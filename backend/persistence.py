"""Async persistence helpers for ingestion metadata and job tracking."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite


def _utc_now() -> str:
    """Return an ISO8601 UTC timestamp with second precision."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


class DocumentStore:
    """Lightweight async wrapper around a SQLite database for document metadata."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def init(self) -> None:
        """Initialise the database (idempotent)."""
        async with self._init_lock:
            if self._initialized:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = await aiosqlite.connect(self.db_path)
            try:
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA synchronous=NORMAL;")
                await conn.execute("PRAGMA foreign_keys=ON;")

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_hash TEXT PRIMARY KEY,
                        original_name TEXT NOT NULL,
                        stored_name TEXT NOT NULL,
                        size INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_ingested_at TEXT
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        doc_hash TEXT NOT NULL,
                        status TEXT NOT NULL,
                        error TEXT,
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        FOREIGN KEY(doc_hash) REFERENCES documents(doc_hash) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_documents_status
                    ON documents(status)
                    """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_jobs_status
                    ON jobs(status)
                    """
                )

                await conn.commit()
            finally:
                await conn.close()

            self._initialized = True

    async def _conn(self) -> aiosqlite.Connection:
        await self.init()
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn

    async def upsert_document(
        self,
        *,
        doc_hash: str,
        original_name: str,
        stored_name: str,
        size: int,
    ) -> None:
        now = _utc_now()
        conn = await self._conn()
        try:
            await conn.execute(
                """
                INSERT INTO documents (doc_hash, original_name, stored_name, size, status, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', NULL, ?, ?)
                ON CONFLICT(doc_hash) DO UPDATE SET
                    original_name=excluded.original_name,
                    stored_name=excluded.stored_name,
                    size=excluded.size,
                    status=CASE
                        WHEN documents.status = 'processed' THEN documents.status
                        ELSE 'pending'
                    END,
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (doc_hash, original_name, stored_name, size, now, now),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_document(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cursor = await conn.execute(
                "SELECT * FROM documents WHERE doc_hash = ?", (doc_hash,)
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return None
            return dict(row)
        finally:
            await conn.close()

    async def list_documents(self) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cursor = await conn.execute(
                "SELECT * FROM documents ORDER BY updated_at DESC"
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def count_documents(self, status: Optional[str] = None) -> int:
        conn = await self._conn()
        try:
            if status is None:
                cursor = await conn.execute("SELECT COUNT(*) FROM documents")
                row = await cursor.fetchone()
            else:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE status = ?",
                    (status,)
                )
                row = await cursor.fetchone()
            await cursor.close()
            return int(row[0] if row else 0)
        finally:
            await conn.close()

    async def update_document_status(
        self, doc_hash: str, status: str, *, error: Optional[str] = None
    ) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                """
                UPDATE documents
                SET status = ?, error = ?, updated_at = ?
                WHERE doc_hash = ?
                """,
                (status, error, _utc_now(), doc_hash),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def mark_document_processed(self, doc_hash: str) -> None:
        conn = await self._conn()
        now = _utc_now()
        try:
            await conn.execute(
                """
                UPDATE documents
                SET status = 'processed', error = NULL, updated_at = ?, last_ingested_at = ?
                WHERE doc_hash = ?
                """,
                (now, now, doc_hash),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def create_job(self, job_id: str, doc_hash: str) -> None:
        conn = await self._conn()
        now = _utc_now()
        try:
            await conn.execute(
                """
                INSERT INTO jobs (job_id, doc_hash, status, error, created_at)
                VALUES (?, ?, 'queued', NULL, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    doc_hash=excluded.doc_hash,
                    status='queued',
                    error=NULL,
                    created_at=excluded.created_at,
                    started_at=NULL,
                    finished_at=NULL
                """,
                (job_id, doc_hash, now),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def mark_job_started(self, job_id: str) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = ?, error = NULL
                WHERE job_id = ?
                """,
                (_utc_now(), job_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def finish_job(
        self, job_id: str, status: str, *, error: Optional[str] = None
    ) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                """
                UPDATE jobs
                SET status = ?, error = ?, finished_at = ?
                WHERE job_id = ?
                """,
                (status, error, _utc_now(), job_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cursor = await conn.execute(
                """
                SELECT j.job_id,
                       j.doc_hash,
                       j.status,
                       j.error,
                       j.created_at,
                       j.started_at,
                       j.finished_at,
                       d.original_name
                FROM jobs j
                LEFT JOIN documents d ON d.doc_hash = j.doc_hash
                ORDER BY j.created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            results: List[Dict[str, Any]] = []
            for row in rows:
                data = dict(row)
                data["file"] = data.pop("original_name", None)
                results.append(data)
            return results
        finally:
            await conn.close()

    async def list_active_jobs(self) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cursor = await conn.execute(
                """
                SELECT job_id, doc_hash, status
                FROM jobs
                WHERE status IN ('queued', 'running')
                ORDER BY created_at ASC
                """
            )
            rows = await cursor.fetchall()
            await cursor.close()
            return [dict(row) for row in rows]
        finally:
            await conn.close()

