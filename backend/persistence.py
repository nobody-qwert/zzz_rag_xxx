"""Async SQLite persistence for documents, jobs, extractions, chunks, embeddings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiosqlite
import numpy as np


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


@dataclass
class EmbeddingRow:
    chunk_id: str
    doc_hash: str
    dim: int
    model: str
    vector: np.ndarray  # float32


class DocumentStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def init(self) -> None:
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
                    CREATE TABLE IF NOT EXISTS extractions (
                        doc_hash TEXT NOT NULL,
                        parser TEXT NOT NULL,
                        text TEXT NOT NULL,
                        meta TEXT,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (doc_hash, parser),
                        FOREIGN KEY(doc_hash) REFERENCES documents(doc_hash) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_hash TEXT NOT NULL,
                        parser TEXT NOT NULL,
                        order_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        token_count INTEGER NOT NULL,
                        FOREIGN KEY(doc_hash) REFERENCES documents(doc_hash) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        chunk_id TEXT PRIMARY KEY,
                        doc_hash TEXT NOT NULL,
                        dim INTEGER NOT NULL,
                        model TEXT NOT NULL,
                        vector BLOB NOT NULL,
                        FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        doc_hash TEXT PRIMARY KEY,
                        pymupdf_time_sec REAL,
                        mineru_time_sec REAL,
                        chunking_time_sec REAL,
                        embedding_time_sec REAL,
                        total_time_sec REAL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(doc_hash) REFERENCES documents(doc_hash) ON DELETE CASCADE
                    )
                    """
                )

                await conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(status)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_ext_doc ON extractions(doc_hash)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_hash)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_emb_doc ON embeddings(doc_hash)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_doc ON performance_metrics(doc_hash)")

                await conn.commit()
            finally:
                await conn.close()
            self._initialized = True

    async def _conn(self) -> aiosqlite.Connection:
        await self.init()
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        return conn

    # Documents
    async def upsert_document(self, *, doc_hash: str, original_name: str, stored_name: str, size: int) -> None:
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
                    status=CASE WHEN documents.status='processed' THEN documents.status ELSE 'pending' END,
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (doc_hash, original_name, stored_name, size, now, now),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def update_document_status(self, doc_hash: str, status: str, *, error: Optional[str] = None) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "UPDATE documents SET status=?, error=?, updated_at=? WHERE doc_hash=?",
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
                "UPDATE documents SET status='processed', error=NULL, updated_at=?, last_ingested_at=? WHERE doc_hash=?",
                (now, now, doc_hash),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_document(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute("SELECT * FROM documents WHERE doc_hash=?", (doc_hash,))
            row = await cur.fetchone()
            await cur.close()
            return dict(row) if row else None
        finally:
            await conn.close()

    async def list_documents(self) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute("SELECT * FROM documents ORDER BY updated_at DESC")
            rows = await cur.fetchall()
            await cur.close()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def count_documents(self, status: Optional[str] = None) -> int:
        conn = await self._conn()
        try:
            if status is None:
                cur = await conn.execute("SELECT COUNT(*) FROM documents")
            else:
                cur = await conn.execute("SELECT COUNT(*) FROM documents WHERE status=?", (status,))
            row = await cur.fetchone()
            await cur.close()
            return int(row[0] if row else 0)
        finally:
            await conn.close()

    # Jobs
    async def create_job(self, job_id: str, doc_hash: str) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "INSERT INTO jobs (job_id, doc_hash, status, error, created_at) VALUES (?, ?, 'queued', NULL, ?)",
                (job_id, doc_hash, _utc_now()),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def mark_job_started(self, job_id: str) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "UPDATE jobs SET status='running', started_at=?, error=NULL WHERE job_id=?",
                (_utc_now(), job_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def finish_job(self, job_id: str, status: str, *, error: Optional[str] = None) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                "UPDATE jobs SET status=?, error=?, finished_at=? WHERE job_id=?",
                (status, error, _utc_now(), job_id),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                """
                SELECT j.job_id, j.doc_hash, j.status, j.error, j.created_at, j.started_at, j.finished_at,
                       d.original_name as file
                FROM jobs j LEFT JOIN documents d ON d.doc_hash=j.doc_hash
                ORDER BY j.created_at DESC LIMIT ?
                """,
                (limit,),
            )
            rows = await cur.fetchall()
            await cur.close()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    async def list_active_jobs(self) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                "SELECT job_id, doc_hash, status FROM jobs WHERE status IN ('queued','running') ORDER BY created_at ASC"
            )
            rows = await cur.fetchall()
            await cur.close()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    # Extractions
    async def upsert_extraction(self, doc_hash: str, parser: str, *, text: str, meta_json: Optional[str]) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                """
                INSERT INTO extractions (doc_hash, parser, text, meta, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_hash, parser) DO UPDATE SET text=excluded.text, meta=excluded.meta, created_at=excluded.created_at
                """,
                (doc_hash, parser, text, meta_json, _utc_now()),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_extraction(self, doc_hash: str, parser: str) -> Optional[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                "SELECT doc_hash, parser, text, meta, created_at FROM extractions WHERE doc_hash=? AND parser=?",
                (doc_hash, parser),
            )
            row = await cur.fetchone()
            await cur.close()
            return dict(row) if row else None
        finally:
            await conn.close()

    # Chunks
    async def replace_chunks(self, doc_hash: str, parser: str, items: Sequence[Tuple[str, int, str, int]]) -> None:
        """Replace chunk rows for a document and parser.

        items: iterable of (chunk_id, order_index, text, token_count)
        """
        conn = await self._conn()
        try:
            await conn.execute("DELETE FROM chunks WHERE doc_hash=? AND parser=?", (doc_hash, parser))
            await conn.executemany(
                "INSERT INTO chunks (chunk_id, doc_hash, parser, order_index, text, token_count) VALUES (?, ?, ?, ?, ?, ?)",
                [(cid, doc_hash, parser, idx, txt, tok) for (cid, idx, txt, tok) in items],
            )
            await conn.commit()
        finally:
            await conn.close()

    async def fetch_chunks(self, doc_hash: Optional[str] = None, parser: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = await self._conn()
        try:
            if doc_hash and parser:
                cur = await conn.execute(
                    "SELECT * FROM chunks WHERE doc_hash=? AND parser=? ORDER BY order_index ASC",
                    (doc_hash, parser),
                )
            elif doc_hash:
                cur = await conn.execute(
                    "SELECT * FROM chunks WHERE doc_hash=? ORDER BY order_index ASC",
                    (doc_hash,),
                )
            else:
                cur = await conn.execute("SELECT * FROM chunks ORDER BY doc_hash, order_index ASC")
            rows = await cur.fetchall()
            await cur.close()
            return [dict(r) for r in rows]
        finally:
            await conn.close()

    # Embeddings
    @staticmethod
    def _pack_vector(v: np.ndarray) -> bytes:
        v = np.asarray(v, dtype=np.float32)
        return v.tobytes()

    @staticmethod
    def _unpack_vector(b: bytes, dim: int) -> np.ndarray:
        return np.frombuffer(b, dtype=np.float32, count=dim)

    async def replace_embeddings(self, rows: Iterable[EmbeddingRow]) -> None:
        conn = await self._conn()
        try:
            # It's acceptable to delete by doc scope per batch; group rows
            rows_list = list(rows)
            if not rows_list:
                return
            doc_hashes = {r.doc_hash for r in rows_list}
            for dh in doc_hashes:
                await conn.execute("DELETE FROM embeddings WHERE doc_hash=?", (dh,))
            await conn.executemany(
                "INSERT INTO embeddings (chunk_id, doc_hash, dim, model, vector) VALUES (?, ?, ?, ?, ?)",
                [
                    (r.chunk_id, r.doc_hash, r.dim, r.model, self._pack_vector(r.vector))
                    for r in rows_list
                ],
            )
            await conn.commit()
        finally:
            await conn.close()

    async def fetch_embeddings_for_chunks(self, chunk_ids: Sequence[str]) -> List[EmbeddingRow]:
        if not chunk_ids:
            return []
        conn = await self._conn()
        try:
            # Build query with variable placeholders
            qs = ",".join(["?"] * len(chunk_ids))
            cur = await conn.execute(
                f"SELECT chunk_id, doc_hash, dim, model, vector FROM embeddings WHERE chunk_id IN ({qs})",
                tuple(chunk_ids),
            )
            rows = await cur.fetchall()
            await cur.close()
            out: List[EmbeddingRow] = []
            for r in rows:
                dim = int(r["dim"])  # type: ignore
                out.append(
                    EmbeddingRow(
                        chunk_id=r["chunk_id"],
                        doc_hash=r["doc_hash"],
                        dim=dim,
                        model=r["model"],
                        vector=self._unpack_vector(r["vector"], dim),
                    )
                )
            return out
        finally:
            await conn.close()

    # Performance Metrics
    async def save_performance_metrics(
        self,
        doc_hash: str,
        *,
        pymupdf_time_sec: Optional[float] = None,
        mineru_time_sec: Optional[float] = None,
        chunking_time_sec: Optional[float] = None,
        embedding_time_sec: Optional[float] = None,
        total_time_sec: Optional[float] = None,
    ) -> None:
        conn = await self._conn()
        try:
            await conn.execute(
                """
                INSERT INTO performance_metrics (
                    doc_hash, pymupdf_time_sec, mineru_time_sec, chunking_time_sec,
                    embedding_time_sec, total_time_sec, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_hash) DO UPDATE SET
                    pymupdf_time_sec=excluded.pymupdf_time_sec,
                    mineru_time_sec=excluded.mineru_time_sec,
                    chunking_time_sec=excluded.chunking_time_sec,
                    embedding_time_sec=excluded.embedding_time_sec,
                    total_time_sec=excluded.total_time_sec,
                    created_at=excluded.created_at
                """,
                (
                    doc_hash,
                    pymupdf_time_sec,
                    mineru_time_sec,
                    chunking_time_sec,
                    embedding_time_sec,
                    total_time_sec,
                    _utc_now(),
                ),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_performance_metrics(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        conn = await self._conn()
        try:
            cur = await conn.execute(
                "SELECT * FROM performance_metrics WHERE doc_hash=?",
                (doc_hash,),
            )
            row = await cur.fetchone()
            await cur.close()
            return dict(row) if row else None
        finally:
            await conn.close()
