from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException, UploadFile

try:
    from ..chunking import sliding_token_chunks
    from ..embeddings import EmbeddingClient
    from ..persistence import EmbeddingRow
    from ..dependencies import document_store, jobs_registry, settings
    from ..utils.files import safe_filename
except ImportError:  # pragma: no cover
    from chunking import sliding_token_chunks  # type: ignore
    from embeddings import EmbeddingClient  # type: ignore
    from persistence import EmbeddingRow  # type: ignore
    from dependencies import document_store, jobs_registry, settings  # type: ignore
    from utils.files import safe_filename  # type: ignore

logger = logging.getLogger(__name__)


async def ingest_file(file: UploadFile) -> Dict[str, Any]:
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    display_name = safe_filename(file.filename or "upload.bin")
    content = await file.read()
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    doc_hash = hashlib.sha256(content).hexdigest()
    existing = await document_store.get_document(doc_hash)
    if existing and existing.get("status") == "processed":
        return {
            "job_id": None,
            "status": "skipped",
            "file": existing.get("original_name") or display_name,
            "hash": doc_hash,
            "message": "Document already ingested",
        }

    suffix = Path(display_name).suffix
    stored_name = f"{doc_hash}{suffix.lower()}" if suffix else doc_hash
    dest_path = settings.data_dir / stored_name
    dest_path.write_bytes(content)

    await document_store.upsert_document(
        doc_hash=doc_hash,
        original_name=display_name,
        stored_name=stored_name,
        size=len(content),
    )

    job_id = await _queue_job(dest_path, doc_hash, display_name)
    return {"job_id": job_id, "status": "queued", "file": display_name, "hash": doc_hash}


async def retry_ingest(doc_hash: str) -> Dict[str, Any]:
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    stored_name = doc.get("stored_name") or doc_hash
    source_path = settings.data_dir / stored_name
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Stored document missing")

    display_name = doc.get("original_name") or stored_name
    job_id = await _queue_job(source_path, doc_hash, display_name)
    return {"job_id": job_id, "status": "queued", "file": display_name, "hash": doc_hash, "retry": True}


def get_job_status(job_id: str) -> Dict[str, Any]:
    info = jobs_registry.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **info}


async def delete_document(doc_hash: str) -> Dict[str, Any]:
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    active_jobs = [
        jid for jid, info in jobs_registry.items()
        if info.get("hash") == doc_hash and info.get("status") in {"queued", "running"}
    ]
    if active_jobs:
        raise HTTPException(status_code=409, detail="Document is currently processing and cannot be removed")

    stored_name = doc.get("stored_name") or doc_hash
    file_path = settings.data_dir / stored_name
    file_removed = False
    if file_path.exists():
        try:
            file_path.unlink()
            file_removed = True
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Failed to delete stored file %s: %s", file_path, exc)

    deleted = await document_store.delete_document(doc_hash)
    removed_jobs: List[str] = []
    for jid, info in list(jobs_registry.items()):
        if info.get("hash") == doc_hash:
            removed_jobs.append(jid)
            jobs_registry.pop(jid, None)

    return {
        "hash": doc_hash,
        "deleted": bool(deleted),
        "file_removed": file_removed,
        "removed_jobs": removed_jobs,
    }


async def _queue_job(source_path: Path, doc_hash: str, display_name: str) -> str:
    from uuid import uuid4

    job_id = str(uuid4())
    jobs_registry[job_id] = {"status": "queued", "file": display_name, "hash": doc_hash}
    await document_store.create_job(job_id, doc_hash)
    asyncio.create_task(_process_job(job_id, source_path, doc_hash, display_name))
    return job_id


async def _process_job(job_id: str, doc_path: Path, doc_hash: str, display_name: str) -> None:
    async def update(status: str, *, error: Optional[str] = None) -> None:
        jobs_registry[job_id]["status"] = status if not error else f"error: {error}"
        await document_store.update_document_status(doc_hash, status if not error else "error", error=error)

    start_total = time.perf_counter()
    pymupdf_time = None
    mineru_time = None
    chunking_time = None
    embedding_time = None

    try:
        await document_store.mark_job_started(job_id)
        await document_store.update_document_status(doc_hash, "processing")

        start_pymupdf = time.perf_counter()
        py_out = await _extract_pymupdf(doc_path)
        pymupdf_time = time.perf_counter() - start_pymupdf
        await document_store.upsert_extraction(
            doc_hash,
            "pymupdf",
            text=py_out["text"],
            meta_json=json.dumps({"pages": py_out.get("pages", 0)}),
        )

        pymupdf_text = py_out.get("text", "") or ""
        chosen_text = pymupdf_text
        chosen_meta: Dict[str, Any] = {"parser_used": "pymupdf", "reason": "default"}

        run_ocr = settings.parser_mode != "pymupdf"
        ocr_allowed = _is_pdf_file(doc_path)
        if run_ocr and not ocr_allowed:
            run_ocr = False
            logger.debug("Skipping OCR for non-PDF file %s", doc_path)

        if run_ocr:
            start_mineru = time.perf_counter()
            try:
                ocr_result = await _call_ocr_module(doc_hash, doc_path)
                mineru_time = time.perf_counter() - start_mineru
                ocr_text = ocr_result.get("text") or ""
                ocr_meta = ocr_result.get("metadata") or {}
                await document_store.upsert_extraction(
                    doc_hash,
                    settings.ocr_parser_key,
                    text=ocr_text or pymupdf_text,
                    meta_json=json.dumps({**ocr_meta, "source": "ocr", "fallback_used": not bool(ocr_text)}),
                )
                if ocr_text:
                    chosen_text = ocr_text
                    chosen_meta = {"parser_used": settings.ocr_parser_key}
                else:
                    chosen_meta = {
                        "parser_used": "pymupdf",
                        "reason": "ocr_empty_text",
                    }
            except Exception as exc:
                mineru_time = None
                logger.warning("OCR module failed for %s: %s — falling back to PyMuPDF text", doc_path, exc)
                fallback_meta = {
                    "parser_used": "pymupdf",
                    "reason": f"ocr_module_failed: {exc}",
                }
                chosen_text = pymupdf_text
                chosen_meta = fallback_meta
                await document_store.upsert_extraction(
                    doc_hash,
                    settings.ocr_parser_key,
                    text=pymupdf_text,
                    meta_json=json.dumps({**fallback_meta, "source": "pymupdf"}),
                )
        else:
            skip_reason = "non_pdf" if not ocr_allowed else "pymupdf_only"
            chosen_meta = {"parser_used": "pymupdf", "reason": skip_reason}
            await document_store.upsert_extraction(
                doc_hash,
                settings.ocr_parser_key,
                text=pymupdf_text,
                meta_json=json.dumps({"parser_used": "pymupdf", "reason": skip_reason, "source": "pymupdf"}),
            )

        start_chunking = time.perf_counter()
        chunks = sliding_token_chunks(chosen_text, size=settings.chunk_size, overlap=settings.chunk_overlap)
        chunk_rows = [(c.chunk_id, c.order_index, c.text, c.token_count) for c in chunks]
        await document_store.replace_chunks(doc_hash, settings.ocr_parser_key, chunk_rows)
        chunking_time = time.perf_counter() - start_chunking

        start_embedding = time.perf_counter()
        emb_client = EmbeddingClient()
        rows = await _compute_embeddings_for_chunks(
            [
                {"chunk_id": cid, "order_index": idx, "text": txt, "token_count": tok}
                for (cid, idx, txt, tok) in chunk_rows
            ],
            emb_client,
            doc_hash,
        )
        await document_store.replace_embeddings(rows)
        embedding_time = time.perf_counter() - start_embedding

        total_time = time.perf_counter() - start_total
        try:
            await document_store.save_performance_metrics(
                doc_hash,
                pymupdf_time_sec=pymupdf_time,
                mineru_time_sec=mineru_time,
                chunking_time_sec=chunking_time,
                embedding_time_sec=embedding_time,
                total_time_sec=total_time,
            )
        except Exception as perf_exc:  # pragma: no cover - metrics best effort
            logger.warning("Failed to save performance metrics for %s: %s", doc_hash, perf_exc)

        await document_store.mark_document_processed(doc_hash)
        await document_store.finish_job(job_id, "done")
        jobs_registry[job_id]["status"] = "done"
    except asyncio.CancelledError:
        await document_store.finish_job(job_id, "cancelled", error="cancelled")
        await update("error", error="cancelled")
        raise
    except Exception as exc:  # pragma: no cover - background task logging
        await document_store.finish_job(job_id, "error", error=str(exc))
        await update("error", error=str(exc))
        logger.exception("Job %s failed: %s", job_id, exc)


async def _extract_pymupdf(path: Path) -> Dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyMuPDF not installed") from exc

    def _run() -> Dict[str, Any]:
        doc = fitz.open(path)
        try:
            parts: List[str] = []
            for page in doc:
                text = page.get_text("text")
                if text and text.strip():
                    parts.append(text)
            return {"text": "\n\n".join(p.strip() for p in parts if p.strip()), "pages": len(doc)}
        finally:
            doc.close()

    return await asyncio.to_thread(_run)


def _is_pdf_file(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            sig = handle.read(5)
            if isinstance(sig, bytes) and sig.startswith(b"%PDF-"):
                return True
    except Exception:
        pass
    return path.suffix.lower() == ".pdf"


async def _call_ocr_module(doc_hash: str, doc_path: Path) -> Dict[str, Any]:
    if not doc_path.exists():
        raise RuntimeError(f"OCR source file missing: {doc_path}")

    form_data = {"doc_hash": doc_hash}
    filename = doc_path.name or f"{doc_hash}.bin"
    content_type = "application/pdf" if doc_path.suffix.lower() == ".pdf" else "application/octet-stream"

    timeout = httpx.Timeout(settings.ocr_module_timeout, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            with doc_path.open("rb") as file_obj:
                files = {"file": (filename, file_obj, content_type)}
                response = await client.post(f"{settings.ocr_module_url}/parse", data=form_data, files=files)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"OCR module returned {exc.response.status_code}: {detail}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"OCR module request failed: {exc}") from exc

    return data


async def _compute_embeddings_for_chunks(
    chunks: List[Dict[str, Any]],
    client: EmbeddingClient,
    doc_hash: str,
) -> List[EmbeddingRow]:
    texts = [chunk["text"] for chunk in chunks]
    vectors = await client.embed_batch(texts)
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding result size mismatch")
    if client.dim is None:
        raise RuntimeError("Embedding dimension is unknown after embedding call")

    rows: List[EmbeddingRow] = []
    for chunk, vector in zip(chunks, vectors):
        rows.append(
            EmbeddingRow(
                chunk_id=chunk["chunk_id"],
                doc_hash=doc_hash,
                dim=client.dim,
                model=client.model,
                vector=vector,
            )
        )
    return rows


async def warmup_mineru() -> Dict[str, Any]:
    timeout = httpx.Timeout(settings.ocr_module_timeout, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(f"{settings.ocr_module_url}/warmup")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise HTTPException(status_code=500, detail=f"OCR module warmup failed: {detail}") from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=500, detail=f"OCR module warmup request failed: {exc}") from exc
