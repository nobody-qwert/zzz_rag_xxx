from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx
from fastapi import HTTPException, UploadFile

try:
    from ..chunking import ChunkWindowSpec, chunk_text_multi
    from ..embeddings import EmbeddingClient
    from ..persistence import EmbeddingRow
    from ..dependencies import document_store, jobs_registry, settings, gpu_phase_manager
    from ..utils.files import safe_filename
except ImportError:  # pragma: no cover
    from chunking import ChunkWindowSpec, chunk_text_multi  # type: ignore
    from embeddings import EmbeddingClient  # type: ignore
    from persistence import EmbeddingRow  # type: ignore
    from dependencies import document_store, jobs_registry, settings, gpu_phase_manager  # type: ignore
    from utils.files import safe_filename  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class _QueuedBatchDoc:
    doc_path: Path
    doc_hash: str
    display_name: str
    job_record_id: Optional[str] = None


@dataclass
class _QueuedBatchJob:
    job_id: str
    docs: List[_QueuedBatchDoc]


_job_queue: "asyncio.Queue[_QueuedBatchJob]" = asyncio.Queue()
_worker_task: Optional["asyncio.Task[None]"] = None
_worker_lock = asyncio.Lock()
_reprocess_all_lock = asyncio.Lock()


async def ingest_file(file: UploadFile) -> Dict[str, Any]:
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    return await ingest_files([file])


async def ingest_files(files: Sequence[UploadFile]) -> Dict[str, Any]:
    uploads = [f for f in files if f is not None]
    if not uploads:
        raise HTTPException(status_code=400, detail="No files uploaded")

    prepared_docs: List[_QueuedBatchDoc] = []
    results: List[Dict[str, Any]] = []

    for upload in uploads:
        doc_info, payload = await _prepare_upload(upload)
        results.append(payload)
        if doc_info:
            prepared_docs.append(doc_info)

    job_id: Optional[str] = None
    if prepared_docs:
        job_id = await _queue_batch_job(prepared_docs)
        for payload in results:
            if payload.get("status") == "queued":
                payload["job_id"] = job_id

    skipped = sum(1 for r in results if r.get("status") == "skipped")
    return {
        "job_id": job_id,
        "results": results,
        "queued_count": len(prepared_docs),
        "skipped_count": skipped,
        "total": len(results),
    }


async def retry_ingest(doc_hash: str) -> Dict[str, Any]:
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    stored_name = doc.get("stored_name") or doc_hash
    source_path = settings.data_dir / stored_name
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Stored document missing")

    display_name = doc.get("original_name") or stored_name
    job_doc = _QueuedBatchDoc(doc_path=source_path, doc_hash=doc_hash, display_name=display_name)
    job_id = await _queue_batch_job([job_doc])
    return {
        "job_id": job_id,
        "status": "queued",
        "file": display_name,
        "hash": doc_hash,
        "retry": True,
    }


async def reprocess_after_ocr(doc_hash: str, *, ensure_gpu_phase: bool = True) -> Dict[str, Any]:
    """Re-run chunking and embeddings using existing OCR text."""
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    active_jobs = [
        jid
        for jid, info in jobs_registry.items()
        if info.get("status") in {"queued", "running"}
        and any(d.get("hash") == doc_hash for d in info.get("docs", []))
    ]
    if active_jobs:
        raise HTTPException(status_code=409, detail=f"Document is currently processing (jobs: {', '.join(active_jobs)})")

    extraction = await document_store.get_extraction(doc_hash, settings.ocr_parser_key)
    if not extraction:
        raise HTTPException(status_code=404, detail="No OCR extraction found; re-run OCR first")

    ocr_text = (extraction.get("text") or "").strip()
    if not ocr_text:
        raise HTTPException(status_code=400, detail="OCR extraction is empty; re-run OCR first")

    if ensure_gpu_phase:
        try:
            await gpu_phase_manager.switch_to_no_llm(reason=f"reprocess:{doc_hash}")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"GPU not available for preprocessing: {exc}") from exc

    await document_store.update_document_status(doc_hash, "processing")

    chunk_specs = [
        ChunkWindowSpec(
            name="small",
            core_size=settings.chunk_size,
            left_padding=settings.chunk_overlap,
            right_padding=settings.chunk_overlap,
            step_size=settings.chunk_size,
        ),
        ChunkWindowSpec(
            name="large",
            core_size=settings.large_chunk_size,
            left_padding=settings.large_chunk_left_overlap,
            right_padding=settings.large_chunk_right_overlap,
            step_size=settings.large_chunk_size,
        ),
    ]

    try:
        emb_client = EmbeddingClient()

        start_chunking = time.perf_counter()
        chunk_views = chunk_text_multi(ocr_text, chunk_specs)
        small_chunks = chunk_views.get("small", [])
        large_chunks = chunk_views.get("large", [])
        if not small_chunks and not large_chunks:
            raise RuntimeError("Chunking produced no chunks; check chunking settings")

        small_chunk_rows = [(c.chunk_id, c.order_index, c.text, c.token_count) for c in small_chunks]
        large_chunk_rows = [(c.chunk_id, c.order_index, c.text, c.token_count) for c in large_chunks]
        await document_store.replace_chunks(doc_hash, settings.chunk_config_small_id, small_chunk_rows)
        await document_store.replace_chunks(doc_hash, settings.chunk_config_large_id, large_chunk_rows)
        chunking_time = time.perf_counter() - start_chunking

        start_embedding = time.perf_counter()
        rows = await _compute_embeddings_for_chunks(
            [
                {"chunk_id": cid, "order_index": idx, "text": txt, "token_count": tok}
                for (cid, idx, txt, tok) in (small_chunk_rows + large_chunk_rows)
            ],
            emb_client,
            doc_hash,
        )
        await document_store.replace_embeddings(rows)
        embedding_time = time.perf_counter() - start_embedding

        prev_metrics = await document_store.get_performance_metrics(doc_hash)
        ocr_time_val = None
        if prev_metrics:
            try:
                ocr_time_val = float(prev_metrics.get("ocr_time_sec")) if prev_metrics.get("ocr_time_sec") is not None else None
            except (TypeError, ValueError):
                ocr_time_val = prev_metrics.get("ocr_time_sec")
        total_time = (ocr_time_val or 0.0) + chunking_time + embedding_time

        await document_store.save_performance_metrics(
            doc_hash,
            ocr_time_sec=ocr_time_val,
            chunking_time_sec=chunking_time,
            embedding_time_sec=embedding_time,
            total_time_sec=total_time,
        )
        await document_store.mark_document_processed(doc_hash)

        return {
            "hash": doc_hash,
            "document_name": doc.get("original_name") or doc_hash,
            "status": "processed",
            "chunk_count": len(small_chunk_rows) + len(large_chunk_rows),
            "small_chunks": len(small_chunk_rows),
            "large_chunks": len(large_chunk_rows),
            "total_embeddings": len(rows),
            "chunking_time_sec": chunking_time,
            "embedding_time_sec": embedding_time,
            "ocr_time_sec": ocr_time_val,
            "total_time_sec": total_time,
        }
    except HTTPException:
        # Pass through HTTP-level errors unchanged
        raise
    except Exception as exc:
        error_msg = str(exc)
        await document_store.update_document_status(doc_hash, "error", error=error_msg)
        raise HTTPException(status_code=500, detail=error_msg) from exc


async def reprocess_all_documents() -> Dict[str, Any]:
    """Reprocess every document that already completed OCR."""
    if _reprocess_all_lock.locked():
        raise HTTPException(status_code=409, detail="A bulk reprocess is already running")

    async with _reprocess_all_lock:
        docs = await document_store.list_documents()
        doc_hashes = [doc.get("doc_hash") for doc in docs if doc.get("doc_hash")]
        total = len(doc_hashes)
        if total == 0:
            return {"total": 0, "processed": 0, "skipped": 0, "failed": 0, "results": []}

        try:
            await gpu_phase_manager.switch_to_no_llm(reason="reprocess_all")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"GPU not available for preprocessing: {exc}") from exc

        results: List[Dict[str, Any]] = []
        processed = 0
        skipped = 0
        failed = 0

        for doc_hash in doc_hashes:
            if not doc_hash:
                skipped += 1
                results.append({"hash": None, "status": "skipped", "reason": "missing_hash"})
                continue
            try:
                payload = await reprocess_after_ocr(doc_hash, ensure_gpu_phase=False)
                processed += 1
                results.append(
                    {
                        "hash": doc_hash,
                        "status": "processed",
                        "chunk_count": payload.get("chunk_count"),
                        "total_embeddings": payload.get("total_embeddings"),
                    }
                )
            except HTTPException as exc:
                if isinstance(exc.detail, str):
                    detail = exc.detail
                else:
                    detail = json.dumps(exc.detail) if isinstance(exc.detail, (dict, list)) else str(exc.detail)
                if exc.status_code in (404, 409):
                    skipped += 1
                    results.append({"hash": doc_hash, "status": "skipped", "reason": detail})
                else:
                    failed += 1
                    results.append({"hash": doc_hash, "status": "error", "reason": detail})
            except Exception as exc:  # pragma: no cover - bulk guard
                failed += 1
                results.append({"hash": doc_hash, "status": "error", "reason": str(exc)})

        return {
            "total": total,
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            "results": results,
        }


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
        jid
        for jid, info in jobs_registry.items()
        if info.get("status") in {"queued", "running"}
        and any(d.get("hash") == doc_hash for d in info.get("docs", []))
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
        docs = info.get("docs") or []
        if any(d.get("hash") == doc_hash for d in docs):
            removed_jobs.append(jid)
            jobs_registry.pop(jid, None)

    return {
        "hash": doc_hash,
        "deleted": bool(deleted),
        "file_removed": file_removed,
        "removed_jobs": removed_jobs,
    }


async def _prepare_upload(file: UploadFile) -> tuple[Optional[_QueuedBatchDoc], Dict[str, Any]]:
    display_name = safe_filename(file.filename or "upload.bin")
    content = await file.read()
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{display_name}' is empty")

    doc_hash = hashlib.sha256(content).hexdigest()
    existing = await document_store.get_document(doc_hash)
    if existing and existing.get("status") == "processed":
        return (
            None,
            {
                "job_id": None,
                "status": "skipped",
                "file": existing.get("original_name") or display_name,
                "hash": doc_hash,
                "message": "Document already ingested",
            },
        )

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

    return (
        _QueuedBatchDoc(doc_path=dest_path, doc_hash=doc_hash, display_name=display_name),
        {
            "job_id": None,
            "status": "queued",
            "file": display_name,
            "hash": doc_hash,
        },
    )


async def _queue_batch_job(docs: Sequence[_QueuedBatchDoc]) -> str:
    from uuid import uuid4

    batch_docs = [doc for doc in docs if doc.doc_path.exists()]
    if not batch_docs:
        raise HTTPException(status_code=400, detail="No documents to ingest")

    job_id = str(uuid4())
    for idx, doc in enumerate(batch_docs, start=1):
        job_record_id = f"{job_id}:{idx}"
        doc.job_record_id = job_record_id
        await document_store.create_job(job_record_id, doc.doc_hash)

    _init_job_entry(job_id, batch_docs)
    await _job_queue.put(_QueuedBatchJob(job_id=job_id, docs=batch_docs))
    await _ensure_worker_running()
    return job_id


def _init_job_entry(job_id: str, docs: Sequence[_QueuedBatchDoc]) -> None:
    doc_entries: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        doc_entries.append(
            {
                "hash": doc.doc_hash,
                "file": doc.display_name,
                "order": idx,
                "status": "queued",
                "phase": "queued",
                "progress": {"phase": "queued", "stage": "queued", "percent": 0.0},
            }
        )

    jobs_registry[job_id] = {
        "status": "queued",
        "phase": "queued",
        "progress": {"phase": "queued", "stage": "queued", "percent": 0.0},
        "docs": doc_entries,
        "total_docs": len(doc_entries),
        "file": doc_entries[0]["file"] if len(doc_entries) == 1 else f"{len(doc_entries)} documents",
        "hash": doc_entries[0]["hash"] if len(doc_entries) == 1 else None,
    }


def _set_job_progress(job_id: str, phase: str, percent: float, *, stage: Optional[str] = None, **extra: Any) -> None:
    info = jobs_registry.get(job_id)
    if not info:
        return
    payload = {
        "phase": phase,
        "stage": stage or phase,
        "percent": max(0.0, min(100.0, float(percent))),
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    info["phase"] = phase
    info["progress"] = payload


def _update_doc_progress(doc_entry: Dict[str, Any], phase: str, percent: float, *, stage: Optional[str] = None, **extra: Any) -> None:
    payload = {
        "phase": phase,
        "stage": stage or phase,
        "percent": max(0.0, min(100.0, float(percent))),
    }
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    doc_entry["phase"] = phase
    doc_entry["progress"] = payload


async def _ensure_worker_running() -> None:
    global _worker_task
    async with _worker_lock:
        if _worker_task is None or _worker_task.done():
            _worker_task = asyncio.create_task(_job_worker())


async def _job_worker() -> None:
    while True:
        job = await _job_queue.get()
        try:
            await _process_job(job.job_id, job.docs)
        except asyncio.CancelledError:
            _job_queue.put_nowait(job)
            raise
        except Exception:  # pragma: no cover - worker guard
            logger.exception("Unhandled error while processing queued job %s", job.job_id)
        finally:
            _job_queue.task_done()


async def _process_job(job_id: str, docs: Sequence[_QueuedBatchDoc]) -> None:
    info = jobs_registry.get(job_id)
    if not info:
        return

    total_docs = len(docs)
    if total_docs == 0:
        info["status"] = "done"
        _set_job_progress(job_id, "completed", 100.0, stage="completed")
        return

    doc_entries = info.get("docs", [])
    doc_by_hash = {entry.get("hash"): entry for entry in doc_entries}
    info["status"] = "running"
    ocr_phase_acquired = False

    try:
        await gpu_phase_manager.switch_to_no_llm(reason=f"ingest:{job_id}")
        ocr_phase_acquired = True
    except Exception as exc:
        error_msg = f"GPU not available for OCR: {exc}"
        _set_job_progress(job_id, "error", 0.0, stage="gpu_unavailable", message=error_msg)
        for entry in doc_entries:
            entry["status"] = "error"
            _update_doc_progress(entry, "error", 0.0, stage="gpu_unavailable", error=error_msg)
        for doc in docs:
            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "error", error=error_msg)
        info["status"] = "error"
        return

    _set_job_progress(job_id, "ocr", 0.0, stage="starting", total_docs=total_docs)

    ocr_payloads: Dict[str, Dict[str, Any]] = {}
    successful_docs: List[_QueuedBatchDoc] = []
    errors: Dict[str, str] = {}

    for index, doc in enumerate(docs, start=1):
        doc_entry = doc_by_hash.get(doc.doc_hash)
        if not doc_entry:
            continue
        await document_store.update_document_status(doc.doc_hash, "ocr_pending")
        if doc.job_record_id:
            await document_store.mark_job_started(doc.job_record_id)
        doc_entry["status"] = "ocr_running"
        _update_doc_progress(doc_entry, "ocr", 0.0, stage="starting")

        start_ocr = time.perf_counter()

        def ocr_progress_callback(data: Dict[str, Any]) -> None:
            if not data:
                return
            stage = str(data.get("stage") or "ocr")
            percent_raw = data.get("percent")
            try:
                percent_val = float(percent_raw) if percent_raw is not None else float(doc_entry.get("progress", {}).get("percent", 0.0))
            except (TypeError, ValueError):
                percent_val = float(doc_entry.get("progress", {}).get("percent", 0.0))
            _update_doc_progress(
                doc_entry,
                "ocr",
                percent_val,
                stage=stage,
                current=data.get("current"),
                total=data.get("total"),
            )

        try:
            ocr_result = await _call_ocr_module(doc.doc_hash, doc.doc_path, progress_cb=ocr_progress_callback)
            ocr_time = time.perf_counter() - start_ocr
            ocr_text_raw = ocr_result.get("text") or ""
            ocr_text = ocr_text_raw.strip()
            if not ocr_text:
                raise RuntimeError(f"OCR parser returned no text for document '{doc.display_name}'")
            ocr_meta = ocr_result.get("metadata") or {}
            await document_store.upsert_extraction(
                doc.doc_hash,
                settings.ocr_parser_key,
                text=ocr_text,
                meta_json=json.dumps({**ocr_meta, "source": "ocr"}),
            )
            await document_store.update_document_status(doc.doc_hash, "waiting_postprocess")
            doc_entry["status"] = "ocr_completed"
            _update_doc_progress(doc_entry, "ocr", 100.0, stage="completed")
            ocr_payloads[doc.doc_hash] = {"text": ocr_text, "metadata": ocr_meta, "ocr_time": ocr_time}
            successful_docs.append(doc)
        except Exception as exc:  # pragma: no cover - background task logging
            error_msg = str(exc)
            errors[doc.doc_hash] = error_msg
            doc_entry["status"] = "error"
            _update_doc_progress(doc_entry, "ocr", doc_entry.get("progress", {}).get("percent", 0.0), stage="failed", error=error_msg)
            await document_store.update_document_status(doc.doc_hash, "error", error=error_msg)
            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "error", error=error_msg)
            logger.exception("OCR failed for %s: %s", doc.doc_path, exc)
        finally:
            percent = (index / total_docs) * 50.0
            _set_job_progress(job_id, "ocr", percent, stage=f"{index}/{total_docs}")

    if not successful_docs:
        info["status"] = "error"
        _set_job_progress(job_id, "error", 50.0, stage="ocr_failed", message="All OCR attempts failed")
        return

    chunk_specs = [
        ChunkWindowSpec(
            name="small",
            core_size=settings.chunk_size,
            left_padding=settings.chunk_overlap,
            right_padding=settings.chunk_overlap,
            step_size=settings.chunk_size,
        ),
        ChunkWindowSpec(
            name="large",
            core_size=settings.large_chunk_size,
            left_padding=settings.large_chunk_left_overlap,
            right_padding=settings.large_chunk_right_overlap,
            step_size=settings.large_chunk_size,
        ),
    ]

    _set_job_progress(job_id, "postprocess", 50.0, stage="starting", total_docs=len(successful_docs))
    emb_client = EmbeddingClient()
    post_total = len(successful_docs)
    processed_docs = 0

    for doc in successful_docs:
        doc_entry = doc_by_hash.get(doc.doc_hash)
        if not doc_entry:
            continue
        payload = ocr_payloads.get(doc.doc_hash) or {}
        ocr_text = payload.get("text") or ""
        if not ocr_text:
            doc_entry["status"] = "error"
            _update_doc_progress(doc_entry, "chunking", 0.0, stage="failed", error="missing_ocr_text")
            continue

        await document_store.update_document_status(doc.doc_hash, "processing")

        def chunk_progress_callback(data: Dict[str, Any]) -> None:
            if not data:
                return
            spec_name_raw = data.get("spec")
            spec_stage_raw = data.get("stage")
            spec_name = str(spec_name_raw).strip() if spec_name_raw else "chunk"
            spec_stage = str(spec_stage_raw).strip() if isinstance(spec_stage_raw, str) else ""
            if not spec_stage:
                spec_stage = "processing"
            try:
                spec_percent = float(data.get("percent") or 0.0)
            except (TypeError, ValueError):
                spec_percent = 0.0
            try:
                spec_index = int(data.get("spec_index") or 1)
            except (TypeError, ValueError):
                spec_index = 1
            try:
                spec_total = int(data.get("spec_total") or 1)
            except (TypeError, ValueError):
                spec_total = 1
            spec_index = max(1, spec_index)
            spec_total = max(1, spec_total)
            spec_fraction = max(0.0, min(1.0, spec_percent / 100.0))
            overall_fraction = ((spec_index - 1) + spec_fraction) / spec_total
            overall_percent = max(0.0, min(100.0, overall_fraction * 100.0))
            stage_label = f"{spec_name}:{spec_stage}" if spec_stage else spec_name
            _update_doc_progress(
                doc_entry,
                "chunking",
                overall_percent,
                stage=stage_label,
                spec=spec_name,
                spec_index=spec_index,
                spec_total=spec_total,
                spec_percent=spec_percent,
                chunk_count=data.get("chunk_count"),
            )

        try:
            doc_entry["status"] = "chunking"
            _update_doc_progress(doc_entry, "chunking", 0.0, stage="starting")
            start_chunking = time.perf_counter()
            chunk_views = chunk_text_multi(ocr_text, chunk_specs, progress_cb=chunk_progress_callback)
            small_chunks = chunk_views.get("small", [])
            large_chunks = chunk_views.get("large", [])
            if not small_chunks:
                raise RuntimeError(f"No OCR text extracted for document '{doc.display_name}'")

            small_chunk_rows = [(c.chunk_id, c.order_index, c.text, c.token_count) for c in small_chunks]
            large_chunk_rows = [(c.chunk_id, c.order_index, c.text, c.token_count) for c in large_chunks]
            await document_store.replace_chunks(doc.doc_hash, settings.chunk_config_small_id, small_chunk_rows)
            await document_store.replace_chunks(doc.doc_hash, settings.chunk_config_large_id, large_chunk_rows)
            chunking_time = time.perf_counter() - start_chunking

            doc_entry["status"] = "embedding"

            def embedding_progress_callback(data: Dict[str, Any]) -> None:
                if not data:
                    return
                try:
                    percent_val = float(data.get("percent") or 0.0)
                except (TypeError, ValueError):
                    percent_val = 0.0
                _update_doc_progress(
                    doc_entry,
                    "embedding",
                    percent_val,
                    stage="batch",
                    processed=data.get("processed"),
                    total=data.get("total"),
                )

            _update_doc_progress(doc_entry, "embedding", 0.0, stage="starting")
            start_embedding = time.perf_counter()
            rows = await _compute_embeddings_for_chunks(
                [
                    {"chunk_id": cid, "order_index": idx, "text": txt, "token_count": tok}
                    for (cid, idx, txt, tok) in (small_chunk_rows + large_chunk_rows)
                ],
                emb_client,
                doc.doc_hash,
                progress_cb=embedding_progress_callback,
            )
            await document_store.replace_embeddings(rows)
            embedding_time = time.perf_counter() - start_embedding

            total_time = (payload.get("ocr_time", 0.0) or 0.0) + chunking_time + embedding_time
            try:
                await document_store.save_performance_metrics(
                    doc.doc_hash,
                    ocr_time_sec=payload.get("ocr_time"),
                    chunking_time_sec=chunking_time,
                    embedding_time_sec=embedding_time,
                    total_time_sec=total_time,
                )
            except Exception as perf_exc:  # pragma: no cover - metrics best effort
                logger.warning("Failed to save performance metrics for %s: %s", doc.doc_hash, perf_exc)

            await document_store.mark_document_processed(doc.doc_hash)
            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "done")

            doc_entry["status"] = "completed"
            _update_doc_progress(
                doc_entry,
                "completed",
                100.0,
                stage="done",
                embeddings=len(rows),
                small_embeddings=len(small_chunk_rows),
                large_embeddings=len(large_chunk_rows),
            )
        except Exception as exc:
            error_msg = str(exc)
            doc_entry["status"] = "error"
            _update_doc_progress(doc_entry, "chunking", doc_entry.get("progress", {}).get("percent", 0.0), stage="failed", error=error_msg)
            await document_store.update_document_status(doc.doc_hash, "error", error=error_msg)
            if doc.job_record_id:
                await document_store.finish_job(doc.job_record_id, "error", error=error_msg)
            logger.exception("Chunking/embedding failed for %s: %s", doc.doc_hash, exc)
            continue

        processed_docs += 1
        percent = 50.0 + (processed_docs / post_total) * 50.0
        _set_job_progress(job_id, "postprocess", percent, stage=f"{processed_docs}/{post_total}")

    had_errors = bool(errors) or any(entry.get("status") == "error" for entry in doc_entries)
    info["status"] = "error" if had_errors else "done"
    final_stage = "failed" if had_errors else "done"
    _set_job_progress(job_id, "completed" if not had_errors else "error", 100.0, stage=final_stage)


async def _call_ocr_module(
    doc_hash: str,
    doc_path: Path,
    *,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
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
            job_info = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise RuntimeError(f"OCR module returned {exc.response.status_code}: {detail}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"OCR module request failed: {exc}") from exc

        job_id = job_info.get("job_id")
        if not job_id:
            raise RuntimeError("OCR module response missing job_id")
        status_url = f"{settings.ocr_module_url}/jobs/{job_id}"
        result_url = f"{status_url}/result"
        deadline = time.perf_counter() + settings.ocr_module_timeout
        poll_interval = max(0.5, float(settings.ocr_status_poll_interval))

        retry_delay = min(2.0, poll_interval)

        while True:
            if time.perf_counter() > deadline:
                raise RuntimeError(f"OCR module job {job_id} timed out after {settings.ocr_module_timeout} seconds")
            try:
                status_resp = await client.get(status_url)
                status_resp.raise_for_status()
                status_data = status_resp.json()
                if progress_cb:
                    progress_payload = status_data.get("progress")
                    if isinstance(progress_payload, dict):
                        progress_cb(progress_payload)
            except httpx.TransportError as exc:
                logger.warning("OCR status poll for job %s failed: %s — retrying", job_id, exc)
                await asyncio.sleep(retry_delay)
                continue
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                raise RuntimeError(f"OCR module status error {exc.response.status_code}: {detail}") from exc
            except httpx.RequestError as exc:
                raise RuntimeError(f"OCR module status request failed: {exc}") from exc

            status_value = (status_data.get("status") or "").lower()
            if status_value == "done":
                try:
                    result_resp = await client.get(result_url)
                    result_resp.raise_for_status()
                    result_data = result_resp.json()
                    if progress_cb:
                        progress_cb({"stage": "completed", "percent": 100.0})
                    return result_data
                except httpx.TransportError as exc:
                    logger.warning("OCR result fetch for job %s failed: %s — retrying", job_id, exc)
                    await asyncio.sleep(retry_delay)
                    continue
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    raise RuntimeError(f"OCR module result error {exc.response.status_code}: {detail}") from exc
                except httpx.RequestError as exc:
                    raise RuntimeError(f"OCR module result request failed: {exc}") from exc
            if status_value == "error":
                error_detail = status_data.get("error") or "unknown error"
                raise RuntimeError(f"OCR module job {job_id} failed: {error_detail}")

            await asyncio.sleep(poll_interval)


async def _compute_embeddings_for_chunks(
    chunks: List[Dict[str, Any]],
    client: EmbeddingClient,
    doc_hash: str,
    *,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[EmbeddingRow]:
    total = len(chunks)
    if total == 0:
        if progress_cb:
            progress_cb({"percent": 100.0, "processed": 0, "total": 0})
        return []

    rows: List[EmbeddingRow] = []
    batch_size = getattr(client, "max_batch", 1) or 1

    last_percent_reported = 0.0

    for start in range(0, total, batch_size):
        batch_chunks = chunks[start : start + batch_size]
        texts = [chunk["text"] for chunk in batch_chunks]
        vectors = await client.embed_batch(texts)
        if len(vectors) != len(batch_chunks):
            raise RuntimeError("Embedding result size mismatch")
        if client.dim is None:
            raise RuntimeError("Embedding dimension is unknown after embedding call")

        for chunk, vector in zip(batch_chunks, vectors):
            rows.append(
                EmbeddingRow(
                    chunk_id=chunk["chunk_id"],
                    doc_hash=doc_hash,
                    dim=client.dim,
                    model=client.model,
                    vector=vector,
                )
            )

        if progress_cb:
            done = min(total, start + len(batch_chunks))
            percent = 100.0 if total == 0 else (done / total) * 100.0
            last_percent_reported = percent
            progress_cb({"percent": percent, "processed": done, "total": total})

    if progress_cb and last_percent_reported < 100.0:
        progress_cb({"percent": 100.0, "processed": total, "total": total})

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
