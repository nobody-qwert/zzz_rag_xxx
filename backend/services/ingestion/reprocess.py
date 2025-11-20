from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

try:
    from ..embeddings import EmbeddingClient
except ImportError:  # pragma: no cover
    from embeddings import EmbeddingClient  # type: ignore

from .context import document_store, gpu_phase_manager, jobs_registry, settings
from .pipeline import build_chunk_specs, run_postprocess_pipeline

_reprocess_all_lock = asyncio.Lock()


async def reprocess_after_ocr(doc_hash: str, *, ensure_gpu_phase: bool = True) -> Dict[str, Any]:
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

    try:
        emb_client = EmbeddingClient()
        prev_metrics = await document_store.get_performance_metrics(doc_hash)
        ocr_time_val: Optional[float] = None
        if prev_metrics:
            try:
                raw_val = prev_metrics.get("ocr_time_sec")
                ocr_time_val = float(raw_val) if raw_val is not None else None
            except (TypeError, ValueError):
                ocr_time_val = prev_metrics.get("ocr_time_sec")

        result = await run_postprocess_pipeline(
            doc_hash,
            ocr_text=ocr_text,
            emb_client=emb_client,
            chunk_specs=build_chunk_specs(),
            ocr_time=ocr_time_val,
        )

        chunking_time = result["chunking_time_sec"]
        embedding_time = result["embedding_time_sec"]
        total_time = (ocr_time_val or 0.0) + chunking_time + embedding_time

        return {
            "hash": doc_hash,
            "document_name": doc.get("original_name") or doc_hash,
            "status": "processed",
            "chunk_count": result["total_chunks"],
            "small_chunks": result["small_chunks"],
            "large_chunks": result["large_chunks"],
            "total_embeddings": result["total_embeddings"],
            "chunking_time_sec": chunking_time,
            "embedding_time_sec": embedding_time,
            "ocr_time_sec": ocr_time_val,
            "total_time_sec": total_time,
        }
    except HTTPException:
        raise
    except Exception as exc:
        error_msg = str(exc)
        await document_store.update_document_status(doc_hash, "error", error=error_msg)
        raise HTTPException(status_code=500, detail=error_msg) from exc


async def reprocess_all_documents() -> Dict[str, Any]:
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
                detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
                if exc.status_code in (404, 409):
                    skipped += 1
                    results.append({"hash": doc_hash, "status": "skipped", "reason": detail})
                else:
                    failed += 1
                    results.append({"hash": doc_hash, "status": "error", "reason": detail})
            except Exception as exc:  # pragma: no cover
                failed += 1
                results.append({"hash": doc_hash, "status": "error", "reason": str(exc)})

        return {
            "total": total,
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            "results": results,
        }
