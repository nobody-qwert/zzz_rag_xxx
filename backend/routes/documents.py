from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

try:
    from ..chunking import count_text_tokens
    from ..dependencies import document_store, settings
    from .helpers import format_document_row
    from ..services.ingestion import delete_document as delete_document_service
except ImportError:  # pragma: no cover
    from chunking import count_text_tokens  # type: ignore
    from dependencies import document_store, settings  # type: ignore
    from routes.helpers import format_document_row  # type: ignore
    from services.ingestion import delete_document as delete_document_service  # type: ignore

router = APIRouter()


@router.get("/documents")
async def list_documents() -> List[Dict[str, Any]]:
    docs = await document_store.list_documents()
    results: List[Dict[str, Any]] = []
    for doc in docs:
        doc_data = format_document_row(doc)
        doc_hash = doc.get("doc_hash")
        status = str(doc_data.get("status") or "").strip().lower()

        ocr_available = False
        ocr_created_at: Optional[str] = None
        small_embeddings = 0
        large_embeddings = 0
        total_embeddings = 0

        if doc_hash:
            extraction = await document_store.get_extraction(doc_hash, settings.ocr_parser_key)
            if extraction:
                ocr_available = True
                ocr_created_at = extraction.get("created_at")

            embedding_counts = await document_store.count_embeddings_by_config(doc_hash)
            small_embeddings = int(embedding_counts.get(settings.chunk_config_small_id, 0))
            large_embeddings = int(embedding_counts.get(settings.chunk_config_large_id, 0))
            total_embeddings = int(sum(embedding_counts.values()))

        doc_data["ocr_available"] = ocr_available
        doc_data["ocr_extracted_at"] = ocr_created_at
        doc_data["small_embeddings"] = small_embeddings
        doc_data["large_embeddings"] = large_embeddings
        doc_data["total_embeddings"] = total_embeddings
        doc_data["embedding_available"] = total_embeddings > 0

        if doc_hash and status in settings.completed_doc_statuses:
            metrics = await document_store.get_performance_metrics(doc_hash)
            doc_data["performance"] = metrics

        results.append(doc_data)
    return results


@router.delete("/documents/{doc_hash}")
async def delete_document(doc_hash: str) -> Dict[str, Any]:
    return await delete_document_service(doc_hash)


@router.get("/parsers")
async def list_parsers() -> Dict[str, Any]:
    return {"default": settings.ocr_parser_key, "options": [settings.ocr_parser_key]}


@router.get("/debug/parsed_text/{doc_hash}")
async def debug_parsed_text(
    doc_hash: str,
    parser: Optional[str] = None,
    max_chars: Optional[int] = 2000,
) -> Dict[str, Any]:
    parser_key = parser or settings.ocr_parser_key
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    row = await document_store.get_extraction(doc_hash, parser_key)
    if not row:
        raise HTTPException(status_code=404, detail=f"No extraction found for parser '{parser_key}'")
    text = row["text"] or ""
    total_tokens = count_text_tokens(text)
    embedding_counts = await document_store.count_embeddings_by_config(doc_hash)
    small_embeddings = int(embedding_counts.get(settings.chunk_config_small_id, 0))
    large_embeddings = int(embedding_counts.get(settings.chunk_config_large_id, 0))
    total_embeddings = small_embeddings + large_embeddings

    text_length = len(text)
    unlimited_preview = max_chars is None or max_chars <= 0
    preview_limit = text_length if unlimited_preview else int(max_chars)
    preview_chars = min(preview_limit, text_length)

    return {
        "parser": parser_key,
        "document_name": doc.get("original_name", "unknown"),
        "file_size": doc.get("size", 0),
        "extracted_chars": len(text),
        "total_tokens": total_tokens,
        "preview_chars": preview_chars,
        "truncated": (not unlimited_preview) and text_length > preview_limit,
        "preview": text[:preview_chars],
        "chunk_count": total_embeddings,
        "total_embeddings": total_embeddings,
        "small_embeddings": small_embeddings,
        "large_embeddings": large_embeddings,
    }
