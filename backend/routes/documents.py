from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

try:
    from ..dependencies import document_store, settings
    from .helpers import format_document_row
    from ..services.ingestion import delete_document as delete_document_service
except ImportError:  # pragma: no cover
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
        status = str(doc_data.get("status") or "").strip().lower()
        if doc.get("doc_hash") and status in settings.completed_doc_statuses:
            metrics = await document_store.get_performance_metrics(doc["doc_hash"])
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
    max_chars: int = 2000,
) -> Dict[str, Any]:
    parser_key = parser or settings.ocr_parser_key
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    row = await document_store.get_extraction(doc_hash, parser_key)
    if not row:
        raise HTTPException(status_code=404, detail=f"No extraction found for parser '{parser_key}'")
    text = row["text"] or ""

    return {
        "parser": parser_key,
        "document_name": doc.get("original_name", "unknown"),
        "file_size": doc.get("size", 0),
        "extracted_chars": len(text),
        "preview_chars": min(max_chars, len(text)),
        "truncated": len(text) > max_chars,
        "preview": text[:max_chars],
    }
