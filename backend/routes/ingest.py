from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, File, UploadFile

try:
    from ..services.ingestion import get_job_status, ingest_file, retry_ingest
except ImportError:  # pragma: no cover
    from services.ingestion import get_job_status, ingest_file, retry_ingest  # type: ignore

router = APIRouter()


@router.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await ingest_file(file)


@router.get("/status/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    return get_job_status(job_id)


@router.post("/ingest/{doc_hash}/retry")
async def retry_ingest_route(doc_hash: str) -> Dict[str, Any]:
    return await retry_ingest(doc_hash)
