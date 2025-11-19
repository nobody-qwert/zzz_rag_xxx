from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, UploadFile

try:
    from ..services.ingestion import get_job_status, ingest_files, retry_ingest
except ImportError:  # pragma: no cover
    from services.ingestion import get_job_status, ingest_files, retry_ingest  # type: ignore

router = APIRouter()


@router.post("/ingest")
async def ingest(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    uploads: List[UploadFile] = []
    if files:
        uploads.extend(files)
    if file:
        uploads.append(file)
    return await ingest_files(uploads)


@router.get("/status/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    return get_job_status(job_id)


@router.post("/ingest/{doc_hash}/retry")
async def retry_ingest_route(doc_hash: str) -> Dict[str, Any]:
    return await retry_ingest(doc_hash)
