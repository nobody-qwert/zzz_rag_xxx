from __future__ import annotations

from fastapi import APIRouter

try:
    from ..dependencies import document_store, jobs_registry
    from .helpers import format_document_row
    from ..services.ingestion import warmup_mineru
    from ..services.rag import warmup_llm
except ImportError:  # pragma: no cover
    from dependencies import document_store, jobs_registry  # type: ignore
    from routes.helpers import format_document_row  # type: ignore
    from services.ingestion import warmup_mineru  # type: ignore
    from services.rag import warmup_llm  # type: ignore

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/status")
async def system_status() -> dict:
    docs = await document_store.list_documents()
    ready = any((doc.get("status") == "processed") for doc in docs)
    running = [
        jid for jid, info in jobs_registry.items()
        if info.get("status") in {"queued", "running"}
    ]
    return {
        "ready": ready,
        "has_running_jobs": len(running) > 0,
        "running_jobs": [
            {
                "job_id": jid,
                "doc_hash": info.get("hash"),
                "status": info.get("status"),
                "file": info.get("file"),
            }
            for jid, info in jobs_registry.items()
            if info.get("status") in {"queued", "running"}
        ],
        "total_jobs": len(jobs_registry),
        "docs_count": sum(1 for doc in docs if str(doc.get("status")).lower() == "processed"),
        "total_docs": len(docs),
        "jobs": [
            {
                "job_id": jid,
                "doc_hash": info.get("hash"),
                "status": info.get("status"),
                "file": info.get("file"),
            }
            for jid, info in jobs_registry.items()
        ],
        "documents": [format_document_row(doc) for doc in docs],
        "llm_ready": False,
    }


@router.get("/ready")
async def ready() -> dict[str, bool]:
    ready_flag = (await document_store.count_documents(status="processed")) > 0
    return {"ready": ready_flag}


@router.post("/warmup")
async def warmup() -> dict:
    return await warmup_llm()


@router.post("/warmup/mineru")
async def warmup_mineru_route() -> dict:
    return await warmup_mineru()
