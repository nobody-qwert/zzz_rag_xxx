import os
import json
import hashlib
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uuid import uuid4

from persistence import DocumentStore

class AskRequest(BaseModel):
    query: str = Field(..., description="The natural language question to ask the RAG system")

class AskResponse(BaseModel):
    answer: str = Field(..., description="The generated response to the query")
    sources: List[Any] = Field(default=[], description="List of source documents or chunks used to generate the answer")

jobs: dict[str, dict] = {}


# Environment/config
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/indices"))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "lmstudio-or-token")

# Ensure directories exist (also created as volumes by docker-compose)
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

DOC_STORE_PATH = Path(os.environ.get("DOC_STORE_PATH") or (DATA_DIR / "rag_meta.db"))

document_store = DocumentStore(DOC_STORE_PATH)

_llm_warm: bool = False


logger = logging.getLogger(__name__)


async def _ingest_pdf_with_pymupdf(
    rag: Any,
    path: Path,
    display_name: str,
    doc_hash: str,
) -> None:
    """Rapid PDF text ingestion using PyMuPDF, falling back to LightRAG insert."""

    try:
        import fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - handled by fallback
        raise RuntimeError(
            "PyMuPDF is not installed; add 'PyMuPDF' to backend requirements."
        ) from exc

    def _extract_pages() -> List[Dict[str, Any]]:
        doc = fitz.open(path)
        try:
            items: List[Dict[str, Any]] = []
            for page_index, page in enumerate(doc):
                text = page.get_text("text")
                if text and text.strip():
                    items.append(
                        {
                            "type": "text",
                            "page_idx": page_index,
                            "page_number": page_index + 1,
                            "text": text,
                        }
                    )
            return items
        finally:
            doc.close()

    content_items = await asyncio.to_thread(_extract_pages)
    if not content_items:
        raise ValueError("PyMuPDF extracted no textual content from the PDF")

    # Combine content similar to raganything.utils.separate_content
    combined_text = "\n\n".join(item["text"].strip() for item in content_items if item["text"].strip())
    if not combined_text:
        raise ValueError("PyMuPDF produced empty text after cleanup")

    try:
        from raganything.utils import insert_text_content  # type: ignore
    except Exception as exc:  # pragma: no cover - should not happen in runtime image
        raise RuntimeError(f"Failed to import raganything utilities: {exc}") from exc

    await insert_text_content(
        rag.lightrag,
        input=combined_text,
        file_paths=display_name,
        ids=doc_hash,
    )

    doc_status_accessor = getattr(rag, "lightrag", None)
    if doc_status_accessor is None:
        return

    doc_status = getattr(doc_status_accessor, "doc_status", None)
    if doc_status is None:
        return

    try:
        current = await doc_status.get_by_id(doc_hash)
        payload: Dict[str, Any] = {
            "doc_id": doc_hash,
            "status": "PROCESSED",
            "text_processed": True,
            "multimodal_processed": True,
        }
        if current:
            payload.update({k: v for k, v in current.items() if k not in payload})
        await doc_status.upsert(payload)
        if hasattr(doc_status, "index_done_callback"):
            await doc_status.index_done_callback()
    except Exception:
        logger.exception("Failed to update LightRAG doc_status for %s", doc_hash)


def _sanitize_doc_status_artifacts() -> None:
    """Remove incompatible fields from LightRAG doc status cache files."""
    try:
        if not INDEX_DIR.exists():
            return
        for path in INDEX_DIR.rglob("*doc_status*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

            if not isinstance(data, dict):
                continue

            modified = False
            for doc_id, meta in list(data.items()):
                if isinstance(meta, dict) and "multimodal_processed" in meta:
                    meta = dict(meta)
                    meta.pop("multimodal_processed", None)
                    data[doc_id] = meta
                    modified = True

            if modified:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, sort_keys=True)
                except OSError:
                    continue
    except Exception:
        # Best effort cleanup; ignore unexpected errors.
        pass


def _purge_doc_status_entry(doc_id: str) -> None:
    """Remove a document entry from LightRAG doc status caches."""
    try:
        if not INDEX_DIR.exists():
            return
        for path in INDEX_DIR.rglob("*doc_status*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

            if not isinstance(data, dict) or doc_id not in data:
                continue

            try:
                del data[doc_id]
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, sort_keys=True)
            except OSError:
                continue
    except Exception:
        pass


async def _restore_jobs() -> None:
    """Best-effort reconciliation of job state after a backend restart."""
    global _llm_warm
    _llm_warm = False
    active_jobs = await document_store.list_active_jobs()
    jobs.clear()
    if not active_jobs:
        return

    documents = {doc["doc_hash"]: doc for doc in await document_store.list_documents()}

    interruption_reason = "Interrupted during restart"
    for record in active_jobs:
        doc_hash = record.get("doc_hash") or "unknown"
        doc_meta = documents.get(doc_hash)
        display_name = (doc_meta or {}).get("original_name") or doc_hash

        jobs[record["job_id"]] = {
            "status": "error: interrupted",
            "file": display_name,
            "hash": doc_hash,
        }

        # Mark the job/doc as errored so new ingests can proceed cleanly.
        await document_store.finish_job(
            record["job_id"],
            status="error",
            error=interruption_reason,
        )

        if doc_meta and doc_meta.get("status") != "processed":
            await document_store.update_document_status(
                doc_hash,
                "error",
                error=interruption_reason,
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await document_store.init()
    await _restore_jobs()
    yield


app = FastAPI(title="RAG-Anything Backend", version="0.1.0", lifespan=lifespan)

# CORS for local dev frontend (adjust as needed)
frontend_origin = f"http://localhost:{os.environ.get('FRONTEND_PORT', '5173')}"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rag_instance: Optional[Any] = None
_rag_module: Optional[Any] = None

# LLM model function cache for LightRAG
_LLM_MODEL_FUNC: Optional[Callable[..., str]] = None

def _get_llm_model_func() -> Callable[..., str]:
    """
    Create (once) and return a function(prompt, ...) -> str that calls LM Studio via its
    OpenAI-compatible API. The function signature is flexible to match LightRAG expectations.
    """
    global _LLM_MODEL_FUNC
    if _LLM_MODEL_FUNC is not None:
        return _LLM_MODEL_FUNC

    base = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    model = (os.environ.get("LLM_MODEL") or "").strip()
    if not base or not key or not model:
        raise RuntimeError("OPENAI_BASE_URL, OPENAI_API_KEY, and LLM_MODEL must be set")

    # Import locally to avoid import error before deps are installed
    try:
        from openai import AsyncOpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(f"OpenAI SDK not available: {e}")

    client = AsyncOpenAI(base_url=base, api_key=key)

    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        text = resp.choices[0].message.content or ""
        return text.strip()

    _LLM_MODEL_FUNC = llm_model_func
    return _LLM_MODEL_FUNC


def _get_embedding_func() -> Any:
    """
    Create and return an embedding function that calls LM Studio's OpenAI-compatible embeddings API.
    """
    base = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    model = (os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small").strip()
    
    if not base or not key or not model:
        raise RuntimeError("OPENAI_BASE_URL, OPENAI_API_KEY, and EMBEDDING_MODEL must be set")

    # Infer embedding dimension based on model
    dim = 768  # Default for nomic-embed-text-v1.5
    model_lower = model.lower()
    if "text-embedding-3-large" in model_lower:
        dim = 3072
    elif "text-embedding-3-small" in model_lower:
        dim = 1536

    # Import locally to avoid import error before deps are installed
    try:
        from lightrag.utils import EmbeddingFunc  # type: ignore
        from lightrag.llm.openai import openai_embed  # type: ignore
    except Exception as e:
        raise RuntimeError(f"LightRAG dependencies not available: {e}")

    async def _embed_async(texts: List[str]) -> List[List[float]]:
        embeddings = await openai_embed(
            texts=texts,
            model=model,
            base_url=base,
            api_key=key,
        )
        return embeddings.tolist()

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=8192,
        func=_embed_async,
    )


def _load_rag() -> Any:
    """
    Initialize RAG-Anything with proper LLM and embedding functions.
    """
    global _rag_instance, _rag_module
    if _rag_instance is not None:
        _sanitize_doc_status_artifacts()
        return _rag_instance

    try:
        import raganything as rag  # type: ignore
        from raganything import RAGAnything, RAGAnythingConfig  # type: ignore
        _rag_module = rag
    except Exception as e:
        raise RuntimeError(f"Failed to import raganything: {e}")

    # Get required functions
    try:
        llm = _get_llm_model_func()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM function: {e}")
    
    try:
        embedding = _get_embedding_func()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize embedding function: {e}")

    # Create configuration (reads from environment variables)
    cfg = RAGAnythingConfig(working_dir=str(INDEX_DIR))

    # Initialize RAGAnything with both functions
    try:
        _rag_instance = RAGAnything(
            config=cfg,
            llm_model_func=llm,
            embedding_func=embedding,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize RAG-Anything: {e}")

    # Propagate OpenAI-compatible env for LM Studio
    os.environ.setdefault("OPENAI_BASE_URL", OPENAI_BASE_URL or "")
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY or "lmstudio-or-token")

    # Compatibility: LightRAG versions before multimodal support reject extra doc_status fields.
    if hasattr(_rag_instance, "_mark_multimodal_processing_complete"):
        async def _noop_mark_multimodal(doc_id: str) -> None:
            return None

        _rag_instance._mark_multimodal_processing_complete = _noop_mark_multimodal  # type: ignore[attr-defined]

    _sanitize_doc_status_artifacts()

    return _rag_instance


async def _ensure_rag_initialized(rag: Any) -> None:
    """Ensure the provided RAG-Anything instance has LightRAG ready for use."""
    if rag is None:
        raise RuntimeError("RAG-Anything instance unavailable")

    async def _run(fn: Callable[[], Any]) -> None:
        result = await _maybe_await(fn())
        if isinstance(result, dict):
            if not result.get("success", True):
                raise RuntimeError(result.get("error") or "LightRAG initialization failed")
        elif isinstance(result, bool):
            if not result:
                raise RuntimeError("LightRAG initialization failed")

    for attr in ("_ensure_lightrag_initialized", "ensure_lightrag_initialized"):
        ensure_fn = getattr(rag, attr, None)
        if ensure_fn is None:
            continue
        await _run(ensure_fn)
        break
    else:
        if getattr(rag, "lightrag", None) is None:
            raise RuntimeError("No LightRAG instance available and no initializer found")


async def _maybe_await(result_or_coro: Any) -> Any:
    if asyncio.iscoroutine(result_or_coro):
        return await result_or_coro
    return result_or_coro


def _safe_filename(name: str) -> str:
    # Basic sanitization
    keep = (" ", ".", "_", "-", "(", ")")
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip().strip(".")
    return cleaned or "upload.bin"


def _has_index() -> bool:
    """Return True if at least one index file exists in INDEX_DIR."""
    return any(p.is_file() for p in INDEX_DIR.iterdir())

async def _has_ready_index() -> bool:
    """
    Return True only when at least one document is fully processed and has chunks indexed.
    Prefer LightRAG's doc_status, with a filesystem fallback.
    """
    if await document_store.count_documents(status="processed") > 0:
        return True

    try:
        rag = _load_rag()
        l = getattr(rag, "lightrag", None)
        if l is not None and hasattr(l, "doc_status") and hasattr(l.doc_status, "get_all"):
            all_status = await _maybe_await(l.doc_status.get_all())
            if all_status:
                for s in all_status:
                    try:
                        status = str((s.get("status") or "")).upper()
                        chunks = int(s.get("chunks_count") or 0)
                        if status == "PROCESSED" and chunks > 0:
                            return True
                    except Exception:
                        continue
    except Exception:
        # Ignore and fallback to filesystem probe
        pass

    # Fallback: look for index-like artifacts in INDEX_DIR
    try:
        for p in INDEX_DIR.rglob("*"):
            if p.is_file() and p.stat().st_size > 0 and p.suffix.lower() in {".json", ".parquet", ".bin", ".idx", ".faiss", ".sqlite", ".db"}:
                return True
    except Exception:
        pass
    return False

async def _process_job(job_id: str, path: Path, doc_hash: str, display_name: str) -> None:
    """
    Background task that processes a document and updates the jobs dict.
    """
    start_time = time.perf_counter()
    total_steps = 6

    def _log(step: int, message: str) -> None:
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Ingest %s (%s) step %d/%d: %s [%.2fs elapsed]",
            doc_hash[:8],
            display_name,
            step,
            total_steps,
            message,
            elapsed,
        )

    try:
        await document_store.mark_job_started(job_id)
        _log(1, "Job marked as started")
    except Exception:
        # Non-fatal: continue best-effort even if persistence update fails.
        pass

    job_record = jobs.setdefault(job_id, {"status": "running", "file": display_name, "hash": doc_hash})
    job_record.update({"status": "running", "file": display_name, "hash": doc_hash})

    try:
        await document_store.update_document_status(doc_hash, "processing")
        _log(2, "Document status set to processing")
    except Exception:
        pass

    try:
        _purge_doc_status_entry(doc_hash)
        _log(3, "Cleared cached doc status metadata")
        rag = _load_rag()
        await _ensure_rag_initialized(rag)
        _log(4, "Loaded RAG-Anything pipeline")

        used_pymupdf = False
        if path.suffix.lower() == ".pdf":
            try:
                _log(5, "Using PyMuPDF fast-path for PDF text ingestion")
                await _ingest_pdf_with_pymupdf(rag, path, display_name, doc_hash)
                used_pymupdf = True
            except Exception as exc:
                logger.exception(
                    "PyMuPDF ingestion failed for %s; falling back to RagAnything pipeline (%s)",
                    display_name,
                    exc,
                )

        if not used_pymupdf:
            _log(5, "Delegating document to RagAnything parser pipeline")
            # Prefer instance method if available
            if hasattr(rag, "process_document_complete"):
                await rag.process_document_complete(
                    str(path),
                    output_dir=str(INDEX_DIR),
                    parse_method="auto",
                    doc_id=doc_hash,
                )
            else:
                # Fallback to module-level function
                if hasattr(_rag_module, "process_document_complete"):
                    result = _rag_module.process_document_complete(  # type: ignore
                        str(path),
                        output_dir=str(INDEX_DIR),
                        parse_method="auto",
                        doc_id=doc_hash,
                    )
                    if asyncio.iscoroutine(result):
                        await result
        jobs[job_id]["status"] = "done"
        await document_store.mark_document_processed(doc_hash)
        await document_store.finish_job(job_id, "done")
        _log(6, "Document ingestion finished and metadata persisted")
    except asyncio.CancelledError:
        jobs[job_id]["status"] = "cancelled"
        await document_store.finish_job(job_id, "cancelled", error="Processing cancelled")
        await document_store.update_document_status(doc_hash, "error", error="Processing cancelled")
        _log(total_steps, "Document ingestion cancelled")
        raise
    except Exception as e:
        jobs[job_id]["status"] = f"error: {e}"
        await document_store.finish_job(job_id, "error", error=str(e))
        await document_store.update_document_status(doc_hash, "error", error=str(e))
        _log(total_steps, f"Document ingestion failed: {e}")
        raise


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/docs")
async def list_docs() -> List[Dict[str, Any]]:
    documents = await document_store.list_documents()
    items: List[Dict[str, Any]] = []
    for doc in documents:
        stored_name = doc.get("stored_name") or doc.get("doc_hash") or "unknown"
        file_path = DATA_DIR / stored_name
        size = doc.get("size")
        if size in (None, 0) and file_path.exists():
            try:
                size = file_path.stat().st_size
            except OSError:
                size = 0

        items.append(
            {
                "name": doc.get("original_name") or stored_name,
                "size": size or 0,
                "status": doc.get("status") or "unknown",
                "hash": doc.get("doc_hash"),
                "stored_name": stored_name,
                "path": f"/data/{stored_name}",
                "last_ingested_at": doc.get("last_ingested_at"),
                "error": doc.get("error"),
                "updated_at": doc.get("updated_at"),
                "created_at": doc.get("created_at"),
            }
        )
    return items


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> Dict[str, Any]:
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    display_name = _safe_filename(file.filename or "upload.bin")

    try:
        content = await file.read()
    finally:
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

    if existing and existing.get("status") in {"pending", "processing"}:
        active_job_id = next(
            (
                job_id
                for job_id, info in jobs.items()
                if info.get("hash") == doc_hash
                and info.get("status") in {"queued", "running"}
            ),
            None,
        )
        if active_job_id is None:
            active_records = await document_store.list_active_jobs()
            for record in active_records:
                if record.get("doc_hash") == doc_hash:
                    active_job_id = record.get("job_id")
                    break
        return {
            "job_id": active_job_id,
            "status": "already_processing",
            "file": existing.get("original_name") or display_name,
            "hash": doc_hash,
            "message": "Document ingestion already in progress",
        }

    suffix = Path(display_name).suffix
    stored_name = f"{doc_hash}{suffix.lower()}" if suffix else doc_hash
    dest_path = DATA_DIR / stored_name

    if existing and existing.get("stored_name") and existing["stored_name"] != stored_name:
        old_path = DATA_DIR / existing["stored_name"]
        if old_path.exists():
            try:
                old_path.unlink()
            except Exception:
                pass

    with open(dest_path, "wb") as f:
        f.write(content)

    await document_store.upsert_document(
        doc_hash=doc_hash,
        original_name=display_name,
        stored_name=stored_name,
        size=len(content),
    )

    _purge_doc_status_entry(doc_hash)

    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued", "file": display_name, "hash": doc_hash}
    await document_store.create_job(job_id, doc_hash)

    asyncio.create_task(_process_job(job_id, dest_path, doc_hash, display_name))

    return {
        "job_id": job_id,
        "status": "queued",
        "file": display_name,
        "hash": doc_hash,
    }


@app.post("/ingest/{doc_hash}/retry")
async def retry_ingest(doc_hash: str) -> Dict[str, Any]:
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    status = str(doc.get("status") or "").lower()
    if status in {"pending", "processing"}:
        raise HTTPException(status_code=409, detail="Document ingestion already in progress")

    stored_name = doc.get("stored_name") or doc_hash
    source_path = DATA_DIR / stored_name
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Stored document not found on disk")

    for job_id, info in jobs.items():
        if info.get("hash") == doc_hash and info.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="Document ingestion already queued")

    display_name = doc.get("original_name") or stored_name

    await document_store.update_document_status(doc_hash, "pending", error=None)

    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued", "file": display_name, "hash": doc_hash}
    await document_store.create_job(job_id, doc_hash)

    _purge_doc_status_entry(doc_hash)

    asyncio.create_task(_process_job(job_id, source_path, doc_hash, display_name))

    return {
        "job_id": job_id,
        "status": "queued",
        "file": display_name,
        "hash": doc_hash,
        "retry": True,
    }


@app.get("/status/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    info = jobs.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": info.get("status"),
        "file": info.get("file"),
        "hash": info.get("hash"),
    }


@app.get("/ready")
async def ready() -> Dict[str, bool]:
    """Return True when at least one indexed document exists and is processed."""
    return {"ready": await _has_ready_index()}


@app.get("/status")
async def global_status() -> Dict[str, Any]:
    """Return global system status including running jobs, readiness, document count, and job summaries."""
    processed_docs = await document_store.count_documents(status="processed")
    total_docs = await document_store.count_documents()
    job_summaries = await document_store.list_jobs()

    running_jobs = [
        job_id
        for job_id, info in jobs.items()
        if info.get("status") == "running"
    ]
    active_jobs = [
        job_id
        for job_id, info in jobs.items()
        if info.get("status") in {"queued", "running"}
    ]

    ready = processed_docs > 0
    if not ready:
        ready = await _has_ready_index()

    return {
        "ready": ready,
        "has_running_jobs": len(active_jobs) > 0,
        "running_jobs": running_jobs,
        "total_jobs": len(job_summaries),
        "docs_count": processed_docs,
        "total_docs": total_docs,
        "jobs": job_summaries,
        "llm_ready": _llm_warm,
    }


@app.post("/warmup")
async def warmup() -> Dict[str, Any]:
    """Send a simple warmup query to ensure LLM is loaded and ready."""
    global _llm_warm
    try:
        rag = _load_rag()
        await _ensure_rag_initialized(rag)
        llm = _get_llm_model_func()

        await llm("Warm up for upcoming chat sessions.")
        _llm_warm = True
        return {"status": "ready", "warmup_complete": True}
    except Exception as e:
        _llm_warm = False
        return {"status": "error", "error": str(e), "warmup_complete": False}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    # Ensure indices are fully ready (embeddings/graph built)
    processed_docs = await document_store.count_documents(status="processed")
    if processed_docs == 0:
        raise HTTPException(
            status_code=400,
            detail="Documents are not fully indexed yet – wait for ingestion to complete",
        )

    try:
        rag = _load_rag()
        await _ensure_rag_initialized(rag)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG-Anything unavailable: {e}")

    # Try preferred async query
    async def _run_query() -> Any:
        llm = None
        try:
            llm = _get_llm_model_func()
        except Exception:
            llm = None

        # Try instance aquery
        if hasattr(rag, "aquery"):
            # Prefer to disable rerank if supported to avoid missing-model warnings
            try:
                if llm:
                    return await _maybe_await(rag.aquery(req.query, llm_model_func=llm, enable_rerank=False))
            except TypeError:
                pass
            try:
                return await _maybe_await(rag.aquery(req.query, enable_rerank=False))
            except TypeError:
                pass
            try:
                if llm:
                    return await _maybe_await(rag.aquery(req.query, llm_model_func=llm))
            except TypeError:
                pass
            return await _maybe_await(rag.aquery(req.query))

        # Try instance query
        if hasattr(rag, "query"):
            try:
                if llm:
                    return await _maybe_await(rag.query(req.query, llm_model_func=llm, enable_rerank=False))
            except TypeError:
                pass
            try:
                return await _maybe_await(rag.query(req.query, enable_rerank=False))
            except TypeError:
                pass

        # Try module-level calls
        if _rag_module is not None:
            for fn_name in ["aquery", "query", "rag_query"]:
                if hasattr(_rag_module, fn_name):
                    fn = getattr(_rag_module, fn_name)
                    try:
                        if llm:
                            return await _maybe_await(fn(req.query, llm_model_func=llm))  # type: ignore
                    except TypeError:
                        pass
                    return await _maybe_await(fn(req.query))  # type: ignore
        raise RuntimeError("No query method found on RAG-Anything")

    try:
        result = await _run_query()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    # Normalize result shape
    answer: str = ""
    sources: List[Any] = []

    if isinstance(result, dict):
        answer = (
            result.get("answer")
            or result.get("text")
            or result.get("output")
            or result.get("response")
            or ""
        )
        for key in ["citations", "sources", "documents", "refs", "chunks"]:
            v = result.get(key)
            if v:
                sources = v if isinstance(v, list) else [v]
                break
    elif hasattr(result, "answer"):
        answer = getattr(result, "answer", "")
        for key in ["citations", "sources", "documents", "refs", "chunks"]:
            if hasattr(result, key):
                v = getattr(result, key)
                sources = v if isinstance(v, list) else [v]
                break
    else:
        answer = str(result)

    return AskResponse(answer=answer, sources=sources)


# For local dev run: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
