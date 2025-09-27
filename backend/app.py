import os
import io
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Environment/config
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/indices"))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "lmstudio-or-token")

# Ensure directories exist (also created as volumes by docker-compose)
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG-Anything Backend", version="0.1.0")

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


def _load_rag() -> Any:
    """
    Lazy-load and initialize RAG-Anything. Tries a few common initialization patterns
    so this backend can work across raganything versions.
    """
    global _rag_instance, _rag_module
    if _rag_instance is not None:
        return _rag_instance

    try:
        import raganything as rag  # type: ignore
        _rag_module = rag
    except Exception as e:
        raise RuntimeError(f"Failed to import raganything: {e}")

    # Try to create an instance if the library exposes a class
    candidate_instance = None
    for cls_name in ["RAGAnything", "RAG", "RAGAny"]:
        if hasattr(_rag_module, cls_name):
            cls = getattr(_rag_module, cls_name)
            try:
                # Try various ctor signatures
                try:
                    candidate_instance = cls(index_dir=str(INDEX_DIR), data_dir=str(DATA_DIR))
                except Exception:
                    candidate_instance = cls()
                break
            except Exception:
                pass

    # If no class, some versions might be function-only; we'll store the module itself
    _rag_instance = candidate_instance if candidate_instance is not None else _rag_module

    # Propagate OpenAI-compatible env for LM Studio
    os.environ.setdefault("OPENAI_BASE_URL", OPENAI_BASE_URL or "")
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY or "lmstudio-or-token")

    return _rag_instance


async def _maybe_await(result_or_coro: Any) -> Any:
    if asyncio.iscoroutine(result_or_coro):
        return await result_or_coro
    return result_or_coro


def _safe_filename(name: str) -> str:
    # Basic sanitization
    keep = (" ", ".", "_", "-", "(", ")")
    cleaned = "".join(c for c in name if c.isalnum() or c in keep).strip().strip(".")
    return cleaned or "upload.bin"


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Any] = []


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/docs")
async def list_docs() -> List[Dict[str, Any]]:
    if not DATA_DIR.exists():
        return []
    items: List[Dict[str, Any]] = []
    for p in sorted(DATA_DIR.iterdir()):
        if not p.is_file():
            continue
        stat = p.stat()
        items.append(
            {
                "name": p.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "path": f"/data/{p.name}",  # inside container path, for reference
            }
        )
    return items


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> Dict[str, Any]:
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Save upload to DATA_DIR
    filename = _safe_filename(file.filename or "upload.bin")
    dest_path = DATA_DIR / filename

    try:
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
    finally:
        await file.close()

    # Run RAG-Anything ingestion
    try:
        rag = _load_rag()

        # Prefer instance method if available
        if hasattr(rag, "process_document_complete"):
            result = rag.process_document_complete(
                str(dest_path),
                output_dir=str(INDEX_DIR),
            )
            if asyncio.iscoroutine(result):
                await result
        else:
            # Try module-level function
            if hasattr(_rag_module, "process_document_complete"):
                result = _rag_module.process_document_complete(  # type: ignore
                    str(dest_path),
                    output_dir=str(INDEX_DIR),
                )
                if asyncio.iscoroutine(result):
                    await result
            # else: best effort; some versions may auto-index on read/query
    except Exception as e:
        # Do not fail hard on ingest; return error info for UI
        return {
            "status": "uploaded",
            "file": filename,
            "indexed": False,
            "error": f"Ingestion failed: {e}",
        }

    return {"status": "ok", "file": filename, "indexed": True}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    try:
        rag = _load_rag()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG-Anything unavailable: {e}")

    # Try preferred async query
    async def _run_query() -> Any:
        # Try instance aquery
        if hasattr(rag, "aquery"):
            return await _maybe_await(rag.aquery(req.query))
        # Try instance query
        if hasattr(rag, "query"):
            return await _maybe_await(rag.query(req.query))
        # Try module-level calls
        if _rag_module is not None:
            for fn_name in ["aquery", "query", "rag_query"]:
                if hasattr(_rag_module, fn_name):
                    fn = getattr(_rag_module, fn_name)
                    return await _maybe_await(fn(req.query))  # type: ignore
        raise RuntimeError("No query method found on RAG-Anything")

    try:
        result = await _run_query()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    # Normalize result shape
    answer: str = ""
    sources: List[Any] = []

    # Common shapes: dict with answer/sources/citations; object with attributes; plain string
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
                if isinstance(v, list):
                    sources = v
                else:
                    sources = [v]
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
