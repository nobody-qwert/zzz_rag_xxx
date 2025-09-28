import os
import io
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

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
    model = (os.environ.get("EMBEDDING_MODEL") or "text-embedding-nomic-embed-text-v1.5").strip()
    
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
    cfg = RAGAnythingConfig()

    # Initialize RAGAnything with both functions
    try:
        _rag_instance = RAGAnything(
            config=cfg,
            llm_model_func=llm,
            embedding_func=embedding,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize RAGAnything: {e}")

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
            await rag.process_document_complete(
                str(dest_path),
                output_dir=str(INDEX_DIR),
                parse_method="auto",
            )
        else:
            # Try module-level function
            if hasattr(_rag_module, "process_document_complete"):
                result = _rag_module.process_document_complete(  # type: ignore
                    str(dest_path),
                    output_dir=str(INDEX_DIR),
                    parse_method="auto",
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
        llm = None
        try:
            llm = _get_llm_model_func()
        except Exception:
            # If LLM unavailable, still try legacy methods
            llm = None

        # Try instance aquery
        if hasattr(rag, "aquery"):
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
                    return await _maybe_await(rag.query(req.query, llm_model_func=llm))
            except TypeError:
                pass
            return await _maybe_await(rag.query(req.query))

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
