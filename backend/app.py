import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from uuid import uuid4

from persistence import DocumentStore, EmbeddingRow
from mineru_wrapper import run_mineru, warmup_mineru
from chunking import sliding_token_chunks
from embeddings import EmbeddingClient


logger = logging.getLogger(__name__)


DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/indices"))
DOC_STORE_PATH = Path(os.environ.get("DOC_STORE_PATH") or (DATA_DIR / "rag_meta.db"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

document_store = DocumentStore(DOC_STORE_PATH)

# Parser strategy: 'mineru' | 'pymupdf' | 'auto'
PARSER_MODE = (os.environ.get("PARSER_MODE") or "mineru").strip().lower()
MIN_PYMUPDF_CHARS_PER_PAGE = int(os.environ.get("MIN_PYMUPDF_CHARS_PER_PAGE", "300") or 300)
MINERU_LANG = (os.environ.get("MINERU_LANG") or "en").strip()
MINERU_PARSE_METHOD = (os.environ.get("MINERU_PARSE_METHOD") or "auto").strip().lower()
MINERU_TABLE_ENABLE = (os.environ.get("MINERU_TABLE_ENABLE") or "true").strip().lower()
MINERU_FORMULA_ENABLE = (os.environ.get("MINERU_FORMULA_ENABLE") or "true").strip().lower()
MINERU_WARMUP_ON_STARTUP = (os.environ.get("MINERU_WARMUP_ON_STARTUP") or "false").strip().lower()


class AskRequest(BaseModel):
    query: str = Field(...)
    top_k: int = Field(5, ge=1, le=50)


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)


jobs: Dict[str, Dict[str, Any]] = {}


def _safe_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (".", "_", "-", " ")).strip()[:255] or "upload.bin"


async def _extract_pymupdf(path: Path) -> Dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyMuPDF not installed") from exc

    def _run() -> Dict[str, Any]:
        doc = fitz.open(path)
        try:
            parts: List[str] = []
            for page in doc:
                t = page.get_text("text")
                if t and t.strip():
                    parts.append(t)
            return {"text": "\n\n".join(p.strip() for p in parts if p.strip()), "pages": len(doc)}
        finally:
            doc.close()

    return await asyncio.to_thread(_run)


async def _compute_embeddings_for_chunks(chunks: List[Dict[str, Any]], client: EmbeddingClient, doc_hash: str) -> List[EmbeddingRow]:
    texts = [c["text"] for c in chunks]
    vectors = await client.embed_batch(texts)
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding result size mismatch")

    rows: List[EmbeddingRow] = []
    for c, v in zip(chunks, vectors):
        rows.append(
            EmbeddingRow(
                chunk_id=c["chunk_id"],
                doc_hash=doc_hash,
                dim=client.dim,
                model=client.model,
                vector=v,
            )
        )
    return rows


async def _process_job(job_id: str, doc_path: Path, doc_hash: str, display_name: str) -> None:
    import time
    
    async def update(status: str, *, error: Optional[str] = None) -> None:
        jobs[job_id]["status"] = status if not error else f"error: {error}"
        await document_store.update_document_status(doc_hash, status if not error else "error", error=error)

    # Performance tracking
    start_total = time.perf_counter()
    pymupdf_time = None
    mineru_time = None
    chunking_time = None
    embedding_time = None

    try:
        await document_store.mark_job_started(job_id)
        await document_store.update_document_status(doc_hash, "processing")

        # 1) PyMuPDF fast extraction
        start_pymupdf = time.perf_counter()
        py_out = await _extract_pymupdf(doc_path)
        pymupdf_time = time.perf_counter() - start_pymupdf
        await document_store.upsert_extraction(doc_hash, "pymupdf", text=py_out["text"], meta_json=json.dumps({"pages": py_out.get("pages", 0)}))

        # Decide whether to run MinerU based on PARSER_MODE
        use_mineru = True
        pymupdf_text = py_out.get("text", "") or ""
        pages = int(py_out.get("pages", 0) or 0)
        if PARSER_MODE == "pymupdf":
            use_mineru = False
        elif PARSER_MODE == "auto":
            # If PyMuPDF extracted sufficient text per page, skip OCR-rich path
            cpp = (len(pymupdf_text) / max(1, pages)) if pages else len(pymupdf_text)
            use_mineru = cpp < float(MIN_PYMUPDF_CHARS_PER_PAGE)

        if use_mineru:
            # 2) MinerU rich extraction (GPU when available)
            start_mineru = time.perf_counter()
            mineru_dir = INDEX_DIR / f"mineru/{doc_hash}"
            mineru_res = await asyncio.to_thread(
                run_mineru,
                doc_path,
                mineru_dir,
                parse_method=MINERU_PARSE_METHOD,
                lang=MINERU_LANG,
                table_enable=(MINERU_TABLE_ENABLE in {"1", "true", "yes", "on"}),
                formula_enable=(MINERU_FORMULA_ENABLE in {"1", "true", "yes", "on"}),
            )
            mineru_time = time.perf_counter() - start_mineru
            await document_store.upsert_extraction(doc_hash, "mineru", text=mineru_res.text, meta_json=json.dumps(mineru_res.metadata))
            chosen_text = mineru_res.text
            chosen_meta = {"parser_used": "mineru"}
        else:
            # Skip MinerU; use PyMuPDF-only text for downstream steps but store under 'mineru' chunks for compatibility
            chosen_text = pymupdf_text
            chosen_meta = {"parser_used": "pymupdf", "reason": "auto-skip or configured"}
            await document_store.upsert_extraction(doc_hash, "mineru", text=chosen_text, meta_json=json.dumps(chosen_meta))

        # 3) Chunk from MinerU text
        start_chunking = time.perf_counter()
        size = int(os.environ.get("CHUNK_SIZE", "500") or 500)
        overlap = int(os.environ.get("CHUNK_OVERLAP", "100") or 100)
        chunks = sliding_token_chunks(chosen_text, size=size, overlap=overlap)
        chunk_rows = [(c.chunk_id, c.order_index, c.text, c.token_count) for c in chunks]
        await document_store.replace_chunks(doc_hash, "mineru", chunk_rows)
        chunking_time = time.perf_counter() - start_chunking

        # 4) Embeddings (LM Studio)
        start_embedding = time.perf_counter()
        emb_client = EmbeddingClient()
        rows = await _compute_embeddings_for_chunks(
            [
                {"chunk_id": cid, "order_index": idx, "text": txt, "token_count": tok}
                for (cid, idx, txt, tok) in chunk_rows
            ],
            emb_client,
            doc_hash,
        )
        await document_store.replace_embeddings(rows)
        embedding_time = time.perf_counter() - start_embedding

        total_time = time.perf_counter() - start_total

        # Save performance metrics
        await document_store.save_performance_metrics(
            doc_hash,
            pymupdf_time_sec=pymupdf_time,
            mineru_time_sec=mineru_time,
            chunking_time_sec=chunking_time,
            embedding_time_sec=embedding_time,
            total_time_sec=total_time,
        )

        await document_store.mark_document_processed(doc_hash)
        await document_store.finish_job(job_id, "done")
        jobs[job_id]["status"] = "done"
    except asyncio.CancelledError:
        await document_store.finish_job(job_id, "cancelled", error="cancelled")
        await update("error", error="cancelled")
        raise
    except Exception as exc:
        await document_store.finish_job(job_id, "error", error=str(exc))
        await update("error", error=str(exc))
        logger.exception("Job %s failed: %s", job_id, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await document_store.init()
    # Optional MinerU warmup to eliminate first-ingest model init latency
    if MINERU_WARMUP_ON_STARTUP in {"1", "true", "yes", "on"}:
        async def _do_warmup() -> None:
            try:
                await asyncio.to_thread(
                    warmup_mineru,
                    parse_method=MINERU_PARSE_METHOD,
                    lang=MINERU_LANG,
                    table_enable=(MINERU_TABLE_ENABLE in {"1", "true", "yes", "on"}),
                    formula_enable=(MINERU_FORMULA_ENABLE in {"1", "true", "yes", "on"}),
                )
                logger.info("MinerU warmup completed")
            except Exception as exc:
                logger.warning("MinerU warmup failed: %s", exc)

        asyncio.create_task(_do_warmup())
    yield


app = FastAPI(title="RAG MinerU Backend", version="0.1.0", lifespan=lifespan)

frontend_origin = f"http://localhost:{os.environ.get('FRONTEND_PORT', '5173')}"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://localhost", "http://127.0.0.1"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _format_document_row(doc: Dict[str, Any]) -> Dict[str, Any]:
    stored_name = doc.get("stored_name") or doc.get("doc_hash")
    file_path = DATA_DIR / stored_name
    size = doc.get("size")
    try:
        if (size in (None, 0)) and file_path.exists():
            size = file_path.stat().st_size
    except Exception:
        size = size or 0
    return {
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


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
async def system_status() -> Dict[str, Any]:
    docs = await document_store.list_documents()
    ready = any((d.get("status") == "processed") for d in docs)
    active = [jid for jid, info in jobs.items() if info.get("status") in {"queued", "running"}]
    return {
        "ready": ready,
        "has_running_jobs": len(active) > 0,
        "running_jobs": [
            {
                "job_id": jid,
                "doc_hash": info.get("hash"),
                "status": info.get("status"),
                "file": info.get("file"),
            }
            for jid, info in jobs.items()
            if info.get("status") in {"queued", "running"}
        ],
        "total_jobs": len(jobs),
        "docs_count": sum(1 for d in docs if str(d.get("status")).lower() == "processed"),
        "total_docs": len(docs),
        "jobs": [
            {
                "job_id": jid,
                "doc_hash": info.get("hash"),
                "status": info.get("status"),
                "file": info.get("file"),
            }
            for jid, info in jobs.items()
        ],
        "documents": [_format_document_row(d) for d in docs],
        "llm_ready": False,
    }


@app.get("/documents")
async def list_docs() -> List[Dict[str, Any]]:
    docs = await document_store.list_documents()
    result = []
    for d in docs:
        doc_data = _format_document_row(d)
        # Add performance metrics if available
        if d.get("doc_hash"):
            metrics = await document_store.get_performance_metrics(d["doc_hash"])
            doc_data["performance"] = metrics
        result.append(doc_data)
    return result


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> Dict[str, Any]:
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    display_name = _safe_filename(file.filename or "upload.bin")
    content = await file.read()
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

    suffix = Path(display_name).suffix
    stored_name = f"{doc_hash}{suffix.lower()}" if suffix else doc_hash
    dest_path = DATA_DIR / stored_name
    dest_path.write_bytes(content)

    await document_store.upsert_document(
        doc_hash=doc_hash,
        original_name=display_name,
        stored_name=stored_name,
        size=len(content),
    )

    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued", "file": display_name, "hash": doc_hash}
    await document_store.create_job(job_id, doc_hash)
    asyncio.create_task(_process_job(job_id, dest_path, doc_hash, display_name))

    return {"job_id": job_id, "status": "queued", "file": display_name, "hash": doc_hash}


@app.get("/status/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    info = jobs.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **info}


@app.post("/ingest/{doc_hash}/retry")
async def retry_ingest(doc_hash: str) -> Dict[str, Any]:
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    stored_name = doc.get("stored_name") or doc_hash
    source_path = DATA_DIR / stored_name
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Stored document missing")

    display_name = doc.get("original_name") or stored_name
    job_id = str(uuid4())
    jobs[job_id] = {"status": "queued", "file": display_name, "hash": doc_hash}
    await document_store.create_job(job_id, doc_hash)
    asyncio.create_task(_process_job(job_id, source_path, doc_hash, display_name))
    return {"job_id": job_id, "status": "queued", "file": display_name, "hash": doc_hash, "retry": True}


@app.get("/debug/parsed_text/{doc_hash}")
async def debug_parsed_text(doc_hash: str, parser: str = "mineru", max_chars: int = 2000) -> Dict[str, Any]:
    doc = await document_store.get_document(doc_hash)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    row = await document_store.get_extraction(doc_hash, parser)
    if not row:
        raise HTTPException(status_code=404, detail=f"No extraction found for parser '{parser}'")
    text = row["text"] or ""
    
    # Get chunk statistics for this document
    chunks = await document_store.fetch_chunks(doc_hash=doc_hash, parser=parser)
    total_tokens = sum(c.get("token_count", 0) for c in chunks)
    chunk_count = len(chunks)
    
    return {
        "parser": parser,
        "document_name": doc.get("original_name", "unknown"),
        "file_size": doc.get("size", 0),
        "extracted_chars": len(text),
        "total_tokens": total_tokens,
        "chunk_count": chunk_count,
        "preview_chars": min(max_chars, len(text)),
        "truncated": len(text) > max_chars,
        "preview": text[:max_chars],
    }


async def _embed_query(text: str, client: EmbeddingClient) -> Any:
    vecs = await client.embed_batch([text])
    return vecs[0] if vecs else None


def _cosine_sim(a, b) -> float:
    import numpy as np

    a = a.astype("float32").ravel()
    b = b.astype("float32").ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    processed = await document_store.count_documents(status="processed")
    if processed == 0:
        raise HTTPException(status_code=400, detail="No processed documents yet")

    emb_client = EmbeddingClient()
    qvec = await _embed_query(q, emb_client)
    if qvec is None:
        raise HTTPException(status_code=500, detail="Failed to embed query")

    # naive retrieval across all chunks (could add WHERE doc_hash in the future)
    chunks = await document_store.fetch_chunks(parser="mineru")
    chunk_ids = [c["chunk_id"] for c in chunks]
    emb_rows = await document_store.fetch_embeddings_for_chunks(chunk_ids)
    emb_by_id = {r.chunk_id: r for r in emb_rows}

    # Get document metadata for all chunks
    docs = await document_store.list_documents()
    docs_by_hash = {d["doc_hash"]: d for d in docs}
    
    # Count total chunks per document
    chunks_per_doc: Dict[str, int] = {}
    for c in chunks:
        dh = c["doc_hash"]
        chunks_per_doc[dh] = chunks_per_doc.get(dh, 0) + 1

    scored: List[Dict[str, Any]] = []
    for c in chunks:
        r = emb_by_id.get(c["chunk_id"])  # type: ignore
        if not r:
            continue
        score = _cosine_sim(r.vector, qvec)
        scored.append({"chunk": c, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)

    top_k = max(1, min(req.top_k, 20))
    top = scored[:top_k]
    context = "\n\n".join(f"[source {i+1}]\n" + t["chunk"]["text"] for i, t in enumerate(top))

    # Call LM Studio chat for answer
    base = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    model = (os.environ.get("LLM_MODEL") or "").strip()
    if not base or not key or not model:
        raise HTTPException(status_code=500, detail="OPENAI_BASE_URL, OPENAI_API_KEY, and LLM_MODEL must be set")

    prompt = (
        "Answer the question using the provided document context.\n"
        "Cite sources by [source N] where appropriate.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}"
    )

    try:
        from openai import AsyncOpenAI  # type: ignore

        client = AsyncOpenAI(base_url=base, api_key=key)
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for RAG."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
            stream=False,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {exc}")

    sources = []
    for t in top:
        chunk = t["chunk"]
        doc_hash = chunk["doc_hash"]
        doc = docs_by_hash.get(doc_hash, {})
        total_chunks = chunks_per_doc.get(doc_hash, 0)
        
        sources.append({
            "chunk_id": chunk["chunk_id"],
            "score": t["score"],
            "doc_hash": doc_hash,
            "order_index": chunk["order_index"],
            "document_name": doc.get("original_name", "unknown"),
            "total_chunks": total_chunks,
            "chunk_text_preview": chunk["text"][:200] if chunk.get("text") else "",
        })
    
    return AskResponse(answer=answer, sources=sources)


@app.get("/ready")
async def ready() -> Dict[str, bool]:
    return {"ready": (await document_store.count_documents(status="processed")) > 0}


@app.post("/warmup")
async def warmup() -> Dict[str, Any]:
    # Minimal LLM warmup by sending a short prompt
    base = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    model = (os.environ.get("LLM_MODEL") or "").strip()
    if not base or not key or not model:
        return {"warmup_complete": False, "error": "LLM env missing"}
    try:
        from openai import AsyncOpenAI  # type: ignore

        client = AsyncOpenAI(base_url=base, api_key=key)
        await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Warmup"},
                {"role": "user", "content": "Say ready."},
            ],
            max_tokens=4,
            temperature=0,
        )
        return {"warmup_complete": True, "status": "ready"}
    except Exception as exc:
        return {"warmup_complete": False, "error": str(exc)}


@app.post("/warmup/mineru")
async def warmup_mineru_route() -> Dict[str, Any]:
    try:
        info = await asyncio.to_thread(
            warmup_mineru,
            parse_method=MINERU_PARSE_METHOD,
            lang=MINERU_LANG,
            table_enable=(MINERU_TABLE_ENABLE in {"1", "true", "yes", "on"}),
            formula_enable=(MINERU_FORMULA_ENABLE in {"1", "true", "yes", "on"}),
        )
        return {"warmup_complete": True, **info}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MinerU warmup failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
