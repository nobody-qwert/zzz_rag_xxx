RAG MinerU (root-level)

This is a GPU-ready RAG ingestion demo that:
- Extracts text with PyMuPDF (fast path)
- Extracts rich text/markdown with MinerU (GPU if available)
- Chunks MinerU text (500 tokens, 100 overlap)
- Computes embeddings using a local llama.cpp OpenAI-compatible endpoint
- Stores documents, extractions, chunks, and embeddings in SQLite
- Exposes a FastAPI backend compatible with the rag_test frontend APIs

## Prerequisites

### GPU Support (Required for optimal performance)

This application uses MinerU with GPU acceleration for fast PDF processing. To enable GPU support in Docker:

#### For WSL2 on Windows (with NVIDIA GPU):

1. **Verify NVIDIA drivers are installed in Windows**
   - Open PowerShell and run: `nvidia-smi`
   - You should see your GPU listed with driver version

2. **Install NVIDIA Container Toolkit in WSL**
   ```bash
   # Add NVIDIA package repository
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

   # Install the toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit

   # Configure Docker to use NVIDIA runtime
   sudo nvidia-ctk runtime configure --runtime=docker

   # Restart Docker
   sudo service docker restart
   ```

3. **Verify GPU access in Docker**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```
   You should see your GPU information displayed.

#### For Linux (with NVIDIA GPU):

Follow the same steps as WSL2 above. The NVIDIA Container Toolkit works identically on native Linux.

#### For systems without GPU:

The application will automatically fall back to CPU mode. Set `MINERU_DEVICE_MODE=cpu` in your `.env` file.

## Quick Start

1. **Create environment configuration**
     - Copy `.env.example` to `.env`
     - Configure the following variables:
       ```
        LLM_BASE_URL=http://llm:8000/v1
        LLM_API_KEY=local-llm
        LLM_MODEL=qwen/qwen3-4b-2507
        EMBEDDING_BASE_URL=http://embed:8080/v1
        EMBEDDING_API_KEY=local-embed
        EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
     BACKEND_PORT=8000
     FRONTEND_PORT=5173
     DATA_DIR=/data
     INDEX_DIR=/indices
     MINERU_DEVICE_MODE=auto  # or cuda/cpu
     ```

   These URLs match the service names defined in `docker-compose.yml`. When the backend needs to reach the models from outside Docker, point `LLM_BASE_URL` / `EMBEDDING_BASE_URL` to the host ports instead (for example, `http://localhost:8010/v1` and `http://localhost:8011/v1`).

2. **Stage model weights**
   - Place your GGUF files under `models/` (or update `.env` to point elsewhere). The defaults expect:
     - `models/llm/Qwen3-4B-Instruct-2507-Q8_0.gguf`
     - `models/embed/nomic-embed-text-v1.5.Q8_0.gguf`
     - Example download commands:
       ```bash
       huggingface-cli download lmstudio-community/Qwen3-4B-Instruct-2507-GQ44x4 --local-dir ./models/llm --include Qwen3-4B-Instruct-2507-Q8_0.gguf
       huggingface-cli download nomic-ai/nomic-embed-text-v1.5 --local-dir ./models/embed --include nomic-embed-text-v1.5.Q8_0.gguf
     ```
     (Any equivalent GGUF files work—just update the filenames in `.env`.)

3. **Build and run**
   ```bash
   docker compose up --build
   ```
   The first build also creates the CUDA-enabled llama.cpp image defined in `docker/llama-cpp.Dockerfile`, so expect a longer initial build while Python dependencies (llama-cpp-python + uvicorn) install.

4. **Access the UI**
   - Open your browser to: `http://localhost:5173` (or your configured FRONTEND_PORT)
   - The UI proxies `/api` requests to the backend service

## Local LLM + Embedding Services

The compose stack automatically launches two CUDA-enabled `llama.cpp` servers alongside the backend:

- `llm`: serves chat/completions using your Qwen3 4B GGUF build.
- `embed`: serves embeddings using your nomic-embed-text GGUF build.

Both services share the custom image built from `docker/llama-cpp.Dockerfile` (Python + `llama-cpp-python[cuda]`), so they support the same command-line flags and GPU acceleration.

> Note: The image now compiles `llama-cpp-python` with `-DGGML_CUDA=on -DGGML_CUDA_F16=on` inside an NVIDIA CUDA *devel* base. Rebuild the services (`docker compose build llm embed`) so the refreshed wheel runs matmuls on your 5090 instead of the CPU.

Useful knobs (see `.env.example`):

- `LOCAL_QWEN_MODEL_DIR` / `LOCAL_QWEN_MODEL_FILE`
- `LOCAL_EMBED_MODEL_DIR` / `LOCAL_EMBED_MODEL_FILE`
- `LLM_CONTEXT_SIZE` (default 10k tokens)
- `EMBED_CONTEXT_SIZE` (default 2k tokens)
- `LLM_HOST_PORT` / `EMBED_HOST_PORT` (default 8010/8011)
- `LLM_API_KEY` / `EMBEDDING_API_KEY` (shared secrets for the OpenAI-compatible endpoints)

Use these curl commands to verify the services after `docker compose up`:

```bash
curl http://localhost:${LLM_HOST_PORT:-8010}/v1/models -H "Authorization: Bearer ${LLM_API_KEY:-local-llm}"
curl http://localhost:${LLM_HOST_PORT:-8010}/v1/chat/completions \
  -H "Authorization: Bearer ${LLM_API_KEY:-local-llm}" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen/qwen3-4b-2507","messages":[{"role":"user","content":"Say hi"}]}'
curl http://localhost:${EMBED_HOST_PORT:-8011}/v1/embeddings \
  -H "Authorization: Bearer ${EMBEDDING_API_KEY:-local-embed}" \
  -H "Content-Type: application/json" \
  -d '{"model":"text-embedding-nomic-embed-text-v1.5","input":"embed me"}'
```

## Performance Notes

- **With GPU (CUDA)**: PDF processing is 10-20x faster (~2-5 seconds per page)
- **Without GPU (CPU)**: Processing is slower (~30-60 seconds per page)
- The `MINERU_DEVICE_MODE=auto` setting will automatically detect and use GPU if available

### Make First Parse Fast

On the first ingestion, MinerU downloads multiple model weights and initializes several pipelines. This can add 30–60 seconds (or more) to the first run even on a fast GPU. Options to remove that “first hit” latency:

- Build-time prefetch (opt-in):
  - Set `PREFETCH_MODELS=1` and rebuild: `docker compose build --build-arg PREFETCH_MODELS=1`
  - This runs a minimal warmup during the backend image build to populate the cache.
- Runtime warmup endpoint:
  - Call `POST /api/warmup/mineru` after the stack starts (frontend proxies `/api`).
  - Or enable `MINERU_WARMUP_ON_STARTUP=true` to warm up automatically on boot.

### Persistent Model Cache

The backend is configured to cache model files under `/indices` (a mounted volume):

- `HF_HOME=/indices/hf_cache` and `TRANSFORMERS_CACHE=/indices/hf_cache`
- `ULTRALYTICS_HOME=/indices/ultralytics`

This keeps downloads between restarts, so subsequent runs avoid re-fetching.

### Skip Heavy Steps When Not Needed

You can significantly speed up ingestion for digital PDFs (not scans) by using extracted text from PyMuPDF and skipping OCR-heavy MinerU steps.

- `PARSER_MODE=mineru|pymupdf|auto`
  - `pymupdf`: fastest text-only; no OCR/structure
  - `mineru`: full OCR + layout + tables
  - `auto`: use PyMuPDF if enough text was found (threshold below), otherwise fall back to MinerU
- `MIN_PYMUPDF_CHARS_PER_PAGE=300` (auto mode threshold)
- `MINERU_PARSE_METHOD=auto` (see MinerU docs for options)
- `MINERU_LANG=en`
- `MINERU_TABLE_ENABLE=true|false` and `MINERU_FORMULA_ENABLE=true|false` to disable table/formula passes for speed when unneeded

Tip: For docs without complex tables or formulas, disabling those passes saves time and VRAM.

## Supported Formats

- PDFs only. Image files (PNG/JPG), Office docs, and other formats are not parsed by MinerU in this demo.
- If you need to ingest images, convert them to PDF first (most OS print dialogs can “Save as PDF”).

Troubleshooting
- Error: "Failed to load document (PDFium: Data format error)"
  - Cause: MinerU (via PDFium) was asked to open a non‑PDF file or a corrupted PDF.
  - Fixes in this repo:
    - The frontend file picker now only accepts `.pdf`.
    - The backend skips MinerU for non‑PDFs and falls back to PyMuPDF text extraction instead of failing the job.
  - If the error persists on a real PDF, re‑download the file or try printing it to a new PDF (to repair damaged structure).

Notes
- This app lives outside sample_apps per project guidance
- The backend uses uv for fast installs and installs CUDA torch explicitly

Endpoints
- Backend (proxied via frontend at /api):
  - POST /api/ingest: upload a PDF, starts background job
  - GET /api/docs: list documents
  - GET /api/status: system + jobs summary (used by UI)
  - GET /api/status/{job_id}: single job status
  - POST /api/ingest/{doc_hash}/retry: retry processing a doc
  - GET /api/debug/parsed_text/{doc_hash}?parser=mineru|pymupdf: preview extracted text
  - POST /api/ask: simple RAG over SQLite-stored embeddings
  - GET /api/ready, POST /api/warmup

What the chat uses
- Retrieval: cosine similarity over embeddings stored in SQLite (per-chunk)
- Context: top-k chunk texts (k is configurable on the request)
- Generation: local llama.cpp OpenAI-compatible chat API (LLM_MODEL)
