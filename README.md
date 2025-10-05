RAG MinerU (root-level)

This is a GPU-ready RAG ingestion demo that:
- Extracts text with PyMuPDF (fast path)
- Extracts rich text/markdown with MinerU (GPU if available)
- Chunks MinerU text (500 tokens, 100 overlap)
- Computes embeddings using an LM Studio OpenAI-compatible endpoint
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
     OPENAI_BASE_URL=http://host:1234/v1
     OPENAI_API_KEY=lm-studio
     LLM_MODEL=your-model-name
     EMBEDDING_MODEL=text-embedding-3-small
     BACKEND_PORT=8000
     FRONTEND_PORT=5173
     DATA_DIR=/data
     INDEX_DIR=/indices
     MINERU_DEVICE_MODE=auto  # or cuda/cpu
     ```

2. **Build and run**
   ```bash
   docker compose up --build
   ```

3. **Access the UI**
   - Open your browser to: `http://localhost:5173` (or your configured FRONTEND_PORT)
   - The UI proxies `/api` requests to the backend service

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
- Generation: LM Studio via OpenAI-compatible chat API (LLM_MODEL)
