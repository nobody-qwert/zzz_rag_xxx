# RAG-Anything + Remote LM Studio + Frontend (Docker Compose)

This project runs a complete RAG app in Docker and connects to a remote LM Studio server via its OpenAI-compatible API. A minimal React frontend provides:
- Upload/ingest of documents
- List of ingested docs
- Chat to ask questions against your indexed data

Architecture:
- Remote LLM (LM Studio) on another machine, exposing http://192.168.88.11:1234/v1 (adjust IP/port)
- rag-backend (FastAPI) in Docker uses RAG-Anything for ingest/index/query
- frontend (Vite/React -> static Nginx) calls backend. In production, Nginx proxies /api → rag-backend:8000

Note: No Ollama stack is required here.

--------------------------------------------------------------------------------

Prerequisites
- Windows 11 + WSL2 recommended for running docker compose
- Docker Desktop (WSL2 engine enabled)
- Remote LM Studio reachable from your Docker host
  - LM Studio must bind to 0.0.0.0 and allow inbound connections on its server port
  - In LM Studio, enable the “OpenAI compatible server” and ensure the base URL uses the /v1 suffix

--------------------------------------------------------------------------------

Project Layout
- backend/        FastAPI app (RAG-Anything integration)
- frontend/       React/Vite app (built and served by Nginx in production)
- data/           User-uploaded raw files (mounted into backend)
- indices/        Persistent indices/artifacts (mounted into backend)
- docker-compose.yml
- .env.example    Template for environment variables (copy to .env)

--------------------------------------------------------------------------------

Environment Setup
1) Copy the example env and edit values:
   cp .env.example .env

   Required values:
   - OPENAI_BASE_URL=http://192.168.88.11:1234/v1
   - OPENAI_API_KEY=any-non-empty-string
   - BACKEND_PORT=8000
   - FRONTEND_PORT=5173
   - DATA_DIR=/data
   - INDEX_DIR=/indices

2) Ensure LM Studio (remote) is running and reachable from this machine:
   - Bind to 0.0.0.0
   - OpenAI-compatible server URL must include /v1
   - Make sure firewall rules allow inbound traffic on the port (e.g., 1234)

--------------------------------------------------------------------------------

Run with Docker Compose
From the project root:
   docker compose up --build

- Backend will be available on http://localhost:8000
- Frontend will be available on http://localhost:5173

In production, the frontend’s Nginx proxies /api → rag-backend:8000. In local dev (Vite), the Vite dev server proxies /api → http://localhost:8000.

--------------------------------------------------------------------------------

Usage (Frontend)
- Open http://localhost:5173
- Upload a document; it will appear in the Docs list
- Ask questions in the Chat panel

--------------------------------------------------------------------------------

API Endpoints (Backend)
Base URL: http://localhost:8000

- POST /ingest
  - multipart/form-data with field: file
  - Example:
    curl -F "file=@/path/to/file.pdf" http://localhost:8000/ingest

- GET /docs
  - Lists ingested documents
  - Example:
    curl http://localhost:8000/docs

- POST /ask
  - JSON body: { "query": "..." }
  - Example:
    curl -H "Content-Type: application/json" -d '{"query":"What is in the docs?"}' http://localhost:8000/ask

- GET /healthz
  - Simple health check

--------------------------------------------------------------------------------

Local Development (optional)
You can run services outside Docker if desired.

Backend:
- Install deps: pip install -r backend/requirements.txt
- Run: uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
- Ensure env variables are set in your shell (OPENAI_BASE_URL, OPENAI_API_KEY, DATA_DIR, INDEX_DIR)

Frontend:
- cd frontend
- npm install
- npm run dev
- Vite dev server at http://localhost:5173 will proxy /api → http://localhost:8000

--------------------------------------------------------------------------------

Persistence
- data/ and indices/ are bind-mounted into the backend container.
- Files survive container rebuilds.

--------------------------------------------------------------------------------

Troubleshooting
- 404/Network errors from LM Studio:
  - Verify OPENAI_BASE_URL ends with /v1
  - Check the server is bound to 0.0.0.0 and is reachable (e.g., curl http://192.168.88.11:1234/v1/models)
  - Ensure firewall rules allow inbound connections and no VPN or routing rules block the path

- Ingestion fails for certain formats:
  - Some Office/PDF/image parsers may require extra system packages. The current backend image is python:3.10-slim + raganything[all].
  - If parsing fails for large/complex docs, consider extending the backend image with additional packages (e.g., poppler, tesseract, libreoffice) per RAG-Anything’s docs.

- CORS issues in dev:
  - Vite dev proxy routes /api to http://localhost:8000
  - Backend enables CORS for http://localhost:5173 by default

--------------------------------------------------------------------------------

Notes
- Do not commit secrets; keep .env locally. Update .env.example only for defaults.
- To change models, switch the model inside LM Studio; the backend stays the same.
- For performance, consider batching ingestion and prebuilding indices.

--------------------------------------------------------------------------------

License
- MIT or your preferred license.
