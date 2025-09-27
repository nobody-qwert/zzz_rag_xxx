# Plan: RAG-Anything in Docker + Remote LM Studio (OpenAI-compatible) + Frontend

This plan runs the **RAG pipeline entirely inside Docker** (service: `rag-backend`) and connects to a **remote LM Studio server** for inference via its **OpenAI-compatible** endpoint. A separate **frontend** container provides a chat UI and shows the list of ingested documents.

---

## Architecture

* **Remote LLM (LM Studio):** running on another machine, exposing `http://192.168.88.11:1234/v1` (adjust IP/port as needed). Qwen 3 14B (or any other model) can be loaded there; GPU lives on the remote box.
* **rag-backend (Docker):** Python API that wraps **RAG-Anything** (ingest, index, query). Talks to LM Studio via `OPENAI_BASE_URL`.
* **frontend (Docker):** web UI to:

  * Upload/ingest documents
  * List ingested items
  * Chat with the RAG app (ask questions)

> No Ollama stack is required in this plan.

---

## Prerequisites (Windows + WSL2 host)

* Windows with latest NVIDIA driver (if you also run local GPU workloads; not strictly required here since the LLM is remote).
* Docker Desktop (WSL2 engine). Run `docker compose` from your WSL shell inside the project directory for best performance.
* Remote LM Studio reachable from your host on `192.168.88.11:1234` (make sure firewall allows inbound; LM Studio server should bind to `0.0.0.0`).

---

## Project layout

```
project/
├─ data/                 # user-uploaded docs (mounted into backend)
├─ indices/              # persistent indices / artifacts (mounted into backend)
├─ frontend/             # frontend app source (React/Vite suggested)
│  ├─ Dockerfile
│  └─ src/ ...
├─ backend/
│  ├─ app.py             # FastAPI app wiring RAG-Anything
│  ├─ requirements.txt   # fastapi, uvicorn, raganything[all], python-multipart, pydantic, etc.
│  └─ Dockerfile
├─ .env
└─ docker-compose.yml
```

---

## Environment (.env)

```env
# Remote LM Studio (OpenAI-compatible). IMPORTANT: include /v1 suffix
OPENAI_BASE_URL=http://192.168.88.11:1234/v1
OPENAI_API_KEY=lmstudio-or-token   # LM Studio often accepts any non-empty string

# Backend paths
DATA_DIR=/data
INDEX_DIR=/indices

# Backend server
BACKEND_PORT=8000

# Frontend server
FRONTEND_PORT=5173
```

---

## Backend service: `rag-backend`

**Goals**

* REST endpoints:

  * `POST /ingest` — upload/ingest a document (PDF/DOCX/IMG/ZIP) → updates indices
  * `GET  /docs` — list ingested docs (file names / IDs)
  * `POST /ask` — body `{ "query": "..." }` → runs RAG query and returns answer + sources
  * (Optional) `DELETE /docs/{id}` to remove
* Uses **RAG-Anything** for parsing and query (`process_document_complete`, `aquery`, etc.).
* Stores artifacts under `/indices` and raw files under `/data`.

**backend/requirements.txt** (suggested)

```
fastapi
uvicorn[standard]
raganything[all]
python-multipart
pydantic>=2
```

**backend/Dockerfile** (example)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ /app/
ENV DATA_DIR=/data INDEX_DIR=/indices \
    OPENAI_BASE_URL=${OPENAI_BASE_URL} OPENAI_API_KEY=${OPENAI_API_KEY}
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**backend/app.py** (high-level outline)

* On `POST /ingest`:

  1. Save file to `${DATA_DIR}`.
  2. Call `rag.process_document_complete(path, output_dir=INDEX_DIR, parse_method="auto", parser="mineru")`.
* On `POST /ask`:

  * Call `rag.aquery(query)` and return `{ answer, citations }`.
* On `GET /docs`:

  * List files in `${DATA_DIR}` with basic metadata.

> Implement one shared `RAGAnything()` instance per process (lazy-init) and reuse; pass env vars through to the OpenAI-compatible client as RAG-Anything expects.

---

## Frontend service: `frontend`

**Goals**

* Minimal UI with:

  * **Uploader** → `POST /ingest`
  * **Docs list** → `GET /docs`
  * **Chat pane** → calls `POST /ask`
* Stack suggestion: **React + Vite** (fast), fetch calls to backend at `http://localhost:${BACKEND_PORT}` (wired via Compose network name `rag-backend`).

**frontend/Dockerfile** (example, multi-stage)

```dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

> During development you can also run a Vite dev server, but the Nginx stage is simpler for deployment.

---

## docker-compose.yml

```yaml
name: rag-remote-lmstudio
services:
  rag-backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    env_file: .env
    environment:
      OPENAI_BASE_URL: ${OPENAI_BASE_URL}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      DATA_DIR: ${DATA_DIR}
      INDEX_DIR: ${INDEX_DIR}
    volumes:
      - ./data:${DATA_DIR}
      - ./indices:${INDEX_DIR}
    ports:
      - "${BACKEND_PORT}:8000"

  frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    depends_on:
      - rag-backend
    ports:
      - "${FRONTEND_PORT}:80"
```

---

## Networking & security notes

* Ensure the **remote LM Studio** binds to `0.0.0.0` and that **192.168.88.11:1234** is reachable from your Docker host.
* Windows Firewall on the remote machine must allow inbound TCP on the chosen port.
* If your network uses different subnets/VLANs, add routes/firewall rules accordingly.
* Use HTTPS and auth if you expose the backend/front-end beyond LAN (e.g., place a reverse proxy like Traefik/Caddy with TLS and an auth layer).

---

## Run

```bash
# from the project folder (WSL terminal recommended)
docker compose up --build
```

* Open the frontend at `http://localhost:${FRONTEND_PORT}`.
* Upload docs; they appear in **Docs** list.
* Chat in the UI → calls backend `/ask` → backend calls **remote LM Studio** at `${OPENAI_BASE_URL}`.

---

## Tuning & tips

* **Model selection:** change the model inside LM Studio; the backend stays the same.
* **Throughput:** consider batching ingestion and prebuilding indices.
* **Embeddings:** start simple (use the same LM Studio endpoint). If you need different embeddings, add a dedicated embeddings function or a separate provider.
* **Persistence:** the `data/` and `indices/` folders are volumes—safe across container rebuilds.
* **Large docs:** if you ingest Office formats, you may add LibreOffice to the backend image or switch RAG-Anything parser.
