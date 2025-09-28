# RAG-Anything + Remote LM Studio Integration

A complete RAG (Retrieval-Augmented Generation) application that integrates RAG-Anything with LM Studio for document processing and intelligent querying. Features a React frontend and FastAPI backend, all containerized with Docker Compose.

## ✨ Features

- **Document Processing**: Upload and process PDFs, images, Office documents, and more using RAG-Anything with MinerU parser
- **Intelligent Querying**: Ask questions about your documents using LightRAG's advanced retrieval system
- **LM Studio Integration**: Connect to remote LM Studio for both LLM and embedding models via OpenAI-compatible API
- **Modern Frontend**: Clean React interface for document upload and chat-based querying
- **Containerized**: Full Docker Compose setup for easy deployment
- **Persistent Storage**: Documents and indices survive container restarts

## 🏗️ Architecture

- **Remote LLM**: LM Studio server (e.g., `http://192.168.88.11:1234/v1`)
- **Backend**: FastAPI with RAG-Anything integration for document processing and querying
- **Frontend**: React/Vite app served by Nginx with API proxy
- **Storage**: Persistent volumes for uploaded documents and search indices

## 📋 Prerequisites

- **Docker Desktop** with WSL2 engine (Windows 11 recommended)
- **Remote LM Studio** server accessible from your Docker host
  - Must bind to `0.0.0.0` (not just localhost)
  - OpenAI-compatible server enabled with `/v1` endpoint
  - Both LLM and embedding models loaded (e.g., `text-embedding-nomic-embed-text-v1.5`)

## 🚀 Quick Start

### 1. Environment Setup

Copy and configure environment variables:

```bash
cp .env.example .env
```

Edit `.env` with your LM Studio details:

```env
# Remote LM Studio (OpenAI-compatible). IMPORTANT: include /v1 suffix
OPENAI_BASE_URL=http://192.168.88.11:1234/v1
OPENAI_API_KEY=lmstudio-or-token
# Model id as shown by LM Studio (/v1/models). Must match a loaded model.
LLM_MODEL=Your-Model-Id-From-LM-Studio
# Embedding model for RAG-Anything (must be available in LM Studio)
EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5

# Backend paths (mounted into the backend container)
DATA_DIR=/data
INDEX_DIR=/indices

# Backend server
BACKEND_PORT=8000

# Frontend server (host port exposed by docker compose)
FRONTEND_PORT=5173
```

### 2. Verify LM Studio Connection

Test that your LM Studio server is accessible:

```bash
curl http://192.168.88.11:1234/v1/models
```

You should see a list of loaded models including both your LLM and embedding model.

### 3. Launch the Application

```bash
docker compose up --build
```

The application will be available at:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📖 Usage

### Web Interface

1. **Upload Documents**: 
   - Navigate to http://localhost:5173
   - Use the upload interface to add PDFs, images, or Office documents
   - Documents are processed automatically using RAG-Anything + MinerU

2. **Query Documents**:
   - Use the chat interface to ask questions about your uploaded documents
   - The system uses LightRAG's hybrid search (vector + knowledge graph) for intelligent responses

### API Endpoints

#### Document Ingestion
```bash
curl -F "file=@document.pdf" http://localhost:8000/ingest
```

#### List Documents
```bash
curl http://localhost:8000/docs
```

#### Query Documents
```bash
curl -H "Content-Type: application/json" \
     -d '{"query":"What are the key concepts in the document?"}' \
     http://localhost:8000/ask
```

#### Health Check
```bash
curl http://localhost:8000/healthz
```

## 🔧 Configuration

### Supported File Types

RAG-Anything with MinerU parser supports:
- **PDFs**: Direct parsing with OCR fallback
- **Images**: PNG, JPEG, BMP, TIFF, GIF, WebP
- **Office Documents**: DOC, DOCX, PPT, PPTX, XLS, XLSX (via LibreOffice conversion)
- **Text Files**: TXT, MD (via ReportLab conversion)

### LM Studio Models

Recommended models for optimal performance:
- **LLM**: Any chat model loaded in LM Studio
- **Embeddings**: `text-embedding-nomic-embed-text-v1.5` (768 dimensions)
- **Alternative Embeddings**: `text-embedding-3-small` (1536d) or `text-embedding-3-large` (3072d)

### Processing Features

- **Multimodal Content**: Automatic processing of images, tables, and equations
- **Context-Aware**: Surrounding content extraction for better understanding
- **Caching**: Parse results cached for faster reprocessing
- **Knowledge Graph**: LightRAG builds entity-relationship graphs for enhanced retrieval

## 🛠️ Development

### Local Development (Optional)

**Backend**:
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Frontend**:
```bash
cd frontend
npm install
npm run dev
```

### Project Structure

```
├── backend/              # FastAPI application
│   ├── app.py           # Main application with RAG-Anything integration
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile       # Backend container
├── frontend/            # React application
│   ├── src/            # React components
│   ├── package.json    # Node.js dependencies
│   └── Dockerfile      # Frontend container
├── data/               # Uploaded documents (persistent)
├── indices/            # Search indices (persistent)
├── docker-compose.yml  # Container orchestration
└── .env.example       # Environment template
```

## 🔍 Troubleshooting

### Connection Issues

**LM Studio not reachable**:
- Verify `OPENAI_BASE_URL` includes `/v1` suffix
- Ensure LM Studio binds to `0.0.0.0`, not just `localhost`
- Check firewall rules allow inbound connections
- Test with: `curl http://your-lm-studio-ip:1234/v1/models`

**Model not found**:
- Verify `LLM_MODEL` matches exactly what's shown in `/v1/models`
- Ensure both LLM and embedding models are loaded in LM Studio
- Check model names are case-sensitive

### Processing Issues

**Document parsing fails**:
- Large/complex documents may require more memory
- Some formats may need additional system packages
- Check backend logs for specific parser errors

**Slow processing**:
- Consider using smaller models for faster processing
- Adjust `max_concurrent_files` in RAG-Anything config
- Monitor system resources during processing

### Query Issues

**No results returned**:
- Ensure documents were successfully ingested (check `/docs` endpoint)
- Try different query phrasings
- Check if embedding model is properly loaded

**Poor quality responses**:
- Experiment with different LLM models
- Adjust query modes (local, global, hybrid)
- Consider reranking models for better relevance

## 📊 Performance Tips

- **Memory**: Allocate sufficient RAM for LM Studio models
- **Storage**: Use SSD storage for faster index operations
- **Network**: Ensure stable, low-latency connection to LM Studio
- **Batching**: Process multiple documents together when possible

## 🔒 Security Notes

- Keep `.env` file secure and never commit it to version control
- Use strong API keys in production environments
- Consider network security when exposing LM Studio server
- Regularly update dependencies for security patches

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests for any improvements.

## 📞 Support

For issues related to:
- **RAG-Anything**: Check the [RAG-Anything documentation](https://github.com/aigc-apps/RAG-Anything)
- **LightRAG**: See [LightRAG repository](https://github.com/HKUDS/LightRAG)
- **LM Studio**: Visit [LM Studio documentation](https://lmstudio.ai/docs)
