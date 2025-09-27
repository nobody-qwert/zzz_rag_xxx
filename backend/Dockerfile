FROM python:3.10-slim

WORKDIR /app

# Recommended Python env flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps required by raganything/mineru parsers (PDF/OCR etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    poppler-utils ghostscript tesseract-ocr \
  && rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY backend/ /app/

# Runtime env (values injected via docker-compose env_file)
ENV DATA_DIR=/data \
    INDEX_DIR=/indices \
    OPENAI_BASE_URL=${OPENAI_BASE_URL} \
    OPENAI_API_KEY=${OPENAI_API_KEY}

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
