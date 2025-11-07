FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG LLAMA_CPP_PYTHON_REF=main

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates \
    build-essential cmake ninja-build \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
  && python3 -m pip install --no-cache-dir \
      "llama-cpp-python[cuda] @ git+https://github.com/abetlen/llama-cpp-python.git@${LLAMA_CPP_PYTHON_REF}" \
      "uvicorn>=0.30.0" \
      "anyio>=4.0.0" \
      "fastapi>=0.110" \
      "starlette>=0.37" \
      "starlette-context>=0.3.6" \
      "pydantic-settings>=2.0.0" \
      "sse-starlette>=2.0.0"

WORKDIR /models

CMD ["python3", "-m", "llama_cpp.server", "--help"]
