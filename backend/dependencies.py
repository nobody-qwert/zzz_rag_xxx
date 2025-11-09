from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI

try:  # When imported as part of package
    from .config import load_settings
    from .persistence import DocumentStore
except ImportError:  # pragma: no cover - script fallback
    from config import load_settings  # type: ignore
    from persistence import DocumentStore  # type: ignore

settings = load_settings()
document_store = DocumentStore(settings.doc_store_path)
jobs_registry: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await document_store.init()
    yield
