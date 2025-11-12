from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Set


@dataclass(frozen=True)
class AppSettings:
    data_dir: Path
    index_dir: Path
    doc_store_path: Path
    parser_mode: str
    min_pymupdf_chars_per_page: int
    ocr_parser_key: str
    ocr_module_url: str
    ocr_module_timeout: float
    chat_context_window: int
    chat_completion_max_tokens: int
    chat_completion_reserve: int
    min_context_similarity: float
    no_context_response: str
    system_prompt: str
    continue_prompt: str
    completed_doc_statuses: Set[str]
    frontend_origin: str
    chunk_size: int
    chunk_overlap: int
    large_chunk_size: int
    large_chunk_left_overlap: int
    large_chunk_right_overlap: int
    large_chunk_parser_key: str
    llm_base_url: str
    llm_api_key: str
    llm_model: str


def _int_env(name: str, default: str) -> int:
    return int(os.environ.get(name, default) or default)


def _float_env(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        return float(raw or default)
    except (TypeError, ValueError):
        return float(default)


def _str_env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or default).strip()


def load_settings() -> AppSettings:
    data_dir = Path(os.environ.get("DATA_DIR", "/app_data/docs"))
    index_dir = Path(os.environ.get("INDEX_DIR", "/app_data/runtime"))
    doc_store_path = Path(os.environ.get("DOC_STORE_PATH") or (data_dir / "rag_meta.db"))

    data_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    chat_completion_max_tokens = _int_env("CHAT_COMPLETION_MAX_TOKENS", "2048")
    chat_completion_reserve = _int_env("CHAT_COMPLETION_RESERVE", str(chat_completion_max_tokens))

    ocr_parser_key = _str_env("OCR_PARSER_KEY", "mineru").lower()
    large_chunk_size = _int_env("LARGE_CHUNK_SIZE", "1600")
    large_chunk_left_overlap = _int_env("LARGE_CHUNK_LEFT_OVERLAP", "100")
    large_chunk_right_overlap = _int_env("LARGE_CHUNK_RIGHT_OVERLAP", "100")

    return AppSettings(
        data_dir=data_dir,
        index_dir=index_dir,
        doc_store_path=doc_store_path,
        parser_mode=_str_env("PARSER_MODE", "ocr").lower(),
        min_pymupdf_chars_per_page=_int_env("MIN_PYMUPDF_CHARS_PER_PAGE", "300"),
        ocr_parser_key=ocr_parser_key,
        ocr_module_url=_str_env("OCR_MODULE_URL", "http://ocr-module:8000").rstrip("/"),
        ocr_module_timeout=_float_env("OCR_MODULE_TIMEOUT", "120"),
        chat_context_window=_int_env("CHAT_CONTEXT_WINDOW", "10000"),
        chat_completion_max_tokens=chat_completion_max_tokens,
        chat_completion_reserve=chat_completion_reserve,
        min_context_similarity=float(os.environ.get("MIN_CONTEXT_SIMILARITY", "0.35") or 0.35),
        no_context_response="I couldn't find relevant information for that in the available documents.",
        system_prompt=(
            "You are a retrieval-augmented assistant. Answer strictly using the provided document snippets and cite them as [source N]. "
            "When the snippets contain only partial information, summarize what they do cover and note any gaps; do not invent details. "
            "Only state that no relevant information exists when no snippets are supplied."
        ),
        continue_prompt=(
            "Continue the previous answer to the user's last question. "
            "Resume exactly where it stopped without repeating earlier content."
        ),
        completed_doc_statuses={s.strip().lower() for s in ("processed", "done", "completed", "ready")},
        frontend_origin=f"http://localhost:{os.environ.get('FRONTEND_PORT', '5173')}",
        chunk_size=_int_env("CHUNK_SIZE", "200"),
        chunk_overlap=_int_env("CHUNK_OVERLAP", "60"),
        large_chunk_size=large_chunk_size,
        large_chunk_left_overlap=large_chunk_left_overlap,
        large_chunk_right_overlap=large_chunk_right_overlap,
        large_chunk_parser_key=f"{ocr_parser_key}:large",
        llm_base_url=_str_env("LLM_BASE_URL"),
        llm_api_key=_str_env("LLM_API_KEY"),
        llm_model=_str_env("LLM_MODEL"),
    )
