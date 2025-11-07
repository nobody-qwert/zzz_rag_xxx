from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, List, Optional

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None

_DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def _load_encoder(model: Optional[str] = None):  # type: ignore[override]
    if not tiktoken:
        return None
    try:
        if model:
            return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except Exception:
        return None


def estimate_tokens(text: str, *, model: Optional[str] = None) -> int:
    if not text:
        return 0
    encoder = _load_encoder(model)
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    # Rough fallback: ~4 characters per token
    return max(1, math.ceil(len(text) / 4))


def estimate_messages_tokens(messages: Iterable[dict], *, model: Optional[str] = None) -> int:
    total = 0
    for message in messages:
        total += 4  # per-message structural overhead heuristic
        total += estimate_tokens(str(message.get("content", "")), model=model)
    return total + 2  # assistant priming per OpenAI guideline


def truncate_text_to_tokens(text: str, max_tokens: int, *, model: Optional[str] = None) -> str:
    if max_tokens <= 0 or not text:
        return ""
    encoder = _load_encoder(model)
    if encoder is None:
        # fallback to character-level truncation
        approx_chars = max_tokens * 4
        return text[:approx_chars]
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text
    trimmed = tokens[:max_tokens]
    try:
        return encoder.decode(trimmed)
    except Exception:
        return text[: max_tokens * 4]
