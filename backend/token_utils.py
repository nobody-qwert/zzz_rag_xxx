from __future__ import annotations

import math
from typing import Iterable, Optional

try:
    from .tokenizer_registry import (
        count_chat_tokens,
        count_text_tokens,
        record_tokenizer_fallback,
        truncate_text,
    )
except ImportError:  # pragma: no cover
    from tokenizer_registry import (  # type: ignore
        count_chat_tokens,
        count_text_tokens,
        record_tokenizer_fallback,
        truncate_text,
    )

_FALLBACK_CHARS_PER_TOKEN = 4


def _fallback_token_estimate(text: str) -> int:
    return max(1, math.ceil(len(text) / _FALLBACK_CHARS_PER_TOKEN))


def estimate_tokens(
    text: str,
    *,
    tokenizer_id: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    if not text:
        return 0
    identifier = (tokenizer_id or model or "").strip()
    if identifier:
        token_count = count_text_tokens(text, identifier)
        if token_count is not None:
            return token_count
        record_tokenizer_fallback(identifier, "text_count_fallback")
    return _fallback_token_estimate(text)


def estimate_messages_tokens(
    messages: Iterable[dict],
    *,
    tokenizer_id: Optional[str] = None,
    model: Optional[str] = None,
) -> int:
    identifier = (tokenizer_id or model or "").strip()
    cached_messages = list(messages)
    if identifier:
        token_count = count_chat_tokens(cached_messages, identifier)
        if token_count is not None:
            return token_count
        record_tokenizer_fallback(identifier, "chat_count_fallback")
    total = 0
    for message in cached_messages:
        total += 4  # per-message structural overhead heuristic
        total += estimate_tokens(str(message.get("content", "")), tokenizer_id=identifier or None)
    return total + 2  # assistant priming per OpenAI guideline


def truncate_text_to_tokens(
    text: str,
    max_tokens: int,
    *,
    tokenizer_id: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    if max_tokens <= 0 or not text:
        return ""
    identifier = (tokenizer_id or model or "").strip()
    if identifier:
        truncated = truncate_text(text, max_tokens, identifier)
        if truncated is not None:
            return truncated
        record_tokenizer_fallback(identifier, "truncate_fallback")
    approx_chars = max_tokens * _FALLBACK_CHARS_PER_TOKEN
    return text[:approx_chars]
