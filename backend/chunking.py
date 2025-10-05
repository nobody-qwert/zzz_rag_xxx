from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Chunk:
    chunk_id: str
    order_index: int
    text: str
    token_count: int


def _tokenizer() -> any:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return enc
    except Exception:  # pragma: no cover - fallback
        return None


def _encode(enc, text: str) -> List[int]:
    if enc is None:
        # naive fallback: 1 token per ~4 chars
        n = max(1, len(text) // 4)
        return list(range(n))
    return enc.encode(text)


def _decode(enc, tokens: Sequence[int]) -> str:
    if enc is None:
        # cannot reconstruct; not used in fallback path
        return ""
    return enc.decode(list(tokens))


def sliding_token_chunks(text: str, *, size: int = 500, overlap: int = 100) -> List[Chunk]:
    enc = _tokenizer()
    token_ids = _encode(enc, text)
    if not token_ids:
        return []

    step = max(1, size - max(0, overlap))
    out: List[Chunk] = []
    order = 0

    for start in range(0, len(token_ids), step):
        end = min(len(token_ids), start + size)
        if end <= start:
            break
        if enc is None:
            # fallback: slice original text proportionally
            span_len = max(1, (end - start) * 4)
            piece = text[start * 4 : start * 4 + span_len]
            content = piece
        else:
            content = _decode(enc, token_ids[start:end])
        content_norm = (content or "").strip()
        if not content_norm:
            continue
        cid = "chunk-" + md5(content_norm.encode("utf-8")).hexdigest()
        out.append(Chunk(chunk_id=cid, order_index=order, text=content_norm, token_count=end - start))
        order += 1

        if end == len(token_ids):
            break

    return out

