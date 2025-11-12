from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from typing import Dict, List, Optional, Sequence


@dataclass
class Chunk:
    chunk_id: str
    order_index: int
    text: str
    token_count: int


@dataclass(frozen=True)
class ChunkWindowSpec:
    """Defines a sliding window with a fixed-sized core and optional padding."""

    name: str
    core_size: int
    left_padding: int = 0
    right_padding: int = 0
    step_size: Optional[int] = None

    def step(self) -> int:
        base = self.step_size if self.step_size is not None else self.core_size
        return max(1, base)


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


def _materialize_span(
    enc,
    token_ids: Sequence[int],
    text: str,
    start: int,
    end: int,
) -> str:
    if enc is None:
        approx_chars_per_token = 4
        raw_start = min(len(text), start * approx_chars_per_token)
        span_len = max(1, (end - start) * approx_chars_per_token)
        raw_end = min(len(text), raw_start + span_len)
        return text[raw_start:raw_end]
    return _decode(enc, token_ids[start:end])


def _chunks_for_spec(enc, token_ids: Sequence[int], text: str, spec: ChunkWindowSpec) -> List[Chunk]:
    total_tokens = len(token_ids)
    if total_tokens == 0 or spec.core_size <= 0:
        return []

    out: List[Chunk] = []
    step = spec.step()

    for core_start in range(0, total_tokens, step):
        core_end = min(total_tokens, core_start + spec.core_size)
        if core_end <= core_start:
            break

        window_start = max(0, core_start - max(0, spec.left_padding))
        window_end = min(total_tokens, core_end + max(0, spec.right_padding))
        if window_end <= window_start:
            continue

        content = _materialize_span(enc, token_ids, text, window_start, window_end)
        content_norm = (content or "").strip()
        if not content_norm:
            continue

        chunk_hash = md5(f"{spec.name}:{content_norm}".encode("utf-8")).hexdigest()
        chunk_id = f"chunk-{spec.name}-{chunk_hash}"
        out.append(
            Chunk(
                chunk_id=chunk_id,
                order_index=len(out),
                text=content_norm,
                token_count=window_end - window_start,
            )
        )

        if core_end == total_tokens:
            break

    return out


def chunk_text_multi(text: str, specs: Sequence[ChunkWindowSpec]) -> Dict[str, List[Chunk]]:
    """Return chunk lists for each spec using a shared tokenization."""
    enc = _tokenizer()
    token_ids = _encode(enc, text)
    results: Dict[str, List[Chunk]] = {}

    for spec in specs:
        results[spec.name] = _chunks_for_spec(enc, token_ids, text, spec)

    return results

