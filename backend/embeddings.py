from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


class EmbeddingClient:
    def __init__(self) -> None:
        base = (os.environ.get("EMBEDDING_BASE_URL") or "").strip()
        key = (os.environ.get("EMBEDDING_API_KEY") or "").strip()
        model = (os.environ.get("EMBEDDING_MODEL") or "").strip()
        if not base or not key or not model:
            raise RuntimeError("EMBEDDING_BASE_URL, EMBEDDING_API_KEY, and EMBEDDING_MODEL must be set")
        self.base = base
        self.key = key
        self.model = model
        self.dim: Optional[int] = None

        from openai import AsyncOpenAI  # type: ignore

        self._client = AsyncOpenAI(base_url=self.base, api_key=self.key)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        vectors = await self._client.embeddings.create(model=self.model, input=texts)  # type: ignore[attr-defined]
        data = vectors.data  # type: ignore

        out: List[np.ndarray] = []
        for item in data:
            v = np.asarray(item.embedding, dtype=np.float32)  # type: ignore[attr-defined]
            current_dim = int(v.shape[0])
            if self.dim is None:
                self.dim = current_dim
            elif self.dim != current_dim:
                raise RuntimeError(f"Embedding dimension changed from {self.dim} to {current_dim}")
            out.append(v)
        if self.dim is None:
            raise RuntimeError("Failed to determine embedding dimension from embeddings response")
        return out
