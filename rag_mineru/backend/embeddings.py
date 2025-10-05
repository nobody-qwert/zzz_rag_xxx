from __future__ import annotations

import os
from typing import List

import numpy as np


class EmbeddingClient:
    def __init__(self) -> None:
        base = (os.environ.get("OPENAI_BASE_URL") or "").strip()
        key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        model = (os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small").strip()
        if not base or not key or not model:
            raise RuntimeError("OPENAI_BASE_URL, OPENAI_API_KEY, and EMBEDDING_MODEL must be set")
        self.base = base
        self.key = key
        self.model = model

        # dimension hints for LM Studio/OpenAI models
        ml = model.lower()
        if "text-embedding-3-large" in ml:
            self.dim = 3072
        elif "text-embedding-3-small" in ml:
            self.dim = 1536
        else:
            # common default for many local embedders
            self.dim = 768

        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(base_url=self.base, api_key=self.key)
            self._async = False
        except Exception:
            # fallback to async client if only that works in env
            from openai import AsyncOpenAI  # type: ignore

            self._client = AsyncOpenAI(base_url=self.base, api_key=self.key)
            self._async = True

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        if not texts:
            return []
        if self._async:
            vectors = await self._client.embeddings.create(model=self.model, input=texts)  # type: ignore[attr-defined]
            data = vectors.data  # type: ignore
        else:
            vectors = self._client.embeddings.create(model=self.model, input=texts)  # type: ignore[attr-defined]
            data = vectors.data  # type: ignore

        out: List[np.ndarray] = []
        for item in data:
            v = np.asarray(item.embedding, dtype=np.float32)  # type: ignore[attr-defined]
            out.append(v)
        return out

