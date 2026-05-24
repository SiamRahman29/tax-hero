"""BGE-M3 embedding wrapper.

BGE-M3 is the default for Bangla because it is multilingual, strong on
South Asian languages, and — critically for hybrid retrieval — returns
both a dense vector and a lexical-sparse representation in a single
forward pass. Running one model instead of two halves the latency and
keeps the two retrievers' tokenisation consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from config import EMBEDDING_MODEL


@dataclass
class Embedding:
    dense: np.ndarray              # shape (1024,), float32
    sparse: dict[int, float]       # token-id -> weight


class Embedder:
    """Thin wrapper around BGEM3FlagModel."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, use_fp16: bool = True):
        # use_fp16=True is faster on GPU; ignored on CPU.
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed(self, texts: list[str], batch_size: int = 8) -> list[Embedding]:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,   # ColBERT-style adds latency; skip for v1
        )
        dense_vecs = out["dense_vecs"]
        sparse_dicts = out["lexical_weights"]
        return [
            Embedding(
                dense=np.asarray(d, dtype=np.float32),
                sparse={int(k): float(v) for k, v in s.items()},
            )
            for d, s in zip(dense_vecs, sparse_dicts)
        ]

    def embed_query(self, query: str) -> Embedding:
        return self.embed([query])[0]
