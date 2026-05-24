"""Hybrid retrieval with Reciprocal Rank Fusion.

We run dense and sparse search separately and combine the ranked lists
with RRF — a simple, parameter-free fusion that beats most weighted
schemes empirically. The dense retriever catches semantic matches; the
sparse retriever catches exact terminology like section numbers, SRO
references, and Bangla-specific terms that dense embeddings sometimes
smear together.

For production, plug a cross-encoder reranker (BAAI/bge-reranker-v2-m3
or Cohere Rerank) on top of the fused list. That single addition is
usually the biggest quality jump in the whole pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.embedding import Embedder
from src.indexing import HybridStore
from config import RRF_K, TOP_K_DENSE, TOP_K_FINAL, TOP_K_SPARSE


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    payload: dict


def _rrf_fuse(dense_hits: list, sparse_hits: list, top_k: int) -> list[RetrievedChunk]:
    """Combine two ranked lists via Reciprocal Rank Fusion.

    Each document's RRF score is the sum over the lists it appears in of
    1 / (k + rank). k = 60 is the standard from Cormack et al.; it has
    almost no influence on the final ordering and is rarely worth tuning.
    """
    scores: dict[int, float] = {}
    payloads: dict[int, dict] = {}

    for rank, hit in enumerate(dense_hits):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        payloads[hit.id] = hit.payload

    for rank, hit in enumerate(sparse_hits):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        payloads.setdefault(hit.id, hit.payload)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        RetrievedChunk(
            chunk_id=payloads[pid]["chunk_id"],
            text=payloads[pid]["text"],
            score=score,
            payload=payloads[pid],
        )
        for pid, score in ranked[:top_k]
    ]


def retrieve(
    query: str,
    embedder: Embedder,
    store: HybridStore,
    top_k: int = TOP_K_FINAL,
) -> list[RetrievedChunk]:
    """Run hybrid retrieval for a single query.

    Returns the top_k chunks after RRF fusion of dense and sparse hits.
    """
    emb = embedder.embed_query(query)
    dense_hits = store.search_dense(emb.dense.tolist(), limit=TOP_K_DENSE)
    sparse_hits = store.search_sparse(emb.sparse, limit=TOP_K_SPARSE)
    return _rrf_fuse(dense_hits, sparse_hits, top_k)
