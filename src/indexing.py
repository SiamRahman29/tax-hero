"""Hybrid vector store backed by Qdrant in embedded local mode.

Qdrant indexes dense and sparse vectors in the same collection and supports
metadata filters out of the box. For the POC we run it in local mode — no
Docker, no separate process, just a directory on disk. For production,
change the client to point at a real Qdrant cluster; nothing else changes.
"""
from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

from src.chunking import Chunk
from src.embedding import Embedder
from config import (
    COLLECTION_NAME,
    DENSE_DIM,
    DENSE_VECTOR_NAME,
    QDRANT_PATH,
    SPARSE_VECTOR_NAME,
)


def _sparse_to_qdrant(weights: dict[int, float]) -> SparseVector:
    """Convert BGE-M3's {token_id: weight} dict to a Qdrant SparseVector."""
    indices = list(weights.keys())
    values = [float(weights[i]) for i in indices]
    return SparseVector(indices=indices, values=values)


def _point_id(chunk_id: str) -> int:
    """Qdrant accepts u64 ids; map our 16-hex-char chunk ids deterministically."""
    return int(chunk_id, 16) & ((1 << 63) - 1)


class HybridStore:
    """A persistent hybrid (dense + sparse) index over chunks."""

    def __init__(self, path: str = QDRANT_PATH, collection: str = COLLECTION_NAME):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=path)
        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams()),
            },
        )

    def upsert(self, chunks: list[Chunk], embedder: Embedder, batch_size: int = 16) -> None:
        """Embed `chunks` and write them to the index."""
        for i in tqdm(range(0, len(chunks), batch_size), desc="indexing"):
            batch = chunks[i:i + batch_size]
            embeds = embedder.embed([c.text for c in batch])
            points = [
                PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector={
                        DENSE_VECTOR_NAME: emb.dense.tolist(),
                        SPARSE_VECTOR_NAME: _sparse_to_qdrant(emb.sparse),
                    },
                    payload=chunk.to_dict(),
                )
                for chunk, emb in zip(batch, embeds)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search_dense(self, dense: list[float], limit: int):
        return self.client.search(
            collection_name=self.collection,
            query_vector=NamedVector(name=DENSE_VECTOR_NAME, vector=dense),
            limit=limit,
        )

    def search_sparse(self, sparse: dict[int, float], limit: int):
        return self.client.search(
            collection_name=self.collection,
            query_vector=NamedSparseVector(
                name=SPARSE_VECTOR_NAME,
                vector=_sparse_to_qdrant(sparse),
            ),
            limit=limit,
        )

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count
