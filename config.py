"""Central configuration for the Bangladeshi tax RAG system.

All knobs live here. Edit the defaults or override via environment variables
(loaded from .env automatically).
"""
from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = DATA_DIR / "eval"
QDRANT_PATH = os.getenv("QDRANT_PATH", str(DATA_DIR / "qdrant_storage"))

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# BGE-M3 is the default because it is multilingual, strong on Bangla, and
# returns dense + lexical-sparse embeddings in a single forward pass — which
# is exactly what hybrid retrieval needs. Swap to "intfloat/multilingual-e5-large"
# if you would rather keep a sentence-transformers-only stack.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Claude Sonnet 4.6 is a good cost/quality balance for the POC. For the
# hardest temporal / multi-hop questions, Opus 4.7 is worth the price.
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Conservative chunk size for mixed Bangla + English. BGE-M3 supports up
# to 8192 tokens, but smaller chunks retrieve more precisely.
CHUNK_TARGET_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_DENSE = 20
TOP_K_SPARSE = 20
TOP_K_FINAL = 8       # how many chunks reach the LLM after fusion
RRF_K = 60            # standard RRF constant (Cormack et al.)

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tax_chunks")
DENSE_DIM = 1024      # BGE-M3 dense dimension
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
