"""Ingestion CLI.

Walks data/raw/*.pdf (or any paths passed on the command line), parses
each PDF into blocks, chunks them, and writes them to the hybrid index.

Usage:
  python scripts/ingest.py                          # all PDFs in data/raw
  python scripts/ingest.py path/to/one.pdf two.pdf  # specific files
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` and `config` importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import chunk_blocks                            # noqa: E402
from src.embedding import Embedder                               # noqa: E402
from src.indexing import HybridStore                             # noqa: E402
from src.parsing import parse_pdf                                # noqa: E402
from config import RAW_DIR                                       # noqa: E402


def main() -> None:
    if len(sys.argv) > 1:
        pdfs = [Path(p) for p in sys.argv[1:]]
    else:
        pdfs = sorted(Path(RAW_DIR).glob("*.pdf"))

    if not pdfs:
        print(f"no PDFs found in {RAW_DIR} — drop some in or pass paths on argv")
        sys.exit(1)

    embedder = Embedder()
    store = HybridStore()

    total_chunks = 0
    for pdf in pdfs:
        print(f"\n=== {pdf.name} ===")
        blocks = parse_pdf(pdf)
        chunks = chunk_blocks(blocks)
        print(f"  {len(blocks)} blocks → {len(chunks)} chunks")
        store.upsert(chunks, embedder)
        total_chunks += len(chunks)

    print(f"\nindexed {total_chunks} chunks; collection now holds {store.count()} points")


if __name__ == "__main__":
    main()
