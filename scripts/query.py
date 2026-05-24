"""Query CLI.

Runs hybrid retrieval and generation for one question.
Prints the top retrieved chunks with their fusion scores so you can see
exactly what the model was reasoning over, then prints the answer.

Usage:
  python scripts/query.py "What is the tax-free threshold for individuals?"
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding import Embedder                               # noqa: E402
from src.generation import answer                                # noqa: E402
from src.indexing import HybridStore                             # noqa: E402
from src.retrieval import retrieve                               # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python scripts/query.py "<question>"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    embedder = Embedder()
    store = HybridStore()

    chunks = retrieve(query, embedder, store)

    print(f"\n=== retrieved {len(chunks)} chunks ===")
    for i, c in enumerate(chunks, start=1):
        meta = c.payload
        head = (
            f"[{i}] score={c.score:.4f}  "
            f"{Path(meta.get('source','?')).name}  "
            f"p.{meta.get('page','?')}  "
            f"{meta.get('section') or '—'}  "
            f"({meta.get('block_type')})"
        )
        print(head)
        preview = c.text.replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + " …"
        print("    " + preview)

    print("\n=== answer ===")
    print(answer(query, chunks).text)


if __name__ == "__main__":
    main()
