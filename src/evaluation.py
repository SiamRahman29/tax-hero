"""Evaluation harness.

Three metrics that together cover most of what matters for a tax RAG:

  retrieval_ok   — did the gold source document appear in the top-k?
                   This is the bluntest possible retrieval metric, but it
                   is dead reliable as a regression signal.
  cited          — did the answer actually include inline citations?
                   A surprising number of LLMs ignore citation instructions
                   silently; measure it.
  faithful       — does an LLM judge agree the answer is supported by the
                   retrieved evidence? Cheap proxy for hallucination rate.

The eval set is a JSONL file where each line is:
  {
    "question": "...",
    "expected_source": "filename.pdf",     # filename match is enough
    "expected_section": "Section 33",      # optional, for tighter scoring
    "category": "factual_current"          # for breakdowns
  }

Run as a module:
  python -m src.evaluation data/eval/eval_set.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from anthropic import Anthropic

from src.embedding import Embedder
from src.generation import answer
from src.indexing import HybridStore
from src.retrieval import retrieve
from config import ANSWER_MODEL, ANTHROPIC_API_KEY


CITATION_PATTERN = re.compile(r"\[\d+\]")

JUDGE_PROMPT = """You are a strict grader. The candidate answer is shown
below along with the evidence it was meant to use. Reply with exactly one
word — "yes" if every factual claim in the answer is supported by the
evidence, "no" otherwise. Stylistic differences and reordering are fine;
only factual support matters.

Question:
{question}

Evidence:
{evidence}

Candidate answer:
{candidate}

Is the candidate fully supported by the evidence? Reply yes or no."""


@dataclass
class EvalResult:
    question: str
    category: str
    retrieved_sources: list[str]
    retrieval_ok: bool
    answer_text: str
    cited: bool
    faithful: bool | None


def _judge_faithful(question: str, evidence: str, candidate: str) -> bool | None:
    """LLM-as-judge: returns True/False, or None if the call fails."""
    if not ANTHROPIC_API_KEY:
        return None
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        resp = client.messages.create(
            model=ANSWER_MODEL,
            max_tokens=4,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question, evidence=evidence, candidate=candidate,
                ),
            }],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return text.strip().lower().startswith("y")
    except Exception:
        return None


def run_eval(eval_path: str | Path, embedder: Embedder, store: HybridStore) -> list[EvalResult]:
    results: list[EvalResult] = []
    with open(eval_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    for row in rows:
        question = row["question"]
        expected = row.get("expected_source", "")
        category = row.get("category", "uncategorised")

        chunks = retrieve(question, embedder, store)
        retrieved_sources = [Path(c.payload.get("source", "")).name for c in chunks]
        retrieval_ok = any(expected and expected in s for s in retrieved_sources)

        ans = answer(question, chunks)
        cited = bool(CITATION_PATTERN.search(ans.text))

        evidence = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(chunks))
        faithful = _judge_faithful(question, evidence, ans.text)

        results.append(EvalResult(
            question=question,
            category=category,
            retrieved_sources=retrieved_sources,
            retrieval_ok=retrieval_ok,
            answer_text=ans.text,
            cited=cited,
            faithful=faithful,
        ))
        print(f"[{category}] retrieval_ok={retrieval_ok} cited={cited} faithful={faithful}")

    return results


def summarize(results: list[EvalResult]) -> dict:
    by_cat: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    def pct(rs: list[EvalResult], attr: str) -> float:
        vals = [getattr(r, attr) for r in rs if getattr(r, attr) is not None]
        return round(100 * sum(bool(v) for v in vals) / max(1, len(vals)), 1)

    summary = {"overall": {
        "n": len(results),
        "retrieval_ok": pct(results, "retrieval_ok"),
        "cited": pct(results, "cited"),
        "faithful": pct(results, "faithful"),
    }}
    for cat, rs in by_cat.items():
        summary[cat] = {
            "n": len(rs),
            "retrieval_ok": pct(rs, "retrieval_ok"),
            "cited": pct(rs, "cited"),
            "faithful": pct(rs, "faithful"),
        }
    return summary


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m src.evaluation <eval_set.jsonl>")
        sys.exit(1)
    eval_path = sys.argv[1]
    embedder = Embedder()
    store = HybridStore()
    results = run_eval(eval_path, embedder, store)
    print("\n=== summary ===")
    print(json.dumps(summarize(results), indent=2, ensure_ascii=False))

    # Also dump per-question results for inspection.
    out_path = Path(eval_path).with_suffix(".results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
