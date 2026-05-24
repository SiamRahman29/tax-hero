"""Answer generation with mandatory citations.

The system prompt is bilingual on purpose: most tax questions arrive in
Bangla or in code-mixed Bangla+English, and we want the model to answer
in whichever language the user wrote in. Citations are non-negotiable —
a Bangladeshi tax answer without a source reference is unusable, because
the user (or their accountant) needs to be able to verify it.

If you don't have an ANTHROPIC_API_KEY set, this module will raise at
construction time rather than later — fail loud, not silent.
"""
from __future__ import annotations

from dataclasses import dataclass

from anthropic import Anthropic

from src.retrieval import RetrievedChunk
from config import ANSWER_MODEL, ANTHROPIC_API_KEY


SYSTEM = """You are a careful assistant for Bangladeshi tax law questions.

Rules:
1. Answer ONLY from the evidence below. If the evidence does not cover the
   question, say so plainly — do not guess and do not fall back on general
   knowledge of tax law from other jurisdictions.
2. Cite every factual claim inline with bracketed numbers like [1], [2]
   that match the evidence ids. A sentence without a citation is a bug.
3. Answer in the same language the user used. If the question is in Bangla,
   answer in Bangla. If it mixes Bangla and English, mirror the mix.
4. When a rate table or threshold is relevant, reproduce the table in
   Markdown. Do not paraphrase a numeric table into prose.
5. If the evidence contains conflicting versions of the same provision
   (e.g. amended vs original), flag the conflict and report both, with
   their citations. Do not silently pick one.
"""


@dataclass
class Answer:
    text: str
    citations: list[RetrievedChunk]
    model: str


def _format_evidence(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered evidence block."""
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = c.payload
        source = meta.get("source", "?")
        page = meta.get("page", "?")
        section = meta.get("section") or "—"
        header = f"[{i}] source={source} page={page} section={section}"
        lines.append(header)
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines)


def answer(query: str, chunks: list[RetrievedChunk]) -> Answer:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set; copy .env.example to .env")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    evidence = _format_evidence(chunks)

    user = (
        f"Question:\n{query}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Write the answer following the rules above."
    )

    resp = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )

    # The response is a list of content blocks; concatenate text blocks.
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return Answer(text=text, citations=chunks, model=ANSWER_MODEL)
