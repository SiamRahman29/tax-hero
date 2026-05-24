"""Chunking with metadata preservation.

POC strategy:
  - Tables are atomic. Never split a rate schedule.
  - Long text blocks are split on paragraph boundaries, then merged into
    target-sized chunks with a small overlap so context bleeds across.
  - Bangla sentence boundary "।" (danda) is recognised alongside Latin
    periods.

Every chunk carries `effective_from`, `effective_to`, and `supersedes`
fields. They are unused in the POC — the amendment tagger populates them
later — but they exist so retrieval-time temporal filtering becomes a
one-line addition with no schema change.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

from src.parsing import Block
from config import CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS


_SENTENCE_SPLIT = re.compile(r"(?<=[.;।])\s+")


def _approx_tokens(text: str) -> int:
    """Rough token count.

    BGE-M3 uses a multilingual tokenizer; ~4 characters per token is a
    serviceable approximation for mixed Bangla + English. Replace with
    the real tokenizer if you need precision.
    """
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int
    section: str | None
    block_type: str                                # "text" or "table"
    # Versioning metadata — empty in the POC, populated by the amendment
    # tagger later. Leaving the fields here keeps the storage schema stable.
    effective_from: str | None = None              # ISO date
    effective_to: str | None = None                # ISO date
    supersedes: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _hash_id(*parts: str) -> str:
    """Deterministic 16-hex-char id derived from the chunk's content + source."""
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def _split_paragraphs(text: str) -> list[str]:
    """Split prose into paragraphs; fall back to sentence-ish splits if a
    paragraph alone is bigger than the target chunk size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    out: list[str] = []
    for p in paragraphs:
        if _approx_tokens(p) <= CHUNK_TARGET_TOKENS:
            out.append(p)
        else:
            sentences = [s.strip() for s in _SENTENCE_SPLIT.split(p) if s.strip()]
            out.extend(sentences or [p])
    return out


def chunk_blocks(blocks: Iterable[Block]) -> list[Chunk]:
    """Convert parsed Blocks into Chunks ready for embedding."""
    chunks: list[Chunk] = []

    for block in blocks:
        if block.type == "table":
            chunks.append(Chunk(
                chunk_id=_hash_id(block.source, str(block.page), "table", block.content[:64]),
                text=block.content,
                source=block.source,
                page=block.page,
                section=block.section,
                block_type="table",
            ))
            continue

        pieces = _split_paragraphs(block.content)
        buffer: list[str] = []
        buffer_tokens = 0

        def _flush() -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            text = "\n\n".join(buffer)
            chunks.append(Chunk(
                chunk_id=_hash_id(block.source, str(block.page), text[:64]),
                text=text,
                source=block.source,
                page=block.page,
                section=block.section,
                block_type="text",
            ))
            # Carry a tail forward as overlap so context flows across chunks.
            overlap: list[str] = []
            overlap_tokens = 0
            for prev in reversed(buffer):
                if overlap_tokens + _approx_tokens(prev) > CHUNK_OVERLAP_TOKENS:
                    break
                overlap.insert(0, prev)
                overlap_tokens += _approx_tokens(prev)
            buffer = overlap
            buffer_tokens = overlap_tokens

        for piece in pieces:
            piece_tokens = _approx_tokens(piece)
            if buffer and buffer_tokens + piece_tokens > CHUNK_TARGET_TOKENS:
                _flush()
            buffer.append(piece)
            buffer_tokens += piece_tokens

        _flush()

    return chunks
