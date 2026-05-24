"""PDF parsing with table-aware extraction.

Most NBR documents are text-based PDFs, which means pdfplumber handles them
well. For scanned documents, swap the body of `parse_pdf` for a Tesseract
(`lang="ben"`) or hosted-OCR call — the rest of the pipeline does not change.

Tables are extracted separately so they can be embedded as their own chunks.
Splitting a rate schedule in the middle of its rows destroys retrievability,
so we keep tables atomic from this point on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pdfplumber


BlockType = Literal["text", "table"]


@dataclass
class Block:
    """A semantic unit extracted from a document.

    Either a run of prose or an extracted table. Carries the metadata we
    need to keep downstream chunks traceable back to their source.
    """
    type: BlockType
    content: str             # plain text for prose; Markdown for tables
    source: str              # absolute or relative path to the source PDF
    page: int
    section: str | None = None
    metadata: dict = field(default_factory=dict)


# Matches "Section 12", "Sec. 12", or Bangla "ধারা ১২" (with Bangla digits).
SECTION_PATTERN = re.compile(
    r"(?:Section|Sec\.?|ধারা)\s*[\d\u09E6-\u09EF]+",
    re.IGNORECASE,
)


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Render a pdfplumber table as Markdown.

    Keeping tables in Markdown preserves enough structure for an LLM to read
    rate slabs and threshold tables correctly without us needing a richer
    representation in the POC.
    """
    if not table or not table[0]:
        return ""

    def cells(row: list[str | None]) -> list[str]:
        return [(c or "").strip().replace("|", "\\|").replace("\n", " ") for c in row]

    header, *rows = table
    lines = ["| " + " | ".join(cells(header)) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        lines.append("| " + " | ".join(cells(row)) + " |")
    return "\n".join(lines)


def _detect_section(text: str) -> str | None:
    """Return the first section / ধারা reference found in `text`, or None."""
    m = SECTION_PATTERN.search(text)
    return m.group(0).strip() if m else None


def parse_pdf(path: str | Path) -> list[Block]:
    """Extract a list of Blocks from a PDF.

    Text blocks remember the most recent section heading we have seen so
    that chunks created from them inherit a useful `section` label.
    """
    path = Path(path)
    blocks: list[Block] = []
    current_section: str | None = None

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Tables first. pdfplumber returns one list-of-rows per table.
            for table in page.extract_tables() or []:
                md = _table_to_markdown(table)
                if md.strip():
                    blocks.append(Block(
                        type="table",
                        content=md,
                        source=str(path),
                        page=page_num,
                        section=current_section,
                    ))

            text = page.extract_text() or ""
            if not text.strip():
                continue

            # Update running section from this page's headings.
            detected = _detect_section(text)
            if detected:
                current_section = detected

            blocks.append(Block(
                type="text",
                content=text,
                source=str(path),
                page=page_num,
                section=current_section,
            ))

    return blocks
