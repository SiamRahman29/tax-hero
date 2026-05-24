# bn-tax-rag

A retrieval-augmented generation system for Bangladeshi tax law.
POC stage: a working hybrid pipeline that ingests NBR PDFs, retrieves
across Bangla and English, and answers with citations.

## What's in here

### POC scope (what this repo does today)
- `pdfplumber`-based parsing with table extraction
- BGE-M3 embeddings — multilingual; returns dense + lexical-sparse in one model
- Qdrant local hybrid index (dense + sparse vectors in one collection)
- Reciprocal Rank Fusion of dense and sparse hits
- Claude generation with a citation-required prompt
- Eval harness with three baseline metrics: retrieval@k, citation rate, faithfulness

### Deliberately out of scope for the POC
The production architecture diagram shows the full target. The pieces below
get added only when the eval set says they matter — not before.
- Cross-encoder reranking (BAAI/bge-reranker-v2-m3 or Cohere Rerank)
- Amendment graph and effective-date filtering
- Query understanding (intent, entities, temporal anchor)
- Tool calls for numerical computation (rate slab arithmetic)
- Observability (Langfuse / Phoenix)
- Real OCR fallback for scanned PDFs

## Layout

```
.
├── config.py                    Central config; reads .env
├── pyproject.toml
├── .env.example                 Copy to .env and set ANTHROPIC_API_KEY
├── src/
│   ├── parsing.py               PDF → Blocks (text + tables)
│   ├── chunking.py              Blocks → Chunks (with metadata slots)
│   ├── embedding.py             BGE-M3 wrapper
│   ├── indexing.py              Qdrant hybrid store
│   ├── retrieval.py             Hybrid search + RRF fusion
│   ├── generation.py            Claude with citation prompt
│   └── evaluation.py            Eval runner
├── scripts/
│   ├── ingest.py                CLI: ingest PDFs
│   └── query.py                 CLI: ask one question
└── data/
    ├── raw/                     Drop NBR PDFs here
    └── eval/
        └── eval_set.example.jsonl   Format for the eval set
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

# put your NBR PDFs into data/raw/
# put your hand-curated eval set into data/eval/eval_set.jsonl
```

The first run downloads BGE-M3 (~2.2 GB). It runs on CPU but is much
faster on a GPU.

## Use

```bash
# Ingest PDFs from data/raw/
python scripts/ingest.py

# Ask a question
python scripts/query.py "What is the tax-free threshold for individuals?"
python scripts/query.py "ব্যক্তি করদাতার জন্য সর্বনিম্ন আয়সীমা কত?"

# Run the eval set
python -m src.evaluation data/eval/eval_set.jsonl
```

## Design notes worth knowing

**Why BGE-M3 over multilingual-e5 or OpenAI.** BGE-M3 returns dense and
lexical-sparse embeddings in one forward pass, which is the exact shape
hybrid retrieval needs. It is also among the strongest open multilingual
models for South Asian languages. `multilingual-e5-large` is a reasonable
alternative if you'd rather avoid the FlagEmbedding dependency, but you'd
then need a separate sparse retriever.

**Why Qdrant local mode.** Qdrant indexes both dense and sparse vectors
in the same collection and supports filtered queries — which is what you
need once amendment metadata lands on each chunk. Local mode means no
Docker for the POC; flip a URL to graduate to a real server later.

**Why RRF instead of weighted fusion.** Reciprocal Rank Fusion is
parameter-free, robust, and beats most weight-tuned schemes when the
two retrievers have very different score scales (which dense cosine
and sparse-lexical always do). Save the tuning for the cross-encoder
reranker that comes next.

**Chunks carry `effective_from` / `effective_to` and `supersedes` from
day one.** They are empty in the POC, but the fields exist so retrieval-
time temporal filtering becomes a one-line addition later — no schema
migration needed.

**Why no BM25 alongside the sparse vectors.** BGE-M3's lexical-sparse
output is, effectively, a learned BM25. For Bangla specifically it
beats classical BM25 because it tokenizes via the multilingual model
instead of a brittle word splitter. If you later need raw BM25 for
specific identifiers (sections, SRO numbers), add it then — the
indexer is structured to accept a third retriever.

## Where to extend (in order)

1. **Eval set first.** Until you have 50+ verified questions, every
   architecture change is unmeasurable. Don't skip this.
2. **Reranker.** Add `BAAI/bge-reranker-v2-m3` after the RRF step. On a
   real eval set this typically moves retrieval quality 10–15 points.
3. **Amendment tagger.** A separate ingestion step that reads each
   Finance Act and identifies the parent-Act sections it amends.
   Persist into Postgres as edges in a `(provision, amended_by,
   effective_from)` table. Add a `valid_at` filter to retrieval.
4. **Query understanding.** A small LLM call before retrieval that
   extracts language, intent (factual / temporal / tabular / multi-hop),
   entities, and a temporal anchor ("as of" date). Use the anchor to
   filter chunks by `effective_from`/`effective_to`.
5. **Cross-modal page retrieval.** For pages where pdfplumber mangles a
   complex rate schedule, run ColPali on the page image and treat the
   page-embedding as a fallback retriever.
