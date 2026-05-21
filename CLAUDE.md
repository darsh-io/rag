# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A ground-up RAG (Retrieval-Augmented Generation) build with no framework abstractions. Every component is implemented directly using raw APIs and libraries. The goal is to understand RAG deeply by building each piece from scratch.

**Current state:** Full retrieval-to-answer pipeline is working end-to-end. Ingest a PDF, ask questions, get cited answers with conversation history. FastAPI backend and HTML frontend are not yet built.

**Current stack:** OpenRouter (embeddings + LLM) → Chroma (vector DB) → BM25 (sparse retrieval) → RRF (fusion) → Cohere (reranker)

**Not yet built:** FastAPI backend, HTML/CSS/JS frontend

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add OPENROUTER_API_KEY and COHERE_API_KEY
```

Run:
```bash
python src/main.py
```

Test the PDF chunker standalone:
```bash
python src/ragPipeline/chunk.py
```

## Architecture

```
src/
├── main.py                   # Entry point: setup(), run_ingest(), run_query_loop()
├── query.py                  # Full RAG pipeline + LLM call (rag_query, call_llm)
└── ragPipeline/
    ├── embeddings.py         # OpenRouter embeddings API client (raw requests)
    ├── cosineSim.py          # Cosine similarity, pure Python (no numpy)
    ├── chunk.py              # PDF → sentence-aware overlapping chunks with metadata
    ├── vectorstore.py        # Chroma collection: get_collection, ingest, query
    ├── bm25.py               # Sparse BM25 retrieval over full corpus
    ├── rrf.py                # Reciprocal Rank Fusion
    └── reranker.py           # Cohere Rerank v2 API client
config/
└── config.yaml               # embeddings-model, llm-model, reranker-model names
resources/
└── Attention-Is-All-You-Need.pdf  # test document
chroma_db/                    # persisted vector store (gitignored)
```

## Data Flow

```
PDF → chunk_pdf() → chunks[]
                        ↓
              getEmbeddings() per chunk → Chroma (ingest)

Query → getEmbeddings() → Chroma top-10 (dense)
      → BM25Okapi → top-10 (sparse)
      → reciprocal_rank_fusion() → top-10
      → Cohere rerank → top-5
      → call_llm(system + history + context + question) → cited answer
```

## Key Patterns

**Config loading:** `main.py:setup()` loads `.env` via `python-dotenv`, reads `config/config.yaml` via PyYAML, returns a single `cfg` dict. All functions receive what they need via parameters — no globals.

**Conversation history:** `run_query_loop()` maintains a `history` list of `{role, content}` dicts. Appended *after* each successful call. Passed into `rag_query()` and injected between the system prompt and the current user message. `/clear` resets it; `/quit` returns to mode selection.

**Chunk IDs:** Formatted as `<filename>_chunk<index>` — stable across re-ingests, so re-uploading the same PDF overwrites rather than duplicates.

**Chunker details:** Splits by sentences (lookbehind regex on `.!?`), builds overlapping chunks by character count (`chunk_size=1000`, `overlap=200`), tracks page numbers by character position via a `char_to_page` array.

## Commit Conventions

| Tag | Meaning |
|-----|---------|
| `INIT` | First commit or major milestone |
| `ADD` | New feature or file |
| `DEL` | Removed something |
| `FIX` | Bug fix |
| `REF` | Refactor (no behavior change) |
| `DOCS` | Documentation only |
| `TEST` | Tests only |
| `CONF` | Config changes |
| `SEC` | Security fix |
| `IMPR` | Improvement to existing feature |
| `MISC` | Anything else |

Format: `TAG: Short description` (e.g., `ADD: Chroma vector store integration`)

## Key Constraints

- **No frameworks:** No LangChain, LlamaIndex, or similar. Raw `requests`, raw math.
- **OpenRouter only:** All LLM and embedding calls go through `https://openrouter.ai/api/v1/`. Key in `.env` as `OPENROUTER_API_KEY`.
- **Cohere for reranking:** Rerank endpoint is `https://api.cohere.com/v2/rerank`. Key in `.env` as `COHERE_API_KEY`.
- **Pure Python math:** `cosineSim.py` uses no numpy — keep it that way unless there's a strong reason.

