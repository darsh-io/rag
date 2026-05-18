# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 21-day ground-up RAG (Retrieval-Augmented Generation) build with no framework abstractions. Every component is implemented directly using raw APIs and libraries. The goal is to understand RAG deeply by building each piece from scratch.

**Current state:** Day 1–2 complete — embeddings, cosine similarity, and PDF chunker are working. Vector database, retrieval, reranking, API backend, and frontend are all planned but not yet built.

**Planned full stack:** OpenRouter (embeddings/LLM) → Chroma (vector DB) → Cohere (reranker) → FastAPI (backend) → HTML/CSS/JS (frontend)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add OPENROUTER_API_KEY
```

Run the current entry point:
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
├── main.py              # Entry point: loads config, runs similarity demo
└── ragPipeline/
    ├── embeddings.py    # OpenRouter embeddings API client (raw requests)
    ├── cosineSim.py     # Cosine similarity, pure Python (no numpy)
    └── chunk.py         # PDF → sentence-aware overlapping chunks with metadata
config/
└── config.yaml          # embeddings-model name (OpenRouter model ID)
resources/
└── Attention-Is-All-You-Need.pdf  # test document
```

**Data flow so far:** PDF → `chunk_pdf()` → list of `{text, source, page, chunk_index}` dicts → `getEmbeddings()` → `cosineSimilarity()`

**Config loading pattern:** `main.py:setup()` loads `.env` via `python-dotenv`, then reads `config/config.yaml` via PyYAML. Config is passed explicitly to functions — no globals.

**Chunker details:** Splits by sentences (regex on `.?!`), builds overlapping chunks by character count (`chunk_size=1000`, `overlap=200`), tracks page numbers by character position across pages.

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
- **OpenRouter only:** All model calls go through OpenRouter (`https://openrouter.ai/api/v1/`). API key in `.env` as `OPENROUTER_API_KEY`.
- **Pure Python math:** `cosineSim.py` uses no numpy — keep it that way unless there's a strong reason.
