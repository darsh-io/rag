# Project RAGged

A 21-day ground-up build of a production RAG (Retrieval-Augmented Generation) system — no framework abstractions, no black boxes. Every component is written and understood before the next one is added.

## What it is

A document Q&A system that lets you ask questions about your own files and get grounded, cited answers. Built incrementally: embeddings first, then chunking, vector storage, retrieval, reranking, a FastAPI backend, a streaming frontend, and finally deployed to a live URL.

**Planned stack**

| Layer | Tool |
|---|---|
| LLM + Embeddings | OpenRouter |
| Vector DB | Qdrant Cloud |
| Backend | FastAPI |
| Frontend | HTML / CSS / JS |
| Deployment | Railway |
| Reranking | Cohere (free tier) |

## Current state — Day 1

The embedding + similarity foundation is in place:

- `src/embeddings.py` — calls the OpenRouter embeddings endpoint with raw `requests` (no SDK wrappers)
- `src/cosineSim.py` — cosine similarity implemented from scratch in pure Python
- `src/main.py` — takes two sentences, embeds both, prints their cosine similarity score
- `config/config.yaml` — model config (`nvidia/llama-nemotron-embed-vl-1b-v2:free`)

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
```

**Run:**

```bash
cd src
python main.py
```

You'll be prompted to enter two sentences. The script prints their cosine similarity (0 = unrelated, 1 = identical meaning).

COMMIT TYPES:
```
INIT   initialized project/file
ADD    added feature
DEL    removed something
FIX    fixed bug
REF    refactor (no behavior change)
DOCS   documentation changes
TEST   tests added/updated
CONF   config changes
SEC    security improvements
MISC   catch-all
```