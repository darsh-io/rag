# Project RAGged

A ground-up build of a RAG (Retrieval-Augmented Generation) system — no framework abstractions, no black boxes. Every component is written from scratch and understood before the next one is added.

## What it is

A document Q&A system that lets you ingest PDFs and ask questions about them, getting grounded, cited answers. The retrieval pipeline combines dense vector search with sparse BM25, fuses the results via Reciprocal Rank Fusion, and reranks the final candidates with Cohere before sending them to an LLM.

## Stack

| Layer | Tool |
|---|---|
| Embeddings + LLM | OpenRouter (raw `requests`) |
| Vector DB | Chroma (local, persistent) |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Reranking | Cohere Rerank v2 |
| Deployment | Local CLI |

## Current state

The full retrieval-to-answer pipeline is working end-to-end:

```
src/
├── main.py                   # CLI entry point — ingest/query loop
├── query.py                  # Full RAG pipeline + LLM call
└── ragPipeline/
    ├── embeddings.py         # OpenRouter embeddings API client
    ├── cosineSim.py          # Cosine similarity, pure Python
    ├── chunk.py              # PDF → sentence-aware overlapping chunks
    ├── vectorstore.py        # Chroma collection management (ingest + query)
    ├── bm25.py               # Sparse BM25 retrieval over full corpus
    ├── rrf.py                # Reciprocal Rank Fusion
    └── reranker.py           # Cohere rerank API client
```

**Retrieval pipeline per query:**
1. Dense retrieval — top 10 chunks via Chroma cosine similarity
2. Sparse retrieval — top 10 chunks via BM25Okapi over the full corpus
3. RRF fusion — merge both ranked lists into a single top-10
4. Cohere rerank — re-score and cut to top 5
5. LLM answer — top 5 chunks + conversation history sent to the model

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
cp .env.example .env
```

Add your keys to `.env`:

```
OPENROUTER_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
```

## Usage

```bash
cd src
python main.py
```

You'll be prompted for a mode on each loop iteration:

| Mode | What it does |
|---|---|
| `ingest` | Chunk a PDF, embed it, and store it in Chroma |
| `query` | Start an interactive Q&A session with history |
| `quit` | Exit |

Inside query mode:

| Command | Effect |
|---|---|
| (any text) | Run the full RAG pipeline and print a cited answer |
| `/clear` | Reset conversation history |
| `/quit` | Return to mode selection |

## Config

Model names are set in `config/config.yaml`:

```yaml
embeddings-model:
  name: ...
llm-model:
  name: ...
reranker-model:
  name: ...
```

## Commit types

| Tag | Meaning |
|---|---|
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
