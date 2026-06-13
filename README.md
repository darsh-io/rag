# rewise

> Your documents, finally worth talking to.

A fully custom Retrieval-Augmented Generation (RAG) system built from the ground up — no LangChain, no LlamaIndex, no magic black boxes. Every layer of the pipeline is hand-rolled and understood before the next one was added.

Built with a real problem in mind: give teachers a way to turn their class materials into an intelligent, always-available study assistant for their students. Upload your lecture notes, textbooks, and slides. Group them by topic. Let students ask questions and get cited, grounded answers — at 2 AM before the exam, with no one to disturb.

---

## What It Does

Imagine your entire course reader could answer questions. You ask *"what causes inflation?"* and instead of flipping through 300 pages, it reads everything in seconds, finds the most relevant passages, and replies with a clear answer that tells you exactly which document and page it came from. That's rewise — for any subject, any file format, any student, at any hour.

The magic is in what happens between your question and the answer. Most AI tools either hallucinate from memory or do a single keyword search. rewise does neither. It runs your question through a five-stage retrieval pipeline that combines semantic search, keyword matching, ranked fusion, and a dedicated reranking model — before the LLM ever sees a single word.

---

## The Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│  INGEST                                                            │
│                                                                    │
│  File (PDF / DOCX / PPTX / image / …)                             │
│       │                                                            │
│       ▼                                                            │
│  chunk_file()  ──► sentence-aware overlapping chunks + metadata   │
│       │                                                            │
│       ▼                                                            │
│  getEmbeddings()  ──► OpenRouter text-embedding-3-small           │
│       │                                                            │
│       ▼                                                            │
│  Chroma EphemeralClient  ──► in-memory vector store (cosine)      │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│  QUERY                                                             │
│                                                                    │
│  User question                                                     │
│       │                                                            │
│       ▼                                                            │
│  ① HyDE  ──► LLM writes a hypothetical answer to the question     │
│              (embeds far richer than a short question alone)       │
│       │                                                            │
│       ├──────────────────────────────────────────────┐            │
│       ▼                                              ▼            │
│  ② Dense retrieval                        ③ Sparse retrieval     │
│     Chroma cosine similarity                 BM25Okapi            │
│     top-10 chunks                            top-10 chunks        │
│       │                                              │            │
│       └──────────────┬───────────────────────────────┘            │
│                      ▼                                             │
│  ④ Reciprocal Rank Fusion  ──► merged top-10                      │
│                      │                                             │
│                      ▼                                             │
│  ⑤ Cohere Rerank v2  ──► top-5 by true semantic relevance        │
│                      │                                             │
│                      ▼                                             │
│  ⑥ LLM  ──► cited, grounded answer streamed token-by-token       │
└────────────────────────────────────────────────────────────────────┘
```

**Topic filtering** is enforced at both the dense and sparse layers — Chroma's `where` filter restricts vector search to a topic's assigned sources, and `get_filtered()` pre-filters the BM25 corpus so sparse retrieval never crosses topic boundaries.

---

## Why No Frameworks?

LangChain and LlamaIndex abstract away the parts that matter most when you're trying to understand or control RAG behavior. When retrieval fails silently, you need to know whether the embeddings are wrong, the chunker is too aggressive, BM25 isn't tuned, or the reranker is penalizing good results. That diagnosis is impossible when the pipeline is a black box.

rewise builds every piece explicitly:

| Component | Why it matters |
|-----------|---------------|
| Custom chunker | Sentence-aware splits with configurable overlap — no arbitrary hard cuts |
| Dual retrieval | Dense (semantic) + sparse (keyword) catch different failure modes |
| RRF fusion | Merges both ranked lists without needing calibrated scores from either |
| Cohere reranking | Cross-encoder re-scores question+passage together — far more accurate than embedding similarity alone |
| HyDE | Embeds a *hypothetical answer* instead of the raw question for better semantic match |
| Raw API calls | Every request is a `requests.post()` you can read, log, and modify |

---

## Features

### For Students

| Feature | Detail |
|---------|--------|
| Natural language Q&A | Ask anything about ingested documents in plain English |
| Topic-scoped answers | Select a topic and answers draw only from its assigned documents |
| Cited answers | Every factual claim references `[Source: file, Page: N]` |
| Conversation history | Multi-turn context — ask follow-ups naturally |
| Persistent chats | Sessions saved per-user, encrypted, completely private |
| Answer feedback | Rate each answer 👍/👎 with an optional comment |

### For Teachers (Admin)

| Feature | Detail |
|---------|--------|
| Topic management | Create named topics, assign and remove source documents |
| File ingestion | Upload directly inside each topic — no separate upload step |
| Browse all chats | Filter every student's chat history by user or topic |
| Feedback dashboard | See all student ratings and comments to spot confusing content |
| User management | Create, deactivate, and delete accounts; assign student/teacher/supradmin roles |

### Security & Privacy

| Mechanism | Implementation |
|-----------|----------------|
| Authentication | JWT HS256 via `PyJWT`, 7-day expiry, ephemeral secret per process |
| Passwords | `bcrypt.hashpw()` — no plaintext storage, no `passlib` |
| Chat encryption | Fernet symmetric encryption per user; key derived via HKDF-SHA256 (server secret + username as salt) |
| Role enforcement | FastAPI `Depends` chain; 3 roles: `student`, `teacher`, `supradmin` |
| Topic isolation | Source filter enforced server-side on both retrieval layers |

---

## How the Retrieval Stages Actually Work

**HyDE (Hypothetical Document Embeddings)**
Short questions embed poorly — they don't contain enough semantic signal. Instead of embedding *"What causes inflation?"*, the system first asks the LLM to write a short passage that would answer that question. That hypothetical answer embeds into a much richer space and retrieves far more relevant chunks.

**Dual Retrieval**
Dense retrieval (cosine similarity over embeddings) excels at semantic matching but misses exact terminology. Sparse retrieval (BM25Okapi) is a statistical model that rewards term frequency and penalizes common words — catching what dense misses. Running both is strictly better than either alone.

**Reciprocal Rank Fusion**
Rather than picking a winner between dense and sparse, RRF computes `score = Σ 1/(k + rank)` across both ranked lists. No calibration needed between the two systems — position alone does the work.

**Cohere Reranking**
The top-10 fused candidates get re-scored by a cross-encoder that reads the question and each passage *together*. Unlike embedding similarity (which encodes question and passage independently), a cross-encoder sees both at once and produces a true relevance judgment. Final top-5 go to the LLM.

---

## File Structure

```
serve.py                      # Entry point — python serve.py
setup.py                      # First-run setup wizard
src/
├── app.py                    # FastAPI app wiring + lifespan
├── db.py                     # SQLite schema, migrations, connection helper
├── deps.py                   # Auth dependencies (JWT decode, role guards)
├── encryption.py             # Per-user Fernet encryption via HKDF
├── config.py                 # Config loader (config.yaml + .env)
├── query.py                  # HyDE + full retrieval pipeline + streaming LLM calls
├── routers/
│   ├── auth.py               # Login, self-registration, /me
│   ├── users.py              # Supradmin user CRUD
│   ├── classes.py            # Class CRUD, teacher/student assignment
│   ├── topics.py             # Topic CRUD per class
│   ├── documents.py          # Document ingest + delete per topic
│   ├── chats.py              # Chat creation, SSE streaming query, history
│   └── feedback.py           # Per-message ratings and comments
├── static/
│   ├── index.html            # SPA shell
│   ├── app.js                # All frontend logic
│   └── styles.css            # Dark-theme design system
└── ragPipeline/
    ├── chunk.py              # Universal file chunker (20+ formats, vision for images)
    ├── embeddings.py         # OpenRouter embeddings client (raw requests)
    ├── vectorstore.py        # Chroma PersistentClient: ingest, filtered query
    ├── bm25.py               # BM25Okapi sparse retrieval
    ├── rrf.py                # Reciprocal Rank Fusion
    ├── reranker.py           # Cohere Rerank v2 client
    └── cosineSim.py          # Pure Python cosine similarity (no numpy)
config/
├── config.yaml               # Model names
├── rewise.db                 # SQLite database (gitignored)
├── chroma/                   # Persisted vector store (gitignored)
└── .server_secret            # 32-byte HKDF master secret (gitignored)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS SPA |
| Embeddings | OpenRouter → `text-embedding-3-small` (raw `requests`) |
| LLM | OpenRouter → any model (streaming SSE) |
| Vector store | ChromaDB `PersistentClient` (on-disk, cosine) |
| Sparse retrieval | `rank-bm25` BM25Okapi |
| Reranking | Cohere Rerank v2 |
| Auth | PyJWT + bcrypt |
| Encryption | `cryptography` — Fernet + HKDF-SHA256 |
| Image ingestion | Vision model via OpenRouter |
| Document parsing | PyPDF2, python-docx, openpyxl, python-pptx, ebooklib, odfpy, Pillow |

---

## Supported File Types

`PDF` · `DOCX` · `PPTX` · `XLSX` · `XLS` · `EPUB` · `ODT` · `ODS` · `ODP` · `TXT` · `MD` · `JSON` · `CSV` · `HTML` · `YAML` · `TOML` · `RST` · `INI` · `CFG` · `CONF` · `LOG` · `JPG` · `PNG` · `GIF` · `WEBP` · `TIFF` · `BMP`

Images are processed by a vision model that produces a textual description before chunking — scanned pages and diagrams are not dead ends.

---

## Setup

**Requirements:** Python 3.10+, an [OpenRouter](https://openrouter.ai) API key, a [Cohere](https://cohere.com) API key.

```bash
git clone https://github.com/darsh-io/rewise.git
cd rag
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
OPENROUTER_API_KEY=sk-or-...
COHERE_API_KEY=...

# Creates the first admin account on first run
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_password
```

Start the server:

```bash
python serve.py
```

Open `http://localhost:8000` and sign in.

**Getting started:**
1. Sign in as the admin account set in `.env`
2. Go to **Admin → Classes** and create a class
3. Go to **Admin → Topics** and create a topic, then upload documents into it
4. Students self-register at the login screen — they pick their class on sign-up
5. Students sign in, select a topic, and start asking

---

## Config

`config/config.yaml` controls which models are used:

```yaml
embeddings-model:
  name: openai/text-embedding-3-small
llm-model:
  name: nex-agi/nex-n2-pro:free
reranker-model:
  name: rerank-english-v2.0
vision-model:
  name: openai/gpt-4o
```

Any OpenRouter-compatible model slug works for embeddings and LLM.

---

## Vision

The problem this is solving: students get a folder of PDFs at the start of term and are expected to learn from static files. When they have a question at 11 PM before an exam, there's no one to ask.

rewise replaces that with a curated, context-aware study assistant — where a teacher defines the knowledge base, and students get an AI that only knows what the teacher put in. Not random internet hallucinations. Not a generic chatbot. Just your syllabus, accurately recalled, with sources.

When students consistently ask the same question and mark its answer as unhelpful, the teacher sees it in the feedback dashboard — and knows exactly which concept needs a better explanation.

Self-hostable. No third-party subscriptions. No data leaving your control. Every line of the pipeline readable and replaceable.

---

## Commit Convention

| Tag | Meaning |
|-----|---------|
| `INIT` | First commit or major milestone |
| `ADD` | New feature or file |
| `DEL` | Removed something |
| `FIX` | Bug fix |
| `REF` | Refactor (no behavior change) |
| `DOCS` | Documentation only |
| `TEST` | Tests only |
| `CONF` | Config change |
| `SEC` | Security fix |
| `IMPR` | Improvement to existing feature |
| `MISC` | Anything else |

---

*Built with Python. No magic. Every line readable.*
