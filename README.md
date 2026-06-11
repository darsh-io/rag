# RAGged

> **Your documents, finally worth talking to.**

A fully custom Retrieval-Augmented Generation (RAG) system built from the ground up — no LangChain, no LlamaIndex, no magic black boxes. Every layer of the pipeline is hand-rolled and understood before the next one was added. Built with a specific vision: give teachers a way to turn their class materials into an intelligent, always-available study assistant for their students.

---

## The Pitch — For Everyone

**For grandma:** Imagine your textbook could talk back. You ask it "what caused World War II?" and instead of flipping through 300 pages, it reads the whole thing in seconds, finds the most relevant parts, and gives you a clear answer with page references. RAGged does exactly that — for any document, any subject, any student, at any hour.

**For the student who just discovered RAG:** RAG stands for Retrieval-Augmented Generation. The idea is simple: instead of asking an AI to guess from memory, you give it your actual documents and tell it to only answer from those. RAGged takes this further — it searches your documents *two different ways* (like Google + a keyword search), merges both results intelligently, re-ranks everything by relevance, and *then* asks the AI. The result is answers that are grounded, cited, and honest about what they don't know.

**For the technical recruiter:** This is a production-quality RAG implementation written entirely in Python — no orchestration frameworks. The author implemented dense retrieval, sparse BM25, reciprocal rank fusion, Cohere reranking, HyDE, JWT auth, bcrypt password hashing, Fernet-encrypted per-user chat storage with HKDF key derivation, a streaming SSE FastAPI backend, and a dark-theme single-page frontend from scratch. Every component is replaceable, debuggable, and documented at the code level.

**For the startup founder:** The EdTech problem: students get a folder of PDFs at the start of term and are expected to learn from static files. RAGged replaces that with a curated, AI-powered study buddy where teachers define *topics*, assign documents to each topic, and students instantly have a context-aware assistant that only knows what the teacher intended — organized by class, by subject, by week. It's private per-user, encrypted at rest, and gives teachers insight into what's confusing students through a feedback dashboard.

---

## The Problem It Solves

Right now, when a teacher shares class materials, students get a folder of PDFs they'll never fully read. When they have a question at 11 PM, there's no one to ask.

RAGged flips this:

- **Teachers** (admins) upload their lecture notes, textbooks, and slides. They group them into **Topics** (e.g., *"Week 3 — Thermodynamics"* or *"Chapter 5 Review"*).
- **Students** (users) sign in, pick a topic, and ask questions in plain English.
- The system finds the most relevant passages, synthesizes an answer, and cites exactly which document and page it came from.
- Students can rate answers (👍 / 👎) and leave comments. Teachers see this feedback to know what's still confusing their class.

---

## Why No Frameworks?

LangChain and LlamaIndex are powerful tools — but they abstract away the parts that matter most when you're trying to *understand* RAG or *control* its behavior. When retrieval fails silently, you need to know whether the embeddings are wrong, the chunker is too aggressive, BM25 isn't tuned, or the reranker is penalizing good results. That diagnosis is impossible when the pipeline is a black box.

RAGged builds every piece explicitly:

| What | Why it matters |
|------|---------------|
| Custom chunker | Sentence-aware splits with configurable overlap — no arbitrary 512-token hard cuts |
| Dual retrieval | Dense (semantic) + sparse (keyword) catch different failure modes |
| RRF fusion | Merges both ranked lists without needing calibrated scores |
| Cohere reranking | Final semantic filter before the LLM sees anything |
| HyDE | Embeds a *hypothetical answer* instead of the question for better semantic match |
| No ORMs, no magic | Every API call is a raw `requests.post()` you can read and modify |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  INGEST PIPELINE                                                   │
│                                                                    │
│  File (PDF/DOCX/image/…)                                           │
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
│  QUERY PIPELINE (per question)                                     │
│                                                                    │
│  User question                                                     │
│       │                                                            │
│       ▼                                                            │
│  ① HyDE  ──► LLM writes a hypothetical answer to the question     │
│       │      (embeds much better than the question itself)         │
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
│  ⑥ LLM (system prompt + history + context + question)             │
│       ──► cited, grounded answer streamed token-by-token          │
└────────────────────────────────────────────────────────────────────┘
```

**Topic filtering** is enforced at both the dense and sparse layers — Chroma's `where` filter restricts vector search to a topic's sources, and `get_filtered()` pre-filters the BM25 corpus so sparse retrieval never crosses topic boundaries.

---

## File Structure

```
src/
├── app.py                    # FastAPI server — all endpoints, auth, chat, feedback
├── query.py                  # HyDE + retrieval pipeline + LLM calls (streaming + sync)
├── static/
│   └── index.html            # Full SPA — dark-theme UI, no JS framework
└── ragPipeline/
    ├── chunk.py              # Universal file chunker (20+ formats, vision for images)
    ├── embeddings.py         # OpenRouter embeddings (raw requests)
    ├── vectorstore.py        # Chroma: ingest, query with source filter, list sources
    ├── bm25.py               # BM25Okapi sparse retrieval
    ├── rrf.py                # Reciprocal Rank Fusion
    ├── reranker.py           # Cohere Rerank v2 client
    └── cosineSim.py          # Pure Python cosine similarity (no numpy)
config/
├── config.yaml               # Model names
├── users.json                # Bcrypt-hashed credentials (gitignored)
├── topics.json               # Topic definitions + source assignments (gitignored)
├── feedback.json             # User ratings + comments (gitignored)
├── .server_secret            # 32-byte HKDF master secret (gitignored)
└── chats/
    └── {username}.enc        # Per-user Fernet-encrypted chat histories (gitignored)
```

---

## Feature Breakdown

### For Students

| Feature | Detail |
|---------|--------|
| Ask questions | Natural language Q&A over any ingested document |
| Topic-scoped answers | Select a topic and answers draw only from its assigned documents |
| Cited answers | Every factual claim references `[Source: file, Page: N]` |
| Conversation history | Multi-turn context — ask follow-ups naturally |
| Persistent chats | Chat history saved per-session, encrypted, private to you |
| Feedback | Rate each answer 👍/👎 with an optional comment |

### For Teachers (Admin)

| Feature | Detail |
|---------|--------|
| Topic management | Create named topics, assign and remove source documents |
| File ingestion | Upload PDFs, Word docs, PowerPoints, images, spreadsheets, and more |
| Browse all chats | Filter every student's chat history by user or topic |
| Feedback dashboard | See all student ratings/comments to identify confusing content |
| User management | Create/delete student accounts, promote to admin |

### Security & Privacy

| Mechanism | Implementation |
|-----------|----------------|
| Authentication | JWT (HS256, 7-day expiry) via `PyJWT`, ephemeral secret per process |
| Password storage | bcrypt hashing via `bcrypt.hashpw()` — no plaintext, no `passlib` |
| Chat encryption | Fernet symmetric encryption per user; key derived via HKDF-SHA256 from a persisted server secret + username as salt |
| Role enforcement | FastAPI `Depends` chain: `_require_auth` → `_require_admin` |
| Topic isolation | Source filter enforced server-side on both dense and sparse retrieval |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS SPA (no React, no Vue) |
| Embeddings | OpenRouter → `text-embedding-3-small` (raw `requests`) |
| LLM | OpenRouter → any model (streaming SSE) |
| Vector store | ChromaDB `EphemeralClient` (in-memory, cosine space) |
| Sparse retrieval | `rank-bm25` BM25Okapi |
| Reranking | Cohere Rerank v2 |
| Auth | PyJWT + bcrypt |
| Encryption | `cryptography` (Fernet + HKDF-SHA256) |
| Image ingestion | Vision model via OpenRouter (GPT-4o / similar) |
| Document parsing | PyPDF2, python-docx, openpyxl, python-pptx, ebooklib, odfpy, Pillow |

---

## Setup

**Requirements:** Python 3.10+, an [OpenRouter](https://openrouter.ai) API key, a [Cohere](https://cohere.com) API key.

```bash
git clone https://github.com/darsh-io/RAGged
cd RAGged
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
cd src
uvicorn app:app --reload
```

Open `http://localhost:8000` — sign in with your admin credentials.

**First steps:**
1. Go to Settings (⚙) and enter your API keys if not using `.env`
2. Click the Topics icon to create a topic (e.g., *"Chapter 1 — Forces"*)
3. Upload your documents directly into the topic
4. Create student accounts via the Users icon
5. Students sign in, select the topic, and start asking

---

## Supported File Types

RAGged ingests virtually any document a teacher might have:

`PDF` · `DOCX` · `PPTX` · `XLSX` · `XLS` · `EPUB` · `ODT` · `ODS` · `ODP` · `TXT` · `MD` · `JSON` · `CSV` · `HTML` · `YAML` · `TOML` · `RST` · `INI` · `CFG` · `CONF` · `LOG` · `JPG` · `PNG` · `GIF` · `WEBP` · `TIFF` · `BMP`

Images are processed by a vision model that generates a textual description before chunking — so scanned pages and diagrams are not dead ends.

---

## How Retrieval Actually Works (The Detail)

Most RAG tutorials stop at "embed your docs, do cosine similarity, done." RAGged goes further:

**1. HyDE (Hypothetical Document Embeddings)**
Instead of embedding *"What causes inflation?"* (a short question), the system first asks the LLM to write a short passage that *would answer* that question. That hypothetical passage embeds into a much richer semantic space, dramatically improving retrieval recall on short or vague questions.

**2. Dual Retrieval**
Dense retrieval (cosine similarity over embeddings) is great at semantic matching but misses exact keyword hits. Sparse retrieval (BM25) is the opposite — it's a statistical model that rewards term frequency and penalizes common words, catching the cases dense retrieval misses. Running both and combining the results is strictly better than either alone.

**3. Reciprocal Rank Fusion**
Rather than picking one winner between dense and sparse, RRF takes both ranked lists and computes a combined score based on rank position: `score = 1/(k + rank)` for each list, summed per document. This works without needing calibrated scores between the two systems.

**4. Cohere Reranking**
The top-10 fused candidates get re-scored by Cohere's cross-encoder model, which reads the question and each passage *together* — far more accurate than embedding similarity alone. The final top-5 go to the LLM.

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

## Vision

RAGged started as a learning exercise — build RAG without the training wheels. It became something more: a practical tool for the specific problem of school-based knowledge delivery.

The goal is a system where:
- A chemistry teacher uploads their semester slides on a Sunday evening
- By Monday morning, 30 students have a tutor that has read every page, never gets tired, is available at 2 AM before an exam, and only knows what the teacher put in — not random internet hallucinations
- When students consistently ask the same question and mark its answer as unhelpful, the teacher sees it in the feedback dashboard and knows exactly which concept needs a better explanation

No third-party AI tutor subscriptions. No data leaving your control (self-hostable). No generic GPT-wrapper that doesn't know your syllabus. Just your documents, your students, and a pipeline you can inspect and trust.

---

*Built with Python. No magic. Every line readable.*
