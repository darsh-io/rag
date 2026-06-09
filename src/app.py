import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from query import build_rag_context, call_llm_stream, rag_query
from ragPipeline.chunk import SUPPORTED_EXTENSIONS
from ragPipeline.vectorstore import get_collection, ingest


# --- Pydantic models ---------------------------------------------------------

class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    source: str
    page: int
    relevance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


# --- App setup ---------------------------------------------------------------

def _load_cfg():
    """Load model config from config.yaml; API keys default to empty (can be supplied per-request)."""
    load_dotenv()

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return {
        # Keys from .env are the fallback; headers sent by the client take priority
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "cohere_api_key": os.getenv("COHERE_API_KEY", ""),
        "embed_model": config["embeddings-model"]["name"],
        "llm_model": config["llm-model"]["name"],
        "reranker_model": config["reranker-model"]["name"],
        "vision_model": config["vision-model"]["name"],
        "embed_url": "https://openrouter.ai/api/v1/embeddings",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
    }


def _resolve_cfg(request: Request) -> dict:
    """Merge server-side config with per-request API keys from headers (headers take priority)."""
    cfg = dict(request.app.state.cfg)
    or_key = request.headers.get("x-openrouter-key")
    co_key = request.headers.get("x-cohere-key")
    if or_key:
        cfg["api_key"] = or_key
    if co_key:
        cfg["cohere_api_key"] = co_key
    return cfg


def _require_keys(cfg: dict):
    """Raise a descriptive 401 if either API key is missing."""
    if not cfg["api_key"]:
        raise HTTPException(
            status_code=401,
            detail="OpenRouter API key not set. Add OPENROUTER_API_KEY to .env or enter it in the app settings.",
        )
    if not cfg["cohere_api_key"]:
        raise HTTPException(
            status_code=401,
            detail="Cohere API key not set. Add COHERE_API_KEY to .env or enter it in the app settings.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cfg = _load_cfg()
    db_path = str(Path(__file__).parent.parent / "chroma_db")
    app.state.collection = get_collection(db_path=db_path)
    yield


app = FastAPI(title="RAGged", lifespan=lifespan)

# Allow the HTML file to be opened directly from disk (file:// origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness check — the frontend polls this to detect whether the server is running."""
    return {"ok": True}


@app.get("/")
async def serve_frontend():
    """Serve the single-page frontend."""
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: Request, file: UploadFile = File(...)):
    """Accept a PDF upload, run the ingest pipeline, and return a summary."""
    cfg = _resolve_cfg(request)
    _require_keys(cfg)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Supported: {supported}")

    # write upload to a temp file preserving the extension so the extractor can dispatch correctly
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    vision_cfg = {
        "api_key": cfg["api_key"],
        "chat_url": cfg["chat_url"],
        "vision_model": cfg["vision_model"],
    }

    try:
        chunks_ingested = ingest(
            tmp_path, request.app.state.collection,
            cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
            vision_cfg=vision_cfg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return IngestResponse(filename=file.filename, chunks_ingested=chunks_ingested)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, body: QueryRequest):
    """Run the RAG pipeline for a question and return the full answer with cited sources."""
    cfg = _resolve_cfg(request)
    _require_keys(cfg)

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        answer, chunks = rag_query(
            body.question, [], request.app.state.collection,
            cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
            cfg["chat_url"], cfg["llm_model"],
            cfg["cohere_api_key"], cfg["reranker_model"],
            top_k=body.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        Source(source=meta["source"], page=meta["page"], relevance=round(score, 4))
        for _, _, meta, score in chunks
    ]
    return QueryResponse(answer=answer, sources=sources)


@app.post("/query/stream")
async def query_stream_endpoint(request: Request, body: QueryRequest):
    """Stream the LLM answer token-by-token as SSE; sources are sent as the first event."""
    cfg = _resolve_cfg(request)
    _require_keys(cfg)

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        # retrieval is synchronous — run it in a thread so we don't block the event loop
        chunks, messages, hyde_doc = await asyncio.to_thread(
            build_rag_context,
            body.question, [], request.app.state.collection,
            cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
            cfg["chat_url"], cfg["llm_model"],
            cfg["cohere_api_key"], cfg["reranker_model"],
            body.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        {"source": meta["source"], "page": meta["page"], "relevance": round(score, 4)}
        for _, _, meta, score in chunks
    ]

    async def event_stream():
        yield f"data: {json.dumps({'type': 'hyde', 'text': hyde_doc})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def stream_in_thread():
            try:
                for delta in call_llm_stream(messages, cfg["api_key"], cfg["chat_url"], cfg["llm_model"]):
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"__error__": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, stream_in_thread)

        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, dict) and "__error__" in item:
                yield f"data: {json.dumps({'type': 'error', 'message': item['__error__']})}\n\n"
                break
            yield f"data: {json.dumps({'type': 'delta', 'text': item})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
