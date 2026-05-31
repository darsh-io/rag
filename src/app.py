import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from query import build_rag_context, call_llm, call_llm_stream, rag_query
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
    """Load credentials and model config from .env and config.yaml."""
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing in .env")
    if not cohere_api_key:
        raise RuntimeError("COHERE_API_KEY missing in .env")

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return {
        "api_key": api_key,
        "cohere_api_key": cohere_api_key,
        "embed_model": config["embeddings-model"]["name"],
        "llm_model": config["llm-model"]["name"],
        "reranker_model": config["reranker-model"]["name"],
        "embed_url": "https://openrouter.ai/api/v1/embeddings",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cfg = _load_cfg()
    db_path = str(Path(__file__).parent.parent / "chroma_db")
    app.state.collection = get_collection(db_path=db_path)
    yield


app = FastAPI(title="RAGged", lifespan=lifespan)


# --- Endpoints ---------------------------------------------------------------

@app.get("/")
async def serve_frontend():
    """Serve the single-page frontend."""
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: Request, file: UploadFile = File(...)):
    """Accept a PDF upload, run the ingest pipeline, and return a summary."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # write upload to a temp file so the ingest pipeline can read it by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        cfg = request.app.state.cfg
        chunks_ingested = ingest(
            tmp_path, request.app.state.collection,
            cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
        )
    finally:
        os.unlink(tmp_path)

    return IngestResponse(filename=file.filename, chunks_ingested=chunks_ingested)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, body: QueryRequest):
    """Run the RAG pipeline for a question and return the full answer with cited sources."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    cfg = request.app.state.cfg

    # history is stateless per HTTP request; session history belongs in a future layer
    answer, chunks = rag_query(
        body.question, [], request.app.state.collection,
        cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
        cfg["chat_url"], cfg["llm_model"],
        cfg["cohere_api_key"], cfg["reranker_model"],
        top_k=body.top_k,
    )

    sources = [
        Source(source=meta["source"], page=meta["page"], relevance=round(score, 4))
        for _, _, meta, score in chunks
    ]

    return QueryResponse(answer=answer, sources=sources)


@app.post("/query/stream")
async def query_stream_endpoint(request: Request, body: QueryRequest):
    """Stream the LLM answer token-by-token as SSE; sources are sent as the first event."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    cfg = request.app.state.cfg

    # retrieval is synchronous — run it in a thread so we don't block the event loop
    chunks, messages, hyde_doc = await asyncio.to_thread(
        build_rag_context,
        body.question, [], request.app.state.collection,
        cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
        cfg["chat_url"], cfg["llm_model"],
        cfg["cohere_api_key"], cfg["reranker_model"],
        body.top_k,
    )

    sources = [
        {"source": meta["source"], "page": meta["page"], "relevance": round(score, 4)}
        for _, _, meta, score in chunks
    ]

    async def event_stream():
        # emit the hypothetical doc so the client can show what HyDE generated
        yield f"data: {json.dumps({'type': 'hyde', 'text': hyde_doc})}\n\n"
        # second event: all sources so the client can render citations immediately
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # stream LLM deltas from a background thread via a queue
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def stream_in_thread():
            try:
                for delta in call_llm_stream(messages, cfg["api_key"], cfg["chat_url"], cfg["llm_model"]):
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        asyncio.get_event_loop().run_in_executor(None, stream_in_thread)

        while True:
            delta = await queue.get()
            if delta is None:
                break
            yield f"data: {json.dumps({'type': 'delta', 'text': delta})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
