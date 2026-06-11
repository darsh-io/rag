import asyncio
import json
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import bcrypt
import jwt
import yaml
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from query import build_rag_context, call_llm_stream, rag_query
from ragPipeline.chunk import SUPPORTED_EXTENSIONS
from ragPipeline.vectorstore import get_collection, ingest


# --- Auth --------------------------------------------------------------------

_JWT_SECRET  = secrets.token_hex(32)   # fresh per process; tokens invalidate on restart
_JWT_ALG     = "HS256"
_JWT_DAYS    = 7
_pw_hash: bytes | None = None          # set at startup if APP_PASSWORD is configured
_bearer      = HTTPBearer(auto_error=False)


def _auth_enabled() -> bool:
    return _pw_hash is not None


def _make_token() -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=_JWT_DAYS)
    return jwt.encode({"exp": exp}, _JWT_SECRET, algorithm=_JWT_ALG)


def _valid_token(token: str) -> bool:
    try:
        jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALG])
        return True
    except Exception:
        return False


def _require_auth(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    """FastAPI dependency — raises 401 if auth is enabled and token is missing/invalid."""
    if not _auth_enabled():
        return
    if not credentials or not _valid_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --- Pydantic models ---------------------------------------------------------

class LoginRequest(BaseModel):
    password: str


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
        "api_key":       os.getenv("OPENROUTER_API_KEY", ""),
        "cohere_api_key": os.getenv("COHERE_API_KEY", ""),
        "embed_model":   config["embeddings-model"]["name"],
        "llm_model":     config["llm-model"]["name"],
        "reranker_model": config["reranker-model"]["name"],
        "vision_model":  config["vision-model"]["name"],
        "embed_url":     "https://openrouter.ai/api/v1/embeddings",
        "chat_url":      "https://openrouter.ai/api/v1/chat/completions",
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
    global _pw_hash
    load_dotenv()
    raw_pw = os.getenv("APP_PASSWORD", "").strip()
    if raw_pw:
        _pw_hash = bcrypt.hashpw(raw_pw.encode(), bcrypt.gensalt())
        print("Auth enabled — APP_PASSWORD is set.")
    else:
        print("Auth disabled — set APP_PASSWORD in .env to enable login protection.")

    app.state.cfg        = _load_cfg()
    app.state.collection = get_collection()
    yield


app = FastAPI(title="RAGged", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth endpoints (public) -------------------------------------------------

@app.post("/auth/login")
async def login(body: LoginRequest):
    """Verify password and return a signed JWT. If auth is disabled, always succeeds."""
    if _auth_enabled() and not bcrypt.checkpw(body.password.encode(), _pw_hash):
        raise HTTPException(status_code=401, detail="Incorrect password.")
    return {"token": _make_token(), "auth_enabled": _auth_enabled()}


@app.get("/auth/status")
async def auth_status(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    """Return whether auth is enabled and whether the supplied token is valid."""
    if not _auth_enabled():
        return {"auth_enabled": False, "authenticated": True}
    ok = bool(credentials and _valid_token(credentials.credentials))
    if not ok:
        raise HTTPException(status_code=401, detail="Login required.")
    return {"auth_enabled": True, "authenticated": True}


# --- App endpoints (protected) -----------------------------------------------

@app.get("/health")
async def health():
    """Public liveness check — does not require auth so the frontend can detect the server."""
    return {"ok": True}


@app.get("/")
async def serve_frontend():
    """Serve the single-page frontend."""
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(_require_auth)])
async def ingest_endpoint(request: Request, file: UploadFile = File(...)):
    """Accept a file upload, run the ingest pipeline, and return a summary."""
    cfg = _resolve_cfg(request)
    _require_keys(cfg)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Supported: {supported}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    vision_cfg = {
        "api_key":      cfg["api_key"],
        "chat_url":     cfg["chat_url"],
        "vision_model": cfg["vision_model"],
    }

    try:
        chunks_ingested = ingest(
            tmp_path, request.app.state.collection,
            cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
            vision_cfg=vision_cfg,
            source_name=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    return IngestResponse(filename=file.filename, chunks_ingested=chunks_ingested)


@app.delete("/documents/{source_name:path}", dependencies=[Depends(_require_auth)])
async def delete_document(source_name: str, request: Request):
    """Delete all chunks belonging to a source document from the collection."""
    try:
        request.app.state.collection.delete(where={"source": {"$eq": source_name}})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"deleted": source_name}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_require_auth)])
async def query_endpoint(request: Request, body: QueryRequest):
    """Run the full RAG pipeline and return the answer with cited sources."""
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


@app.post("/query/stream", dependencies=[Depends(_require_auth)])
async def query_stream_endpoint(request: Request, body: QueryRequest):
    """Stream the LLM answer token-by-token as SSE."""
    cfg = _resolve_cfg(request)
    _require_keys(cfg)

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
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
