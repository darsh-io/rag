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
_bearer      = HTTPBearer(auto_error=False)
_USERS_PATH  = Path(__file__).parent.parent / "config" / "users.json"
_users: dict = {}  # {username: {"pw_hash_hex": str, "role": "admin"|"user"}}


def _load_users() -> dict:
    if _USERS_PATH.exists():
        try:
            return json.loads(_USERS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_users() -> None:
    _USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _USERS_PATH.write_text(json.dumps(_users, indent=2), encoding="utf-8")


def _auth_enabled() -> bool:
    return bool(_users)


def _hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).hex()


def _check_pw(password: str, pw_hash_hex: str) -> bool:
    return bcrypt.checkpw(password.encode(), bytes.fromhex(pw_hash_hex))


def _make_token(username: str, role: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=_JWT_DAYS)
    return jwt.encode({"sub": username, "role": role, "exp": exp}, _JWT_SECRET, algorithm=_JWT_ALG)


def _decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALG])
    except Exception:
        return None


def _require_auth(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """FastAPI dependency — returns JWT payload dict, raises 401 if invalid."""
    if not _auth_enabled():
        return {"sub": "anonymous", "role": "admin"}
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(credentials.credentials)
    if not payload or payload.get("sub") not in _users:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def _require_admin(payload: dict = Depends(_require_auth)) -> dict:
    """FastAPI dependency — raises 403 if the caller is not an admin."""
    if payload["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return payload


# --- Pydantic models ---------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


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
    global _users
    load_dotenv()
    _users = _load_users()

    admin_user = os.getenv("ADMIN_USERNAME", "").strip()
    admin_pw   = os.getenv("ADMIN_PASSWORD", "").strip()
    if admin_user and admin_pw and admin_user not in _users:
        _users[admin_user] = {"pw_hash_hex": _hash_pw(admin_pw), "role": "admin"}
        _save_users()
        print(f"Created admin account '{admin_user}'.")

    if _auth_enabled():
        print(f"Auth enabled — {len(_users)} user(s) registered.")
    else:
        print("Auth disabled — set ADMIN_USERNAME and ADMIN_PASSWORD in .env to enable.")

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


# --- Auth endpoints ----------------------------------------------------------

@app.post("/auth/login")
async def login(body: LoginRequest):
    """Verify username + password and return a signed JWT."""
    if not _auth_enabled():
        token = _make_token("anonymous", "admin")
        return {"token": token, "username": "anonymous", "role": "admin", "auth_enabled": False}
    user = _users.get(body.username)
    if not user or not _check_pw(body.password, user["pw_hash_hex"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    token = _make_token(body.username, user["role"])
    return {"token": token, "username": body.username, "role": user["role"], "auth_enabled": True}


@app.get("/auth/status")
async def auth_status(payload: dict = Depends(_require_auth)):
    """Return current user info. Raises 401 if unauthenticated."""
    return {
        "auth_enabled": _auth_enabled(),
        "authenticated": True,
        "username": payload["sub"],
        "role": payload["role"],
    }


@app.get("/auth/users")
async def list_users(payload: dict = Depends(_require_admin)):
    """Return the full user list. Admin only."""
    return [{"username": u, "role": d["role"]} for u, d in _users.items()]


@app.post("/auth/users")
async def create_user(body: CreateUserRequest, payload: dict = Depends(_require_admin)):
    """Create a new user account. Admin only."""
    if not body.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty.")
    if body.username in _users:
        raise HTTPException(status_code=409, detail=f"User '{body.username}' already exists.")
    if body.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'user'.")
    if not body.password:
        raise HTTPException(status_code=400, detail="Password cannot be empty.")
    _users[body.username] = {"pw_hash_hex": _hash_pw(body.password), "role": body.role}
    _save_users()
    return {"username": body.username, "role": body.role}


@app.delete("/auth/users/{username}")
async def delete_user(username: str, payload: dict = Depends(_require_admin)):
    """Delete a user account. Admin only. Cannot delete yourself."""
    if username not in _users:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found.")
    if username == payload["sub"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own account.")
    del _users[username]
    _save_users()
    return {"deleted": username}


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
