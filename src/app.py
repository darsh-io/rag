import asyncio
import base64
import json
import os
import secrets
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bcrypt
import jwt
import yaml
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from query import build_rag_context, call_llm_stream, rag_query
from ragPipeline.chunk import SUPPORTED_EXTENSIONS
from ragPipeline.vectorstore import get_collection, ingest, list_sources


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


# --- Topics ------------------------------------------------------------------

_TOPICS_PATH = Path(__file__).parent.parent / "config" / "topics.json"
_topics: dict = {}  # {topic_id: {id, name, sources: [], created_by, created_at}}


def _load_topics() -> dict:
    if _TOPICS_PATH.exists():
        try:
            return json.loads(_TOPICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_topics() -> None:
    _TOPICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TOPICS_PATH.write_text(json.dumps(_topics, indent=2), encoding="utf-8")


# --- Feedback storage --------------------------------------------------------

_FEEDBACK_PATH = Path(__file__).parent.parent / "config" / "feedback.json"


def _load_feedback() -> list:
    if _FEEDBACK_PATH.exists():
        try:
            return json.loads(_FEEDBACK_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_feedback(fb: list) -> None:
    _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FEEDBACK_PATH.write_text(json.dumps(fb, indent=2), encoding="utf-8")


# --- Chat storage (encrypted per-user) --------------------------------------

_SECRET_PATH = Path(__file__).parent.parent / "config" / ".server_secret"
_CHATS_DIR   = Path(__file__).parent.parent / "config" / "chats"
_server_secret: bytes = b""


def _load_or_create_secret() -> bytes:
    if _SECRET_PATH.exists():
        return bytes.fromhex(_SECRET_PATH.read_text().strip())
    raw = secrets.token_bytes(32)
    _SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SECRET_PATH.write_text(raw.hex())
    return raw


def _fernet(username: str) -> Fernet:
    """Derive a stable per-user Fernet key from the server secret via HKDF."""
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=username.encode(), info=b"rewise-chat")
    return Fernet(base64.urlsafe_b64encode(hkdf.derive(_server_secret)))


def _load_chats(username: str) -> dict:
    path = _CHATS_DIR / f"{username}.enc"
    if not path.exists():
        return {}
    try:
        return json.loads(_fernet(username).decrypt(path.read_bytes()))
    except Exception:
        return {}


def _save_chats(username: str, chats: dict) -> None:
    _CHATS_DIR.mkdir(parents=True, exist_ok=True)
    (_CHATS_DIR / f"{username}.enc").write_bytes(
        _fernet(username).encrypt(json.dumps(chats).encode())
    )


# --- Pydantic models ---------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class CreateTopicRequest(BaseModel):
    name: str


class AddTopicSourceRequest(BaseModel):
    source_name: str


class UpsertChatRequest(BaseModel):
    id: str
    title: str
    createdAt: str
    topicId: str | None = None
    messages: list[Any] = []


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int


class FeedbackRequest(BaseModel):
    chat_id: str
    chat_title: str = ""
    topic_id: str | None = None
    topic_name: str = ""
    message_index: int
    question: str
    answer: str
    rating: str  # "up" or "down"
    comment: str = ""


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    topic_id: str | None = None


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
    global _users, _topics, _server_secret
    load_dotenv()

    _server_secret = _load_or_create_secret()
    _users  = _load_users()
    _topics = _load_topics()

    admin_user = os.getenv("ADMIN_USERNAME", "").strip()
    admin_pw   = os.getenv("ADMIN_PASSWORD", "").strip()
    if admin_user and admin_pw and admin_user not in _users:
        _users[admin_user] = {"pw_hash_hex": _hash_pw(admin_pw), "role": "admin"}
        _save_users()
        print(f"Created admin account '{admin_user}'.")

    if _auth_enabled():
        print(f"Auth enabled — {len(_users)} user(s), {len(_topics)} topic(s).")
    else:
        print("Auth disabled — set ADMIN_USERNAME and ADMIN_PASSWORD in .env to enable.")

    app.state.cfg        = _load_cfg()
    app.state.collection = get_collection()
    yield


app = FastAPI(title="rewise", lifespan=lifespan)

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


# --- Topics endpoints --------------------------------------------------------

@app.get("/topics")
async def get_topics(payload: dict = Depends(_require_auth)):
    """Return all topics. Any authenticated user can read."""
    return list(_topics.values())


@app.post("/topics")
async def create_topic(body: CreateTopicRequest, payload: dict = Depends(_require_admin)):
    """Create a new topic. Admin only."""
    if not body.name.strip():
        raise HTTPException(status_code=400, detail="Topic name cannot be empty.")
    topic_id = secrets.token_hex(8)
    _topics[topic_id] = {
        "id": topic_id,
        "name": body.name.strip(),
        "sources": [],
        "created_by": payload["sub"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_topics()
    return _topics[topic_id]


@app.delete("/topics/{topic_id}")
async def delete_topic(topic_id: str, payload: dict = Depends(_require_admin)):
    """Delete a topic. Admin only."""
    if topic_id not in _topics:
        raise HTTPException(status_code=404, detail="Topic not found.")
    del _topics[topic_id]
    _save_topics()
    return {"deleted": topic_id}


@app.post("/topics/{topic_id}/sources")
async def add_topic_source(topic_id: str, body: AddTopicSourceRequest, payload: dict = Depends(_require_admin)):
    """Add a source to a topic. Admin only."""
    if topic_id not in _topics:
        raise HTTPException(status_code=404, detail="Topic not found.")
    if body.source_name not in _topics[topic_id]["sources"]:
        _topics[topic_id]["sources"].append(body.source_name)
        _save_topics()
    return _topics[topic_id]


@app.delete("/topics/{topic_id}/sources/{source_name:path}")
async def remove_topic_source(topic_id: str, source_name: str, payload: dict = Depends(_require_admin)):
    """Remove a source from a topic. Admin only."""
    if topic_id not in _topics:
        raise HTTPException(status_code=404, detail="Topic not found.")
    if source_name in _topics[topic_id]["sources"]:
        _topics[topic_id]["sources"].remove(source_name)
        _save_topics()
    return _topics[topic_id]


# --- Sources endpoint --------------------------------------------------------

@app.get("/sources")
async def get_sources(request: Request, payload: dict = Depends(_require_auth)):
    """Return unique source names currently in the vector store."""
    return list_sources(request.app.state.collection)


# --- Chat endpoints ----------------------------------------------------------

@app.get("/chats")
async def list_chats(payload: dict = Depends(_require_auth)):
    """Return metadata list of the current user's chat sessions."""
    chats = _load_chats(payload["sub"])
    return sorted(
        [{"id": s["id"], "title": s.get("title", ""), "createdAt": s.get("createdAt", ""),
          "topicId": s.get("topicId"), "messageCount": len(s.get("messages", []))}
         for s in chats.values()],
        key=lambda s: s["createdAt"], reverse=True,
    )


@app.get("/chats/{session_id}")
async def get_chat(session_id: str, payload: dict = Depends(_require_auth)):
    """Return a single full chat session for the current user."""
    chats = _load_chats(payload["sub"])
    if session_id not in chats:
        raise HTTPException(status_code=404, detail="Session not found.")
    return chats[session_id]


@app.put("/chats/{session_id}")
async def upsert_chat(session_id: str, body: UpsertChatRequest, payload: dict = Depends(_require_auth)):
    """Create or update a chat session for the current user."""
    chats = _load_chats(payload["sub"])
    chats[session_id] = body.model_dump()
    _save_chats(payload["sub"], chats)
    return {"ok": True}


@app.delete("/chats/{session_id}")
async def delete_chat(session_id: str, payload: dict = Depends(_require_auth)):
    """Delete a chat session for the current user."""
    chats = _load_chats(payload["sub"])
    if session_id not in chats:
        raise HTTPException(status_code=404, detail="Session not found.")
    del chats[session_id]
    _save_chats(payload["sub"], chats)
    return {"deleted": session_id}


# --- Feedback endpoints ------------------------------------------------------

@app.post("/feedback", dependencies=[Depends(_require_auth)])
async def submit_feedback(body: FeedbackRequest, payload: dict = Depends(_require_auth)):
    """Store user feedback for an answer. Replaces any prior rating for the same message."""
    if body.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="Rating must be 'up' or 'down'.")
    fb = _load_feedback()
    fb = [f for f in fb if not (
        f["username"] == payload["sub"] and
        f["chat_id"] == body.chat_id and
        f["message_index"] == body.message_index
    )]
    fb.append({
        "id": secrets.token_hex(8),
        "username": payload["sub"],
        "chat_id": body.chat_id,
        "chat_title": body.chat_title,
        "topic_id": body.topic_id,
        "topic_name": body.topic_name,
        "message_index": body.message_index,
        "question": body.question,
        "answer": body.answer,
        "rating": body.rating,
        "comment": body.comment,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_feedback(fb)
    return {"ok": True}


# --- Admin endpoints ---------------------------------------------------------

@app.get("/admin/chats")
async def admin_list_chats(
    username: str | None = None,
    topic_id: str | None = None,
    payload: dict = Depends(_require_admin),
):
    """List chat metadata for all users, optionally filtered by username and/or topic. Admin only."""
    targets = [username] if username else list(_users.keys())
    result = []
    for u in targets:
        for s in _load_chats(u).values():
            if topic_id and s.get("topicId") != topic_id:
                continue
            result.append({
                "username": u,
                "id": s["id"],
                "title": s.get("title", ""),
                "topicId": s.get("topicId"),
                "createdAt": s.get("createdAt", ""),
                "messageCount": len(s.get("messages", [])),
            })
    return sorted(result, key=lambda s: s["createdAt"], reverse=True)


@app.get("/admin/chats/{username}/{chat_id}")
async def admin_get_chat(username: str, chat_id: str, payload: dict = Depends(_require_admin)):
    """Get the full content of any user's chat session. Admin only."""
    if username not in _users:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found.")
    chats = _load_chats(username)
    if chat_id not in chats:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return chats[chat_id]


@app.get("/admin/feedback")
async def get_admin_feedback(
    username: str | None = None,
    rating: str | None = None,
    payload: dict = Depends(_require_admin),
):
    """Return all feedback, optionally filtered by username or rating. Admin only."""
    fb = _load_feedback()
    if username:
        fb = [f for f in fb if f["username"] == username]
    if rating:
        fb = [f for f in fb if f["rating"] == rating]
    return sorted(fb, key=lambda f: f["created_at"], reverse=True)


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


def _source_filter_for(topic_id: str | None) -> list | None:
    """Return the source list for a topic_id, or None for no filter."""
    if not topic_id or topic_id not in _topics:
        return None
    return _topics[topic_id]["sources"] or None


def _assert_topic_has_sources(topic_id: str | None, source_filter: list | None) -> None:
    if topic_id and topic_id in _topics and not source_filter:
        raise HTTPException(status_code=400, detail="This topic has no sources assigned yet.")


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_require_auth)])
async def query_endpoint(request: Request, body: QueryRequest):
    """Run the full RAG pipeline and return the answer with cited sources."""
    cfg = _resolve_cfg(request)
    _require_keys(cfg)

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    source_filter = _source_filter_for(body.topic_id)
    _assert_topic_has_sources(body.topic_id, source_filter)
    try:
        answer, chunks = rag_query(
            body.question, [], request.app.state.collection,
            cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
            cfg["chat_url"], cfg["llm_model"],
            cfg["cohere_api_key"], cfg["reranker_model"],
            top_k=body.top_k,
            source_filter=source_filter,
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

    source_filter = _source_filter_for(body.topic_id)
    _assert_topic_has_sources(body.topic_id, source_filter)
    try:
        chunks, messages, hyde_doc = await asyncio.to_thread(
            build_rag_context,
            body.question, [], request.app.state.collection,
            cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
            cfg["chat_url"], cfg["llm_model"],
            cfg["cohere_api_key"], cfg["reranker_model"],
            body.top_k,
            source_filter,
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
