"""Chat CRUD + streaming RAG query."""
import asyncio, json
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from src.db import get_db, new_id, now
from src.deps import get_current_user, require_role, class_access, own_chat, UserInToken
from src.encryption import encrypt, decrypt
from src.config import cfg, CHROMA_DIR
from src.ragPipeline.vectorstore import get_collection
from src.query import build_rag_context, call_llm_stream

router = APIRouter(tags=["chats"])
_teacher_or_admin = require_role("teacher", "supradmin")


class CreateChat(BaseModel):
    topic_id: Optional[str] = None
    title: Optional[str] = None


class QueryRequest(BaseModel):
    question: str


# ── student/self chat endpoints ───────────────────────────────────────────────

@router.get("/classes/{class_id}/chats/me")
def list_my_chats(
    class_id: str,
    topic_id: Optional[str] = None,
    caller: UserInToken = Depends(get_current_user),
    cls=Depends(class_access("member")),
):
    with get_db() as conn:
        q = "SELECT id,title,topic_id,created_at,updated_at FROM chats WHERE user_id=? AND class_id=?"
        params = [caller.id, class_id]
        if topic_id:
            q += " AND topic_id=?"
            params.append(topic_id)
        q += " ORDER BY updated_at DESC"
        rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


@router.post("/classes/{class_id}/chats")
def create_chat(
    class_id: str,
    body: CreateChat,
    caller: UserInToken = Depends(get_current_user),
    cls=Depends(class_access("member")),
):
    chat_id = new_id()
    ts = now()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chats (id,user_id,topic_id,class_id,title,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
            (chat_id, caller.id, body.topic_id, class_id, body.title or "New chat", ts, ts),
        )
    return {"id": chat_id}


@router.get("/chats/{chat_id}")
def get_chat(
    chat_id: str,
    caller: UserInToken = Depends(get_current_user),
    chat=Depends(own_chat()),
):
    with get_db() as conn:
        msgs = conn.execute(
            "SELECT id,role,content,sources_json,feedback_rating,feedback_comment,created_at "
            "FROM chat_messages WHERE chat_id=? ORDER BY created_at",
            (chat_id,),
        ).fetchall()

    decrypted = []
    owner_id = chat["user_id"]
    for m in msgs:
        row = dict(m)
        try:
            row["content"] = decrypt(owner_id, row["content"])
        except Exception:
            pass
        decrypted.append(row)

    return {**chat, "messages": decrypted}


@router.delete("/chats/{chat_id}")
def delete_chat(
    chat_id: str,
    caller: UserInToken = Depends(get_current_user),
    chat=Depends(own_chat()),
):
    if caller.role == "student" and chat["user_id"] != caller.id:
        raise HTTPException(403, "Not your chat")
    with get_db() as conn:
        conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    return {"ok": True}


@router.post("/chats/{chat_id}/query/stream")
def query_stream(
    chat_id: str,
    body: QueryRequest,
    caller: UserInToken = Depends(get_current_user),
    chat=Depends(own_chat()),
):
    if not body.question.strip():
        raise HTTPException(400, "Question must not be empty")

    if not cfg["api_key"]:
        raise HTTPException(401, "OpenRouter API key not configured")
    if not cfg["cohere_api_key"]:
        raise HTTPException(401, "Cohere API key not configured")

    class_id = chat["class_id"]
    topic_id = chat["topic_id"]

    # build conversation history from stored messages
    with get_db() as conn:
        msgs = conn.execute(
            "SELECT role, content FROM chat_messages WHERE chat_id=? ORDER BY created_at",
            (chat_id,),
        ).fetchall()

    owner_id = chat["user_id"]
    history = []
    for m in msgs:
        try:
            text = decrypt(owner_id, m["content"])
        except Exception:
            text = m["content"]
        history.append({"role": m["role"], "content": text})

    # resolve source filter from topic documents
    source_filter = None
    if topic_id:
        with get_db() as conn:
            docs = conn.execute(
                "SELECT source_name FROM topic_documents WHERE topic_id=?", (topic_id,)
            ).fetchall()
        if docs:
            source_filter = [d["source_name"] for d in docs]

    collection = get_collection(f"class_{class_id}", CHROMA_DIR)

    def event_stream():
        try:
            chunks, messages, hyde_doc = build_rag_context(
                body.question, history, collection,
                cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
                cfg["chat_url"], cfg["llm_model"],
                cfg["cohere_api_key"], cfg["reranker_model"],
                source_filter=source_filter,
            )
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        sources = [
            {"source": meta["source"], "page": meta["page"], "relevance": round(score, 4)}
            for _, _, meta, score in chunks
        ]
        yield f"data: {json.dumps({'type': 'hyde', 'text': hyde_doc})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        full_answer = []
        try:
            for delta in call_llm_stream(messages, cfg["api_key"], cfg["chat_url"], cfg["llm_model"]):
                full_answer.append(delta)
                yield f"data: {json.dumps({'type': 'delta', 'text': delta})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        answer_text = "".join(full_answer)

        if not answer_text.strip():
            yield f"data: {json.dumps({'type': 'error', 'message': 'The model returned an empty response. The context may be too long, or the model hit a rate limit.'})}\n\n"
            return

        # persist both turns
        ts = now()
        user_msg_id = new_id()
        asst_msg_id = new_id()
        with get_db() as conn:
            conn.execute(
                "INSERT INTO chat_messages (id,chat_id,role,content,created_at) VALUES (?,?,?,?,?)",
                (user_msg_id, chat_id, "user", encrypt(owner_id, body.question), ts),
            )
            conn.execute(
                "INSERT INTO chat_messages (id,chat_id,role,content,sources_json,hyde_text,created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (asst_msg_id, chat_id, "assistant",
                 encrypt(owner_id, answer_text),
                 json.dumps(sources), hyde_doc, ts),
            )
            conn.execute(
                "UPDATE chats SET updated_at=?, title=CASE WHEN title='New chat' THEN ? ELSE title END WHERE id=?",
                (ts, body.question[:60], chat_id),
            )

        yield f"data: {json.dumps({'type': 'done', 'message_id': asst_msg_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── teacher/admin: view all chats in a class ─────────────────────────────────

@router.get("/classes/{class_id}/chats")
def list_class_chats(
    class_id: str,
    user_id: Optional[str] = None,
    topic_id: Optional[str] = None,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    with get_db() as conn:
        q = ("SELECT c.id, c.title, c.topic_id, c.user_id, u.username, c.created_at, c.updated_at "
             "FROM chats c JOIN users u ON u.id=c.user_id "
             "WHERE c.class_id=?")
        params = [class_id]
        if user_id:
            q += " AND c.user_id=?"
            params.append(user_id)
        if topic_id:
            q += " AND c.topic_id=?"
            params.append(topic_id)
        q += " ORDER BY c.updated_at DESC"
        rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


@router.get("/classes/{class_id}/chats/{chat_id}/view")
def teacher_view_chat(
    class_id: str,
    chat_id: str,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    """Teacher/admin view of a student chat — full content, decrypted."""
    with get_db() as conn:
        chat = conn.execute(
            "SELECT * FROM chats WHERE id=? AND class_id=?", (chat_id, class_id)
        ).fetchone()
    if not chat:
        raise HTTPException(404, "Chat not found")

    with get_db() as conn:
        msgs = conn.execute(
            "SELECT id,role,content,sources_json,feedback_rating,feedback_comment,created_at "
            "FROM chat_messages WHERE chat_id=? ORDER BY created_at",
            (chat_id,),
        ).fetchall()

    owner_id = chat["user_id"]
    decrypted = []
    for m in msgs:
        row = dict(m)
        try:
            row["content"] = decrypt(owner_id, row["content"])
        except Exception:
            pass
        decrypted.append(row)

    return {**dict(chat), "messages": decrypted}
