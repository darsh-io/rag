"""Chat CRUD + streaming RAG query."""
import asyncio, json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from src.db import get_db, new_id, now
from src.deps import get_current_user, require_role, class_access, own_chat, UserInToken
from src.encryption import encrypt, decrypt
from src.config import cfg, CHROMA_DIR
from src.logger import get_logger
from src.ragPipeline.vectorstore import get_collection
from src.query import build_rag_context, call_llm_stream

log = get_logger("rewise.chats")

router = APIRouter(tags=["chats"])
_teacher_or_admin = require_role("teacher", "supradmin")

_SENTINEL = object()


class CreateChat(BaseModel):
    topic_id: Optional[str] = None
    title: Optional[str] = None


class QueryRequest(BaseModel):
    question: str


# ── student/self chat endpoints ───────────────────────────────────────────────

@router.get("/classes/{class_id}/chats/me")
async def list_my_chats(
    class_id: str,
    topic_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    caller: UserInToken = Depends(get_current_user),
    cls=Depends(class_access("member")),
):
    def _query():
        with get_db() as conn:
            base = "FROM chats WHERE user_id=? AND class_id=?"
            params_count = [caller.id, class_id]
            params_rows  = [caller.id, class_id]
            if topic_id:
                base += " AND topic_id=?"
                params_count.append(topic_id)
                params_rows.append(topic_id)
            total = conn.execute(f"SELECT COUNT(*) {base}", params_count).fetchone()[0]
            rows  = conn.execute(
                f"SELECT id,title,topic_id,created_at,updated_at {base} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                params_rows + [limit, offset],
            ).fetchall()
            return total, rows
    total, rows = await asyncio.to_thread(_query)
    return {"items": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}


@router.post("/classes/{class_id}/chats")
async def create_chat(
    class_id: str,
    body: CreateChat,
    caller: UserInToken = Depends(get_current_user),
    cls=Depends(class_access("member")),
):
    chat_id = new_id()
    ts = now()
    def _insert():
        with get_db() as conn:
            conn.execute(
                "INSERT INTO chats (id,user_id,topic_id,class_id,title,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (chat_id, caller.id, body.topic_id, class_id, body.title or "New chat", ts, ts),
            )
    await asyncio.to_thread(_insert)
    return {"id": chat_id}


@router.get("/chats/{chat_id}")
async def get_chat(
    chat_id: str,
    caller: UserInToken = Depends(get_current_user),
    chat=Depends(own_chat()),
):
    def _query():
        with get_db() as conn:
            return conn.execute(
                "SELECT id,role,content,sources_json,feedback_rating,feedback_comment,created_at "
                "FROM chat_messages WHERE chat_id=? ORDER BY created_at",
                (chat_id,),
            ).fetchall()
    msgs = await asyncio.to_thread(_query)

    owner_id = chat["user_id"]
    decrypted = []
    for m in msgs:
        row = dict(m)
        try:
            row["content"] = decrypt(owner_id, row["content"])
        except Exception:
            pass
        decrypted.append(row)

    return {**chat, "messages": decrypted}


@router.delete("/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    caller: UserInToken = Depends(get_current_user),
    chat=Depends(own_chat()),
):
    if caller.role == "student" and chat["user_id"] != caller.id:
        raise HTTPException(403, "Not your chat")
    def _delete():
        with get_db() as conn:
            conn.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    await asyncio.to_thread(_delete)
    return {"ok": True}


@router.post("/chats/{chat_id}/query/stream")
async def query_stream(
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
    owner_id = chat["user_id"]

    def _load_history():
        with get_db() as conn:
            return conn.execute(
                "SELECT role, content FROM chat_messages WHERE chat_id=? ORDER BY created_at",
                (chat_id,),
            ).fetchall()

    def _load_source_filter():
        if not topic_id:
            return None
        with get_db() as conn:
            docs = conn.execute(
                "SELECT source_name FROM topic_documents WHERE topic_id=? AND status='ready'",
                (topic_id,),
            ).fetchall()
        return [d["source_name"] for d in docs] if docs else None

    msgs_raw, source_filter = await asyncio.gather(
        asyncio.to_thread(_load_history),
        asyncio.to_thread(_load_source_filter),
    )

    history = []
    for m in msgs_raw:
        try:
            text = decrypt(owner_id, m["content"])
        except Exception:
            text = m["content"]
        history.append({"role": m["role"], "content": text})

    collection = get_collection(f"class_{class_id}", CHROMA_DIR)

    async def event_stream():
        log.info("stream_start", extra={"chat_id": chat_id, "question": body.question[:80]})

        # RAG context (blocking) in a thread
        try:
            log.info("rag_context_start", extra={"chat_id": chat_id, "source_filter": source_filter})
            chunks, messages, hyde_doc = await asyncio.to_thread(
                build_rag_context,
                body.question, history, collection,
                cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
                cfg["chat_url"], cfg["llm_model"],
                cfg["cohere_api_key"], cfg["reranker_model"],
                source_filter,
            )
            log.info("rag_context_done", extra={"chat_id": chat_id, "chunks": len(chunks)})
        except ValueError as e:
            log.warning("rag_context_value_error", extra={"chat_id": chat_id, "error": str(e)})
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return
        except Exception as e:
            log.error("rag_context_error", extra={"chat_id": chat_id, "error": str(e)}, exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        sources = [
            {"source": meta["source"], "page": meta["page"], "relevance": round(score, 4)}
            for _, _, meta, score in chunks
        ]
        yield f"data: {json.dumps({'type': 'hyde', 'text': hyde_doc})}\n\n"
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # LLM streaming — capture the running loop HERE (async context), close over it in _stream
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()  # safe: we are in a coroutine

        def _stream():
            log.info("llm_stream_thread_start", extra={"chat_id": chat_id})
            try:
                delta_count = 0
                for delta in call_llm_stream(messages, cfg["api_key"], cfg["chat_url"], cfg["llm_model"]):
                    loop.call_soon_threadsafe(queue.put_nowait, delta)
                    delta_count += 1
                log.info("llm_stream_thread_done", extra={"chat_id": chat_id, "deltas": delta_count})
            except Exception as e:
                log.error("llm_stream_thread_error", extra={"chat_id": chat_id, "error": str(e)}, exc_info=True)
                loop.call_soon_threadsafe(queue.put_nowait, ("__error__", str(e)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        stream_task = loop.run_in_executor(None, _stream)
        log.info("llm_stream_executor_started", extra={"chat_id": chat_id})

        full_answer = []
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                log.info("llm_stream_sentinel", extra={"chat_id": chat_id, "answer_chars": sum(len(t) for t in full_answer)})
                break
            if isinstance(item, tuple) and item[0] == "__error__":
                log.error("llm_stream_queue_error", extra={"chat_id": chat_id, "error": item[1]})
                yield f"data: {json.dumps({'type': 'error', 'message': item[1]})}\n\n"
                await stream_task
                return
            full_answer.append(item)
            yield f"data: {json.dumps({'type': 'delta', 'text': item})}\n\n"

        await stream_task
        answer_text = "".join(full_answer)

        if not answer_text.strip():
            log.warning("llm_stream_empty_response", extra={"chat_id": chat_id})
            yield f"data: {json.dumps({'type': 'error', 'message': 'The model returned an empty response. The context may be too long, or the model hit a rate limit.'})}\n\n"
            return

        # persist both turns
        ts = now()
        user_msg_id = new_id()
        asst_msg_id = new_id()

        def _persist():
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

        await asyncio.to_thread(_persist)
        log.info("stream_done", extra={"chat_id": chat_id, "message_id": asst_msg_id})
        yield f"data: {json.dumps({'type': 'done', 'message_id': asst_msg_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── teacher/admin: view all chats in a class ─────────────────────────────────

@router.get("/classes/{class_id}/chats")
async def list_class_chats(
    class_id: str,
    user_id: Optional[str] = None,
    topic_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    def _query():
        with get_db() as conn:
            where = "WHERE c.class_id=?"
            params = [class_id]
            if user_id:
                where += " AND c.user_id=?"
                params.append(user_id)
            if topic_id:
                where += " AND c.topic_id=?"
                params.append(topic_id)
            total = conn.execute(
                f"SELECT COUNT(*) FROM chats c JOIN users u ON u.id=c.user_id {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT c.id, c.title, c.topic_id, c.user_id, u.username, c.created_at, c.updated_at "
                f"FROM chats c JOIN users u ON u.id=c.user_id {where} "
                f"ORDER BY c.updated_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            return total, rows
    total, rows = await asyncio.to_thread(_query)
    return {"items": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}


@router.get("/classes/{class_id}/chats/{chat_id}/view")
async def teacher_view_chat(
    class_id: str,
    chat_id: str,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    """Teacher/admin view of a student chat — full content, decrypted."""
    def _query():
        with get_db() as conn:
            chat = conn.execute(
                "SELECT * FROM chats WHERE id=? AND class_id=?", (chat_id, class_id)
            ).fetchone()
            if not chat:
                return None, []
            msgs = conn.execute(
                "SELECT id,role,content,sources_json,feedback_rating,feedback_comment,created_at "
                "FROM chat_messages WHERE chat_id=? ORDER BY created_at",
                (chat_id,),
            ).fetchall()
            return dict(chat), [dict(m) for m in msgs]

    chat, msgs = await asyncio.to_thread(_query)
    if not chat:
        raise HTTPException(404, "Chat not found")

    owner_id = chat["user_id"]
    decrypted = []
    for row in msgs:
        try:
            row["content"] = decrypt(owner_id, row["content"])
        except Exception:
            pass
        decrypted.append(row)

    return {**chat, "messages": decrypted}
