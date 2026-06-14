"""Feedback — submit rating on a message; teacher/admin can view per class."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.db import get_db, now
from src.deps import get_current_user, require_role, class_access, UserInToken

router = APIRouter(tags=["feedback"])
_teacher_or_admin = require_role("teacher", "supradmin")


class FeedbackBody(BaseModel):
    rating: str  # "up" | "down"
    comment: Optional[str] = None


@router.post("/chats/{chat_id}/messages/{msg_id}/feedback")
def submit_feedback(
    chat_id: str,
    msg_id: str,
    body: FeedbackBody,
    caller: UserInToken = Depends(get_current_user),
):
    if body.rating not in ("up", "down"):
        raise HTTPException(400, "Rating must be 'up' or 'down'")

    with get_db() as conn:
        chat = conn.execute(
            "SELECT user_id FROM chats WHERE id=?", (chat_id,)
        ).fetchone()
    if not chat:
        raise HTTPException(404, "Chat not found")
    if caller.role == "student" and chat["user_id"] != caller.id:
        raise HTTPException(403, "Not your chat")

    with get_db() as conn:
        msg = conn.execute(
            "SELECT id FROM chat_messages WHERE id=? AND chat_id=? AND role='assistant'",
            (msg_id, chat_id),
        ).fetchone()
    if not msg:
        raise HTTPException(404, "Message not found")

    with get_db() as conn:
        conn.execute(
            "UPDATE chat_messages SET feedback_rating=?, feedback_comment=? WHERE id=?",
            (body.rating, body.comment, msg_id),
        )
    return {"ok": True}


@router.get("/classes/{class_id}/feedback")
def list_class_feedback(
    class_id: str,
    rating: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    with get_db() as conn:
        base = (
            "FROM chat_messages m "
            "JOIN chats c ON c.id=m.chat_id "
            "JOIN users u ON u.id=c.user_id "
            "WHERE c.class_id=? AND m.feedback_rating IS NOT NULL"
        )
        params = [class_id]
        if rating:
            base += " AND m.feedback_rating=?"
            params.append(rating)
        if user_id:
            base += " AND c.user_id=?"
            params.append(user_id)
        total = conn.execute(f"SELECT COUNT(*) {base}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT m.id, m.feedback_rating, m.feedback_comment, m.created_at, "
            f"c.id AS chat_id, c.title AS chat_title, c.user_id, u.username, c.topic_id "
            f"{base} ORDER BY m.created_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
    return {"items": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}
