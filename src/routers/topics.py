"""Topic CRUD — scoped to a class."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.db import get_db, new_id, now
from src.deps import require_role, class_access, get_current_user, UserInToken

router = APIRouter(prefix="/classes/{class_id}/topics", tags=["topics"])
_teacher_or_admin = require_role("teacher", "supradmin")


class TopicBody(BaseModel):
    name: str
    description: Optional[str] = None


@router.get("")
def list_topics(
    class_id: str,
    caller: UserInToken = Depends(get_current_user),
    cls=Depends(class_access("member")),
):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT t.id, t.name, t.description, t.created_at, "
            "COUNT(d.id) AS doc_count "
            "FROM topics t LEFT JOIN topic_documents d ON d.topic_id=t.id "
            "WHERE t.class_id=? GROUP BY t.id ORDER BY t.name",
            (class_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("")
def create_topic(
    class_id: str,
    body: TopicBody,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    tid = new_id()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO topics (id,class_id,name,description,created_by,created_at) VALUES (?,?,?,?,?,?)",
            (tid, class_id, body.name, body.description, caller.id, now()),
        )
    return {"id": tid, "name": body.name}


@router.patch("/{topic_id}")
def update_topic(
    class_id: str,
    topic_id: str,
    body: TopicBody,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    with get_db() as conn:
        conn.execute(
            "UPDATE topics SET name=?, description=? WHERE id=? AND class_id=?",
            (body.name, body.description, topic_id, class_id),
        )
    return {"ok": True}


@router.delete("/{topic_id}")
def delete_topic(
    class_id: str,
    topic_id: str,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM topics WHERE id=? AND class_id=?", (topic_id, class_id)
        )
    return {"ok": True}


@router.get("/{topic_id}/documents")
def list_topic_documents(
    class_id: str,
    topic_id: str,
    caller: UserInToken = Depends(get_current_user),
    cls=Depends(class_access("member")),
):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,filename,source_name,chunks_ingested,uploaded_at,status,error_message "
            "FROM topic_documents WHERE topic_id=? ORDER BY uploaded_at",
            (topic_id,),
        ).fetchall()
    return [dict(r) for r in rows]
