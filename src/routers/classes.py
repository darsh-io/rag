"""Class CRUD, teacher assignment, student enrollment."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from src.db import get_db, new_id, now
from src.deps import require_role, get_current_user, class_access, UserInToken

router = APIRouter(prefix="/classes", tags=["classes"])
_admin = require_role("supradmin")
_teacher_or_admin = require_role("teacher", "supradmin")


class CreateClass(BaseModel):
    name: str
    description: Optional[str] = None


class AssignBody(BaseModel):
    user_id: str


# ── supradmin CRUD ────────────────────────────────────────────────────────────

@router.get("")
def list_all_classes(caller: UserInToken = Depends(_admin)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,name,description,created_at,is_active FROM classes ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("")
def create_class(body: CreateClass, caller: UserInToken = Depends(_admin)):
    cid = new_id()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO classes (id,name,description,created_by,created_at) VALUES (?,?,?,?,?)",
            (cid, body.name, body.description, caller.id, now()),
        )
    return {"id": cid, "name": body.name}


@router.patch("/{class_id}")
def update_class(class_id: str, body: CreateClass, caller: UserInToken = Depends(_admin)):
    with get_db() as conn:
        conn.execute(
            "UPDATE classes SET name=?, description=? WHERE id=?",
            (body.name, body.description, class_id),
        )
    return {"ok": True}


@router.delete("/{class_id}")
def delete_class(class_id: str, caller: UserInToken = Depends(_admin)):
    with get_db() as conn:
        conn.execute("UPDATE classes SET is_active=0 WHERE id=?", (class_id,))
    return {"ok": True}


@router.patch("/{class_id}/deactivate")
def deactivate_class(class_id: str, caller: UserInToken = Depends(_admin)):
    with get_db() as conn:
        row = conn.execute("SELECT id FROM classes WHERE id=?", (class_id,)).fetchone()
        if not row:
            from fastapi import HTTPException
            raise HTTPException(404, "Class not found")
        conn.execute("UPDATE classes SET is_active=0 WHERE id=?", (class_id,))
    return {"ok": True}


@router.patch("/{class_id}/activate")
def activate_class(class_id: str, caller: UserInToken = Depends(_admin)):
    with get_db() as conn:
        row = conn.execute("SELECT id FROM classes WHERE id=?", (class_id,)).fetchone()
        if not row:
            from fastapi import HTTPException
            raise HTTPException(404, "Class not found")
        conn.execute("UPDATE classes SET is_active=1 WHERE id=?", (class_id,))
    return {"ok": True}


# ── teacher / student assignment ──────────────────────────────────────────────

@router.post("/{class_id}/teachers")
def assign_teacher(class_id: str, body: AssignBody, caller: UserInToken = Depends(_admin)):
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO class_teachers (class_id,user_id,assigned_at,assigned_by) VALUES (?,?,?,?)",
                (class_id, body.user_id, now(), caller.id),
            )
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(409, "Already assigned")
        raise
    return {"ok": True}


@router.delete("/{class_id}/teachers/{user_id}")
def remove_teacher(class_id: str, user_id: str, caller: UserInToken = Depends(_admin)):
    with get_db() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM class_teachers WHERE class_id=?", (class_id,)
        ).fetchone()[0]
        if count <= 1:
            raise HTTPException(409, "Cannot remove the last teacher from a class")
        conn.execute(
            "DELETE FROM class_teachers WHERE class_id=? AND user_id=?", (class_id, user_id)
        )
    return {"ok": True}


@router.get("/{class_id}/teachers")
def list_teachers(class_id: str, caller: UserInToken = Depends(_admin)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT u.id,u.username,u.display_name FROM users u "
            "JOIN class_teachers ct ON ct.user_id=u.id WHERE ct.class_id=?",
            (class_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/{class_id}/students")
def enroll_student(
    class_id: str,
    body: AssignBody,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO class_students (class_id,user_id,enrolled_at,enrolled_by) VALUES (?,?,?,?)",
                (class_id, body.user_id, now(), caller.id),
            )
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(409, "Already enrolled")
        raise
    return {"ok": True}


@router.delete("/{class_id}/students/{user_id}")
def unenroll_student(
    class_id: str,
    user_id: str,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM class_students WHERE class_id=? AND user_id=?", (class_id, user_id)
        )
    return {"ok": True}


@router.get("/{class_id}/students")
def list_students(
    class_id: str,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT u.id,u.username,u.display_name FROM users u "
            "JOIN class_students cs ON cs.user_id=u.id WHERE cs.class_id=?",
            (class_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── /classes/me — returns classes for the current user ───────────────────────
# Must be registered before /{class_id} so "me" isn't treated as a class ID.

@router.get("/me")
def my_classes(caller: UserInToken = Depends(get_current_user)):
    with get_db() as conn:
        if caller.role == "supradmin":
            rows = conn.execute(
                "SELECT id,name,description FROM classes WHERE is_active=1 ORDER BY name"
            ).fetchall()
        elif caller.role == "teacher":
            rows = conn.execute(
                "SELECT c.id,c.name,c.description FROM classes c "
                "JOIN class_teachers ct ON ct.class_id=c.id "
                "WHERE ct.user_id=? AND c.is_active=1 ORDER BY c.name",
                (caller.id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT c.id,c.name,c.description FROM classes c "
                "JOIN class_students cs ON cs.class_id=c.id "
                "WHERE cs.user_id=? AND c.is_active=1 ORDER BY c.name",
                (caller.id,),
            ).fetchall()
    return [dict(r) for r in rows]
