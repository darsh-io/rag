"""User management — supradmin only (create, list, deactivate, password reset)."""
import bcrypt
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from src.db import get_db, new_id, now
from src.deps import require_role, UserInToken

router = APIRouter(prefix="/users", tags=["users"])
_admin = require_role("supradmin")


class CreateUser(BaseModel):
    username: str
    password: str
    role: str
    display_name: Optional[str] = None


class UpdatePassword(BaseModel):
    password: str


@router.get("")
def list_users(user: UserInToken = Depends(_admin)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, username, role, display_name, created_at, is_active FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("")
def create_user(body: CreateUser, user: UserInToken = Depends(_admin)):
    if body.role not in ("student", "teacher", "supradmin"):
        raise HTTPException(400, "Invalid role")
    ph = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    uid = new_id()
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (id,username,pass_hash,role,display_name,created_at) VALUES (?,?,?,?,?,?)",
                (uid, body.username, ph, body.role, body.display_name, now()),
            )
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(409, "Username already taken")
        raise
    return {"id": uid, "username": body.username, "role": body.role}


@router.patch("/{user_id}/deactivate")
def deactivate_user(user_id: str, caller: UserInToken = Depends(_admin)):
    if user_id == caller.id:
        raise HTTPException(400, "Cannot deactivate yourself")
    with get_db() as conn:
        conn.execute("UPDATE users SET is_active=0 WHERE id=?", (user_id,))
    return {"ok": True}


@router.patch("/{user_id}/activate")
def activate_user(user_id: str, caller: UserInToken = Depends(_admin)):
    with get_db() as conn:
        conn.execute("UPDATE users SET is_active=1 WHERE id=?", (user_id,))
    return {"ok": True}


@router.patch("/{user_id}/password")
def reset_password(user_id: str, body: UpdatePassword, caller: UserInToken = Depends(_admin)):
    ph = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    with get_db() as conn:
        conn.execute("UPDATE users SET pass_hash=? WHERE id=?", (ph, user_id))
    return {"ok": True}
