"""Authentication — login, register, status."""
import bcrypt
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from src.db import get_db, new_id, now
from src.deps import make_token, get_current_user, UserInToken

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    class_id: str
    display_name: Optional[str] = None


@router.post("/login")
def login(body: LoginRequest):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, username, pass_hash, role, display_name "
            "FROM users WHERE username=? AND is_active=1",
            (body.username,),
        ).fetchone()
    if not row:
        raise HTTPException(401, "Invalid credentials")
    if not bcrypt.checkpw(body.password.encode(), row["pass_hash"].encode()):
        raise HTTPException(401, "Invalid credentials")
    token = make_token(row["id"], row["username"], row["role"])
    return {
        "token": token,
        "user": {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "display_name": row["display_name"],
        },
    }


@router.get("/classes")
def public_classes():
    """Public — list active classes for the self-registration form."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, description FROM classes WHERE is_active=1 ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/register")
def register(body: RegisterRequest):
    """Self-registration — creates a student account and enrolls in the chosen class."""
    if not body.username.strip():
        raise HTTPException(400, "Username cannot be empty")
    if len(body.password) < 4:
        raise HTTPException(400, "Password must be at least 4 characters")

    with get_db() as conn:
        cls = conn.execute(
            "SELECT id FROM classes WHERE id=? AND is_active=1", (body.class_id,)
        ).fetchone()
    if not cls:
        raise HTTPException(404, "Class not found")

    ph = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    uid = new_id()
    ts = now()
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (id,username,pass_hash,role,display_name,created_at) VALUES (?,?,?,?,?,?)",
                (uid, body.username.strip(), ph, "student", body.display_name, ts),
            )
    except Exception as e:
        if "UNIQUE" in str(e):
            raise HTTPException(409, "Username already taken")
        raise

    with get_db() as conn:
        conn.execute(
            "INSERT INTO class_students (class_id,user_id,enrolled_at,enrolled_by) VALUES (?,?,?,?)",
            (body.class_id, uid, ts, uid),
        )

    token = make_token(uid, body.username.strip(), "student")
    return {
        "token": token,
        "user": {"id": uid, "username": body.username.strip(), "role": "student", "display_name": body.display_name},
    }


@router.get("/me")
def me(user: UserInToken = Depends(get_current_user)):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, username, role, display_name FROM users WHERE id=?", (user.id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "User not found")
    return dict(row)
