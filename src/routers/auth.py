"""Authentication — login, register, token refresh, status."""
import bcrypt
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional
from src.db import get_db, new_id, now
from src.deps import (
    make_token, make_refresh_token, verify_refresh_token,
    get_current_user, require_role, UserInToken,
)
from src.limiter import limiter

router = APIRouter(prefix="/auth", tags=["auth"])
_admin = require_role("supradmin")


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    class_id: str
    display_name: Optional[str] = None


class RefreshRequest(BaseModel):
    refresh_token: str


class RegistrationSettingBody(BaseModel):
    enabled: bool


@router.post("/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest):
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
    token         = make_token(row["id"], row["username"], row["role"])
    refresh_token = make_refresh_token(row["id"])
    return {
        "token": token,
        "refresh_token": refresh_token,
        "user": {
            "id":           row["id"],
            "username":     row["username"],
            "role":         row["role"],
            "display_name": row["display_name"],
        },
    }


@router.post("/refresh")
@limiter.limit("20/minute")
def refresh(request: Request, body: RefreshRequest):
    user_id = verify_refresh_token(body.refresh_token)
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, username, role FROM users WHERE id=? AND is_active=1", (user_id,)
        ).fetchone()
    if not row:
        raise HTTPException(401, "User not found or deactivated")
    token         = make_token(row["id"], row["username"], row["role"])
    refresh_token = make_refresh_token(row["id"])
    return {"token": token, "refresh_token": refresh_token}


@router.get("/classes")
def public_classes():
    """Public — list active classes for the self-registration form."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, description FROM classes WHERE is_active=1 ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/register")
@limiter.limit("5/minute")
def register(request: Request, body: RegisterRequest):
    """Self-registration — creates a student account and enrolls in the chosen class."""
    # Check whether self-registration is enabled
    with get_db() as conn:
        setting = conn.execute(
            "SELECT value FROM settings WHERE key='self_registration_enabled'"
        ).fetchone()
    if setting and setting["value"] == "0":
        raise HTTPException(403, "Self-registration is currently disabled")

    if not body.username.strip():
        raise HTTPException(400, "Username cannot be empty")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    with get_db() as conn:
        cls = conn.execute(
            "SELECT id FROM classes WHERE id=? AND is_active=1", (body.class_id,)
        ).fetchone()
    if not cls:
        raise HTTPException(404, "Class not found")

    ph  = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
    uid = new_id()
    ts  = now()
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

    token         = make_token(uid, body.username.strip(), "student")
    refresh_token = make_refresh_token(uid)
    return {
        "token": token,
        "refresh_token": refresh_token,
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


@router.patch("/settings/registration")
def set_registration(body: RegistrationSettingBody, caller: UserInToken = Depends(_admin)):
    """Supradmin toggle for self-registration."""
    val = "1" if body.enabled else "0"
    with get_db() as conn:
        conn.execute(
            "INSERT INTO settings (key,value,updated_by,updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_by=excluded.updated_by, updated_at=excluded.updated_at",
            ("self_registration_enabled", val, caller.id, now()),
        )
    return {"self_registration_enabled": body.enabled}
