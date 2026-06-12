"""Authentication — login, status."""
import bcrypt
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.db import get_db
from src.deps import make_token, get_current_user, UserInToken

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


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


@router.get("/me")
def me(user: UserInToken = Depends(get_current_user)):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, username, role, display_name FROM users WHERE id=?", (user.id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "User not found")
    return dict(row)
