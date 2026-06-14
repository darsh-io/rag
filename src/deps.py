"""FastAPI dependency factories for auth and resource-scoped access."""
import os, jwt
from pathlib import Path
from typing import NamedTuple
from datetime import datetime, timezone, timedelta
from fastapi import Header, HTTPException, Path as FPath, Depends
from src.db import get_db

_SECRET_PATH     = Path(__file__).parent.parent / "config" / ".server_secret"
_JWT_SECRET_PATH = Path(__file__).parent.parent / "config" / ".jwt_secret"
_ALGORITHM = "HS256"
_ACCESS_TTL  = timedelta(days=7)
_REFRESH_TTL = timedelta(days=30)


def _load_or_create(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_bytes().hex()
    secret = os.urandom(32)
    path.write_bytes(secret)
    return secret.hex()


def _encryption_secret() -> str:
    return _load_or_create(_SECRET_PATH)


def _jwt_secret() -> str:
    return _load_or_create(_JWT_SECRET_PATH)


class UserInToken(NamedTuple):
    id: str
    username: str
    role: str


def make_token(user_id: str, username: str, role: str) -> str:
    exp = datetime.now(timezone.utc) + _ACCESS_TTL
    return jwt.encode(
        {"sub": user_id, "username": username, "role": role, "type": "access", "exp": exp},
        _jwt_secret(),
        algorithm=_ALGORITHM,
    )


def make_refresh_token(user_id: str) -> str:
    exp = datetime.now(timezone.utc) + _REFRESH_TTL
    return jwt.encode(
        {"sub": user_id, "type": "refresh", "exp": exp},
        _jwt_secret(),
        algorithm=_ALGORITHM,
    )


def verify_refresh_token(token: str) -> str:
    """Decode a refresh token and return the user_id, or raise HTTPException."""
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Refresh token expired")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid refresh token")
    if payload.get("type") != "refresh":
        raise HTTPException(401, "Not a refresh token")
    return payload["sub"]


def get_current_user(authorization: str = Header(default=None)) -> UserInToken:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")
    if payload.get("type") == "refresh":
        raise HTTPException(401, "Use access token, not refresh token")
    return UserInToken(id=payload["sub"], username=payload["username"], role=payload["role"])


def require_role(*roles: str):
    """Dependency factory — raises 403 if the user's role is not in *roles."""
    def dep(user: UserInToken = Depends(get_current_user)) -> UserInToken:
        if user.role not in roles:
            raise HTTPException(403, "Insufficient role")
        return user
    return dep


def class_access(mode: str = "member"):
    """
    Factory — verifies the caller can access a class.
    mode='member'  → student enrolled OR teacher assigned OR supradmin
    mode='teacher' → teacher assigned OR supradmin
    mode='admin'   → supradmin only
    """
    def dep(
        class_id: str = FPath(...),
        user: UserInToken = Depends(get_current_user),
    ) -> dict:
        if user.role == "supradmin":
            with get_db() as conn:
                row = conn.execute(
                    "SELECT id,name FROM classes WHERE id=? AND is_active=1", (class_id,)
                ).fetchone()
            if not row:
                raise HTTPException(404, "Class not found")
            return dict(row)

        if mode == "admin":
            raise HTTPException(403, "Insufficient role")

        with get_db() as conn:
            cls_row = conn.execute(
                "SELECT id,name FROM classes WHERE id=? AND is_active=1", (class_id,)
            ).fetchone()
            if not cls_row:
                raise HTTPException(404, "Class not found")

            if user.role == "teacher":
                ok = conn.execute(
                    "SELECT 1 FROM class_teachers WHERE class_id=? AND user_id=?",
                    (class_id, user.id),
                ).fetchone()
                if not ok:
                    raise HTTPException(403, "Not a teacher in this class")
                return dict(cls_row)

            # student
            if mode == "teacher":
                raise HTTPException(403, "Teachers only")
            ok = conn.execute(
                "SELECT 1 FROM class_students WHERE class_id=? AND user_id=?",
                (class_id, user.id),
            ).fetchone()
            if not ok:
                raise HTTPException(403, "Not enrolled in this class")
            return dict(cls_row)

    return dep


def topic_teacher():
    """Verifies the topic belongs to a class the caller teaches (or is supradmin)."""
    def dep(
        topic_id: str = FPath(...),
        user: UserInToken = Depends(get_current_user),
    ) -> dict:
        with get_db() as conn:
            row = conn.execute(
                "SELECT t.*, c.name AS class_name FROM topics t "
                "JOIN classes c ON c.id=t.class_id "
                "WHERE t.id=?",
                (topic_id,),
            ).fetchone()
        if not row:
            raise HTTPException(404, "Topic not found")
        if user.role == "supradmin":
            return dict(row)
        if user.role != "teacher":
            raise HTTPException(403, "Teachers only")
        with get_db() as conn:
            ok = conn.execute(
                "SELECT 1 FROM class_teachers WHERE class_id=? AND user_id=?",
                (row["class_id"], user.id),
            ).fetchone()
        if not ok:
            raise HTTPException(403, "Not a teacher in this class")
        return dict(row)

    return dep


def own_chat():
    """Verifies the chat belongs to the caller (students) or is in a teacher's class."""
    def dep(
        chat_id: str = FPath(...),
        user: UserInToken = Depends(get_current_user),
    ) -> dict:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Chat not found")
        if user.role == "supradmin":
            return dict(row)
        if user.role == "teacher":
            with get_db() as conn:
                ok = conn.execute(
                    "SELECT 1 FROM class_teachers WHERE class_id=? AND user_id=?",
                    (row["class_id"], user.id),
                ).fetchone()
            if not ok:
                raise HTTPException(403, "Not a teacher in this class")
            return dict(row)
        # student — must own it
        if row["user_id"] != user.id:
            raise HTTPException(403, "Not your chat")
        return dict(row)

    return dep
