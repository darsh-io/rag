"""JWT / auth dependency tests."""
import pytest
import jwt as pyjwt
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException
from src.deps import (
    make_token, make_refresh_token, verify_refresh_token, _jwt_secret,
    get_current_user, require_role, class_access, topic_teacher, own_chat,
    _encryption_secret, UserInToken,
)


def test_make_token_decodable():
    token = make_token("uid-1", "alice", "teacher")
    payload = pyjwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    assert payload["sub"] == "uid-1"
    assert payload["role"] == "teacher"
    assert payload["type"] == "access"


def test_access_token_has_expiry():
    token = make_token("uid-1", "alice", "teacher")
    payload = pyjwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    assert "exp" in payload


def test_refresh_token_type():
    token = make_refresh_token("uid-2")
    payload = pyjwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    assert payload["type"] == "refresh"
    assert payload["sub"] == "uid-2"


def test_verify_refresh_token_ok():
    token = make_refresh_token("uid-3")
    user_id = verify_refresh_token(token)
    assert user_id == "uid-3"


def test_verify_refresh_token_rejects_access_token():
    token = make_token("uid-4", "bob", "student")
    with pytest.raises(HTTPException) as exc:
        verify_refresh_token(token)
    assert exc.value.status_code == 401


def test_verify_refresh_token_rejects_garbage():
    with pytest.raises(HTTPException) as exc:
        verify_refresh_token("not.a.token")
    assert exc.value.status_code == 401


def test_verify_refresh_token_expired():
    expired = pyjwt.encode(
        {"sub": "uid-x", "type": "refresh", "exp": datetime(2000, 1, 1, tzinfo=timezone.utc)},
        _jwt_secret(),
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        verify_refresh_token(expired)
    assert exc.value.status_code == 401


def test_encryption_secret_returns_hex():
    secret = _encryption_secret()
    assert isinstance(secret, str)
    assert len(secret) > 0


def test_get_current_user_no_header():
    with pytest.raises(HTTPException) as exc:
        get_current_user(authorization=None)
    assert exc.value.status_code == 401


def test_get_current_user_malformed_header():
    with pytest.raises(HTTPException) as exc:
        get_current_user(authorization="Token abc")
    assert exc.value.status_code == 401


def test_get_current_user_expired_token():
    expired = pyjwt.encode(
        {"sub": "uid-x", "username": "alice", "role": "teacher", "type": "access",
         "exp": datetime(2000, 1, 1, tzinfo=timezone.utc)},
        _jwt_secret(),
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        get_current_user(authorization=f"Bearer {expired}")
    assert exc.value.status_code == 401


def test_get_current_user_invalid_token():
    with pytest.raises(HTTPException) as exc:
        get_current_user(authorization="Bearer not.valid.jwt")
    assert exc.value.status_code == 401


def test_get_current_user_rejects_refresh_token():
    token = make_refresh_token("uid-5")
    with pytest.raises(HTTPException) as exc:
        get_current_user(authorization=f"Bearer {token}")
    assert exc.value.status_code == 401


def test_get_current_user_ok():
    token = make_token("uid-6", "carol", "teacher")
    user = get_current_user(authorization=f"Bearer {token}")
    assert user.id == "uid-6"
    assert user.role == "teacher"


def test_require_role_passes():
    dep_fn = require_role("teacher", "supradmin")
    inner = dep_fn
    user = UserInToken(id="u1", username="t", role="teacher")
    result = inner(user=user)
    assert result.role == "teacher"


def test_require_role_rejects():
    dep_fn = require_role("supradmin")
    user = UserInToken(id="u2", username="t", role="student")
    with pytest.raises(HTTPException) as exc:
        dep_fn(user=user)
    assert exc.value.status_code == 403


def test_class_access_admin_mode_rejects_non_supradmin():
    dep_fn = class_access("admin")
    user = UserInToken(id="u3", username="t", role="teacher")
    with pytest.raises(HTTPException) as exc:
        dep_fn(class_id="some-class", user=user)
    assert exc.value.status_code == 403


def test_class_access_supradmin_class_not_found():
    dep_fn = class_access("member")
    user = UserInToken(id="u4", username="sa", role="supradmin")
    with pytest.raises(HTTPException) as exc:
        dep_fn(class_id="nonexistent-class-id", user=user)
    assert exc.value.status_code == 404


def test_class_access_teacher_not_assigned():
    dep_fn = class_access("member")
    user = UserInToken(id="u5", username="t", role="teacher")
    with pytest.raises(HTTPException) as exc:
        dep_fn(class_id="nonexistent-class-id", user=user)
    assert exc.value.status_code == 404


def test_class_access_student_teacher_mode():
    dep_fn = class_access("teacher")
    user = UserInToken(id="u6", username="s", role="student")
    with pytest.raises(HTTPException) as exc:
        dep_fn(class_id="nonexistent-class-id", user=user)
    assert exc.value.status_code == 404


def test_topic_teacher_not_found():
    dep_fn = topic_teacher()
    user = UserInToken(id="u7", username="t", role="teacher")
    with pytest.raises(HTTPException) as exc:
        dep_fn(topic_id="nonexistent-topic", user=user)
    assert exc.value.status_code == 404


def test_topic_teacher_student_rejected():
    dep_fn = topic_teacher()
    user = UserInToken(id="u8", username="s", role="student")
    with pytest.raises(HTTPException) as exc:
        dep_fn(topic_id="nonexistent-topic", user=user)
    assert exc.value.status_code == 404


def test_own_chat_not_found():
    dep_fn = own_chat()
    user = UserInToken(id="u9", username="s", role="student")
    with pytest.raises(HTTPException) as exc:
        dep_fn(chat_id="nonexistent-chat", user=user)
    assert exc.value.status_code == 404


def test_load_or_create_new_file(tmp_path):
    from src.deps import _load_or_create
    new_path = tmp_path / "subdir" / "test_secret"
    assert not new_path.exists()
    secret = _load_or_create(new_path)
    assert new_path.exists()
    assert isinstance(secret, str)
    assert len(secret) == 64  # 32 random bytes → 64 hex chars


def test_load_or_create_existing_file(tmp_path):
    from src.deps import _load_or_create
    p = tmp_path / "existing_secret"
    p.write_bytes(bytes.fromhex("ab" * 32))
    secret = _load_or_create(p)
    assert secret == "ab" * 32


def _admin_id():
    from src.db import get_db
    with get_db() as conn:
        row = conn.execute("SELECT id FROM users WHERE role='supradmin' LIMIT 1").fetchone()
    return row["id"]


def _make_class(name):
    from src.db import get_db, new_id, now
    class_id = new_id()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO classes (id, name, is_active, created_by, created_at) VALUES (?, ?, 1, ?, ?)",
            (class_id, name, _admin_id(), now()),
        )
    return class_id


def test_class_access_teacher_not_in_class(setup_db):
    """Teacher trying to access a class they're not assigned to → 403."""
    class_id = _make_class("Dep Test Class T1")
    dep_fn = class_access("member")
    user = UserInToken(id="nonexistent-teacher-id", username="t", role="teacher")
    with pytest.raises(HTTPException) as exc:
        dep_fn(class_id=class_id, user=user)
    assert exc.value.status_code == 403


def test_class_access_student_not_enrolled(setup_db):
    """Student trying to access a class they're not enrolled in → 403."""
    class_id = _make_class("Dep Test Class S1")
    dep_fn = class_access("member")
    user = UserInToken(id="nonexistent-student-id", username="s", role="student")
    with pytest.raises(HTTPException) as exc:
        dep_fn(class_id=class_id, user=user)
    assert exc.value.status_code == 403


def test_class_access_student_teacher_mode_raises(setup_db):
    """Student accessing teacher-only endpoint → 403."""
    class_id = _make_class("Dep Test Class S2")
    dep_fn = class_access("teacher")
    user = UserInToken(id="nonexistent-student-id", username="s", role="student")
    with pytest.raises(HTTPException) as exc:
        dep_fn(class_id=class_id, user=user)
    assert exc.value.status_code == 403


def _make_user(role):
    import bcrypt
    from src.db import get_db, new_id, now
    import uuid
    user_id = new_id()
    ph = bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (id, username, pass_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, f"dep_test_{uuid.uuid4().hex[:8]}", ph, role, now()),
        )
    return user_id


def test_class_access_teacher_in_class_ok(setup_db):
    """Teacher assigned to class passes class_access."""
    from src.db import get_db, now
    class_id = _make_class("Dep Test Class T2")
    user_id = _make_user("teacher")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO class_teachers (class_id, user_id, assigned_by, assigned_at) VALUES (?, ?, ?, ?)",
            (class_id, user_id, _admin_id(), now()),
        )
    dep_fn = class_access("member")
    user = UserInToken(id=user_id, username="t", role="teacher")
    result = dep_fn(class_id=class_id, user=user)
    assert result["id"] == class_id


def test_class_access_student_enrolled_ok(setup_db):
    """Student enrolled in class passes class_access."""
    from src.db import get_db, now
    class_id = _make_class("Dep Test Class S3")
    user_id = _make_user("student")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO class_students (class_id, user_id, enrolled_by, enrolled_at) VALUES (?, ?, ?, ?)",
            (class_id, user_id, _admin_id(), now()),
        )
    dep_fn = class_access("member")
    user = UserInToken(id=user_id, username="s", role="student")
    result = dep_fn(class_id=class_id, user=user)
    assert result["id"] == class_id
