"""SQLite layer — schema, connection helpers, seeding."""
import sqlite3, uuid, os
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

DB_PATH = Path(__file__).parent.parent / "config" / "rewise.db"

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    id           TEXT PRIMARY KEY,
    username     TEXT UNIQUE NOT NULL,
    pass_hash    TEXT NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('student','teacher','supradmin')),
    display_name TEXT,
    created_at   TEXT NOT NULL,
    is_active    INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS classes (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    created_by  TEXT NOT NULL REFERENCES users(id),
    created_at  TEXT NOT NULL,
    is_active   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS class_teachers (
    class_id    TEXT NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL REFERENCES users(id)   ON DELETE CASCADE,
    assigned_at TEXT NOT NULL,
    assigned_by TEXT NOT NULL REFERENCES users(id),
    PRIMARY KEY (class_id, user_id)
);

CREATE TABLE IF NOT EXISTS class_students (
    class_id    TEXT NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL REFERENCES users(id)   ON DELETE CASCADE,
    enrolled_at TEXT NOT NULL,
    enrolled_by TEXT NOT NULL REFERENCES users(id),
    PRIMARY KEY (class_id, user_id)
);

CREATE TABLE IF NOT EXISTS topics (
    id          TEXT PRIMARY KEY,
    class_id    TEXT NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    created_by  TEXT NOT NULL REFERENCES users(id),
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS topic_documents (
    id              TEXT PRIMARY KEY,
    topic_id        TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    source_name     TEXT NOT NULL,
    chunks_ingested INTEGER NOT NULL DEFAULT 0,
    uploaded_by     TEXT NOT NULL REFERENCES users(id),
    uploaded_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chats (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id   TEXT REFERENCES topics(id)  ON DELETE SET NULL,
    class_id   TEXT REFERENCES classes(id) ON DELETE SET NULL,
    title      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id               TEXT PRIMARY KEY,
    chat_id          TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content          TEXT NOT NULL,
    sources_json     TEXT,
    hyde_text        TEXT,
    feedback_rating  TEXT CHECK (feedback_rating IN ('up','down',NULL)),
    feedback_comment TEXT,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_by TEXT REFERENCES users(id),
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ct_user  ON class_teachers(user_id);
CREATE INDEX IF NOT EXISTS idx_cs_user  ON class_students(user_id);
CREATE INDEX IF NOT EXISTS idx_top_cls  ON topics(class_id);
CREATE INDEX IF NOT EXISTS idx_doc_top  ON topic_documents(topic_id);
CREATE INDEX IF NOT EXISTS idx_chat_usr ON chats(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_cls ON chats(class_id);
CREATE INDEX IF NOT EXISTS idx_msg_cht  ON chat_messages(chat_id);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db():
    """Yield an open connection; commit on exit, rollback on error."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Run migrations, then seed the supradmin account and default settings."""
    from src.migrations import run_migrations
    run_migrations()
    _seed_supradmin()
    _seed_settings()


def _seed_settings() -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO settings (key,value,updated_at) VALUES (?,?,?)",
            ("self_registration_enabled", "1", now()),
        )


def _seed_supradmin() -> None:
    import bcrypt
    from dotenv import load_dotenv
    load_dotenv()
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "changeme")
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE role='supradmin'"
        ).fetchone()
        if not existing:
            uid = new_id()
            ph  = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                "INSERT INTO users (id,username,pass_hash,role,created_at) VALUES (?,?,?,?,?)",
                (uid, username, ph, "supradmin", now()),
            )
            print(f"[rewise] Supradmin created → {username}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())
