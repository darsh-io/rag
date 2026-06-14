"""Versioned SQLite migration runner."""
from src.db import get_db

_MIGRATIONS = [
    # v1 — baseline schema (matches _SCHEMA in db.py) + status/error_message on topic_documents
    """
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
        uploaded_at     TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'ready',
        error_message   TEXT
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

    CREATE TABLE IF NOT EXISTS schema_migrations (
        version    INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_ct_user  ON class_teachers(user_id);
    CREATE INDEX IF NOT EXISTS idx_cs_user  ON class_students(user_id);
    CREATE INDEX IF NOT EXISTS idx_top_cls  ON topics(class_id);
    CREATE INDEX IF NOT EXISTS idx_doc_top  ON topic_documents(topic_id);
    CREATE INDEX IF NOT EXISTS idx_chat_usr ON chats(user_id);
    CREATE INDEX IF NOT EXISTS idx_chat_cls ON chats(class_id);
    CREATE INDEX IF NOT EXISTS idx_msg_cht  ON chat_messages(chat_id);
    """,
]


def _add_column_if_missing(conn, table: str, column: str, definition: str) -> None:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def run_migrations() -> None:
    """Apply any unapplied migrations in order."""
    with get_db() as conn:
        # Bootstrap the migrations table before running scripts that create it
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version    INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)
        applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()}

    for version, sql in enumerate(_MIGRATIONS, start=1):
        if version in applied:
            continue
        with get_db() as conn:
            conn.executescript(sql)
            # Patch existing DBs that predate the status/error_message columns
            _add_column_if_missing(conn, "topic_documents", "status",        "TEXT NOT NULL DEFAULT 'ready'")
            _add_column_if_missing(conn, "topic_documents", "error_message", "TEXT")
            from src.db import now
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version,applied_at) VALUES (?,?)",
                (version, now()),
            )
