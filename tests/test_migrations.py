"""Migration system tests."""
import pytest
from src.db import get_db


def test_schema_migrations_table_exists():
    with get_db() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        ).fetchone()
    assert row is not None


def test_migration_v1_applied():
    with get_db() as conn:
        row = conn.execute("SELECT version FROM schema_migrations WHERE version=1").fetchone()
    assert row is not None


def test_topic_documents_has_status_column():
    with get_db() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(topic_documents)").fetchall()]
    assert "status" in cols
    assert "error_message" in cols


def test_settings_table_exists():
    with get_db() as conn:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='settings'").fetchone()
    assert row is not None


def test_self_registration_seeded():
    with get_db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key='self_registration_enabled'").fetchone()
    assert row is not None
    assert row["value"] in ("0", "1")
