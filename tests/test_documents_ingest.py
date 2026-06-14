"""Document ingest background task tests."""
import os
import pytest
from unittest.mock import patch, MagicMock
from src.db import get_db, new_id, now


@pytest.fixture
def doc_record(admin_headers):
    """Insert a topic_documents row in 'processing' state and return doc_id."""
    from tests.conftest import _tmp_dir
    doc_id = new_id()
    # We need a topic to satisfy FK; create a minimal one inline
    with get_db() as conn:
        # Get any existing user/class/topic or skip
        topic = conn.execute("SELECT id FROM topics LIMIT 1").fetchone()
        user  = conn.execute("SELECT id FROM users WHERE role='supradmin' LIMIT 1").fetchone()
        if not topic or not user:
            return None
        conn.execute(
            "INSERT INTO topic_documents "
            "(id,topic_id,filename,source_name,chunks_ingested,uploaded_by,uploaded_at,status) "
            "VALUES (?,?,?,?,0,?,'now','processing')",
            (doc_id, topic["id"], "test.pdf", "test_topic12345678", user["id"]),
        )
    return doc_id


def test_do_ingest_success(tmp_path):
    """_do_ingest marks the record ready on success."""
    from src.routers.documents import _do_ingest
    from src.db import get_db, new_id, now

    # Ensure we have a topic and user
    with get_db() as conn:
        topic = conn.execute("SELECT id FROM topics LIMIT 1").fetchone()
        user  = conn.execute("SELECT id FROM users WHERE role='supradmin' LIMIT 1").fetchone()
    if not topic or not user:
        pytest.skip("No topic/user in DB")

    doc_id = new_id()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO topic_documents "
            "(id,topic_id,filename,source_name,chunks_ingested,uploaded_by,uploaded_at,status) "
            "VALUES (?,?,?,?,0,?,?,'processing')",
            (doc_id, topic["id"], "test.pdf", "test_topicABCD1234", user["id"], now()),
        )

    tmp_file = tmp_path / "fake.pdf"
    tmp_file.write_bytes(b"%PDF fake")

    with patch("src.routers.documents.get_collection"), \
         patch("src.routers.documents.ingest", return_value=5) as mock_ingest, \
         patch("src.routers.documents.invalidate_bm25_cache"):
        _do_ingest(doc_id, str(tmp_file), "class123", "test_topicABCD1234", {})

    with get_db() as conn:
        row = conn.execute("SELECT status, chunks_ingested FROM topic_documents WHERE id=?", (doc_id,)).fetchone()
    assert row["status"] == "ready"
    assert row["chunks_ingested"] == 5


def test_do_ingest_error():
    """_do_ingest marks the record error on exception."""
    from src.routers.documents import _do_ingest
    from src.db import get_db, new_id, now

    with get_db() as conn:
        topic = conn.execute("SELECT id FROM topics LIMIT 1").fetchone()
        user  = conn.execute("SELECT id FROM users WHERE role='supradmin' LIMIT 1").fetchone()
    if not topic or not user:
        pytest.skip("No topic/user in DB")

    doc_id = new_id()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO topic_documents "
            "(id,topic_id,filename,source_name,chunks_ingested,uploaded_by,uploaded_at,status) "
            "VALUES (?,?,?,?,0,?,?,'processing')",
            (doc_id, topic["id"], "fail.pdf", "test_topicFAIL1234", user["id"], now()),
        )

    with patch("src.routers.documents.get_collection"), \
         patch("src.routers.documents.ingest", side_effect=RuntimeError("embed failed")):
        _do_ingest(doc_id, "/nonexistent/path.pdf", "classX", "test_topicFAIL1234", {})

    with get_db() as conn:
        row = conn.execute("SELECT status, error_message FROM topic_documents WHERE id=?", (doc_id,)).fetchone()
    assert row["status"] == "error"
    assert "embed failed" in row["error_message"]
