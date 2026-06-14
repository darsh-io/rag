"""Topic and document endpoint tests."""
import pytest


@pytest.fixture
async def class_and_topic(client, admin_headers):
    cls_r = await client.post("/classes", headers=admin_headers, json={"name": "Topic Tests"})
    class_id = cls_r.json()["id"]
    topic_r = await client.post(
        f"/classes/{class_id}/topics", headers=admin_headers, json={"name": "Ch 1", "description": "Intro"}
    )
    topic_id = topic_r.json()["id"]
    return class_id, topic_id


@pytest.mark.asyncio
async def test_create_topic(client, admin_headers, class_and_topic):
    class_id, topic_id = class_and_topic
    r = await client.get(f"/classes/{class_id}/topics", headers=admin_headers)
    assert r.status_code == 200
    assert any(t["id"] == topic_id for t in r.json())


@pytest.mark.asyncio
async def test_update_topic(client, admin_headers, class_and_topic):
    class_id, topic_id = class_and_topic
    r = await client.patch(
        f"/classes/{class_id}/topics/{topic_id}",
        headers=admin_headers,
        json={"name": "Chapter 1 Updated"},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_list_topic_documents(client, admin_headers, class_and_topic):
    class_id, topic_id = class_and_topic
    r = await client.get(
        f"/classes/{class_id}/topics/{topic_id}/documents", headers=admin_headers
    )
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_delete_topic(client, admin_headers, class_and_topic):
    class_id, topic_id = class_and_topic
    r = await client.delete(
        f"/classes/{class_id}/topics/{topic_id}", headers=admin_headers
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_upload_document_size_limit(client, admin_headers, class_and_topic):
    """Upload endpoint rejects files over the size limit."""
    import io
    import src.routers.documents as docs_mod

    class_id, topic_id = class_and_topic
    original_max = docs_mod._MAX_BYTES
    docs_mod._MAX_BYTES = 5  # 5 bytes — anything realistic will exceed it
    try:
        r = await client.post(
            f"/classes/{class_id}/topics/{topic_id}/documents",
            headers=admin_headers,
            files={"file": ("test.pdf", io.BytesIO(b"%PDF-1.4 fake content here"), "application/pdf")},
        )
        assert r.status_code == 413
    finally:
        docs_mod._MAX_BYTES = original_max
