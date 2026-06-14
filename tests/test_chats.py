"""Chat CRUD tests (no streaming — that requires a live LLM)."""
import pytest


@pytest.fixture
async def setup_class_and_chat(client, admin_headers):
    cls_r = await client.post("/classes", headers=admin_headers, json={"name": "Chat Test Class"})
    class_id = cls_r.json()["id"]

    topic_r = await client.post(
        f"/classes/{class_id}/topics", headers=admin_headers, json={"name": "Topic A"}
    )
    topic_id = topic_r.json()["id"]

    chat_r = await client.post(
        f"/classes/{class_id}/chats", headers=admin_headers,
        json={"topic_id": topic_id, "title": "My chat"},
    )
    chat_id = chat_r.json()["id"]
    return class_id, topic_id, chat_id


@pytest.mark.asyncio
async def test_create_and_get_chat(client, admin_headers, setup_class_and_chat):
    class_id, topic_id, chat_id = setup_class_and_chat
    r = await client.get(f"/chats/{chat_id}", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == chat_id
    assert body["messages"] == []


@pytest.mark.asyncio
async def test_list_my_chats(client, admin_headers, setup_class_and_chat):
    class_id, topic_id, chat_id = setup_class_and_chat
    r = await client.get(f"/classes/{class_id}/chats/me", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert any(c["id"] == chat_id for c in body["items"])


@pytest.mark.asyncio
async def test_list_class_chats_teacher(client, admin_headers, setup_class_and_chat):
    class_id, topic_id, chat_id = setup_class_and_chat
    r = await client.get(f"/classes/{class_id}/chats", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert "items" in body


@pytest.mark.asyncio
async def test_delete_chat(client, admin_headers, setup_class_and_chat):
    class_id, _, chat_id = setup_class_and_chat
    r = await client.delete(f"/chats/{chat_id}", headers=admin_headers)
    assert r.status_code == 200

    r2 = await client.get(f"/chats/{chat_id}", headers=admin_headers)
    assert r2.status_code == 404


@pytest.mark.asyncio
async def test_teacher_view_chat(client, admin_headers, setup_class_and_chat):
    class_id, _, chat_id = setup_class_and_chat
    r = await client.get(f"/classes/{class_id}/chats/{chat_id}/view", headers=admin_headers)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_query_stream_empty_question(client, admin_headers, setup_class_and_chat):
    _, _, chat_id = setup_class_and_chat
    r = await client.post(f"/chats/{chat_id}/query/stream", headers=admin_headers,
                          json={"question": "  "})
    assert r.status_code == 400
