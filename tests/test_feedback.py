"""Feedback endpoint tests."""
import pytest


@pytest.fixture
async def chat_with_message(client, admin_headers):
    """Create a class, a chat, and simulate a message in the DB for feedback testing."""
    from src.db import get_db, new_id, now
    from src.encryption import encrypt

    cls_r = await client.post("/classes", headers=admin_headers, json={"name": "FB Test Class"})
    class_id = cls_r.json()["id"]

    chat_r = await client.post(f"/classes/{class_id}/chats", headers=admin_headers, json={})
    chat_id = chat_r.json()["id"]

    # Get admin user id
    me_r = await client.get("/auth/me", headers=admin_headers)
    user_id = me_r.json()["id"]

    msg_id = new_id()
    ts = now()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id,chat_id,role,content,created_at) VALUES (?,?,?,?,?)",
            (msg_id, chat_id, "assistant", encrypt(user_id, "Test answer"), ts),
        )

    return class_id, chat_id, msg_id


@pytest.mark.asyncio
async def test_submit_feedback_up(client, admin_headers, chat_with_message):
    class_id, chat_id, msg_id = chat_with_message
    r = await client.post(
        f"/chats/{chat_id}/messages/{msg_id}/feedback",
        headers=admin_headers,
        json={"rating": "up", "comment": "Great answer!"},
    )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_submit_feedback_invalid_rating(client, admin_headers, chat_with_message):
    class_id, chat_id, msg_id = chat_with_message
    r = await client.post(
        f"/chats/{chat_id}/messages/{msg_id}/feedback",
        headers=admin_headers,
        json={"rating": "meh"},
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_list_feedback_paginated(client, admin_headers, chat_with_message):
    class_id, chat_id, msg_id = chat_with_message
    await client.post(
        f"/chats/{chat_id}/messages/{msg_id}/feedback",
        headers=admin_headers,
        json={"rating": "down"},
    )
    r = await client.get(f"/classes/{class_id}/feedback", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert "total" in body
