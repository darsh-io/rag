"""Class and topic CRUD tests."""
import pytest


@pytest.fixture
async def class_id(client, admin_headers):
    r = await client.post("/classes", headers=admin_headers, json={"name": "Test Class"})
    assert r.status_code == 200
    return r.json()["id"]


@pytest.mark.asyncio
async def test_create_and_list_classes(client, admin_headers):
    await client.post("/classes", headers=admin_headers, json={"name": "Physics 101"})
    r = await client.get("/classes", headers=admin_headers)
    assert r.status_code == 200
    names = [c["name"] for c in r.json()]
    assert "Physics 101" in names


@pytest.mark.asyncio
async def test_create_topic(client, admin_headers, class_id):
    r = await client.post(
        f"/classes/{class_id}/topics",
        headers=admin_headers,
        json={"name": "Week 1"},
    )
    assert r.status_code == 200
    assert r.json()["name"] == "Week 1"


@pytest.mark.asyncio
async def test_list_topics(client, admin_headers, class_id):
    await client.post(f"/classes/{class_id}/topics", headers=admin_headers, json={"name": "Intro"})
    r = await client.get(f"/classes/{class_id}/topics", headers=admin_headers)
    assert r.status_code == 200
    assert any(t["name"] == "Intro" for t in r.json())


@pytest.mark.asyncio
async def test_deactivate_class(client, admin_headers, class_id):
    r = await client.patch(f"/classes/{class_id}/deactivate", headers=admin_headers)
    assert r.status_code == 200
    r2 = await client.patch(f"/classes/{class_id}/activate", headers=admin_headers)
    assert r2.status_code == 200
