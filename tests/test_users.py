"""User management endpoint tests."""
import pytest
import uuid


@pytest.mark.asyncio
async def test_list_users_paginated(client, admin_headers):
    r = await client.get("/users", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert "total" in body
    assert "limit" in body
    assert "offset" in body


@pytest.mark.asyncio
async def test_create_user_and_deactivate(client, admin_headers):
    r = await client.post("/users", headers=admin_headers, json={
        "username": "testteacher", "password": "password123", "role": "teacher"
    })
    assert r.status_code == 200
    uid = r.json()["id"]

    r2 = await client.patch(f"/users/{uid}/deactivate", headers=admin_headers)
    assert r2.status_code == 200

    r3 = await client.patch(f"/users/{uid}/activate", headers=admin_headers)
    assert r3.status_code == 200


@pytest.mark.asyncio
async def test_create_user_requires_admin(client):
    r = await client.post("/users", json={
        "username": "hacker", "password": "password123", "role": "student"
    })
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_create_user_invalid_role(client, admin_headers):
    r = await client.post("/users", headers=admin_headers, json={
        "username": f"u_{uuid.uuid4().hex[:8]}", "password": "password123", "role": "hacker"
    })
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_create_user_duplicate_username(client, admin_headers):
    uname = f"dup_{uuid.uuid4().hex[:8]}"
    await client.post("/users", headers=admin_headers, json={
        "username": uname, "password": "password123", "role": "student"
    })
    r2 = await client.post("/users", headers=admin_headers, json={
        "username": uname, "password": "password123", "role": "student"
    })
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_deactivate_self_fails(client, admin_headers):
    login = await client.post("/auth/login", json={"username": "admin", "password": "changeme1"})
    admin_id = login.json()["user"]["id"]
    r = await client.patch(f"/users/{admin_id}/deactivate", headers=admin_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_delete_user(client, admin_headers):
    uid_r = await client.post("/users", headers=admin_headers, json={
        "username": f"del_{uuid.uuid4().hex[:8]}", "password": "password123", "role": "student"
    })
    uid = uid_r.json()["id"]
    r = await client.delete(f"/users/{uid}", headers=admin_headers)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_delete_self_fails(client, admin_headers):
    login = await client.post("/auth/login", json={"username": "admin", "password": "changeme1"})
    admin_id = login.json()["user"]["id"]
    r = await client.delete(f"/users/{admin_id}", headers=admin_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_delete_nonexistent_user(client, admin_headers):
    r = await client.delete("/users/nonexistent-id-xyz", headers=admin_headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_reset_password_min_length(client, admin_headers):
    r = await client.post("/users", headers=admin_headers, json={
        "username": "pwtest", "password": "password123", "role": "student"
    })
    uid = r.json()["id"]
    r2 = await client.patch(f"/users/{uid}/password", headers=admin_headers, json={"password": "short"})
    assert r2.status_code == 400


@pytest.mark.asyncio
async def test_reset_password_success(client, admin_headers):
    r = await client.post("/users", headers=admin_headers, json={
        "username": f"pwok_{uuid.uuid4().hex[:8]}", "password": "password123", "role": "student"
    })
    uid = r.json()["id"]
    r2 = await client.patch(f"/users/{uid}/password", headers=admin_headers, json={"password": "newpassword1"})
    assert r2.status_code == 200
