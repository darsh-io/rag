"""Auth endpoint tests."""
import pytest


@pytest.mark.asyncio
async def test_login_success(client):
    r = await client.post("/auth/login", json={"username": "admin", "password": "changeme1"})
    assert r.status_code == 200
    body = r.json()
    assert "token" in body
    assert "refresh_token" in body
    assert body["user"]["role"] == "supradmin"


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    r = await client.post("/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_user(client):
    r = await client.post("/auth/login", json={"username": "nobody", "password": "password123"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token(client):
    login = await client.post("/auth/login", json={"username": "admin", "password": "changeme1"})
    refresh_token = login.json()["refresh_token"]
    r = await client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert r.status_code == 200
    assert "token" in r.json()


@pytest.mark.asyncio
async def test_refresh_with_access_token_fails(client):
    login = await client.post("/auth/login", json={"username": "admin", "password": "changeme1"})
    access_token = login.json()["token"]
    r = await client.post("/auth/refresh", json={"refresh_token": access_token})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_me_requires_auth(client):
    r = await client.get("/auth/me")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_me_with_token(client, admin_headers):
    r = await client.get("/auth/me", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["role"] == "supradmin"


@pytest.mark.asyncio
async def test_register_password_too_short(client):
    r = await client.post("/auth/register", json={
        "username": "newstudent", "password": "short", "class_id": "fake"
    })
    assert r.status_code == 400
    assert "8 characters" in r.json()["detail"]


@pytest.mark.asyncio
async def test_register_empty_username(client):
    r = await client.post("/auth/register", json={
        "username": "   ", "password": "longpassword", "class_id": "fake"
    })
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_register_class_not_found(client):
    r = await client.post("/auth/register", json={
        "username": "validuser1", "password": "longpassword", "class_id": "nonexistent-class"
    })
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_register_success(client, admin_headers):
    cls_r = await client.post("/classes", headers=admin_headers, json={"name": "Register Test Class"})
    class_id = cls_r.json()["id"]
    import uuid
    uname = f"student_{uuid.uuid4().hex[:8]}"
    r = await client.post("/auth/register", json={
        "username": uname, "password": "longpassword", "class_id": class_id
    })
    assert r.status_code == 200
    body = r.json()
    assert "token" in body
    assert body["user"]["role"] == "student"


@pytest.mark.asyncio
async def test_register_duplicate_username(client, admin_headers):
    cls_r = await client.post("/classes", headers=admin_headers, json={"name": "Dup Test Class"})
    class_id = cls_r.json()["id"]
    import uuid
    uname = f"dupuser_{uuid.uuid4().hex[:8]}"
    await client.post("/auth/register", json={
        "username": uname, "password": "longpassword", "class_id": class_id
    })
    r2 = await client.post("/auth/register", json={
        "username": uname, "password": "longpassword", "class_id": class_id
    })
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_refresh_deactivated_user(client, admin_headers):
    import uuid
    uname = f"deact_{uuid.uuid4().hex[:8]}"
    r = await client.post("/users", headers=admin_headers, json={
        "username": uname, "password": "password123", "role": "teacher"
    })
    uid = r.json()["id"]
    login = await client.post("/auth/login", json={"username": uname, "password": "password123"})
    refresh_token = login.json()["refresh_token"]
    await client.patch(f"/users/{uid}/deactivate", headers=admin_headers)
    r2 = await client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert r2.status_code == 401


@pytest.mark.asyncio
async def test_toggle_registration(client, admin_headers):
    r = await client.patch("/auth/settings/registration", headers=admin_headers, json={"enabled": False})
    assert r.status_code == 200
    assert r.json()["self_registration_enabled"] is False

    r2 = await client.post("/auth/register", json={
        "username": "blocked_user", "password": "longpassword", "class_id": "fake"
    })
    assert r2.status_code == 403

    # Re-enable
    await client.patch("/auth/settings/registration", headers=admin_headers, json={"enabled": True})


@pytest.mark.asyncio
async def test_public_classes(client):
    r = await client.get("/auth/classes")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
