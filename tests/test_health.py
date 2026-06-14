"""Health check tests."""
import pytest


@pytest.mark.asyncio
async def test_health_returns_checks(client):
    r = await client.get("/health")
    # may be 200 or 503 depending on env; either way it must have checks
    assert r.status_code in (200, 503)
    body = r.json()
    assert "checks" in body
    assert "sqlite" in body["checks"]
    assert "chroma" in body["checks"]


@pytest.mark.asyncio
async def test_health_sqlite_ok(client):
    r = await client.get("/health")
    assert r.json()["checks"]["sqlite"] == "ok"
