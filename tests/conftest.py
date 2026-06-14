"""Shared fixtures for the test suite."""
import os
import pytest
import tempfile
from httpx import AsyncClient, ASGITransport

# Point DB and Chroma at temp paths before any app import
_tmp_dir = tempfile.mkdtemp()
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("COHERE_API_KEY",     "test-key")
os.environ.setdefault("ADMIN_USERNAME",      "admin")
os.environ.setdefault("ADMIN_PASSWORD",      "changeme1")
os.environ.setdefault("CORS_ORIGINS",        "http://localhost:8000")
# Disable rate limiting in tests
os.environ["RATELIMIT_ENABLED"] = "0"

import src.db as _db
_db.DB_PATH = type(_db.DB_PATH)(_tmp_dir) / "test.db"

import src.config as _cfg
_cfg.CHROMA_DIR = os.path.join(_tmp_dir, "chroma")

from src.app import app
from src.db import init_db
from src.limiter import limiter

# Disable slowapi rate limiting during tests
limiter._enabled = False


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def admin_token(client):
    r = await client.post("/auth/login", json={"username": "admin", "password": "changeme1"})
    assert r.status_code == 200, r.text
    return r.json()["token"]


@pytest.fixture
async def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}
