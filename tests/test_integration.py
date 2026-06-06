"""Integration tests — require a real PostgreSQL instance with pgvector.

These tests spin up the FastAPI app against a live database to verify the full
request/response cycle, model persistence, and API contract.

Run with:
    DATABASE_URL=postgresql+asyncpg://flora:flora@localhost:5432/flora \\
        pytest tests/test_integration.py -v -m integration

The CI workflow provisions a pgvector/pgvector:pg17 service container, so these
tests run automatically on push. They are skipped when DATABASE_URL points to
SQLite or is absent.
"""
from __future__ import annotations

import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Skip entire module if no PostgreSQL DATABASE_URL is configured
pytestmark = pytest.mark.integration

_DB_URL = os.getenv("DATABASE_URL", "")
_HAS_PG = _DB_URL.startswith("postgresql")

if not _HAS_PG:
    pytest.skip(
        "Integration tests require DATABASE_URL pointing to PostgreSQL",
        allow_module_level=True,
    )


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client with a test-scoped DB engine.

    Creates a fresh SQLAlchemy engine per test so connections are bound
    to the current event loop (avoids asyncpg 'another operation in
    progress' errors).
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

    from database import Base, get_db
    from main import app

    test_engine = create_async_engine(_DB_URL, echo=False)
    test_session_factory = async_sessionmaker(
        test_engine, expire_on_commit=False,
    )

    async with test_engine.begin() as conn:
        await conn.exec_driver_sql(
            "CREATE EXTENSION IF NOT EXISTS vector",
        )
        await conn.exec_driver_sql(
            "CREATE EXTENSION IF NOT EXISTS pg_trgm",
        )
        await conn.run_sync(Base.metadata.create_all)

    async def _test_get_db() -> AsyncGenerator[  # type: ignore[misc]
        AsyncSession, None,
    ]:
        async with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _test_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as c:
        yield c

    app.dependency_overrides.clear()
    await test_engine.dispose()


# ── Health check ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Flower CRUD ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_flower(client: AsyncClient):
    resp = await client.post(
        "/flowers", json={"latin_name": "Testus plantus integration"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["latin_name"] == "Testus plantus integration"
    assert data["status"] == "pending"
    assert data["id"] > 0


@pytest.mark.asyncio
async def test_create_flower_duplicate_returns_409(client: AsyncClient):
    await client.post(
        "/flowers", json={"latin_name": "Testus duplicatus"},
    )
    resp = await client.post(
        "/flowers", json={"latin_name": "Testus duplicatus"},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_list_flowers(client: AsyncClient):
    await client.post(
        "/flowers", json={"latin_name": "Testus listus"},
    )
    resp = await client.get("/flowers")
    assert resp.status_code == 200
    flowers = resp.json()
    assert isinstance(flowers, list)
    assert any(f["latin_name"] == "Testus listus" for f in flowers)


@pytest.mark.asyncio
async def test_list_flowers_status_filter(client: AsyncClient):
    await client.post(
        "/flowers", json={"latin_name": "Testus pending filter"},
    )
    resp = await client.get("/flowers?status=pending")
    assert resp.status_code == 200
    flowers = resp.json()
    assert all(f["status"] == "pending" for f in flowers)


@pytest.mark.asyncio
async def test_get_flower_by_id(client: AsyncClient):
    create_resp = await client.post(
        "/flowers", json={"latin_name": "Testus getbyid"},
    )
    flower_id = create_resp.json()["id"]

    resp = await client.get(f"/flowers/{flower_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == flower_id


@pytest.mark.asyncio
async def test_get_flower_not_found(client: AsyncClient):
    resp = await client.get("/flowers/999999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_flower(client: AsyncClient):
    create_resp = await client.post(
        "/flowers", json={"latin_name": "Testus deleteme"},
    )
    flower_id = create_resp.json()["id"]

    del_resp = await client.delete(f"/flowers/{flower_id}")
    assert del_resp.status_code == 204

    get_resp = await client.get(f"/flowers/{flower_id}")
    assert get_resp.status_code == 404


