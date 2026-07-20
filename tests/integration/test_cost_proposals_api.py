"""Integration tests for the cost-proposal Review-inbox endpoints
(api/routes/pothole.py). Talks through the real ASGI app so the read/write
surface + the advise-only marker round-trip are proven at the route.

Isolated: InMemoryBackend + a tmp storage path; nothing touches a real store.
"""
from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import ApiAuthConfig, ApiConfig, StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config(tmp_path):
    return TjConfig(
        version="1",
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        storage=StorageConfig(path=str(tmp_path / "telemetry.duckdb")),
    )


@pytest.fixture
def app(config, db):
    # Seed low-cache-efficacy spans so the `cache` analyzer flags a row and the
    # refresh produces at least one cost proposal.
    now = utcnow()
    for i in range(12):
        db.insert_span(make_llm_span(
            agent_id="svc-a", provider="anthropic", model="claude-sonnet-5",
            billing_account="anthropic", input_tokens=15_000, output_tokens=200,
            cache_tokens=400, session_id=f"s-{i}",
            start_time=now - timedelta(days=2, minutes=i),
        ))
    pipeline = IngestPipeline(db=db, config=config)
    return create_app(config=config, db=db, ingest_pipeline=pipeline)


@pytest.fixture
def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_cost_proposals_never_run_before_refresh(client):
    r = await client.get("/api/v1/pothole/cost-proposals")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "never_run"
    assert body["proposals"] == []


async def test_cost_refresh_requires_write_token(client):
    r = await client.post("/api/v1/pothole/cost-proposals/refresh")
    assert r.status_code == 401


async def test_cost_refresh_then_proposals_listed(app, client):
    token = app.state.pothole_write_token
    r = await client.post(
        "/api/v1/pothole/cost-proposals/refresh",
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ready"

    r2 = await client.get("/api/v1/pothole/cost-proposals")
    body = r2.json()
    assert body["status"] == "ready"
    assert len(body["proposals"]) >= 1
    cache_props = [p for p in body["proposals"] if p["analyzer"] == "cache"]
    assert cache_props, body["proposals"]
    prop = cache_props[0]
    assert prop["kind"] == "cost"
    assert prop["advise_only"] is True


async def test_mark_cost_applied_round_trip(app, client):
    token = app.state.pothole_write_token
    await client.post(
        "/api/v1/pothole/cost-proposals/refresh",
        headers={"X-TJ-Local-Token": token},
    )
    proposals = (await client.get("/api/v1/pothole/cost-proposals")).json()["proposals"]
    prop = next(p for p in proposals if p["analyzer"] == "cache")

    # Mark applied (the marker) — requires the write token.
    unauth = await client.post("/api/v1/pothole/cost-proposals/apply", json=prop)
    assert unauth.status_code == 401

    r = await client.post(
        "/api/v1/pothole/cost-proposals/apply", json=prop,
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 200
    rec = r.json()
    assert rec["state"] == "applied"
    assert rec["applied_at"]

    applied = (await client.get("/api/v1/pothole/cost-applied")).json()
    assert len(applied["applied"]) == 1
    assert "total_realized_usd" in applied["ledger"]

    # Revert stops the ledger counting it.
    rev = await client.post(
        f"/api/v1/pothole/cost-applied/{rec['id']}/revert",
        headers={"X-TJ-Local-Token": token},
    )
    assert rev.status_code == 200
    assert rev.json()["state"] == "reverted"
