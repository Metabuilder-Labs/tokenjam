"""Integration tests for the cost-proposal Review-inbox endpoints
(api/routes/relearn.py). Talks through the real ASGI app so the read/write
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
    r = await client.get("/api/v1/relearn/cost-proposals")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "never_run"
    assert body["proposals"] == []


async def test_cost_refresh_requires_write_token(client):
    r = await client.post("/api/v1/relearn/cost-proposals/refresh")
    assert r.status_code == 401


async def test_cost_refresh_then_proposals_listed(app, client):
    token = app.state.relearn_write_token
    r = await client.post(
        "/api/v1/relearn/cost-proposals/refresh",
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ready"

    r2 = await client.get("/api/v1/relearn/cost-proposals")
    body = r2.json()
    assert body["status"] == "ready"
    assert len(body["proposals"]) >= 1
    cache_props = [p for p in body["proposals"] if p["analyzer"] == "cache"]
    assert cache_props, body["proposals"]
    prop = cache_props[0]
    assert prop["kind"] == "cost"
    assert prop["advise_only"] is True


async def test_mark_cost_applied_round_trip(app, client):
    token = app.state.relearn_write_token
    await client.post(
        "/api/v1/relearn/cost-proposals/refresh",
        headers={"X-TJ-Local-Token": token},
    )
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    prop = next(p for p in proposals if p["analyzer"] == "cache")

    # Mark applied (the marker) — requires the write token.
    unauth = await client.post("/api/v1/relearn/cost-proposals/apply", json=prop)
    assert unauth.status_code == 401

    r = await client.post(
        "/api/v1/relearn/cost-proposals/apply", json=prop,
        headers={"X-TJ-Local-Token": token},
    )
    assert r.status_code == 200
    rec = r.json()
    assert rec["state"] == "applied"
    assert rec["applied_at"]

    applied = (await client.get("/api/v1/relearn/cost-applied")).json()
    assert len(applied["applied"]) == 1
    assert "total_realized_usd" in applied["ledger"]

    # Revert stops the ledger counting it.
    rev = await client.post(
        f"/api/v1/relearn/cost-applied/{rec['id']}/revert",
        headers={"X-TJ-Local-Token": token},
    )
    assert rev.status_code == 200
    assert rev.json()["state"] == "reverted"


async def test_cost_apply_workspace_writes_note_and_records_marker(app, client, db, monkeypatch, tmp_path):
    """A CC-origin subagent proposal routes a reversible rung-1 note through the
    existing relearn apply path, then records the cost marker for delta-verify."""
    from tokenjam.core.optimize import relearn_apply as pa

    # over_powered subagent fan-out on a premium model, in-window.
    now = utcnow()
    for i in range(4):
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", provider="anthropic", model="claude-opus-4-8",
            billing_account="anthropic", input_tokens=60_000, output_tokens=400,
            cost_usd=0.5, session_id="s1", sub_agent_id=f"sa{i}",
            start_time=now - timedelta(days=2, minutes=i),
        ))
    # Home-anchored target allowlist: point Path.home() at tmp so the CLAUDE.md is "inside".
    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: home))
    target = home / "proj" / "CLAUDE.md"

    token = app.state.relearn_write_token
    hdr = {"X-TJ-Local-Token": token}
    await client.post("/api/v1/relearn/cost-proposals/refresh", headers=hdr)
    proposals = (await client.get("/api/v1/relearn/cost-proposals")).json()["proposals"]
    sub = next(p for p in proposals if p["analyzer"] == "subagent")
    assert sub["apply_capable"] is True
    assert sub["advise_only"] is False

    body = {**sub, "target_path": str(target)}

    # Dry-run: a diff, nothing written, no cost marker.
    dry = await client.post(
        "/api/v1/relearn/cost-proposals/apply-workspace",
        json={**body, "go": False}, headers=hdr,
    )
    assert dry.status_code == 200
    assert dry.json()["applied"]["dry_run"] is True
    assert not target.exists()

    # Write: the note lands, reversibly, and a cost marker is recorded.
    wrote = await client.post(
        "/api/v1/relearn/cost-proposals/apply-workspace",
        json={**body, "go": True}, headers=hdr,
    )
    assert wrote.status_code == 200
    assert target.exists()
    assert "tokenjam" in target.read_text(encoding="utf-8")
    assert wrote.json()["cost_record"] is not None

    applied = (await client.get("/api/v1/relearn/cost-applied")).json()
    assert any(r["analyzer"] == "subagent" for r in applied["applied"])
