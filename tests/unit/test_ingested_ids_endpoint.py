"""Tests for POST /api/v1/sessions/ingested-ids (#642 + Greptile P2).

The endpoint returns the subset of a candidate session-id list that exists in
the `sessions` table — used by `tj backfill status` when `tj serve` holds the DB
write-lock. Auth is off by default on the local daemon, so the endpoint must cap
the candidate-count to avoid an unbounded request exhausting daemon memory /
blocking the event loop (Greptile P2).
"""
from __future__ import annotations

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.api.routes.sessions import MAX_INGESTED_ID_CANDIDATES
from tokenjam.core.config import (
    ApiAuthConfig,
    ApiConfig,
    SecurityConfig,
    TjConfig,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tests.factories import make_llm_span


def _app(db):
    config = TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="s"),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
    )
    pipeline = IngestPipeline(db=db, config=config)
    pipeline.process(make_llm_span(
        agent_id="claude-code", session_id="sess-a", conversation_id="a"))
    pipeline.process(make_llm_span(
        agent_id="claude-code", session_id="sess-b", conversation_id="b"))
    return create_app(config=config, db=db, ingest_pipeline=pipeline)


@pytest.mark.asyncio
async def test_returns_only_the_ingested_subset():
    db = InMemoryBackend()
    try:
        app = _app(db)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/sessions/ingested-ids",
                json={"session_ids": ["sess-a", "sess-missing", "sess-b"]},
            )
            assert resp.status_code == 200
            assert sorted(resp.json()["ingested"]) == ["sess-a", "sess-b"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_rejects_over_limit_candidate_array():
    """Greptile P2: an oversized id array is rejected with 413, not processed."""
    db = InMemoryBackend()
    try:
        app = _app(db)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            huge = [f"sess-{i}" for i in range(MAX_INGESTED_ID_CANDIDATES + 1)]
            resp = await c.post(
                "/api/v1/sessions/ingested-ids", json={"session_ids": huge},
            )
            assert resp.status_code == 413
            assert "maximum" in resp.json()["error"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_accepts_a_list_at_the_limit():
    """Exactly at the cap is allowed (boundary is inclusive)."""
    db = InMemoryBackend()
    try:
        app = _app(db)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            at_limit = ["sess-a"] + [
                f"sess-{i}" for i in range(MAX_INGESTED_ID_CANDIDATES - 1)
            ]
            assert len(at_limit) == MAX_INGESTED_ID_CANDIDATES
            resp = await c.post(
                "/api/v1/sessions/ingested-ids", json={"session_ids": at_limit},
            )
            assert resp.status_code == 200
            assert resp.json()["ingested"] == ["sess-a"]
    finally:
        db.close()
