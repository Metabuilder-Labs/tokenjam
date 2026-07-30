"""Endpoint tests pinning the dollars+tokens sweep for /agents and
/cost/tenants: every dollar-bearing read route should also carry the token
figure that was already in scope for the same query, rather than leaking
one unit and dropping the other.
"""
from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.models import AgentRecord
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


def _app(db, config):
    return create_app(config=config, db=db, ingest_pipeline=IngestPipeline(db=db, config=config))


@pytest.mark.asyncio
async def test_agents_route_carries_lifetime_tokens_and_framing():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    now = utcnow()
    db.upsert_agent(AgentRecord(agent_id="alpha", first_seen=now, last_seen=now))
    session = make_session(agent_id="alpha", plan_tier="api")
    db.upsert_session(session)
    db.insert_span(make_llm_span(
        agent_id="alpha", session_id=session.session_id,
        input_tokens=10_000, output_tokens=500,
        cache_tokens=1_000, cache_write_tokens=200,
        cost_usd=0.05,
    ))
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/agents")).json()

    assert d["agents"], "expected the seeded agent to appear"
    row = d["agents"][0]
    assert row["agent_id"] == "alpha"
    assert row["lifetime_cost_usd"] == pytest.approx(0.05)
    # Cache-write tokens must be included in the sum (a recurring omission —
    # see root CLAUDE.md's cache-token-types note).
    assert row["lifetime_tokens"] == 10_000 + 500 + 1_000 + 200
    assert "framing" in d
    assert d["framing"]["pricing_mode"] == "api"


@pytest.mark.asyncio
async def test_cost_tenants_totals_carry_token_counterparts():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    now = utcnow()
    session = make_session(agent_id="alpha", plan_tier="api")
    db.upsert_session(session)
    db.insert_span(make_llm_span(
        agent_id="alpha", session_id=session.session_id,
        model="claude-haiku-4-5", provider="anthropic",
        input_tokens=8_000, output_tokens=1_000,
        cache_tokens=500, cache_write_tokens=100,
        cost_usd=0.10, start_time=now - timedelta(hours=1),
        tenant_id="acme",
    ))
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/tenants?since=7d")).json()

    assert d["has_data"] is True
    assert d["total_tokens"] == 8_000 + 1_000 + 500 + 100
    assert d["attributed_tokens"] == d["total_tokens"]
    assert d["unattributed_tokens"] == 0
    row = d["rows"][0]
    assert row["tenant_id"] == "acme"
    assert row["input_tokens"] == 8_000
    assert row["cache_write_tokens"] == 100
