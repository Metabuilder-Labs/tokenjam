"""Endpoint tests for the SDK cost-attribution dimensions (tenant/feature/
environment/prompt version) — the multi-tenant Cost view breakdown.

Covers:
- GET /cost group_by=tenant|feature|environment|prompt_version + the equality
  filters for each dimension.
- GET /cost attribution_coverage — the per-dimension "is there any data"
  flags that power the UI's honest empty state.
- GET /cost/tenants — the top-N-tenants-by-spend concentration view: shares
  computed against TOTAL window spend, the unattributed remainder, and the
  honest empty state when no span carries a tenant_id.
"""
from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


def _app(db, config):
    return create_app(config=config, db=db, ingest_pipeline=IngestPipeline(db=db, config=config))


@pytest.mark.asyncio
async def test_cost_group_by_tenant():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    session = make_session(session_id="s1")
    db.upsert_session(session)
    db.insert_span(make_llm_span(
        session_id="s1", model="claude-haiku-4-5", cost_usd=5.0, tenant_id="acme-corp",
    ))
    db.insert_span(make_llm_span(
        session_id="s1", model="claude-haiku-4-5", cost_usd=1.0, tenant_id="small-co",
    ))
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost?since=30d&group_by=tenant")).json()
    groups = {r["group"]: r["cost_usd"] for r in d["rows"]}
    assert groups == {"acme-corp": pytest.approx(5.0), "small-co": pytest.approx(1.0)}


@pytest.mark.asyncio
async def test_cost_filter_by_tenant_id():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    session = make_session(session_id="s1")
    db.upsert_session(session)
    db.insert_span(make_llm_span(
        session_id="s1", model="claude-haiku-4-5", cost_usd=3.0, tenant_id="acme-corp",
    ))
    db.insert_span(make_llm_span(
        session_id="s1", model="gpt-4o", provider="openai", cost_usd=9.0, tenant_id="other-co",
    ))
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost?since=30d&group_by=model&tenant_id=acme-corp")).json()
    assert len(d["rows"]) == 1
    assert d["rows"][0]["cost_usd"] == pytest.approx(3.0)
    assert d["total_cost_usd"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_cost_attribution_coverage_reports_no_data_for_unset_dimension():
    """Zero spans in the window carry tenant_id/feature/environment/
    prompt_version — attribution_coverage must say so explicitly (the honest
    empty-state contract), not just return empty rows silently."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    session = make_session(session_id="s1")
    db.upsert_session(session)
    db.insert_span(make_llm_span(session_id="s1", model="claude-haiku-4-5", cost_usd=1.0))
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost?since=30d")).json()
    cov = d["attribution_coverage"]
    assert cov["tenant"]["has_data"] is False
    assert cov["tenant"]["attribute"] == "tokenjam.tenant_id"
    assert cov["feature"]["has_data"] is False
    assert cov["environment"]["has_data"] is False
    assert cov["prompt_version"]["has_data"] is False


@pytest.mark.asyncio
async def test_cost_attribution_coverage_true_when_dimension_set():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    session = make_session(session_id="s1")
    db.upsert_session(session)
    db.insert_span(make_llm_span(
        session_id="s1", model="claude-haiku-4-5", cost_usd=1.0, tenant_id="acme-corp",
    ))
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost?since=30d")).json()
    assert d["attribution_coverage"]["tenant"]["has_data"] is True
    assert d["attribution_coverage"]["feature"]["has_data"] is False


@pytest.mark.asyncio
async def test_cost_tenants_empty_state_when_no_tenant_data():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    session = make_session(session_id="s1")
    db.upsert_session(session)
    db.insert_span(make_llm_span(session_id="s1", model="claude-haiku-4-5", cost_usd=42.0))
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/tenants?since=30d")).json()
    assert d["has_data"] is False
    assert d["rows"] == []
    assert d["attribute"] == "tokenjam.tenant_id"
    # The window still had real spend — an empty tenant breakdown must not be
    # confused with "no spend at all".
    assert d["total_cost_usd"] == pytest.approx(42.0)
    assert d["unattributed_cost_usd"] == pytest.approx(42.0)


@pytest.mark.asyncio
async def test_cost_tenants_concentration_shares_and_unattributed():
    """Top-N tenants, ordered by spend, with share computed against TOTAL
    window spend (including the unattributed remainder) — never inflated by
    silently excluding spend that has no tenant_id."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    session = make_session(session_id="s1")
    db.upsert_session(session)
    db.insert_span(make_llm_span(
        session_id="s1", model="claude-haiku-4-5", cost_usd=80.0, tenant_id="whale-corp",
    ))
    db.insert_span(make_llm_span(
        session_id="s1", model="claude-haiku-4-5", cost_usd=10.0, tenant_id="small-co",
    ))
    # Unattributed spend — must count toward total_cost_usd and shrink shares,
    # never silently dropped.
    db.insert_span(make_llm_span(session_id="s1", model="claude-haiku-4-5", cost_usd=10.0))

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/tenants?since=30d")).json()

    assert d["has_data"] is True
    assert d["total_cost_usd"] == pytest.approx(100.0)
    assert d["attributed_cost_usd"] == pytest.approx(90.0)
    assert d["unattributed_cost_usd"] == pytest.approx(10.0)
    rows = {r["tenant_id"]: r for r in d["rows"]}
    assert rows["whale-corp"]["cost_usd"] == pytest.approx(80.0)
    # Share is against the FULL window total (100), not the attributed
    # subset (90) — 80/100 = 0.8, not 80/90.
    assert rows["whale-corp"]["share_of_total"] == pytest.approx(0.8)
    assert rows["small-co"]["share_of_total"] == pytest.approx(0.1)
    # Biggest spender first.
    assert d["rows"][0]["tenant_id"] == "whale-corp"


@pytest.mark.asyncio
async def test_cost_tenants_trend_vs_prior_window():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    session = make_session(session_id="s1")
    db.upsert_session(session)
    now = utcnow()
    # Prior 7-day window (8-15 days ago): tenant spent 10.
    db.insert_span(make_llm_span(
        session_id="s1", model="claude-haiku-4-5", cost_usd=10.0, tenant_id="acme-corp",
        start_time=now - timedelta(days=10),
    ))
    # Current 7-day window: tenant spent 20 — a 100% increase.
    db.insert_span(make_llm_span(
        session_id="s1", model="claude-haiku-4-5", cost_usd=20.0, tenant_id="acme-corp",
        start_time=now - timedelta(days=1),
    ))
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/tenants?since=7d")).json()
    row = next(r for r in d["rows"] if r["tenant_id"] == "acme-corp")
    assert row["cost_usd"] == pytest.approx(20.0)
    assert row["previous_cost_usd"] == pytest.approx(10.0)
    assert row["delta_pct"] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_analytics_group_by_tenant():
    """The generalized /analytics explorer pivot also supports the new
    attribution dimensions (#SDK dashboard shape) — a single compute path,
    not a second re-derivation."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    session = make_session(session_id="s1")
    db.upsert_session(session)
    db.insert_span(make_llm_span(
        session_id="s1", model="claude-haiku-4-5", cost_usd=4.0, tenant_id="acme-corp",
    ))
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/analytics?since=30d&metric=spend&group_by=tenant")).json()
    assert "acme-corp" in d["groups"]
