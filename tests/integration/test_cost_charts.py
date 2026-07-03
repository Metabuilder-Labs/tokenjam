"""Endpoint tests for the Lens Visualizations Wave 1 cost charts (#212, #213).

Covers the server-side data shaping the charts consume:
- /cost `series` carries the reusable (bucket, agent, model, provider) group-by
  shape with the full token-component split (#213 stacked + #210 explorer reuse).
- /cost/cache returns per-bucket hit-rate + measured captured savings + the
  cache analyzer's *estimated recoverable*, plus a plan-tier framing block.
"""
from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import ProviderBudget, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span


def _app(db, config):
    return create_app(config=config, db=db, ingest_pipeline=IngestPipeline(db=db, config=config))


def _seed(db, plan_tier="api"):
    now = utcnow()
    for i in range(3):
        s = make_session(session_id=f"s{i}", plan_tier=plan_tier)
        db.upsert_session(s)
        db.insert_span(make_llm_span(
            session_id=f"s{i}", model="claude-haiku-4-5", provider="anthropic",
            input_tokens=10_000, output_tokens=500,
            cache_tokens=4_000, cache_write_tokens=1_000,
            cost_usd=0.02, start_time=now - timedelta(days=i),
        ))


@pytest.mark.asyncio
async def test_cost_series_carries_reusable_groupby_shape():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost?since=30d&group_by=model")).json()
    assert d["series"], "expected a window series"
    row = d["series"][0]
    # The group-by shape carries every dimension + the full token split so the
    # stacked chart and the future explorer can pivot client-side.
    for key in ("bucket", "agent_id", "model", "provider", "cost_usd",
                "input_tokens", "output_tokens", "cache_tokens", "cache_write_tokens"):
        assert key in row, f"series row missing {key}"
    assert row["provider"] == "anthropic"
    assert row["cache_tokens"] == 4_000


@pytest.mark.asyncio
async def test_cost_cache_endpoint_hitrate_and_captured():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/cache?since=30d")).json()

    assert d["series"], "expected per-bucket cache series"
    p = d["series"][0]
    # hit-rate = cache_read / (cache_read + input) = 4000 / 14000
    assert abs(p["hit_rate"] - (4000 / 14000)) < 1e-3
    # captured = cache_read * (input_rate - cache_read_rate) / 1e6
    #          = 4000 * (1.00 - 0.10) / 1e6 = 0.0036  (Haiku 4.5)
    assert abs(p["captured_usd"] - 0.0036) < 1e-6
    assert p["captured_tokens"] == 4000
    # window totals + estimated recoverable from the cache analyzer
    assert d["total_captured_tokens"] == 12_000
    assert d["estimated_recoverable_usd"] is not None
    # framing block present (single compute path)
    assert d["framing"]["pricing_mode"] == "api"


@pytest.mark.asyncio
async def test_cost_cache_framing_subscription_suppresses_dollars():
    """Subscription plan → framing suppresses dollars; tokens stay meaningful."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="max_20x")})
    _seed(db, plan_tier="max_20x")
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/cache?since=30d")).json()
    assert d["framing"]["pricing_mode"] == "subscription"
    # token figures are still present so the UI can render the tokens framing
    assert d["total_captured_tokens"] == 12_000


@pytest.mark.asyncio
async def test_cost_cache_empty_window_is_safe():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/cache?since=7d")).json()
    assert d["series"] == []
    assert d["total_captured_usd"] == 0
    assert "framing" in d


# --- #211: cost-by-component + recoverable-waste overlay ------------------- #
def _seed_downsize_and_cache(db, plan_tier="api"):
    """Small Opus sessions (downsize candidate) with cache tokens (cache
    recoverable) + tool spans, so build_report yields multiple recoverables."""
    now = utcnow()
    for i in range(6):
        s = make_session(session_id=f"s{i}", plan_tier=plan_tier, agent_id="cc")
        db.upsert_session(s)
        llm = make_llm_span(
            session_id=f"s{i}", agent_id="cc", model="claude-opus-4-7",
            provider="anthropic", input_tokens=1200, output_tokens=200,
            cache_tokens=3000, cache_write_tokens=800, cost_usd=0.05,
            start_time=now - timedelta(days=i),
        )
        db.insert_span(llm)
        for _ in range(2):
            t = make_tool_span(agent_id="cc", tool_name="Read", trace_id=llm.trace_id)
            t.session_id = f"s{i}"
            t.start_time = now - timedelta(days=i)
            db.insert_span(t)


@pytest.mark.asyncio
async def test_components_split_priced_per_component():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed_downsize_and_cache(db)
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/components?since=30d")).json()
    keys = [c["key"] for c in d["components"]]
    assert keys == ["input", "output", "cache_read", "cache_write"]
    by = {c["key"]: c for c in d["components"]}
    # Opus 4.7 input rate; 6 × 1200 = 7200 input tokens. Cost must be > 0 and
    # token volume exact.
    assert by["input"]["tokens"] == 7200
    assert by["cache_read"]["tokens"] == 18000
    assert by["input"]["cost_usd"] > 0
    assert d["framing"]["pricing_mode"] == "api"


@pytest.mark.asyncio
async def test_components_recoverable_is_registry_driven_and_honest():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed_downsize_and_cache(db)
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/components?since=30d")).json()
    rec = {r["analyzer"]: r for r in d["recoverable"]}
    # downsize is the typed slot; reuse comes from the findings dict — its
    # presence proves the collector is registry-driven (not a hard-coded list).
    assert "downsize" in rec
    assert "reuse" in rec
    # each carries a component attribution
    assert rec["downsize"]["component"] == "call"
    assert rec["cache"]["component"] == "input" if "cache" in rec else True
    # downsize keeps its mandatory caveat verbatim (Rule 14)
    from tokenjam.core.optimize.types import MODEL_DOWNGRADE_CAVEAT
    assert rec["downsize"]["caveat"] == MODEL_DOWNGRADE_CAVEAT
    # honesty: the payload never calls recoverable "saved"
    import json
    assert "saved" not in json.dumps(d).lower()
    assert d["total_recoverable_usd"] > 0


@pytest.mark.asyncio
async def test_components_subscription_framing_suppresses_dollars():
    db = InMemoryBackend()
    cfg = TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="max_20x")})
    _seed_downsize_and_cache(db, plan_tier="max_20x")
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/components?since=30d")).json()
    assert d["framing"]["pricing_mode"] == "subscription"
    # token volumes stay present so the UI can render the tokens framing
    assert sum(c["tokens"] for c in d["components"]) > 0


@pytest.mark.asyncio
async def test_components_empty_window_is_safe():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/cost/components?since=7d")).json()
    assert d["total_cost_usd"] == 0
    assert d["recoverable"] == []
    assert "framing" in d
