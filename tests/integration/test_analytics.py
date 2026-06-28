"""Endpoint tests for the Analytics pivot explorer (#210).

Covers the generalized group-by: metric × group_by × stack_by × filters → a
grouped series + KPI totals + the plan-tier framing block. The explorer's
line / bar / hbar views are all client-side pivots of this one `rows` shape.
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


def _seed(db, plan="api"):
    now = utcnow()
    for d in range(4):
        for model, provider, cost in [("claude-opus-4-7", "anthropic", 0.06),
                                       ("gpt-4o", "openai", 0.03)]:
            sid = f"s{d}-{model}"
            db.upsert_session(make_session(session_id=sid, plan_tier=plan, agent_id="cc"))
            llm = make_llm_span(
                session_id=sid, agent_id="cc", model=model, provider=provider,
                input_tokens=2000, output_tokens=300, cost_usd=cost,
                start_time=now - timedelta(days=d),
            )
            db.insert_span(llm)
            for tool in ["Read", "Bash", "Grep"]:
                t = make_tool_span(agent_id="cc", tool_name=tool, trace_id=llm.trace_id)
                t.session_id = sid
                t.start_time = now - timedelta(days=d)
                db.insert_span(t)


async def _get(db, cfg, qs):
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return (await c.get("/api/v1/analytics?" + qs)).json()


@pytest.mark.asyncio
async def test_spend_by_model_grouped_series_and_kpis():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    d = await _get(db, cfg, "metric=spend&group_by=model&since=30d")
    assert d["metric"] == "spend" and d["value_unit"] == "usd"
    # groups sorted by total desc; opus (0.06×4) > gpt (0.03×4)
    assert d["groups"] == ["claude-opus-4-7", "gpt-4o"]
    assert d["rows"], "expected grouped rows"
    for r in d["rows"]:
        assert {"bucket", "group", "stack", "value", "tokens"} <= set(r)
    # KPIs are window totals, independent of group_by
    assert d["kpis"]["spend"] == pytest.approx(0.36)
    assert d["kpis"]["sessions"] == 8
    assert d["framing"]["pricing_mode"] == "api"


@pytest.mark.asyncio
@pytest.mark.parametrize("metric,unit", [
    ("tokens", "tokens"), ("sessions", "count"), ("events", "count"),
])
async def test_metric_dimension_matrix(metric, unit):
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    d = await _get(db, cfg, f"metric={metric}&group_by=agent&since=30d")
    assert d["value_unit"] == unit
    assert d["groups"] == ["cc"]
    assert d["rows"]


@pytest.mark.asyncio
async def test_tool_category_dimension_includes_tool_spans():
    """tool / tool_category breakdowns must see tool spans (NULL model), which
    the LLM-cost `model IS NOT NULL` gate would otherwise hide."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    d = await _get(db, cfg, "metric=events&group_by=tool_category&since=30d")
    assert set(d["groups"]) == {"file", "shell", "search"}  # Read/Edit, Bash, Grep
    assert "(none)" not in d["groups"]
    # KPI totals reflect the FULL window (LLM spend included), not the tool-only
    # subtype gate the breakdown applies — so "events by tool" never zeroes Spend.
    assert d["kpis"]["spend"] > 0
    assert d["kpis"]["tokens"] > 0


@pytest.mark.asyncio
async def test_stack_by_returns_second_dimension():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    d = await _get(db, cfg, "metric=spend&group_by=provider&stack_by=model&since=30d")
    assert d["stack_by"] == "model"
    assert set(d["groups"]) == {"anthropic", "openai"}
    assert set(d["stacks"]) == {"claude-opus-4-7", "gpt-4o"}
    assert any(r["stack"] for r in d["rows"])


@pytest.mark.asyncio
async def test_filters_scope_the_window():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    d = await _get(db, cfg, "metric=spend&group_by=model&provider=anthropic&since=30d")
    assert d["groups"] == ["claude-opus-4-7"]
    assert "gpt-4o" not in d["groups"]


@pytest.mark.asyncio
async def test_subscription_framing_suppresses_dollars():
    db = InMemoryBackend()
    cfg = TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="max_20x")})
    _seed(db, plan="max_20x")
    d = await _get(db, cfg, "metric=spend&group_by=model&since=30d")
    assert d["framing"]["pricing_mode"] == "subscription"
    # token volume is present per row so the UI renders token-share, not $
    assert all("tokens" in r for r in d["rows"])


@pytest.mark.asyncio
async def test_unknown_metric_and_dimension_rejected():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        assert (await c.get("/api/v1/analytics?metric=bogus")).status_code == 400
        assert (await c.get("/api/v1/analytics?group_by=bogus")).status_code == 400


@pytest.mark.asyncio
async def test_empty_window_is_safe():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    d = await _get(db, cfg, "metric=spend&group_by=model&since=7d")
    assert d["rows"] == [] and d["groups"] == []
    assert d["kpis"]["spend"] == 0
    assert "framing" in d
    # #217: the sparkline/delta fields are always present. With a window set,
    # the delta keys exist but are null (no prior data → no divide-by-zero).
    assert d["kpi_series"] == []
    assert set(d["kpi_deltas"]) == {"spend", "tokens", "events", "sessions"}
    assert all(v is None for v in d["kpi_deltas"].values())


@pytest.mark.asyncio
async def test_kpi_series_and_deltas_for_sparklines():
    """#217: the explorer returns a per-bucket `kpi_series` (all four metrics)
    and a period-over-period `kpi_deltas` vs the prior equal-length window —
    both computed server-side so the UI never aggregates per-bucket in JS."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)  # 4 days of data, two models/day
    d = await _get(db, cfg, "metric=spend&group_by=model&since=30d")
    # Per-bucket series carries every KPI metric for each bucket.
    assert d["kpi_series"], "expected a non-empty kpi_series"
    for b in d["kpi_series"]:
        assert {"bucket", "spend", "tokens", "events", "sessions"} <= set(b)
    # Series spend reconciles with the window KPI total.
    assert sum(b["spend"] for b in d["kpi_series"]) == pytest.approx(d["kpis"]["spend"])
    # Prior window had no data (seed is the most recent 4 days) → prev=0 → null
    # delta, never a divide-by-zero.
    assert set(d["kpi_deltas"]) == {"spend", "tokens", "events", "sessions"}
    assert d["kpi_deltas"]["spend"] is None
    assert d["kpi_prev"]["spend"] == 0


@pytest.mark.asyncio
async def test_kpi_delta_non_null_when_prior_window_has_data():
    """#217: a real period-over-period % when both the current and prior
    equal-length windows carry data (the seed spans 4 consecutive days, so a
    2-day window's prior 2 days are non-empty)."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    d = await _get(db, cfg, "metric=spend&group_by=model&since=2d")
    assert d["kpi_prev"]["spend"] > 0, "prior window should have seeded data"
    assert d["kpi_deltas"]["spend"] is not None
    assert isinstance(d["kpi_deltas"]["spend"], float)


@pytest.mark.asyncio
async def test_session_spanning_buckets_counted_once_in_totals_by_group():
    """#45: a session active across multiple time buckets must count once in
    `totals_by_group` for the sessions metric (COUNT(DISTINCT) is non-additive),
    so the grouped totals reconcile with the window-wide Sessions KPI rather than
    inflating to the per-bucket sum."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    now = utcnow()
    sid = "spanning"
    db.upsert_session(make_session(session_id=sid, plan_tier="api", agent_id="cc"))
    # One session with two LLM spans on consecutive days -> two daily buckets.
    for d in (0, 1):
        db.insert_span(make_llm_span(
            session_id=sid, agent_id="cc", model="claude-opus-4-7",
            provider="anthropic", input_tokens=1000, output_tokens=100,
            cost_usd=0.05, start_time=now - timedelta(days=d),
        ))
    d = await _get(db, cfg, "metric=sessions&group_by=agent&since=30d")
    assert d["metric"] == "sessions"
    # True window-wide distinct session count per group is 1, not the 2 you get
    # by summing the per-bucket distinct counts.
    assert d["totals_by_group"]["cc"] == 1
    # And it reconciles with the Sessions KPI tile on the same screen.
    assert d["totals_by_group"]["cc"] == d["kpis"]["sessions"]
