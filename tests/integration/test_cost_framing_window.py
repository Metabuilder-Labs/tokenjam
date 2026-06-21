"""#177 — /api/v1/cost plan-tier framing must be window-INDEPENDENT.

The cost chart's tokens-vs-dollars unit and the qualifier banner are a property
of the user's plan, not the selected time window. Before the fix, the framing
mix was scoped to the window via ``session.started_at``; a 24h window with no
in-window session *starts* (but recent spans from long-running sessions)
collapsed to the empty-mix "api" default and rendered dollars, while a 7d/30d
window rendered the subscription / unknown framing for the very same user.

These tests seed long-running sessions whose ``started_at`` is outside the 24h
window while their spans are recent, then assert the framing block is identical
across 24h / 7d / 30d. Uses a real DuckDBBackend because the route's mix query
runs against ``db.conn`` (InMemoryBackend has none and short-circuits to {}).
"""
from __future__ import annotations

import dataclasses
from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import ProviderBudget, StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch, tmp_path):
    """Point Path.home() at an empty dir so config_declared_plan's global
    fallback never reads the dev machine's ~/.config/tj/config.toml."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


def _seed_long_running(db, plan_tier: str) -> None:
    """40 sessions that STARTED 3–14 days ago but each have a span in the last
    hour — the divergence that made 24h framing differ from 30d (#177)."""
    now = utcnow()
    for i in range(40):
        days = 3 + (i % 12)
        s = make_session(session_id=f"s-{i}", plan_tier=plan_tier)
        s = dataclasses.replace(s, started_at=now - timedelta(days=days))
        db.upsert_session(s)
        db.insert_span(make_llm_span(
            session_id=f"s-{i}", billing_account="anthropic",
            start_time=now - timedelta(days=days),
            input_tokens=5000, output_tokens=800, cost_usd=40.0,
        ))
        db.insert_span(make_llm_span(
            session_id=f"s-{i}", billing_account="anthropic",
            start_time=now - timedelta(hours=1, minutes=i % 50),
            input_tokens=1000, output_tokens=200, cost_usd=5.0,
        ))


async def _framings_across_windows(db, config):
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    transport = httpx.ASGITransport(app=app)
    out = {}
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        for win in ("24h", "7d", "30d"):
            out[win] = (await c.get(f"/api/v1/cost?since={win}")).json()["framing"]
    return out


@pytest.mark.asyncio
async def test_framing_consistent_across_windows_subscription(tmp_path):
    """Declared subscription user → tokens framing on every window (AC #2)."""
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="max_5x")})
    _seed_long_running(db, plan_tier="unknown")  # data unstamped; plan declared
    framings = await _framings_across_windows(db, config)
    modes = {w: f["pricing_mode"] for w, f in framings.items()}
    rules = {w: f["display_rule"] for w, f in framings.items()}
    assert set(modes.values()) == {"subscription"}, modes
    assert len(set(rules.values())) == 1, rules
    db.close()


@pytest.mark.asyncio
async def test_framing_consistent_across_windows_subscription_from_data(tmp_path):
    """Undeclared user whose sessions are subscription-stamped → tokens on
    every window (no 24h dollars / 30d tokens split)."""
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = TjConfig(version="1")  # no declared plan
    _seed_long_running(db, plan_tier="max_5x")
    framings = await _framings_across_windows(db, config)
    modes = {w: f["pricing_mode"] for w, f in framings.items()}
    assert set(modes.values()) == {"subscription"}, modes
    db.close()


@pytest.mark.asyncio
async def test_framing_consistent_across_windows_unknown(tmp_path):
    """Undeclared user with all-unknown sessions → the SAME unknown framing on
    every window (AC #4: one consistent choice, not 24h api / 30d unknown)."""
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = TjConfig(version="1")  # no declared plan
    _seed_long_running(db, plan_tier="unknown")
    framings = await _framings_across_windows(db, config)
    modes = {w: f["pricing_mode"] for w, f in framings.items()}
    rules = {w: f["display_rule"] for w, f in framings.items()}
    assert set(modes.values()) == {"unknown"}, modes
    assert len(set(rules.values())) == 1, rules
    # The qualifier banner is the same on every window, too.
    quals = {w: f["qualifier_text"] for w, f in framings.items()}
    assert len(set(quals.values())) == 1, quals
    db.close()


@pytest.mark.asyncio
async def test_window_totals_still_vary_by_window(tmp_path):
    """Framing is window-independent, but the dollar/token TOTALS the chart
    plots must still grow with the window — only the *framing* is pinned."""
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="api")})
    _seed_long_running(db, plan_tier="api")
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        c24 = (await c.get("/api/v1/cost?since=24h")).json()
        c30 = (await c.get("/api/v1/cost?since=30d")).json()
    assert c24["framing"]["pricing_mode"] == c30["framing"]["pricing_mode"] == "api"
    assert c30["total_cost_usd"] > c24["total_cost_usd"]
    db.close()
