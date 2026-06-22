"""#197 — `tj cost` (CLI) and `/api/v1/cost` (Lens) must render IDENTICAL
plan-tier framing for the same DB / window / plan state.

Both surfaces consume ``core/framing``, but the CLI direct-conn path framed off
a window-SCOPED session mix while the API moved to a window-INDEPENDENT mix
(#177). When the in-window mix differed from full history (daemon-off), they
disagreed — the CLI suppressed dollars while Lens showed them, or vice-versa.

These tests seed sessions that STARTED outside a 24h window but have recent
spans, so the window-scoped mix is empty while full history carries the real
plan — the exact divergence. They then assert the CLI's ``_cost_framing`` and
the live ``/api/v1/cost`` framing block agree across api / subscription /
unknown. (Pre-fix, the subscription and unknown cases fail.)
"""
from __future__ import annotations

import dataclasses
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.cli.cmd_cost import _cost_framing
from tokenjam.core.config import ProviderBudget, StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session

# The framing fields that express "units + qualifier" (AC #1) — the decision,
# not the window-scoped dollar totals carried alongside it.
_DECISION_FIELDS = (
    "pricing_mode",
    "plan_tier",
    "plan_label",
    "plan_labels",
    "display_rule",
    "qualifier_text",
    "subscription_share_pct",
    "api_share_pct",
)


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch, tmp_path):
    """Point Path.home() at an empty dir so config_declared_plan's global
    fallback never reads the dev machine's ~/.config/tj/config.toml."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


def _seed_out_of_window(db, plan_tier: str) -> None:
    """20 sessions that STARTED 5 days ago (outside a 24h window) but each have
    a span in the last hour. Window-scoped session mix → empty; full-history
    mix → {plan_tier: 20}. That gap is exactly what made the CLI and API
    disagree daemon-off (#197)."""
    now = utcnow()
    for i in range(20):
        s = make_session(session_id=f"s-{i}", plan_tier=plan_tier)
        s = dataclasses.replace(s, started_at=now - timedelta(days=5))
        db.upsert_session(s)
        db.insert_span(make_llm_span(
            session_id=f"s-{i}", billing_account="anthropic",
            start_time=now - timedelta(minutes=10 + i),
            input_tokens=1000, output_tokens=200, cost_usd=5.0,
        ))


def _decision(framing_dict: dict) -> dict:
    return {k: framing_dict.get(k) for k in _DECISION_FIELDS}


async def _api_framing(db, config) -> dict:
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/api/v1/cost?since=24h")
    return resp.json()["framing"]


def _cli_framing(db, config) -> dict:
    now = utcnow()
    ctx = SimpleNamespace(obj={"config": config})
    framing = _cost_framing(
        ctx, db,
        since="24h",
        since_dt=now - timedelta(hours=24),
        until_dt=now,
        agent=None,
        total_cost=0.0,
        total_tokens=0,
    )
    assert framing is not None  # direct-conn path: config + conn both present
    return framing.to_dict()


@pytest.mark.asyncio
@pytest.mark.parametrize("plan_tier,budgets,expected_mode", [
    ("api", {"anthropic": ProviderBudget(plan="api")}, "api"),
    ("max_5x", {"anthropic": ProviderBudget(plan="max_5x")}, "subscription"),
    ("unknown", {}, "unknown"),
])
async def test_cli_and_api_framing_agree(tmp_path, plan_tier, budgets, expected_mode):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = TjConfig(version="1", budgets=budgets)
    _seed_out_of_window(db, plan_tier=plan_tier)

    api = _decision(await _api_framing(db, config))
    cli = _decision(_cli_framing(db, config))

    # Both must reach the same units + qualifier for the same DB/window/plan.
    assert cli == api, f"CLI {cli} != API {api}"
    # And the shared decision must be the expected plan-driven mode (not the
    # empty-window "api" default the CLI used to collapse to).
    assert api["pricing_mode"] == expected_mode, api
    db.close()
