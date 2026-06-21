"""Unit tests for plan-tier session promotion and framing fallback."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from tokenjam.core.framing import (
    DISPLAY_SHOW_DOLLARS,
    DISPLAY_SUPPRESS_UNKNOWN,
    WindowSummary,
    apply_declared_plans_to_sessions,
    compute_framing,
)
from tokenjam.core.backfill import session_record_from_parsed, ParsedSession
from tokenjam.core.db import InMemoryBackend
from tests.factories import make_llm_span, make_session


@dataclass
class _Budget:
    plan: str | None = None


class _Config:
    def __init__(self, budgets: dict | None = None):
        self.budgets = budgets or {}


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


def test_apply_declared_plans_promotes_unknown_sessions():
    db = InMemoryBackend()
    session = make_session(session_id="s1", plan_tier="unknown")
    db.upsert_session(session)
    span = make_llm_span(session_id="s1", billing_account="anthropic")
    db.insert_span(span)

    cfg = _Config({"anthropic": _Budget(plan="api")})
    n = apply_declared_plans_to_sessions(db.conn, cfg)
    assert n == 1

    row = db.conn.execute(
        "SELECT plan_tier FROM sessions WHERE session_id = $1", ["s1"]
    ).fetchone()
    assert row[0] == "api"


def test_apply_declared_plans_skips_known_sessions():
    db = InMemoryBackend()
    session = make_session(session_id="s1", plan_tier="max_5x")
    db.upsert_session(session)
    span = make_llm_span(session_id="s1", billing_account="anthropic")
    db.insert_span(span)

    cfg = _Config({"anthropic": _Budget(plan="api")})
    n = apply_declared_plans_to_sessions(db.conn, cfg)
    assert n == 0

    row = db.conn.execute(
        "SELECT plan_tier FROM sessions WHERE session_id = $1", ["s1"]
    ).fetchone()
    assert row[0] == "max_5x"


def test_apply_declared_plans_uses_global_config_fallback(monkeypatch, tmp_path):
    global_cfg = tmp_path / ".config" / "tj" / "config.toml"
    global_cfg.parent.mkdir(parents=True)
    global_cfg.write_text(
        '[budget.anthropic]\nplan = "api"\ncycle_start_day = 1\n'
    )
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    db = InMemoryBackend()
    db.upsert_session(make_session(session_id="s1", plan_tier="unknown"))
    db.insert_span(make_llm_span(session_id="s1", billing_account="anthropic"))

    n = apply_declared_plans_to_sessions(db.conn, _Config())
    assert n == 1
    row = db.conn.execute(
        "SELECT plan_tier FROM sessions WHERE session_id = $1", ["s1"]
    ).fetchone()
    assert row[0] == "api"


def test_apply_declared_plans_reconcile_updates_known_sessions():
    db = InMemoryBackend()
    session = make_session(session_id="s1", plan_tier="max_5x")
    db.upsert_session(session)
    span = make_llm_span(session_id="s1", billing_account="anthropic")
    db.insert_span(span)

    cfg = _Config({"anthropic": _Budget(plan="api")})
    n = apply_declared_plans_to_sessions(db.conn, cfg, reconcile=True)
    assert n == 1

    row = db.conn.execute(
        "SELECT plan_tier FROM sessions WHERE session_id = $1", ["s1"]
    ).fetchone()
    assert row[0] == "api"


def test_apply_declared_plans_scopes_by_billing_account():
    db = InMemoryBackend()
    db.upsert_session(make_session(session_id="anthropic-s", plan_tier="unknown"))
    db.upsert_session(make_session(session_id="openai-s", plan_tier="unknown"))
    db.insert_span(make_llm_span(session_id="anthropic-s", billing_account="anthropic"))
    db.insert_span(
        make_llm_span(
            session_id="openai-s",
            provider="openai",
            model="gpt-4o",
            billing_account="openai",
        )
    )

    cfg = _Config({"anthropic": _Budget(plan="api")})
    n = apply_declared_plans_to_sessions(db.conn, cfg)
    assert n == 1

    anthropic = db.conn.execute(
        "SELECT plan_tier FROM sessions WHERE session_id = $1", ["anthropic-s"]
    ).fetchone()[0]
    openai = db.conn.execute(
        "SELECT plan_tier FROM sessions WHERE session_id = $1", ["openai-s"]
    ).fetchone()[0]
    assert anthropic == "api"
    assert openai == "unknown"


def test_compute_framing_all_unknown_falls_back_to_config():
    cfg = _Config({"anthropic": _Budget(plan="api")})
    f = compute_framing(
        cfg,
        WindowSummary(
            total_cost_usd=10.0,
            total_tokens=500,
            sessions=5,
            plan_tier_mix={"unknown": 5},
        ),
    )
    assert f.pricing_mode == "api"
    assert f.plan_tier == "api"
    assert f.plan_label == "API billing"
    assert f.display_rule == DISPLAY_SHOW_DOLLARS
    assert f.qualifier_text is None


def test_compute_framing_all_unknown_no_config_still_suppressed():
    f = compute_framing(
        _Config(),
        WindowSummary(
            total_cost_usd=10.0,
            total_tokens=500,
            sessions=5,
            plan_tier_mix={"unknown": 5},
        ),
    )
    assert f.display_rule == DISPLAY_SUPPRESS_UNKNOWN
    assert "claude-code --reconfigure" in (f.qualifier_text or "")


def test_session_record_from_parsed_accepts_plan_tier():
    from datetime import datetime, timezone

    parsed = ParsedSession(
        session_id="sess-1",
        agent_id="claude-code-proj",
        cwd="/proj",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        spans=[],
        total_input_tokens=0,
        total_output_tokens=0,
        total_cache_tokens=0,
        total_cost_usd=0.0,
        tool_call_count=0,
    )
    rec = session_record_from_parsed(parsed, plan_tier="api")
    assert rec.plan_tier == "api"


def test_upsert_session_preserves_known_plan_tier_on_conflict():
    db = InMemoryBackend()
    db.upsert_session(make_session(session_id="s1", plan_tier="max_5x"))
    db.upsert_session(make_session(session_id="s1", plan_tier="api"))
    row = db.conn.execute(
        "SELECT plan_tier FROM sessions WHERE session_id = $1", ["s1"]
    ).fetchone()
    assert row[0] == "max_5x"
