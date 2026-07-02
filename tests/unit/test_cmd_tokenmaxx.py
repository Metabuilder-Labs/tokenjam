"""Unit tests for `tj tokenmaxx` — the quota/efficiency card (#7).

Reframed from the old spend-tier card: the card now leads with context
COMPOSITION (overhead vs real work) pulled from the #4 diagnostic, is
quota-native for subscription plans (NO dollar spend-brag — mirrors #5's
polarity), and has a weekly "quota Wrapped" recap mode.

Two layers under test:
  * efficiency-tier classification from the overhead (re-read) share;
  * the CLI render / JSON over a synthetic multi-session fixture, asserting the
    quota-native framing (composition + reclaimed, never a spend flex) and the
    plan-tier suppression of dollars.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pytest
from click.testing import CliRunner

from tokenjam.cli.cmd_tokenmaxx import (
    Tier,
    _TIERS,
    _classify,
    cmd_tokenmaxx,
)
from tokenjam.core.config import CaptureConfig, ProviderBudget, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session

# Anchor the fixture a couple of hours before "now" so a relative `--since`
# window (parsed against utcnow() in the CLI) always covers it.
BASE = utcnow() - timedelta(hours=2)


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch, tmp_path):
    """Keep framing's global-config fallback from reading this machine's
    ~/.config/tj/config.toml during the no-plan tests."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


# ───────────────────────────── classification ─────────────────────────────

def test_classify_lean_context_is_minimizer():
    # Very low overhead → the leanest tier. Efficiency is the brag now, not spend.
    assert _classify(0.0).label == "TokenMinimizer"
    assert _classify(0.30).label == "TokenMinimizer"


def test_classify_walks_overhead_boundaries_lean_to_heavy():
    # Five efficiency tiers, boundaries at 30% / 50% / 70% / 85% overhead.
    # Lower overhead = leaner context = a better tier.
    cases = [
        (0.00, "TokenMinimizer"),
        (0.30, "TokenMinimizer"),
        (0.3001, "LeanOperator"),
        (0.50, "LeanOperator"),
        (0.5001, "SteadyState"),
        (0.70, "SteadyState"),
        (0.7001, "ContextHeavy"),
        (0.85, "ContextHeavy"),
        (0.8501, "QuotaSink"),
        (1.00, "QuotaSink"),
    ]
    for share, expected in cases:
        assert _classify(share).label == expected, f"failed at overhead {share}"


def test_classify_is_monotonic_more_overhead_never_better():
    # Sanity: as overhead climbs, the tier index never decreases. Guards a
    # future table edit from accidentally rewarding waste.
    labels = [t.label for t in _TIERS]
    for share in (0.1, 0.4, 0.6, 0.8, 0.95):
        tier = _classify(share)
        assert tier.label in labels


def test_every_tier_has_emoji_and_quip():
    # The card's social-shareability depends on every tier having readable text.
    for tier in _TIERS:
        assert isinstance(tier, Tier)
        assert tier.label
        assert tier.emoji
        assert tier.quip


# ───────────────────────────── fixtures ───────────────────────────────────

def _seed_overhead_heavy(db, plan_tier: str = "max_5x") -> None:
    """One re-read-heavy session: ~95% of tokens are cache re-reads (overhead),
    plus a big-cache turn that clears the compact threshold so the card can
    surface a 'reclaimable' figure.
    """
    sess = make_session(session_id="sess-a", plan_tier=plan_tier,
                        duration_seconds=120.0)
    db.upsert_session(sess)
    span = make_llm_span(
        model="claude-opus-4-6",
        input_tokens=4_000,            # net-new work
        output_tokens=1_000,           # work produced
        cache_tokens=250_000,          # re-reading history / CLAUDE.md (overhead)
        cache_write_tokens=0,
        cost_usd=2.5,
        session_id="sess-a",
    )
    span.start_time = BASE
    db.insert_span(span)


def _seed_lean(db, plan_tier: str = "max_5x") -> None:
    """A lean session: mostly net-new work, low overhead share."""
    sess = make_session(session_id="sess-lean", plan_tier=plan_tier,
                        duration_seconds=30.0)
    db.upsert_session(sess)
    span = make_llm_span(
        model="claude-haiku-4-5",
        input_tokens=5_000,
        output_tokens=3_000,
        cache_tokens=500,              # negligible overhead
        cost_usd=0.1,
        session_id="sess-lean",
    )
    span.start_time = BASE
    db.insert_span(span)


def _config(plan_tier: str | None) -> TjConfig:
    budgets = {}
    if plan_tier is not None:
        budgets["anthropic"] = ProviderBudget(plan=plan_tier)
    return TjConfig(
        version="1",
        capture=CaptureConfig(tool_inputs=True),
        budgets=budgets,
    )


def _invoke(db, config, *args) -> object:
    runner = CliRunner()
    return runner.invoke(
        cmd_tokenmaxx, list(args),
        obj={"db": db, "config": config, "agent": None},
    )


# ───────────────────────── render: composition framing ────────────────────

def test_card_leads_with_overhead_vs_work_composition(db):
    _seed_overhead_heavy(db, plan_tier="max_5x")
    result = _invoke(db, _config("max_5x"))
    assert result.exit_code == 0, result.output
    out = result.output
    # The headline is the COMPOSITION — overhead vs real work — not a spend tier.
    assert "overhead" in out.lower()
    assert "real work" in out.lower()
    # A heavily re-read window classifies into a high-overhead tier.
    assert "QuotaSink" in out or "ContextHeavy" in out


def test_card_is_quota_native_for_subscription_no_dollar_spend_brag(db):
    # The load-bearing reframe: a subscription user sees a token-share headline
    # and NEVER a dollar spend figure (no "$X in last 30d" flex).
    _seed_overhead_heavy(db, plan_tier="max_5x")
    result = _invoke(db, _config("max_5x"))
    assert result.exit_code == 0, result.output
    out = result.output
    assert "% of cycle tokens" in out
    # No dollar figure anywhere on a subscription card.
    assert "$" not in out
    assert "Implied API value" not in out


def test_card_surfaces_reclaimable_quota_not_spend(db):
    # The action line frames "what you can reclaim" in quota terms (compact
    # candidates), pointing at tj context — the payoff, not a brag.
    _seed_overhead_heavy(db, plan_tier="max_5x")
    result = _invoke(db, _config("max_5x"))
    assert result.exit_code == 0, result.output
    out = result.output
    assert "reclaim" in out.lower()
    assert "tj context" in out


def test_api_plan_shows_implied_dollars_as_secondary_only(db):
    # API users DO get a dollar calibration line — but demoted, labeled
    # "Implied API value", never the headline (mirrors quota-audit #5).
    _seed_overhead_heavy(db, plan_tier="api")
    result = _invoke(db, _config("api"))
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Implied API value" in out
    assert "calibration only" in out
    # Still composition-led, not spend-led.
    assert "overhead" in out.lower()


def test_lean_context_rewards_an_efficiency_tier(db):
    _seed_lean(db, plan_tier="max_5x")
    result = _invoke(db, _config("max_5x"))
    assert result.exit_code == 0, result.output
    out = result.output
    assert "TokenMinimizer" in out or "LeanOperator" in out
    # Even lean cards are quota-native (no dollars for subscription).
    assert "$" not in out


def test_weekly_recap_mode_renders_wrapped_title(db):
    _seed_overhead_heavy(db, plan_tier="max_5x")
    result = _invoke(db, _config("max_5x"), "--weekly")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Weekly Recap" in out
    assert "this week" in out


def test_no_data_prints_onboarding_hint(db):
    # Empty DB → friendly onboarding hint, not a crash.
    result = _invoke(db, _config("max_5x"))
    assert result.exit_code == 0, result.output
    assert "onboard --claude-code" in result.output


# ───────────────────────────── JSON output ────────────────────────────────

def test_json_emits_composition_shares_not_spend_tier(db):
    _seed_overhead_heavy(db, plan_tier="max_5x")
    result = _invoke(db, _config("max_5x"), "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # The JSON is composition-shaped: overhead/work shares + token totals.
    assert "overhead_share" in payload
    assert "work_share" in payload
    assert payload["overhead_share"] > 0.80
    assert payload["total_reread_tokens"] > payload["total_work_tokens"]
    # Quota-native: pricing mode reflects the subscription plan.
    assert payload["pricing_mode"] == "subscription"
    assert payload["plan_tier"] == "max_5x"
    # No spend-brag keys carried over from the old card.
    assert "tier" in payload  # efficiency tier label still present
    assert payload["tier"] in {t.label for t in _TIERS}


def test_json_weekly_flag_round_trips(db):
    _seed_overhead_heavy(db, plan_tier="max_5x")
    result = _invoke(db, _config("max_5x"), "--weekly", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["weekly"] is True


# ───────────────────────────── error paths ────────────────────────────────

def test_requires_direct_db_connection(db):
    # Mirrors cmd_context / cmd_quota_audit: a connection-less backend (API shim
    # style) is rejected with a clear daemon-down hint.
    class _NoConn:
        pass

    result = _invoke(_NoConn(), _config("max_5x"))
    assert result.exit_code != 0
    assert "direct database connection" in str(result.output) + str(result.exception)
