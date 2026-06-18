"""Plan-tier framing for `tj cost --compare` (issue #120)."""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from tokenjam.cli.cmd_cost import _diff_note, _render_diff, _suppresses_dollars
from tokenjam.core.framing import Framing
from tokenjam.utils.formatting import console
from tokenjam.utils.time_parse import utcnow


def _fake_diff():
    now = utcnow()
    cur = SimpleNamespace(since=now - timedelta(days=7), until=now, sessions=10,
                          total_tokens=2_000_000, total_cost_usd=148.0)
    prev = SimpleNamespace(since=now - timedelta(days=14), until=now - timedelta(days=7),
                           sessions=8, total_tokens=1_500_000, total_cost_usd=120.0)
    return SimpleNamespace(
        current=cur, previous=prev,
        cost_delta_usd=28.0, cost_delta_pct=23.3,
        tokens_delta=500_000, tokens_delta_pct=33.3,
        by_agent=[{"group": "a1", "previous_cost": 50.0, "current_cost": 70.0, "delta": 20.0}],
        by_model=[{"group": "claude-opus-4-7", "previous_cost": 80.0, "current_cost": 90.0, "delta": 10.0}],
    )


def _render(framing):
    with console.capture() as cap:
        _render_diff(_fake_diff(), framing)
    return cap.get()


# --- pure helpers ---------------------------------------------------------- #
@pytest.mark.parametrize("mode,expected", [
    ("api", False), ("unknown", False), ("subscription", True), ("local", True),
])
def test_suppresses_dollars(mode, expected):
    assert _suppresses_dollars(mode) is expected


def test_diff_note_per_mode():
    assert _diff_note("api") is None
    assert "Subscription plan" in _diff_note("subscription")
    assert "Local inference" in _diff_note("local")
    assert "unknown" in _diff_note("unknown").lower()


# --- rendering ------------------------------------------------------------- #
def test_api_mode_byte_identical_to_no_framing():
    # api framing must render exactly like the pre-#120 (no-framing) output.
    assert _render(None) == _render(Framing(pricing_mode="api"))


def test_api_mode_shows_dollars_and_no_note():
    out = _render(Framing(pricing_mode="api"))
    assert "$148" in out and "Cost delta:" in out
    assert "Subscription plan" not in out
    assert "Top shifts by agent" in out


def test_subscription_suppresses_dollars():
    out = _render(Framing(pricing_mode="subscription", plan_monthly_usd=100.0))
    assert "Subscription plan" in out
    assert "Token delta:" in out
    assert "Cost delta:" not in out
    assert "$148" not in out and "$120" not in out
    # dollar-denominated per-agent/model shifts are suppressed too
    assert "Top shifts by agent" not in out
    assert "Top shifts by model" not in out


def test_local_suppresses_dollars():
    out = _render(Framing(pricing_mode="local"))
    assert "Local inference" in out
    assert "Cost delta:" not in out
    assert "Token delta:" in out


def test_unknown_keeps_dollars_with_qualifier():
    out = _render(Framing(pricing_mode="unknown",
                          qualifier_text="Plan tier unknown — figures may overstate actual cost."))
    assert "Cost delta:" in out
    assert "$148" in out
    assert "may overstate" in out
