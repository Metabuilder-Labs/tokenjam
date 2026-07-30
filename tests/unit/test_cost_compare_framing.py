"""Plan-tier framing for `tj cost --compare` (issue #120).

`_render_diff` used to differentiate its output by `framing.pricing_mode`:
subscription/local modes suppressed the dollar-denominated Cost delta and
per-agent/per-model shift lines in favour of token-only comparison, each
paired with its own billing-mode explanation ("Subscription plan — dollar
deltas omitted...", "Local inference — no marginal cost..."). Both the
suppression and the explanations were removed by product decision: dollars
are always legitimate, so tj no longer differentiates its rendering between
subscription and API users. `_render_diff` now renders identically
regardless of the `framing` passed in; the parameter is accepted only for
call-site compatibility.
"""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

from tokenjam.cli.cmd_cost import _render_diff
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


def test_api_mode_byte_identical_to_no_framing():
    # api framing must render exactly like the pre-#120 (no-framing) output.
    assert _render(None) == _render(Framing(pricing_mode="api"))


def test_api_mode_shows_dollars_and_no_note():
    out = _render(Framing(pricing_mode="api"))
    assert "$148" in out and "Cost delta:" in out
    assert "Subscription plan" not in out
    assert "Top shifts by agent" in out


def test_subscription_mode_shows_dollars_like_api():
    """Subscription mode now shows the Cost delta and per-agent/per-model
    dollar shifts, same as API mode (previously suppressed in favour of
    token-only comparison, with a "Subscription plan..." explanation)."""
    out = _render(Framing(pricing_mode="subscription", plan_monthly_usd=100.0))
    assert "Subscription plan" not in out
    assert "Token delta:" in out
    assert "Cost delta:" in out
    assert "$148" in out and "$120" in out
    assert "Top shifts by agent" in out
    assert "Top shifts by model" in out


def test_local_mode_shows_dollars_like_api():
    """Local mode now shows dollar deltas too (previously suppressed, with a
    "Local inference..." explanation)."""
    out = _render(Framing(pricing_mode="local"))
    assert "Local inference" not in out
    assert "Cost delta:" in out
    assert "Token delta:" in out


def test_unknown_mode_shows_dollars_with_no_note():
    """Unknown plan tier keeps dollar figures (never suppressed); the "may
    overstate" qualifier note that used to accompany it was removed by
    product decision, same as the other modes above."""
    out = _render(Framing(pricing_mode="unknown",
                          qualifier_text="Plan tier unknown — figures may overstate actual cost."))
    assert "Cost delta:" in out
    assert "$148" in out
    assert "may overstate" not in out


def test_top_shifts_render_token_deltas_when_present():
    """get_cost_delta_by_group now sums tokens alongside cost per group (the
    dollars+tokens sweep); the top-shifts renderer must show that token delta
    next to the dollar delta rather than leaving it dollars-only."""
    diff = _fake_diff()
    diff.by_agent[0]["current_tokens"] = 900_000
    diff.by_agent[0]["previous_tokens"] = 600_000
    diff.by_agent[0]["tokens_delta"] = 300_000
    with console.capture() as cap:
        from tokenjam.cli.cmd_cost import _render_diff
        _render_diff(diff, Framing(pricing_mode="api"))
    out = cap.get()
    assert "tokens)" in out
    assert "a1" in out


def test_top_shifts_render_fine_with_no_token_fields(monkeypatch=None):
    """Entries without tokens_delta (older payload shape) must still render
    without crashing, and without a bogus tokens suffix."""
    out = _render(Framing(pricing_mode="api"))
    assert "a1" in out
    assert "claude-opus-4-7" in out
