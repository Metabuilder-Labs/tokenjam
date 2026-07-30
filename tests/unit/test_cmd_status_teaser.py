"""`tj status` recoverable teaser (`_recoverable_teaser`, cmd_status.py).

Nothing in `tj status`, `tj doctor`, `tj statusline`, or the banner ever
pointed at `tj optimize` — a user could run tokenjam for months without
learning the command exists. `build_report` is monkeypatched here rather
than seeded through real analyzer thresholds: the teaser is a thin
aggregation layer over the existing #111 recoverable contract, and what's
under test is that aggregation + the honesty/silence gates around it, not
the analyzers themselves (those have their own tests).

The teaser used to gate on the user's plan tier (`plan_tier_mix` +
`pricing_mode_for`), staying silent for subscription/local plans. That gate
was removed by product decision: dollars are always legitimate, so tj no
longer differentiates its output between subscription and API users — the
teaser now fires for any plan tier with a large-enough recoverable figure.
"""
from __future__ import annotations

from types import SimpleNamespace

from tokenjam.cli.cmd_status import _recoverable_teaser


class _FakeDB:
    def __init__(self, conn):
        self.conn = conn


def _report(*, downgrade_usd=None, finding_usd=None):
    findings = {}
    if finding_usd is not None:
        findings["cache"] = SimpleNamespace(past_overspend_usd=finding_usd)
    downgrade = None
    if downgrade_usd is not None:
        downgrade = SimpleNamespace(past_overspend_usd=downgrade_usd)
    return SimpleNamespace(downgrade=downgrade, findings=findings)


def test_teaser_prints_largest_single_estimate_not_a_sum(monkeypatch):
    """The teaser must mirror `largest_recoverable_usd` in api/routes/cost.py:
    downsize and cache price OVERLAPPING angles on the same spans (#111), so
    summing 2.0 + 3.5 into "$5.50" would print an inflated, non-additive
    headline the product's own API disclaims via `recoverable_additive: False`
    and `_recoverable_overlap_note`. The honest figure is the larger of the
    two on its own: $3.50."""
    monkeypatch.setattr(
        "tokenjam.core.optimize.build_report",
        lambda **kw: _report(downgrade_usd=2.0, finding_usd=3.5),
    )

    out = _recoverable_teaser(_FakeDB(conn=object()), config=object())

    assert out is not None
    assert "$3.50" in out
    assert "$5.50" not in out
    assert "tj optimize" in out


def test_teaser_silent_below_minimum_threshold(monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.optimize.build_report",
        lambda **kw: _report(finding_usd=0.42),
    )

    assert _recoverable_teaser(_FakeDB(conn=object()), config=object()) is None


def test_teaser_silent_without_direct_db_connection():
    """Daemon holds the write lock (API-shim mode) — no `.conn` to build a
    report from. Must stay silent, never raise."""
    assert _recoverable_teaser(_FakeDB(conn=None), config=object()) is None


def test_teaser_shows_for_subscription_plan_too(monkeypatch):
    """The `tj optimize` pointer no longer differentiates by billing mode
    (product decision, reversing the prior subscription/local silence):
    dollars are always legitimate, so a subscription-tier user with a large
    recoverable total gets the same dollar teaser an API user would."""
    monkeypatch.setattr(
        "tokenjam.core.optimize.build_report",
        lambda **kw: _report(finding_usd=50.0),
    )

    out = _recoverable_teaser(_FakeDB(conn=object()), config=object())
    assert out is not None
    assert "$50.00" in out
    assert "tj optimize" in out


def test_teaser_silent_on_build_report_failure(monkeypatch):
    """Never let the teaser computation break `tj status` itself."""
    def boom(**kw):
        raise RuntimeError("optimize requires a direct DuckDB connection")

    monkeypatch.setattr("tokenjam.core.optimize.build_report", boom)

    assert _recoverable_teaser(_FakeDB(conn=object()), config=object()) is None
