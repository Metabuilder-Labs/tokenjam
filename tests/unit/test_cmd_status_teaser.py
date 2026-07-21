"""`tj status` recoverable teaser (`_recoverable_teaser`, cmd_status.py).

Nothing in `tj status`, `tj doctor`, `tj statusline`, or the banner ever
pointed at `tj optimize` — a user could run tokenjam for months without
learning the command exists. `build_report` and `plan_tier_mix` are
monkeypatched here rather than seeded through real analyzer thresholds: the
teaser is a thin aggregation layer over the existing #111 recoverable
contract, and what's under test is that aggregation + the honesty/silence
gates around it, not the analyzers themselves (those have their own tests).
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
        findings["cache"] = SimpleNamespace(estimated_recoverable_usd=finding_usd)
    downgrade = None
    if downgrade_usd is not None:
        downgrade = SimpleNamespace(estimated_recoverable_usd=downgrade_usd)
    return SimpleNamespace(downgrade=downgrade, findings=findings)


def test_teaser_prints_dollar_figure_and_points_to_optimize(monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.framing.plan_tier_mix", lambda conn, since, until, agent_id: {"api": 10},
    )
    monkeypatch.setattr(
        "tokenjam.core.optimize.build_report",
        lambda **kw: _report(downgrade_usd=2.0, finding_usd=3.5),
    )

    out = _recoverable_teaser(_FakeDB(conn=object()), config=object())

    assert out is not None
    assert "$5.50" in out
    assert "tj optimize" in out


def test_teaser_silent_below_minimum_threshold(monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.framing.plan_tier_mix", lambda conn, since, until, agent_id: {"api": 10},
    )
    monkeypatch.setattr(
        "tokenjam.core.optimize.build_report",
        lambda **kw: _report(finding_usd=0.42),
    )

    assert _recoverable_teaser(_FakeDB(conn=object()), config=object()) is None


def test_teaser_silent_without_direct_db_connection():
    """Daemon holds the write lock (API-shim mode) — no `.conn` to build a
    report from. Must stay silent, never raise."""
    assert _recoverable_teaser(_FakeDB(conn=None), config=object()) is None


def test_teaser_silent_for_non_api_pricing_mode(monkeypatch):
    """A subscription/local plan pays a flat fee — a raw dollar figure would
    misrepresent it, so the teaser must not print one even with a large
    recoverable total."""
    monkeypatch.setattr(
        "tokenjam.core.framing.plan_tier_mix",
        lambda conn, since, until, agent_id: {"pro": 10},
    )
    monkeypatch.setattr(
        "tokenjam.core.optimize.build_report",
        lambda **kw: _report(finding_usd=50.0),
    )

    assert _recoverable_teaser(_FakeDB(conn=object()), config=object()) is None


def test_teaser_silent_on_build_report_failure(monkeypatch):
    """Never let the teaser computation break `tj status` itself."""
    monkeypatch.setattr(
        "tokenjam.core.framing.plan_tier_mix", lambda conn, since, until, agent_id: {"api": 10},
    )

    def boom(**kw):
        raise RuntimeError("optimize requires a direct DuckDB connection")

    monkeypatch.setattr("tokenjam.core.optimize.build_report", boom)

    assert _recoverable_teaser(_FakeDB(conn=object()), config=object()) is None
