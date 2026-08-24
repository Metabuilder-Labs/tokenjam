"""Core ranking used by both the CLI text view and GET /optimize."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from tokenjam.core.optimize.rank import (
    ALWAYS_FULL_FINDINGS,
    CARD_FINDING_NAMES,
    rank_findings,
    reclaimable_share,
)
from tokenjam.core.optimize.types import OptimizeReport, WindowSummary

UTC = timezone.utc


def _report(**kwargs) -> OptimizeReport:
    summary = WindowSummary(
        since=datetime(2026, 1, 1, tzinfo=UTC),
        until=datetime(2026, 1, 8, tzinfo=UTC),
        days=7,
        sessions=3,
        spans=10,
        total_tokens=1000,
        total_cost_usd=1.0,
        thin_data=False,
    )
    return OptimizeReport(window=summary, **kwargs)


def test_reclaimable_share_none_without_estimate():
    assert reclaimable_share(SimpleNamespace(), 1000) is None
    assert reclaimable_share(SimpleNamespace(past_overspend_tokens=None), 1000) is None
    assert reclaimable_share(SimpleNamespace(past_overspend_tokens=50), 0) is None


def test_reclaimable_share_clamps_negative():
    assert reclaimable_share(SimpleNamespace(past_overspend_tokens=-10), 1000) == 0.0


def test_rank_orders_by_share_then_name_order():
    report = _report(
        downgrade=SimpleNamespace(past_overspend_tokens=100),
        findings={
            "resend": SimpleNamespace(past_overspend_tokens=400),
            "cache": SimpleNamespace(past_overspend_tokens=400),
            "trim": SimpleNamespace(past_overspend_tokens=50),
        },
    )
    ranked = rank_findings(report, None)
    names = [name for name, _ in ranked]
    # cache is listed before resend in CARD_FINDING_NAMES; equal share keeps that order
    assert names[:3] == ["cache", "resend", "downsize"]
    assert names[3] == "trim"


def test_rank_skips_downsize_when_not_requested():
    report = _report(
        downgrade=SimpleNamespace(past_overspend_tokens=999),
        findings={"cache": SimpleNamespace(past_overspend_tokens=10)},
    )
    ranked = rank_findings(report, ["cache"])
    assert [name for name, _ in ranked] == ["cache"]


def test_rank_includes_empty_downsize_when_requested():
    report = _report(downgrade=None, findings={})
    ranked = rank_findings(report, None)
    assert ranked == [("downsize", None)]


def test_rank_drops_unknown_finding_names():
    report = _report(
        downgrade=None,
        findings={"budget-projection": SimpleNamespace(past_overspend_tokens=500)},
    )
    ranked = rank_findings(report, ["cache"])
    assert ranked == []


def test_relearn_is_unranked_even_with_a_token_figure():
    report = _report(
        downgrade=None,
        findings={"relearn": SimpleNamespace(past_overspend_tokens=800)},
    )
    ranked = rank_findings(report, ["relearn"])
    assert ranked == [("relearn", None)]
    assert "relearn" in ALWAYS_FULL_FINDINGS


def test_card_finding_names_match_cli_renderers():
    from tokenjam.cli.cmd_optimize import _ALWAYS_FULL_FINDINGS, _FINDING_RENDERERS

    assert tuple(_FINDING_RENDERERS) == CARD_FINDING_NAMES
    assert _ALWAYS_FULL_FINDINGS is ALWAYS_FULL_FINDINGS


def test_api_optimize_route_does_not_import_cli():
    source = (
        Path(__file__).resolve().parents[2]
        / "tokenjam"
        / "api"
        / "routes"
        / "optimize.py"
    ).read_text(encoding="utf-8")
    assert "tokenjam.cli" not in source
    assert "rank_findings" in source
