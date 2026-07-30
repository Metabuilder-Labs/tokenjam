"""No inbox row's money may be absent from the headline above it.

`past_overspend_rollup` deduplicates by `signature`, which is correct only while
a signature identifies exactly one rendered card. When two cards that render
separately — separate titles, separate figures, separate apply targets — share
one signature, the rollup keeps whichever it saw first and silently discards the
rest, so the headline understates a total whose parts the user can see listed
underneath it.

The live instance was `downsize`: `build_agent_price_rows` groups candidates by
(agent, provider, model, alt_model), so an agent that ran two over-sized models
produced two rows, while the signature was keyed on the agent alone.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from tokenjam.utils.time_parse import utcnow

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import cost_proposals as cp
from tokenjam.core.optimize.analyzers.downsize_agents import build_agent_price_rows
from tokenjam.core.optimize.types import DowngradeFinding, OptimizeReport, WindowSummary

NOW = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)


def _candidate(session_id, agent_id, model, alt_model):
    return {
        "session_id": session_id, "agent_id": agent_id, "provider": "anthropic",
        "model": model, "alt_model": alt_model,
        "input_tokens": 100_000, "output_tokens": 20_000,
        "cache_tokens": 500_000, "cache_write_tokens": 40_000,
        "started_at": utcnow() - timedelta(days=1),
    }


def _report_with_two_models_on_one_agent():
    """One agent, two over-sized models — two price rows, two cards."""
    rows = build_agent_price_rows(
        [
            _candidate("s1", "svc-a", "claude-opus-4-8", "claude-haiku-4-5"),
            _candidate("s2", "svc-a", "claude-sonnet-4-5", "claude-haiku-4-5"),
        ],
        30.0,
    )
    assert len(rows) == 2, "fixture must produce two rows for one agent"
    finding = DowngradeFinding(
        candidate_sessions=4, total_sessions=10, actual_cost_usd=5.0,
        alternative_cost_usd=2.0, monthly_savings_usd=3.0, percent_of_sessions=40.0,
        examples=[], suggestions={"claude-opus-4-8": "claude-haiku-4-5"},
        past_overspend_usd=3.0, percent_of_tokens=35.0,
        estimate_basis="downsize basis", per_agent=rows,
    )
    window = WindowSummary(
        since=NOW - timedelta(days=30), until=NOW, days=30, sessions=10,
        spans=100, total_tokens=1, total_cost_usd=10.0, thin_data=False,
    )
    return OptimizeReport(window=window, downgrade=finding, findings={})


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(
        version="1",
        storage=StorageConfig(path=str(tmp_path / "t.duckdb")),
        agents={},
    )


def test_two_models_on_one_agent_get_distinct_signatures(cfg):
    proposals = cp.cost_proposals_from_report(
        _report_with_two_models_on_one_agent(), config=cfg,
    )
    downsize = [p for p in proposals if p.analyzer == "downsize"]

    assert len(downsize) == 2
    assert len({p.signature for p in downsize}) == 2
    # The grain that makes them distinct is the one the rows were grouped on.
    for p in downsize:
        assert p.baseline["model"] in p.signature
        assert p.baseline["alt_model"] in p.signature


def test_headline_sums_every_downsize_row_it_renders(cfg):
    proposals = cp.cost_proposals_from_report(
        _report_with_two_models_on_one_agent(), config=cfg,
    )
    downsize = [p for p in proposals if p.analyzer == "downsize"]
    rollup = cp.past_overspend_rollup(proposals)

    rendered = sum(p.past_overspend_usd or 0.0 for p in downsize)
    entry = next(e for e in rollup["by_analyzer"] if e["analyzer"] == "downsize")

    assert rendered > 0
    assert entry["usd"] == pytest.approx(rendered)
    assert entry["count"] == len(downsize)


def test_every_rendered_proposal_carries_a_unique_signature(cfg):
    """The general rule, guarding the whole class rather than one analyzer.

    Any proposal the assembler emits is a row the inbox renders; the rollup
    keeps one row per signature. Two emitted proposals sharing a signature is
    therefore always money rendered but not summed, whichever analyzer does it.
    """
    proposals = cp.cost_proposals_from_report(
        _report_with_two_models_on_one_agent(), config=cfg,
    )
    signatures = [p.signature for p in proposals]

    assert len(signatures) == len(set(signatures)), (
        f"duplicate signature(s) among rendered proposals: "
        f"{sorted({s for s in signatures if signatures.count(s) > 1})}"
    )
