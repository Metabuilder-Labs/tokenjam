"""Unit tests for the subagent right-sizing analyzer."""
from __future__ import annotations

from datetime import timedelta

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize.analyzers.subagent_rightsizing import run as run_subagent
from tokenjam.core.optimize.runner import report_from_dict, report_to_dict
from tokenjam.core.optimize.types import (
    AnalyzerContext,
    OptimizeReport,
    WindowSummary,
)
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span


def _ctx(db: InMemoryBackend, window_cost_usd: float):
    now = utcnow()
    since = now - timedelta(days=1)
    until = now + timedelta(minutes=5)
    summary = WindowSummary(
        since=since, until=until, days=1.0, sessions=1, spans=0,
        total_tokens=0, total_cost_usd=window_cost_usd, thin_data=False,
    )
    ctx = AnalyzerContext(
        conn=db.conn, config=None, since=since, until=until, agent_id=None,
        window_days=1.0, summary=summary, report=OptimizeReport(window=summary),
    )
    return ctx, now


def test_flags_over_powered_and_over_provisioned_subagents():
    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=1.62)
        # Main thread — NOT a subagent, must be excluded entirely.
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", input_tokens=5000, output_tokens=5000,
            cost_usd=1.0, session_id="s1", sub_agent_id=None, start_time=now,
        ))
        # Subagent A: Opus, huge context, tiny output, no tools -> both flags.
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", input_tokens=80_000, output_tokens=100,
            cost_usd=0.60, session_id="s1", sub_agent_id="agentA", start_time=now,
        ))
        # Subagent B: Haiku, modest, cheap -> below the noise floor, unflagged.
        db.insert_span(make_llm_span(
            model="claude-haiku-4-5", input_tokens=2000, output_tokens=3000,
            cost_usd=0.02, session_id="s1", sub_agent_id="agentB", start_time=now,
        ))

        run_subagent(ctx)
        f = ctx.report.findings["subagent"]

        assert f.total_subagents == 2          # A + B; main excluded
        assert f.sessions_with_subagents == 1
        assert abs(f.subagent_cost_usd - 0.62) < 1e-6
        assert f.window_cost_usd == 1.62
        assert abs(f.percent_of_cost - (0.62 / 1.62)) < 1e-3

        # Only A is a candidate, flagged on both axes.
        assert len(f.flagged) == 1
        a = f.flagged[0]
        assert a.sub_agent_id == "agentA"
        assert "over_powered" in a.flags
        assert "over_provisioned" in a.flags
        assert abs(f.flagged_cost_usd - 0.60) < 1e-6

        # B is present in the breakdown but carries no flags.
        b = next(r for r in f.rows if r.sub_agent_id == "agentB")
        assert b.flags == []
    finally:
        db.close()


def test_no_finding_when_no_subagents():
    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=1.0)
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", input_tokens=1000, output_tokens=200,
            cost_usd=1.0, session_id="s1", sub_agent_id=None, start_time=now,
        ))
        run_subagent(ctx)
        assert "subagent" not in ctx.report.findings
    finally:
        db.close()


def test_finding_survives_dict_round_trip():
    """The MCP / REST path serialises the report to a dict and back; the
    subagent finding (incl. per-row flags) must reconstruct."""
    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=0.60)
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", input_tokens=80_000, output_tokens=100,
            cost_usd=0.60, session_id="s1", sub_agent_id="agentA", start_time=now,
        ))
        run_subagent(ctx)

        restored = report_from_dict(report_to_dict(ctx.report))
        f = restored.findings["subagent"]
        assert f.total_subagents == 1
        assert f.flagged[0].sub_agent_id == "agentA"
        assert "over_powered" in f.flagged[0].flags
    finally:
        db.close()
