"""Unit tests for the subagent right-sizing analyzer."""
from __future__ import annotations

from datetime import timedelta

import pytest

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


def _ctx(db: InMemoryBackend, window_cost_usd: float, config=None):
    now = utcnow()
    since = now - timedelta(days=1)
    until = now + timedelta(minutes=5)
    summary = WindowSummary(
        since=since, until=until, days=1.0, sessions=1, spans=0,
        total_tokens=0, total_cost_usd=window_cost_usd, thin_data=False,
    )
    ctx = AnalyzerContext(
        conn=db.conn, config=config, since=since, until=until, agent_id=None,
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


def test_config_lowers_noise_floor_flags_previously_hidden_subagent():
    """An over_powered-shaped subagent costing less than the default
    MIN_FLAG_COST_USD is unflagged; lowering [optimize] min_flag_cost_usd
    (threaded through ctx.config) flags the identical span."""
    from tokenjam.core.config import OptimizeConfig, TjConfig
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
        MIN_FLAG_COST_USD,
    )

    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=0.03)
        # Opus-powered, tiny output, no tools -> over_powered shape, but
        # costs less than the default noise floor.
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", input_tokens=1000, output_tokens=100,
            cost_usd=0.03, session_id="s1", sub_agent_id="agentC", start_time=now,
        ))

        run_subagent(ctx)
        default_finding = ctx.report.findings["subagent"]
        assert default_finding.flagged == []
        assert default_finding.min_flag_cost_usd == MIN_FLAG_COST_USD

        lowered_ctx, _ = _ctx(
            db, window_cost_usd=0.03,
            config=TjConfig(version="1", optimize=OptimizeConfig(min_flag_cost_usd=0.01)),
        )
        run_subagent(lowered_ctx)
        lowered_finding = lowered_ctx.report.findings["subagent"]
        assert len(lowered_finding.flagged) == 1
        assert lowered_finding.flagged[0].sub_agent_id == "agentC"
        assert lowered_finding.min_flag_cost_usd == 0.01
    finally:
        db.close()


def test_flags_over_powered_fable_subagent():
    """A Fable-powered subagent (the tier above Opus) doing trivial work must be
    flagged over_powered — before the shared predicate, only "opus" matched, so
    Fable spawns were invisible to right-sizing."""
    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=2.0)
        # Fable subagent: premium model, tiny output, few tools -> over_powered.
        db.insert_span(make_llm_span(
            model="claude-fable-5", input_tokens=4_000, output_tokens=150,
            cost_usd=0.40, session_id="s1", sub_agent_id="fableA", start_time=now,
        ))
        run_subagent(ctx)
        f = ctx.report.findings["subagent"]

        assert len(f.flagged) == 1
        a = f.flagged[0]
        assert a.sub_agent_id == "fableA"
        assert a.model == "claude-fable-5"
        assert "over_powered" in a.flags
    finally:
        db.close()


def test_over_powered_subagent_carries_quantified_estimate():
    """An over_powered subagent must fill estimated_recoverable_* with the
    model-swap delta (premium cost − cheaper same-family cost over the same
    tokens), so it can compete for a ranked slot (#101)."""
    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=1.0)
        # Opus subagent, big context, tiny output -> over_powered (+over_provisioned).
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", provider="anthropic",
            input_tokens=80_000, output_tokens=100,
            cost_usd=0.60, session_id="s1", sub_agent_id="agentA", start_time=now,
        ))
        run_subagent(ctx)
        f = ctx.report.findings["subagent"]

        # Swap target for opus subagents is claude-sonnet-5 (one tier down,
        # not model_downgrade's opus->haiku two-tier jump). Priced over the
        # SAME tokens:
        #   input  80_000 @ $2.00/Mtok  = 0.16
        #   output    100 @ $10.00/Mtok = 0.001
        # alt_cost = 0.161; delta = 0.60 − 0.161 = 0.439.
        assert f.past_overspend_usd == pytest.approx(0.439, abs=1e-6)
        # Recoverable tokens = the quota sitting in the priced over_powered rows.
        assert f.past_overspend_tokens == 80_100
        assert f.estimate_confidence == "heuristic"
        assert "review" in f.estimate_basis.lower()

        # CLAUDE.md Critical Rule 28: past_overspend_usd / past_overspend_tokens
        # must land inside a real price band (never divide two figures that
        # answer different questions).
        from tokenjam.core.pricing import get_rates
        cheapest = get_rates("anthropic", "claude-haiku-4-5")
        priciest = get_rates("anthropic", "claude-fable-5")
        implied_rate = f.past_overspend_usd / f.past_overspend_tokens * 1_000_000
        assert cheapest.cache_read_per_mtok <= implied_rate <= priciest.input_per_mtok
    finally:
        db.close()


def test_over_powered_estimate_prices_cache_write_on_both_sides():
    """`_alt_cost_for_row` must price cache-write tokens on the ALTERNATIVE
    side too, not just the actual side (`cost_usd` already bills cache-write
    on the original model). This is the same asymmetry filed against
    `model_downgrade._alt_unit_cost` — check it does not recur here.
    Subagents are heavily cache-write-bearing (Task dispatch primes a fresh
    cache), so a dropped cache-write class on the alt side would inflate this
    card's numbers specifically."""
    from tokenjam.core.pricing import get_rates

    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=100.0)
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", provider="anthropic",
            input_tokens=1_000, output_tokens=100, cache_write_tokens=50_000,
            cost_usd=1.0, session_id="s1", sub_agent_id="agentCW", start_time=now,
        ))
        run_subagent(ctx)
        f = ctx.report.findings["subagent"]

        rates = get_rates("anthropic", "claude-sonnet-5")
        assert rates is not None
        correct_alt_cost = (
            1_000 / 1e6 * rates.input_per_mtok
            + 100 / 1e6 * rates.output_per_mtok
            + 50_000 / 1e6 * rates.cache_write_per_mtok
        )
        broken_alt_cost = (  # what it would be if cache-write were dropped
            1_000 / 1e6 * rates.input_per_mtok
            + 100 / 1e6 * rates.output_per_mtok
        )
        correct_delta = 1.0 - correct_alt_cost
        broken_delta = 1.0 - broken_alt_cost
        assert correct_delta < broken_delta
        assert f.past_overspend_usd == pytest.approx(correct_delta, abs=1e-6)
    finally:
        db.close()


def test_over_powered_flags_high_output_full_agent_loop_subagent():
    """The over_powered gate used to require output_tokens < 2_000 AND
    tool_calls <= 5, which made a Task subagent that ran as a full agent loop
    (many LLM calls, large output — the shape CLAUDE.md Critical Rule 29
    warns compounds with session length) LESS eligible to be flagged, not
    more. Measured on a real corpus, only 6.5% of premium-tier subagent
    spend cleared both of the old clauses. A premium-model subagent must be
    flagged regardless of how much output it produced or how many tool calls
    it made."""
    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=100.0)
        # Opus subagent: big output (well over the old 2_000 floor) AND many
        # tool calls (well over the old 5-call ceiling) -> must still be
        # over_powered under the new gate.
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", provider="anthropic",
            input_tokens=10_000, output_tokens=237_813, tool_name="Read",
            cost_usd=81.01, session_id="s1", sub_agent_id="agentBig", start_time=now,
        ))
        for i in range(50):  # simulate a many-tool-call agent loop
            db.insert_span(make_llm_span(
                model="claude-opus-4-8", provider="anthropic",
                input_tokens=100, output_tokens=10, tool_name="Read",
                cost_usd=0.001, session_id="s1", sub_agent_id="agentBig",
                start_time=now,
            ))
        run_subagent(ctx)
        f = ctx.report.findings["subagent"]

        assert len(f.flagged) == 1
        a = f.flagged[0]
        assert a.sub_agent_id == "agentBig"
        assert a.tool_calls > 5
        assert "over_powered" in a.flags
        assert f.past_overspend_usd is not None and f.past_overspend_usd > 0
    finally:
        db.close()


def test_over_powered_swap_target_is_sonnet_5_not_model_downgrades_haiku():
    """The subagent analyzer must price its own explicit one-tier-down swap
    (claude-sonnet-5), never silently inherit model_downgrade.lookup_downgrade's
    opus->haiku two-tier jump (that ladder is tuned for a different
    heuristic — the whole-session premium quota audit)."""
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
        _subagent_downgrade_target,
    )
    from tokenjam.core.optimize.analyzers.model_downgrade import lookup_downgrade

    assert _subagent_downgrade_target("anthropic", "claude-opus-4-8") == "claude-sonnet-5"
    # Sanity: this really does differ from the shared ladder's target.
    assert lookup_downgrade("anthropic", "claude-opus-4-8") == "claude-haiku-4-5"


def test_over_provisioned_only_subagent_has_no_estimate():
    """A subagent flagged only over_provisioned (a NON-premium model handed a
    large context) contributes nothing to the estimate when its dispatch
    cohort (same calling agent + model) is too small to have a meaningful
    median baseline — one subagent is a cohort of one, well under
    MIN_COHORT_SESSIONS — so the finding stays unranked (estimate None),
    honestly, rather than inventing a baseline (#101). See
    test_over_provisioned_estimate_prices_context_excess_over_cohort_median
    for the case where a real cohort baseline exists."""
    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=1.0)
        # Sonnet (not premium): big context, tiny output -> over_provisioned only.
        db.insert_span(make_llm_span(
            model="claude-sonnet-4-6", provider="anthropic",
            input_tokens=80_000, output_tokens=100,
            cost_usd=0.30, session_id="s1", sub_agent_id="agentS", start_time=now,
        ))
        run_subagent(ctx)
        f = ctx.report.findings["subagent"]

        assert f.flagged[0].flags == ["over_provisioned"]
        assert f.past_overspend_usd is None
        assert f.past_overspend_tokens is None
    finally:
        db.close()


def test_over_provisioned_estimate_prices_context_excess_over_cohort_median():
    """With a real dispatch cohort (>= MIN_COHORT_SESSIONS same-agent,
    same-model subagents), an over_provisioned outlier's context excess over
    the cohort's own median is priced at the cache-read rate (context arrives
    overwhelmingly as cache reads) -- never against zero context, never a
    made-up target size."""
    from tokenjam.core.pricing import get_rates

    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=1.0)
        # 4 ordinary same-shape dispatches: modest context, well under the
        # over_provisioned threshold -> form the cohort's baseline.
        for i in range(4):
            db.insert_span(make_llm_span(
                model="claude-sonnet-4-6", provider="anthropic",
                input_tokens=10_000, output_tokens=500, cost_usd=0.05,
                session_id=f"s-peer-{i}", sub_agent_id=f"peer{i}", start_time=now,
            ))
        # The outlier: same agent + model, huge context, tiny output ->
        # over_provisioned, with 4 like-shaped peers to baseline against.
        db.insert_span(make_llm_span(
            model="claude-sonnet-4-6", provider="anthropic",
            input_tokens=80_000, output_tokens=100, cost_usd=0.30,
            session_id="s-outlier", sub_agent_id="outlier", start_time=now,
        ))

        run_subagent(ctx)
        f = ctx.report.findings["subagent"]

        outlier = next(r for r in f.flagged if r.sub_agent_id == "outlier")
        assert outlier.flags == ["over_provisioned"]

        # Cohort contexts: [10_000, 10_000, 10_000, 10_000, 80_000] -> median
        # 10_000. Excess = 80_000 - 10_000 = 70_000, priced at the cache-read
        # rate (never the fresh-input rate).
        rates = get_rates("anthropic", "claude-sonnet-4-6")
        expected_tokens = 70_000
        expected_usd = expected_tokens / 1_000_000 * rates.cache_read_per_mtok

        assert f.past_overspend_tokens == expected_tokens
        assert f.past_overspend_usd == pytest.approx(expected_usd, abs=1e-9)
        assert "cohort" in f.estimate_basis.lower()
    finally:
        db.close()


def test_over_powered_estimate_ranks_in_numbered_slot():
    """A CC-heavy window with significant over_powered subagent spend must earn a
    numbered (major) slot in the ranked report, not fall to the unranked tail —
    the exact regression #101 fixes."""
    from tokenjam.cli.cmd_optimize import (
        DE_MINIMIS_SHARE,
        _rank_findings,
        _reclaimable_share,
    )

    db = InMemoryBackend()
    try:
        ctx, now = _ctx(db, window_cost_usd=2.0)
        # The window's token total (drives the reclaimable-share ranking).
        ctx.report.window.total_tokens = 200_000
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", provider="anthropic",
            input_tokens=80_000, output_tokens=100,
            cost_usd=0.60, session_id="s1", sub_agent_id="agentA", start_time=now,
        ))
        run_subagent(ctx)
        f = ctx.report.findings["subagent"]

        share = _reclaimable_share(f, ctx.report.window.total_tokens)
        assert share is not None                      # now quantified, not unranked
        assert share >= DE_MINIMIS_SHARE              # 80_100 / 200_000 = 0.40 -> major

        ranked = _rank_findings(ctx.report, requested=None)
        major = [name for name, s in ranked if s is not None and s >= DE_MINIMIS_SHARE]
        assert "subagent" in major
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
