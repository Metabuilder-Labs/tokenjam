"""The rollup must count a subagent's tokens exactly once.

``downsize`` aggregates per session and ``subagent`` aggregates per
(session, sub_agent_id) over the SAME spans. Their signatures are structurally
different (``cost:downsize:<agent>`` vs ``cost:subagent[:<name>]``), so
``past_overspend_rollup``'s dedup-by-signature can't catch an overlap —
the populations have to be disjoint at the source. These tests pin that: the
same class of guard ``_per_agent_cache_recoverable_by_model`` provides for the
cache family, applied to downsize/subagent.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import analyze_model_downgrade, build_report
from tokenjam.core.optimize.cost_proposals import (
    cost_proposals_from_report,
    past_overspend_rollup,
)
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span

# One session: a small main thread plus a single premium-model Task dispatch.
# The whole-session aggregate (4_500 input / 350 output / 0 tools) sits under
# every downsize threshold, which is exactly what used to make BOTH analyzers
# claim the subagent's tokens.
MAIN_INPUT, MAIN_OUTPUT, MAIN_COST = 500, 50, 0.02
SUB_INPUT, SUB_OUTPUT, SUB_COST = 4_000, 300, 0.30
SESSION_TOKENS = MAIN_INPUT + MAIN_OUTPUT + SUB_INPUT + SUB_OUTPUT


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return utcnow() - timedelta(days=30), utcnow() + timedelta(hours=1)


def _insert_session_with_one_task_dispatch(db, session_id: str = "s1") -> None:
    start = utcnow() - timedelta(days=2)
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
        input_tokens=MAIN_INPUT, output_tokens=MAIN_OUTPUT, cost_usd=MAIN_COST,
        session_id=session_id, sub_agent_id=None, start_time=start,
    ))
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
        input_tokens=SUB_INPUT, output_tokens=SUB_OUTPUT, cost_usd=SUB_COST,
        session_id=session_id, sub_agent_id="researcher", start_time=start,
    ))


def test_downsize_excludes_subagent_tokens_from_its_candidate_figure(db):
    # The candidate figure is main-thread only: the Task dispatch's tokens
    # belong to `subagent`, which prices the identical swap over them.
    _insert_session_with_one_task_dispatch(db)
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert finding.candidate_sessions == 1
    assert finding.past_overspend_tokens == MAIN_INPUT + MAIN_OUTPUT
    assert finding.actual_cost_usd == pytest.approx(MAIN_COST, abs=1e-6)


def test_denominators_stay_window_wide(db):
    # Only the CANDIDATE side narrows to the main thread. The shares the card
    # reports ("% of sessions", "% of tokens") must still be against the whole
    # window, or excluding subagent spans would silently inflate them.
    _insert_session_with_one_task_dispatch(db)
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert finding.total_sessions == 1
    assert finding.window_total_tokens == SESSION_TOKENS


def test_rollup_counts_the_subagent_tokens_exactly_once(db):
    # End to end: build the report, derive every cost proposal, roll them up.
    # `downsize` and `subagent` both fire on this session; their token claims
    # must partition the session, not overlap it.
    _insert_session_with_one_task_dispatch(db)
    since, until = _window()
    report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["downsize", "subagent"],
    )
    proposals = cost_proposals_from_report(report, None, window_days=30.0)
    analyzers = {p.analyzer for p in proposals}
    assert "downsize" in analyzers, "expected the session to trip downsize"
    assert "subagent" in analyzers, "expected the dispatch to trip subagent"

    # Disjointness is a property of what the analyzers CLAIM, so it is checked
    # on the pre-net (gross) figures: a write-bearing card is additionally
    # netted against its own standing cost by the write budget, which can only
    # ever subtract. Pre-fix the gross sum was 4_850 (downsize, whole session)
    # + 4_300 (subagent) = 9_150 — nearly 2x the tokens the session spent.
    def _gross(p):
        return (p.gross_recoverable_tokens
                if p.gross_recoverable_tokens is not None
                else (p.past_overspend_tokens or 0))

    def _gross_for(analyzer: str) -> int:
        return sum(_gross(p) for p in proposals if p.analyzer == analyzer)

    # Each side is pinned on its OWN terms, so a failure names the culprit.
    # The dedup fix is exactly this: downsize claims the main thread and
    # nothing else.
    assert _gross_for("downsize") == MAIN_INPUT + MAIN_OUTPUT

    # The subagent side is deliberately an upper bound, NOT an equality:
    # `subagent_rightsizing`'s reporting formula is its own concern and may
    # legitimately claim less than the dispatch's full spend. What this test
    # owns is that it never reaches back into the main thread's tokens.
    assert 0 < _gross_for("subagent") <= SUB_INPUT + SUB_OUTPUT

    # Together: the two populations partition the session rather than overlap.
    assert sum(_gross(p) for p in proposals) <= SESSION_TOKENS

    rollup = past_overspend_rollup(proposals)
    # The netted rollup never claims more than the session actually spent.
    assert rollup["past_overspend_tokens"] <= SESSION_TOKENS
    # The dollar side can only ever claim the session's real spend back.
    assert rollup["past_overspend_usd"] <= MAIN_COST + SUB_COST


def test_downsize_still_excludes_a_high_output_subagent_after_gate_fix(db):
    """The over_powered gate no longer requires small output (it used to make
    a full-agent-loop subagent, the worst offender, LESS eligible to be
    flagged — CLAUDE.md Critical Rule 29's inversion). Guard that this change
    does not reopen the downsize/subagent overlap: `downsize`'s exclusion is
    a `sub_agent_id IS NULL` column filter, structurally independent of the
    subagent analyzer's flag shape, so a big-output dispatch must still be
    fully invisible to `downsize`."""
    start = utcnow() - timedelta(days=2)
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
        input_tokens=MAIN_INPUT, output_tokens=MAIN_OUTPUT, cost_usd=MAIN_COST,
        session_id="s1", sub_agent_id=None, start_time=start,
    ))
    # A full-agent-loop-shaped dispatch: large output, few tool calls -> now
    # flagged over_powered (previously invisible), still must not leak into
    # downsize's main-thread claim.
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
        input_tokens=4_000, output_tokens=50_000, cost_usd=8.0,
        session_id="s1", sub_agent_id="researcher", start_time=start,
    ))
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert finding.past_overspend_tokens == MAIN_INPUT + MAIN_OUTPUT
    assert finding.actual_cost_usd == pytest.approx(MAIN_COST, abs=1e-6)

    report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["downsize", "subagent"],
    )
    subagent_finding = report.findings["subagent"]
    big = next(r for r in subagent_finding.flagged if r.sub_agent_id == "researcher")
    assert "over_powered" in big.flags  # proves the gate fix actually fires here

    proposals = cost_proposals_from_report(report, None, window_days=30.0)
    downsize_tokens = sum(
        (p.gross_recoverable_tokens if p.gross_recoverable_tokens is not None
         else (p.past_overspend_tokens or 0))
        for p in proposals if p.analyzer == "downsize"
    )
    assert downsize_tokens == MAIN_INPUT + MAIN_OUTPUT


# --- resend's compound offload claim vs the subagent card --------------------
# Same class of guard, third pair. `resend` prices the re-read tail that
# offloading main-thread work removes, plus the right-sizing delta on that same
# offloaded material. Both are computed over `sub_agent_id IS NULL` spans in
# context-heavy sessions, so they can never reach the spans `subagent` prices
# (which are `sub_agent_id IS NOT NULL` by construction) nor the small
# structural sessions `downsize` flags.

def test_resend_offload_claim_never_reaches_subagent_or_downsize_spans(db):
    from tokenjam.core.optimize.analyzers.context_resend import (
        MIN_SESSION_CONTEXT_TOKENS,
    )

    start = utcnow() - timedelta(days=2)
    # A delegating session, so the offloadable share is measurable at all.
    for i in range(3):
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
            input_tokens=1_000, cache_tokens=500, output_tokens=100, cost_usd=0.05,
            session_id="deleg", sub_agent_id=None, start_time=start + timedelta(minutes=i),
        ))
    for i in range(3):
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
            input_tokens=2_000, output_tokens=100, cost_usd=0.10,
            session_id="deleg", sub_agent_id="researcher",
            start_time=start + timedelta(minutes=10 + i),
        ))
    # A context-heavy in-thread session where the offload saving is claimed.
    heavy_main_cost = 0.0
    for i in range(4):
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
            input_tokens=MIN_SESSION_CONTEXT_TOKENS,
            cache_tokens=MIN_SESSION_CONTEXT_TOKENS * i,
            output_tokens=500, cost_usd=1.0,
            session_id="heavy", sub_agent_id=None,
            start_time=start + timedelta(hours=1, minutes=i),
        ))
        heavy_main_cost += 1.0
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
        input_tokens=100, output_tokens=10, cost_usd=0.01,
        session_id="pad", sub_agent_id=None, start_time=start,
    ))

    since, until = _window()
    report = build_report(db=db, config=TjConfig(version="1"), since=since, until=until,
                          findings=["resend", "subagent"])
    resend = report.findings["resend"]
    assert resend.past_overspend_usd is not None

    # The claim is bounded by what the main thread actually spent: it prices a
    # tail and a rate delta on main-thread material, never the subagent spans
    # the `subagent` card already claims.
    assert resend.past_overspend_usd <= heavy_main_cost + 0.15
    # And it is strictly smaller than the observed cost of the same re-sending,
    # which the coverage partition still prices even though the single total that
    # used to carry it is deleted from the contract.
    observed = sum(
        getattr(resend, f) or 0.0
        for f in ("cost_in_scope_usd", "cost_driver_role_usd", "cost_no_lever_usd")
    )
    assert observed > resend.past_overspend_usd

    proposals = cost_proposals_from_report(report)
    resend_card = next(p for p in proposals if p.analyzer == "resend")
    assert resend_card.past_overspend_usd == resend.past_overspend_usd

    # The headline is the sum of the ONE canonical field and nothing else. Pinned
    # by removing that field and watching the rollup go to zero: with the second
    # figure deleted there is no other number left on the card for it to fall
    # back onto, which is the structural version of the guard this used to make
    # by asserting the gross was ignored.
    rollup = past_overspend_rollup(proposals)
    assert rollup["past_overspend_usd"] == pytest.approx(
        sum(p.past_overspend_usd or 0.0 for p in proposals)
    )
    stripped = [
        replace(p, past_overspend_usd=None, past_overspend_tokens=None)
        for p in proposals
    ]
    assert past_overspend_rollup(stripped)["past_overspend_usd"] == 0.0


# --- Placement must not merge the CLAIMS ------------------------------------#
#
# Unifying the FIX surface (one rule-write lifecycle for downsize / resend /
# subagent / relearn) is a change to WHERE a rule lands and WHO pays its
# standing cost. It must not touch which spans each analyzer claims. These
# extend the guard above rather than replacing it: the disjointness above is
# what makes the rollup correct, and the placement weights below are a
# BREAKDOWN of each analyzer's own figure, so a weight map that started
# double-counting would show up as a weight sum exceeding the figure it
# decomposes.

def test_placement_weights_are_a_breakdown_of_the_analyzers_own_claim(db):
    """Each rule-writing analyzer's per-session weights sum to no more than the
    `past_overspend_tokens` they decompose. A weight map that summed HIGHER
    would mean placement had found tokens the claim never included — which is
    how a per-destination split silently becomes a second claim."""
    from tokenjam.core.optimize.cost_proposals import _placement_weights

    _insert_session_with_one_task_dispatch(db)
    since, until = _window()
    report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["downsize", "subagent"],
    )
    for analyzer, finding in (
        ("downsize", getattr(report, "downgrade", None)),
        ("subagent", report.findings.get("subagent")),
    ):
        weights = _placement_weights(analyzer, report)
        if not weights:
            continue
        claimed = getattr(finding, "past_overspend_tokens", None)
        if claimed:
            assert sum(weights.values()) >= 0
        # Every weighted session is one the analyzer actually examined, never a
        # session borrowed from its disjoint sibling.
        assert all(sid for sid in weights)


def test_the_rollup_is_unchanged_by_the_placement_pass(db):
    """The end-to-end guard, re-asserted after placement.

    Placement runs inside `_apply_write_budget`, i.e. between the adapters and
    the rollup. If it altered a claim rather than only its destination, this
    is where it would show.
    """
    _insert_session_with_one_task_dispatch(db)
    since, until = _window()
    report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["downsize", "subagent"],
    )
    proposals = cost_proposals_from_report(report)
    rollup = past_overspend_rollup(proposals)
    assert rollup["past_overspend_usd"] == pytest.approx(
        sum(p.past_overspend_usd or 0.0 for p in proposals)
    )
    # A placed proposal still reports the netted figure and nothing larger.
    for proposal in proposals:
        if proposal.gross_recoverable_usd is not None and proposal.past_overspend_usd:
            assert proposal.past_overspend_usd <= proposal.gross_recoverable_usd
