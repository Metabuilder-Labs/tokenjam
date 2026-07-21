"""Unit tests for the cache analyzer's root-caused per-agent proposals:
A1 (uncached agent), A2 (cache thrash), A3 (20-block lookback miss).

Mirrors ``test_cache_efficacy.py``'s fixtures/style. Detection positive,
negative, and threshold-edge cases per check, plus the cross-check dedup rule
(A1 beats A2 beats A3 — one agent, at most one card).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import OptimizeConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers.cache_efficacy import (
    MIN_CALLS_FOR_ROOT_CAUSE,
    MIN_LOOKBACK_MISS_RECURRENCE,
    MIN_UNCACHED_MEDIAN_INPUT_TOKENS,
    THRASH_READ_WRITE_RATIO_THRESHOLD,
    _compute_root_cause_candidates,
)
from tokenjam.core.optimize.runner import report_from_dict, report_to_dict
from tests.factories import make_llm_span, make_tool_span

BASE = datetime(2026, 6, 1, tzinfo=timezone.utc)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return BASE - timedelta(hours=1), BASE + timedelta(hours=6)


# --------------------------------------------------------------------------- #
# A1 — uncached agent
# --------------------------------------------------------------------------- #

def _seed_uncached(db, *, agent="agent-a1", calls=20, input_tokens=4000,
                    session_prefix="s"):
    for i in range(calls):
        db.insert_span(make_llm_span(
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=input_tokens, cache_tokens=0, cache_write_tokens=0,
            session_id=f"{session_prefix}-{i // 5}",
            start_time=BASE + timedelta(minutes=i),
        ))


def test_a1_flagged_when_never_cached_and_prefix_large(db):
    _seed_uncached(db, calls=MIN_CALLS_FOR_ROOT_CAUSE, input_tokens=4000)
    since, until = _window()
    uncached, thrash, lookback = _compute_root_cause_candidates(db.conn, since, until, None)
    assert len(uncached) == 1
    c = uncached[0]
    assert c.agent_id == "agent-a1"
    assert c.calls == MIN_CALLS_FOR_ROOT_CAUSE
    assert c.assumed_prefix_tokens > 0
    assert c.estimated_recoverable_usd is not None
    assert c.estimated_recoverable_usd >= 0
    assert "cache_control" in c.cache_control_snippet
    assert not thrash and not lookback


def test_a1_boundary_exactly_at_thresholds_flags(db):
    # Exactly 20 calls, exactly the median-input floor — inclusive boundary.
    _seed_uncached(db, calls=MIN_CALLS_FOR_ROOT_CAUSE,
                    input_tokens=MIN_UNCACHED_MEDIAN_INPUT_TOKENS)
    since, until = _window()
    uncached, _, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert len(uncached) == 1


def test_a1_not_flagged_below_call_volume(db):
    _seed_uncached(db, calls=MIN_CALLS_FOR_ROOT_CAUSE - 1, input_tokens=4000)
    since, until = _window()
    uncached, _, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert uncached == []


def test_min_calls_override_flags_below_default_call_volume(db):
    """`_compute_root_cause_candidates`'s `min_calls` param (what run() threads
    from `[optimize] min_calls_for_root_cause`) surfaces the exact data from
    test_a1_not_flagged_below_call_volume once lowered."""
    _seed_uncached(db, calls=MIN_CALLS_FOR_ROOT_CAUSE - 1, input_tokens=4000)
    since, until = _window()
    uncached, _, _ = _compute_root_cause_candidates(
        db.conn, since, until, None, min_calls=MIN_CALLS_FOR_ROOT_CAUSE - 1,
    )
    assert len(uncached) == 1


def test_run_reads_min_calls_for_root_cause_from_ctx_config(db):
    """The registered run(ctx) entry point reads
    `ctx.config.optimize.min_calls_for_root_cause`."""
    _seed_uncached(db, calls=MIN_CALLS_FOR_ROOT_CAUSE - 1, input_tokens=4000)
    since, until = _window()

    default_report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["cache"],
    )
    assert default_report.findings["cache"].uncached_agents == []

    lowered_config = TjConfig(
        version="1",
        optimize=OptimizeConfig(min_calls_for_root_cause=MIN_CALLS_FOR_ROOT_CAUSE - 1),
    )
    lowered_report = build_report(
        db=db, config=lowered_config, since=since, until=until,
        findings=["cache"],
    )
    lowered_finding = lowered_report.findings["cache"]
    assert len(lowered_finding.uncached_agents) == 1
    assert lowered_finding.min_calls_for_root_cause == MIN_CALLS_FOR_ROOT_CAUSE - 1


def test_a1_not_flagged_when_prefix_too_small(db):
    _seed_uncached(db, calls=MIN_CALLS_FOR_ROOT_CAUSE,
                    input_tokens=MIN_UNCACHED_MEDIAN_INPUT_TOKENS - 1)
    since, until = _window()
    uncached, _, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert uncached == []


def test_a1_not_flagged_when_any_call_already_caches(db):
    _seed_uncached(db, calls=MIN_CALLS_FOR_ROOT_CAUSE, input_tokens=4000)
    # One call in the group DOES cache — A1 requires zero caching on EVERY call.
    db.insert_span(make_llm_span(
        agent_id="agent-a1", provider="anthropic", model="claude-sonnet-5",
        input_tokens=4000, cache_tokens=500, cache_write_tokens=0,
        session_id="s-caching", start_time=BASE + timedelta(minutes=50),
    ))
    since, until = _window()
    uncached, _, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert uncached == []


# --------------------------------------------------------------------------- #
# A2 — cache thrash
# --------------------------------------------------------------------------- #

def _seed_thrash_pairs(db, *, agent="agent-a2", n_sessions=15, gap_minutes=10.0,
                        second_call_reads=False):
    """n_sessions sessions, each with two calls: the first always a cache
    write, the second either a read (cache_tokens>0) or a bare uncached call —
    exactly at the "attempted regularly" boundary (write_events == calls//2)."""
    for i in range(n_sessions):
        sid = f"thrash-{i}"
        t0 = BASE + timedelta(hours=i)
        db.insert_span(make_llm_span(
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=3000, cache_tokens=0, cache_write_tokens=5000,
            session_id=sid, start_time=t0,
        ))
        db.insert_span(make_llm_span(
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=3000,
            cache_tokens=5000 if second_call_reads else 0,
            cache_write_tokens=0,
            session_id=sid, start_time=t0 + timedelta(minutes=gap_minutes),
        ))


def test_a2_ttl_cause_when_gap_over_five_minutes(db):
    _seed_thrash_pairs(db, gap_minutes=10.0)
    since, until = _window()
    _, thrash, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert len(thrash) == 1
    c = thrash[0]
    assert c.agent_id == "agent-a2"
    assert c.cause == "ttl"
    assert c.inter_call_gap_p50_minutes == pytest.approx(10.0, abs=0.1)
    assert c.read_write_ratio < THRASH_READ_WRITE_RATIO_THRESHOLD
    assert "1h" in c.cache_control_snippet or "cache_control" in c.cache_control_snippet


def test_a2_instability_cause_when_gap_under_five_minutes(db):
    _seed_thrash_pairs(db, gap_minutes=2.0)
    since, until = _window()
    _, thrash, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert len(thrash) == 1
    c = thrash[0]
    assert c.cause == "instability"
    assert c.ttl_worth_it is None
    assert "timestamp" in c.cache_control_snippet or "invalidator" in c.cache_control_snippet.lower()
    # The instability checklist IS an actionable fix (fix your prompt-assembly
    # code), so the wasted-spend headline stays populated for this variant.
    assert c.estimated_recoverable_usd is not None


def _seed_thrash_chain(db, *, agent="agent-a2-chain", n_sessions=3, calls_per_session=6,
                        gap_minutes=10.0):
    """n_sessions sessions, each with calls_per_session consecutive cache-write
    calls (no reads), spaced gap_minutes apart — many write-repeats per burst,
    the shape where switching to a 1-hour TTL actually pays off."""
    for i in range(n_sessions):
        sid = f"chain-{i}"
        t0 = BASE + timedelta(hours=i)
        for j in range(calls_per_session):
            db.insert_span(make_llm_span(
                agent_id=agent, provider="anthropic", model="claude-sonnet-5",
                input_tokens=3000, cache_tokens=0, cache_write_tokens=5000,
                session_id=sid, start_time=t0 + timedelta(minutes=gap_minutes * j),
            ))


def test_a2_ttl_worth_it_carries_positive_recoverable_usd(db):
    _seed_thrash_chain(db)
    since, until = _window()
    _, thrash, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert len(thrash) == 1
    c = thrash[0]
    assert c.cause == "ttl"
    assert c.ttl_breakeven_usd is not None
    assert c.ttl_breakeven_usd > 0
    assert c.ttl_worth_it is True
    # Rollup contract: when the recommended fix DOES recover money, the
    # headline figure is populated (and positive) so it counts.
    assert c.estimated_recoverable_usd is not None
    assert c.estimated_recoverable_usd > 0


def test_a2_ttl_breakeven_can_be_negative(db):
    """Sparse per-burst reuse (one write per session, the second call never
    reads back) means the 1-hour TTL's write premium doesn't clear — the
    honest arithmetic must come out negative, not oversell."""
    _seed_thrash_pairs(db, n_sessions=15, gap_minutes=10.0, second_call_reads=False)
    since, until = _window()
    _, thrash, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert len(thrash) == 1
    c = thrash[0]
    assert c.cause == "ttl"
    assert c.ttl_breakeven_usd is not None
    assert c.ttl_breakeven_usd < 0
    assert c.ttl_worth_it is False
    # Rollup contract: the "not worth it" variant must not carry a positive
    # headline figure — its own recommended fix (TTL switch) doesn't recover
    # anything, so estimated_recoverable_usd is excluded (None), never a
    # positive number a downstream rollup would sum.
    assert c.estimated_recoverable_usd is None


def test_a2_not_flagged_when_ratio_at_or_above_threshold(db):
    # Reads roughly match writes -> ratio ~1.0, not below threshold.
    _seed_thrash_pairs(db, n_sessions=15, gap_minutes=10.0, second_call_reads=True)
    since, until = _window()
    _, thrash, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert thrash == []


def test_a2_not_flagged_when_caching_not_attempted_regularly(db):
    # Only ONE session out of many has any cache_write at all.
    agent = "agent-a2-sparse"
    for i in range(20):
        db.insert_span(make_llm_span(
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=3000, cache_tokens=0, cache_write_tokens=0,
            session_id=f"s-{i}", start_time=BASE + timedelta(minutes=i),
        ))
    db.insert_span(make_llm_span(
        agent_id=agent, provider="anthropic", model="claude-sonnet-5",
        input_tokens=3000, cache_tokens=0, cache_write_tokens=5000,
        session_id="s-write", start_time=BASE + timedelta(minutes=100),
    ))
    since, until = _window()
    _, thrash, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert thrash == []


# --------------------------------------------------------------------------- #
# A3 — 20-block lookback miss
# --------------------------------------------------------------------------- #

def _seed_lookback_agent(db, *, agent="agent-a3", tool_calls_before_miss=11,
                          repeats=3):
    """One session per agent: a bootstrap cache write, then `repeats` rounds
    of [long tool-heavy turn -> miss -> hit]. write_events stays at 1 (well
    under the "attempted regularly" floor) so A2 never fires; A1 never fires
    because the bootstrap call caches."""
    sid = "a3-session"
    t = BASE
    db.insert_span(make_llm_span(
        agent_id=agent, provider="anthropic", model="claude-sonnet-5",
        input_tokens=3000, cache_tokens=0, cache_write_tokens=2000,
        session_id=sid, start_time=t,
    ))
    t += timedelta(seconds=10)
    for _ in range(repeats):
        for j in range(tool_calls_before_miss):
            db.insert_span(make_tool_span(
                agent_id=agent, tool_name="Read", session_id=sid,
                start_time=t + timedelta(seconds=j),
            ))
        t += timedelta(seconds=20)
        db.insert_span(make_llm_span(  # the miss
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=3000, cache_tokens=0, cache_write_tokens=0,
            session_id=sid, start_time=t,
        ))
        t += timedelta(seconds=10)
        db.insert_span(make_llm_span(  # the recovered hit
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=200, cache_tokens=1800, cache_write_tokens=0,
            session_id=sid, start_time=t,
        ))
        t += timedelta(seconds=20)


def test_a3_flagged_on_recurring_post_long_turn_misses(db):
    _seed_lookback_agent(db, tool_calls_before_miss=11, repeats=MIN_LOOKBACK_MISS_RECURRENCE)
    since, until = _window()
    uncached, thrash, lookback = _compute_root_cause_candidates(db.conn, since, until, None)
    assert not uncached and not thrash
    assert len(lookback) == 1
    c = lookback[0]
    assert c.agent_id == "agent-a3"
    assert c.miss_count == MIN_LOOKBACK_MISS_RECURRENCE
    assert c.avg_prior_turn_blocks > 20
    assert "breakpoint" in c.cache_control_snippet.lower()


def test_a3_not_flagged_below_block_limit(db):
    # Only 5 tool calls before each miss -> 10 estimated blocks, under the limit.
    _seed_lookback_agent(db, tool_calls_before_miss=5, repeats=MIN_LOOKBACK_MISS_RECURRENCE)
    since, until = _window()
    _, _, lookback = _compute_root_cause_candidates(db.conn, since, until, None)
    assert lookback == []


def test_a3_not_flagged_below_recurrence_floor(db):
    _seed_lookback_agent(db, tool_calls_before_miss=11, repeats=MIN_LOOKBACK_MISS_RECURRENCE - 1)
    since, until = _window()
    _, _, lookback = _compute_root_cause_candidates(db.conn, since, until, None)
    assert lookback == []


# --------------------------------------------------------------------------- #
# Dedup: A1 beats A2 beats A3, one card per agent
# --------------------------------------------------------------------------- #

def test_uncached_agent_never_also_appears_as_thrash_or_lookback(db):
    _seed_uncached(db, agent="agent-dedup", calls=MIN_CALLS_FOR_ROOT_CAUSE, input_tokens=4000)
    since, until = _window()
    uncached, thrash, lookback = _compute_root_cause_candidates(db.conn, since, until, None)
    agents_seen = {c.agent_id for c in uncached} | {c.agent_id for c in thrash} | {
        c.agent_id for c in lookback
    }
    assert agents_seen == {"agent-dedup"}
    assert any(c.agent_id == "agent-dedup" for c in uncached)
    assert not any(c.agent_id == "agent-dedup" for c in thrash)
    assert not any(c.agent_id == "agent-dedup" for c in lookback)


def test_thrash_shaped_agent_never_also_appears_as_lookback(db):
    """An agent whose data satisfies A2 (regular writes, low read:write ratio)
    ALSO has long, tool-heavy turns shaped like an A3 lookback miss. A2's
    priority means A3 is never even evaluated for this agent — one card, not
    two, and never across BOTH the ``cache`` and ``cache_thrash`` analyzer
    values (the two ``CostProposal.analyzer`` strings this check family uses)."""
    agent = "agent-both"
    for i in range(15):
        sid = f"both-{i}"
        t0 = BASE + timedelta(hours=i)
        db.insert_span(make_llm_span(
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=3000, cache_tokens=0, cache_write_tokens=5000,
            session_id=sid, start_time=t0,
        ))
        # A3-shaped structure layered on top: >10 tool calls before the miss.
        for j in range(11):
            db.insert_span(make_tool_span(
                agent_id=agent, tool_name="Read", session_id=sid,
                start_time=t0 + timedelta(minutes=5, seconds=j),
            ))
        db.insert_span(make_llm_span(
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=3000, cache_tokens=0, cache_write_tokens=0,
            session_id=sid, start_time=t0 + timedelta(minutes=10),
        ))
    since, until = _window()
    uncached, thrash, lookback = _compute_root_cause_candidates(db.conn, since, until, None)
    assert not any(c.agent_id == agent for c in uncached)
    assert any(c.agent_id == agent for c in thrash)
    assert not any(c.agent_id == agent for c in lookback)

    from tokenjam.core.optimize.analyzers.cache_efficacy import CacheEfficacyFinding
    from tokenjam.core.optimize.cost_proposals import cost_proposals_from_report
    from tokenjam.core.optimize.types import OptimizeReport, WindowSummary

    report = OptimizeReport(
        window=WindowSummary(since=since, until=until, days=1, sessions=15, spans=1,
                              total_tokens=1, total_cost_usd=1.0, thin_data=False),
        findings={"cache": CacheEfficacyFinding(
            uncached_agents=uncached, thrash_agents=thrash, lookback_miss_agents=lookback,
        )},
    )
    props_for_agent = [
        p for p in cost_proposals_from_report(report)
        if p.analyzer in ("cache", "cache_thrash") and p.baseline.get("agent_id") == agent
    ]
    assert len(props_for_agent) == 1
    assert props_for_agent[0].analyzer == "cache_thrash"


# --------------------------------------------------------------------------- #
# report_to_dict / report_from_dict round-trip (the daemon -> CLI JSON path)
# --------------------------------------------------------------------------- #

def test_root_cause_candidates_survive_report_dict_round_trip(db):
    _seed_uncached(db, agent="agent-rt", calls=MIN_CALLS_FOR_ROOT_CAUSE, input_tokens=4000)
    since, until = _window()
    report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["cache"],
    )
    finding = report.findings["cache"]
    assert len(finding.uncached_agents) == 1

    rebuilt = report_from_dict(report_to_dict(report))
    rebuilt_finding = rebuilt.findings["cache"]
    assert len(rebuilt_finding.uncached_agents) == 1
    assert rebuilt_finding.uncached_agents[0].agent_id == "agent-rt"
    assert (
        rebuilt_finding.uncached_agents[0].estimated_recoverable_usd
        == finding.uncached_agents[0].estimated_recoverable_usd
    )
    assert rebuilt_finding.thrash_agents == []
    assert rebuilt_finding.lookback_miss_agents == []
