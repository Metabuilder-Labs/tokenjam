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
    # Wide enough to comfortably contain every fixture in this file,
    # including `_seed_thrash_pairs`'s default 15 sessions spaced one hour
    # apart (max offset ~14h10m) — narrower bounds silently truncated the
    # seeded call count below `MIN_CALLS_FOR_ROOT_CAUSE` once the min-calls
    # floor started gating A2/A3 uniformly (previously it gated only A1),
    # which made those tests pass for the wrong reason (an empty/undersized
    # result, not a correctly-classified one).
    return BASE - timedelta(hours=1), BASE + timedelta(hours=24)


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


def _seed_thrash_chain(db, *, agent="agent-a2-chain", n_sessions=4, calls_per_session=6,
                        gap_minutes=10.0):
    """n_sessions sessions, each with calls_per_session consecutive cache-write
    calls (no reads), spaced gap_minutes apart — many write-repeats per burst,
    the shape where switching to a 1-hour TTL actually pays off. n_sessions=4
    (24 total calls) clears MIN_CALLS_FOR_ROOT_CAUSE now that the min-calls
    floor gates A2 (previously A1-only) — 3 sessions (18 calls) fell short."""
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
                          repeats=3, padding_calls=15, miss_cache_write_tokens=0):
    """One session per agent: a bootstrap cache write, then `repeats` rounds
    of [long tool-heavy turn -> miss -> hit]. write_events stays at 1 (well
    under the "attempted regularly" floor) so A2 never fires; A1 never fires
    because the bootstrap call caches.

    `padding_calls` seeds extra cache-hit calls in a SEPARATE session so the
    agent's total call volume clears MIN_CALLS_FOR_ROOT_CAUSE (now that the
    min-calls floor gates A3 too, previously A1-only) without perturbing the
    A3 pairing logic: each padding call has cache_tokens > 0, so it is
    skipped as "not a miss" wherever it's evaluated, and it lives in its own
    session so it is never paired against the real bootstrap/miss/hit calls.

    `miss_cache_write_tokens` controls whether the "miss" call itself paid a
    cache-write cost. 0 (the default) models a genuinely uncached call riding
    along with the pattern -- it must still count toward `miss_count`, but
    must NOT be priced into the dollar estimate."""
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
            input_tokens=3000, cache_tokens=0,
            cache_write_tokens=miss_cache_write_tokens,
            session_id=sid, start_time=t,
        ))
        t += timedelta(seconds=10)
        db.insert_span(make_llm_span(  # the recovered hit
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=200, cache_tokens=1800, cache_write_tokens=0,
            session_id=sid, start_time=t,
        ))
        t += timedelta(seconds=20)

    padding_sid = "a3-padding"
    for i in range(padding_calls):
        db.insert_span(make_llm_span(
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=200, cache_tokens=1800, cache_write_tokens=0,
            session_id=padding_sid, start_time=t + timedelta(seconds=i),
        ))


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
    # Every "miss" call here has cache_write_tokens == 0 (a genuinely
    # uncached call, not a rewritten cache prefix) -- it must still count
    # toward miss_count (asserted above) but must NOT be priced as if the
    # whole input_tokens were a rewritten prefix.
    assert c.estimated_recoverable_usd == 0.0
    assert c.estimated_recoverable_tokens == 0


def test_a3_zero_cache_write_miss_excluded_from_dollar_estimate(db):
    """A3's dollar estimate must only price misses that actually paid a
    cache-write cost. With every miss call carrying cache_write_tokens=0,
    the recoverable estimate must be exactly zero -- even though each miss's
    input_tokens (3000) is well above zero and would inflate the estimate if
    (incorrectly) folded in via `cache_write_tokens or input_tokens`."""
    _seed_lookback_agent(
        db, tool_calls_before_miss=11, repeats=MIN_LOOKBACK_MISS_RECURRENCE,
        miss_cache_write_tokens=0,
    )
    since, until = _window()
    _, _, lookback = _compute_root_cause_candidates(db.conn, since, until, None)
    assert len(lookback) == 1
    c = lookback[0]
    assert c.miss_count == MIN_LOOKBACK_MISS_RECURRENCE
    assert c.estimated_recoverable_usd == 0.0
    assert c.estimated_recoverable_tokens == 0


def test_a3_priced_only_over_misses_with_cache_write(db):
    """Contrast case: when the miss calls DID pay a cache-write cost (a real
    rewritten-prefix miss), the dollar estimate must be positive and priced
    off the actual cache_write_tokens -- confirming the zero-dollar result
    above comes from the cache_write_tokens gate, not from some other break."""
    _seed_lookback_agent(
        db, tool_calls_before_miss=11, repeats=MIN_LOOKBACK_MISS_RECURRENCE,
        miss_cache_write_tokens=1500,
    )
    since, until = _window()
    _, _, lookback = _compute_root_cause_candidates(db.conn, since, until, None)
    assert len(lookback) == 1
    c = lookback[0]
    assert c.miss_count == MIN_LOOKBACK_MISS_RECURRENCE
    assert c.estimated_recoverable_usd is not None
    assert c.estimated_recoverable_usd > 0
    assert c.estimated_recoverable_tokens == 1500 * MIN_LOOKBACK_MISS_RECURRENCE


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
# Sessionless calls must never collapse into one shared pseudo-session
# --------------------------------------------------------------------------- #

def test_sessionless_calls_get_unique_synthetic_session_ids(db):
    """Calls lacking a session id must never collapse into a single shared
    "" pseudo-session -- A2 keys by c.session_id and A3 keys by
    (agent_id, session_id), so unrelated sessionless calls being merged would
    corrupt gap-based TTL-vs-thrash decisions and A3 pairing. Each sessionless
    row must get its own unique synthetic id that can't collide with a real
    session id."""
    from tokenjam.core.optimize.analyzers.cache_efficacy import _fetch_agent_calls

    for i in range(5):
        db.insert_span(make_llm_span(
            agent_id="agent-sessionless", provider="anthropic", model="claude-sonnet-5",
            input_tokens=1000, cache_tokens=0, cache_write_tokens=0,
            session_id=None, start_time=BASE + timedelta(minutes=i),
        ))
    # One call with a real session id, seeded alongside, to confirm the
    # synthetic ids never collide with it.
    db.insert_span(make_llm_span(
        agent_id="agent-sessionless", provider="anthropic", model="claude-sonnet-5",
        input_tokens=1000, cache_tokens=0, cache_write_tokens=0,
        session_id="s-real", start_time=BASE + timedelta(minutes=10),
    ))
    since, until = _window()
    by_agent = _fetch_agent_calls(db.conn, since, until, None)
    calls = by_agent["agent-sessionless"]
    assert len(calls) == 6

    sessionless_ids = [c.session_id for c in calls if c.session_id != "s-real"]
    assert len(sessionless_ids) == 5
    # Every sessionless call gets its own distinct, non-empty synthetic id.
    assert all(sid for sid in sessionless_ids)
    assert len(set(sessionless_ids)) == 5
    # None of the synthetic ids collide with the real session id.
    assert "s-real" not in sessionless_ids


def test_sessionless_calls_never_thrash_paired_via_shared_gap(db):
    """Regression for the shared-"" pseudo-session bug: before the fix, N
    sessionless calls from one agent were grouped under the same session_id
    key, so `_inter_call_gap_minutes` computed real (small) gaps between
    otherwise-unrelated calls -- corrupting A2's TTL-vs-instability gap
    classification. With unique synthetic ids, each sessionless call is its
    own single-call "session", so no gap is ever computed between them."""
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        _fetch_agent_calls,
        _inter_call_gap_minutes,
    )

    for i in range(6):
        db.insert_span(make_llm_span(
            agent_id="agent-sessionless-a2", provider="anthropic", model="claude-sonnet-5",
            input_tokens=3000, cache_tokens=0, cache_write_tokens=5000,
            session_id=None, start_time=BASE + timedelta(seconds=i * 30),
        ))
    since, until = _window()
    by_agent = _fetch_agent_calls(db.conn, since, until, None)
    calls = by_agent["agent-sessionless-a2"]
    assert len(calls) == 6

    # No gaps at all: every sessionless call is alone in its own synthetic
    # session, so there's no "previous call in the same session" to pair
    # against (the old shared-"" behavior would have produced five ~30s gaps
    # here instead).
    assert _inter_call_gap_minutes(calls) == []


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
