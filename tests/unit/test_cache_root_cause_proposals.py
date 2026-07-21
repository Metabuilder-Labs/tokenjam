"""Unit tests for wiring the cache analyzer's root-caused per-agent findings
(A1 uncached / A2 thrash / A3 lookback miss) into Review-inbox proposals.

Mirrors ``test_cost_proposals.py``'s fixtures/style: an ``InMemoryBackend``,
storage under ``tmp_path``, nothing touching a real ``~/.tj``.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers.cache_efficacy import (
    CacheEfficacyFinding,
    LookbackMissCandidate,
    ThrashAgentCandidate,
    UncachedAgentCandidate,
    _compute_root_cause_candidates,
)
from tokenjam.core.optimize.cost_proposals import (
    _cache_thrash_to_proposals,
    _cache_to_proposals,
    _per_agent_cache_recoverable_by_model,
    cost_proposals_from_report,
)
from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
from tests.factories import make_llm_span

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
MARKER = NOW - timedelta(days=5)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return WindowSummary(since=MARKER, until=NOW, days=5, sessions=10, spans=100,
                          total_tokens=1, total_cost_usd=5.0, thin_data=False)


def _uncached_candidate(agent_id="svc-uncached"):
    return UncachedAgentCandidate(
        agent_id=agent_id, provider="anthropic", model="claude-sonnet-5",
        calls=25, sessions=5, assumed_prefix_tokens=4000,
        cache_control_snippet='{"cache_control": {"type": "ephemeral"}}',
        estimated_recoverable_usd=1.5, estimated_recoverable_tokens=90000,
        estimate_basis="p25 prefix basis",
    )


def _thrash_candidate(agent_id="svc-thrash", cause="ttl", ttl_worth_it=True,
                       ttl_breakeven_usd=0.4):
    return ThrashAgentCandidate(
        agent_id=agent_id, provider="anthropic", model="claude-sonnet-5",
        calls=30, cache_write_tokens=50000, cache_read_tokens=10000,
        read_write_ratio=0.2, cause=cause, inter_call_gap_p50_minutes=12.0,
        ttl_worth_it=ttl_worth_it, ttl_breakeven_usd=ttl_breakeven_usd,
        cache_control_snippet="checklist or ttl snippet",
        estimated_recoverable_usd=0.6, estimate_basis="thrash basis",
    )


def _lookback_candidate(agent_id="svc-lookback"):
    return LookbackMissCandidate(
        agent_id=agent_id, provider="anthropic", model="claude-sonnet-5",
        miss_count=4, avg_prior_turn_blocks=28.0,
        cache_control_snippet="add an intermediate breakpoint",
        estimated_recoverable_usd=0.3, estimated_recoverable_tokens=12000,
        estimate_basis="lookback basis",
    )


def _report(**finding_kwargs):
    cache = CacheEfficacyFinding(**finding_kwargs)
    return OptimizeReport(window=_window(), findings={"cache": cache})


# --- Adapter: A1 uncached -----------------------------------------------------

def test_uncached_adapter_produces_advise_only_cost_card():
    report = _report(uncached_agents=[_uncached_candidate()])
    props = [p for p in cost_proposals_from_report(report) if p.analyzer == "cache"
             and p.signature.startswith("cost:cache-uncached:")]
    assert len(props) == 1
    p = props[0]
    assert p.kind == "cost"
    assert p.advise_only is True
    assert p.signature == "cost:cache-uncached:svc-uncached"
    assert p.target_key == {"agent_id": "svc-uncached", "provider": "anthropic",
                             "model": "claude-sonnet-5"}
    assert p.agent_id == "svc-uncached"
    assert p.estimated_recoverable_usd == 1.5
    assert "cache_control" in p.suggestion
    assert "25 calls" in p.evidence


# --- Adapter: A2 thrash --------------------------------------------------------

def test_thrash_adapter_ttl_worth_it_card():
    report = _report(thrash_agents=[_thrash_candidate(cause="ttl", ttl_worth_it=True)])
    props = [p for p in cost_proposals_from_report(report) if p.analyzer == "cache_thrash"]
    assert len(props) == 1
    p = props[0]
    assert p.signature == "cost:cache-thrash:svc-thrash"
    assert "pay off" in p.advise_text
    assert "not worth it" not in p.advise_text


def test_thrash_adapter_ttl_not_worth_it_card_says_so_verbatim():
    """Acceptance criterion: when the honest break-even arithmetic is negative,
    the card must say the fix isn't worth it, never oversell."""
    report = _report(thrash_agents=[
        _thrash_candidate(cause="ttl", ttl_worth_it=False, ttl_breakeven_usd=-0.2),
    ])
    props = [p for p in cost_proposals_from_report(report) if p.analyzer == "cache_thrash"]
    assert len(props) == 1
    assert "caching not worth it at this cadence" in props[0].advise_text


def test_thrash_adapter_not_worth_it_excludes_recoverable_usd_end_to_end(db):
    """Rollup contract (Component E's `estimated_recoverable_rollup` sums
    ``CostProposal.estimated_recoverable_usd`` with no analyzer allowlist):
    when the TTL variant's honest break-even is negative, the real analyzer ->
    adapter pipeline must not hand the rollup a positive figure to sum."""
    # Sparse per-burst reuse: one write per session, second call never reads
    # back -> negative TTL break-even (see test_cache_root_cause.py's
    # equivalent analyzer-level case for the arithmetic).
    for i in range(15):
        sid = f"thrash-{i}"
        t0 = MARKER + timedelta(hours=i)
        db.insert_span(make_llm_span(
            agent_id="svc-thrash-negative", provider="anthropic", model="claude-sonnet-5",
            input_tokens=3000, cache_tokens=0, cache_write_tokens=5000,
            session_id=sid, start_time=t0,
        ))
        db.insert_span(make_llm_span(
            agent_id="svc-thrash-negative", provider="anthropic", model="claude-sonnet-5",
            input_tokens=3000, cache_tokens=0, cache_write_tokens=0,
            session_id=sid, start_time=t0 + timedelta(minutes=10),
        ))
    since, until = MARKER - timedelta(hours=1), MARKER + timedelta(hours=20)
    _, thrash, _ = _compute_root_cause_candidates(db.conn, since, until, None)
    assert len(thrash) == 1
    assert thrash[0].cause == "ttl"
    assert thrash[0].ttl_worth_it is False

    finding = CacheEfficacyFinding(thrash_agents=thrash)
    props = _cache_thrash_to_proposals(finding)
    assert len(props) == 1
    assert props[0].estimated_recoverable_usd is None
    assert "caching not worth it at this cadence" in props[0].advise_text


def test_thrash_adapter_instability_card_lists_checklist():
    report = _report(thrash_agents=[_thrash_candidate(cause="instability", ttl_worth_it=None)])
    props = [p for p in cost_proposals_from_report(report) if p.analyzer == "cache_thrash"]
    assert len(props) == 1
    assert "timestamp" in props[0].advise_text
    assert "UUID" in props[0].advise_text or "uuid" in props[0].advise_text.lower()


# --- Adapter: A3 lookback ------------------------------------------------------

def test_lookback_adapter_card():
    report = _report(lookback_miss_agents=[_lookback_candidate()])
    props = [p for p in cost_proposals_from_report(report)
             if p.signature.startswith("cost:cache-lookback:")]
    assert len(props) == 1
    p = props[0]
    assert p.analyzer == "cache"
    assert "20" in p.evidence
    assert "4 cache miss" in p.evidence
    assert p.estimated_recoverable_tokens == 12000


# --- Signatures never collide with the existing per-(provider,model) card ----

def test_signatures_are_distinct_across_all_cache_check_kinds():
    report = _report(
        uncached_agents=[_uncached_candidate()],
        thrash_agents=[_thrash_candidate()],
        lookback_miss_agents=[_lookback_candidate()],
    )
    sigs = [p.signature for p in cost_proposals_from_report(report)]
    assert len(sigs) == len(set(sigs))


# --- The generic per-(provider,model) card CAN overlap a per-agent card ------
#
# The generic ``cost:cache:<provider>:<model>`` card and the new per-agent
# cards read the SAME underlying spans (a flagged agent's calls are part of
# the aggregate the generic row's efficacy is computed over). This is a real
# overlap, not a hypothetical: an agent that's the dominant (or sole) driver
# of a model's window-wide low efficacy trips both. The generic card's dollar
# figure must be reduced by whatever the per-agent cards already claim for
# that model, so the rollup never counts the same spend twice.

def test_per_agent_recoverable_by_model_sums_across_all_three_checks():
    uc = _uncached_candidate(agent_id="a")
    th = _thrash_candidate(agent_id="b")
    lb = _lookback_candidate(agent_id="c")
    finding = CacheEfficacyFinding(uncached_agents=[uc], thrash_agents=[th],
                                    lookback_miss_agents=[lb])
    totals = _per_agent_cache_recoverable_by_model(finding)
    key = ("anthropic", "claude-sonnet-5")  # all three fixtures share this model
    assert totals[key][0] == pytest.approx(uc.estimated_recoverable_usd
                                            + th.estimated_recoverable_usd
                                            + lb.estimated_recoverable_usd)
    assert totals[key][1] == (uc.estimated_recoverable_tokens or 0) + (lb.estimated_recoverable_tokens or 0)


def test_per_agent_recoverable_by_model_ignores_non_overlapping_models():
    uc = _uncached_candidate(agent_id="a")
    finding = CacheEfficacyFinding(uncached_agents=[uc])
    totals = _per_agent_cache_recoverable_by_model(finding)
    assert ("openai", "gpt-4o") not in totals


def test_generic_per_model_card_is_reduced_by_overlapping_per_agent_claim(db):
    """End-to-end: one agent is the model's ENTIRE window traffic, so it trips
    both the generic per-model card AND its own A1 uncached card off the same
    spans. The generic card must not still claim the full, unreduced figure."""
    agent = "agent-solo"
    for i in range(25):
        db.insert_span(make_llm_span(
            agent_id=agent, provider="anthropic", model="claude-sonnet-5",
            input_tokens=4000, cache_tokens=0, cache_write_tokens=0,
            session_id=f"s-{i // 5}",
            start_time=MARKER + timedelta(minutes=i),
        ))
    since, until = MARKER - timedelta(hours=1), MARKER + timedelta(hours=1)
    report = build_report(db=db, config=TjConfig(version="1"), since=since, until=until,
                           findings=["cache"])
    finding = report.findings["cache"]
    assert len(finding.uncached_agents) == 1        # A1 fires
    assert len(finding.flagged) == 1                # the generic row also fires

    # What the generic card WOULD claim with no per-agent overlap subtracted.
    unclaimed_variant = replace(finding, uncached_agents=[], thrash_agents=[],
                                 lookback_miss_agents=[])
    original = _cache_to_proposals(unclaimed_variant)[0].estimated_recoverable_usd

    proposals = cost_proposals_from_report(report)
    generic = next(p for p in proposals if p.signature == "cost:cache:anthropic:claude-sonnet-5")
    agent_card = next(p for p in proposals if p.signature == "cost:cache-uncached:agent-solo")

    assert generic.estimated_recoverable_usd < original
    assert generic.estimated_recoverable_usd == pytest.approx(
        max(0.0, original - agent_card.estimated_recoverable_usd), abs=1e-4,
    )
    # The two cards combined never exceed what the single generic card would
    # have claimed alone — no inflation from counting the same spend twice.
    assert (
        generic.estimated_recoverable_usd + agent_card.estimated_recoverable_usd
        <= original + 1e-6
    )
    assert "double-count" in generic.estimate_basis

