"""`claude-mythos-5` reaches the premium-gated analyzers.

It shipped with a full set of rates in pricing/models.toml and a row in
`DOWNGRADE_CANDIDATES`, but no entry in `model_tiers.TIER_SUBSTRINGS`. So
`is_premium_tier("claude-mythos-5")` was False and it fell out of every
premium-gated flag — while being billed at Fable's rate. Nothing failed; the
figures were simply computed over a population that silently excluded it.

`test_model_tiers.py` pins the classification and the fail-closed guard that
stops the next family landing half-added. This pins the CONSEQUENCE: that each
gate actually admits the model now. A tier constant nothing consumes would pass
that test and still leave every analyzer blind, so the two are separate.

Each test asserts mythos behaves as its price twin `claude-fable-5` does —
comparative rather than absolute, so retuning a threshold moves both sides and
these keep asking the same question. The twinning is on RATES only: the two have
different `DOWNGRADE_CANDIDATES` targets (mythos drops to sonnet-5, fable to
sonnet-4-6), so any figure priced against the alternative legitimately differs
and is compared on the half that does not depend on the target.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.model_tiers import is_premium_tier
from tokenjam.core.optimize import analyze_model_downgrade
from tokenjam.core.optimize.analyzers.model_downgrade import audit_opus_quota
from tokenjam.core.optimize.analyzers.resend_tail import (
    MIN_SESSION_CONTEXT_TOKENS,
    premium_driver_role,
)
from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
    _subagent_downgrade_target,
    run as run_subagent,
)
from tokenjam.core.optimize.types import (
    AnalyzerContext,
    OptimizeReport,
    WindowSummary,
)
from tokenjam.core.pricing import get_rates
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span

MYTHOS = "claude-mythos-5"
FABLE = "claude-fable-5"
CHEAP = "claude-haiku-4-5"


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return utcnow() - timedelta(days=30), utcnow() + timedelta(hours=1)


def _config() -> TjConfig:
    return TjConfig(version="1", capture=CaptureConfig())


def test_mythos_and_fable_are_priced_identically():
    """The premise the rest of this file rests on. If the two ever diverge in
    the table, "mythos should behave like fable" stops being the right claim and
    these comparisons need re-deriving rather than re-running."""
    mythos = get_rates("anthropic", MYTHOS)
    fable = get_rates("anthropic", FABLE)
    assert mythos is not None and fable is not None
    assert mythos == fable


def test_mythos_is_premium_like_fable():
    assert is_premium_tier(MYTHOS) is is_premium_tier(FABLE) is True
    assert is_premium_tier(CHEAP) is False


# --------------------------------------------------------------------------
# Gate 1 — downsize's driver-role case
# --------------------------------------------------------------------------

def _seed_driver_session(db, session_id: str, model: str) -> None:
    """A context-heavy session whose every turn runs tools and never delegates
    — the shape `premium_driver_role` flags. Mirrors the corpus in
    test_downsize_driver_role.py, parameterised on the model."""
    start = utcnow() - timedelta(days=2)
    for i in range(6):
        turn_at = start + timedelta(minutes=i * 10)
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", model=model, provider="anthropic",
            input_tokens=MIN_SESSION_CONTEXT_TOKENS,
            cache_tokens=MIN_SESSION_CONTEXT_TOKENS * i,
            output_tokens=800, cost_usd=1.0,
            session_id=session_id, sub_agent_id=None, start_time=turn_at,
        ))
        for j in range(3):
            db.insert_span(make_tool_span(
                agent_id="claude-code-x", tool_name="Read",
                session_id=session_id,
                start_time=turn_at + timedelta(seconds=j + 1),
            ))


def _turns_for(db, session_id: str):
    from tokenjam.core.context_diagnostic import load_turn_compositions

    since, until = _window()
    return [
        t for t in load_turn_compositions(
            db.conn, since, until, None, ordered=True, with_tool_activity=True,
        )
        if t.session_id == session_id
    ]


def test_a_mythos_driver_session_is_flagged(db):
    _seed_driver_session(db, "mythos", MYTHOS)
    assert premium_driver_role(_turns_for(db, "mythos")) == ("anthropic", MYTHOS)


def test_the_driver_role_case_prices_a_mythos_session(db):
    """Detection is not enough — the case has to produce a dollar figure, which
    additionally needs a priced downgrade target for the model."""
    _seed_driver_session(db, "mythos", MYTHOS)
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert finding.driver_sessions == 1
    assert finding.driver_recoverable_usd > 0


def test_a_mythos_driver_session_offloads_the_same_as_a_fable_one(db):
    """Compared on the OFFLOAD half only.

    `driver_offload_usd` prices the re-read tail at the DRIVER model's own
    cache-read rate, which mythos and fable share, so it must match exactly.
    `driver_tier_usd` reprices the same turns at the ALTERNATIVE, and the two
    have different downgrade targets — a difference there is the ladder doing
    its job, not the gate treating them unequally."""
    _seed_driver_session(db, "mythos", MYTHOS)
    since, until = _window()
    mythos = analyze_model_downgrade(db.conn, since, until, None, 30.0)

    other = InMemoryBackend()
    try:
        _seed_driver_session(other, "fable", FABLE)
        fable = analyze_model_downgrade(other.conn, since, until, None, 30.0)
    finally:
        other.close()
    assert mythos.driver_sessions == fable.driver_sessions == 1
    assert mythos.driver_offload_usd == pytest.approx(fable.driver_offload_usd)
    assert mythos.driver_tier_usd > 0 and fable.driver_tier_usd > 0


# --------------------------------------------------------------------------
# Gate 2 — subagent right-sizing's `over_powered`
# --------------------------------------------------------------------------

def _seed_premium_subagents(db, model: str) -> None:
    """Subagents on a premium model, costly enough to clear the noise floor."""
    start = utcnow() - timedelta(days=2)
    for s in range(3):
        sid = f"sess{s}"
        db.upsert_session(make_session(session_id=sid, plan_tier="api"))
        for i in range(6):
            db.insert_span(make_llm_span(
                session_id=sid, agent_id="svc-a", sub_agent_id=f"sub{s}",
                provider="anthropic", model=model,
                input_tokens=4_000, output_tokens=900,
                cache_tokens=120_000, cache_write_tokens=9_000,
                cost_usd=2.4,
                start_time=start + timedelta(days=s, minutes=5 * i),
            ))


def _subagent_finding(db, model: str):
    """Runs the analyzer through its own context, as its unit tests do.

    `build_report` is not used here: it skips the subagent analyzer when the
    window summary reports no cost, which would make every assertion below pass
    vacuously by finding no rows at all."""
    _seed_premium_subagents(db, model)
    since, until = _window()
    summary = WindowSummary(
        since=since, until=until, days=30.0, sessions=3, spans=0,
        total_tokens=0, total_cost_usd=200.0, thin_data=False,
    )
    ctx = AnalyzerContext(
        conn=db.conn, config=_config(), since=since, until=until, agent_id=None,
        window_days=30.0, summary=summary, report=OptimizeReport(window=summary),
    )
    run_subagent(ctx)
    return ctx.report.findings["subagent"]


def test_mythos_subagents_are_flagged_over_powered(db):
    finding = _subagent_finding(db, MYTHOS)
    flagged = [r for r in finding.rows if "over_powered" in r.flags]
    assert flagged, "no mythos subagent reached the over_powered flag"
    assert finding.past_overspend_usd is not None
    assert finding.past_overspend_usd > 0


def test_mythos_subagents_price_the_same_as_fable_ones(db):
    """Exact equality is the right claim HERE, unlike the driver-role case:
    this analyzer swaps both families to the same one-tier-down target, so the
    only remaining input is the rate, which they share."""
    assert (
        _subagent_downgrade_target("anthropic", MYTHOS)
        == _subagent_downgrade_target("anthropic", FABLE)
    )
    mythos = _subagent_finding(db, MYTHOS)
    other = InMemoryBackend()
    try:
        fable = _subagent_finding(other, FABLE)
    finally:
        other.close()
    assert mythos.past_overspend_usd == pytest.approx(fable.past_overspend_usd)


def test_a_cheap_subagent_is_still_not_over_powered(db):
    """The gate did not simply widen to everything: mythos joined the premium
    tier, haiku did not."""
    finding = _subagent_finding(db, CHEAP)
    assert not [r for r in finding.rows if "over_powered" in r.flags]


# --------------------------------------------------------------------------
# Gate 3 — the premium quota audit
# --------------------------------------------------------------------------

def _seed_cheap_shaped_premium_turns(db, model: str) -> None:
    """Premium turns in contiguous structurally-cheap stretches — the shape the
    quota audit calls misallocated."""
    start = utcnow() - timedelta(days=3)
    for s in range(4):
        sid = f"qa{s}"
        db.upsert_session(make_session(session_id=sid, plan_tier="api"))
        for i in range(10):
            db.insert_span(make_llm_span(
                session_id=sid, agent_id="svc-a", provider="anthropic",
                model=model,
                input_tokens=400, output_tokens=120,
                cache_tokens=40_000, cache_write_tokens=2_000,
                cost_usd=0.5,
                start_time=start + timedelta(days=s, minutes=4 * i),
            ))


def _audit_for(db, model: str):
    _seed_cheap_shaped_premium_turns(db, model)
    since, until = _window()
    return audit_opus_quota(db.conn, since, until, None, 30.0, _config())


def test_the_quota_audit_counts_mythos_as_premium(db):
    audit = _audit_for(db, MYTHOS)
    assert audit.opus_sessions > 0, "mythos sessions invisible to the quota audit"
    assert audit.opus_tokens > 0
    assert audit.percent_quota_misallocated > 0


def test_the_quota_audit_treats_mythos_exactly_as_fable(db):
    mythos = _audit_for(db, MYTHOS)
    other = InMemoryBackend()
    try:
        fable = _audit_for(other, FABLE)
    finally:
        other.close()
    assert mythos.opus_sessions == fable.opus_sessions
    assert mythos.opus_tokens == fable.opus_tokens
    assert mythos.percent_quota_misallocated == fable.percent_quota_misallocated


def test_a_cheap_model_still_carries_no_premium_quota(db):
    audit = _audit_for(db, CHEAP)
    assert audit.opus_sessions == 0
    assert audit.opus_tokens == 0
