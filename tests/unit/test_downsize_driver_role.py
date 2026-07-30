"""`downsize`'s primary case: a premium model in the DRIVER role.

Two things are pinned here.

**Detection.** A premium-tier model that drives a long, tool-heavy session
without ever dispatching a subagent is flagged; each of the four conditions is
pinned by a session that fails exactly that one and nothing else.

**Disjointness (Critical Rule 27).** The driver-role claim prices the same
mechanism `resend` prices — material re-read by every later main-thread turn —
so the two must never draw from the same sessions. `resend` skips exactly the
sessions `premium_driver_role` flags, and both call that one shared predicate.
The end-to-end test walks `build_report` -> `cost_proposals_from_report` ->
`past_overspend_rollup`, which is the only level at which a cross-analyzer
overlap is visible at all.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import analyze_model_downgrade, build_report
from tokenjam.core.optimize.analyzers.resend_tail import (
    DRIVER_TOOL_FANOUT_FLOOR,
    MIN_SESSION_CONTEXT_TOKENS,
    premium_driver_role,
    tool_driven_stretch_mask,
)
from tokenjam.core.optimize.cost_proposals import (
    cost_proposals_from_report,
    past_overspend_rollup,
)
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_tool_span

PREMIUM = "claude-opus-4-7"
CHEAP = "claude-haiku-4-5"
TURNS = 6
TOOLS_PER_TURN = 3


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return utcnow() - timedelta(days=30), utcnow() + timedelta(hours=1)


def _seed_driver_session(
    db,
    session_id: str = "driver",
    *,
    model: str = PREMIUM,
    turns: int = TURNS,
    tools_per_turn: int = TOOLS_PER_TURN,
    delegate: bool = False,
    context_tokens: int = MIN_SESSION_CONTEXT_TOKENS,
) -> float:
    """A context-heavy session whose turns each run tools. Returns its cost."""
    start = utcnow() - timedelta(days=2)
    cost = 0.0
    for i in range(turns):
        turn_at = start + timedelta(minutes=i * 10)
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", model=model, provider="anthropic",
            input_tokens=context_tokens,
            # Growing cache reads: the accumulating context each later turn
            # re-reads, which is exactly what offloading removes.
            cache_tokens=context_tokens * i,
            output_tokens=800, cost_usd=1.0,
            session_id=session_id, sub_agent_id=None, start_time=turn_at,
        ))
        cost += 1.0
        for j in range(tools_per_turn):
            db.insert_span(make_tool_span(
                agent_id="claude-code-x", tool_name="Read",
                session_id=session_id,
                start_time=turn_at + timedelta(seconds=j + 1),
            ))
    if delegate:
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", model=model, provider="anthropic",
            input_tokens=5_000, output_tokens=200, cost_usd=0.2,
            session_id=session_id, sub_agent_id="researcher",
            start_time=start + timedelta(minutes=turns * 10),
        ))
        cost += 0.2
    return cost


def _turns_for(db, session_id: str):
    from tokenjam.core.context_diagnostic import load_turn_compositions

    since, until = _window()
    return [
        t for t in load_turn_compositions(
            db.conn, since, until, None, ordered=True, with_tool_activity=True,
        )
        if t.session_id == session_id
    ]


# --- detection ---------------------------------------------------------------

def test_premium_driver_with_undelegated_tool_work_is_flagged(db):
    _seed_driver_session(db)
    assert premium_driver_role(_turns_for(db, "driver")) == ("anthropic", PREMIUM)


def test_a_cheap_driver_is_not_this_finding(db):
    # The whole claim is about the PRICE of the model doing inline work. A
    # Haiku-driven session re-sends just as much context and is still not this.
    _seed_driver_session(db, session_id="cheap", model=CHEAP)
    assert premium_driver_role(_turns_for(db, "cheap")) is None


def test_a_session_that_already_delegates_is_not_this_finding(db):
    # A session that dispatches a subagent is not making this mistake, and its
    # subagent spans belong to the `subagent` analyzer.
    _seed_driver_session(db, session_id="deleg", delegate=True)
    assert premium_driver_role(_turns_for(db, "deleg")) is None


def test_a_tool_light_session_is_not_this_finding(db):
    # A premium conversation with two greps in it had no routable work.
    _seed_driver_session(db, session_id="light", tools_per_turn=0)
    assert premium_driver_role(_turns_for(db, "light")) is None


def test_a_small_context_session_is_not_this_finding(db):
    # Below the context floor there is no tail to remove, so there is no lever.
    _seed_driver_session(db, session_id="small", context_tokens=1_000)
    assert premium_driver_role(_turns_for(db, "small")) is None


def test_tool_activity_is_required_to_evaluate_the_predicate(db):
    # `tool_fanout` and `delegates` default to inert values when turns are
    # loaded without the tool-span join, which would make every premium session
    # look like a flagged one. Both callers pass `with_tool_activity=True`; this
    # pins that the predicate genuinely depends on it.
    from tokenjam.core.context_diagnostic import load_turn_compositions

    _seed_driver_session(db)
    since, until = _window()
    bare = load_turn_compositions(db.conn, since, until, None, ordered=True)
    assert premium_driver_role(bare) is None


def test_a_lone_tool_turn_is_not_a_stretch():
    # One isolated tool call between conversational turns is not the
    # "long tool-output loop" shape a worker replaces.
    from tokenjam.core.context_diagnostic import TurnComposition

    def turn(fanout: int) -> TurnComposition:
        return TurnComposition(
            session_id="s", sub_agent_id=None, model=PREMIUM,
            reread_tokens=0, new_input_tokens=1, output_tokens=1,
            cache_write_tokens=0, cost_usd=0.0, tool_fanout=fanout,
        )

    assert tool_driven_stretch_mask([turn(0), turn(1), turn(0)]) == [False, False, False]
    assert tool_driven_stretch_mask([turn(1), turn(1), turn(0)]) == [True, True, False]


# --- the finding -------------------------------------------------------------

def test_the_finding_carries_a_dollar_figure_and_names_the_substitute(db):
    _seed_driver_session(db)
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert finding.driver_sessions == 1
    assert finding.driver_recoverable_usd > 0
    # Both halves are real and are what the headline sums to.
    assert finding.driver_offload_usd > 0
    assert finding.driver_tier_usd > 0
    assert finding.driver_recoverable_usd == pytest.approx(
        finding.driver_offload_usd + finding.driver_tier_usd, abs=1e-6,
    )
    # The substitute worker tier is named — a counterfactual whose substitute
    # is unstated cannot be inspected.
    assert finding.driver_substitutes == {PREMIUM: CHEAP}
    assert CHEAP in finding.driver_estimate_basis
    assert str(DRIVER_TOOL_FANOUT_FLOOR) in finding.driver_estimate_basis


def test_the_claim_never_exceeds_what_the_session_actually_spent(db):
    cost = _seed_driver_session(db)
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert 0 < finding.driver_recoverable_usd < cost


def test_the_tiny_session_case_survives_as_a_secondary_contribution(db):
    # A structurally tiny session with a cheaper same-family target still
    # produces the old claim, on its own, with no driver session in the window.
    start = utcnow() - timedelta(days=2)
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model=PREMIUM, provider="anthropic",
        input_tokens=400, output_tokens=40, cost_usd=0.02,
        session_id="tiny", sub_agent_id=None, start_time=start,
    ))
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert finding.driver_sessions == 0
    assert finding.candidate_sessions == 1
    assert finding.past_overspend_usd > 0


def test_a_driver_session_is_excluded_from_the_tiny_session_case(db):
    # The two gates can both admit one session: the SMALL_* gate reads UNCACHED
    # input, so a session with little uncached input but a huge cache-read tail
    # clears it while still being a context-heavy driver session. Both cases
    # feed one `past_overspend_usd`, so the exclusion is what stops the
    # finding double-counting against itself.
    start = utcnow() - timedelta(days=2)
    for i in range(TURNS):
        turn_at = start + timedelta(minutes=i * 10)
        db.insert_span(make_llm_span(
            agent_id="claude-code-x", model=PREMIUM, provider="anthropic",
            input_tokens=100,                       # under SMALL_INPUT_TOKENS
            cache_tokens=MIN_SESSION_CONTEXT_TOKENS * (i + 1),
            output_tokens=10,                       # under SMALL_OUTPUT_TOKENS
            cost_usd=1.0,
            session_id="both", sub_agent_id=None, start_time=turn_at,
        ))
        for j in range(TOOLS_PER_TURN):
            db.insert_span(make_tool_span(
                agent_id="claude-code-x", tool_name="Read", session_id="both",
                start_time=turn_at + timedelta(seconds=j + 1),
            ))
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert finding.driver_sessions == 1
    # Claimed by the driver case, so the tiny-session walk must not see it —
    # even though its summed shape clears every SMALL_* threshold.
    assert finding.candidate_sessions == 0
    assert finding.past_overspend_usd == pytest.approx(
        finding.driver_recoverable_usd, abs=1e-6,
    )


# --- disjointness against resend and subagent --------------------------------

def test_resend_hands_driver_sessions_to_downsize(db):
    # A delegating session so `offloadable_share` is measurable at all, plus a
    # driver session that resend must NOT claim.
    _seed_driver_session(db, session_id="deleg", delegate=True)
    _seed_driver_session(db, session_id="driver")
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model=PREMIUM, provider="anthropic",
        input_tokens=100, output_tokens=10, cost_usd=0.01,
        session_id="pad", sub_agent_id=None, start_time=utcnow() - timedelta(days=2),
    ))
    since, until = _window()
    report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["resend", "downsize"],
    )
    resend = report.findings["resend"]
    assert resend.driver_role_sessions == 1
    assert report.downgrade is not None
    assert report.downgrade.driver_sessions == 1
    # The partition is visible on the payload, not an invisible subtraction.
    assert any("model-role card" in n for n in resend.notes)


def test_the_rollup_counts_a_driver_session_exactly_once(db):
    # End to end. `downsize` claims the driver session; `resend` claims the
    # delegating one. Neither analyzer's signature can dedup against the other
    # (`cost:downsize:driver-role` vs `cost:resend`), so the populations have to
    # be disjoint at the source — which is what this pins.
    deleg_cost = _seed_driver_session(db, session_id="deleg", delegate=True)
    driver_cost = _seed_driver_session(db, session_id="driver")
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model=PREMIUM, provider="anthropic",
        input_tokens=100, output_tokens=10, cost_usd=0.01,
        session_id="pad", sub_agent_id=None, start_time=utcnow() - timedelta(days=2),
    ))
    since, until = _window()
    report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["resend", "downsize", "subagent"],
    )
    proposals = cost_proposals_from_report(report, None, window_days=30.0)
    driver_cards = [p for p in proposals if p.signature == "cost:downsize:driver-role"]
    assert len(driver_cards) == 1, "exactly one window-wide driver card, never one per agent"
    assert CHEAP in driver_cards[0].evidence

    rollup = past_overspend_rollup(proposals)
    # Nothing may claim back more than the window actually spent.
    assert rollup["past_overspend_usd"] <= deleg_cost + driver_cost + 0.01
    # And the driver card's own claim stays inside its own session's spend.
    assert 0 < (driver_cards[0].past_overspend_usd or 0.0) < driver_cost


def test_the_driver_card_and_the_tiny_card_do_not_claim_the_same_dollars(db):
    # Both cases fire in one window. The finding's `past_overspend_usd`
    # is their sum, so the two CARDS must split it rather than each carrying
    # the total — the failure mode Critical Rule 27 describes.
    _seed_driver_session(db, session_id="deleg", delegate=True)
    _seed_driver_session(db, session_id="driver")
    db.insert_span(make_llm_span(
        agent_id="claude-code-x", model=PREMIUM, provider="anthropic",
        input_tokens=400, output_tokens=40, cost_usd=0.02,
        session_id="tiny", sub_agent_id=None, start_time=utcnow() - timedelta(days=2),
    ))
    since, until = _window()
    report = build_report(
        db=db, config=TjConfig(version="1"), since=since, until=until,
        findings=["downsize"],
    )
    finding = report.downgrade
    assert finding is not None and finding.driver_sessions == 1
    assert finding.candidate_sessions == 1

    proposals = [p for p in cost_proposals_from_report(report, None, window_days=30.0)
                 if p.analyzer == "downsize"]
    driver = [p for p in proposals if p.signature == "cost:downsize:driver-role"]
    others = [p for p in proposals if p.signature != "cost:downsize:driver-role"]
    assert len(driver) == 1
    assert driver[0].past_overspend_usd == pytest.approx(
        finding.driver_recoverable_usd, abs=1e-6,
    )
    # The tiny-session cards carry the tiny-session share and nothing more.
    # Their own arithmetic re-derives the current cost from token rates rather
    # than the recorded `cost_usd`, so this is an upper bound, not an equality —
    # what it forbids is the driver dollars appearing a second time here.
    tiny_savings = finding.actual_cost_usd - finding.alternative_cost_usd
    assert 0 < sum(p.past_overspend_usd or 0.0 for p in others) <= tiny_savings * 1.01
    # Nothing claims the combined figure twice.
    assert sum(p.past_overspend_usd or 0.0 for p in proposals) <= (
        finding.past_overspend_usd + 1e-6
    )
