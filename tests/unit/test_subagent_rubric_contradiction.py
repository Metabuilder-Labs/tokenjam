"""The rule a card writes must ERASE the number the card found.

`past_overspend_usd` is the maximum its analyzer knows: applying the fix should
remove the behaviour that produced it. A rule that leaves the number roughly
where it was is a correctness bug, not a copy preference.

The subagent rubric broke that. It ended with "a subagent that does little tool
work and returns a short result rarely needs the premium tier" — which is
exactly the gate `analyzers/subagent_rightsizing.py` DELETED this cycle
(`output_tokens < 2000` and `tool_calls <= 5`), because on a real Claude Code
corpus a Task subagent is a full agent loop and those clauses systematically
excluded the EXPENSIVE dispatches (Critical Rule 29's gate inversion). Every
dispatch in the analyzer's claim is large, tool-heavy and long-output —
precisely what that sentence waved through. The compound resend fix carried the
same sentence in a different costume ("doing broad reads and returning a short
conclusion").

Per Critical Rule 23 these are INVERTED assertions, not deleted ones: the same
pins now defend the correct state.
"""
from __future__ import annotations

import re
from pathlib import Path

from tokenjam.core.optimize.analyzers import subagent_rightsizing
from tokenjam.core.optimize.analyzers.context_resend import RIGHTSIZE_FIX_TEMPLATE
from tokenjam.core.optimize.cost_proposals import SUBAGENT_RUBRIC_INTRO

#: The shape the deleted analyzer gate keyed on. A rubric that gives this shape
#: a pass re-imposes the gate in prose, on the very population the analyzer was
#: fixed to include.
_PASSES_THE_EXPENSIVE_SHAPE = re.compile(
    r"(little tool work|few tool calls|short (?:result|conclusion|output))"
    r"[^.]*rarely needs",
    re.IGNORECASE,
)


def test_the_rubric_does_not_wave_through_the_shape_the_analyzer_bills_for():
    assert _PASSES_THE_EXPENSIVE_SHAPE.search(SUBAGENT_RUBRIC_INTRO) is None
    assert _PASSES_THE_EXPENSIVE_SHAPE.search(RIGHTSIZE_FIX_TEMPLATE) is None
    # And it says so positively, so a future edit cannot quietly restore the
    # pass by rewording it: tool volume and output length are named as NOT
    # reasons to keep the premium tier.
    assert "not on that list" in SUBAGENT_RUBRIC_INTRO
    assert "expensive one, not the hard one" in SUBAGENT_RUBRIC_INTRO
    assert "not reasons to keep" in RIGHTSIZE_FIX_TEMPLATE


def test_the_shape_based_default_down_survives():
    """The half that is correct and is the actual lever. Removing the bad
    escape hatch must not take the rule with it."""
    for text in (SUBAGENT_RUBRIC_INTRO, RIGHTSIZE_FIX_TEMPLATE):
        assert "cheapest same-family model that fits its shape" in text


def test_the_premium_escape_hatch_is_checkable_rather_than_self_assessed():
    """"Unless the subtask genuinely needs deep reasoning" asks the dispatching
    agent to rate its own task's difficulty, and an agent asked that question
    answers yes — the exception then swallows the rule. The replacement is
    stated as conditions that are either written into the dispatch or not, so
    an outside reader of the transcript can decide whether it applied."""
    assert "deep reasoning" not in SUBAGENT_RUBRIC_INTRO.lower()
    assert "genuinely needs" not in SUBAGENT_RUBRIC_INTRO.lower()
    # The criterion is a property of the DISPATCH, not of the agent's opinion.
    assert "the dispatch itself states" in SUBAGENT_RUBRIC_INTRO
    assert "already attempted it in this session and its output was rejected" in (
        SUBAGENT_RUBRIC_INTRO
    )


def test_the_analyzer_gate_this_rubric_must_agree_with_is_still_shape_free():
    """The rubric and the gate have to describe one policy.

    `SMALL_OUTPUT_TOKENS` still exists — `over_provisioned` uses it — but
    `over_powered` must not, or the prose and the code disagree again.
    """
    # The constant survives — `over_provisioned` still uses it — so its mere
    # presence is not evidence the gate came back.
    assert subagent_rightsizing.SMALL_OUTPUT_TOKENS == 2_000
    docstring = subagent_rightsizing.__doc__ or ""
    assert "REGARDLESS of how much output it" in docstring
    assert "over_powered only" in (
        subagent_rightsizing.__doc__ or ""
    ) or "Used by over_provisioned only" in Path(
        subagent_rightsizing.__file__,
    ).read_text(encoding="utf-8")


def test_the_rubric_does_not_strengthen_any_claim(monkeypatch):
    """Critical Rule 14: rewriting the rule must not turn a candidate flag into
    a promise. No "safe", no "would have worked", no equivalence claim."""
    for text in (SUBAGENT_RUBRIC_INTRO, RIGHTSIZE_FIX_TEMPLATE):
        lowered = text.lower()
        for banned in ("safe to", "would have worked", "no quality", "guaranteed"):
            assert banned not in lowered


# --- The evidence= field: a declaration/enforcement pair, not a catalog string -
#
# Every test above scans `SUBAGENT_RUBRIC_INTRO` / `RIGHTSIZE_FIX_TEMPLATE` —
# catalog constants. The `evidence=` sentence on an `over_powered` card is built
# inline in `cost_proposals._subagent_to_proposals`, at call time, from the
# finding's own rows: none of the tests above can see it, which is exactly how
# it said "did little work (small output, few tool calls)" — the
# `over_provisioned` gate's language, not `over_powered`'s — while every test in
# this file stayed green. `over_powered` is set by `is_premium_tier(model)` plus
# `MIN_FLAG_COST_USD` alone (`subagent_rightsizing._flags_for`); it never reads
# `output_tokens` or `tool_calls`. These tests build a real proposal through the
# actual construction path and pin the two to AGREE: the evidence states only
# what the gate tests, on a fixture whose numbers would directly contradict the
# banned language if it ever came back.

def _over_powered_row(**overrides):
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import SubagentRow

    fields = dict(
        session_id="s1", sub_agent_id="sa0", model="claude-opus-4-8",
        llm_calls=200, tool_calls=340, input_tokens=400_000, output_tokens=40_000,
        cache_tokens=0, cache_write_tokens=0, cost_usd=6.5, provider="anthropic",
        flags=["over_powered"],
    )
    fields.update(overrides)
    return SubagentRow(**fields)


def test_over_powered_evidence_agrees_with_its_own_gate():
    """The gate this card fires on (`is_premium_tier` + the cost floor) is what
    the evidence sentence must state — never output size or tool-call count,
    which is `over_provisioned`'s test, not this one."""
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
        SubagentRightsizingFinding,
    )
    from tokenjam.core.optimize.cost_proposals import _subagent_to_proposals

    # Deliberately the OPPOSITE of "little work": hundreds of tool calls and a
    # large output. If the banned language ever came back, this fixture is
    # exactly the case that would make it self-contradicting on the same card.
    row = _over_powered_row()
    finding = SubagentRightsizingFinding(
        flagged=[row], percent_of_cost=0.5, flagged_cost_usd=6.5,
        subagent_cost_usd=6.5, past_overspend_usd=2.0, past_overspend_tokens=440_000,
    )
    props = _subagent_to_proposals(finding)
    assert len(props) == 1
    evidence = props[0].evidence

    # Guard: the extractor actually parsed something. A test asserting only
    # absence of the banned phrases would pass vacuously against an empty or
    # unrelated string once the code moves.
    assert evidence and "premium-tier" in evidence and row.model in evidence

    lowered = evidence.lower()
    for banned in (
        "did little work", "small output", "few tool calls",
        "little tool work", "short result", "short conclusion", "short output",
    ):
        assert banned not in lowered, (
            f"over_powered evidence claims {banned!r}, a property its gate "
            "never tests"
        )


def test_over_provisioned_keeps_its_own_low_output_language():
    """The other half of the pair: `over_provisioned` legitimately earns the
    low-output framing (it is exactly what its gate tests), and it must not
    lose that language to the fix above. `_derived_effort` (cost_proposals.py)
    is where that framing survives today — it pins `effort: low` only for an
    `over_provisioned` row, never for `over_powered` alone."""
    from tokenjam.core.optimize.cost_proposals import _derived_effort

    assert _derived_effort({"over_provisioned": True}) == "low"
    assert _derived_effort({"over_provisioned": False}) is None
    assert _derived_effort({}) is None
