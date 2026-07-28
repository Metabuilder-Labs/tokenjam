"""The fix catalog and the lint that keeps its texts honest.

Every property here corresponds to a defect that ALREADY SHIPPED, twice each,
past readers who were looking. That is the argument for a lint rather than a
review checklist: none of these are subtle once stated, they are just invisible
when a policy has no single home to be checked in.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from tokenjam.core.fixes import (
    FIX_CATALOG,
    PERSONA_CLAUDE_CODE,
    PERSONA_SDK,
    FixRecord,
    fix_for,
    fix_text,
    fixes_for,
    lint_catalog,
    lint_fix,
)


def _record(**kw) -> FixRecord:
    base = dict(
        key="test.fix",
        text="Default every worker to the cheapest model that fits its shape.",
        delivery="claude_md_rule",
        personas=frozenset({PERSONA_CLAUDE_CODE}),
        analyzers=frozenset({"subagent"}),
        answers="an observation",
    )
    base.update(kw)
    return FixRecord(**base)   # type: ignore[arg-type]


# --- the catalog holds --------------------------------------------------------#

def test_every_catalogued_fix_passes_its_own_lint():
    """The gate. A shipped fix that violates a published property is exactly
    the class of defect this catalog was built after."""
    assert lint_catalog() == {}


def test_a_fix_is_defined_once_and_a_duplicate_key_is_refused():
    """Two copies of a policy is two policies — which is how the identical
    sizing contradiction shipped in the subagent rubric AND the resend
    template in different words, and was fixed in only one of them."""
    from tokenjam.core.fixes.catalog import register

    with pytest.raises(ValueError, match="duplicate fix key"):
        register(_record(key="subagent.sizing_rubric"))


def test_the_analyzer_constants_read_from_the_catalog_rather_than_redefining_it():
    """The two constants that drifted now resolve THROUGH the catalog, so a
    correction lands in both places or neither."""
    from tokenjam.core.optimize.analyzers.context_resend import (
        COMPACTION_FIX,
        RESEND_SDK_TRIM_FIX,
        RIGHTSIZE_FIX_TEMPLATE,
    )
    from tokenjam.core.optimize.analyzers.output_verbosity import (
        VERBOSITY_REMEDY_SNIPPET,
    )
    from tokenjam.core.optimize.cost_proposals import (
        CACHE_NO_LEVER_TEXT,
        SUBAGENT_RUBRIC_INTRO,
        _DOWNSIZE_CC_LEVER,
        _REUSE_NOTE_INTRO,
        _SCRIPT_SKILL_INTRO,
    )

    for constant, key in (
        (SUBAGENT_RUBRIC_INTRO, "subagent.sizing_rubric"),
        (RIGHTSIZE_FIX_TEMPLATE, "resend.rightsize_worker"),
        (COMPACTION_FIX, "resend.compaction_relief"),
        (RESEND_SDK_TRIM_FIX, "resend.sdk_trim_context"),
        (VERBOSITY_REMEDY_SNIPPET, "verbosity.prefer_shorter_when_it_serves"),
        (CACHE_NO_LEVER_TEXT, "cache.no_claude_code_lever"),
        (_DOWNSIZE_CC_LEVER, "downsize.claude_code_levers"),
        (_SCRIPT_SKILL_INTRO, "script.replace_agent_turn_with_script"),
        (_REUSE_NOTE_INTRO, "reuse.template_the_plan_skeleton"),
    ):
        assert constant == fix_text(key), key


def test_a_missing_key_raises_rather_than_returning_empty_text():
    """A card with an empty fix is the same quiet failure as a frontmatter key
    the harness ignores: it looks fine and does nothing."""
    with pytest.raises(KeyError):
        fix_text("nope.not.a.fix")


# --- the persona dimension ----------------------------------------------------#

def test_a_fix_is_only_offered_to_a_persona_that_can_act_on_it():
    """What a user can edit differs completely by persona. A Claude Code user
    edits instruction files, agent files and hooks; an SDK service edits
    routing code and redeploys. Handing either the other's fix names an action
    they cannot take."""
    cc = {r.key for r in fixes_for("resend", PERSONA_CLAUDE_CODE)}
    sdk = {r.key for r in fixes_for("resend", PERSONA_SDK)}
    assert "resend.offload_to_subagent" in cc
    assert "resend.offload_to_subagent" not in sdk
    assert "resend.sdk_cache_breakpoint" in sdk
    assert "resend.sdk_cache_breakpoint" not in cc


def test_model_and_effort_are_separate_levers():
    """They answer different diagnoses and neither substitutes for the other:
    a worker that did not know enough needs a bigger model, one that did not
    try hard enough needs more effort. Collapsing them loses half the remedy."""
    from tokenjam.core.fixes import LEVER_EFFORT, LEVER_MODEL

    levers = {r.key: r.lever for r in fixes_for("subagent")}
    assert levers["subagent.sizing_rubric"] == LEVER_MODEL
    assert levers["subagent.pin_effort"] == LEVER_EFFORT


# --- the lint's own properties ------------------------------------------------#

def test_the_lint_catches_a_fix_that_relicenses_what_its_analyzer_bills_for():
    """THE load-bearing check. `past_overspend_usd` is the maximum its analyzer
    knows: applying the fix should ERASE it. A fix that gives a pass to the
    exact shape the analyzer flagged leaves the number where it was."""
    rubric = fix_for("subagent.sizing_rubric")
    assert rubric is not None
    reopened = replace(
        rubric,
        text=rubric.text + " A subagent that does little tool work rarely "
                           "needs the premium tier.",
    )
    problems = lint_fix(reopened)
    assert any("re-licenses the behaviour" in p for p in problems)
    assert any("little tool work" in p for p in problems)


def test_every_relicense_guard_actually_fires():
    """A declared guard that cannot fire is decoration.

    ``must_not_relicense`` is the load-bearing property — applying a fix has to
    be able to ERASE the number its analyzer found — but it is data, and data
    can be wrong in a way that always passes: a shape nobody would ever write,
    or one already absent by construction. So every declared shape is injected
    into its own record and must produce a NAMED violation.

    It also covers the migration's own risk. The guard used to exist on the
    sizing family alone, which is the one place it was already known to be
    needed; the records that came in from the analyzers arrived with none, so
    the property held over three records while reading as if it held over all.
    """
    guarded = [r for r in FIX_CATALOG.values() if r.must_not_relicense]
    assert len(guarded) > 3, "the guard covers barely more than it started with"
    for record in guarded:
        for shape in sorted(record.must_not_relicense):
            reopened = replace(record, text=f"{record.text} Also fine: {shape}.")
            problems = lint_fix(reopened)
            assert any("re-licenses the behaviour" in p for p in problems), (
                record.key, shape,
            )
            assert any(shape in p for p in problems), (record.key, shape)


def test_the_lint_catches_a_self_graded_escape_hatch():
    """"Unless the subtask genuinely needs deep reasoning" asks the agent to
    rate its own task's difficulty, and an agent asked that answers yes — so
    the exception swallows the rule."""
    for hatch in (
        "Use the cheap model unless the task genuinely needs deep reasoning.",
        "Route down, except when the subtask truly requires a larger model.",
        "Prefer the small model unless the work needs deep thinking.",
    ):
        problems = lint_fix(_record(text=hatch))
        assert any("grades itself" in p for p in problems), hatch


def test_the_lint_catches_a_fix_that_says_to_do_nothing_but_is_offered():
    """A card whose fix text says there is nothing to do must not occupy an
    apply slot. The observation still stands — only the offer is withdrawn."""
    problems = lint_fix(_record(
        text="The harness already handles this, so no rule or hook is needed.",
        advisory_only=False,
    ))
    assert any("advisory_only" in p for p in problems)
    # And marked correctly, it passes.
    assert lint_fix(_record(
        text="The harness already handles this, so no rule or hook is needed.",
        advisory_only=True,
    )) == []


def test_the_lint_catches_a_label_masquerading_as_a_fix():
    assert any("too short" in p for p in lint_fix(_record(text="Use haiku.")))


def test_the_lint_catches_instruction_text_past_the_published_ceiling():
    from tokenjam.core.fixes import MAX_FIX_LINES

    long_text = "\n".join(["Route context-heavy work to a worker."] * (MAX_FIX_LINES + 1))
    assert any("exceeds the" in p for p in lint_fix(_record(text=long_text)))


def test_the_lint_says_escalate_to_a_hook_for_fixed_point_behaviour():
    """Text that must run at a fixed point is a hook's job. Delivered as a
    rule it is a request the agent may or may not honour at that instant."""
    problems = lint_fix(_record(
        text="Before every Bash call, check the working directory is the repo root.",
    ))
    assert any("Escalate to a hook" in p for p in problems)


def test_the_no_action_pattern_tolerates_a_list_of_nouns():
    """Regression on the lint itself. Its first draft matched only a single
    noun, so "no rule or hook is needed" — the natural phrasing — reported a
    correctly-marked advisory record as mis-marked. A false positive teaches
    the next author to reach for an exception rather than fix the text."""
    assert lint_fix(_record(
        text="The harness already blocks this, so no rule or hook is needed here.",
        advisory_only=True,
    )) == []


def test_the_catalog_covers_the_analyzers_that_write_rules():
    """Coverage is the whole point, and a floor is how it stays that way.

    The catalog held eight records while every other analyzer kept its fix text
    as its own constants — and the lint only guards what is catalogued, so a
    green lint read as "the fix text is checked" while most of it was not. That
    is worse to be complacent about than having no lint at all.
    """
    for analyzer in (
        "subagent", "resend", "relearn", "downsize", "cache",
        "verbosity", "trim", "script", "reuse", "batch",
    ):
        assert fixes_for(analyzer), analyzer
    # A floor rather than an equality: adding a fix must not need this edited,
    # but silently dropping the surface back toward the original eight must
    # fail. The number is a coverage claim, not a measurement of the product.
    assert len(FIX_CATALOG) > 8


# --- one instruction, one record (the three-analyzers-one-rule case) ---------#

def test_no_two_records_carry_substantially_the_same_instruction():
    """Three fixes told the agent to delegate context-heavy work to a subagent
    in three separately-authored wordings, so three near-identical blocks could
    land in one CLAUDE.md — and the write budget's one-block-per-family rule
    could not see it, because they were three families from three analyzers.

    The harm is not untidiness: length and redundancy REDUCE adherence, so
    writing a rule three times makes it less likely to be followed than writing
    it once. Each analyzer's duplicate actively defeats the others."""
    from tokenjam.core.fixes.lint import lint_duplicates

    assert lint_duplicates() == {}


def test_all_three_offload_analyzers_reference_one_record():
    """Their CLAIMS stay disjoint — they legitimately price different span
    populations (Critical Rule 27) — but they must not each author their own
    copy of the same instruction."""
    from tokenjam.core.optimize.analyzers.context_resend import SUBAGENT_OFFLOAD_FIX
    from tokenjam.core.optimize.analyzers.relearn import _FAMILY_BY_KEY
    from tokenjam.core.optimize.cost_proposals import (
        _driver_role_advice as _DRIVER_ROLE_ADVICE,
    )

    canonical = fix_text("resend.offload_to_subagent")
    assert SUBAGENT_OFFLOAD_FIX == canonical
    # The other two lead in with their own framing, then quote the ONE rule.
    assert canonical in _DRIVER_ROLE_ADVICE()
    assert canonical in _FAMILY_BY_KEY["context_overflow"]["fix"]
    # One record, three analyzers.
    record = fix_for("resend.offload_to_subagent")
    assert record is not None
    assert record.analyzers == frozenset({"resend", "downsize", "relearn"})


def test_the_duplicate_check_uses_containment_not_jaccard():
    """The metric matters, and the first draft got it wrong.

    Jaccard divides by the union, so it collapses when two texts differ in
    LENGTH — which is exactly the shape of this defect. Measured against the
    real wordings, the driver-role text scored 22% Jaccard against the rule it
    duplicated: far under any usable threshold, so a Jaccard check would have
    certified the defect as absent. A check that cannot catch what it was
    written for is worse than no check.

    Compared against the COMPOSED artifact, because that is what each of these
    wordings actually duplicated: the driver-role text restated both halves —
    where the work runs and what it runs on — which is why it reads as a near
    copy of the pair rather than of either record alone.
    """
    from tokenjam.core.fixes.lint import NEAR_DUPLICATE_OVERLAP, _overlap
    from tokenjam.core.optimize.cost_proposals import compound_offload_fix

    composed = compound_offload_fix(
        {},
        fix_text("resend.offload_to_subagent"),
        fix_text("resend.rightsize_worker"),
    )
    for shipped_wording in (
        "Route this shape of work to workers instead of doing it inline. Add a "
        "standing rule to CLAUDE.md telling the agent to dispatch a subagent "
        "for context-heavy sub-tasks (broad file reads, multi-file search, "
        "long tool-output loops, exploratory investigation) rather than "
        "running them in the main thread, and pin the worker's model in its "
        "own frontmatter so every dispatch inherits the cheaper tier.",
        "The durable fix is to keep bulk content off the main thread: delegate "
        "whole-file reads, log sweeps and multi-file investigations to a "
        "subagent (its tool output lives in its own context and is never "
        "re-sent on a later parent turn).",
    ):
        containment = _overlap(shipped_wording, composed)
        assert containment >= NEAR_DUPLICATE_OVERLAP, containment
        # And the metric that would have missed it, pinned so the choice
        # cannot be quietly reverted: Jaccard on the same pair.
        a = set(shipped_wording.lower().split()) & set(composed.lower().split())
        b = set(shipped_wording.lower().split()) | set(composed.lower().split())
        assert len(a) / len(b) < NEAR_DUPLICATE_OVERLAP


def test_an_unrelated_fix_is_not_flagged_as_a_duplicate():
    """The check has to be usable: two genuinely different instructions that
    share this domain's vocabulary must not trip it."""
    from tokenjam.core.fixes.lint import NEAR_DUPLICATE_OVERLAP, _overlap

    assert _overlap(
        fix_text("resend.offload_to_subagent"),
        fix_text("resend.sdk_cache_breakpoint"),
    ) < NEAR_DUPLICATE_OVERLAP


def test_no_composable_group_states_one_instruction_twice():
    """The check the pairwise one structurally cannot make.

    ``lint_duplicates`` compares whole records, so it only sees redundancy
    between two records taken entire. This defect does not have that shape: a
    model-pinning sentence folded into the offload rule scored 42% against the
    record that owns pinning — under threshold, because each record was only
    PARTLY redundant — while the compound card rendered both back to back and
    put the same instruction in front of the user twice. It was found by
    rendering the artifact and reading it, which is not a check.

    The groups are derived from the catalog rather than hand-listed, so a record
    added tomorrow is covered without anyone remembering to extend a list.
    """
    from tokenjam.core.fixes.lint import lint_composed

    assert lint_composed() == {}


def test_the_artifacts_the_product_actually_emits_are_checked_too():
    """Derived groups are necessary but not sufficient.

    ``composable_groups`` reasons about which records COULD co-land. This
    renders what the product DOES emit, through the real composition helper both
    the card and the CLI lead with — the one place where a change to how the
    parts are joined, rather than to any record, could reintroduce the defect.
    """
    from tokenjam.core.fixes.lint import lint_composed_text
    from tokenjam.core.optimize.cost_proposals import compound_offload_fix

    where = fix_text("resend.offload_to_subagent")
    what_on = fix_text("resend.rightsize_worker")
    # Both shapes the helper emits: with and without a concrete right-sizing
    # target, since the third part only appears when one was observed.
    for rightsize in (
        {},
        {
            "agent_name": "Explore", "proposed_model": "claude-haiku-4-5",
            "target_path": ".claude/agents/Explore.md",
        },
    ):
        composed = compound_offload_fix(rightsize, where, what_on)
        assert lint_composed_text("compound_offload_fix", [
            ("resend.offload_to_subagent", where),
            ("resend.rightsize_worker", what_on),
            ("rendered:concrete", composed[len(where) + len(what_on):]),
        ]) == []


def test_the_composed_check_catches_the_defect_that_motivated_it():
    """The check's own regression test, on the real shape it missed.

    A composed check that cannot fail is indistinguishable from one that
    passes. This reinstates the pinning sentence the offload record used to
    carry and asserts the check reports it — the pairwise check, run on the same
    pair, does not, which is the whole reason this one exists.
    """
    from tokenjam.core.fixes.lint import (
        NEAR_DUPLICATE_OVERLAP,
        _overlap,
        lint_composed_text,
    )

    reopened = fix_text("resend.offload_to_subagent") + (
        " Pin the worker's model in its own definition file so every dispatch "
        "inherits the cheaper tier instead of the parent's."
    )
    what_on = fix_text("resend.rightsize_worker")

    problems = lint_composed_text("compound", [
        ("resend.offload_to_subagent", reopened),
        ("resend.rightsize_worker", what_on),
    ])
    assert any("states one instruction twice" in p for p in problems)
    # And the pairwise check on the same pair does NOT fire — which is the
    # blind spot, pinned, so nobody "simplifies" this back to one check.
    assert _overlap(reopened, what_on) < NEAR_DUPLICATE_OVERLAP


def test_the_composed_check_does_not_fire_on_a_shared_delivery_label():
    """The false-positive direction, which matters more than the gap.

    Many fix texts open with the mechanism they are delivered through
    ("CLAUDE.md/skill note:"). That names the DELIVERY, which is already a field
    on the record, so leaving it in the comparison makes every record of one
    kind look partly like every other — two unrelated Read rules scored 71% on
    the prefix alone. A false positive teaches the next author to reach for an
    exception rather than to fix the text, and an exception here is permanent.
    """
    from tokenjam.core.fixes.lint import lint_composed_text

    assert lint_composed_text("read-rules", [
        ("relearn.read_directory", fix_text("relearn.read_directory")),
        ("relearn.read_too_large", fix_text("relearn.read_too_large")),
    ]) == []


def test_each_record_carries_ONE_instruction_so_composition_is_safe():
    """The compound artifact renders two records back to back, so a record that
    strays into its neighbour's job puts the same instruction in front of the
    user twice — the very defect, in miniature, inside one block.

    This was caught by rendering the composed artifact and READING it: the
    pairwise catalog lint scored the pair at 42%, under threshold, because each
    record was only partly redundant. A pairwise check cannot see redundancy
    that only appears once two records are concatenated.
    """
    from tokenjam.core.fixes.lint import _overlap
    from tokenjam.core.optimize.cost_proposals import compound_offload_fix

    where = fix_text("resend.offload_to_subagent")
    what_on = fix_text("resend.rightsize_worker")
    # The offload record is about WHERE the work runs; pinning belongs to the
    # right-sizing record, which owns "what it runs on".
    assert "pin" not in where.lower()
    assert "pin both that model" in what_on
    assert _overlap(where, what_on) < 0.30

    composed = compound_offload_fix({}, where, what_on)
    # One mention of the pinning instruction in the artifact a user receives.
    assert composed.lower().count("definition file") == 1
