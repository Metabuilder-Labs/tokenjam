"""The two delivery kinds the seam was built for: path-scoped rules and hooks.

Both exist to price a fix HONESTLY rather than to make it look cheap. That
distinction is the whole file: a path-scoped rule really does cost almost
nothing, and an injecting hook really does cost something that a blanket
"a hook is executed, so a hook is free" would call free.
"""
from __future__ import annotations

import ast
import json

import pytest

from tokenjam.core.fixes import fix_for
from tokenjam.core.optimize.rule_scope import globs_for, may_be_path_scoped
from tokenjam.core.rulewrite.delivery import (
    DELIVERY_CLAUDE_MD_RULE,
    DELIVERY_EXECUTING_HOOK,
    DELIVERY_INJECTING_HOOK,
    DELIVERY_KINDS,
    DELIVERY_PATH_SCOPED_RULE,
    MAX_BRACE_EXPANSIONS,
    standing_tokens_per_session,
)
from tokenjam.core.rulewrite.types import RuleDestination, RuleWrite, RuleWriteRefused
from tokenjam.core.summarize import load_semantics as ls


def _rule(**kw) -> RuleWrite:
    base = dict(
        signature="relearn:migration_read_before_edit",
        analyzer="relearn",
        title="Read a migration before editing it",
        artifact_text=(
            "Read a migration file in full before editing it. Migrations are "
            "append-only and ordered, so an edit written from a remembered "
            "shape lands in the wrong place."
        ),
        delivery=DELIVERY_PATH_SCOPED_RULE,
        paths=("**/migrations/**",),
        destinations=(
            RuleDestination(path="/repo/a/.claude/rules/x.md", scope="project", sessions=40),
        ),
    )
    base.update(kw)
    return RuleWrite(**base)   # type: ignore[arg-type]


# --- load semantics: the frontmatter is what makes it cheap ------------------#

@pytest.mark.parametrize("body,expected", [
    ('---\npaths:\n  - "src/**/*.py"\n---\n\nBody.', ls.PATH_SCOPED),
    ('---\npaths: "src/**/*.py"\n---\n\nBody.', ls.PATH_SCOPED),
    ('---\npaths: ["src/**"]\n---\n\nBody.', ls.PATH_SCOPED),
    # No globs -> loads at launch with the same priority as CLAUDE.md.
    ('---\ndescription: x\n---\n\nBody.', ls.ALWAYS),
    ('---\npaths:\ndescription: x\n---\n\nBody.', ls.ALWAYS),
    ('No frontmatter, but the prose says paths: something.', ls.ALWAYS),
])
def test_a_rules_file_is_only_cheap_when_it_actually_carries_globs(body, expected):
    """The `paths:` frontmatter is the ENTIRE reason a path-scoped rule is
    cheap. A rule in the same directory without it loads at launch — so
    classifying on the directory would keep pricing a rule as cheap after it
    lost the one line that made it so."""
    assert ls.classify(".claude/rules/x.md", body) == expected


def test_without_the_text_a_rules_file_is_priced_as_always_resident():
    """The safe direction for a cost: unknown never understates."""
    assert ls.classify(".claude/rules/x.md") == ls.ALWAYS


def test_a_path_scoped_rule_has_no_invocation_key():
    """It is pulled in by a file READ matching a glob, not by an invocation
    anyone records under a name. Returning a key would make it look like a
    measurable quantity we have."""
    assert ls.invocation_key(".claude/rules/x.md", ls.PATH_SCOPED) == ""


# --- the delivery kind --------------------------------------------------------#

def test_a_path_scoped_rule_costs_a_fraction_of_the_same_words_in_claude_md():
    rule = _rule()
    md = DELIVERY_KINDS[DELIVERY_CLAUDE_MD_RULE]
    ps = DELIVERY_KINDS[DELIVERY_PATH_SCOPED_RULE]
    md_cost = md.standing_tokens(rule, md.render(rule, ""), "")
    ps_cost = ps.standing_tokens(rule, ps.render(rule, ""), "")
    assert 0 < ps_cost < md_cost
    # Only the frontmatter is resident, so the gap is large, not marginal.
    assert ps_cost < md_cost * 0.4


def test_a_path_scoped_rule_is_never_priced_at_zero():
    """The frontmatter really is carried. Pricing it at nothing would be the
    hooks-are-free claim in a different costume — which is the exact claim this
    seam exists to stop being made by default."""
    rule = _rule()
    kind = DELIVERY_KINDS[DELIVERY_PATH_SCOPED_RULE]
    assert kind.standing_tokens(rule, kind.render(rule, ""), "") > 0
    assert kind.carries_prompt_text is True


def test_a_path_scoped_kind_with_no_globs_refuses_rather_than_degrading():
    """A path-scoped rule emitting no `paths:` has silently become an
    always-resident rule while still being priced as if it were not. The file
    is written, the rule works, and only the COST is wrong — which is why this
    refuses instead of falling back."""
    kind = DELIVERY_KINDS[DELIVERY_PATH_SCOPED_RULE]
    with pytest.raises(RuleWriteRefused, match="no path globs"):
        kind.render(_rule(paths=()), "")


def test_a_glob_past_the_brace_budget_is_refused():
    """Past the expansion budget the pattern is used UNEXPANDED and matches
    nothing — so the rule would read as applied while never firing."""
    kind = DELIVERY_KINDS[DELIVERY_PATH_SCOPED_RULE]
    # Derived from the budget rather than hardcoded, so retuning the constant
    # retunes the test with it.
    side = int(MAX_BRACE_EXPANSIONS ** 0.5) + 2
    group = ",".join(str(i) for i in range(side))
    huge = "src/{" + group + "}/{" + group + "}/**"
    with pytest.raises(RuleWriteRefused, match="brace-expansion budget"):
        kind.render(_rule(paths=(huge,)), "")
    # And one just inside the budget is accepted, so the guard is a ceiling
    # rather than a blanket refusal of every brace pattern.
    small = "src/{a,b,c}/**"
    assert "paths:" in kind.render(_rule(paths=(small,)), "")


def test_the_rendered_rule_carries_its_globs_in_frontmatter():
    rendered = DELIVERY_KINDS[DELIVERY_PATH_SCOPED_RULE].render(_rule(), "")
    assert rendered.startswith("---\n")
    assert "paths:" in rendered
    assert '"**/migrations/**"' in rendered
    # And it classifies as path-scoped by the same reader the pricer uses.
    assert ls.classify(".claude/rules/x.md", rendered) == ls.PATH_SCOPED


# --- selection is DERIVED, not preferred -------------------------------------#

def test_an_action_shape_rule_may_never_be_path_scoped():
    """A rule about the shape of the NEXT ACTION — which model to dispatch on,
    how much effort, whether to delegate — decides something before any file is
    read. Scoped to globs it would load only after the decision it governs had
    been made: present, well-formed, and useless. That failure is worse than
    paying the rent, and it is invisible."""
    for key in (
        "subagent.sizing_rubric", "subagent.pin_effort",
        "resend.offload_to_subagent", "resend.sdk_cache_breakpoint",
    ):
        record = fix_for(key)
        assert record is not None
        assert may_be_path_scoped(record) is False, key
        assert globs_for(record) == (), key


def test_a_file_shaped_rule_may_be_scoped_when_it_names_its_globs():
    record = fix_for("relearn.migration_read_before_edit")
    assert record is not None
    assert may_be_path_scoped(record) is True
    assert globs_for(record) == ("**/migrations/**", "**/migrate/**")


def test_no_glob_is_ever_inferred_from_prose():
    """A scopeable record that names no globs gets none. Inferring one would be
    guessing which files an instruction is about, and a wrong guess produces a
    rule that is silently never loaded."""
    record = fix_for("relearn.edit_before_read")
    assert record is not None
    assert may_be_path_scoped(record) is True     # awareness-class...
    assert globs_for(record) == ()                # ...but it named none.


# --- the pricing split survives; the offer decision it used to feed is gone -#

def test_path_scoped_pricing_is_still_cheaper_but_never_gates_the_offer():
    """Standing cost pricing (per delivery kind) is still real and still
    differs between a path-scoped rule and a CLAUDE.md rule — but it is purely
    informational now. There is no budget left for it to feed: BOTH kinds are
    offered regardless of how expensive either prices out, however large the
    artifact.
    """
    rule = _rule()
    huge_text = "x" * 50_000
    per_session = {}
    for kind_name in (DELIVERY_CLAUDE_MD_RULE, DELIVERY_PATH_SCOPED_RULE):
        kind = DELIVERY_KINDS[kind_name]
        rendered = kind.render(rule, "") + huge_text
        per_session[kind_name] = kind.standing_tokens(rule, rendered, "")

    # The rent genuinely differs (the path-scoped rule only carries its
    # frontmatter, not the huge body).
    assert per_session[DELIVERY_PATH_SCOPED_RULE] < per_session[DELIVERY_CLAUDE_MD_RULE]
    # And neither figure gates anything: a `RuleWrite` for either kind is
    # `offered` purely from its own construction (see `core/rulewrite/plan.py`),
    # never from what `standing_tokens` returns.


# --- hooks: the zero is earned by ONE of the two kinds ----------------------#

def test_an_injecting_hook_does_not_price_to_zero():
    """THE assertion. An injecting hook puts text in front of the model — and
    its cost is worse-behaved than a rule's, since the injected block lands in
    the conversation and is re-sent every subsequent turn.

    The budget agrees with the mechanism now, which it did not when the answer
    was derived from the artifact's shape: both sides are asked the same
    question and both charge.
    """
    rule = _rule(delivery=DELIVERY_INJECTING_HOOK)
    kind = DELIVERY_KINDS[DELIVERY_INJECTING_HOOK]

    assert standing_tokens_per_session(
        DELIVERY_INJECTING_HOOK, rule.artifact_text,
    ) > 0
    assert kind.standing_tokens(rule, "", "") > 0
    assert kind.carries_prompt_text is True


def test_an_executing_hook_still_prices_to_zero():
    """The one case a zero was ever earned: run by the harness, never sent to
    the model. Removing this would over-charge a genuinely free fix."""
    rule = _rule(delivery=DELIVERY_EXECUTING_HOOK)
    kind = DELIVERY_KINDS[DELIVERY_EXECUTING_HOOK]
    assert standing_tokens_per_session(
        DELIVERY_EXECUTING_HOOK, rule.artifact_text,
    ) == 0
    assert kind.standing_tokens(rule, "", "") == 0
    assert kind.carries_prompt_text is False


def test_the_injecting_price_assumes_the_cap_the_hook_enforces():
    """The price has to assume the same ceiling the script honours; if they
    drift the product charges for one behaviour and ships another."""
    from tokenjam.core.optimize.relearn_apply import _REACTIVE_SPECS, _render_reactive_hook
    from tokenjam.core.rulewrite.delivery import MAX_NUDGES_PER_SESSION

    source = _render_reactive_hook(_REACTIVE_SPECS["stale_read_race"], "T", "sig")
    assert f"_MAX_NUDGES_PER_SESSION = {MAX_NUDGES_PER_SESSION}" in source


# --- the generated hook's own rails ------------------------------------------#

def test_the_generated_hook_nests_additional_context_correctly():
    """`additionalContext` at the TOP level is silently ignored — the hook
    runs, exits 0, and injects nothing, which is indistinguishable from
    working. It must be nested inside `hookSpecificOutput`."""
    from tokenjam.core.optimize.relearn_apply import _REACTIVE_SPECS, _render_reactive_hook

    source = _render_reactive_hook(_REACTIVE_SPECS["cwd_confusion"], "T", "sig")
    ast.parse(source)          # it is valid Python
    assert '"hookSpecificOutput"' in source
    # The key appears INSIDE the nested dict, never as a sibling of it.
    nested = source[source.index('"hookSpecificOutput"'):]
    assert '"additionalContext"' in nested
    before = source[: source.index('"hookSpecificOutput"')]
    assert '"additionalContext"' not in before


def test_the_generated_hook_dedups_per_session_and_caps_itself():
    """`PostToolUseFailure` fires per matching call — only `SessionStart` is
    structurally once-per-session — and the `once` field is honored ONLY in
    skill frontmatter, never in a settings file. So the cap lives in the
    script, keyed on the `session_id` passed on stdin."""
    from tokenjam.core.optimize.relearn_apply import _REACTIVE_SPECS, _render_reactive_hook

    source = _render_reactive_hook(_REACTIVE_SPECS["edit_string_not_found"], "T", "sig")
    assert "_nudge_budget_spent" in source
    assert 'payload.get("session_id")' in source
    assert "_MAX_NUDGES_PER_SESSION" in source


def test_the_generated_hook_is_advisory_and_fails_open():
    """A misfiring blocking hook degrades every session until removed, which is
    a far worse failure than an ignored rule. That asymmetry is why injection
    is advisory-only and the body runs under a blanket except."""
    from tokenjam.core.optimize.relearn_apply import _REACTIVE_SPECS, _render_reactive_hook

    source = _render_reactive_hook(_REACTIVE_SPECS["stale_read_race"], "T", "sig")
    assert "except Exception:" in source
    assert "fail-open" in source.lower()
    # It never emits a blocking decision.
    assert '"permissionDecision"' not in source
    assert '"deny"' not in source


def test_the_settings_patch_uses_the_matcher_group_shape():
    """Outer element is a matcher group; inner is a handler. Hook entries MERGE
    across settings layers rather than overriding, so this stacks on the user's
    own hooks — it cannot clobber them, and it must not assume it is alone."""
    from pathlib import Path

    from tokenjam.core.optimize.relearn_apply import render_settings_patch

    patch = render_settings_patch(Path("/tmp/h.py"), "PostToolUseFailure", "Edit|Write")
    group = patch["hooks"]["PostToolUseFailure"][0]
    assert group["matcher"] == "Edit|Write"
    assert group["hooks"][0]["type"] == "command"
    json.dumps(patch)          # it is serialisable as written
