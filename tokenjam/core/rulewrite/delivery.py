"""HOW a rule reaches the agent — the delivery mechanism, as a seam.

Appending markdown to a ``CLAUDE.md`` is one way to put guidance in front of an
agent. It is not the only one, and for several of these analyzers it is not the
best one: a rule at the top of a long context competes with everything after
it, while a Claude Code hook can put the same guidance in view **at the moment
of the decision** it is meant to change.

So delivery is a dimension of the fix, not a hardcoded behaviour. A rule
carries WHAT to say (``artifact_text``), WHERE it lands (its destinations) and
HOW it gets there (this module). Adding a second mechanism should be a new
:class:`DeliveryKind` registered here plus its renderer — not a refactor of the
staging, diff, apply, undo or budget machinery, none of which knows what a
CLAUDE.md is.

**The pricing trap this seam exists to keep visible.** "A hook is free" is a
statement about an EXECUTING hook — a formatter, a lint gate, a guard that
blocks a command — and it is **wrong for a context-injecting one**. A
``UserPromptSubmit`` re-injection and a ``PreToolUse`` ``additionalContext``
nudge are prompt text; they merely arrive on a different schedule than a file
read at session start. Worse, once injected they land in the conversation and
are re-sent with every subsequent turn, so their cost is not even bounded by
the number of injections.

That is why :attr:`DeliveryKind.carries_prompt_text` is a property of the
DELIVERY: asked of the mechanism, answered by it, and never inferred from what
the artifact looks like. Two hooks that look alike price differently because
they ARE different mechanisms, each carrying its own pricer. Read the flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from tokenjam.core.rulewrite.kinds import (
    DEFAULT_DELIVERY,
    DELIVERY_CLAUDE_MD_RULE,
    DELIVERY_EXECUTING_HOOK,
    DELIVERY_INJECTING_HOOK,
    DELIVERY_PATH_SCOPED_RULE,
    DELIVERY_SKILL,
    MAX_NUDGES_PER_SESSION,
    UNRESOLVED_DELIVERY,
)
from tokenjam.core.rulewrite.legacy import UNRESOLVED_DELIVERY_LABEL
from tokenjam.core.rulewrite.types import RuleWrite, RuleWriteRefused


@dataclass(frozen=True)
class DeliveryKind:
    """One way of getting a rule in front of the agent.

    ``render`` takes ``(rule, existing_content)`` and returns the FULL new
    content of the target. Returning the whole artifact rather than a patch is
    what lets staging persist output instead of a recipe, so what a reviewer
    approved in the diff is byte-for-byte what apply writes — a property every
    delivery kind has to preserve.

    ``standing_tokens`` takes ``(rule, rendered, existing)`` and returns the
    per-session tokens this delivery adds. It exists so a mechanism prices
    ITSELF rather than inheriting an answer derived from what it resembles; see
    the module docstring for why an injecting hook cannot inherit a hook's zero.
    """

    name: str
    #: Human label, for a CLI column, an expanded card and a UI badge. One
    #: form, not one per surface: a second "short" spelling was tried here and
    #: removed the moment the last caller stopped needing it, because an
    #: abstraction nothing consumes is how the thing this module replaced got
    #: started.
    label: str
    #: Whether this mechanism puts tokens in front of the model at all. The
    #: budget's netting is meaningless for a mechanism that does not, and
    #: dangerously wrong for one that does but is assumed not to.
    carries_prompt_text: bool
    render: Callable[[RuleWrite, str], str]
    standing_tokens: Callable[[RuleWrite, str, str], int]


def _cluster_for(rule: RuleWrite) -> dict:
    """The cluster shape ``relearn_apply``'s renderers read, from a rule.

    One place builds it, so a field the renderers depend on cannot be present
    for one delivery and missing for another.
    """
    return {
        "title": rule.title or rule.signature,
        "proposed_fix": rule.artifact_text,
        "sessions": sum(d.sessions for d in rule.destinations),
        "repos": [d.path for d in rule.destinations],
    }


def _render_claude_md(rule: RuleWrite, existing: str) -> str:
    """Delegates to ``relearn_apply``'s renderers so the marker comments — the
    thing that makes a re-apply replace rather than duplicate, and makes the
    existing Revert path able to find the block — are produced in exactly one
    place."""
    from tokenjam.core.optimize.relearn_apply import render_note_content

    return render_note_content(existing, _cluster_for(rule), rule.signature)


def _render_skill(rule: RuleWrite, existing: str) -> str:
    """A ``SKILL.md``, from the same renderer the apply path uses.

    A skill file is one skill per directory by construction, so there is no
    sibling content to preserve and the existing bytes are not read.
    """
    from tokenjam.core.optimize.relearn_apply import render_skill_content, slugify

    del existing
    return render_skill_content(
        _cluster_for(rule), rule.signature, slugify(rule.title or rule.signature),
    )


def _standing_tokens_claude_md(rule: RuleWrite, rendered: str, existing: str) -> int:
    """Priced on the DELTA this write introduces, through
    ``write_budget.standing_tokens_per_session`` so one module owns the
    always-resident model rather than each caller restating it.

    A re-apply that replaces an existing block adds nothing, and pricing it as
    if it added a whole block would overstate the cost of keeping a rule
    current.
    """
    from tokenjam.core.optimize.write_budget import standing_tokens_per_session

    added = max(0, len(rendered) - len(existing))
    return standing_tokens_per_session(
        DELIVERY_CLAUDE_MD_RULE, rendered[:added] if added else "",
    )


def _standing_tokens_skill(rule: RuleWrite, rendered: str, existing: str) -> int:
    """A skill's always-loaded surface: its frontmatter name + description,
    bounded however long the body is. Same delta discipline as the rule above —
    a re-apply that replaces an existing file adds nothing new to carry."""
    from tokenjam.core.optimize.write_budget import standing_tokens_per_session

    del rule
    added = max(0, len(rendered) - len(existing))
    return standing_tokens_per_session(
        DELIVERY_SKILL, rendered[:added] if added else "",
    )


#: Globs a path-scoped rule may carry before the brace-expansion budget is a
#: concern. Claude Code expands brace patterns against a shared budget of 1,000
#: expanded patterns, and a pattern that EXCEEDS it is used unexpanded — which
#: matches nothing at all. A rule that matches nothing is not a cheap rule, it
#: is an absent one that still reads as applied, so the renderer refuses to
#: emit a pattern it cannot vouch for rather than shipping a silent no-op.
MAX_BRACE_EXPANSIONS = 1_000


def _brace_expansions(pattern: str) -> int:
    """How many patterns ``pattern`` expands to. 1 when it has no braces.

    Multiplicative across brace groups, which is how the budget is actually
    consumed: ``{a,b}/{c,d,e}`` is six patterns, not five.
    """
    import re as _re

    total = 1
    for group in _re.findall(r"\{([^{}]*)\}", pattern or ""):
        total *= max(1, len(group.split(",")))
    return total


def render_path_scoped_rule(rule: RuleWrite, existing: str) -> str:
    """A `.claude/rules/<slug>.md` whose frontmatter scopes it to the globs the
    rule was derived from.

    **The ``paths:`` line is the entire reason this delivery is cheap.** A rule
    in the same directory WITHOUT it loads at launch with the same priority as
    `CLAUDE.md` — so a path-scoped kind that emits no globs has silently become
    an always-resident rule while still being priced as if it were not. This
    refuses rather than degrading, because the failure is invisible: the file
    is written, the rule works, and only the cost is wrong.
    """
    globs = tuple(rule.paths)
    if not globs:
        raise RuleWriteRefused(
            f"{rule.signature} has no path globs, so a path-scoped rule would "
            "carry no `paths:` frontmatter — which makes it load at launch "
            "like a CLAUDE.md rule while being priced as if it did not. "
            "Write it as a CLAUDE.md rule instead, or derive the globs.",
        )
    for glob in globs:
        expansions = _brace_expansions(glob)
        if expansions > MAX_BRACE_EXPANSIONS:
            raise RuleWriteRefused(
                f"the glob {glob!r} expands to {expansions} patterns, over the "
                f"{MAX_BRACE_EXPANSIONS} brace-expansion budget. Past it the "
                "pattern is used UNEXPANDED and matches nothing, so the rule "
                "would read as applied while never firing.",
            )
    from tokenjam.core.optimize.relearn_apply import render_note_content, slugify

    body = render_note_content("", _cluster_for(rule), rule.signature)
    paths_block = "\n".join(f"  - {glob!r}".replace("'", '"') for glob in globs)
    frontmatter = (
        "---\n"
        f"description: {slugify(rule.title or rule.signature).replace('-', ' ')}\n"
        "paths:\n"
        f"{paths_block}\n"
        "---\n\n"
    )
    # Re-applying replaces the whole file: a path-scoped rule is one rule per
    # file by construction, so there is no sibling content to preserve and
    # appending would duplicate the block.
    del existing
    return frontmatter + body


def _standing_tokens_path_scoped(
    rule: RuleWrite, rendered: str, existing: str,
) -> int:
    """What a path-scoped rule adds to a session, priced through the SAME
    load-semantics split the read side uses.

    Only the frontmatter is resident — that is how the harness knows the rule
    exists and what it matches — and the instruction body arrives only on a
    read that matches the glob. So the standing cost is the frontmatter, not
    the block, and the difference is the whole point of the delivery.

    Deliberately NOT zero. The frontmatter really is carried, and a delivery
    that priced itself at nothing would be making the same unearned free claim
    this seam exists to stop being made by default.
    """
    from tokenjam.core.optimize.write_budget import tokens_from_chars
    from tokenjam.core.summarize.load_semantics import (
        PATH_SCOPED,
        split_always_resident,
    )

    del rule, existing
    resident, _on_demand = split_always_resident(rendered, PATH_SCOPED)
    return tokens_from_chars(len(resident))


def _render_hook(rule: RuleWrite, existing: str) -> str:
    """The hook script for this rule, from ``relearn_apply``'s renderers.

    Delegates rather than re-implementing: hook generation already ships and
    works (guard + reactive renderers, the settings patch, the enforcement
    wiring), and a second generator would be the scattered-implementation
    problem this catalog exists to close, with the added hazard that two
    generators can disagree about a script that runs in the user's loop.
    """
    from tokenjam.core.optimize.relearn_apply import render_hook_content

    del existing
    return render_hook_content(
        {
            "family_key": rule.signature.split(":")[-1],
            "title": rule.title or rule.signature,
            "delivery": rule.delivery,
            "proposed_fix": rule.artifact_text,
        },
        rule.signature,
    )


def _standing_tokens_executing_hook(
    rule: RuleWrite, rendered: str, existing: str,
) -> int:
    """Zero, and this is the one delivery where that is EARNED.

    An executing hook is run by the harness; its source is never sent to the
    model. Nothing about it is resident, so nothing is charged.
    """
    del rule, rendered, existing
    return 0


def _standing_tokens_injecting_hook(
    rule: RuleWrite, rendered: str, existing: str,
) -> int:
    """What an injecting hook costs — never zero.

    Priced on the TEXT IT INJECTS, not on the size of the script that injects
    it: the script is executed, the injected block is what the model reads.
    Charged per nudge and multiplied by the per-session cap, because a nudge
    that fires twice costs twice — and because the injected block then rides in
    the conversation for every later turn, this is a floor rather than a
    ceiling. Erring low on a COST is the wrong direction, so it is stated as a
    floor rather than dressed up as exact.
    """
    from tokenjam.core.optimize.write_budget import tokens_from_chars

    del rendered, existing
    injected = rule.artifact_text or ""
    return tokens_from_chars(len(injected)) * MAX_NUDGES_PER_SESSION


#: Every delivery mechanism this product can offer, by name.
#:
#: Adding one means adding an entry here plus its two functions. It must NOT
#: mean touching ``apply.py``, ``store.py``, ``plan.py`` or ``write_budget`` —
#: if it does, the seam has been welded shut again and that is the thing to fix
#: rather than to work around.
DELIVERY_KINDS: dict[str, DeliveryKind] = {
    DELIVERY_CLAUDE_MD_RULE: DeliveryKind(
        name=DELIVERY_CLAUDE_MD_RULE,
        label="CLAUDE.md rule",
        carries_prompt_text=True,
        render=_render_claude_md,
        standing_tokens=_standing_tokens_claude_md,
    ),
    DELIVERY_SKILL: DeliveryKind(
        name=DELIVERY_SKILL,
        label="skill",
        # Only the frontmatter is resident, but resident is resident: a skill
        # the harness lists is a skill whose description is sent every session.
        carries_prompt_text=True,
        render=_render_skill,
        standing_tokens=_standing_tokens_skill,
    ),
    DELIVERY_PATH_SCOPED_RULE: DeliveryKind(
        name=DELIVERY_PATH_SCOPED_RULE,
        label="path-scoped rule",
        # It carries prompt text — just far less of it, and only the
        # frontmatter unconditionally. Flagging it False would be the
        # hooks-are-free mistake in a different costume.
        carries_prompt_text=True,
        render=render_path_scoped_rule,
        standing_tokens=_standing_tokens_path_scoped,
    ),
    DELIVERY_EXECUTING_HOOK: DeliveryKind(
        name=DELIVERY_EXECUTING_HOOK,
        label="hook (blocking guard)",
        carries_prompt_text=False,
        render=_render_hook,
        standing_tokens=_standing_tokens_executing_hook,
    ),
    DELIVERY_INJECTING_HOOK: DeliveryKind(
        name=DELIVERY_INJECTING_HOOK,
        label="hook (context nudge)",
        carries_prompt_text=True,
        render=_render_hook,
        standing_tokens=_standing_tokens_injecting_hook,
    ),
}


def resolve_delivery(name: str | None) -> DeliveryKind:
    """The mechanism named by ``name``, defaulting to the markdown rule.

    An UNKNOWN name is refused rather than silently falling back: a staged
    entry written by a later build that names a mechanism this build cannot
    render must not be quietly rendered as a markdown block into whatever file
    it named. That is a wrong write to a real file, which is the one failure
    this package exists to make impossible.
    """
    key = str(name or DEFAULT_DELIVERY)
    kind = DELIVERY_KINDS.get(key)
    if kind is not None:
        return kind
    if key == UNRESOLVED_DELIVERY:
        # Not a typo and not a future build — a stored record from before
        # mechanisms had names, whose artifact could not be established from
        # anything the record carries. Say that plainly: the user's action is
        # to re-run the analysis, not to look for a broken name.
        raise RuleWriteRefused(
            "this fix was saved by an older version that did not record which "
            "kind of artifact it writes, and nothing in the saved record "
            "establishes it. Guessing would write the wrong file. Re-run the "
            "analysis to refresh it.",
        )
    raise RuleWriteRefused(
        f"unknown delivery mechanism {key!r} — this build knows "
        f"{', '.join(sorted(DELIVERY_KINDS))}. Refusing to guess how to "
        "write it.",
    )


def delivery_label(name: str | None) -> str:
    """The human label for a mechanism — a card, a UI badge, a CLI column.

    Never raises, unlike :func:`resolve_delivery`: a surface that merely NAMES
    an artifact must keep rendering for a record this build cannot resolve, and
    it says so ("unknown") rather than dropping the column or guessing a label.
    Refusing is the right answer when about to WRITE the artifact; it is the
    wrong answer when about to describe one.
    """
    kind = DELIVERY_KINDS.get(str(name or ""))
    return kind.label if kind else UNRESOLVED_DELIVERY_LABEL


__all__ = [
    "DEFAULT_DELIVERY",
    "DELIVERY_CLAUDE_MD_RULE",
    "DELIVERY_EXECUTING_HOOK",
    "DELIVERY_INJECTING_HOOK",
    "DELIVERY_KINDS",
    "DELIVERY_PATH_SCOPED_RULE",
    "DELIVERY_SKILL",
    "DeliveryKind",
    "MAX_NUDGES_PER_SESSION",
    "delivery_label",
    "resolve_delivery",
]
