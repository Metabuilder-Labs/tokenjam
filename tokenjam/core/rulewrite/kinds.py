"""The names of the delivery mechanisms, and nothing else.

A dependency-free vocabulary module. ``delivery`` owns each mechanism's
BEHAVIOUR (its renderer and its pricer) and imports these names; ``types``,
``relearn_apply`` and ``legacy`` need only the names, and importing them from
here keeps every one of those out of a cycle with the behaviour module.

There is exactly one vocabulary for "what kind of artifact is this". It used to
be two — a numeric ladder rung alongside these names — and the two disagreed
about what the number meant: one module read rung 1 as *the file the artifact
lands in*, another read it as *the artifact's shape*. A number that has to be
translated back into a word before anyone can act on it was never the concept;
the word was.
"""
from __future__ import annotations

#: A markdown block appended to a ``CLAUDE.md``, loaded at session start.
DELIVERY_CLAUDE_MD_RULE = "claude_md_rule"

#: A `.claude/rules/<slug>.md` carrying a ``paths:`` glob. It loads only when
#: Claude READS a file matching the pattern — not at launch, and not on every
#: tool use — so its standing cost is a small fraction of the same words in a
#: `CLAUDE.md`.
#:
#: Not merely a cheaper way to say the same thing. Standing cost is an INPUT to
#: whether a write is offered at all, so a near-zero-rent destination makes
#: fixes net-positive that are net-negative against a `CLAUDE.md`. It widens
#: what the product can legitimately recommend rather than just discounting
#: what it already does.
DELIVERY_PATH_SCOPED_RULE = "path_scoped_rule"

#: A `.claude/skills/<slug>/SKILL.md`. Only its frontmatter is resident; the
#: body arrives when the skill is invoked.
DELIVERY_SKILL = "skill"

#: A hook that RUNS CODE and injects nothing — a guard that blocks a command, a
#: formatter. Executed, never sent to the model as text, so it is genuinely
#: free of standing prompt cost. This is the one case where a zero is EARNED,
#: and the reason it must be claimed by a named mechanism rather than inherited
#: by everything that happens to be a hook.
DELIVERY_EXECUTING_HOOK = "executing_hook"

#: A hook that puts TEXT in front of the model — a ``PostToolUseFailure`` nudge,
#: a ``UserPromptSubmit`` re-injection. Prompt text on a different schedule.
#: **Not free, and worse-behaved than a rule:** the injected block lands in the
#: conversation and is re-sent on every subsequent turn, so its cost is not even
#: bounded by session count the way a session-start read is.
#:
#: Most of the hook families that ship are this kind, and a blanket "a hook is
#: executed, so a hook is free" priced every one of them at zero.
DELIVERY_INJECTING_HOOK = "injecting_hook"

#: The default for a rule that names no mechanism.
DEFAULT_DELIVERY = DELIVERY_CLAUDE_MD_RULE

#: A stored record that DOES describe an artifact but not well enough to say
#: which mechanism wrote it — see ``rulewrite/legacy``. Deliberately a real
#: name rather than an empty string, because empty means "names no mechanism"
#: and takes the default: an ambiguous legacy record must not be able to
#: masquerade as a fresh rule and be rendered as a CLAUDE.md block into
#: whatever path it happens to carry. This name is in no registry, so every
#: lookup of it refuses.
UNRESOLVED_DELIVERY = "unresolved"

#: The mechanisms that write an executable artifact and a settings patch, and so
#: are staged DISABLED until a human separately confirms enabling them. This is
#: a statement about what the artifact IS, not about how strong an intervention
#: it is: both hook kinds need the confirmation, and they differ from each other
#: only in what they cost, which is a separate question asked separately.
ENFORCEMENT_DELIVERIES = frozenset({DELIVERY_EXECUTING_HOOK, DELIVERY_INJECTING_HOOK})

#: Mirrors the cap the generated injecting hook enforces on itself, keyed on the
#: stdin ``session_id``. The PRICE has to assume the same ceiling the script
#: honours; if they drift, the product charges for one behaviour and ships
#: another. Lives here so a pricer can price an injecting hook without
#: importing the behaviour module.
MAX_NUDGES_PER_SESSION = 2


__all__ = [
    "DEFAULT_DELIVERY",
    "DELIVERY_CLAUDE_MD_RULE",
    "DELIVERY_EXECUTING_HOOK",
    "DELIVERY_INJECTING_HOOK",
    "DELIVERY_PATH_SCOPED_RULE",
    "DELIVERY_SKILL",
    "ENFORCEMENT_DELIVERIES",
    "MAX_NUDGES_PER_SESSION",
    "UNRESOLVED_DELIVERY",
]
