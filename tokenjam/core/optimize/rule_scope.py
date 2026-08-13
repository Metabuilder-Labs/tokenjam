"""Whether a rule's observation is confined to identifiable paths.

A `.claude/rules/*.md` carrying a ``paths:`` glob loads only when Claude reads
a file matching the pattern, so it costs a small fraction of the same words in
a `CLAUDE.md`. That makes it worth reaching for — but only when it is TRUE.

**The selection has to be derived, not preferred.** Two rules look identical on
the page and are not:

* a rule whose observation is confined to identifiable paths ("this project's
  migrations are edited without reading them first") genuinely only needs to
  be in context when one of those files is in play;
* a rule that must hold on every turn regardless of what is being touched
  ("default subagents to the cheapest model that fits") is about the shape of
  the NEXT action, not about a file — scoping it to globs makes it silently
  stop applying exactly when it matters, and the failure is invisible because
  the file is written, the rule is well-formed, and nothing errors.

The second failure is worse than paying the rent, which is why this module
answers "may this be scoped" conservatively and returns nothing when it cannot
tell. A missing scope costs tokens; a wrong scope costs the fix.

Nothing here guesses a glob. Scoping needs a lever the observation already
names — the file class the behaviour happens in — and where the observation
does not name one, there is no glob to derive and the rule stays where it is.
"""
from __future__ import annotations

from tokenjam.core.fixes.catalog import (
    LEVER_AWARENESS,
    LEVER_EFFORT,
    LEVER_MODEL,
    LEVER_OFFLOAD,
    LEVER_ROUTING,
    FixRecord,
)

#: Levers whose rule is about the shape of the NEXT ACTION rather than about a
#: file, and therefore may never be path-scoped.
#:
#: Every one of these decides something before any file is read: which model to
#: dispatch on, how much effort to give it, whether to delegate at all, where
#: to route a call. A ``paths:`` glob on one of them would load the rule only
#: after the decision it governs had already been made — the rule would be
#: present, well-formed, and useless, which is the one failure mode worse than
#: paying its rent.
ACTION_SHAPE_LEVERS: frozenset[str] = frozenset({
    LEVER_MODEL, LEVER_EFFORT, LEVER_OFFLOAD, LEVER_ROUTING,
})

#: Levers whose rule is about how to handle particular FILES, and so may carry
#: a glob when the observation names one. Awareness-class fixes are the honest
#: candidates: they say "when you touch this kind of thing, remember X".
FILE_SHAPED_LEVERS: frozenset[str] = frozenset({LEVER_AWARENESS})


def may_be_path_scoped(record: FixRecord) -> bool:
    """Whether this fix's instruction can be confined to a glob at all.

    Answers from the LEVER, which is the record's own statement of what the
    fix changes — not from the text, which reads the same either way. A lever
    that decides the shape of the next action cannot be scoped to files; one
    that governs how a file is handled can.

    Conservative by construction: an unrecognised lever is not scoped. The
    cost of that is tokens; the cost of the opposite is a rule that stops
    applying without saying so.
    """
    if record.lever in ACTION_SHAPE_LEVERS:
        return False
    return record.lever in FILE_SHAPED_LEVERS


def globs_for(record: FixRecord) -> tuple[str, ...]:
    """The path globs this fix should be scoped to, or ``()``.

    ``()`` means "no scope could be derived", which is a real answer and the
    only safe default: it leaves the rule in the destination it would have had,
    priced at what that destination actually costs.

    A record opts in by naming its own globs; nothing here infers a pattern
    from prose. Inferring one would be guessing at which files an instruction
    is about, and a wrong guess produces a rule that is silently never loaded.
    """
    if not may_be_path_scoped(record):
        return ()
    return tuple(record.path_globs)


__all__ = [
    "ACTION_SHAPE_LEVERS",
    "FILE_SHAPED_LEVERS",
    "globs_for",
    "may_be_path_scoped",
]
