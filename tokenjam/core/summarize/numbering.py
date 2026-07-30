"""The never-renumber gate: a relocation may not disturb a numbered list.

Instruction files in this genre carry numbered lists — "Critical Rule 27",
"root anti-pattern 21" — whose numbers are cited from source comments, from
other documents, and from people's memory. Renumbering silently breaks every
one of those citations, and nothing else in the tree tests for it: the files
still parse, the rules are still all present, and the only symptom is that a
reference now resolves to the wrong rule. That is worse than a broken link,
because it reads as correct.

So this is a **gate, not a test**. It sits in the same position as the
structure gate in ``session.check``: it runs on the candidate output before
anything is staged or written, and if it fails, nothing is written at all.

**The multiset is taken over BOTH files together**, source and target, before
and after. That is what makes a MOVE pass and a RENUMBER fail. Relocating a
section that contains ``1.``/``2.``/``3.`` removes those three items from the
source and adds the same three to the target, so the combined multiset is
unchanged — while renumbering the survivors in either file changes it
immediately. Checking the files separately would reject every legitimate move.

Deliberately a multiset (``Counter``) and not a set or a sorted list: two
sections can each open a list at ``1.``, and collapsing the duplicates would
let one of them be dropped without the gate noticing.
"""
from __future__ import annotations

import re
from collections import Counter

from tokenjam.core.summarize.sections import _FENCE_RE

#: A numbered list item at the start of a line: the shape "Critical Rule N" and
#: "root anti-pattern N" are actually written in. Up to three leading spaces so
#: a nested item counts too — a nested list's numbers are cited just as often.
_NUMBERED_ITEM_RE = re.compile(r"^[ \t]{0,3}(\d+)[.)][ \t]")


def numbered_items(text: str) -> Counter[str]:
    """Multiset of the numbers opening list items in ``text``, fences excluded.

    Fenced code is skipped for the same reason :mod:`sections` skips it: a
    ``1.`` inside a shell transcript or a diff is not a list item anyone cites,
    and counting it would make the gate fire on a move that changed nothing.
    """
    out: Counter[str] = Counter()
    fence: str | None = None
    for line in (text or "").splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence is not None:
            if (
                fence_match
                and fence_match.group(1)[0] == fence[0]
                and len(fence_match.group(1)) >= len(fence)
            ):
                fence = None
            continue
        if fence_match:
            fence = fence_match.group(1)
            continue
        m = _NUMBERED_ITEM_RE.match(line)
        if m:
            out[m.group(1)] += 1
    return out


def numbering_drift(
    *,
    source_before: str,
    target_before: str,
    source_after: str,
    target_after: str,
) -> dict[str, int]:
    """What the numbering gate would report: ``{number: delta}``, empty when clean.

    A positive delta is a number that appeared from nowhere, a negative one is a
    number that vanished. Both are failures; a pure move produces neither.
    """
    before = numbered_items(source_before) + numbered_items(target_before)
    after = numbered_items(source_after) + numbered_items(target_after)
    drift: dict[str, int] = {}
    for number in set(before) | set(after):
        delta = after[number] - before[number]
        if delta:
            drift[number] = delta
    return drift


def describe_drift(drift: dict[str, int]) -> str:
    """One line naming what changed, for the refusal message a user reads."""
    if not drift:
        return ""
    parts = [
        f"{number}. x{abs(delta)} {'added' if delta > 0 else 'lost'}"
        for number, delta in sorted(drift.items(), key=lambda kv: int(kv[0]))
    ]
    return (
        "numbered list items changed across the two files ("
        + "; ".join(parts)
        + ") — relocation moves text and must never renumber it, and these "
        "numbers are cited from elsewhere in the codebase"
    )
