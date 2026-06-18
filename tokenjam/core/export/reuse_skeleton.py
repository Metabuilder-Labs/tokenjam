"""
Skeleton rendering for the Reuse analyzer's HTML / Markdown report (#116).

Given the planning text of a cluster's representative session plus the planning
texts of the other example sessions, produce a *skeleton*: the shared literal
structure with the parts that vary across runs replaced by `{{slot_N}}` markers.
The accompanying `slot_map` records the divergent values per slot so a reviewer
can see exactly what changes between runs.

This is deliberately simple in v1 — a position-wise comparison over
whitespace-tokenized text. It won't realign after an insertion/deletion (that
needs proper sequence alignment, which is forward-compatible work), but it's
enough to name the variables a user reviews before turning the skeleton into a
slash command or script. Separated from the analyzer so the report command and
the `--export-templates` shortcut share one implementation.
"""
from __future__ import annotations

# Cap on distinct variable slots. Past this, a cluster's plans diverge so much
# that the "skeleton" is mostly placeholders — the caller flags it as a weak
# match. Further divergences collapse into a single ellipsis marker so the
# skeleton stays readable instead of sprouting hundreds of slots.
MAX_SLOTS = 20

_OVERFLOW_MARKER = "{{…}}"


def render_skeleton(
    plan_text: str,
    example_texts: list[str],
) -> tuple[str, dict[str, list[str]]]:
    """
    Return ``(skeleton_with_slots, slot_map)``.

    - ``skeleton_with_slots``: ``plan_text`` with positions that diverge across
      the examples replaced by ``{{slot_1}}``, ``{{slot_2}}``, … (and
      ``{{…}}`` once ``MAX_SLOTS`` is exhausted).
    - ``slot_map``: ``{slot_name: [sorted distinct values seen at that
      position]}``.

    A position is *literal* only when every example has a token there and they
    all equal the plan token; otherwise it's a slot. ``example_texts`` is the
    comparison set (the plan_text itself need not be included).
    """
    plan_tokens = plan_text.split()
    example_token_lists = [t.split() for t in example_texts]

    out: list[str] = []
    slot_map: dict[str, list[str]] = {}
    slot_count = 0

    for i, tok in enumerate(plan_tokens):
        aligned = [toks[i] for toks in example_token_lists if i < len(toks)]
        all_agree = (
            len(aligned) == len(example_token_lists)
            and all(a == tok for a in aligned)
        )
        if all_agree or not example_token_lists:
            out.append(tok)
            continue
        # Divergent position → a slot.
        if slot_count < MAX_SLOTS:
            slot_count += 1
            name = f"slot_{slot_count}"
            slot_map[name] = sorted({tok, *aligned})
            out.append(f"{{{{{name}}}}}")
        else:
            out.append(_OVERFLOW_MARKER)

    return " ".join(out), slot_map


def is_weak_match(slot_map: dict[str, list[str]]) -> bool:
    """A skeleton is 'weak' once it hits the slot cap — mostly placeholders."""
    return len(slot_map) >= MAX_SLOTS
