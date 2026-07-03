"""Deterministic phase segmentation within a single ask (Task E).

A long autonomous ask — a headless worker spawned by a harness, a ``/govern``
loop — is ONE human exchange, so the work map collapses it to a single milestone
showing only its final outcome. That hides the journey: what the agent explored,
what it tried, how far it got, where the work went.

This splits an ask's step sequence into **phases** using the agent's OWN
narration as the boundary: every assistant text step starts a phase, and the
tool-only steps that follow attach to it (with their tool calls tallied). The
Map then renders the arc within the ask — "explored the adapter -> researched
the auth model -> wrote the fix -> ran validation -> cleaned up" — instead of one
line.

Purely descriptive (repo Critical Rule 14): every phase title is the agent's own
words (its first narrated sentence) and every tally is an exact count. No LLM, no
interpretation, no quality judgement. Pure module: no I/O; folds the step list
``core.transcript`` already built.
"""
from __future__ import annotations

import re
from typing import Any

#: An ask with fewer real steps than this already reads as one milestone — its
#: single headline is enough, so we don't segment it.
MIN_STEPS_TO_SEGMENT = 6

#: Hard cap on phases returned per ask — a payload bound, not the glance. The UI
#: shows a short preview and reveals the rest on "show all", so this is set high
#: enough that a typical long ask sends its WHOLE arc (the journey is the point).
#: Only a pathological ask exceeds it; beyond the cap we keep the first
#: ``HEAD_PHASES`` and last ``TAIL_PHASES`` with an explicit ``{"omitted": N}``
#: marker between them — never a silent drop (mirrors the head+tail step cap in
#: ``core.transcript``).
MAX_PHASES = 80
HEAD_PHASES = 55
TAIL_PHASES = 25

#: Phase title trim (characters).
MAX_PHASE_TITLE_CHARS = 90

#: Splits on the boundary after a sentence terminator followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _first_sentence(text: str) -> str:
    """First sentence of a narration blob, whitespace-collapsed and trimmed."""
    collapsed = " ".join(text.split())
    first = _SENTENCE_END.split(collapsed, 1)[0].strip()
    if len(first) > MAX_PHASE_TITLE_CHARS:
        first = first[:MAX_PHASE_TITLE_CHARS].rstrip() + "…"
    return first


def _new_phase() -> dict[str, Any]:
    return {"title": "", "_counts": {}, "_order": [], "tool_count": 0, "error_count": 0}


def _tally(phase: dict[str, Any], tools: list[dict[str, Any]]) -> None:
    """Fold a step's tool calls into the current phase's per-tool tally."""
    for tool in tools:
        name = tool.get("name") or ""
        if not name:
            continue
        if name not in phase["_counts"]:
            phase["_counts"][name] = 0
            phase["_order"].append(name)
        phase["_counts"][name] += 1
        phase["tool_count"] += 1
        if tool.get("status") == "error":
            phase["error_count"] += 1


def _finalize(phase: dict[str, Any]) -> dict[str, Any]:
    """Render-ready phase: title + ordered tool breakdown + counts."""
    return {
        "title": phase["title"],
        "tools": [{"name": n, "count": phase["_counts"][n]} for n in phase["_order"]],
        "tool_count": phase["tool_count"],
        "error_count": phase["error_count"],
    }


def segment_phases(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split an ask's steps into descriptive phases, or ``[]`` if not worth it.

    Returns a list of ``{title, tools:[{name,count}], tool_count, error_count}``
    phases (with at most one ``{"omitted": N}`` marker when capped), or an empty
    list when the ask is too short to segment or resolves to a single phase
    (one phase isn't a journey). ``steps`` is the ask's step list from
    ``core.transcript`` (may contain ``{"omitted": N}`` cap markers, which are
    skipped here). Never raises.
    """
    real_count = sum(
        1 for s in steps if isinstance(s, dict) and "omitted" not in s
    )
    if real_count < MIN_STEPS_TO_SEGMENT:
        return []

    phases: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for step in steps:
        if not isinstance(step, dict) or "omitted" in step:
            continue
        text = (step.get("text") or "").strip()
        if text or current is None:
            current = _new_phase()
            current["title"] = _first_sentence(text) if text else ""
            phases.append(current)
        _tally(current, step.get("tools") or [])

    # Drop empty noise (a phase with neither a title nor any tool activity).
    phases = [p for p in phases if p["title"] or p["tool_count"]]
    if len(phases) <= 1:
        return []  # one phase isn't a journey worth drawing

    out = [_finalize(p) for p in phases]
    if len(out) > MAX_PHASES:
        omitted = len(out) - HEAD_PHASES - TAIL_PHASES
        out = out[:HEAD_PHASES] + [{"omitted": omitted}] + out[-TAIL_PHASES:]
    return out


__all__ = ["segment_phases", "MIN_STEPS_TO_SEGMENT", "MAX_PHASES"]
