"""Session map (lens ①): the story-derived board data for a session.

The Map's synchronized-swimlane board plots, over one shared time axis, a
session's *phase* arc, its *tool events*, its *context growth* and its *cost
burn*. This module computes the two parts that come straight from the
reconstructed Story (``core.transcript.build_session_asks``):

  * ``events`` — one entry per literal tool call, flattened across every ask and
    step into a single time-ordered list, each tagged with a coarse activity
    ``category`` (read / search / edit / bash / task / web / other / error) and
    the descriptive ``phase`` of the step it belongs to.
  * ``phases`` — the contiguous phase spans across the whole session, derived
    from those per-step phase labels (reusing ``core.phases.segment_phases`` for
    the labelling), expressed as ordinal (and, when present, timestamp) ranges.

The token/cost time-series (``context_series`` / ``cost_series``) are NOT built
here — they come from real spans in the API route. This module is a PURE,
DETERMINISTIC transform with NO I/O, NO LLM, NO interpretation: every field is a
count, an ordinal, a timestamp the transcript already carried, or a label the
agent itself produced. Never imports ``tokenjam.api`` / ``tokenjam.cli``.
"""
from __future__ import annotations

from typing import Any

from tokenjam.core.phases import segment_phases
from tokenjam.core.workmap import (
    _BASH_TOOLS,
    _FILE_TOOLS,
    _SEARCH_TOOLS,
    _SPAWN_TOOLS,
    _WEB_TOOLS,
)

#: Within ``_FILE_TOOLS`` the read-only tool is split out so the board can show
#: "read" apart from "edit" (the write-ish file tools). Every other file tool
#: (Edit / Write / NotebookEdit / MultiEdit) is an edit.
_READ_TOOLS = frozenset({"Read"})


def _category(tool: dict[str, Any]) -> str:
    """Coarse activity category for one tool call.

    A failed call is ``"error"`` regardless of which tool failed; otherwise the
    category comes from the work-map tool sets (with ``_FILE_TOOLS`` split into
    read vs edit). Unknown tools fall through to ``"other"``.
    """
    if tool.get("status") == "error":
        return "error"
    name = tool.get("name") or ""
    if name in _READ_TOOLS:
        return "read"
    if name in _FILE_TOOLS:
        return "edit"
    if name in _SEARCH_TOOLS:
        return "search"
    if name in _BASH_TOOLS:
        return "bash"
    if name in _SPAWN_TOOLS:
        return "task"
    if name in _WEB_TOOLS:
        return "web"
    return "other"


def _step_phase_names(steps: list[dict[str, Any]]) -> list[str | None]:
    """Phase title for each real (non-omitted) step, parallel to ``steps``.

    Reuses ``segment_phases`` for the titles and replicates its boundary rule
    (every narrated step opens a new phase; tool-only steps attach to the
    current one) to align each step to a title. Returns all-``None`` when the ask
    is too short to segment or resolves to a single phase (``segment_phases``
    returns ``[]``) — a short ask has no journey to label.
    """
    seg = segment_phases(steps)
    titles = [p["title"] for p in seg if isinstance(p, dict) and "omitted" not in p]
    names: list[str | None] = []
    if not titles:
        # No segmentation: every real step is unlabelled.
        for step in steps:
            if isinstance(step, dict) and "omitted" not in step:
                names.append(None)
        return names

    idx = -1
    for step in steps:
        if not isinstance(step, dict) or "omitted" in step:
            continue
        text = (step.get("text") or "").strip()
        if text or idx == -1:
            idx += 1
        names.append(titles[idx] if 0 <= idx < len(titles) else None)
    return names


def _phase_spans(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse a single ask's events into contiguous phase spans.

    Consecutive events sharing a (non-``None``) phase name fold into one span
    carrying its ordinal range and, when the underlying steps had timestamps,
    its ts range. Unlabelled events (phase ``None``) contribute no span.
    """
    spans: list[dict[str, Any]] = []
    for event in events:
        name = event.get("phase")
        if name is None:
            continue
        ordinal = event["ordinal"]
        ts = event.get("ts")
        if spans and spans[-1]["name"] == name:
            span = spans[-1]
            span["end_ordinal"] = ordinal
            if ts is not None:
                if span["start_ts"] is None:
                    span["start_ts"] = ts
                span["end_ts"] = ts
        else:
            spans.append({
                "name": name,
                "start_ordinal": ordinal,
                "end_ordinal": ordinal,
                "start_ts": ts,
                "end_ts": ts,
            })
    return spans


def build_session_map(asks_payload: dict[str, Any]) -> dict[str, Any]:
    """Story-derived board data for a session: ``events`` + ``phases``.

    ``asks_payload`` is the dict from ``core.transcript.build_session_asks``
    (``{"asks": [{n, prompt, ts, steps:[{ts, text, tools:[{name, label,
    status}], is_retry, ...}], ...}]}``). Returns::

        {
          "events": [{ts, ordinal, category, label, is_retry, phase}, ...],
          "phases": [{name, start_ordinal, end_ordinal, start_ts, end_ts}, ...],
        }

    ``events`` is one entry per literal tool call, in transcript order, with a
    session-global 0-based ``ordinal``. ``ts`` is the owning step's timestamp or
    ``None`` (the ordinal is the always-present fallback axis position).
    ``{"omitted": N}`` cap markers and tool-less steps contribute no events.
    Phase spans are built per ask (never merged across asks) so each span
    belongs to exactly one exchange. Pure + deterministic; empty in, empty out.
    """
    events: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    ordinal = 0

    for ask in asks_payload.get("asks") or []:
        if not isinstance(ask, dict):
            continue
        steps = ask.get("steps") or []
        phase_names = _step_phase_names(steps)

        ask_events: list[dict[str, Any]] = []
        real_index = 0
        for step in steps:
            if not isinstance(step, dict) or "omitted" in step:
                continue
            phase = (
                phase_names[real_index] if real_index < len(phase_names) else None
            )
            real_index += 1
            ts = step.get("ts")
            is_retry = bool(step.get("is_retry"))
            for tool in step.get("tools") or []:
                if not isinstance(tool, dict):
                    continue
                event = {
                    "ts": ts,
                    "ordinal": ordinal,
                    "category": _category(tool),
                    "label": tool.get("label") or "",
                    "is_retry": is_retry,
                    "phase": phase,
                }
                events.append(event)
                ask_events.append(event)
                ordinal += 1

        phases.extend(_phase_spans(ask_events))

    return {"events": events, "phases": phases}


__all__ = ["build_session_map"]
