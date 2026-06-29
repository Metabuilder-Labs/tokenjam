"""Method spine: fold a session Story into an ordered list of intent-tagged
"moves" describing *how* an agent attempted the work — recursively for subagents.

This is the deterministic data behind the **Approach** surface. It folds the
Story produced by ``core/transcript.build_session_story`` (the agent's own
narration + literal tool calls + ok/error outcomes) into one *move* per step,
tagging each with a ``kind`` that is the agent's method at that step.

HONESTY BOUNDARY (Critical Rule 14). This module emits ONLY intent it can
determine deterministically from structure the transcript already carries:

  * ``delegate`` — the step spawned a subagent (a ``Task``/``Agent`` tool, or a
    resolved ``subagent``/``subagents`` child).
  * ``dead_end`` — the step is a retry of the previous one, OR it ran a revert
    command (``git checkout -- / restore / revert / reset``).
  * ``verify``   — the step ran a recognised test runner (pytest, npm test, go
    test, cargo test, jest, …).
  * ``act``      — everything else (the default).

There are EXACTLY these four kinds. Richer intent labels —
``hypothesize`` / ``reproduce`` / ``insight`` and the like — are a JUDGEMENT,
not a structural fact, so they are NOT emitted here. They belong to the opt-in
LLM distill layer (``core/distill.py``), which is out of scope for this module.
Each move also carries a ``source`` (``agent_words`` when the label is the
agent's own narration, ``structural`` when synthesised from the tools) so a
surface never claims more than the data supports. No move asserts an approach
was good or bad.

Pure module: no I/O; imports only ``core``. Never imports ``tokenjam.api`` /
``tokenjam.cli``.
"""
from __future__ import annotations

import re
from typing import Any

from tokenjam.core.workmap import (
    _BASH_TOOLS,
    _FILE_TOOLS,
    _SEARCH_TOOLS,
    _SPAWN_TOOLS,
    _WEB_TOOLS,
)

#: Max chars for a move's one-line ``label`` (the spine reads as a list).
MAX_LABEL_CHARS = 80
#: Max chars for a move's ``quote`` (the narration first paragraph).
MAX_QUOTE_CHARS = 600

#: A Bash command that UNDOES prior work — a structural dead-end signal. Matches
#: the destructive/revert git verbs only (``git status``/``git diff`` are safe).
_REVERT_RE = re.compile(r"git\s+(checkout\s+--|restore|revert|reset)")

#: A Bash command that runs a test suite — a structural ``verify`` signal. Kept
#: to the common cross-ecosystem runners; matched against the tool's label
#: (which, for Bash, is the command).
_TEST_RUNNER_RE = re.compile(
    r"pytest|unittest|\bnpm\s+(run\s+)?test|yarn\s+test|go\s+test"
    r"|cargo\s+test|make\s+test|tox\b|jest|vitest"
)

#: The cap markers a subagent dict carries instead of an expanded story when a
#: recursion guard tripped (mirrors ``core/workmap._CAP_MARKERS``).
_CAP_MARKERS = {
    "depth_capped": "depth",
    "budget_capped": "budget",
    "cycle": "cycle",
}

#: Per-tool display verb for a synthesised (no-narration) structural label.
_TOOL_VERBS: dict[str, str] = {
    "Read": "read",
    "Edit": "edit",
    "Write": "write",
    "NotebookEdit": "edit",
    "MultiEdit": "edit",
    "Bash": "bash",
    "Grep": "search",
    "Glob": "search",
    "WebFetch": "fetch",
    "WebSearch": "search",
    "Task": "delegate",
    "Agent": "delegate",
}


def _trim(text: str, limit: int) -> str:
    """Whitespace-trim ``text`` and cap it to ``limit`` chars with an ellipsis."""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _first_line(text: str) -> str:
    """First non-empty, stripped line of ``text`` (or "")."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _quote(text: str) -> str | None:
    """The narration's first paragraph (up to the first blank line), or None.

    Deterministic: never an interpretation, just the agent's own opening words,
    length-capped. Returns None when the step carries no narration.
    """
    text = text.strip()
    if not text:
        return None
    paragraph = text.split("\n\n", 1)[0].strip()
    return _trim(paragraph, MAX_QUOTE_CHARS) or None


def _bash_labels(tools: list[dict[str, Any]]) -> list[str]:
    """The labels (commands) of every Bash tool in ``tools``."""
    return [
        (t.get("label") or "")
        for t in tools
        if t.get("name") in _BASH_TOOLS
    ]


def _has_spawn(step: dict[str, Any]) -> bool:
    """True if the step spawned a subagent (a Task/Agent tool or a resolved
    child under ``subagent``/``subagents``)."""
    if step.get("subagent") or step.get("subagents"):
        return True
    return any(t.get("name") in _SPAWN_TOOLS for t in step.get("tools") or [])


def _is_dead_end(step: dict[str, Any]) -> bool:
    """True if the step is a retry, or ran a revert command."""
    if step.get("is_retry"):
        return True
    return any(_REVERT_RE.search(label) for label in _bash_labels(step.get("tools") or []))


def _is_verify(step: dict[str, Any]) -> bool:
    """True if the step ran a recognised test runner."""
    return any(
        _TEST_RUNNER_RE.search(label) for label in _bash_labels(step.get("tools") or [])
    )


def _kind(step: dict[str, Any]) -> str:
    """Deterministic move kind by precedence: delegate > dead_end > verify > act."""
    if _has_spawn(step):
        return "delegate"
    if _is_dead_end(step):
        return "dead_end"
    if _is_verify(step):
        return "verify"
    return "act"


def _one_tool_label(tool: dict[str, Any]) -> str:
    """Structural label for a step whose single tool stands in for the narration."""
    name = tool.get("name") or "tool"
    label = (tool.get("label") or "").strip()
    if name in _BASH_TOOLS:
        return f"bash: {label}" if label else "bash"
    if name in _FILE_TOOLS:
        verb = "read" if name == "Read" else "edit"
        base = label.rsplit("/", 1)[-1] if label else ""
        return f"{verb} {base}".strip()
    verb = _TOOL_VERBS.get(name, name.lower())
    return f"{verb} {label}".strip()


def _structural_label(tools: list[dict[str, Any]]) -> str:
    """Deterministic label synthesised from a step's tools (no narration).

    A single tool reads as ``verb target`` (e.g. ``edit workmap.py``,
    ``bash: pytest``); multiple tools roll up into category counts
    (e.g. ``read 3 files``).
    """
    if not tools:
        return "(no narration)"
    if len(tools) == 1:
        return _one_tool_label(tools[0])

    reads = sum(1 for t in tools if t.get("name") == "Read")
    edits = sum(
        1 for t in tools if t.get("name") in _FILE_TOOLS and t.get("name") != "Read"
    )
    searches = sum(1 for t in tools if t.get("name") in _SEARCH_TOOLS)
    commands = sum(1 for t in tools if t.get("name") in _BASH_TOOLS)
    spawns = sum(1 for t in tools if t.get("name") in _SPAWN_TOOLS)
    webs = sum(1 for t in tools if t.get("name") in _WEB_TOOLS)
    known = _FILE_TOOLS | _SEARCH_TOOLS | _BASH_TOOLS | _SPAWN_TOOLS | _WEB_TOOLS
    others = sum(1 for t in tools if t.get("name") not in known)

    parts: list[str] = []
    if reads:
        parts.append(f"read {reads} file{'s' if reads != 1 else ''}")
    if edits:
        parts.append(f"edit {edits} file{'s' if edits != 1 else ''}")
    if searches:
        parts.append(f"{searches} search{'es' if searches != 1 else ''}")
    if commands:
        parts.append(f"{commands} command{'s' if commands != 1 else ''}")
    if spawns:
        parts.append(f"{spawns} delegation{'s' if spawns != 1 else ''}")
    if webs:
        parts.append(f"{webs} web fetch{'es' if webs != 1 else ''}")
    if others:
        parts.append(f"{others} tool call{'s' if others != 1 else ''}")
    return ", ".join(parts) if parts else "(no narration)"


def _evidence(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact per-tool summary: ``{name, label, status}`` (no inputs/outputs)."""
    return [
        {
            "name": t.get("name"),
            "label": t.get("label") or "",
            "status": t.get("status") or "ok",
        }
        for t in tools
    ]


def _subagent_dicts(step: dict[str, Any]) -> list[dict[str, Any]]:
    """The subagent child dict(s) attached to a step (``subagent`` and/or
    ``subagents``), in order."""
    children: list[dict[str, Any]] = []
    sub = step.get("subagent")
    if isinstance(sub, dict):
        children.append(sub)
    for s in step.get("subagents") or []:
        if isinstance(s, dict):
            children.append(s)
    return children


def _cap_marker(sub: dict[str, Any]) -> str | None:
    """The cap reason (``depth``/``budget``/``cycle``) if the child wasn't
    expanded, else None."""
    return next((label for key, label in _CAP_MARKERS.items() if sub.get(key)), None)


def _delegate_children(
    step: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    """Recursively build the spine(s) of a delegate step's subagent child stories.

    Returns ``(children, capped)`` — the concatenated child spines, plus the cap
    reason when a child hit a recursion guard (depth/budget/cycle) and so has no
    expanded story. Capped children contribute no moves (the marker carries the
    honest "not expanded" signal instead of inventing one).
    """
    children: list[dict[str, Any]] = []
    capped: str | None = None
    for sub in _subagent_dicts(step):
        cap = _cap_marker(sub)
        if cap is not None:
            capped = capped or cap
            continue
        children.extend(build_method_spine(sub))
    return children, capped


def _move(step: dict[str, Any]) -> dict[str, Any]:
    """Fold one Story step into one method-spine move (deterministic)."""
    tools = step.get("tools") or []
    text = step.get("text") or ""
    kind = _kind(step)

    if text.strip():
        label = _trim(_first_line(text), MAX_LABEL_CHARS)
        source = "agent_words"
    else:
        label = _structural_label(tools)
        source = "structural"

    step_errored = bool(step.get("is_error")) or any(
        t.get("status") == "error" for t in tools
    )

    move: dict[str, Any] = {
        "kind": kind,
        "label": label,
        "source": source,
        "quote": _quote(text),
        "evidence": _evidence(tools),
        "is_retry": bool(step.get("is_retry")),
        "failed": kind == "verify" and step_errored,
    }

    if kind == "delegate":
        children, capped = _delegate_children(step)
        move["children"] = children
        if capped is not None:
            move["capped"] = capped

    return move


def build_method_spine(story: dict[str, Any]) -> list[dict[str, Any]]:
    """Fold a session Story into an ordered list of intent-tagged moves.

    ``story`` is the dict from ``core/transcript.build_session_story`` (or a
    persisted snapshot of it, or a nested subagent story — same schema). Each
    real step becomes one move; ``{"omitted": N}`` cap markers are skipped.
    ``delegate`` moves carry the recursively-built spines of their subagents
    under ``children`` (with a ``capped`` marker when a child wasn't expanded).

    Deterministic and honesty-bounded: see the module docstring. Returns ``[]``
    for an empty/absent story.
    """
    steps = story.get("steps") or []
    return [_move(step) for step in steps if "omitted" not in step]


__all__ = ["build_method_spine"]
