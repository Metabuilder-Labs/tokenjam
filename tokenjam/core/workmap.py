"""Agent work map: a render-ready view of what a Claude Code session did.

A session is NOT one task — it's a sequence of human *asks* (exchanges) fired
into the same terminal until the context window fills. So the work map is a list
of asks (newest first), each annotated with a deterministic activity rollup
(files touched, web sources, searches, shell commands, subagents spawned,
errors/retries) plus the subagent subtree that ask spawned, joined to each
subagent's cost / tokens / right-sizing flags.

This module is a PURE TRANSFORM. It folds two already-computed inputs:

  * the ask-segmented session story (``core.transcript.build_session_asks``),
    which carries each ask's prompt + steps + nested subagents with human names
    and per-step tool labels straight from the on-disk transcript, and
  * the per-subagent cost/token/flag breakdown (the API's ``_session_subagents``)
    derived from spans, plus per-ask token/cost totals bucketed by the caller.

There is NO interpretation: every field is a count, a label the agent itself
produced, or a value summed from real spans. It never groups asks into inferred
"tasks" or judges quality — it reports structure + activity so a human can see,
at a glance, what a long run actually did. Caps the story applied
(depth / step-budget / cycle) surface as node markers, and subagents with
recorded cost that never made it into the (bounded) tree are reported as an
``unmapped`` tail — never silently dropped.

Pure module: no I/O; never imports ``tokenjam.api`` / ``tokenjam.cli``.
"""
from __future__ import annotations

from typing import Any

from tokenjam.core.phases import segment_phases

# Tool-name -> activity category. Mirrors the label args the story already
# extracts (core/transcript._TOOL_LABEL_ARGS) so the rollup and the Timeline
# agree on what each tool "is".
_FILE_TOOLS = frozenset({"Read", "Edit", "Write", "NotebookEdit", "MultiEdit"})
_WEB_TOOLS = frozenset({"WebFetch", "WebSearch"})
_SEARCH_TOOLS = frozenset({"Grep", "Glob"})
_BASH_TOOLS = frozenset({"Bash"})
_SPAWN_TOOLS = frozenset({"Task", "Agent"})

#: Cap on the distinct file paths listed per node (the count stays exact).
MAX_FILES_PER_NODE = 8

#: Node provenance — how the agent was deployed, so the UI can mark in-session
#: subagents (full method, rebuilt from the transcript) apart from cross-terminal
#: child sessions (M2: spliced from the run tree, method may be session-level
#: only). Every node the work map builds today is an in-session sidechain.
PROVENANCE_IN_SESSION = "in_session_subagent"
PROVENANCE_CROSS_TERMINAL = "cross_terminal_child"

#: A subagent object carries one of these instead of an expanded story when the
#: story hit a recursion guard. Mapped to a short node ``capped`` marker.
_CAP_MARKERS = {
    "depth_capped": "depth",
    "budget_capped": "budget",
    "cycle": "cycle",
}
_CAP_SUMMARY = {
    "depth": "not expanded — max depth reached",
    "budget": "not expanded — step budget reached",
    "cycle": "cycle — already shown above",
}


def _empty_activity() -> dict[str, Any]:
    """Zeroed activity rollup (for cap/cycle nodes the story didn't expand)."""
    return {
        "steps": 0, "tool_calls": 0, "files": [], "file_count": 0,
        "source_count": 0, "search_count": 0, "bash_count": 0,
        "spawn_count": 0, "other_count": 0, "error_count": 0, "retry_count": 0,
    }


def _rollup_steps(steps: list[dict[str, Any]]) -> tuple[dict[str, Any], str | None]:
    """Deterministic activity rollup over one node's OWN steps + dominant model.

    Nested subagent steps live under each step's ``subagent``/``subagents`` key,
    not in this list, so there's no double-counting. ``{"omitted": N}`` markers
    (from the story's head+tail cap) are skipped.
    """
    files: list[str] = []
    seen_files: set[str] = set()
    sources: set[str] = set()
    search = bash = spawn = other = 0
    tool_calls = error_calls = step_count = retry_count = 0
    model_freq: dict[str, int] = {}

    for step in steps:
        if "omitted" in step:
            continue
        step_count += 1
        if step.get("is_retry"):
            retry_count += 1
        model = step.get("model")
        if isinstance(model, str) and model:
            model_freq[model] = model_freq.get(model, 0) + 1
        for tool in step.get("tools", []):
            name = tool.get("name") or ""
            label = (tool.get("label") or "").strip()
            tool_calls += 1
            if tool.get("status") == "error":
                error_calls += 1
            if name in _FILE_TOOLS:
                if label and label not in seen_files:
                    seen_files.add(label)
                    if len(files) < MAX_FILES_PER_NODE:
                        files.append(label)
            elif name in _WEB_TOOLS:
                if label:
                    sources.add(label)
            elif name in _SEARCH_TOOLS:
                search += 1
            elif name in _BASH_TOOLS:
                bash += 1
            elif name in _SPAWN_TOOLS:
                spawn += 1
            else:
                other += 1

    dominant = max(model_freq, key=lambda m: model_freq[m]) if model_freq else None
    activity = {
        "steps": step_count,
        "tool_calls": tool_calls,
        "files": files,
        "file_count": len(seen_files),
        "source_count": len(sources),
        "search_count": search,
        "bash_count": bash,
        "spawn_count": spawn,
        "other_count": other,
        "error_count": error_calls,
        "retry_count": retry_count,
    }
    return activity, dominant


def _plural(n: int, word: str, suffix: str = "s") -> str:
    return f"{n} {word}" + ("" if n == 1 else suffix)


def _summary(activity: dict[str, Any]) -> str:
    """One-line human summary of a node's activity (presentation convenience)."""
    parts: list[str] = []
    if activity["source_count"]:
        parts.append(_plural(activity["source_count"], "source"))
    if activity["file_count"]:
        parts.append(_plural(activity["file_count"], "file"))
    if activity["search_count"]:
        parts.append(_plural(activity["search_count"], "search", "es"))
    if activity["bash_count"]:
        parts.append(_plural(activity["bash_count"], "cmd"))
    if activity["spawn_count"]:
        parts.append(_plural(activity["spawn_count"], "subagent"))
    if not parts:
        if activity["tool_calls"]:
            parts.append(_plural(activity["tool_calls"], "tool call"))
        else:
            return "no tool activity"
    if activity["error_count"]:
        parts.append(_plural(activity["error_count"], "error"))
    return " · ".join(parts)


def _row_tokens(row: dict[str, Any]) -> int:
    return int(
        (row.get("input_tokens") or 0)
        + (row.get("output_tokens") or 0)
        + (row.get("cache_tokens") or 0)
        + (row.get("cache_write_tokens") or 0)
    )


def _children_from_steps(
    steps: list[dict[str, Any]],
    depth: int,
    sub_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Subagent child nodes attached to a node's steps (one level down)."""
    children: list[dict[str, Any]] = []
    for step in steps:
        if "omitted" in step:
            continue
        if isinstance(step.get("subagent"), dict):
            children.append(_child_node(step["subagent"], depth=depth,
                                        sub_index=sub_index))
        for sub in step.get("subagents") or []:
            if isinstance(sub, dict):
                children.append(_child_node(sub, depth=depth, sub_index=sub_index))
    return children


def _node_from_story(
    story: dict[str, Any],
    *,
    node_id: str,
    name: str,
    depth: int,
    sub_index: dict[str, dict[str, Any]],
    cost_usd: float | None,
    tokens: int | None,
    flags: list[str],
) -> dict[str, Any]:
    """Build one subagent node from a story-shaped dict, recursing into its own
    subagents."""
    steps = story.get("steps") or []
    activity, model = _rollup_steps(steps)
    return {
        "id": node_id,
        "name": name,
        "depth": depth,
        "model": model,
        "task": story.get("task") or "",
        "outcome": story.get("outcome") or "",
        "activity": activity,
        "summary": _summary(activity),
        "cost_usd": cost_usd,
        "tokens": tokens,
        "flags": flags,
        "capped": None,
        "truncated": bool(story.get("truncated")),
        "provenance": PROVENANCE_IN_SESSION,
        "capture_completeness": "capped" if story.get("truncated") else "full",
        "children": _children_from_steps(steps, depth + 1, sub_index),
    }


def _child_node(
    sub: dict[str, Any],
    *,
    depth: int,
    sub_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a subagent node, joining cost/flags by agent_id.

    A cap/cycle marker means the story didn't expand this child — but its cost
    may still be known from spans, so we surface both (cost + the cap reason).
    """
    agent_id = str(sub.get("agent_id") or "")
    name = sub.get("name") or (f"agent-{agent_id[:8]}" if agent_id else "subagent")
    row = sub_index.get(agent_id)
    cost_usd = float(row["cost_usd"]) if row and row.get("cost_usd") is not None else None
    tokens = _row_tokens(row) if row else None
    flags = list(row["flags"]) if row and row.get("flags") else []

    cap = next((label for marker, label in _CAP_MARKERS.items() if sub.get(marker)), None)
    if cap is not None:
        return {
            "id": agent_id, "name": name, "depth": depth, "model": row.get("model") if row else None,
            "task": "", "outcome": "", "activity": _empty_activity(), "summary": _CAP_SUMMARY[cap],
            "cost_usd": cost_usd, "tokens": tokens, "flags": flags,
            "capped": cap, "truncated": False,
            "provenance": PROVENANCE_IN_SESSION, "capture_completeness": "capped",
            "children": [],
        }

    return _node_from_story(
        sub, node_id=agent_id, name=name, depth=depth, sub_index=sub_index,
        cost_usd=cost_usd, tokens=tokens, flags=flags,
    )


def _ask_node(
    ask: dict[str, Any],
    sub_index: dict[str, dict[str, Any]],
    tokens: int | None,
    cost: float | None,
) -> dict[str, Any]:
    """Build a top-level ask node: the exchange's own activity + its subtree."""
    steps = ask.get("steps") or []
    activity, model = _rollup_steps(steps)
    return {
        "n": ask.get("n"),
        "prompt": ask.get("prompt") or "",
        "ts": ask.get("ts"),
        "model": model,
        "activity": activity,
        "summary": _summary(activity),
        "outcome": ask.get("outcome") or "",
        "tokens": tokens,
        "cost_usd": cost,
        "truncated": bool(ask.get("truncated")),
        # Full pre-cap step count (activity["steps"] counts only the capped list)
        # and the descriptive phase breakdown of a long ask (Task E).
        "step_count": ask.get("step_count"),
        "phases": segment_phases(steps),
        "subagents": _children_from_steps(steps, depth=1, sub_index=sub_index),
    }


def _accumulate(node: dict[str, Any], acc: dict[str, Any]) -> None:
    """Walk a subagent node + descendants into an accumulator dict."""
    acc["count"] += 1
    if node["id"]:
        acc["seen"].add(node["id"])
    acc["max_depth"] = max(acc["max_depth"], node["depth"])
    if node["flags"]:
        acc["flagged"] += 1
    if node["capped"]:
        acc["capped"] += 1
    if node["truncated"]:
        acc["truncated"] = True
    for child in node["children"]:
        _accumulate(child, acc)


def build_work_map(
    asks_payload: dict[str, Any],
    subagents: dict[str, Any] | None,
    *,
    ask_tokens: dict[int, int] | None = None,
    ask_costs: dict[int, float] | None = None,
    session_tokens: int | None = None,
    session_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Fold ask-segmented story + per-subagent breakdown into a render-ready list
    of ask nodes (newest first), each with its subagent subtree.

    ``asks_payload`` is the dict from ``build_session_asks`` (``{"asks": [...]}``
    in chronological order). ``subagents`` is the ``_session_subagents`` breakdown
    used to annotate each subagent node with cost / tokens / flags by
    ``sub_agent_id``. ``ask_tokens`` / ``ask_costs`` map an ask's ``n`` to its
    bucketed token/cost total (computed by the caller from span timestamps).

    Returns ``{asks (newest first), ask_count, subagent_count, max_depth,
    flagged, capped, truncated, unmapped_count, unmapped_tokens,
    unmapped_cost_usd, session_tokens, session_cost_usd}``. ``unmapped_*``
    reports subagents that have recorded cost but never appeared in the (bounded)
    tree — surfaced so the map never under-represents a deep/wide run silently.
    """
    rows = (subagents or {}).get("rows") or []
    sub_index = {str(r["sub_agent_id"]): r for r in rows if r.get("sub_agent_id")}
    ask_tokens = ask_tokens or {}
    ask_costs = ask_costs or {}

    nodes: list[dict[str, Any]] = []
    g: dict[str, Any] = {"count": 0, "flagged": 0, "capped": 0,
                         "max_depth": 0, "truncated": False, "seen": set()}

    for ask in asks_payload.get("asks") or []:
        n = ask.get("n")
        node = _ask_node(ask, sub_index, ask_tokens.get(n), ask_costs.get(n))

        local: dict[str, Any] = {"count": 0, "flagged": 0, "capped": 0,
                                 "max_depth": 0, "truncated": False, "seen": set()}
        for child in node["subagents"]:
            _accumulate(child, local)
        node["subagent_count"] = local["count"]
        node["flagged"] = local["flagged"]

        g["count"] += local["count"]
        g["flagged"] += local["flagged"]
        g["capped"] += local["capped"]
        g["max_depth"] = max(g["max_depth"], local["max_depth"])
        g["seen"] |= local["seen"]
        if local["truncated"] or node["truncated"]:
            g["truncated"] = True
        nodes.append(node)

    nodes.reverse()  # newest ask first

    unmapped = [r for sid, r in sub_index.items() if sid not in g["seen"]]
    unmapped_tokens = sum(_row_tokens(r) for r in unmapped)
    unmapped_cost = sum(
        float(r["cost_usd"]) for r in unmapped if r.get("cost_usd") is not None
    )

    return {
        "asks": nodes,
        "ask_count": len(nodes),
        "subagent_count": g["count"],
        "max_depth": g["max_depth"],
        "flagged": g["flagged"],
        "capped": g["capped"],
        "truncated": g["truncated"],
        "unmapped_count": len(unmapped),
        "unmapped_tokens": unmapped_tokens,
        "unmapped_cost_usd": unmapped_cost,
        "session_tokens": session_tokens,
        "session_cost_usd": session_cost_usd,
    }


__all__ = [
    "build_work_map",
    "MAX_FILES_PER_NODE",
    "PROVENANCE_IN_SESSION",
    "PROVENANCE_CROSS_TERMINAL",
]
