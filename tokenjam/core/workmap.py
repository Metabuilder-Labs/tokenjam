"""Agent work map: a compact, render-ready tree of what a Claude Code session
and its subagents actually did.

This module is a PURE TRANSFORM. It folds two already-computed inputs into one
annotated tree:

  * the deterministic session *story* (``core.transcript.build_session_story``),
    which carries the nested subagent tree with human names + per-step tool
    labels straight from the on-disk transcript, and
  * the per-subagent cost/token/flag breakdown (the API's ``_session_subagents``)
    derived from spans.

For each node it emits a deterministic activity rollup — files touched, web
sources, code searches, shell commands, subagents spawned, errors/retries —
joined to that node's cost / tokens / right-sizing flags.

There is NO interpretation here. Every field is a count, a label the agent
itself produced, or a cost summed from real spans. It never groups steps into
"approaches" or judges quality: it reports structure + activity so a human can
see, at a glance, what a long run actually did. Caps the story applied
(depth / step-budget / cycle) are surfaced as node markers, and subagents that
have recorded cost but never made it into the (bounded) tree are reported as an
``unmapped`` tail — never silently dropped.

Pure module: no I/O; never imports ``tokenjam.api`` / ``tokenjam.cli``.
"""
from __future__ import annotations

from typing import Any

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


def _node_from_story(
    story: dict[str, Any],
    *,
    node_id: str,
    name: str,
    depth: int,
    is_root: bool,
    sub_index: dict[str, dict[str, Any]],
    cost_usd: float | None,
    tokens: int | None,
    flags: list[str],
) -> dict[str, Any]:
    """Build one node from a story-shaped dict and recurse into its subagents."""
    steps = story.get("steps") or []
    activity, model = _rollup_steps(steps)

    children: list[dict[str, Any]] = []
    for step in steps:
        if "omitted" in step:
            continue
        if isinstance(step.get("subagent"), dict):
            children.append(_child_node(step["subagent"], depth=depth + 1,
                                        sub_index=sub_index))
        for sub in step.get("subagents") or []:
            if isinstance(sub, dict):
                children.append(_child_node(sub, depth=depth + 1,
                                            sub_index=sub_index))

    return {
        "id": node_id,
        "name": name,
        "depth": depth,
        "is_root": is_root,
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
        "children": children,
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
            "id": agent_id, "name": name, "depth": depth, "is_root": False,
            "model": (row.get("model") if row else None),
            "task": "", "outcome": "",
            "activity": _empty_activity(), "summary": _CAP_SUMMARY[cap],
            "cost_usd": cost_usd, "tokens": tokens, "flags": flags,
            "capped": cap, "truncated": False, "children": [],
        }

    return _node_from_story(
        sub, node_id=agent_id, name=name, depth=depth, is_root=False,
        sub_index=sub_index, cost_usd=cost_usd, tokens=tokens, flags=flags,
    )


def build_work_map(
    story: dict[str, Any],
    subagents: dict[str, Any] | None,
    *,
    root_cost_usd: float | None = None,
    root_tokens: int | None = None,
    root_label: str = "Main agent",
) -> dict[str, Any]:
    """Fold a session ``story`` + per-subagent ``subagents`` breakdown into a
    render-ready work-map tree.

    ``story`` is the dict from ``build_session_story`` (the main-thread story
    with nested ``subagent``/``subagents`` objects). ``subagents`` is the
    ``_session_subagents`` breakdown (``{"rows": [...]}`` keyed by
    ``sub_agent_id``) used to annotate each node with cost / tokens / flags;
    the root carries the session totals passed via ``root_cost_usd`` /
    ``root_tokens``.

    Returns ``{root, node_count, subagent_count, max_depth, flagged, capped,
    truncated, unmapped_count, unmapped_cost_usd}``. ``unmapped_*`` reports
    subagents that have recorded cost but never appeared in the (bounded) tree
    — surfaced so the map never under-represents a deep/wide run silently.
    """
    rows = (subagents or {}).get("rows") or []
    sub_index = {
        str(r["sub_agent_id"]): r for r in rows if r.get("sub_agent_id")
    }

    root = _node_from_story(
        story, node_id="main", name=root_label, depth=0, is_root=True,
        sub_index=sub_index, cost_usd=root_cost_usd, tokens=root_tokens, flags=[],
    )

    seen: set[str] = set()
    stats = {"node_count": 0, "subagent_count": 0, "max_depth": 0,
             "flagged": 0, "capped": 0, "truncated": False}

    def _walk(node: dict[str, Any]) -> None:
        stats["node_count"] += 1
        if not node["is_root"]:
            stats["subagent_count"] += 1
            if node["id"]:
                seen.add(node["id"])
        stats["max_depth"] = max(stats["max_depth"], node["depth"])
        if node["flags"]:
            stats["flagged"] += 1
        if node["capped"]:
            stats["capped"] += 1
        if node["truncated"]:
            stats["truncated"] = True
        for child in node["children"]:
            _walk(child)

    _walk(root)

    unmapped = [r for sid, r in sub_index.items() if sid not in seen]
    unmapped_cost = sum(
        float(r["cost_usd"]) for r in unmapped if r.get("cost_usd") is not None
    )
    unmapped_tokens = sum(_row_tokens(r) for r in unmapped)

    return {
        "root": root,
        "node_count": stats["node_count"],
        "subagent_count": stats["subagent_count"],
        "max_depth": stats["max_depth"],
        "flagged": stats["flagged"],
        "capped": stats["capped"],
        "truncated": stats["truncated"],
        "unmapped_count": len(unmapped),
        "unmapped_cost_usd": unmapped_cost,
        "unmapped_tokens": unmapped_tokens,
    }


__all__ = ["build_work_map", "MAX_FILES_PER_NODE"]
