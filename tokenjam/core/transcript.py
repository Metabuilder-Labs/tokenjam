"""
Session Story: build a deterministic, step-by-step "story" of a Claude Code
session from its on-disk JSONL transcript.

This module SURFACES the prose the Claude Code agent already wrote — the text
blocks it emits between tool calls — threaded with the literal tool calls and
their ok/error outcomes. There is NO LLM, NO generation, and NO interpretation:
the Story is the transcript verbatim-trimmed.

Source of truth = Claude Code's on-disk session files at
``~/.claude/projects/<encoded-cwd>/<session_id>.jsonl``. The tj ``session_id``
equals the CC transcript filename, so the locator is a glob over the projects
root. SDK sessions have no CC transcript -> ``build_session_story`` returns None.

Observed record shapes (verified against real transcripts):
  - ``type == "assistant"``: ``message.content`` is a list of blocks.
      * ``type == "text"``    -> ``.text``  (the agent's narration)
      * ``type == "thinking"`` -> skipped (internal reasoning, not narration)
      * ``type == "tool_use"`` -> ``.name``, ``.id``, ``.input``
      ``message.model`` carries the model id; top-level ``.timestamp`` is ISO-8601.
  - ``type == "user"``: ``message.content`` is either a plain string (the user's
      prompt) or a list of ``tool_result`` blocks:
      ``{type: "tool_result", tool_use_id, content, is_error?}``.
      A tool failed iff its matching ``tool_result`` has ``is_error == true``
      (absent/false/null -> ok).

Privacy/size: this module NEVER includes full tool inputs or tool outputs. It
emits only a short per-tool ``label`` (one trimmed arg) and an ok/error status.
Content exposure is bounded to the agent's own narration + arg labels, all
length-capped. Pure module: reads only files; never imports ``tokenjam.api``.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any

# --- Caps & trims (bounded payload) -----------------------------------------

#: Hard cap on the number of assistant turns surfaced. Beyond this we keep the
#: first ``HEAD_STEPS`` and last ``TAIL_STEPS`` and insert an explicit
#: ``{"omitted": N}`` marker between them — never a silent drop.
MAX_STORY_STEPS = 400
HEAD_STEPS = 350
TAIL_STEPS = 50

#: Text trims (characters).
#: ``MAX_STEP_TEXT_CHARS`` is a safety guard against a pathological single blob,
#: NOT a preview trim — the UI shows only the first line collapsed and the full
#: narration when a step is expanded, so trimming here would make "expand" lie.
#: Keep it high enough that real assistant responses are never cut.
MAX_STEP_TEXT_CHARS = 100_000
MAX_TASK_OUTCOME_CHARS = 600
MAX_TOOL_LABEL_CHARS = 120

#: Subagent recursion guards (a 4h fan-out run can spawn a deep, wide tree).
#: ``MAX_SUBAGENT_DEPTH`` caps how many levels of nesting we descend; beyond it
#: an Agent/Task step carries ``subagent = {"depth_capped": True}``. A SHARED
#: ``step_budget`` is threaded across the WHOLE tree so the total payload stays
#: bounded — when exhausted, deeper subagents get ``{"budget_capped": True}``
#: (never a silent drop). Cycles are blocked by a seen-set of agentIds.
MAX_SUBAGENT_DEPTH = 3
TOTAL_STEP_BUDGET = 4000

#: Tool names whose RESULT carries a spawned subagent's agentId.
_SUBAGENT_TOOL_NAMES = frozenset({"Task", "Agent"})

#: Matches a Claude Code agentId (16-17 lowercase hex chars) inside a Task/Agent
#: tool_result. The result may contain several; the first that resolves to an
#: ``agent-<id>.jsonl`` file in the root subagents dir is the child.
_AGENT_ID_RE = re.compile(r"[0-9a-f]{16,17}")

#: Default Claude Code projects root.
DEFAULT_PROJECTS_ROOT = Path.home() / ".claude" / "projects"

#: Claude Code injects harness context ahead of the user's first words: one or
#: more ``<system-reminder>`` blocks (CLAUDE.md, environment, date) plus, for
#: slash commands, ``<command-*>`` / ``<local-command-*>`` tag wrappers. These
#: are stripped from the first-prompt extraction (``_strip_harness_wrapper``) so
#: a session's surfaced "task" is the human's actual ask, not the boilerplate.
_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE
)
_COMMAND_BLOCK_RE = re.compile(
    r"<((?:local-)?command-[a-z]+)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_COMMAND_TAG_RE = re.compile(r"</?(?:local-)?command-[a-z]+>", re.IGNORECASE)
_COMMAND_NAME_RE = re.compile(
    r"<command-name>(.*?)</command-name>", re.DOTALL | re.IGNORECASE
)
_COMMAND_ARGS_RE = re.compile(
    r"<command-args>(.*?)</command-args>", re.DOTALL | re.IGNORECASE
)

#: Per-tool preference for which single ``input`` arg makes the most useful
#: label. The first key present wins; falls back to the generic order below.
_TOOL_LABEL_ARGS: dict[str, tuple[str, ...]] = {
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "NotebookEdit": ("notebook_path", "file_path"),
    "Bash": ("command", "description"),
    "Glob": ("pattern", "path"),
    "Grep": ("pattern", "path"),
    "Task": ("description", "subagent_type", "prompt"),
    "WebFetch": ("url", "prompt"),
    "WebSearch": ("query",),
    "Skill": ("skill", "args"),
    "TodoWrite": ("description",),
}
#: Generic fallback order (most-useful-first) for unknown tools.
_GENERIC_LABEL_ARGS: tuple[str, ...] = (
    "file_path",
    "path",
    "command",
    "pattern",
    "query",
    "url",
    "description",
    "prompt",
    "name",
)


# --- Small helpers -----------------------------------------------------------

def _trim(text: str, limit: int) -> tuple[str, bool]:
    """Collapse whitespace-trim and cap ``text`` to ``limit`` chars.

    Returns ``(trimmed_text, was_truncated)``.
    """
    text = text.strip()
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "…", True


def _locate_transcript(session_id: str, projects_root: Path) -> Path | None:
    """Glob ``<projects_root>/*/<session_id>.jsonl``; return the file or None.

    ``session_id`` is treated as a literal filename component — any glob
    metacharacters in it are escaped so a crafted id can't widen the search.
    """
    if not session_id or not projects_root.exists():
        return None
    safe = glob.escape(session_id)
    pattern = str(projects_root / "*" / f"{safe}.jsonl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return Path(matches[0])


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dict records, tolerating bad lines."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _block_text(content: Any) -> str:
    """Extract a plain-text string from a CC ``message.content`` value.

    ``content`` is either a string or a list of blocks; we concatenate the
    ``text`` of any ``type == "text"`` blocks (the agent's narration). Other
    block types (``thinking``, ``tool_use``, ``tool_result``) are ignored here.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            piece = block.get("text")
            if isinstance(piece, str) and piece.strip():
                parts.append(piece)
    return "\n".join(parts)


def _tool_label(name: str, tool_input: Any) -> str:
    """Pick the single most-useful arg from ``tool_input`` as a short label.

    Returns "" when no recognized arg is present. NEVER returns the full input.
    """
    if not isinstance(tool_input, dict):
        return ""
    keys = _TOOL_LABEL_ARGS.get(name, ()) + _GENERIC_LABEL_ARGS
    for key in keys:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            label, _ = _trim(value, MAX_TOOL_LABEL_CHARS)
            return label
    return ""


def _build_tool_status(records: list[dict[str, Any]]) -> dict[str, bool]:
    """Map ``tool_use_id -> is_error`` from the ``user`` records' tool_results.

    A tool is marked errored iff its ``tool_result`` block carries
    ``is_error == true``. Missing entries are treated as "ok" by the caller.
    """
    status: dict[str, bool] = {}
    for record in records:
        if record.get("type") != "user":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                continue
            status[tool_use_id] = bool(block.get("is_error"))
    return status


def _tool_result_text(block: dict[str, Any]) -> str:
    """Flatten a ``tool_result`` block's ``content`` to plain text.

    Content is either a string or a list of ``{type:"text", text:...}`` blocks
    (Claude Code emits both shapes). Used only to scan for an agentId — the
    text itself is NEVER surfaced in the story payload.
    """
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for piece in content:
            if isinstance(piece, dict):
                text = piece.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(piece, str):
                parts.append(piece)
        return " ".join(parts)
    return ""


def _build_subagent_ids(
    records: list[dict[str, Any]],
    agent_tool_use_ids: set[str],
    subagents_dir: Path | None,
) -> dict[str, str]:
    """Map each Task/Agent ``tool_use_id`` -> the spawned child's ``agentId``.

    The child's agentId lives in that tool's RESULT content (a Task/Agent
    result text carries several 16-17 hex strings; the FIRST one that resolves
    to an existing ``agent-<id>.jsonl`` in ``subagents_dir`` is the child). When
    no subagents dir exists yet (depth probing) the first matched id is used.
    Returns only entries we could resolve.
    """
    if not agent_tool_use_ids:
        return {}
    resolved: dict[str, str] = {}
    for record in records:
        if record.get("type") != "user":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id")
            if tool_use_id not in agent_tool_use_ids or tool_use_id in resolved:
                continue
            candidate_ids = _AGENT_ID_RE.findall(_tool_result_text(block))
            if not candidate_ids:
                continue
            child_id: str | None = None
            if subagents_dir is not None:
                for cid in candidate_ids:
                    if (subagents_dir / f"agent-{cid}.jsonl").exists():
                        child_id = cid
                        break
            if child_id is None:
                child_id = candidate_ids[0]
            if isinstance(tool_use_id, str):
                resolved[tool_use_id] = child_id
    return resolved


def _subagent_display_name(
    subagents_dir: Path | None,
    agent_id: str,
    fallback: str,
) -> str:
    """Display name for a subagent: ``meta.json`` ``name`` -> ``description``
    -> the parent Task input fallback -> a short agentId.

    Claude Code writes a sidecar ``agent-<id>.meta.json`` carrying ``name`` (a
    human label like ``impl-session-view``), ``description`` and ``agentType``.
    Reads only that small sidecar; tolerant of a missing/garbled file.
    """
    if subagents_dir is not None:
        meta_path = subagents_dir / f"agent-{agent_id}.meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError):
                meta = None
            if isinstance(meta, dict):
                for key in ("name", "description"):
                    value = meta.get(key)
                    if isinstance(value, str) and value.strip():
                        label, _ = _trim(value, MAX_TOOL_LABEL_CHARS)
                        return label
    if fallback.strip():
        return fallback.strip()
    return f"agent-{agent_id[:8]}"


def _slash_command_label(text: str) -> str:
    """Build a ``/cmd args`` label from Claude Code's ``<command-*>`` wrapper.

    Returns "" when the text carries no ``<command-name>``.
    """
    name_match = _COMMAND_NAME_RE.search(text)
    if not name_match:
        return ""
    name = name_match.group(1).strip()
    if not name:
        return ""
    args_match = _COMMAND_ARGS_RE.search(text)
    args = args_match.group(1).strip() if args_match else ""
    label = name if name.startswith("/") else "/" + name
    return f"{label} {args}".strip()


def _strip_harness_wrapper(text: str) -> str:
    """Strip Claude Code's injected wrappers from a first-user message.

    Claude Code prepends the human's first words with one or more
    ``<system-reminder>`` blocks (CLAUDE.md, environment, date) and, for slash
    commands, ``<command-*>`` / ``<local-command-*>`` tag wrappers. Removing them
    leaves the actual ask. If only a slash command remains, return a ``/cmd
    args`` label so a command-only message still yields a meaningful task.
    Returns "" when nothing meaningful is left.
    """
    command = _slash_command_label(text)
    cleaned = _SYSTEM_REMINDER_RE.sub("", text)
    cleaned = _COMMAND_BLOCK_RE.sub("", cleaned)
    cleaned = _COMMAND_TAG_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned or command


def _first_user_prompt(records: list[dict[str, Any]]) -> str:
    """The human's first real ``user`` message (the initial prompt / ticket).

    Skips ``isMeta`` records and tool-result-only user records, and strips the
    Claude Code harness wrapper (``<system-reminder>`` / ``<command-*>`` tags)
    so the result is the actual ask, not the injected context. Falls through to
    the next user message when one is pure wrapper.
    """
    for record in records:
        if record.get("type") != "user" or record.get("isMeta"):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        text = _strip_harness_wrapper(_block_text(message.get("content")))
        if text.strip():
            return text
    return ""


def _build_steps(
    records: list[dict[str, Any]],
    tool_status: dict[str, bool],
) -> list[dict[str, Any]]:
    """One step per ``assistant`` turn, in record order.

    Each step carries the agent's narration text, the literal tool calls (name +
    short label + ok/error status), and small flags (is_error, is_retry, model).
    Assistant turns that have neither narration nor a tool call are skipped
    (e.g. thinking-only or empty turns) so the story stays meaningful.
    """
    steps: list[dict[str, Any]] = []
    prev_signature: tuple[tuple[str, str], ...] | None = None
    n = 0

    for record in records:
        if record.get("type") != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue

        text_raw = _block_text(message.get("content"))
        tools: list[dict[str, Any]] = []
        # Task/Agent spawns in this turn: (tool_use_id, fallback display name).
        spawns: list[tuple[str, str]] = []
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name") or "unknown"
                if not isinstance(name, str):
                    name = "unknown"
                tool_input = block.get("input")
                label = _tool_label(name, tool_input)
                tool_use_id = block.get("id")
                is_error = (
                    bool(tool_status.get(tool_use_id))
                    if isinstance(tool_use_id, str)
                    else False
                )
                tools.append(
                    {
                        "name": name,
                        "label": label,
                        "status": "error" if is_error else "ok",
                    }
                )
                if name in _SUBAGENT_TOOL_NAMES and isinstance(tool_use_id, str):
                    spawns.append((tool_use_id, _spawn_fallback_name(tool_input)))

        # Skip turns with no narration and no tool call (thinking-only / empty).
        if not text_raw.strip() and not tools:
            continue

        n += 1
        text, text_truncated = _trim(text_raw, MAX_STEP_TEXT_CHARS)
        step_is_error = any(t["status"] == "error" for t in tools)

        signature = tuple((t["name"], t["label"]) for t in tools)
        is_retry = bool(signature) and signature == prev_signature
        prev_signature = signature if signature else prev_signature

        model = message.get("model")
        step: dict[str, Any] = {
            "n": n,
            "ts": record.get("timestamp"),
            "text": text,
            "text_truncated": text_truncated,
            "tools": tools,
            "is_error": step_is_error,
            "is_retry": is_retry,
            "model": model if isinstance(model, str) else None,
        }
        if spawns:
            # Internal-only: consumed by the subagent attach pass, removed before
            # the step is returned. Carries the Task/Agent tool_use_ids + a
            # display-name fallback drawn from the Task input.
            step["_spawns"] = spawns
        steps.append(step)

    return steps


def _spawn_fallback_name(tool_input: Any) -> str:
    """Display-name fallback for a spawned subagent from the Task/Agent input.

    Prefers ``subagent_type`` (e.g. ``general-purpose``) then ``description``.
    Used only when the subagent's ``meta.json`` has no usable ``name`` — never
    surfaces full prompt content.
    """
    if not isinstance(tool_input, dict):
        return ""
    for key in ("subagent_type", "description"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            label, _ = _trim(value, MAX_TOOL_LABEL_CHARS)
            return label
    return ""


def _cap_steps(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Apply ``MAX_STORY_STEPS``: keep head + tail with an explicit marker.

    Returns ``(steps_with_marker, truncated)``. When the count exceeds the cap,
    inserts a single ``{"omitted": N}`` marker dict between the kept head and
    tail — never a silent drop.
    """
    if len(steps) <= MAX_STORY_STEPS:
        return steps, False
    head = steps[:HEAD_STEPS]
    tail = steps[-TAIL_STEPS:]
    omitted = len(steps) - len(head) - len(tail)
    capped: list[dict[str, Any]] = [*head, {"omitted": omitted}, *tail]
    return capped, True


# --- Recursion state ---------------------------------------------------------

class _Budget:
    """Mutable shared step budget threaded across the whole subagent tree.

    Each story consumes ``len(steps)`` from the budget. Once it hits zero,
    deeper subagents are not expanded — they get ``{"budget_capped": True}``
    instead, so nothing is silently dropped.
    """

    __slots__ = ("remaining",)

    def __init__(self, total: int) -> None:
        self.remaining = total

    def take(self, n: int) -> None:
        self.remaining = max(0, self.remaining - n)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0


# --- Public API --------------------------------------------------------------

def build_session_story(
    session_id: str,
    projects_root: Path | str | None = None,
    include_subagents: bool = True,
) -> dict[str, Any] | None:
    """Build a deterministic story for a Claude Code session, or None if absent.

    Locates ``<projects_root>/*/<session_id>.jsonl`` (default projects root is
    ``~/.claude/projects``), parses it, and returns:

        {
          "task": str,            # first user prompt, trimmed
          "outcome": str,         # last assistant narration, trimmed
          "step_count": int,      # number of real assistant turns
          "truncated": bool,      # True if steps were capped
          "steps": [ {n, ts, text, text_truncated,
                      tools: [{name, label, status}],
                      is_error, is_retry, model,
                      subagent?: {...}  # only on Task/Agent steps that spawned
                     }, ...,
                     {"omitted": N}  # marker, only when truncated ],
        }

    When ``include_subagents`` is True (default), every ``Task``/``Agent`` step
    whose child transcript exists under
    ``<dir(parent)>/<session_id>/subagents/agent-<agentId>.jsonl`` gets a
    recursive ``subagent`` object (same schema, plus ``agent_id`` / ``name``).
    Recursion is bounded by ``MAX_SUBAGENT_DEPTH``, a shared step budget
    (``TOTAL_STEP_BUDGET``), and a cycle-guard seen-set of agentIds.

    Returns None when no transcript file exists for the session id (SDK session,
    or transcript pruned) — callers translate that into ``available: false``.
    """
    if projects_root is None:
        root = DEFAULT_PROJECTS_ROOT
    else:
        root = Path(projects_root)

    path = _locate_transcript(session_id, root)
    if path is None:
        return None

    # All subagents for a root session live FLAT here (any depth) — see plan.
    subagents_dir = path.parent / session_id / "subagents"

    if not include_subagents:
        story = _build_story_from_path(path, None, None, _Budget(TOTAL_STEP_BUDGET), 0, set())
        return story

    budget = _Budget(TOTAL_STEP_BUDGET)
    seen: set[str] = {session_id}
    return _build_story_from_path(path, subagents_dir, None, budget, 0, seen)


def _build_story_from_path(
    path: Path,
    subagents_dir: Path | None,
    agent_id: str | None,
    budget: _Budget,
    depth: int,
    seen: set[str],
) -> dict[str, Any]:
    """Build one story from a located transcript ``path`` and (optionally) attach
    its subagents recursively, sharing ``budget`` / ``seen`` across the tree."""
    records = _read_records(path)

    tool_status = _build_tool_status(records)
    steps = _build_steps(records, tool_status)
    step_count = len(steps)
    budget.take(step_count)

    task, _ = _trim(_first_user_prompt(records), MAX_TASK_OUTCOME_CHARS)

    # Outcome = last assistant turn that actually narrated something.
    outcome_raw = ""
    for step in reversed(steps):
        if "omitted" in step:
            continue
        if step.get("text"):
            outcome_raw = step["text"]
            break
    outcome, _ = _trim(outcome_raw, MAX_TASK_OUTCOME_CHARS)

    if subagents_dir is not None:
        _attach_subagents(steps, records, subagents_dir, budget, depth, seen)
    else:
        _strip_spawn_markers(steps)

    capped_steps, truncated = _cap_steps(steps)

    story: dict[str, Any] = {
        "task": task,
        "outcome": outcome,
        "step_count": step_count,
        "truncated": truncated,
        "steps": capped_steps,
    }
    if agent_id is not None:
        story["agent_id"] = agent_id
    return story


def _strip_spawn_markers(steps: list[dict[str, Any]]) -> None:
    """Remove the internal ``_spawns`` key when not attaching subagents."""
    for step in steps:
        step.pop("_spawns", None)


def _attach_subagents(
    steps: list[dict[str, Any]],
    records: list[dict[str, Any]],
    subagents_dir: Path,
    budget: _Budget,
    depth: int,
    seen: set[str],
) -> None:
    """For each Task/Agent step that spawned a child, attach its ``subagent``.

    Resolves each spawn's child agentId, and if the child transcript exists in
    ``subagents_dir`` recursively builds the child story. Honors depth / budget
    caps and the cycle-guard seen-set. Mutates ``steps`` in place and removes
    the internal ``_spawns`` markers.
    """
    # Collect the tool_use_ids of all Task/Agent spawns across these steps.
    spawn_ids: set[str] = set()
    for step in steps:
        for tool_use_id, _ in step.get("_spawns", []):
            spawn_ids.add(tool_use_id)

    id_map: dict[str, str] = {}
    if spawn_ids:
        # Primary: regex over each Task/Agent tool_result content for the child
        # agentId (the documented linkage; picks the first id that resolves to
        # an agent-<id>.jsonl in subagents_dir).
        id_map = _build_subagent_ids(records, spawn_ids, subagents_dir)
        # Fallback: the meta.json sidecar's authoritative ``toolUseId`` link,
        # for any spawn the tool_result text didn't resolve.
        unresolved = spawn_ids - set(id_map)
        if unresolved:
            for tool_use_id, child_id in _resolve_spawns_from_dir(
                steps, subagents_dir
            ).items():
                id_map.setdefault(tool_use_id, child_id)

    for step in steps:
        spawns = step.pop("_spawns", None)
        if not spawns:
            continue
        # A step can carry multiple Task spawns; attach the first that resolves
        # (the common case is one Task per step). Multiple parallel spawns in a
        # single turn are represented by attaching each onto a list.
        attached: list[dict[str, Any]] = []
        for tool_use_id, fallback in spawns:
            child_id = id_map.get(tool_use_id)
            if child_id is None:
                continue
            sub = _build_subagent(child_id, fallback, subagents_dir, budget, depth, seen)
            if sub is not None:
                attached.append(sub)
        if len(attached) == 1:
            step["subagent"] = attached[0]
        elif len(attached) > 1:
            step["subagents"] = attached


def _resolve_spawns_from_dir(
    steps: list[dict[str, Any]],
    subagents_dir: Path,
) -> dict[str, str]:
    """Map Task/Agent ``tool_use_id`` -> child ``agentId`` using ``meta.json``.

    Claude Code writes ``agent-<id>.meta.json`` carrying the parent ``toolUseId``
    — the authoritative link. We index the dir's metas by ``toolUseId`` so each
    spawn resolves to its exact child even when several ran in parallel.
    """
    if not subagents_dir.exists():
        return {}
    by_tool_use: dict[str, str] = {}
    for meta_path in subagents_dir.glob("agent-*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        tool_use_id = meta.get("toolUseId")
        # filename: agent-<id>.meta.json
        agent_id = meta_path.name[len("agent-"):-len(".meta.json")]
        if isinstance(tool_use_id, str) and agent_id:
            by_tool_use[tool_use_id] = agent_id

    result: dict[str, str] = {}
    for step in steps:
        for tool_use_id, _ in step.get("_spawns", []):
            if tool_use_id in by_tool_use:
                result[tool_use_id] = by_tool_use[tool_use_id]
    return result


def _build_subagent(
    agent_id: str,
    fallback_name: str,
    subagents_dir: Path,
    budget: _Budget,
    depth: int,
    seen: set[str],
) -> dict[str, Any] | None:
    """Build one subagent reference, recursing into its own story.

    Returns a small dict with ``agent_id``/``name`` plus, when the child can be
    expanded, the recursive story fields (``task``/``outcome``/``steps`` …).
    Marks ``depth_capped`` / ``budget_capped`` / ``cycle`` instead of expanding
    when a guard trips — never a silent drop. Returns None only when the child
    transcript file genuinely doesn't exist.
    """
    child_path = subagents_dir / f"agent-{agent_id}.jsonl"
    if not child_path.exists():
        return None

    name = _subagent_display_name(subagents_dir, agent_id, fallback_name)
    ref: dict[str, Any] = {"agent_id": agent_id, "name": name}

    if agent_id in seen:
        ref["cycle"] = True
        return ref
    if depth + 1 > MAX_SUBAGENT_DEPTH:
        ref["depth_capped"] = True
        return ref
    if budget.exhausted:
        ref["budget_capped"] = True
        return ref

    seen.add(agent_id)
    child_story = _build_story_from_path(
        child_path, subagents_dir, agent_id, budget, depth + 1, seen
    )
    # Merge the recursive story onto the ref (keeps agent_id/name on top).
    child_story.pop("agent_id", None)
    ref.update(child_story)
    ref["name"] = name
    ref["agent_id"] = agent_id
    return ref


def resolve_projects_root(override: Path | str | None = None) -> Path:
    """Resolve the projects root: explicit override -> env -> default.

    Honors the ``TJ_CLAUDE_PROJECTS_ROOT`` env var so the API and tests can
    point the Story at a temp projects directory without touching real files.
    """
    if override is not None:
        return Path(override)
    env = os.environ.get("TJ_CLAUDE_PROJECTS_ROOT")
    if env:
        return Path(env)
    return DEFAULT_PROJECTS_ROOT


__all__ = [
    "build_session_story",
    "resolve_projects_root",
    "DEFAULT_PROJECTS_ROOT",
    "MAX_STORY_STEPS",
    "MAX_STEP_TEXT_CHARS",
    "MAX_TASK_OUTCOME_CHARS",
    "MAX_TOOL_LABEL_CHARS",
    "MAX_SUBAGENT_DEPTH",
    "TOTAL_STEP_BUDGET",
]
