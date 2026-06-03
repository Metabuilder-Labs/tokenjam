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
MAX_STEP_TEXT_CHARS = 400
MAX_TASK_OUTCOME_CHARS = 600
MAX_TOOL_LABEL_CHARS = 120

#: Default Claude Code projects root.
DEFAULT_PROJECTS_ROOT = Path.home() / ".claude" / "projects"

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


def _first_user_prompt(records: list[dict[str, Any]]) -> str:
    """Text of the first real ``user`` message (the initial prompt / ticket).

    Skips ``isMeta`` records and tool-result-only user records — those are not
    the human's prompt.
    """
    for record in records:
        if record.get("type") != "user" or record.get("isMeta"):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        text = _block_text(message.get("content"))
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
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name") or "unknown"
                if not isinstance(name, str):
                    name = "unknown"
                label = _tool_label(name, block.get("input"))
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
        steps.append(
            {
                "n": n,
                "ts": record.get("timestamp"),
                "text": text,
                "text_truncated": text_truncated,
                "tools": tools,
                "is_error": step_is_error,
                "is_retry": is_retry,
                "model": model if isinstance(model, str) else None,
            }
        )

    return steps


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


# --- Public API --------------------------------------------------------------

def build_session_story(
    session_id: str,
    projects_root: Path | str | None = None,
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
                      is_error, is_retry, model}, ...,
                     {"omitted": N}  # marker, only when truncated ],
        }

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

    records = _read_records(path)

    tool_status = _build_tool_status(records)
    steps = _build_steps(records, tool_status)
    step_count = len(steps)

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

    capped_steps, truncated = _cap_steps(steps)

    return {
        "task": task,
        "outcome": outcome,
        "step_count": step_count,
        "truncated": truncated,
        "steps": capped_steps,
    }


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
]
