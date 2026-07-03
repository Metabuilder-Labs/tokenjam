"""Synthesize a compact **resume brief** from a session tj already persists.

When a Claude Code session is compacted, resumed, or continued after an
ephemeral subagent died, the *method* — what the task was, what was already
done, what was tried and abandoned, where it left off, which files are dirty —
is lost, and the next turn re-investigates from scratch. tj already keeps the
raw material to avoid that: the reconstructed Story (``core/transcript`` /
``core/method_spine``) and its durable snapshot (``core/method_capture``, which
survives Claude Code's transcript prune).

This module folds that material into a short, plain-text brief (target
~400 tokens) with five sections — TASK / DONE-PROGRESS / TRIED-DEAD-ENDS /
OPEN-WHERE-IT-LEFT-OFF / WORKING-FILES — so a resuming session can be *handed*
its prior method instead of rediscovering it.

Deterministic, no LLM, no interpretation (repo Critical Rule 14: every line is
the agent's own words or an exact structural fact). **Pure + fail-soft**: every
public function takes plain dicts (the Story / asks shapes produced by
``core/transcript``) and NEVER raises — any sub-extraction that errors degrades
to empty. Imports only ``core`` — never ``tokenjam.api`` / ``tokenjam.cli``.

Two reliability rules learned from the prototype's evaluation:

  * **Skip slash-command-only openers** (``/clear``, ``/model``) when picking the
    TASK — take the first *substantive prose* ask, not the literal first message.
  * **Scope the brief to the last ask/phase** for long multi-task sessions, so
    progress / files don't sprawl across unrelated earlier work.
"""
from __future__ import annotations

import re
from typing import Any

from tokenjam.core.method_spine import build_method_spine
from tokenjam.core.workmap import _FILE_TOOLS

#: Edit-shaped tools whose ``label`` is the touched file path (the likely-dirty
#: set). ``Read`` is in ``_FILE_TOOLS`` but is not a write, so exclude it.
_EDIT_TOOLS = frozenset(_FILE_TOOLS - {"Read"})

#: A user prompt that is ONLY a slash command (``/clear``, ``/model opus``) —
#: harness control, not a task. ``core/transcript`` renders such an opener as a
#: ``/cmd args`` label; a real ask that merely starts with a slash command but
#: carries prose keeps the prose, so this only matches the bare command form.
_SLASH_ONLY_RE = re.compile(r"^/[\w:-]+(?:\s+\S+){0,4}$")

#: Interruption / connection-closed markers — the strongest "it stopped here"
#: signal, but they live only in the raw transcript (the Story parser drops
#: them), so they are surfaced best-effort from ``records`` when available.
_INTERRUPT_RE = re.compile(
    r"API Error: Connection closed|Connection closed mid-response"
    r"|\[Request interrupted|Request was aborted|operation was aborted",
    re.I,
)
_NEXT_RE = re.compile(
    r"\b(next(?:,| step| I| we)|now I['’]?ll|let me|I['’]?ll now|I will now"
    r"|remaining|still need|todo|to do)\b",
    re.I,
)
_TRIED_RE = re.compile(
    r"didn['’]?t work|doesn['’]?t work|that failed|let me try|instead"
    r"|revert|roll ?back|undo|wrong|not working|broke|broken|reverting",
    re.I,
)

# Output caps (a brief is a glance, not the transcript).
_MAX_TASK_CHARS = 280
_MAX_PROGRESS_LINES = 6
_MAX_TRIED_LINES = 6
_MAX_PENDING_TODOS = 5
_MAX_FILES = 12


# --------------------------------------------------------------------------- #
# Ask selection (the two reliability rules)
# --------------------------------------------------------------------------- #
def _is_substantive_prompt(text: str | None) -> bool:
    """True when ``text`` is a real prose ask, not a bare slash command."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if "\n" not in stripped and _SLASH_ONLY_RE.match(stripped):
        return False
    return True


def _asks_list(asks: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The ``asks`` list out of a ``build_session_asks`` dict, or ``[]``."""
    if not isinstance(asks, dict):
        return []
    raw = asks.get("asks")
    return [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []


def select_task_prompt(
    asks: dict[str, Any] | None, story: dict[str, Any] | None
) -> str:
    """The TASK line: the first substantive prose ask (slash openers skipped).

    Falls back to the Story's own ``task`` (itself substantive-checked), then "".
    """
    for ask in _asks_list(asks):
        if _is_substantive_prompt(ask.get("prompt")):
            return (ask.get("prompt") or "").strip()
    task = (story or {}).get("task") if isinstance(story, dict) else ""
    return (task or "").strip() if _is_substantive_prompt(task) else (task or "").strip()


def select_scope_ask(asks: dict[str, Any] | None) -> dict[str, Any] | None:
    """The ask the brief scopes to: the LAST substantive ask (current phase).

    Returns ``None`` when there are no asks (caller falls back to the whole
    Story). Prefers the last substantive ask; if every ask is a slash command,
    falls back to the last ask so a short session still yields something.
    """
    asks_list = _asks_list(asks)
    if not asks_list:
        return None
    for ask in reversed(asks_list):
        if _is_substantive_prompt(ask.get("prompt")):
            return ask
    return asks_list[-1]


def _scoped_story(
    story: dict[str, Any] | None,
    scope_ask: dict[str, Any] | None,
    task_prompt: str,
) -> dict[str, Any]:
    """Build the (possibly ask-scoped) Story the brief summarizes.

    With a ``scope_ask`` (long multi-task session) the summary is confined to
    that ask's steps + outcome; otherwise it's the whole Story. Always a plain
    dict of the ``build_session_story`` shape so ``build_method_spine`` accepts it.
    """
    if scope_ask is not None:
        steps = scope_ask.get("steps") or []
        return {
            "task": task_prompt,
            "outcome": scope_ask.get("outcome") or "",
            "step_count": scope_ask.get("step_count") or len(steps),
            "steps": steps,
        }
    if isinstance(story, dict):
        return {
            "task": task_prompt,
            "outcome": story.get("outcome") or "",
            "step_count": story.get("step_count") or len(story.get("steps") or []),
            "steps": story.get("steps") or [],
        }
    return {"task": task_prompt, "outcome": "", "step_count": 0, "steps": []}


# --------------------------------------------------------------------------- #
# Extraction from Story steps (works from the persisted snapshot, post-prune)
# --------------------------------------------------------------------------- #
def _walk_tools(steps: Any):
    """Yield every tool dict across ``steps`` and their nested subagent steps.

    The persisted Story carries subagent subtrees recursively; a file a
    subagent edited is just as dirty, so descend into them. Cap markers
    (``{"omitted": N}``) and malformed entries are skipped. Never raises.
    """
    if not isinstance(steps, list):
        return
    for step in steps:
        if not isinstance(step, dict) or "omitted" in step:
            continue
        for tool in step.get("tools") or []:
            if isinstance(tool, dict):
                yield tool
        sub = step.get("subagent")
        if isinstance(sub, dict):
            yield from _walk_tools(sub.get("steps") or [])
        for s in step.get("subagents") or []:
            if isinstance(s, dict):
                yield from _walk_tools(s.get("steps") or [])


def extract_todos(steps: Any) -> tuple[list[str], list[str]]:
    """The last TodoWrite payload's ``(in_progress, pending)`` item contents.

    ``core/transcript`` preserves a TodoWrite call's ``todos`` list
    (``[{content, status}]``) on the step's tool — the single best "where it
    left off / what's incomplete" signal — so the brief reads it from the Story
    rather than re-parsing raw JSONL (works after the transcript is pruned).
    """
    last: list[dict[str, Any]] | None = None
    for tool in _walk_tools(steps):
        if tool.get("name") == "TodoWrite":
            todos = tool.get("todos")
            if isinstance(todos, list) and todos:
                last = todos
    if not last:
        return [], []
    in_prog = [
        (t.get("content") or "").strip()
        for t in last
        if isinstance(t, dict) and t.get("status") == "in_progress" and t.get("content")
    ]
    pending = [
        (t.get("content") or "").strip()
        for t in last
        if isinstance(t, dict) and t.get("status") == "pending" and t.get("content")
    ]
    return in_prog, pending


def extract_working_files(steps: Any) -> list[str]:
    """Distinct file paths of every Edit/Write/MultiEdit/NotebookEdit step.

    The touched path is the edit tool's ``label`` (``core/transcript`` reduces
    those tools to their file path). First-seen order; deduped. NEVER raises.
    """
    seen: list[str] = []
    for tool in _walk_tools(steps):
        if tool.get("name") in _EDIT_TOOLS:
            label = (tool.get("label") or "").strip()
            if label and label not in seen:
                seen.append(label)
    return seen


def extract_interruptions(records: Any) -> list[str]:
    """Interruption / connection-closed snippets from raw transcript records.

    Best-effort and live-transcript-only (the markers are dropped by the Story
    parser). Returns short context snippets around each hit. NEVER raises.
    """
    if not isinstance(records, list):
        return []
    hits: list[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for blob in _record_text_blobs(rec):
            match = _INTERRUPT_RE.search(blob)
            if match:
                start = max(0, match.start() - 10)
                hits.append(blob[start:match.start() + 60].strip())
    return hits


def _record_text_blobs(rec: dict[str, Any]) -> list[str]:
    """Every plain-text blob in a raw record (message content or top-level)."""
    blobs: list[str] = []
    msg = rec.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str):
            blobs.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                for key in ("text", "content"):
                    val = block.get(key)
                    if isinstance(val, str):
                        blobs.append(val)
    if isinstance(rec.get("content"), str):
        blobs.append(rec["content"])
    return blobs


# --------------------------------------------------------------------------- #
# Method-trail summary (reuse tj's method spine)
# --------------------------------------------------------------------------- #
def _flatten_spine(spine: Any, out: list | None = None) -> list[dict[str, Any]]:
    """Depth-first flatten of the recursive spine (drops nesting depth)."""
    if out is None:
        out = []
    if not isinstance(spine, list):
        return out
    for move in spine:
        if not isinstance(move, dict):
            continue
        out.append(move)
        for deleg in move.get("delegations") or []:
            if isinstance(deleg, dict):
                _flatten_spine(deleg.get("spine") or [], out)
    return out


def summarize_progress(spine: list[dict[str, Any]]) -> list[str]:
    """DONE lines: passed checks, delegations, and the last narrated moves."""
    flat = _flatten_spine(spine)
    lines: list[str] = []

    verifies = [m for m in flat if m.get("kind") == "verify" and not m.get("failed")]
    if verifies:
        lines.append(
            f"passed checks: {len(verifies)} (e.g. {verifies[-1]['label'][:60]})"
        )

    delegs: list[str] = []
    for move in spine if isinstance(spine, list) else []:
        if not isinstance(move, dict):
            continue
        for deleg in move.get("delegations") or []:
            if isinstance(deleg, dict) and deleg.get("task"):
                delegs.append(deleg["task"])
    for task in delegs[:3]:
        lines.append(f"delegated: {task[:70]}")

    narrated = [
        m for m in flat
        if m.get("source") == "agent_words"
        and m.get("kind") in ("act", "delegate")
        and not m.get("is_retry")
    ]
    for move in narrated[-4:]:
        lines.append(f"- {move.get('label', '')[:80]}")
    return lines[:_MAX_PROGRESS_LINES]


def summarize_tried(spine: list[dict[str, Any]]) -> list[str]:
    """TRIED / DEAD-END lines: retries, reverts, tool errors, failed checks."""
    flat = _flatten_spine(spine)
    lines: list[str] = []
    for move in flat:
        why = None
        if move.get("kind") == "dead_end":
            why = "retry/revert"
        elif move.get("failed"):
            why = "check failed"
        elif any(
            isinstance(e, dict) and e.get("status") == "error"
            for e in move.get("evidence") or []
        ):
            why = "tool error"
        elif move.get("source") == "agent_words" and _TRIED_RE.search(
            move.get("quote") or move.get("label") or ""
        ):
            why = "narrated dead-end"
        if why:
            lines.append(f"- [{why}] {move.get('label', '')[:75]}")
    dedup: list[str] = []
    for line in lines:
        if not dedup or dedup[-1] != line:
            dedup.append(line)
    return dedup[:_MAX_TRIED_LINES]


def summarize_left_off(
    spine: list[dict[str, Any]],
    scoped_story: dict[str, Any],
    in_prog: list[str],
    pending: list[str],
    interrupts: list[str],
) -> list[str]:
    """OPEN lines: interruption, in-progress/pending todos, last words, next-hint."""
    lines: list[str] = []
    if interrupts:
        lines.append(f'INTERRUPTED: "{interrupts[-1][:70]}"')
    for todo in in_prog:
        lines.append(f"in-progress: {todo[:80]}")
    for todo in pending[:_MAX_PENDING_TODOS]:
        lines.append(f"pending: {todo[:80]}")
    outcome = (scoped_story.get("outcome") or "").strip()
    if outcome:
        lines.append(f"last words: {outcome[:160]}")
    for move in _flatten_spine(spine)[-6:]:
        quote = move.get("quote") or ""
        match = _NEXT_RE.search(quote)
        if match:
            frag = quote[match.start():match.start() + 90].strip()
            lines.append(f"next-hint: …{frag}…")
            break
    return lines


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_resume_brief(
    story: dict[str, Any] | None,
    asks: dict[str, Any] | None = None,
    *,
    session_id: str = "",
    records: Any = None,
) -> str:
    """Synthesize a compact resume brief, or "" when there is nothing to brief.

    ``story`` / ``asks`` are the ``build_session_story`` / ``build_session_asks``
    shapes (either freshly built OR pulled from a persisted ``session_story``
    snapshot). ``records`` is the raw JSONL record list (optional; only used for
    interruption markers on the live-transcript path). NEVER raises — any
    failure degrades to empty sections or an empty brief.
    """
    steps_present = isinstance(story, dict) and (story.get("steps") or [])
    asks_present = bool(_asks_list(asks))
    if not steps_present and not asks_present:
        return ""

    task_prompt = select_task_prompt(asks, story)
    scope_ask = select_scope_ask(asks)
    scoped = _scoped_story(story, scope_ask, task_prompt)

    try:
        spine = build_method_spine(scoped)
    except Exception:  # noqa: BLE001 - fail-soft: a brief must never break a session
        spine = []

    steps = scoped.get("steps") or []
    in_prog, pending = extract_todos(steps)
    files = extract_working_files(steps)
    interrupts = extract_interruptions(records)

    # A "current phase" note only when the scoped ask differs from the task line
    # (a genuinely multi-task session), so a single-task brief stays clean.
    current_phase = ""
    if scope_ask is not None:
        scope_prompt = (scope_ask.get("prompt") or "").strip()
        if scope_prompt and scope_prompt != task_prompt.strip():
            current_phase = scope_prompt

    return _render(
        session_id=session_id,
        step_count=scoped.get("step_count") or len(steps),
        task=task_prompt,
        current_phase=current_phase,
        spine=spine,
        scoped=scoped,
        in_prog=in_prog,
        pending=pending,
        files=files,
        interrupts=interrupts,
    )


def _render(
    *,
    session_id: str,
    step_count: int,
    task: str,
    current_phase: str,
    spine: list[dict[str, Any]],
    scoped: dict[str, Any],
    in_prog: list[str],
    pending: list[str],
    files: list[str],
    interrupts: list[str],
) -> str:
    """Assemble the five-section plain-text brief."""
    sid = (session_id or "")[:8] or "session"
    out: list[str] = [
        f"=== RESUME BRIEF  [{sid}]  ({step_count} steps) ===",
        "",
        "TASK",
        f"  {task[:_MAX_TASK_CHARS] or '(no task captured)'}",
    ]
    if current_phase:
        out.append(f"  current phase: {current_phase[:_MAX_TASK_CHARS]}")
    out += ["", "DONE / PROGRESS"]
    progress = summarize_progress(spine)
    out += [f"  {ln}" for ln in (progress or ["(none detected)"])]

    out += ["", "TRIED / DEAD-ENDS"]
    tried = summarize_tried(spine)
    out += [f"  {ln}" for ln in (tried or ["(none detected)"])]

    out += ["", "OPEN / WHERE IT LEFT OFF"]
    left = summarize_left_off(spine, scoped, in_prog, pending, interrupts)
    out += [f"  {ln}" for ln in (left or ["(nothing pending detected)"])]

    out += ["", "WORKING FILES (likely dirty)"]
    if files:
        out += [f"  {f}" for f in files[:_MAX_FILES]]
        if len(files) > _MAX_FILES:
            out.append(f"  … +{len(files) - _MAX_FILES} more")
    else:
        out.append("  (none)")
    return "\n".join(out)


__all__ = [
    "build_resume_brief",
    "select_task_prompt",
    "select_scope_ask",
    "extract_todos",
    "extract_working_files",
    "extract_interruptions",
]
