"""How often each on-demand agent file was ACTUALLY invoked in a window.

``core/summarize/load_semantics`` says a skill/command/agent body reaches the
model only when it is invoked. That makes the invocation COUNT the missing
multiplier: without it the only two options are pricing an on-demand body as if
it were always resident (wildly overstated — a skill nobody ran becomes the
most expensive file a user owns) or pricing it at zero (understated — a 160 KB
skill invoked daily genuinely costs real tokens, and shortening it genuinely
pays).

The count is OBSERVED, never assumed. Claude Code's own transcripts record
every one of the three invocation shapes, by name:

* ``Skill`` tool calls carry ``input.skill`` — the skill (or command) slug.
* A typed slash command arrives as a user-role ``<command-name>/x</command-name>``
  block.
* ``Task`` / ``Agent`` tool calls carry ``input.subagent_type``.

Scans the same corpus, over the same mtime window, with the same persistent
parse cache (``core/transcript_cache``) the ``deadweight`` and ``relearn``
analyzers already use, so a warm re-run costs a few seconds over thousands of
sessions.

Honesty discipline (Critical Rule 28 corollary a): ``observed`` distinguishes
"this file was invoked zero times" — a real measurement, which prices the
on-demand portion at exactly zero — from "there is no transcript corpus here",
which is not a measurement at all and must degrade both the token and the
dollar field together rather than silently reading as zero.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Tool names whose ``input.skill`` names an invoked skill or command.
_SKILL_TOOLS = frozenset({"Skill"})
#: Tool names whose ``input.subagent_type`` names a spawned agent definition.
_AGENT_TOOLS = frozenset({"Task", "Agent"})

#: A typed slash command as Claude Code records it in the user turn.
_COMMAND_NAME_RE = re.compile(r"<command-name>(.*?)</command-name>", re.DOTALL | re.IGNORECASE)

#: Source string for ``estimate_basis`` — names the signal, not just its shape.
INVOCATION_SOURCE = (
    "Claude Code transcripts: `Skill` tool calls (by skill slug), typed "
    "`<command-name>` slash commands, and `Task`/`Agent` spawns (by "
    "`subagent_type`), counted over the same window"
)


@dataclass(frozen=True)
class InvocationCounts:
    """Observed invocations per name, plus whether anything was observable.

    ``counts`` is keyed on the bare slug (``ship``, ``govern``) — the same key
    ``load_semantics.invocation_key`` derives from a file path. A
    plugin-namespaced invocation (``superpowers:brainstorming``) is indexed
    under BOTH the full name and its bare suffix, because a plugin's own
    ``SKILL.md`` on disk is named by the suffix alone.

    ``observed`` is False only when there was no corpus to read — never
    because a particular file happened to be invoked zero times. A zero in
    ``counts`` is a measurement; ``observed=False`` is the absence of one.
    """

    counts: dict[str, int] = field(default_factory=dict)
    sessions_scanned: int = 0
    observed: bool = False
    #: The distinct working directories the scanned sessions recorded. Collected
    #: on this pass rather than by a second walk purely because the corpus walk
    #: is the expensive part and this one is already reading every record;
    #: ``core/summarize/repo_roots`` turns them into scan roots. Empty means the
    #: transcripts carried no ``cwd``, not that the sessions ran nowhere.
    session_cwds: tuple[str, ...] = ()
    #: Distinct invocation EVENTS counted, before the bare-suffix aliasing
    #: below — summing ``counts.values()`` would double-count every
    #: plugin-namespaced name, which is only ever a display figure.
    total_invocations: int = 0

    def get(self, key: str) -> int:
        """Invocations recorded for ``key`` (0 when it was never invoked)."""
        return int(self.counts.get(key, 0)) if key else 0


def _bump(counts: dict[str, int], raw: str) -> bool:
    """Record one invocation of ``raw``; return whether anything was counted.

    A ``plugin:slug`` name is ALSO indexed under its bare slug, because a
    plugin's ``SKILL.md`` on disk is named by the slug alone and that is the
    key ``load_semantics.invocation_key`` derives from its path.
    """
    name = (raw or "").strip().lstrip("/").strip()
    if not name:
        return False
    counts[name] = counts.get(name, 0) + 1
    if ":" in name:
        suffix = name.rsplit(":", 1)[-1].strip()
        if suffix and suffix != name:
            counts[suffix] = counts.get(suffix, 0) + 1
    return True


def _scan_record(record: dict[str, Any], counts: dict[str, int]) -> int:
    """Count every invocation shape carried by one transcript record.

    Returns how many distinct invocation events it held.
    """
    message = record.get("message")
    if not isinstance(message, dict):
        return 0
    content = message.get("content")
    events = 0
    if isinstance(content, str):
        for hit in _COMMAND_NAME_RE.findall(content):
            events += _bump(counts, hit)
        return events
    if not isinstance(content, list):
        return 0
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str) and "<command-name>" in text:
                for hit in _COMMAND_NAME_RE.findall(text):
                    events += _bump(counts, hit)
            continue
        if kind != "tool_use":
            continue
        tool_input = block.get("input")
        if not isinstance(tool_input, dict):
            continue
        name = block.get("name")
        if name in _SKILL_TOOLS:
            events += _bump(counts, str(tool_input.get("skill") or ""))
        elif name in _AGENT_TOOLS:
            events += _bump(counts, str(tool_input.get("subagent_type") or ""))
    return events


def _record_cwd(record: dict[str, Any]) -> str:
    """The working directory one transcript record carries, or ``""``."""
    cwd = record.get("cwd")
    return cwd if isinstance(cwd, str) and cwd else ""


def count_invocations(
    since: datetime,
    until: datetime,
    *,
    projects_root: Path | str | None = None,
    cache_dir: Path | None = None,
) -> InvocationCounts:
    """Observed skill/command/agent invocations over ``since``..``until``.

    Session selection mirrors ``analyzers.deadweight.compute_deadweight_finding``
    exactly — transcripts under the resolved projects root, mtime-filtered to
    the window, sidechain (``subagents/``) transcripts skipped so a spawned
    child is not counted as its own session.

    Also returns each scanned session's recorded working directory
    (``session_cwds``). That is a different question from invocation counting,
    but it is answered from the same records on the same pass: the corpus walk
    is what costs, and asking a second walk for the cwds would double it.

    Never raises: a missing root, an unreadable transcript, or a malformed
    record degrades to fewer counts, and a missing root degrades to
    ``observed=False`` so the caller reports no figure rather than a zero.
    """
    from tokenjam.core.transcript import read_records, resolve_projects_root

    root = resolve_projects_root(projects_root)
    if not root.exists():
        return InvocationCounts()

    counts: dict[str, int] = {}
    cwds: set[str] = set()
    sessions = 0
    events = 0

    for path in sorted(root.rglob("*.jsonl")):
        if path.parent.name == "subagents":
            continue
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < since or mtime >= until:
            continue
        sessions += 1
        try:
            records = read_records(path, cache_dir=cache_dir)
        except Exception:
            logger.debug("invocation scan: unreadable transcript %s", path, exc_info=True)
            continue
        session_cwd = ""
        for record in records:
            if not isinstance(record, dict):
                continue
            events += _scan_record(record, counts)
            if not session_cwd:
                session_cwd = _record_cwd(record)
        if session_cwd:
            cwds.add(session_cwd)

    # A corpus that exists but held no session in this window is still an
    # observation: nothing was invoked because nothing ran. Only a missing
    # root (returned above) is "not measured".
    return InvocationCounts(
        counts=counts, sessions_scanned=sessions, observed=True,
        total_invocations=events, session_cwds=tuple(sorted(cwds)),
    )
