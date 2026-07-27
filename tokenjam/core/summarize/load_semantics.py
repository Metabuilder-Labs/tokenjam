"""How much of an agent-prompt file is resident in EVERY session, and how much
only arrives when the file is invoked.

The catalog (`agent_files.toml`) lumps five very different things together —
`CLAUDE.md`, `.claude/rules/*.md`, `.claude/skills/*/SKILL.md`,
`.claude/commands/*.md`, `.claude/agents/*.md` — and they are NOT loaded the
same way. Pricing every one of them as if its whole body sat in context on
every call of every session is what made a 160 KB skill library read as the
single most expensive prompt file a user owns, when in fact it had not been
invoked once in the window.

Two load classes, both real:

``ALWAYS``
    `CLAUDE.md` / `AGENTS.md` / `GEMINI.md` / `.claude/rules/*.md` and the
    `~/.claude` equivalents. The whole body is injected at the head of every
    session that loads it and re-read on that session's every later call.

``SKILL`` / ``COMMAND`` / ``AGENT``
    Only the YAML frontmatter is always resident: the harness lists these by
    ``name`` + ``description`` so the model knows they exist, and pulls the
    BODY in only when the skill is invoked, the command is typed, or the agent
    is spawned. Verified against a live Claude Code session's own context: the
    ``description`` from `.claude/commands/govern.md`'s frontmatter appears
    verbatim in the available-skills listing, and the body does not; the same
    holds for `.claude/skills/*/SKILL.md` and for the agent-type roster. Note
    the surface is the frontmatter as WRITTEN — it is not truncated — so it is
    measured here rather than bounded by a constant.

This is the read-side statement of the rule ``core/optimize/write_budget.py``
already applies on the write side (``standing_tokens_per_session``: a rung-2
skill write is charged only its always-loaded frontmatter). Both sides now
resolve the split through this module so the product cannot answer the same
question two ways.

Nothing here reads telemetry. How many times an on-demand file was ACTUALLY
invoked is a separate, observed quantity — see ``core/summarize/invocations``.
"""
from __future__ import annotations

import re
from pathlib import Path

#: Whole body re-sent at the head of every session that loads the file.
ALWAYS = "always"
#: Body loaded when the skill is invoked (`Skill` tool / a typed slash command).
SKILL = "skill"
#: Body loaded when the slash command is typed or invoked.
COMMAND = "command"
#: Body loaded when that agent is spawned as a subagent.
AGENT = "agent"

#: The three on-demand classes, in one place so callers don't re-list them.
ON_DEMAND_CLASSES: frozenset[str] = frozenset({SKILL, COMMAND, AGENT})

#: Directory fragment -> load class. Matched against the POSIX-normalised path,
#: so a Windows path and a `~`-relative one classify identically. Mirrors
#: the three directories ``write_budget._always_loaded_chars`` treats as
#: on-demand on the write side (it now classifies through this module).
_CLASS_BY_FRAGMENT: tuple[tuple[str, str], ...] = (
    ("/skills/", SKILL),
    ("/commands/", COMMAND),
    ("/agents/", AGENT),
)

#: A leading YAML frontmatter block: ``---`` on its own first line, closed by a
#: ``---`` (or ``...``) line. Deliberately anchored at position 0 — a ``---``
#: horizontal rule further down a body is not frontmatter.
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n.*?(?:\r?\n)(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL)


def classify(path: str) -> str:
    """Which load class ``path`` belongs to — never raises, defaults to ALWAYS.

    Defaulting an unrecognised path to ALWAYS is the conservative direction for
    a COST estimate only in the sense that it keeps today's behaviour; it is
    also correct for the catalog's own default membership (`CLAUDE.md` and
    friends are matched by name, not by directory).
    """
    normalised = str(path or "").replace("\\", "/")
    for fragment, load_class in _CLASS_BY_FRAGMENT:
        if fragment in normalised:
            return load_class
    return ALWAYS


def invocation_key(path: str, load_class: str | None = None) -> str:
    """The name an invocation of ``path`` is recorded under, or ``""``.

    A skill is invoked by its DIRECTORY name (`.claude/skills/ship/SKILL.md`
    -> ``ship``); a command and an agent by their filename stem
    (`.claude/commands/govern.md` -> ``govern``). ALWAYS-class files are never
    invoked and return ``""``.
    """
    cls = load_class or classify(path)
    if cls not in ON_DEMAND_CLASSES:
        return ""
    p = Path(str(path or "").replace("\\", "/"))
    if cls == SKILL:
        # `<...>/skills/<slug>/SKILL.md` — the slug is the parent directory.
        # A flat `<...>/skills/<slug>.md` (not the documented layout, but cheap
        # to tolerate) falls back to the stem.
        parent = p.parent.name
        return parent if parent and parent != "skills" else p.stem
    return p.stem


def split_always_resident(text: str, load_class: str) -> tuple[str, str]:
    """Split ``text`` into ``(always_resident, on_demand)`` for its load class.

    ALWAYS -> the whole body is resident and nothing is on demand.
    Otherwise -> the leading YAML frontmatter is resident (that is what the
    harness lists the file by) and the rest arrives only on invocation. A file
    of this class with no frontmatter surfaces nothing at all, so its resident
    portion is empty rather than a guessed prefix.
    """
    body = text or ""
    if load_class not in ON_DEMAND_CLASSES:
        return body, ""
    match = _FRONTMATTER_RE.match(body)
    if match is None:
        return "", body
    return body[: match.end()], body[match.end():]
