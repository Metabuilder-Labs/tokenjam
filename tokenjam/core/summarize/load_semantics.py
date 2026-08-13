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

This is the read-side statement of the rule
``core/rulewrite/delivery.standing_tokens_per_session`` already applies on the
write side (a skill write is priced on only its always-loaded frontmatter).
Both sides now resolve the split through this module so the product cannot
answer the same question two ways.

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
#: A `.claude/rules/*.md` carrying a ``paths:`` glob in its frontmatter. It
#: loads only when Claude READS a file matching the pattern — not at launch and
#: not on every tool use — so unlike its unscoped sibling almost none of it is
#: resident.
#:
#: **The frontmatter is the entire reason it is cheap.** A rule in the same
#: directory WITHOUT ``paths:`` loads at launch with the same priority as
#: `CLAUDE.md`, i.e. it is an ALWAYS file wearing a rules/ path. So this class
#: is decided by reading the frontmatter, never by the directory — classifying
#: on the directory alone is how a rule that lost its ``paths:`` line would
#: keep being priced as if it were still cheap.
PATH_SCOPED = "path-scoped"

#: The on-demand classes, in one place so callers don't re-list them.
#:
#: ``PATH_SCOPED`` is a member: its body arrives only on a matching read, which
#: is the same "not resident until something pulls it in" shape as a skill
#: body. What differs is the TRIGGER (a file read matching a glob, rather than
#: an explicit invocation), and that difference lives in
#: :func:`invocation_key`, which returns "" for it — a path-scoped rule is not
#: invoked by name and has no invocation record to look up.
ON_DEMAND_CLASSES: frozenset[str] = frozenset({SKILL, COMMAND, AGENT, PATH_SCOPED})

#: Directory fragment -> load class. Matched against the POSIX-normalised path,
#: so a Windows path and a `~`-relative one classify identically.
_CLASS_BY_FRAGMENT: tuple[tuple[str, str], ...] = (
    ("/skills/", SKILL),
    ("/commands/", COMMAND),
    ("/agents/", AGENT),
)

#: A ``paths:`` key in a rule's frontmatter. Presence is what makes the rule
#: path-scoped; an empty or absent value leaves it always-resident.
_PATHS_KEY_RE = re.compile(r"^paths[ \t]*:[ \t]*(?P<value>.*)$", re.MULTILINE)

#: A leading YAML frontmatter block: ``---`` on its own first line, closed by a
#: ``---`` (or ``...``) line. Deliberately anchored at position 0 — a ``---``
#: horizontal rule further down a body is not frontmatter.
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n.*?(?:\r?\n)(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL)


def has_paths_scope(text: str) -> bool:
    """Whether a rule's frontmatter carries a non-empty ``paths:`` glob.

    Read from the FRONTMATTER only — a ``paths:`` line in the body is prose.
    This is the single question that decides whether a `.claude/rules/*.md`
    costs almost nothing or costs exactly what a `CLAUDE.md` rule costs, so it
    is asked in one place and both the read and the write side call it.
    """
    body = text or ""
    match = _FRONTMATTER_RE.match(body)
    if match is None:
        return False
    front = body[: match.end()]
    found = _PATHS_KEY_RE.search(front)
    if found is None:
        return False
    # Inline form: `paths: "src/**/*.py"`.
    if found.group("value").strip():
        return True
    # BLOCK form: `paths:` with the globs on the following lines as a YAML
    # list. Its inline value is empty, so an inline-only check reads a
    # perfectly good path-scoped rule as unscoped — and since unscoped means
    # always-resident, that error runs in the expensive direction on the read
    # side and the SAFE direction on the write side, which is exactly the kind
    # of asymmetry that hides. Both forms are ordinary YAML; accept both.
    rest = front[found.end():]
    for line in rest.splitlines():
        stripped = line.strip()
        if not stripped or stripped in ("---", "..."):
            continue
        return stripped.startswith("- ") or stripped == "-"
    return False


def classify(path: str, text: str | None = None) -> str:
    """Which load class ``path`` belongs to — never raises, defaults to ALWAYS.

    Defaulting an unrecognised path to ALWAYS is the conservative direction for
    a COST estimate only in the sense that it keeps today's behaviour; it is
    also correct for the catalog's own default membership (`CLAUDE.md` and
    friends are matched by name, not by directory).

    ``text``, when supplied for a `.claude/rules/*.md`, decides between the two
    very different things that directory holds: a rule WITH a ``paths:`` glob
    loads only on a matching read, while one WITHOUT it loads at launch with
    the same priority as `CLAUDE.md`. Without ``text`` the answer stays ALWAYS,
    which is the safe direction for a cost — it never understates.
    """
    normalised = str(path or "").replace("\\", "/")
    for fragment, load_class in _CLASS_BY_FRAGMENT:
        if fragment in normalised:
            return load_class
    if "/rules/" in normalised and text is not None and has_paths_scope(text):
        return PATH_SCOPED
    return ALWAYS


def invocation_key(path: str, load_class: str | None = None) -> str:
    """The name an invocation of ``path`` is recorded under, or ``""``.

    A skill is invoked by its DIRECTORY name (`.claude/skills/ship/SKILL.md`
    -> ``ship``); a command and an agent by their filename stem
    (`.claude/commands/govern.md` -> ``govern``). ALWAYS-class files are never
    invoked and return ``""``.
    """
    cls = load_class or classify(path)
    if cls not in ON_DEMAND_CLASSES or cls == PATH_SCOPED:
        # A path-scoped rule is pulled in by a FILE READ matching its glob, not
        # by an invocation anyone records under a name. There is no invocation
        # key to return, and inventing one would make it look measurable.
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
    if load_class == PATH_SCOPED:
        # Same split as a skill — the frontmatter is what the harness holds so
        # it knows the rule exists and what it matches; the instruction itself
        # arrives only on a matching read.
        if match is None:
            return "", body
        return body[: match.end()], body[match.end():]
    if match is None:
        return "", body
    return body[: match.end()], body[match.end():]
