"""Context-load audit: everything that enters a Claude Code session and what it
costs in context, classified by how it gets there — never by where it sits.

This is the "CleanMyMac for Claude Code" scan: an inventory, not a
recommendation engine. Three classes, evidence-based (never assumed from a
file's directory):

``class1``
    Loads unconditionally, resident every turn: ``CLAUDE.md``/``AGENTS.md``/
    ``GEMINI.md`` files, unscoped ``.claude/rules/*.md`` files, and the
    always-listed frontmatter (name + description) of every skill/command/
    agent — the harness lists these in the tool/skill roster whether or not
    they are ever invoked (see ``core/summarize/load_semantics``, which this
    module classifies through rather than re-deriving).

``class2``
    Fires conditionally, unrequested: hooks wired to an event in
    ``settings.json`` (global or a plugin's own ``hooks/hooks.json``), a
    ``paths:``-scoped rule (fires on a matching file read, never at launch),
    and a skill/agent body whose own description reads as a self-invoking
    trigger ("proactively", "must use this", ...).

``class3``
    Runs only when the user invokes it: a slash command body, and a skill
    body with no auto-invoke language in its description.

A file present on disk that nothing loads (catalog-unknown, or excluded by a
disabled plugin) is reported separately with zero cost — see
``ScopeAudit.unloaded`` — never priced as if it were resident (Critical Rule 22
applies to this scanner as much as to any analyzer surface it feeds).

Reuses, never duplicates: ``catalog`` (on-disk locations),
``candidates._project_targets``/``_global_targets`` shape (re-derived here at
smaller scope since candidates.py's targets are file lists, not classified
rows), and ``load_semantics`` (the always/on-demand split + the
``paths:``-scope test). Plugins are handled separately per instruction — the
catalog deliberately excludes them (see ``agent_files.toml``'s comment) because
enablement depends on ``enabledPlugins`` in ``settings.json``, which is not
on-disk state the catalog can see.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tokenjam.core.summarize import load_semantics
from tokenjam.core.summarize.catalog import load_catalog
from tokenjam.core.summarize.detect import CHARS_PER_TOKEN

log = logging.getLogger(__name__)

CLASS_1 = "class1"
CLASS_2 = "class2"
CLASS_3 = "class3"

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
PLUGINS_DIR = Path.home() / ".claude" / "plugins"
INSTALLED_PLUGINS_FILE = PLUGINS_DIR / "installed_plugins.json"

GLOBAL_SCOPE = "global"

#: hook event name -> how often it fires. Matches the seven events named in
#: the brief; anything unlisted (a future event Claude Code adds) still shows,
#: just with its own bare name as the frequency rather than a guess.
_HOOK_EVENT_FREQUENCY: dict[str, str] = {
    "SessionStart": "per session",
    "SessionEnd": "per session",
    "UserPromptSubmit": "every turn",
    "PreToolUse": "every tool call",
    "PostToolUse": "every tool call",
    "PostToolUseFailure": "every tool call (on failure)",
    "Stop": "per session",
    "SubagentStop": "per subagent",
    "PreCompact": "on compaction",
    "Notification": "on notification",
}

#: Phrases in a skill/agent DESCRIPTION that read as a self-invoking trigger
#: rather than a "use me when asked" reference. Deliberately small and
#: literal — a heuristic, not a parser; false negatives (missed auto-invokes)
#: are the safe direction here since they land the body in class3 (still
#: reported, just under "on demand" instead of "conditional"), not in the
#: unloaded bucket.
_AUTO_INVOKE_MARKERS: tuple[str, ...] = (
    "proactively", "must use this", "must be used before", "before any",
    "you must use", "automatically invoke", "always invoke", "before ANY",
)

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)(?:\r?\n)(?:---|\.\.\.)[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_DESCRIPTION_RE = re.compile(r"^description:[ \t]*(.*)$", re.MULTILINE)


def _tokens(chars: int) -> int:
    return max(0, round(chars / CHARS_PER_TOKEN))


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _read_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _frontmatter_description(text: str) -> str:
    """The frontmatter `description:` value, stripped of quotes — best-effort,
    single-line only (a folded/multi-line YAML value is left as its first
    line, which is enough for the auto-invoke keyword check below)."""
    match = _FRONTMATTER_RE.match(text or "")
    if match is None:
        return ""
    found = _DESCRIPTION_RE.search(match.group(1))
    if found is None:
        return ""
    return found.group(1).strip().strip('"').strip("'")


def _looks_auto_invoked(description: str) -> bool:
    lowered = description.lower()
    return any(marker.lower() in lowered for marker in _AUTO_INVOKE_MARKERS)


def _truncate(text: str, limit: int = 90) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


#: A file's own leading H1 (`# Title`) — the one honest, mechanically-derived
#: "what is this" for a plain prose file (CLAUDE.md, a rule). Never invented:
#: a file with no heading gets an empty description rather than a guess.
_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _first_heading(text: str, path: Path | None = None) -> str:
    """The file's first heading, as its description.

    Two things are handled rather than surfaced verbatim. Em dashes, because
    they are banned in this product's user-facing copy and a heading lifted out
    of someone else's file would smuggle them onto the page. And a heading that
    merely restates the file's own name (``# CLAUDE.md`` on ``CLAUDE.md``),
    which fills the description column while telling the reader nothing the
    Source column has not already said.

    A name-echo heading yields an EMPTY description; it deliberately does not
    fall through to the file's next heading. A long document can carry several
    top-level headings, and the second one is a section title, not a summary of
    the file. Surfacing it would describe a 43k-token CLAUDE.md as, say,
    "Install in dev mode": a confident claim about the largest row on the page
    that happens to be false. Blank is the honest answer when the file does not
    title itself, and the column already renders that case.
    """
    match = _HEADING_RE.search(text or "")
    if not match:
        return ""
    heading = " ".join(match.group(1).replace("—", ":").replace("–", "-").split())
    if path is not None and heading.casefold() in {
        path.name.casefold(),
        Path(path.name).stem.casefold(),
    }:
        return ""
    return _truncate(heading)


def _display_path(path: Path) -> str:
    """``path`` with the user's home collapsed to ``~`` — readable in a group
    label without leaking the full absolute path every row already carries."""
    home = Path.home()
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class Row:
    """One context-audit line: a source that reaches the model somehow.

    ``family_kind`` + ``family_qualifier`` are presentation metadata only —
    they decide how :func:`rows_for_display` GROUPS this row with its
    siblings (a rules directory, a plugin's skill descriptions, ...) and never
    affect ``chars``/``tokens``/totals, which are always computed over these
    individual rows. ``family_kind == ""`` means "never grouped, always its
    own row" (e.g. a lone CLAUDE.md)."""

    source: str
    trigger: str
    chars: int
    frequency: str
    cls: str
    scope: str
    description: str = ""
    family_kind: str = ""
    family_qualifier: str = ""

    @property
    def tokens(self) -> int:
        return _tokens(self.chars)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "trigger": self.trigger, "chars": self.chars,
            "tokens": self.tokens, "frequency": self.frequency,
            "class": self.cls, "scope": self.scope, "description": self.description,
        }


#: family_kind -> the plural noun used in a multi-member group's label.
#: "rule_dir" is handled separately (a directory-shaped label, not a count
#: noun) — see `_group_label`.
_FAMILY_NOUN: dict[str, str] = {
    "skill_desc": "skill descriptions", "command_desc": "command descriptions",
    "agent_desc": "agent descriptions", "skill_body": "skill bodies",
    "command_body": "command bodies", "agent_body": "agent bodies",
    "hook": "hooks",
}


def _group_label(family_kind: str, qualifier: str, n: int) -> str:
    if family_kind == "rule_dir":
        return f"{qualifier}/* x{n}"
    noun = _FAMILY_NOUN.get(family_kind, "items")
    return f"{qualifier} {n} {noun}".strip() if qualifier else f"{n} {noun}"


#: How many member names a group's derived description names before falling
#: back to "+N more" — long enough to be useful, short enough to stay a
#: one-line "what it is" cell rather than a second member table.
_GROUP_DESCRIPTION_NAME_CAP = 6


def _short_member_name(row: Row) -> str:
    """The short, honest name one member contributes to a group's derived
    description — never invented, always something already on the row:

    * a rule file -> its filename stem (``coding-style.md`` -> ``coding-style``)
    * a hook -> the event it fires on (``PreToolUse``, not the shell command)
    * a skill/command/agent description or body -> its invocation slug
      (``load_semantics.invocation_key``, the same name an invocation is
      recorded under), falling back to the filename stem if that's empty.
    """
    if row.family_kind == "rule_dir":
        return Path(row.source).stem
    if row.family_kind == "hook":
        return row.trigger.split(" (matcher:", 1)[0].strip()
    key = load_semantics.invocation_key(row.source)
    return key or Path(row.source).stem


def _group_description(members: Sequence[Row]) -> str:
    """The group's own "what it is" column: a shared description when every
    member happens to carry the identical one (rare), otherwise a
    comma-joined, capped list of member names — e.g. a rules directory's
    description becomes its own file stems ("coding-style, git-workflow,
    testing, ..., +4 more"), so the column is populated from evidence already
    on hand rather than left blank just because the members disagree.
    """
    descriptions = {m.description for m in members if m.description}
    if len(descriptions) == 1:
        return next(iter(descriptions))
    names: list[str] = []
    seen: set[str] = set()
    for m in members:
        name = _short_member_name(m)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    if not names:
        return ""
    if len(names) > _GROUP_DESCRIPTION_NAME_CAP:
        shown = names[:_GROUP_DESCRIPTION_NAME_CAP]
        return ", ".join(shown) + f", +{len(names) - _GROUP_DESCRIPTION_NAME_CAP} more"
    return ", ".join(names)


def rows_for_display(rows: Sequence[Row]) -> list[dict[str, Any]]:
    """Group ``rows`` by family (``(family_kind, family_qualifier)``) into the
    page's display shape, sorted by tokens descending (biggest cost first —
    the whole point of the redesign this exists for).

    A family with exactly one member is never wrapped in a group — it renders
    as a plain row, same as a row with no family at all. Only a REAL group
    (>=2 members) gets a `kind: "group"` entry with its members carried
    alongside for the UI's expand affordance. Un-grouped chars/tokens are
    never altered by this function; it only changes how they are PRESENTED.
    """
    groups: dict[tuple[str, str], list[Row]] = {}
    order: list[tuple[str, str]] = []
    for r in rows:
        key = (r.family_kind, r.family_qualifier) if r.family_kind else ("", r.source)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    display: list[dict[str, Any]] = []
    for key in order:
        members = groups[key]
        if len(members) == 1:
            r = members[0]
            display.append({
                "kind": "row", "label": r.source, "description": r.description,
                "chars": r.chars, "tokens": r.tokens, "frequency": r.frequency,
                "trigger": r.trigger, "members": [],
            })
            continue
        family_kind, qualifier = key
        chars = sum(m.chars for m in members)
        tokens = sum(m.tokens for m in members)
        ranked = sorted(members, key=lambda m: -m.tokens)
        description = _group_description(ranked)
        display.append({
            "kind": "group",
            "label": _group_label(family_kind, qualifier, len(members)),
            "description": description,
            "chars": chars, "tokens": tokens,
            "frequency": members[0].frequency, "trigger": members[0].trigger,
            "members": [
                {"source": m.source, "chars": m.chars, "tokens": m.tokens,
                 "trigger": m.trigger, "description": m.description}
                for m in ranked
            ],
        })
    display.sort(key=lambda d: -d["tokens"])
    return display


@dataclass(frozen=True)
class UnloadedRow:
    """A file present on disk that NOTHING currently loads — zero cost, on
    purpose (Critical Rule 22): existing on disk is not evidence of being
    loaded, and this is where that distinction is made visible."""

    source: str
    chars: int
    reason: str
    scope: str

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "chars": self.chars, "reason": self.reason,
                 "scope": self.scope}


@dataclass(frozen=True)
class ScopeAudit:
    """One scope's (global, or one project root's) three class tables plus
    its not-loaded pile. Totals are computed only over class1 — the headline
    the brief calls for is the always-resident floor, not a cross-class sum
    (class2/class3 rows do not all fire on every turn, so summing them with
    class1 would be a mixed-basis figure)."""

    scope: str
    class1: tuple[Row, ...] = ()
    class2: tuple[Row, ...] = ()
    class3: tuple[Row, ...] = ()
    unloaded: tuple[UnloadedRow, ...] = ()

    @property
    def class1_total_chars(self) -> int:
        return sum(r.chars for r in self.class1)

    @property
    def class1_total_tokens(self) -> int:
        return sum(r.tokens for r in self.class1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "class1": rows_for_display(self.class1),
            "class2": rows_for_display(self.class2),
            "class3": rows_for_display(self.class3),
            "class2_count": len(self.class2),
            "class3_count": len(self.class3),
            "unloaded": [r.to_dict() for r in self.unloaded],
            "unloaded_count": len(self.unloaded),
            "class1_total_chars": self.class1_total_chars,
            "class1_total_tokens": self.class1_total_tokens,
        }


@dataclass(frozen=True)
class ContextAuditResult:
    """The full page payload: one global scope + one per scanned project root,
    plus what the plugin pass saw. Global and project class1 totals are
    reported SEPARATELY and never summed (a project total mixed with the
    always-on global total would double-count the floor every project already
    pays once)."""

    global_scope: ScopeAudit
    projects: tuple[ScopeAudit, ...] = ()
    plugins_enabled: int = 0
    plugins_disabled: int = 0
    last_scanned_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "global": self.global_scope.to_dict(),
            "projects": [p.to_dict() for p in self.projects],
            "plugins_enabled": self.plugins_enabled,
            "plugins_disabled": self.plugins_disabled,
            "last_scanned_at": self.last_scanned_at,
        }


# --------------------------------------------------------------------------- #
# CLAUDE.md / AGENTS.md / rules — reuse the catalog, split unscoped vs
# paths:-scoped rules via load_semantics (never assumed from directory alone).
# --------------------------------------------------------------------------- #

def _is_rule_path(path: Path) -> bool:
    return "/rules/" in path.as_posix()


#: Subdirectories of a `.claude/` dir the catalog already scans (rules,
#: skills, commands, agents) — never re-walked for "unloaded" discovery, since
#: their catalog-known members are already reported as loaded elsewhere.
_KNOWN_CLAUDE_SUBDIRS = frozenset({"rules", "skills", "commands", "agents"})
#: Subdirectories that hold operational state (transcripts, caches, logs, team
#: coordination), never prompt material — walking these would report session
#: history and cache files as "unreferenced markdown", which is noise, not a
#: finding. This list is intentionally an ALLOWLIST-shaped skip: only real,
#: known operational dirs are excluded, so an unrecognised subdirectory is
#: walked (the safe direction for THIS scanner is to surface too much, since
#: unloaded rows cost nothing and are clearly labelled zero-cost).
_UNLOADED_SCAN_SKIP_DIRS = frozenset({
    "plugins", "projects", "cache", "sessions", "session-data", "session-env",
    "teams", "jobs", "daemon", "telemetry", "backups", "file-history", "ide",
    "homunculus", "mcp-servers", "paste-cache", "hud", "hooks", "shell-snapshots",
    "debug", "metrics", "tasks", "skills-archive", "node_modules", ".git",
})
_UNLOADED_SCAN_MAX_FILES = 2000  # hard cap so a huge stray dir can't stall a request


def _discover_unreferenced_md(claude_dir: Path, known: set[Path], scope: str) -> list[UnloadedRow]:
    """Every ``*.md`` under ``claude_dir`` that the catalog does NOT already
    account for (``known``) — reported at zero cost, never priced, per
    Critical Rule 22: a file existing on disk is not evidence of being loaded.

    Bounded to ``.claude/`` itself (never the whole home directory or repo):
    top-level stray files (e.g. a `.claude/NOTES.md` nobody wired up), plus
    any subdirectory that isn't one of the catalog's own known dirs and isn't
    known operational state (sessions, caches, plugin installs — see
    ``_UNLOADED_SCAN_SKIP_DIRS``). The concrete case this exists for: a
    directory like `.claude/product-state/*.md` that holds real prose but
    that nothing in the harness auto-loads.
    """
    if not claude_dir.is_dir():
        return []
    rows: list[UnloadedRow] = []
    count = 0

    def _maybe_add(path: Path, reason: str) -> bool:
        nonlocal count
        if path.suffix.lower() != ".md" or not path.is_file():
            return True
        if path in known:
            return True
        try:
            size = path.stat().st_size
        except OSError:
            return True
        rows.append(UnloadedRow(str(path), size, reason, scope))
        count += 1
        return count < _UNLOADED_SCAN_MAX_FILES

    try:
        for child in sorted(claude_dir.iterdir()):
            if child.is_file():
                if not _maybe_add(child, "top-level file in .claude/, not a catalog-known name"):
                    return rows
            elif child.is_dir() and child.name not in _KNOWN_CLAUDE_SUBDIRS \
                    and child.name not in _UNLOADED_SCAN_SKIP_DIRS:
                for path in sorted(child.rglob("*.md")):
                    if not _maybe_add(path, f"inside .claude/{child.name}/, which nothing auto-loads"):
                        return rows
    except OSError:
        pass
    return rows


def _catalog_prose_rows(paths: Sequence[Path], scope: str) -> tuple[list[Row], list[UnloadedRow]]:
    """CLAUDE.md-shaped files (project_files) and rules (project_globs'
    ``rules`` entries) from the catalog: always class1, UNLESS a rule carries
    ``paths:`` frontmatter, which is class2 (fires only on a matching file
    read — see ``load_semantics.PATH_SCOPED``).

    Rule files are grouped by their containing directory (``family_kind
    "rule_dir"``) — a rules TREE (``rules/ecc/common/*.md``) is one
    conceptual thing, and enumerating each file separately is exactly the
    "93 tables, 989 rows" failure this grouping exists to fix. CLAUDE.md/
    AGENTS.md files are never grouped (``family_kind ""``): there is
    normally exactly one per scope, so grouping would add an expander for a
    "group" of one.
    """
    rows: list[Row] = []
    unloaded: list[UnloadedRow] = []
    for path in paths:
        text = _read(path)
        if text is None:
            unloaded.append(UnloadedRow(str(path), 0, "unreadable", scope))
            continue
        chars = len(text)
        description = _first_heading(text, path)
        is_rule = _is_rule_path(path)
        family_kind = "rule_dir" if is_rule else ""
        family_qualifier = _display_path(path.parent) if is_rule else ""
        if is_rule and load_semantics.has_paths_scope(text):
            rows.append(Row(str(path), "matching file read (paths: scope)", chars,
                             "on file read", CLASS_2, scope, description,
                             family_kind, family_qualifier))
        else:
            trigger = "harness auto-load" if not is_rule else "harness auto-load (unscoped rule)"
            rows.append(Row(str(path), trigger, chars, "every turn", CLASS_1, scope,
                             description, family_kind, family_qualifier))
    return rows, unloaded


# --------------------------------------------------------------------------- #
# Skills / commands / agents — frontmatter is class1 (always listed); the BODY
# is class2 only when the skill/agent's own description reads as
# self-invoking, class3 otherwise. A command body is always class3: nothing
# loads it until the user types it.
# --------------------------------------------------------------------------- #

def _skill_command_agent_rows(
    paths: Sequence[Path], scope: str, *, group_qualifier: str = "",
) -> tuple[list[Row], list[UnloadedRow]]:
    """``group_qualifier`` names WHOSE skills/commands/agents these are
    ("global", "this project", or a plugin's short id) — the label a
    multi-member group renders under (e.g. "oh-my-claudecode 41 skill
    descriptions"). Frontmatter rows group under ``<kind>_desc``, body rows
    under ``<kind>_body``, so the always-resident listing and the on-demand
    body are never merged into one group even though they share a source
    file."""
    rows: list[Row] = []
    unloaded: list[UnloadedRow] = []
    for path in paths:
        text = _read(path)
        if text is None:
            unloaded.append(UnloadedRow(str(path), 0, "unreadable", scope))
            continue
        load_class = load_semantics.classify(str(path), text)
        description = _frontmatter_description(text)
        if load_class not in load_semantics.ON_DEMAND_CLASSES or load_class == load_semantics.PATH_SCOPED:
            # Shouldn't happen for a skills/commands/agents path, but never
            # misclassify silently — fall back to reading it as always-resident.
            rows.append(Row(str(path), "harness auto-load", len(text), "every turn", CLASS_1, scope, description))
            continue
        resident, on_demand = load_semantics.split_always_resident(text, load_class)
        kind = {"skill": "skill", "command": "command", "agent": "agent"}[load_class]
        rows.append(Row(str(path), "tool listing", len(resident),
                         "every turn", CLASS_1, scope, description,
                         f"{kind}_desc", group_qualifier))
        if not on_demand:
            continue  # no body (frontmatter-only file) — nothing else to report
        if load_class == load_semantics.COMMAND:
            rows.append(Row(str(path), "user types the slash command", len(on_demand),
                             "on demand", CLASS_3, scope, description,
                             f"{kind}_body", group_qualifier))
        elif _looks_auto_invoked(description):
            rows.append(Row(str(path), f"description implies auto-invoke ({kind})",
                             len(on_demand), "on demand (auto-triggered)", CLASS_2, scope,
                             description, f"{kind}_body", group_qualifier))
        else:
            rows.append(Row(str(path), f"user or model invokes the {kind}", len(on_demand),
                             "on demand", CLASS_3, scope, description,
                             f"{kind}_body", group_qualifier))
    return rows, unloaded


# --------------------------------------------------------------------------- #
# Global scope
# --------------------------------------------------------------------------- #

def _global_paths() -> list[Path]:
    """Catalog global paths, expanded — mirrors
    ``candidates._global_targets`` at the smaller scope this module needs
    (no ``home`` override, since the audit always describes the real user)."""
    import glob as _glob
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in load_catalog().global_paths:
        expanded = str(Path(raw).expanduser())
        if any(ch in expanded for ch in "*?["):
            found = (Path(x) for x in sorted(_glob.glob(expanded, recursive=True)))
        else:
            found = iter([Path(expanded)])
        for p in found:
            # Catalog globals OVERLAP the same way project globs do (a rules
            # file can match more than one pattern) — dedupe rather than
            # report the same file's cost twice.
            if p not in seen:
                seen.add(p)
                out.append(p)
    return [p for p in out if p.exists()]


def scan_global(claude_dir: "Path | None" = None) -> ScopeAudit:
    """The global scope's catalog-known prose (``CLAUDE.md``, ``~/.claude/rules/**``)
    plus the global skill/command/agent frontmatter+body split. Plugin
    contributions are NOT included here — see ``_scan_plugins`` — since
    plugin enablement is a separate gate (``settings.json``) this function
    does not consult.

    ``claude_dir`` overrides where the "unreferenced markdown" discovery walk
    (``_discover_unreferenced_md``) looks; it defaults to the real
    ``~/.claude`` and only needs overriding in tests, so a test never has to
    monkeypatch ``Path.home`` to keep this scanner off the real filesystem."""
    if claude_dir is None:
        claude_dir = Path.home() / ".claude"
    paths = _global_paths()
    prose = [p for p in paths if not any(frag in p.as_posix() for frag in ("/skills/", "/commands/", "/agents/"))]
    sca = [p for p in paths if p not in prose]
    class1: list[Row] = []
    class2: list[Row] = []
    class3: list[Row] = []
    unloaded: list[UnloadedRow] = []

    prose_rows, prose_unloaded = _catalog_prose_rows(prose, GLOBAL_SCOPE)
    unloaded.extend(prose_unloaded)
    for r in prose_rows:
        (class1 if r.cls == CLASS_1 else class2 if r.cls == CLASS_2 else class3).append(r)

    sca_rows, sca_unloaded = _skill_command_agent_rows(sca, GLOBAL_SCOPE, group_qualifier="global")
    unloaded.extend(sca_unloaded)
    for r in sca_rows:
        (class1 if r.cls == CLASS_1 else class2 if r.cls == CLASS_2 else class3).append(r)

    known = set(paths)
    unloaded.extend(_discover_unreferenced_md(claude_dir, known, GLOBAL_SCOPE))

    return ScopeAudit(
        scope=GLOBAL_SCOPE, class1=tuple(class1), class2=tuple(class2),
        class3=tuple(class3), unloaded=tuple(unloaded),
    )


# --------------------------------------------------------------------------- #
# Plugins — NOT via agent_files.toml (deliberately excluded there). Enablement
# is `enabledPlugins` in settings.json; a disabled plugin contributes NOTHING
# here (its skills/hooks are excluded), but its OWN rules files, if it ever
# wrote any into ~/.claude/rules/, are already counted above regardless of
# plugin state — the harness reads that directory on its own merit.
# --------------------------------------------------------------------------- #

def _enabled_plugin_ids() -> tuple[list[str], int]:
    """(enabled plugin ids, disabled count) from settings.json's
    ``enabledPlugins`` map. Never raises — a missing/malformed settings file
    just means no plugins are counted, not a crash."""
    settings = _read_json(SETTINGS_PATH)
    raw = settings.get("enabledPlugins", {})
    if not isinstance(raw, dict):
        return [], 0
    enabled = [pid for pid, on in raw.items() if on]
    disabled = sum(1 for on in raw.values() if not on)
    return enabled, disabled


def _plugin_install_paths(plugin_id: str) -> list[Path]:
    data = _read_json(INSTALLED_PLUGINS_FILE)
    entries = data.get("plugins", {}).get(plugin_id, [])
    if not isinstance(entries, list):
        return []
    out = []
    for entry in entries:
        install_path = entry.get("installPath") if isinstance(entry, dict) else None
        if install_path:
            p = Path(install_path)
            if p.is_dir():
                out.append(p)
    return out


def _plugin_short_name(plugin_id: str) -> str:
    """``"oh-my-claudecode@omc"`` -> ``"oh-my-claudecode"`` — the marketplace
    suffix is installation bookkeeping, not part of what a user would call
    the plugin when reading a group label."""
    return plugin_id.split("@", 1)[0]


def _plugin_hook_rows(hooks_json: Path, plugin_id: str) -> list[Row]:
    data = _read_json(hooks_json)
    events = data.get("hooks", {})
    if not isinstance(events, dict):
        return []
    return _hook_rows_from_events(
        events, GLOBAL_SCOPE, source_prefix=f"plugin:{plugin_id}",
        group_qualifier=_plugin_short_name(plugin_id),
    )


def _scan_plugins() -> tuple[list[Row], list[Row], list[Row], list[UnloadedRow], int, int]:
    enabled_ids, disabled_count = _enabled_plugin_ids()
    class1: list[Row] = []
    class2: list[Row] = []
    class3: list[Row] = []
    unloaded: list[UnloadedRow] = []

    for plugin_id in enabled_ids:
        install_paths = _plugin_install_paths(plugin_id)
        if not install_paths:
            continue
        qualifier = _plugin_short_name(plugin_id)
        for root in install_paths:
            skill_files = sorted(root.glob("skills/*/SKILL.md"))
            command_files = sorted(root.glob("commands/*.md"))
            agent_files = sorted(root.glob("agents/*.md"))
            rows, row_unloaded = _skill_command_agent_rows(
                [*skill_files, *command_files, *agent_files], GLOBAL_SCOPE,
                group_qualifier=qualifier,
            )
            unloaded.extend(row_unloaded)
            for r in rows:
                target = class1 if r.cls == CLASS_1 else class2 if r.cls == CLASS_2 else class3
                # Tag the source with the plugin id for display, without
                # losing the on-disk path (kept as-is; the plugin id is
                # already implicit in the cache path).
                target.append(r)
            hooks_json = root / "hooks" / "hooks.json"
            if hooks_json.exists():
                class2.extend(_plugin_hook_rows(hooks_json, plugin_id))

    return class1, class2, class3, unloaded, len(enabled_ids), disabled_count


# --------------------------------------------------------------------------- #
# Hooks — settings.json (global). Project-level settings.json (.claude/settings.json,
# .claude/settings.local.json) are read too, scoped to that project.
# --------------------------------------------------------------------------- #

def _hook_rows_from_events(
    events: dict, scope: str, *, source_prefix: str = "", group_qualifier: str = "",
) -> list[Row]:
    rows: list[Row] = []
    for event_name, matchers in events.items():
        if not isinstance(matchers, list):
            continue
        frequency = _HOOK_EVENT_FREQUENCY.get(event_name, event_name)
        for matcher_entry in matchers:
            if not isinstance(matcher_entry, dict):
                continue
            matcher = matcher_entry.get("matcher", "")
            hook_list = matcher_entry.get("hooks", [])
            if not isinstance(hook_list, list):
                continue
            for hook in hook_list:
                if not isinstance(hook, dict):
                    continue
                command = str(hook.get("command") or "")
                if not command:
                    continue
                trigger = f"{event_name} hook" + (f" (matcher: {matcher})" if matcher else "")
                source = f"{source_prefix} {command}".strip() if source_prefix else command
                # The command STRING is what's on disk/in config; the hook's
                # actual stdout at runtime (what really lands in context) is
                # not something a static scan can measure, so this reports
                # the config footprint only — honestly smaller than the real
                # cost, never larger.
                rows.append(Row(source, trigger, len(command), frequency, CLASS_2, scope,
                                 "", "hook", group_qualifier))
    return rows


def _settings_hook_rows(settings_path: Path, scope: str, *, group_qualifier: str = "") -> list[Row]:
    data = _read_json(settings_path)
    events = data.get("hooks", {})
    if not isinstance(events, dict):
        return []
    return _hook_rows_from_events(events, scope, group_qualifier=group_qualifier)


# --------------------------------------------------------------------------- #
# Project scope — the CLAUDE.md hierarchy + rules/skills/commands/agents +
# hooks for one project root, reusing the catalog's project-side globs.
# --------------------------------------------------------------------------- #

def _project_catalog_paths(root: Path) -> list[Path]:
    cat = load_catalog()
    out: list[Path] = []
    seen: set[Path] = set()
    for name in sorted(cat.project_files):
        p = root / name
        if p.exists() and p not in seen:
            seen.add(p)
            out.append(p)
    for pattern in cat.project_globs:
        # Catalog globs OVERLAP by design (`.claude/rules/*.md` and
        # `.claude/rules/**/*.md` both match a top-level rules file — see
        # agent_files.toml's own comment on why both are kept), so the same
        # file can turn up from two patterns. Dedupe here rather than push
        # that burden onto every caller, the way `candidates.list_candidates`
        # does with its own `seen` set.
        for p in sorted(root.glob(pattern)):
            if p.exists() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def scan_project(root: Path) -> ScopeAudit:
    scope = str(root)
    all_paths = _project_catalog_paths(root)
    prose = [p for p in all_paths if not any(frag in p.as_posix() for frag in ("/skills/", "/commands/", "/agents/"))]
    sca = [p for p in all_paths if p not in prose]

    class1: list[Row] = []
    class2: list[Row] = []
    class3: list[Row] = []
    unloaded: list[UnloadedRow] = []

    prose_rows, prose_unloaded = _catalog_prose_rows(prose, scope)
    unloaded.extend(prose_unloaded)
    for r in prose_rows:
        (class1 if r.cls == CLASS_1 else class2 if r.cls == CLASS_2 else class3).append(r)

    sca_rows, sca_unloaded = _skill_command_agent_rows(sca, scope, group_qualifier="this project")
    unloaded.extend(sca_unloaded)
    for r in sca_rows:
        (class1 if r.cls == CLASS_1 else class2 if r.cls == CLASS_2 else class3).append(r)

    for settings_name in (".claude/settings.json", ".claude/settings.local.json"):
        settings_path = root / settings_name
        if settings_path.exists():
            class2.extend(_settings_hook_rows(settings_path, scope, group_qualifier="this project"))

    known = set(all_paths)
    unloaded.extend(_discover_unreferenced_md(root / ".claude", known, scope))

    return ScopeAudit(
        scope=scope, class1=tuple(class1), class2=tuple(class2),
        class3=tuple(class3), unloaded=tuple(unloaded),
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def run_context_audit(project_roots: Sequence[str | Path] = ()) -> ContextAuditResult:
    """The whole page's data in one call: the global scope (catalog globals +
    plugins), plus one ``ScopeAudit`` per given project root. Never raises —
    an unreadable file, a malformed settings.json, or a missing plugin
    install just drops that one contribution rather than failing the scan.
    """
    global_audit = scan_global()
    plugin_c1, plugin_c2, plugin_c3, plugin_unloaded, enabled, disabled = _scan_plugins()
    settings_hooks = _settings_hook_rows(SETTINGS_PATH, GLOBAL_SCOPE, group_qualifier="global")

    global_audit = ScopeAudit(
        scope=GLOBAL_SCOPE,
        class1=tuple(list(global_audit.class1) + plugin_c1),
        class2=tuple(list(global_audit.class2) + plugin_c2 + settings_hooks),
        class3=tuple(list(global_audit.class3) + plugin_c3),
        unloaded=tuple(list(global_audit.unloaded) + plugin_unloaded),
    )

    projects: list[ScopeAudit] = []
    seen_roots: set[str] = set()
    for raw_root in project_roots:
        root = Path(raw_root).expanduser()
        try:
            resolved = str(root.resolve())
        except OSError:
            continue
        if resolved in seen_roots or not root.is_dir():
            continue
        seen_roots.add(resolved)
        try:
            projects.append(scan_project(root))
        except OSError:
            log.debug("context audit: could not scan project root %s", root, exc_info=True)

    return ContextAuditResult(
        global_scope=global_audit,
        projects=tuple(sorted(projects, key=lambda p: p.scope)),
        plugins_enabled=enabled,
        plugins_disabled=disabled,
        last_scanned_at=datetime.now(timezone.utc).isoformat(),
    )
