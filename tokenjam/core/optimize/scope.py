"""Which filesystem a filesystem-reading analyzer is allowed to look at.

`deadweight`, `relearn` and `summarize` take their evidence from the filesystem
by design: the telemetry DB holds spans, while the transcripts, the MCP config
and the always-loaded prompt files these three reason about live on disk. That
split is correct. What was missing is any way to say WHICH disk.

Until this module there was no scope override at all — only the undocumented
`TJ_CLAUDE_PROJECTS_ROOT` env var — so `--db` implied an isolation it never
provided. Pointing `tj serve` at a freshly created, completely empty database
still surfaced recurring-mistake findings carrying real file paths from an
unrelated project on the operator's machine, still defaulted the review inbox's
"where to write it" target to a real path outside that database's world, and
still spent ~30 seconds walking the operator's real `~/.claude/projects` before
the first paint — on a store with nothing in it.

**The contract**, in precedence order:

1. `--projects-root PATH` (or `[optimize] projects_root`) — scan exactly that
   root. This is the escape hatch for every case below.
2. `TJ_CLAUDE_PROJECTS_ROOT` — unchanged, now documented.
3. An explicit `--db`: the filesystem scans are SUPPRESSED. A caller who named
   a database named the corpus they want reasoned about, and mixing a second,
   unrelated corpus into that report is the bug. Suppression is also what makes
   an empty store paint immediately instead of walking a five-thousand-file
   transcript tree to conclude nothing.
4. Otherwise — `~/.claude/projects`, exactly as before. A normal run is
   unchanged: no flag, no env var, no `--db`, same behaviour as always.

Suppression is reported, never silent: `AnalyzerScope.reason` names why, the
analyzers attach it to their findings, and the surfaces distinguish "scanned,
found nothing" from "did not scan" (root anti-pattern 22). A user who does want
machine history alongside a custom database says so with `--projects-root`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: The env var honored since before this module existed. Kept as the second
#: precedence step so every existing test fixture and integration keeps working.
PROJECTS_ROOT_ENV = "TJ_CLAUDE_PROJECTS_ROOT"

def default_claude_home() -> Path:
    """``~/.claude``, resolved LAZILY.

    Never a module-level constant: `Path.home()` baked at import time survives
    a later `HOME` repoint, which is how a test that carefully isolates itself
    still ends up scanning the developer's real transcript tree (the same trap
    `core/agent_config._settings_paths` documents for the global MCP/settings
    read, and the reason `tests/conftest.py` has to patch `config.SEARCH_PATHS`
    by hand).
    """
    return Path.home() / ".claude"


def default_projects_root() -> Path:
    """``~/.claude/projects``, resolved lazily — see `default_claude_home`."""
    return default_claude_home() / "projects"

#: The one sentence every surface prints when scans were suppressed. Written
#: once here so the CLI, the API and the analyzers cannot word it differently.
SUPPRESSED_BY_DB_REASON = (
    "filesystem scans are scoped off because an explicit --db was given; "
    "pass --projects-root PATH to scan a transcript root alongside it"
)


@dataclass(frozen=True)
class AnalyzerScope:
    """The filesystem an analyzer run may read, and where that came from.

    `enabled` is False only for the suppressed case. It is deliberately NOT the
    same question as "does `projects_root` exist" — a scoped root that happens
    to be empty was still scanned, and saying so is the difference between an
    honest empty state and a claim the run never checked.
    """

    projects_root: Path
    claude_home: Path
    #: What `~` expands to inside this scope. `summarize` and `trim` read
    #: always-loaded prompt files across SEVERAL agent homes (`~/.claude`,
    #: `~/.gemini`, `~/.codex`), so scoping them to the Claude home alone would
    #: miss most of the catalog while still escaping to the real home for the
    #: rest. The home is the scope unit for those; the projects root is the
    #: scope unit for transcript scans.
    home: Path
    enabled: bool
    source: str
    reason: str | None = None


def _home_for(claude_home: Path) -> Path:
    """The home directory implied by a Claude home.

    `<home>/.claude` yields `<home>`. A custom scope directory is treated as
    its own home rather than reaching up into whatever contains it — the same
    no-escape rule `_claude_home_for` applies one level down.
    """
    return claude_home.parent if claude_home.name == ".claude" else claude_home


def _claude_home_for(projects_root: Path) -> Path:
    """The Claude config home implied by a projects root.

    A projects root is conventionally `<claude-home>/projects`, so the home is
    its parent. When a caller points `--projects-root` at a directory by
    another name we treat that directory as the home too, rather than reaching
    up into whatever happens to contain it — escaping the scope the caller just
    drew is the one thing this module exists to prevent.
    """
    return projects_root.parent if projects_root.name == "projects" else projects_root


def _scoped(projects_root: Path, source: str) -> AnalyzerScope:
    root = projects_root.expanduser()
    claude_home = _claude_home_for(root)
    return AnalyzerScope(
        projects_root=root,
        claude_home=claude_home,
        home=_home_for(claude_home),
        enabled=True,
        source=source,
    )


@dataclass(frozen=True)
class WriteScope:
    """The two roots the Approve path needs, derived from ONE resolved scope.

    An approved fix has two halves that must agree: the target the card
    SUGGESTS, and the root the API is willing to WRITE under. They were derived
    independently once — the suggestion followed `AnalyzerScope.claude_home`
    while the API's guard stayed hardcoded to the process's real `Path.home()`
    — so with `--projects-root` pointed outside `$HOME` the UI suggested a
    target and the API then refused that exact write. Both halves resolve
    through this one type now, precisely so they cannot drift apart again.

    * `suggest_root` — where `relearn_apply.default_target_path` roots a
      user-global suggestion (the scope's Claude home).
    * `allowed_root` — the OUTERMOST directory an approved write may land in.
      It is the scope's `home`, not its Claude home, because project-scoped
      targets (a repo's own `CLAUDE.md`, `.claude/skills/…`) legitimately sit
      beside `~/.claude` rather than inside it.

    `suggest_root` is always inside-or-equal `allowed_root` by construction
    (`_home_for` either returns the Claude home's parent or the Claude home
    itself), which is what makes "the guard can never reject our own
    suggestion" a property of the type rather than a coincidence.
    """

    suggest_root: Path
    allowed_root: Path
    source: str


def resolve_write_scope(
    config: object | None = None, *, scope: AnalyzerScope | None = None,
) -> WriteScope:
    """The suggest/allow roots for this run — pass `scope` when one is already
    resolved (the analyzer path), `config` when it is not (the API path).

    Never raises and never touches the filesystem, exactly like
    `resolve_analyzer_scope`. With no `--projects-root`, no env var and no
    scope override this returns `~/.claude` and `~`, i.e. the allowed root is
    the real home and the guard behaves byte-for-byte as it always has.
    """
    resolved = scope if scope is not None else resolve_analyzer_scope(config)
    return WriteScope(
        suggest_root=resolved.claude_home.expanduser(),
        allowed_root=resolved.home.expanduser(),
        source=resolved.source,
    )


def resolve_analyzer_scope(config: object | None = None) -> AnalyzerScope:
    """Resolve the filesystem scope for this run — see the module docstring.

    Never raises and never touches the filesystem: a scope is a decision about
    WHERE to look, taken before anything is read.
    """
    configured = _configured_root(config)
    if configured:
        return _scoped(Path(configured), "flag")

    env = os.environ.get(PROJECTS_ROOT_ENV)
    if env:
        return _scoped(Path(env), "env")

    if _db_path_is_explicit(config):
        # A path is still carried so callers that only want to display the
        # scope have something to show; `enabled` is what gates every read.
        return AnalyzerScope(
            projects_root=default_projects_root(),
            claude_home=default_claude_home(),
            home=Path.home(),
            enabled=False,
            source="suppressed_by_db",
            reason=SUPPRESSED_BY_DB_REASON,
        )

    return _scoped(default_projects_root(), "default")


def _configured_root(config: object | None) -> str | None:
    """`[optimize] projects_root`, which `--projects-root` also writes into."""
    optimize = getattr(config, "optimize", None)
    value = getattr(optimize, "projects_root", None)
    return str(value) if value else None


def _db_path_is_explicit(config: object | None) -> bool:
    """Whether this run's store path came from `--db` rather than discovery.

    Set by the CLI at the point it applies the override, so config discovery
    and an explicit flag stay distinguishable after the fact — a config file
    that happens to name the same path is a normal run, not a scoped one.
    """
    storage = getattr(config, "storage", None)
    return bool(getattr(storage, "path_is_explicit", False))
