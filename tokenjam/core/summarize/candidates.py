"""Discover summarize candidates from the known prompt-file catalog (DEC-020/021).

The **net** is flag-gated: bare `tj summarize list` considers only catalog-known
prompt files (by name/location); **any** widening input — a ``path``,
``--repo``, ``--recursive``, or extra extensions — opens it to **all `*.md`**
(plus those extensions). A "scan for me" command should be generous when asked;
the default stays minimal.

**Ranking is sectioned** (DEC-021, refined): what you *asked for* comes first.
  1. the scanned location (an explicit PATH / ``--repo`` / ``--recursive`` / cwd)
     before the always-on catalog **globals** — the requested scope is the focus;
     globals are supplementary and the CLI shows them under a divider;
  2. WITHIN a section, kind is the differentiator: catalog-recognized **prompts**
     first, then other files, grouped by directory (path), biggest first.

Boundary-safe (pure-filesystem `.git` detection; never `/`, home, or a bare
top-level). Advisory only — reads and reports, never writes.
"""
from __future__ import annotations

import glob as _glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator

from tokenjam.core.summarize.catalog import load_catalog
from tokenjam.core.summarize.detect import MIN_PROSE_WORDS, analyze
from tokenjam.core.summarize.estimate import DEFAULT_TARGET_RATIO, tokens_saved

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig

# Directories never descended into during a --recursive walk.
_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", ".tj",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".idea", ".vscode", "site-packages",
}
# Extensions the opened-up net considers by default (the user can add more).
_DEFAULT_EXTS = {".md", ".markdown"}
_MAX_BYTES = 512 * 1024     # never read a file larger than this
_MIN_BYTES = 400            # stat pre-filter: below this can't hold ~100 prose words
_MAX_WALK_FILES = 5000      # hard cap on a --recursive walk (then bail, flagged)


@dataclass(frozen=True)
class Candidate:
    """One file flagged as worth summarizing."""

    path: str
    prose_words: int
    total_chars: int
    protected_blocks: int
    est_tokens_saved: int
    pricing_mode: str
    scope: str                  # "global" | "project" | "repo" | "path"
    is_prompt: bool             # matched a catalog prompt name/location

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "prose_words": self.prose_words,
            "total_chars": self.total_chars,
            "protected_blocks": self.protected_blocks,
            "est_tokens_saved": self.est_tokens_saved,
            "pricing_mode": self.pricing_mode,
            "scope": self.scope,
            "is_prompt": self.is_prompt,
            "kind": "prompt" if self.is_prompt else "other",
        }


@dataclass(frozen=True)
class ScanResult:
    """Candidates plus what was scanned (transparency, DEC-020)."""

    candidates: list[Candidate]
    root: str | None
    recursive: bool
    globals_checked: int
    walk_capped: bool
    note: str

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "count": len(self.candidates),
            "root": self.root,
            "recursive": self.recursive,
            "globals_checked": self.globals_checked,
            "walk_capped": self.walk_capped,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Catalog matching — what counts as a "prompt"
# --------------------------------------------------------------------------- #

def _is_prompt(path: Path) -> bool:
    """True iff ``path`` is a catalog-known prompt file — by exact name, or by
    matching a catalog glob from the right (so a nested `.claude/skills/*/SKILL.md`
    is recognized regardless of where it sits)."""
    cat = load_catalog()
    if path.name in cat.project_files:
        return True
    return any(path.match(pattern) for pattern in cat.project_globs)


def _norm_ext(ext: str) -> str:
    e = ext.strip().lower().lstrip(".")
    return f".{e}" if e else ""


# --------------------------------------------------------------------------- #
# Repo detection + boundary safety — pure filesystem, no git subprocess.
# --------------------------------------------------------------------------- #

def _is_boundary(d: Path, home: Path) -> bool:
    """A dir we must never treat as a repo root: filesystem root, the user's home,
    or any bare top-level dir (<=2 path components)."""
    return d == Path(d.anchor) or d == home or len(d.parts) <= 2


def find_repo_root(start: "str | os.PathLike[str]") -> Path | None:
    """Nearest ancestor of ``start`` containing a ``.git`` (dir or file), or None.

    Walks up by path only — no listing, no subprocess — and STOPS (returns None)
    at the first boundary (filesystem root, the user's home, any bare top-level,
    or a catalog ``forbidden_roots`` entry). So a repo root is always >=2 levels
    below ``/`` and never home/system; a project nested under a system dir
    (``/opt/foo``, ``~/code/x``) still resolves correctly.
    """
    cur = Path(start).expanduser().resolve()
    home = Path.home().resolve()
    extra = {Path(p).expanduser().resolve() for p in load_catalog().forbidden_roots}
    for d in [cur, *cur.parents]:
        if _is_boundary(d, home) or d in extra:
            return None
        if (d / ".git").exists():
            return d
    return None


# --------------------------------------------------------------------------- #
# File -> Candidate
# --------------------------------------------------------------------------- #

def _read(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _candidate(path: Path, mode: str, scope: str, min_prose_words: int,
               ratio: float) -> Candidate | None:
    text = _read(path)
    if text is None:
        return None
    b = analyze(text)
    if b.prose_words < min_prose_words:
        return None
    return Candidate(
        path=str(path), prose_words=b.prose_words, total_chars=b.total_chars,
        protected_blocks=b.protected_blocks, est_tokens_saved=tokens_saved(b, ratio),
        pricing_mode=mode, scope=scope, is_prompt=_is_prompt(path),
    )


def _pricing_mode(config: "TjConfig | None") -> str:
    if config is None:
        return "unknown"
    from tokenjam.core.framing import config_declared_plan, pricing_mode_for
    plan = config_declared_plan(config)
    return pricing_mode_for(plan) if plan else "unknown"


# --------------------------------------------------------------------------- #
# Target enumeration
# --------------------------------------------------------------------------- #

def _global_targets() -> list[Path]:
    """Catalog global/system paths ("~" expanded; glob patterns expanded)."""
    out: list[Path] = []
    for raw in load_catalog().global_paths:
        ep = os.path.expanduser(raw)
        if any(ch in ep for ch in "*?["):
            out.extend(Path(x) for x in sorted(_glob.glob(ep)))
        else:
            out.append(Path(ep))
    return out


def _project_targets(root: Path, ext_set: set[str]) -> Iterator[Path]:
    """Catalog names + globs at ``root`` (always); plus, when ``ext_set`` is
    non-empty (the net is open), every root-level file with a matching extension."""
    cat = load_catalog()
    for name in sorted(cat.project_files):
        yield root / name
    for pattern in cat.project_globs:
        yield from sorted(root.glob(pattern))
    if ext_set:
        try:
            for p in sorted(root.iterdir()):
                if p.is_file() and p.suffix.lower() in ext_set:
                    yield p
        except OSError:
            pass


def _walk_targets(root: Path, ext_set: set[str]) -> tuple[list[Path], bool]:
    """--recursive: catalog filenames + matching extensions under ``root``; skip-dirs
    pruned, stat pre-filtered, capped. Returns ``(paths, capped)``."""
    names = load_catalog().project_files
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if fn not in names and p.suffix.lower() not in ext_set:
                continue
            try:
                if p.stat().st_size < _MIN_BYTES:
                    continue
            except OSError:
                continue
            out.append(p)
            if len(out) >= _MAX_WALK_FILES:
                return out, True
    return out, False


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def list_candidates(
    path: "str | os.PathLike[str] | None" = None,
    *,
    config: "TjConfig | None" = None,
    recursive: bool = False,
    repo: bool = False,
    include_global: bool = True,
    min_prose_words: int = MIN_PROSE_WORDS,
    ratio: float = DEFAULT_TARGET_RATIO,
    extra_exts: Iterable[str] = (),
) -> ScanResult:
    """Find summarize candidates per DEC-020/021. Advisory: reads only, never writes."""
    mode = _pricing_mode(config)
    extra = {e for e in (_norm_ext(x) for x in extra_exts) if e}
    # The net opens to all-md (+ extras) the moment ANY widening input is given.
    widened = (path is not None) or recursive or repo or bool(extra)
    ext_set = (_DEFAULT_EXTS | extra) if widened else set()

    seen: set[str] = set()
    cands: list[Candidate] = []
    note = ""
    walk_capped = False
    root_used: Path | None = None

    def _add(p: Path, scope: str) -> None:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            return
        seen.add(key)
        c = _candidate(p, mode, scope, min_prose_words, ratio)
        if c is not None:
            cands.append(c)

    # 1) Globals (the floor) — always catalog prompts, unless suppressed.
    globals_checked = 0
    if include_global:
        for gp in _global_targets():
            if gp.exists():
                globals_checked += 1
                _add(gp, "global")

    # 2) Project scope.
    explicit = path is not None
    target = Path(path).expanduser() if explicit else Path.cwd()

    if explicit and target.is_file():
        # NOTE: a specific file still invokes the catalog globals (added above as
        # the floor) unless --no-global; scoping to JUST the named file is deferred
        # (DEF-005).
        root_used = target
        _add(target, "path")
    elif recursive:
        walk_root = target if explicit else find_repo_root(Path.cwd())
        if walk_root is None:
            tail = "showing globals only" if cands else "nothing to show"
            note = ("--recursive needs a git repo or an explicit PATH; no safe root "
                    f"found — {tail}.")
        else:
            root_used = walk_root
            paths, walk_capped = _walk_targets(walk_root, ext_set)
            for p in paths:
                _add(p, "path" if explicit else "repo")
    else:
        if repo and not explicit:
            found = find_repo_root(Path.cwd())
            if found is None:                       # --repo but no repo: don't fake a "repo" root
                scope_root, scope = Path.cwd(), "project"
                note = "--repo: no git repo found — scanning the current directory instead."
            else:
                scope_root, scope = found, "repo"
        else:
            scope_root = target
            scope = "path" if explicit else "project"
            if explicit and not target.exists():    # a typo'd PATH shouldn't silently show only globals
                tail = "showing globals only" if cands else "nothing to show"
                note = f"PATH not found: {target} — {tail}."
        root_used = scope_root
        for p in _project_targets(scope_root, ext_set):
            _add(p, scope)

    # Sectioned sort (DEC-021, refined): what the user asked for first — the scanned
    # location (non-global) before the always-on catalog globals (supplementary, shown
    # under a divider). Kind is the WITHIN-section differentiator: prompts before other
    # files; then by directory (path, alpha); size desc within.
    cands.sort(key=lambda c: (1 if c.scope == "global" else 0, 0 if c.is_prompt else 1,
                              str(Path(c.path).parent), -c.est_tokens_saved))
    return ScanResult(
        candidates=cands,
        root=str(root_used) if root_used is not None else None,
        recursive=recursive,
        globals_checked=globals_checked,
        walk_capped=walk_capped,
        note=note,
    )
