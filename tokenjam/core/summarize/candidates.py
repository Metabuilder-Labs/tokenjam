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

import fnmatch
import glob as _glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

from tokenjam.core.summarize import load_semantics
from tokenjam.core.summarize.catalog import load_catalog
from tokenjam.core.summarize.detect import MIN_PROSE_WORDS, analyze
from tokenjam.core.summarize.estimate import UNMEASURED_PRIOR_RATIO, tokens_saved
from tokenjam.core.summarize.relocate import relocatable_content_chars
from tokenjam.core.summarize.route import recommend_route

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
    #: How this file reaches the model — ``core/summarize/load_semantics``.
    #: ``always`` (whole body every session) vs ``skill``/``command``/``agent``
    #: (frontmatter always, body only when invoked). Defaulted so every
    #: existing construction of this dataclass keeps working.
    load_class: str = load_semantics.ALWAYS
    #: The name an invocation of this file is recorded under (``""`` for an
    #: always-resident file, which is never "invoked").
    invocation_key: str = ""
    #: ``est_tokens_saved`` split across the two load semantics: the part
    #: removed from what every session carries, and the part removed from what
    #: arrives only on invocation. They sum to ``est_tokens_saved`` up to the
    #: rounding in :func:`estimate.tokens_saved`, which floors each part
    #: independently.
    always_resident_tokens_saved: int = 0
    on_demand_tokens_saved: int = 0
    #: Source size of the always-resident portion (the whole file for an
    #: ALWAYS-class one, the frontmatter for an on-demand one). The write-side
    #: budget in ``core/optimize/write_budget`` measures the existing agent-file
    #: footprint off this, so read and write price the same quantity.
    always_resident_chars: int = 0
    #: The scan root this file was found under. ``path`` relative to it is the
    #: file's SLOT (``CLAUDE.md``, ``.claude/commands/ship.md``) — the same slot
    #: under two roots is the same file seen twice, which is how a consumer
    #: identifies copies without assuming their bytes still agree. Empty for a
    #: global-scope file, which has no project root.
    scan_root: str = ""
    #: Which route to a smaller file this candidate actually wants — see
    #: ``core/summarize/route``. Compression is one of four routes to the
    #: published size target and the only one that costs specificity, so a
    #: rule-heavy instruction file is flagged as a PRUNE candidate rather than
    #: being offered compression as though it were the obvious move. The full
    #: user-facing reasoning is NOT carried per candidate (it is a paragraph and
    #: a corpus scan holds hundreds of these); surfaces render it via
    #: `route.recommend_route` for the one file the user is looking at.
    reduction_route: str = ""
    #: Share of this file's prose words living in discrete directives — the
    #: evidence behind ``reduction_route``, carried so a consumer can show the
    #: evidence and not only the verdict.
    directive_share: float = 0.0
    #: True when this rule already declares `paths:` frontmatter, so it is never
    #: told to path-scope itself. Advice only; deliberately does not reprice it.
    already_path_scoped: bool = False
    #: Content characters this file would shed by RELOCATING its reference
    #: sections into a non-loaded document — net of the pointer stubs left
    #: behind, and zero unless the classifier is confident (see
    #: ``core/summarize/relocate`` and ``core/summarize/classify``). A different
    #: OPERATION on the same file, not an addition to ``est_tokens_saved``:
    #: relocating a section and then compressing it would price the same text
    #: twice (Critical Rule 27), so consumers pick one, never a sum.
    relocatable_content_chars: int = 0

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
            "load_class": self.load_class,
            "invocation_key": self.invocation_key,
            "always_resident_tokens_saved": self.always_resident_tokens_saved,
            "on_demand_tokens_saved": self.on_demand_tokens_saved,
            "always_resident_chars": self.always_resident_chars,
            "scan_root": self.scan_root,
            "reduction_route": self.reduction_route,
            "directive_share": round(self.directive_share, 4),
            "already_path_scoped": self.already_path_scoped,
            "relocatable_content_chars": self.relocatable_content_chars,
            "relocatable_tokens": self.relocatable_tokens,
        }

    @property
    def relocatable_tokens(self) -> int:
        """One-time always-resident token reduction from relocating, on the
        shared chars->tokens constant so it is comparable with
        ``est_tokens_saved``."""
        from tokenjam.core.summarize.detect import CHARS_PER_TOKEN
        return max(0, round(self.relocatable_content_chars / CHARS_PER_TOKEN))


@dataclass(frozen=True)
class ScanResult:
    """Candidates plus what was scanned (transparency, DEC-020)."""

    candidates: list[Candidate]
    root: str | None
    recursive: bool
    globals_checked: int
    walk_capped: bool
    note: str
    #: How many project roots the scan enumerated: 1 for a single-root scan (an
    #: explicit PATH, ``--repo``, or the default cwd), N when the caller passed
    #: ``project_roots`` — see :func:`list_candidates`.
    project_roots_scanned: int = 0

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "count": len(self.candidates),
            "root": self.root,
            "recursive": self.recursive,
            "globals_checked": self.globals_checked,
            "walk_capped": self.walk_capped,
            "note": self.note,
            "project_roots_scanned": self.project_roots_scanned,
        }


# --------------------------------------------------------------------------- #
# Catalog matching — what counts as a "prompt"
# --------------------------------------------------------------------------- #

def _matches_glob(path: Path, pattern: str) -> bool:
    """``path`` matched against a catalog glob, from the right.

    ``Path.match`` is used for the ordinary patterns, but it does NOT give
    ``**`` its recursive meaning before Python 3.13 — it matches a single
    component, so `.claude/rules/**/*.md` would silently fail to recognize
    `.claude/rules/ecc/common/coding-style.md` as a rules file and it would be
    reported as an unrecognized "other" document. A recursive pattern is
    therefore matched as "this literal directory prefix appears somewhere in the
    path, and the filename matches the tail".
    """
    if "**" not in pattern:
        return path.match(pattern)
    head, _, tail = pattern.partition("**/")
    if not fnmatch.fnmatch(path.name, tail.rsplit("/", 1)[-1]):
        return False
    head = head.strip("/")
    if not head:
        return True
    posix = path.as_posix()
    return f"/{head}/" in posix or posix.startswith(f"{head}/")


def _is_prompt(path: Path) -> bool:
    """True iff ``path`` is a catalog-known prompt file — by exact name, or by
    matching a catalog glob from the right (so a nested `.claude/skills/*/SKILL.md`
    is recognized regardless of where it sits)."""
    cat = load_catalog()
    if path.name in cat.project_files:
        return True
    return any(_matches_glob(path, pattern) for pattern in cat.project_globs)


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


def is_safe_scan_root(path: "str | os.PathLike[str]") -> bool:
    """Whether ``path`` may be used as a scan root at all.

    The same gate :func:`find_repo_root` stops at, exposed for callers that
    derive roots some other way (``core/summarize/repo_roots`` derives them from
    the analysed window's recorded working directories). Refuses the filesystem
    root, the user's home, any bare top-level dir, and every catalog
    ``forbidden_roots`` entry — so no derivation path can point a scan at
    ``$HOME`` or ``/``.
    """
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    home = Path.home().resolve()
    extra = {Path(p).expanduser().resolve() for p in load_catalog().forbidden_roots}
    return not _is_boundary(resolved, home) and resolved not in extra


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
               ratio: float, scan_root: Path | None = None) -> Candidate | None:
    # A symlink is not a candidate, and NOT because it might duplicate another
    # file — because the FIX refuses it. `prepare`, `check`, `apply_staged` and
    # `undo` all bail on a link rather than rewrite through it (the write could
    # land outside where you expect), so a saving offered on one is unreachable
    # by every path this product offers. Pricing it would claim money no fix can
    # collect (Critical Rule 22). The link TARGET is still a fine candidate when
    # a scan reaches it directly; it is the link that is refused, not the file.
    if path.is_symlink():
        return None
    text = _read(path)
    if text is None:
        return None
    b = analyze(text)
    if b.prose_words < min_prose_words:
        return None
    # Split the SAME reduction across the two load semantics, measured on the
    # two halves of the real text rather than apportioned by a ratio: only the
    # always-resident half is worth (sessions x calls), the on-demand half is
    # worth (invocations). See core/summarize/load_semantics.
    load_class = load_semantics.classify(str(path), text)
    resident_text, on_demand_text = load_semantics.split_always_resident(text, load_class)
    # Why this file is long, and therefore which route to a smaller one it
    # wants. Compression is only one of four, and the only one that costs
    # specificity — so a rule-heavy instruction file is not offered compression
    # as though it were the obvious move. Diagnosis only; nothing is written.
    advice = recommend_route(text=text, load_class=load_class)
    # How much of this file is REFERENCE that could be moved out wholesale
    # instead of compressed in place. Measured here, where the text is already
    # in hand, rather than by a second read in the analyzer. Only meaningful
    # for an always-resident file: relocating a skill BODY saves nothing,
    # because the body is not resident until it is invoked.
    relocatable = (
        relocatable_content_chars(text)
        if load_class == load_semantics.ALWAYS else 0
    )
    return Candidate(
        scan_root=str(scan_root) if scan_root is not None else "",
        path=str(path), prose_words=b.prose_words, total_chars=b.total_chars,
        protected_blocks=b.protected_blocks, est_tokens_saved=tokens_saved(b, ratio),
        pricing_mode=mode, scope=scope, is_prompt=_is_prompt(path),
        load_class=load_class,
        invocation_key=load_semantics.invocation_key(str(path), load_class),
        always_resident_tokens_saved=tokens_saved(analyze(resident_text), ratio),
        on_demand_tokens_saved=tokens_saved(analyze(on_demand_text), ratio),
        always_resident_chars=len(resident_text),
        reduction_route=advice.route, directive_share=advice.directive_share,
        already_path_scoped=advice.already_path_scoped,
        relocatable_content_chars=relocatable,
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

def _expand_home(raw: str, home: Path | None) -> str:
    """Expand a leading `~` against `home`, or the real home when None.

    `home` is the analyzer scope's home (see `core/optimize/scope.py`). The
    catalog's global paths span several agent homes (`~/.claude`, `~/.gemini`,
    `~/.codex`), so scoping them means redirecting `~` itself — anything
    narrower would leave most of the catalog reading the operator's real files
    while a `--projects-root` was in force.
    """
    if home is None:
        return os.path.expanduser(raw)
    if raw == "~":
        return str(home)
    if raw.startswith("~/"):
        return str(home / raw[2:])
    return raw


def _global_targets(home: Path | None = None) -> list[Path]:
    """Catalog global/system paths ("~" expanded; glob patterns expanded)."""
    out: list[Path] = []
    for raw in load_catalog().global_paths:
        ep = _expand_home(raw, home)
        if any(ch in ep for ch in "*?["):
            # recursive=True so a `**` in a catalog global path (e.g.
            # `~/.claude/rules/**/*.md`) descends; without it `glob` treats `**`
            # as a plain `*` and sees only the top level of a nested rule tree.
            out.extend(Path(x) for x in sorted(_glob.glob(ep, recursive=True)))
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


def _already_summarized(config: "TjConfig | None", path: str) -> bool:
    """True if we previously summarized ``path`` and it's unchanged since — skip it (PR3).

    Only re-reads the file when a backup actually exists for it (cheap for the common case).
    """
    if config is None:
        return False
    from tokenjam.core.summarize import backup
    from tokenjam.core.summarize.session import sha256
    out = backup.recorded_output(config, path)
    if out is None:
        return False
    try:
        current = Path(path).read_text(encoding="utf-8")
    except OSError:
        return False
    return sha256(current) == out


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
    ratio: float = UNMEASURED_PRIOR_RATIO,
    extra_exts: Iterable[str] = (),
    home: "Path | None" = None,
    project_root: "Path | None" = None,
    project_roots: "Sequence[str | os.PathLike[str]] | None" = None,
) -> ScanResult:
    """Find summarize candidates per DEC-020/021. Advisory: reads only, never writes.

    Three parameters govern WHERE the scan looks, and they are deliberately
    different questions:

    * ``home`` scopes the catalog's ``~``-rooted global paths.
    * ``project_root`` relocates the implicit "where am I" that the single-root
      project and repo discovery start from, without any of the widening a
      positional ``path`` carries: ``path`` marks the scan explicit, which opens
      ``ext_set`` to all-md and relabels the scope from "project" to "path". A
      caller that only needs the scan confined — an analyzer honoring
      ``--projects-root`` — wants the relocation and none of the widening.
      ``None`` means the process's cwd, which is what every caller got before
      this parameter existed.
    * ``project_roots`` replaces that single-root project scope with a scan of
      EVERY given root. It exists for the optimize analyzer, which prices a
      whole telemetry window that spans many repos: pricing only the repo the
      process happens to sit in makes every other repo's always-resident
      ``CLAUDE.md`` contribute exactly nothing. Roots are scanned with the SAME
      catalog-default net as the cwd scan — deliberately NOT via ``path``, for
      the widening reason above. They are the caller's responsibility to derive
      and to confine within whatever boundary applies (see
      ``core/summarize/repo_roots`` and the summarize analyzer's scope
      composition); they are ignored entirely when an explicit
      ``path``/``--recursive``/``--repo`` asks for a specific scope.

    ``project_root`` and ``project_roots`` never both apply: a caller supplying
    an explicit root list has already decided the population, and
    ``project_root`` is the fallback starting point for when it has not.
    """
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
    roots_scanned = 0

    def _add(p: Path, scope: str, scan_root: Path | None = None) -> None:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            return
        seen.add(key)
        c = _candidate(p, mode, scope, min_prose_words, ratio, scan_root)
        if c is not None and not _already_summarized(config, c.path):
            cands.append(c)

    # 1) Globals (the floor) — always catalog prompts, unless suppressed.
    globals_checked = 0
    if include_global:
        for gp in _global_targets(home):
            if gp.exists():
                globals_checked += 1
                _add(gp, "global")

    # 2) Project scope.
    explicit = path is not None
    # Everywhere below that would have consulted the process's cwd consults
    # this instead, so a confined scan cannot reach a project outside its root
    # through the one door the catalog scoping left open.
    here = Path(project_root).expanduser() if project_root is not None else Path.cwd()
    target = Path(path).expanduser() if path is not None else here

    if explicit and not target.exists():
        # A typo'd / missing PATH shouldn't silently show only globals. This MUST be
        # checked before the --recursive branch: with an explicit PATH, walk_root is
        # the (non-existent) target rather than None, so _walk_targets() just returns
        # [] and the error gets swallowed. Single source of truth for the note.
        tail = "showing globals only" if cands else "nothing to show"
        note = f"PATH not found: {target} — {tail}."
        root_used = target
    elif explicit and target.is_file():
        # NOTE: a specific file still invokes the catalog globals (added above as
        # the floor) unless --no-global; scoping to JUST the named file is deferred
        # (DEF-005).
        root_used = target
        _add(target, "path")
    elif recursive:
        walk_root = target if explicit else find_repo_root(here)
        if walk_root is None:
            tail = "showing globals only" if cands else "nothing to show"
            note = ("--recursive needs a git repo or an explicit PATH; no safe root "
                    f"found — {tail}.")
        else:
            root_used = walk_root
            roots_scanned = 1
            paths, walk_capped = _walk_targets(walk_root, ext_set)
            for p in paths:
                _add(p, "path" if explicit else "repo", walk_root)
    elif project_roots is not None:
        # Corpus-wide project scope: the same catalog-default net, once per root.
        for raw_root in project_roots:
            scan_root = Path(raw_root).expanduser()
            if not scan_root.is_dir():
                continue
            roots_scanned += 1
            for p in _project_targets(scan_root, ext_set):
                _add(p, "project", scan_root)
        root_used = None
    else:
        if repo and not explicit:
            found = find_repo_root(here)
            if found is None:                       # --repo but no repo: don't fake a "repo" root
                scope_root, scope = here, "project"
                note = "--repo: no git repo found — scanning the current directory instead."
            else:
                scope_root, scope = found, "repo"
        else:
            scope_root = target
            scope = "path" if explicit else "project"
        root_used = scope_root
        roots_scanned = 1
        for p in _project_targets(scope_root, ext_set):
            _add(p, scope, scope_root)

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
        project_roots_scanned=roots_scanned,
    )
