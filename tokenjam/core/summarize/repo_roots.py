"""Which project roots the analysed window actually worked in.

`core/summarize/candidates` scans project scope from ONE root. For `tj summarize
list` that root is the cwd, which is right: the user is asking about the repo
they are standing in. For the optimize analyzer it is wrong — it prices a whole
telemetry window that spans every repo the user touched, so a single-cwd scan
leaves every other repo's always-resident `CLAUDE.md` contributing nothing at
all, while that same file is re-sent at the head of every one of its own repo's
sessions. Which repo the analyzer's process happens to sit in is an accident of
invocation; the window it prices is not.

The roots are OBSERVED, not guessed: Claude Code records the working directory
in each transcript, and `core/summarize/invocations` already reads every
transcript in the window (with the shared persistent parse cache), so the cwds
come back from a walk that was happening anyway.

Two honesty rules govern what survives that derivation:

* **A recorded directory that no longer exists on disk contributes nothing.**
  There is no file to read, so there is no reduction to offer and no figure to
  quote (Critical Rule 22 — never show a figure the user cannot act on). It is
  counted, so the basis string can say how much of the corpus was skipped, but
  it is never reconstructed.
* **A root must pass the same boundary gate the cwd scan passes**
  (:func:`candidates.is_safe_scan_root`): never ``/``, never ``$HOME``, never a
  bare top-level directory. A derivation path that can be pointed at ``$HOME``
  by a stray recorded cwd is a derivation path that will eventually scan the
  user's whole home directory.

Nothing here reads telemetry or prices anything; it turns recorded paths into
scannable roots and reports what it dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tokenjam.core.summarize.candidates import find_repo_root, is_safe_scan_root


@dataclass(frozen=True)
class ResolvedRoots:
    """Scannable project roots plus what the derivation had to drop."""

    #: Deduplicated, sorted, existing, boundary-safe roots to scan.
    roots: tuple[Path, ...] = ()
    #: Distinct working directories the window recorded, before filtering.
    recorded: int = 0
    #: Distinct recorded directories that no longer exist on disk. Scanned as
    #: nothing, never reconstructed — the blind spot is reported, not filled.
    vanished: int = 0
    #: Distinct recorded directories that exist but are refused as scan roots by
    #: the boundary gate (``$HOME``, ``/``, a bare top-level dir).
    refused: int = 0
    #: Distinct recorded directories that exist and are safe, but fall outside
    #: the caller-supplied ``within`` boundary. Dropped, and counted so the
    #: basis can say the population was confined rather than simply absent.
    out_of_scope: int = 0


def _inside(root: Path, boundary: Path) -> bool:
    """Whether ``root`` sits at or under ``boundary``."""
    try:
        root.relative_to(boundary)
    except ValueError:
        return False
    return True


def resolve_roots(
    cwds: Iterable[str], *, within: Path | None = None,
) -> ResolvedRoots:
    """Turn recorded working directories into scannable project roots.

    For each recorded directory that still exists, BOTH the directory itself and
    its enclosing git repo root (when they differ) become scan roots: Claude Code
    loads `CLAUDE.md` from the working directory and from its ancestors, so a
    session launched in a sub-package of a monorepo really does carry both files.
    Duplicate content across those roots is the caller's problem to collapse —
    the same file reached two ways resolves to one path here, but two byte-equal
    COPIES (a git worktree) are two real files and are returned as such.

    ``within``, when given, is an outer boundary every returned root must sit
    at or under. It is how this derivation composes with an explicitly drawn
    filesystem scope (``--projects-root`` and its env var): the scope decides
    WHERE the run may look, and this function decides WHICH directories inside
    that are worth looking at. A recorded cwd is not itself trustworthy for
    confinement — a transcript stored under a scoped projects root can record a
    working directory anywhere on disk — so the boundary is applied to the
    derived roots rather than assumed from where the transcript lived. ``None``
    means no boundary was drawn and the observed population stands as recorded;
    it is NOT a default root to filter against, since filtering an unscoped run
    against the default projects root would empty the population rather than
    confine it.
    """
    roots: set[Path] = set()
    recorded: set[str] = set()
    vanished: set[str] = set()
    refused: set[str] = set()
    out_of_scope: set[str] = set()

    boundary = within.expanduser().resolve() if within is not None else None

    for raw in cwds:
        if not raw:
            continue
        recorded.add(raw)
        try:
            directory = Path(raw).expanduser()
            if not directory.is_dir():
                vanished.add(raw)
                continue
            resolved = directory.resolve()
        except (OSError, RuntimeError):
            vanished.add(raw)
            continue
        found = False
        confined_out = False
        for candidate_root in (resolved, find_repo_root(resolved)):
            if candidate_root is None or not is_safe_scan_root(candidate_root):
                continue
            if boundary is not None and not _inside(candidate_root, boundary):
                # Safe and real, but past the boundary the caller drew. Counted
                # separately from `refused` so "outside your scope" never reads
                # as "structurally unscannable".
                confined_out = True
                continue
            roots.add(candidate_root)
            found = True
        if found:
            continue
        if confined_out:
            out_of_scope.add(raw)
        else:
            refused.add(raw)

    return ResolvedRoots(
        roots=tuple(sorted(roots)),
        recorded=len(recorded),
        vanished=len(vanished),
        refused=len(refused),
        out_of_scope=len(out_of_scope),
    )
