"""Deriving scannable project roots from the window's recorded working dirs.

The summarize analyzer prices a telemetry window that spans every repo the user
touched. These tests pin the two honesty rules that govern the derivation: a
recorded directory that is gone contributes nothing (and is counted, so the
basis can say so), and no derived root may escape the boundary gate the cwd
scan already passes.
"""
from __future__ import annotations

from pathlib import Path

from tokenjam.core.summarize.repo_roots import resolve_roots


def _repo(root: Path) -> Path:
    (root / ".git").mkdir(parents=True)
    return root


def test_existing_directories_become_roots(tmp_path):
    a, b = _repo(tmp_path / "code" / "a"), _repo(tmp_path / "code" / "b")
    resolved = resolve_roots([str(a), str(b)])

    assert set(resolved.roots) == {a.resolve(), b.resolve()}
    assert resolved.recorded == 2
    assert resolved.vanished == 0


def test_vanished_directory_is_counted_not_reconstructed(tmp_path):
    """A deleted repo has no file to read, so no figure may be quoted for it —
    but the size of that blind spot must still be reportable."""
    live = _repo(tmp_path / "code" / "live")
    resolved = resolve_roots([str(live), str(tmp_path / "code" / "deleted")])

    assert resolved.roots == (live.resolve(),)
    assert resolved.recorded == 2
    assert resolved.vanished == 1


def test_subdirectory_contributes_its_repo_root_too(tmp_path):
    """Claude Code loads `CLAUDE.md` from the working directory AND its
    ancestors, so a session launched inside a sub-package really does carry
    both files."""
    repo = _repo(tmp_path / "code" / "repo")
    inner = repo / "packages" / "api"
    inner.mkdir(parents=True)

    resolved = resolve_roots([str(inner)])

    assert set(resolved.roots) == {inner.resolve(), repo.resolve()}


def test_home_is_never_a_scan_root(tmp_path, monkeypatch):
    """A stray recorded cwd must not point the scan at the user's whole home."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    resolved = resolve_roots([str(tmp_path)])

    assert resolved.roots == ()
    assert resolved.refused == 1
    assert resolved.vanished == 0      # it exists; it is refused, not missing


def test_duplicate_and_empty_recordings_collapse(tmp_path):
    repo = _repo(tmp_path / "code" / "repo")

    resolved = resolve_roots([str(repo), str(repo), "", str(repo)])

    assert resolved.roots == (repo.resolve(),)
    assert resolved.recorded == 1


# --- Composition with the filesystem scope (core/optimize/scope) ------------
#
# The scope is the BOUNDARY, the recorded working directories are the SELECTION
# within it. These pin that direction, because the inverse composes silently
# wrong: a boundary applied when none was drawn collapses the population back to
# the single directory this enumeration exists to replace, and every test still
# passes.

def test_no_boundary_keeps_every_recorded_root(tmp_path):
    """`within=None` means no scope was drawn, so the observed population stands.

    This is the unscoped default, and it must never be filtered against a root
    nobody asked for.
    """
    a, b = _repo(tmp_path / "code" / "a"), _repo(tmp_path / "elsewhere" / "b")
    resolved = resolve_roots([str(a), str(b)], within=None)

    assert set(resolved.roots) == {a.resolve(), b.resolve()}
    assert resolved.out_of_scope == 0


def test_boundary_confines_roots_to_its_subtree(tmp_path):
    inside = _repo(tmp_path / "scope" / "inside")
    outside = _repo(tmp_path / "other" / "outside")
    resolved = resolve_roots(
        [str(inside), str(outside)], within=tmp_path / "scope",
    )

    assert resolved.roots == (inside.resolve(),)
    assert resolved.out_of_scope == 1
    # Excluded by a scope, not by the structural safety gate — the finding's
    # basis words those two differently and must be able to tell them apart.
    assert resolved.refused == 0


def test_boundary_excluding_everything_is_counted_not_silent(tmp_path):
    """The worst case: a drawn scope excludes every recorded root.

    The population is empty, but "no root survived the scope" is a different
    statement from "no session recorded a directory", and the count is what
    lets the basis say which happened.
    """
    outside = _repo(tmp_path / "other" / "outside")
    resolved = resolve_roots([str(outside)], within=tmp_path / "scope")

    assert resolved.roots == ()
    assert resolved.recorded == 1
    assert resolved.out_of_scope == 1
    assert resolved.vanished == 0


def test_boundary_keeps_a_root_equal_to_the_boundary(tmp_path):
    """At the boundary is inside it — a scope root that is itself a repo is
    scannable, not excluded by an off-by-one on the containment test."""
    root = _repo(tmp_path / "scope")
    resolved = resolve_roots([str(root)], within=tmp_path / "scope")

    assert resolved.roots == (root.resolve(),)
    assert resolved.out_of_scope == 0


def test_vanished_beats_out_of_scope_for_a_missing_outside_dir(tmp_path):
    """A recorded directory that is GONE is counted as vanished regardless of
    where it sat: there is no file to read either way, and reporting it as
    scope-excluded would imply the scope is what is hiding it."""
    resolved = resolve_roots(
        [str(tmp_path / "other" / "deleted")], within=tmp_path / "scope",
    )

    assert resolved.vanished == 1
    assert resolved.out_of_scope == 0
