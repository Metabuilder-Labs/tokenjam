"""`--db` must isolate the filesystem-reading analyzers, not just the store.

Reproduced against a freshly created, completely empty throwaway database: the
review inbox still showed recurring-mistake entries carrying real file paths
from an unrelated project, the recurring-fix modal still defaulted its write
target to a real path on the operator's machine, and `/api/v1/optimize` still
took ~30s because `deadweight` and `relearn` walked the operator's real
`~/.claude/projects` regardless of `--db`.

These pin the scope contract that fixes all three. See
`tokenjam.core.optimize.scope` for the contract itself.
"""

from __future__ import annotations

from pathlib import Path

from tokenjam.core.optimize.scope import (
    PROJECTS_ROOT_ENV,
    default_projects_root,
    resolve_analyzer_scope,
)


class _Storage:
    def __init__(self, path="~/.tj/telemetry.duckdb", path_is_explicit=False):
        self.path = path
        self.path_is_explicit = path_is_explicit


class _Optimize:
    def __init__(self, projects_root=None):
        self.projects_root = projects_root


class _Config:
    def __init__(self, projects_root=None, db_explicit=False):
        self.optimize = _Optimize(projects_root)
        self.storage = _Storage(path_is_explicit=db_explicit)


def test_a_normal_run_is_unchanged(monkeypatch):
    """No flag, no env var, no --db: today's behaviour, exactly."""
    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)
    scope = resolve_analyzer_scope(_Config())
    assert scope.enabled is True
    assert scope.projects_root == default_projects_root()
    assert scope.source == "default"
    assert scope.reason is None


def test_an_explicit_db_suppresses_the_filesystem_scans(monkeypatch):
    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)
    scope = resolve_analyzer_scope(_Config(db_explicit=True))
    assert scope.enabled is False
    assert scope.source == "suppressed_by_db"
    # Suppression is never silent — the reason must name the escape hatch.
    assert scope.reason is not None
    assert "--projects-root" in scope.reason


def test_projects_root_wins_over_an_explicit_db(monkeypatch, tmp_path):
    """The escape hatch: a caller who wants machine history alongside a custom
    database says so, and gets it."""
    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)
    scope = resolve_analyzer_scope(
        _Config(projects_root=str(tmp_path), db_explicit=True)
    )
    assert scope.enabled is True
    assert scope.projects_root == tmp_path
    assert scope.source == "flag"


def test_projects_root_wins_over_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "from-env"))
    scope = resolve_analyzer_scope(_Config(projects_root=str(tmp_path / "from-flag")))
    assert scope.projects_root == tmp_path / "from-flag"
    assert scope.source == "flag"


def test_the_env_var_still_works(monkeypatch, tmp_path):
    """Every existing fixture and integration sets this; it must not regress."""
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path))
    scope = resolve_analyzer_scope(_Config())
    assert scope.enabled is True
    assert scope.projects_root == tmp_path
    assert scope.source == "env"


def test_the_env_var_wins_over_an_explicit_db(monkeypatch, tmp_path):
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path))
    scope = resolve_analyzer_scope(_Config(db_explicit=True))
    assert scope.enabled is True
    assert scope.projects_root == tmp_path


def test_a_scoped_but_empty_root_is_enabled_not_suppressed(monkeypatch, tmp_path):
    """"Scanned, found nothing" is a different statement from "did not scan",
    and the surfaces render them differently (root anti-pattern 22)."""
    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)
    absent = tmp_path / "does-not-exist"
    scope = resolve_analyzer_scope(_Config(projects_root=str(absent)))
    assert scope.enabled is True
    assert scope.reason is None


def test_claude_home_is_the_parent_of_a_conventional_projects_root(tmp_path):
    root = tmp_path / ".claude" / "projects"
    scope = resolve_analyzer_scope(_Config(projects_root=str(root)))
    assert scope.claude_home == tmp_path / ".claude"


def test_claude_home_never_escapes_a_custom_scope(tmp_path):
    """A root by another name is treated as the home itself — reaching up into
    whatever contains it would escape the scope the caller just drew."""
    root = tmp_path / "fixture-transcripts"
    scope = resolve_analyzer_scope(_Config(projects_root=str(root)))
    assert scope.claude_home == root
    assert tmp_path not in [scope.claude_home]


def test_a_missing_or_duck_typed_config_never_raises(monkeypatch):
    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)
    assert resolve_analyzer_scope(None).enabled is True
    assert resolve_analyzer_scope(object()).enabled is True


def test_a_user_path_is_expanded(monkeypatch):
    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)
    scope = resolve_analyzer_scope(_Config(projects_root="~/somewhere/projects"))
    assert "~" not in str(scope.projects_root)
    assert scope.projects_root == Path.home() / "somewhere" / "projects"


def test_the_default_root_follows_a_repointed_home(monkeypatch, tmp_path):
    """Resolved lazily, never baked at import: a constant computed at import
    time survives a later HOME repoint, and a test that carefully isolates
    itself then scans the developer's real transcript tree anyway."""
    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_analyzer_scope(_Config()).projects_root == (
        tmp_path / ".claude" / "projects"
    )
