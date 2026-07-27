"""The scope contract, exercised through the analyzers themselves.

`test_analyzer_scope.py` pins the resolver. This module pins that the three
filesystem-reading analyzers actually HONOR it — the resolver being correct is
worth nothing if an analyzer still reaches for `Path.home()` on its own, which
is exactly how the leak existed in the first place.

Each test reproduces one of the three symptoms observed against a freshly
created, completely empty throwaway `--db`:

  * cross-project findings from the machine's global transcript tree,
  * an apply target pointing at a real path outside the served database,
  * a ~30s first paint spent walking a transcript tree the run cannot use.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenjam.core.optimize.analyzers import deadweight, relearn
from tokenjam.core.optimize.analyzers import summarize as summarize_analyzer
from tokenjam.core.optimize.relearn_apply import default_target_path
from tokenjam.core.optimize.scope import PROJECTS_ROOT_ENV, resolve_analyzer_scope
from tokenjam.core.optimize.types import AnalyzerContext, OptimizeReport, WindowSummary
from tokenjam.utils.time_parse import utcnow


class _Storage:
    def __init__(self, path_is_explicit: bool):
        self.path = ":memory:"
        self.retention_days = 90
        self.path_is_explicit = path_is_explicit


class _Optimize:
    def __init__(self, projects_root=None):
        self.projects_root = projects_root


class _Config:
    def __init__(self, *, db_explicit=False, projects_root=None):
        self.storage = _Storage(db_explicit)
        self.optimize = _Optimize(projects_root)


def _ctx(config, *, total_tokens=1_000_000) -> AnalyzerContext:
    until = utcnow()
    since = until.replace(year=until.year - 1)
    summary = WindowSummary(
        since=since, until=until, days=365.0, sessions=1, spans=1,
        total_tokens=total_tokens, total_cost_usd=10.0, thin_data=False,
    )
    return AnalyzerContext(
        conn=None,
        config=config,
        since=since,
        until=until,
        agent_id=None,
        window_days=365.0,
        summary=summary,
        report=OptimizeReport(window=summary),
        scope=resolve_analyzer_scope(config),
    )


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch):
    """The env var would satisfy every scope, hiding what is being tested."""
    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)


# ── Suppression under an explicit --db ─────────────────────────────────────

@pytest.mark.parametrize(
    "module,key",
    [
        (deadweight, "deadweight"),
        (relearn, "relearn"),
        (summarize_analyzer, "summarize"),
    ],
)
def test_an_explicit_db_stops_the_analyzer_reading_the_real_home(
    module, key, monkeypatch,
):
    """The reported leak: an empty `--db` still surfaced findings sourced from
    the operator's real machine. No filesystem call may happen at all."""
    # Built BEFORE the traps: resolving the scope legitimately names the
    # nominal home, it just must never READ from it.
    ctx = _ctx(_Config(db_explicit=True))

    def _boom(*_a, **_kw):
        raise AssertionError(
            f"{key} touched the filesystem under an explicit --db"
        )

    monkeypatch.setattr(Path, "rglob", _boom)
    monkeypatch.setattr(Path, "glob", _boom)
    monkeypatch.setattr(Path, "iterdir", _boom)
    monkeypatch.setattr(Path, "is_file", _boom)
    monkeypatch.setattr(Path, "exists", _boom)

    module.run(ctx)

    assert key in ctx.report.findings
    # And it says WHY it is empty — "did not scan" is not "found nothing".
    assert ctx.report.filesystem_scan_skipped_reason is not None
    assert "--projects-root" in ctx.report.filesystem_scan_skipped_reason


@pytest.mark.parametrize(
    "module,key",
    [
        (deadweight, "deadweight"),
        (relearn, "relearn"),
        (summarize_analyzer, "summarize"),
    ],
)
def test_a_scoped_root_is_scanned_and_reports_no_skip(module, key, tmp_path):
    """A scoped-but-empty root was genuinely scanned, so no skip reason is set
    — the surfaces render "found nothing" rather than "did not look"."""
    ctx = _ctx(_Config(db_explicit=True, projects_root=str(tmp_path)))
    module.run(ctx)
    assert key in ctx.report.findings
    assert ctx.report.filesystem_scan_skipped_reason is None


@pytest.mark.parametrize(
    "module,key",
    [
        (deadweight, "deadweight"),
        (relearn, "relearn"),
        (summarize_analyzer, "summarize"),
    ],
)
def test_a_normal_run_still_scans(module, key, tmp_path, monkeypatch):
    """The default path must be untouched: no flag, no --db, same as before."""
    monkeypatch.setenv("HOME", str(tmp_path))
    ctx = _ctx(_Config())
    assert ctx.scope.enabled is True
    module.run(ctx)
    assert key in ctx.report.findings
    assert ctx.report.filesystem_scan_skipped_reason is None


# ── The apply target ───────────────────────────────────────────────────────

def test_the_user_global_apply_target_follows_the_scope(tmp_path):
    """The recurring-fix modal's "where to write it" must land inside the
    scope the findings came from. A live Approve in a sandboxed review view
    otherwise writes a real file to a real location on the operator's machine."""
    scoped_home = tmp_path / ".claude"
    target = default_target_path(1, "user-global", "", "slug", claude_home=scoped_home)
    assert target == str(scoped_home / "CLAUDE.md")
    assert str(Path.home()) not in target


def test_the_apply_target_is_unchanged_without_a_scope():
    target = default_target_path(1, "user-global", "", "slug")
    assert target == str(Path.home() / ".claude" / "CLAUDE.md")


@pytest.mark.parametrize("rung,tail", [
    (1, ["CLAUDE.md"]),
    (2, ["skills", "slug", "SKILL.md"]),
    (3, ["hooks", "slug.py"]),
])
def test_every_rung_of_the_apply_target_is_scoped(rung, tail, tmp_path):
    scoped_home = tmp_path / ".claude"
    target = Path(
        default_target_path(rung, "user-global", "", "slug", claude_home=scoped_home)
    )
    assert target == scoped_home.joinpath(*tail)


# ── The distill cache write path ───────────────────────────────────────────

def test_the_distill_cache_never_writes_to_the_real_home_under_an_isolated_config():
    """A SECOND cache beside `relearn_store.default_cache_path`, missed when
    that one was threaded through `_storage_base_dir`. It fires only after a
    successful distill LLM call, which is why it went unnoticed."""
    config = _Config(db_explicit=True)
    scoped = relearn._distill_cache_dir(config)
    assert Path.home() / ".tj" not in scoped.parents
    assert str(Path.home() / ".tj") not in str(scoped)
    assert scoped.name == "relearn"
    assert scoped.parent.name == "distill_cache"


def test_the_distill_cache_keeps_its_historical_path_without_a_config():
    assert relearn._distill_cache_dir() == (
        Path.home() / ".tj" / "distill_cache" / "relearn"
    )


# ── The apply-target suggestion and the write guard must agree ─────────────
# The suggestion followed the scope's Claude home while the API's guard stayed
# hardcoded to the process's real `Path.home()`, so with `--projects-root`
# outside `$HOME` the UI suggested a target and the API 403'd that exact write.
# Both halves resolve through `resolve_write_scope` now; these pin that.

def test_the_suggested_target_is_always_inside_the_allowed_root(tmp_path):
    """The invariant that makes the two halves unable to disagree: whatever
    root the suggestion is built from sits inside the root the guard
    authorizes. Asserted for a conventional scope, a custom-named one, and
    the unscoped default."""
    from tokenjam.core.optimize.scope import resolve_write_scope

    for projects_root in (
        str(tmp_path / "demo-home" / ".claude" / "projects"),
        str(tmp_path / "just-transcripts"),
        None,
    ):
        write_scope = resolve_write_scope(_Config(projects_root=projects_root))
        suggest, allowed = write_scope.suggest_root, write_scope.allowed_root
        assert suggest == allowed or allowed in suggest.parents, (
            f"{suggest} is not inside {allowed} for projects_root={projects_root}"
        )


def test_the_allowed_root_is_the_real_home_without_a_scope(monkeypatch):
    """The no-flag default must be byte-for-byte what it always was."""
    from tokenjam.core.optimize.scope import PROJECTS_ROOT_ENV, resolve_write_scope

    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)
    assert resolve_write_scope(_Config()).allowed_root == Path.home()


def test_an_explicit_db_still_authorizes_only_the_real_home(monkeypatch):
    """Suppression scopes READS off; it must not widen the WRITE root."""
    from tokenjam.core.optimize.scope import PROJECTS_ROOT_ENV, resolve_write_scope

    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)
    assert resolve_write_scope(_Config(db_explicit=True)).allowed_root == Path.home()


def test_a_scoped_root_authorizes_its_own_suggested_target(tmp_path):
    """The exact case that 403'd: `--projects-root` under a throwaway home
    outside `$HOME`, whose own suggested target must be writable."""
    from tokenjam.core.optimize.scope import resolve_write_scope

    demo_home = tmp_path / "demo-home"
    write_scope = resolve_write_scope(
        _Config(projects_root=str(demo_home / ".claude" / "projects")),
    )
    assert write_scope.allowed_root == demo_home
    suggested = Path(
        default_target_path(1, "user-global", "", "slug",
                            claude_home=write_scope.suggest_root),
    )
    assert suggested == demo_home / ".claude" / "CLAUDE.md"
    assert write_scope.allowed_root in suggested.parents


# ── The scan must STAY inside the root, not merely start there ─────────────
#
# The tests above assert that a scoped run recorded no skip reason, which says
# it looked — never that it looked only where it was allowed to. `summarize`
# scans two halves: a catalog of `~`-rooted global prompt files, and a project
# scan. Only the first was scoped, so with a root given and a cwd outside it
# the second walked the cwd anyway and surfaced unrelated files as candidates.

def _summarize_candidate_paths(ctx) -> list[str]:
    finding = ctx.report.findings["summarize"]
    return [c.path for c in finding.candidates]


def test_a_scoped_summarize_does_not_scan_a_cwd_outside_its_root(
    tmp_path, monkeypatch,
):
    """A CLAUDE.md sitting in an unrelated cwd must not become a candidate."""
    scoped_root = tmp_path / "scoped"
    (scoped_root / "projects").mkdir(parents=True)
    outsider = tmp_path / "someone-elses-repo"
    outsider.mkdir()
    # Long enough prose to clear the candidate thresholds if it is ever read.
    (outsider / "CLAUDE.md").write_text(
        "# Project notes\n\n" + ("This is a long prose paragraph. " * 300)
    )
    monkeypatch.chdir(outsider)

    ctx = _ctx(_Config(db_explicit=True, projects_root=str(scoped_root / "projects")))
    summarize_analyzer.run(ctx)

    paths = _summarize_candidate_paths(ctx)
    assert all(str(outsider) not in p for p in paths), (
        f"scan escaped its root into the cwd: {paths}"
    )


def test_an_unscoped_summarize_still_scans_the_cwd(tmp_path, monkeypatch):
    """The default path is the one that must not change.

    With no `--projects-root` and no `--db`, project discovery is rooted at
    the cwd exactly as it always was — the fix confines the scan only once a
    boundary has actually been drawn.
    """
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    project = tmp_path / "home" / "work" / "repo"
    project.mkdir(parents=True)
    (project / "CLAUDE.md").write_text(
        "# Project notes\n\n" + ("This is a long prose paragraph. " * 300)
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)

    ctx = _ctx(_Config())
    summarize_analyzer.run(ctx)

    paths = _summarize_candidate_paths(ctx)
    assert any(str(project) in p for p in paths), (
        f"default run stopped scanning the cwd: {paths}"
    )


def test_a_scoped_summarize_keeps_scanning_a_cwd_inside_its_root(
    tmp_path, monkeypatch,
):
    """Confinement, not blanket suppression: a cwd within scope is still scanned."""
    scoped_home = tmp_path / "scoped"
    (scoped_home / ".claude" / "projects").mkdir(parents=True)
    project = scoped_home / "work" / "repo"
    project.mkdir(parents=True)
    (project / "CLAUDE.md").write_text(
        "# Project notes\n\n" + ("This is a long prose paragraph. " * 300)
    )
    monkeypatch.chdir(project)

    ctx = _ctx(_Config(
        db_explicit=True,
        projects_root=str(scoped_home / ".claude" / "projects"),
    ))
    summarize_analyzer.run(ctx)

    paths = _summarize_candidate_paths(ctx)
    assert any(str(project) in p for p in paths), (
        f"in-scope cwd was not scanned: {paths}"
    )


# --- summarize: scope boundary vs session-derived population ----------------
#
# The summarize analyzer derives its project population from the working
# directories the analysed sessions recorded, and confines that population with
# the resolved scope. Two scoping systems that quietly disagree is the failure
# these pin: a boundary applied when none was drawn silently collapses the
# population back to one directory, and a boundary NOT applied when one was
# drawn lets the scan reach past the root the caller asked for. Both produce a
# figure that looks right.

def _scope(source, home):
    from tokenjam.core.optimize.scope import AnalyzerScope
    return AnalyzerScope(
        projects_root=home / ".claude" / "projects",
        claude_home=home / ".claude",
        home=home,
        enabled=True,
        source=source,
    )


def test_scan_boundary_is_none_when_no_root_was_drawn(tmp_path):
    """An unscoped run draws no boundary.

    The default projects root holds no source project, so filtering the
    observed population against it would empty the scan rather than confine
    it — restoring the single-directory population the enumeration replaced.
    """
    from tokenjam.core.optimize.analyzers.summarize import _scan_boundary

    assert _scan_boundary(_scope("default", tmp_path)) is None


def test_scan_boundary_is_the_scope_home_when_a_root_was_drawn(tmp_path):
    from tokenjam.core.optimize.analyzers.summarize import _scan_boundary

    for source in ("flag", "env"):
        assert _scan_boundary(_scope(source, tmp_path)) == tmp_path


def test_population_basis_distinguishes_confined_from_unobserved():
    """"The scope excluded every recorded root" and "no session recorded one"
    are different facts, and the basis may not print the second for the first
    (root anti-pattern 22)."""
    from tokenjam.core.optimize.analyzers.summarize import _population_basis
    from tokenjam.core.summarize.repo_roots import ResolvedRoots

    unobserved = _population_basis(ResolvedRoots(), 0)
    confined = _population_basis(ResolvedRoots(recorded=3, out_of_scope=3), 0)

    assert "No session in this window recorded" in unobserved
    assert "No session in this window recorded" not in confined
    assert "outside the filesystem scope" in confined


def test_population_basis_discloses_a_partial_scope_exclusion():
    from tokenjam.core.optimize.analyzers.summarize import _population_basis
    from tokenjam.core.summarize.repo_roots import ResolvedRoots

    basis = _population_basis(
        ResolvedRoots(roots=(), recorded=5, out_of_scope=2), scanned=3,
    )

    assert "3 project root(s)" in basis
    assert "2 recorded root(s) fall outside the filesystem scope" in basis
