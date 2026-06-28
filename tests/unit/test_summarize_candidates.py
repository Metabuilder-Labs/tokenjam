"""Unit tests for the summarize scan — net, ranking, repo detection, boundary safety.

Every test runs against an isolated catalog (a tmp global file, controlled
project_files) so it never depends on the developer's real ~/.claude.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tokenjam.core.summarize import candidates
from tokenjam.core.summarize.catalog import Catalog


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Controlled catalog: one tmp global prompt; project files CLAUDE.md/AGENTS.md."""
    gfile = tmp_path / "globalhome" / ".claude" / "CLAUDE.md"
    gfile.parent.mkdir(parents=True)
    gfile.write_text("global instructions " * 200)
    fake = Catalog(
        project_files=frozenset({"CLAUDE.md", "AGENTS.md"}),
        project_globs=(".claude/agents/*.md",),
        global_paths=(str(gfile),),
        forbidden_roots=(),
    )
    monkeypatch.setattr(candidates, "load_catalog", lambda: fake)
    return {"global_file": str(gfile)}


def test_is_boundary_predicate():
    home = Path("/home/alice")
    assert candidates._is_boundary(Path("/"), home) is True
    assert candidates._is_boundary(Path("/etc"), home) is True
    assert candidates._is_boundary(Path("/opt"), home) is True
    assert candidates._is_boundary(home, home) is True
    assert candidates._is_boundary(Path("/opt/foo"), home) is False
    assert candidates._is_boundary(Path("/home/alice/code"), home) is False


def test_find_repo_root_nearest_and_boundary(tmp_path, monkeypatch, iso):
    (tmp_path / "outer" / ".git").mkdir(parents=True)
    inner = tmp_path / "outer" / "inner"
    (inner / ".git").mkdir(parents=True)
    deep = inner / "a" / "b"
    deep.mkdir(parents=True)
    assert candidates.find_repo_root(deep) == inner.resolve()      # nearest .git wins

    monkeypatch.setattr(Path, "home", lambda: tmp_path)            # treat tmp_path as home
    norepo = tmp_path / "x" / "y"
    norepo.mkdir(parents=True)
    assert candidates.find_repo_root(norepo) is None              # no .git below the home boundary
    (tmp_path / ".git").mkdir()
    assert candidates.find_repo_root(tmp_path / "x") is None      # a .git AT home is refused


def test_default_is_catalog_only(tmp_path, monkeypatch, iso):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "CLAUDE.md").write_text("instructions " * 200)
    (proj / "README.md").write_text("readme prose " * 300)
    monkeypatch.chdir(proj)
    paths = [c.path for c in candidates.list_candidates(config=None).candidates]
    assert str(proj / "CLAUDE.md") in paths
    assert not any("README" in p for p in paths)                 # catalog-only: doc excluded


def test_any_flag_opens_net_and_ext(tmp_path, monkeypatch, iso):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "CLAUDE.md").write_text("x " * 200)
    (proj / "README.md").write_text("y " * 300)
    (proj / "notes.txt").write_text("z " * 300)
    paths = [c.path for c in candidates.list_candidates(str(proj), config=None).candidates]
    assert any("README.md" in p for p in paths)                  # explicit path opens *.md
    assert not any("notes.txt" in p for p in paths)              # .txt not in default net
    paths2 = [c.path for c in
              candidates.list_candidates(str(proj), config=None, extra_exts=("txt",)).candidates]
    assert any("notes.txt" in p for p in paths2)                 # --ext txt includes it


def test_prompts_rank_before_docs(tmp_path, monkeypatch, iso):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "CLAUDE.md").write_text("small prompt " * 110)        # small prompt
    (repo / "README.md").write_text("big doc " * 4000)           # huge doc
    monkeypatch.chdir(repo)
    res = candidates.list_candidates(recursive=True, config=None, include_global=False)
    kinds = [c.is_prompt for c in res.candidates]
    last_prompt = max((i for i, k in enumerate(kinds) if k), default=-1)
    first_other = next((i for i, k in enumerate(kinds) if not k), len(kinds))
    assert last_prompt < first_other                             # every prompt before every doc
    paths = [c.path for c in res.candidates]
    assert paths.index(str(repo / "CLAUDE.md")) < paths.index(str(repo / "README.md"))


def test_scanned_location_before_global(tmp_path, monkeypatch, iso):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "CLAUDE.md").write_text("proj " * 200)
    monkeypatch.chdir(proj)
    paths = [c.path for c in candidates.list_candidates(config=None).candidates]
    gfile = iso["global_file"]
    assert str(proj / "CLAUDE.md") in paths and gfile in paths
    assert paths.index(str(proj / "CLAUDE.md")) < paths.index(gfile)   # project before global


def test_requested_scope_outranks_global_even_as_a_doc(tmp_path, iso):
    """Scope is the primary split: a requested plain doc ranks before a global prompt
    (kind only orders WITHIN a section). Regression for 'the tmp file buried in globals'."""
    f = tmp_path / "notes.md"
    f.write_text("plain project doc " * 200)               # scope=path, kind=other (not a catalog name)
    res = candidates.list_candidates(str(f), config=None)  # explicit file + the iso global (a CLAUDE.md prompt)
    paths = [c.path for c in res.candidates]
    gfile = iso["global_file"]
    assert str(f) in paths and gfile in paths
    assert paths.index(str(f)) < paths.index(gfile)        # requested doc before global prompt


def test_recursive_refuses_without_repo(tmp_path, monkeypatch, iso):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "CLAUDE.md").write_text("x " * 200)                  # must NOT be walked to
    monkeypatch.chdir(work)
    res = candidates.list_candidates(recursive=True, config=None)
    assert "no safe root" in res.note
    assert all(c.scope == "global" for c in res.candidates)      # globals only — nothing walked
    assert not any("work" in c.path for c in res.candidates)


def test_explicit_file_any_name(tmp_path, monkeypatch, iso):
    f = tmp_path / "weird_name.md"
    f.write_text("prompt content " * 200)
    res = candidates.list_candidates(str(f), config=None, include_global=False)
    paths = [c.path for c in res.candidates]
    assert paths == [str(f)]                                      # vouched file, analyzed by-name


def test_min_prose_override(tmp_path, monkeypatch, iso):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "CLAUDE.md").write_text("w " * 150)                  # 150 prose words
    monkeypatch.chdir(proj)
    base = candidates.list_candidates(config=None, include_global=False).candidates
    assert any(c.scope == "project" for c in base)              # passes the default-100 gate
    raised = candidates.list_candidates(config=None, include_global=False,
                                        min_prose_words=200).candidates
    assert not any(c.scope == "project" for c in raised)        # dropped above 150


def test_repo_without_git_notes_and_uses_project_scope(tmp_path, monkeypatch, iso):
    """--repo outside a git repo: emit a note and use 'project' scope, not a pretend 'repo' root."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)        # boundary safety → no repo root found
    work = tmp_path / "work"
    work.mkdir()
    (work / "CLAUDE.md").write_text("instructions " * 200)
    monkeypatch.chdir(work)
    res = candidates.list_candidates(repo=True, config=None, include_global=False)
    assert "no git repo" in res.note
    assert res.candidates and all(c.scope == "project" for c in res.candidates)   # project, never "repo"


def test_explicit_missing_path_notes_not_found(tmp_path, iso):
    """A non-existent explicit PATH surfaces a 'not found' note whose tail matches what's shown."""
    missing = tmp_path / "nope" / "ghost.md"
    # --no-global: nothing is shown, so the note must NOT claim "globals only"
    res = candidates.list_candidates(str(missing), config=None, include_global=False)
    assert "not found" in res.note.lower() and "nothing to show" in res.note
    assert res.candidates == []
    # with globals included, they ARE shown — the note says so
    res2 = candidates.list_candidates(str(missing), config=None, include_global=True)
    assert "not found" in res2.note.lower() and "globals only" in res2.note
    assert res2.candidates                                     # the iso global is present
