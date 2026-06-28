"""Integration tests for `tj summarize list` via CliRunner."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import TjConfig
from tokenjam.core.summarize import candidates
from tokenjam.core.summarize.catalog import Catalog


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def iso_catalog(monkeypatch):
    """No real globals — controlled catalog so output is deterministic."""
    fake = Catalog(project_files=frozenset({"CLAUDE.md", "AGENTS.md"}),
                   project_globs=(), global_paths=(), forbidden_roots=())
    monkeypatch.setattr(candidates, "load_catalog", lambda: fake)


def _invoke(runner, args, open_db_side_effect=None):
    with patch("tokenjam.cli.main.load_config", return_value=TjConfig(version="1")), \
         patch("tokenjam.cli.main.open_db", side_effect=open_db_side_effect) as open_db:
        result = runner.invoke(cli, args)
    return result, open_db


def _invoke_cfg(runner, args, config, inp=None):
    """Invoke with a specific config and a poisoned open_db (summarize must never touch it)."""
    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db",
               side_effect=AssertionError("summarize must not open the DB")):
        return runner.invoke(cli, args, input=inp)


def _tmp_storage_config(tmp_path):
    from tokenjam.core.config import StorageConfig
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def test_summarize_never_opens_db(runner, tmp_path):
    (tmp_path / "CLAUDE.md").write_text("instructions " * 200)
    result, open_db = _invoke(
        runner, ["summarize", "list", str(tmp_path)],
        open_db_side_effect=AssertionError("summarize must not open the DB"),
    )
    assert result.exit_code == 0, result.output
    open_db.assert_not_called()                       # pins the no_db_commands wiring


def test_list_divides_requested_from_global(runner, tmp_path, monkeypatch):
    """`list` prints the scanned location first, then a divider, then catalog globals."""
    gfile = tmp_path / "ghome" / ".claude" / "CLAUDE.md"
    gfile.parent.mkdir(parents=True)
    gfile.write_text("global instructions " * 200)
    fake = Catalog(project_files=frozenset({"CLAUDE.md"}), project_globs=(),
                   global_paths=(str(gfile),), forbidden_roots=())
    monkeypatch.setattr(candidates, "load_catalog", lambda: fake)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "notes.md").write_text("plain project doc " * 200)        # requested, kind=other
    res = _invoke_cfg(runner, ["summarize", "list", str(proj), "--ext", "md"],
                      _tmp_storage_config(tmp_path))
    assert res.exit_code == 0, res.output
    assert "global / catalog" in res.output                           # the divider line is present
    # requested doc (kind 'other') sits above the divider; the global prompt below it
    assert res.output.index("other") < res.output.index("global / catalog") < res.output.index("prompt")


def test_list_json_shape(runner, tmp_path):
    (tmp_path / "CLAUDE.md").write_text("instructions " * 200)
    result, _ = _invoke(runner, ["summarize", "list", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert {"candidates", "count", "root", "note"} <= data.keys()
    assert any(Path(c["path"]).name == "CLAUDE.md" for c in data["candidates"])
    assert any(c["kind"] == "prompt" for c in data["candidates"])


def test_json_note_carries_caveat(runner, tmp_path):
    (tmp_path / "CLAUDE.md").write_text("instructions " * 200)
    result, _ = _invoke(runner, ["summarize", "list", str(tmp_path), "--json"])
    data = json.loads(result.output)
    assert "review the summary before adopting" in data["note"]   # honesty discipline (Rule 14)


def test_repo_recursive_mutually_exclusive(runner):
    result, _ = _invoke(runner, ["summarize", "list", "--repo", "--recursive"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
