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


# --- prep / check (mechanism via the CLI; no DB, no writes) ---

def _prep_and_markers(runner, config, f):
    data = json.loads(_invoke_cfg(runner, ["summarize", "prep", str(f), "--json"], config).output)
    import re
    markers = re.findall(r'<tj-keep id="\d+"[^>]*?(?:/>|>.*?</tj-keep>)', data["wrapped_prompt"], re.DOTALL)
    return data, "Act carefully; never skip a step. " + " ".join(markers)


def test_prep_then_check_roundtrip(runner, tmp_path):
    config = _tmp_storage_config(tmp_path)
    f = tmp_path / "CLAUDE.md"
    f.write_text("Always act carefully and never skip a required step. " * 30 + "\n```\nx = 1\n```\n")

    data, summary = _prep_and_markers(runner, config, f)
    assert data["source_sha256"] and "<tj-keep" in data["wrapped_prompt"]

    chk = _invoke_cfg(
        runner,
        ["summarize", "check", str(f), "--summary", "-", "--prepped-hash", data["source_sha256"], "--json"],
        config, inp=summary,
    )
    assert chk.exit_code == 0, chk.output
    verdict = json.loads(chk.output)
    assert verdict["structure_ok"] is True and verdict["staged"] is True
    assert "x = 1" in verdict["restored"]


def test_prep_below_gate_note(runner, tmp_path):
    config = _tmp_storage_config(tmp_path)
    f = tmp_path / "tiny.md"
    f.write_text("short prompt with only a few words")
    res = _invoke_cfg(runner, ["summarize", "prep", str(f), "--json"], config)
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["wrapped_prompt"] == "" and "gate" in data["note"]


def test_check_human_output_shows_structure(runner, tmp_path):
    config = _tmp_storage_config(tmp_path)
    f = tmp_path / "CLAUDE.md"
    f.write_text("Always act carefully and never skip a required step. " * 30 + "\n```\nx = 1\n```\n")
    data, summary = _prep_and_markers(runner, config, f)
    chk = _invoke_cfg(
        runner,
        ["summarize", "check", str(f), "--summary", "-", "--prepped-hash", data["source_sha256"]],
        config, inp=summary,
    )
    assert chk.exit_code == 0, chk.output
    assert "structure preserved" in chk.output and "staged for review" in chk.output


def test_check_refuses_changed_file(runner, tmp_path):
    config = _tmp_storage_config(tmp_path)
    f = tmp_path / "CLAUDE.md"
    f.write_text("Always act carefully and never skip a required step. " * 30)
    data, _ = _prep_and_markers(runner, config, f)
    f.write_text("edited after prep")                 # changed since prep → house-voice refuse
    chk = _invoke_cfg(
        runner,
        ["summarize", "check", str(f), "--summary", "-", "--prepped-hash", data["source_sha256"]],
        config, inp="anything",
    )
    assert chk.exit_code != 0
    assert "changed since" in chk.output


def test_prep_human_emits_wrapped_prompt_and_rules(runner, tmp_path):
    """Manual/copy path: bare `prep` (human) prints the rewrite rules + wrapped prompt to copy —
    not just metadata — so the workflow is usable without --json."""
    config = _tmp_storage_config(tmp_path)
    f = tmp_path / "CLAUDE.md"
    f.write_text("Always act carefully and never skip a required step. " * 30 + "\n```\nx = 1\n```\n")
    res = _invoke_cfg(runner, ["summarize", "prep", str(f)], config)
    assert res.exit_code == 0, res.output
    assert "<tj-keep" in res.output                    # the wrapped prompt (the payload to copy)
    assert "compress AI system prompts" in res.output  # the rewrite rules (WRAP_SUMM_SYS)
    assert "summarize check" in res.output             # the next-step hint
