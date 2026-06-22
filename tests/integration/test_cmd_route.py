"""Integration tests for the `tj route` CLI surface (CliRunner)."""
from __future__ import annotations

import json
from datetime import timedelta
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_tool_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config():
    return TjConfig(version="1")


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _exports_in_tmp(monkeypatch, tmp_path):
    """Redirect ~/.config/tokenjam/exports to a tmp dir so tests never touch
    the real home directory."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return tmp_path


def _seed_downsize_candidate(db, session_id="s-small"):
    """One small Opus session + 2 tool spans — matches the downsize heuristic,
    so build_report yields a downgrade finding with a suggestion."""
    start = utcnow() - timedelta(days=2)
    llm = make_llm_span(
        agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
        input_tokens=1000, output_tokens=200, cost_usd=0.030,
        session_id=session_id, start_time=start,
    )
    db.insert_span(llm)
    for _ in range(2):
        tool = make_tool_span(agent_id="claude-code-x", tool_name="Read",
                              trace_id=llm.trace_id)
        tool.session_id = session_id
        tool.start_time = start
        db.insert_span(tool)


def _invoke(runner, db, config, args):
    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db", return_value=db):
        return runner.invoke(cli, args)


# --- export writes a valid config with honesty comments --------------------

@pytest.mark.parametrize("target,ext", [("ccr", "jsonc"), ("litellm", "yaml")])
def test_export_writes_config_with_caveat(runner, db, config, _exports_in_tmp, target, ext):
    _seed_downsize_candidate(db)
    result = _invoke(runner, db, config, ["route", "export", "--target", target])
    assert result.exit_code == 0, result.output

    exports = _exports_in_tmp / ".config" / "tokenjam" / "exports"
    written = list(exports.glob(f"{target}-*.{ext}"))
    assert len(written) == 1, f"expected one {target} export, got {written}"
    body = written[0].read_text()
    # honesty strings embedded
    assert "Candidate-flagging heuristic" in body          # MODEL_DOWNGRADE_CAVEAT
    assert "evidence: L1" in body
    assert "Derivation window:" in body
    assert "Derived at:" in body
    # the suggestion made it in
    assert "claude-haiku-4-5" in body


def test_export_only_writes_to_exports_dir(runner, db, config, _exports_in_tmp):
    """Nothing outside ~/.config/tokenjam/exports/ is created."""
    _seed_downsize_candidate(db)
    _invoke(runner, db, config, ["route", "export", "--target", "litellm"])
    # The only thing under the fake home is the exports tree.
    created = [p for p in _exports_in_tmp.rglob("*") if p.is_file()]
    assert created, "expected an export file"
    for p in created:
        assert ".config/tokenjam/exports" in str(p), p


def test_export_json_output(runner, db, config):
    _seed_downsize_candidate(db)
    result = _invoke(runner, db, config,
                     ["route", "export", "--target", "ccr", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["target"] == "ccr"
    assert payload["rule_count"] == 1
    assert payload["path"].endswith(".jsonc")


def test_export_requires_target_or_check(runner, db, config):
    _seed_downsize_candidate(db)
    result = _invoke(runner, db, config, ["route", "export"])
    assert result.exit_code != 0
    assert "--target" in result.output


# --- --check staleness ------------------------------------------------------

def test_check_reports_no_export_then_current_then_stale(runner, db, config):
    _seed_downsize_candidate(db)

    # 1. No export yet.
    r1 = _invoke(runner, db, config,
                 ["route", "export", "--check", "--target", "ccr", "--json"])
    assert r1.exit_code == 0, r1.output
    assert json.loads(r1.output)["results"][0]["status"] == "no_export"

    # 2. Export, then check → current.
    _invoke(runner, db, config, ["route", "export", "--target", "ccr"])
    r2 = _invoke(runner, db, config,
                 ["route", "export", "--check", "--target", "ccr", "--json"])
    assert json.loads(r2.output)["results"][0]["status"] == "current"

    # 3. Findings change (add a second distinct candidate model) → stale.
    start = utcnow() - timedelta(days=1)
    llm = make_llm_span(
        agent_id="claude-code-x", model="claude-sonnet-4-5", provider="anthropic",
        input_tokens=900, output_tokens=150, cost_usd=0.02,
        session_id="s-small-2", start_time=start,
    )
    db.insert_span(llm)
    for _ in range(2):
        t = make_tool_span(agent_id="claude-code-x", tool_name="Read",
                           trace_id=llm.trace_id)
        t.session_id = "s-small-2"
        t.start_time = start
        db.insert_span(t)
    r3 = _invoke(runner, db, config,
                 ["route", "export", "--check", "--target", "ccr", "--json"])
    assert json.loads(r3.output)["results"][0]["status"] == "stale"


# --- the existing claude_code alias must still work ------------------------

def test_optimize_export_config_still_works(runner, db, config, _exports_in_tmp):
    _seed_downsize_candidate(db)
    result = _invoke(runner, db, config,
                     ["optimize", "--export-config", "claude-code"])
    assert result.exit_code == 0, result.output
    exports = _exports_in_tmp / ".config" / "tokenjam" / "exports"
    written = list(exports.glob("claude-code-*.jsonc"))
    assert len(written) == 1, written
    assert "routing_recommendations" in written[0].read_text()
