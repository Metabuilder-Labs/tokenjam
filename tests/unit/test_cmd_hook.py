"""CLI-level tests for `tj hook cap-output` — the PostToolUse entrypoint.

Uses Click's CliRunner and patches `load_config` (the canonical pattern from
tests/integration/test_cli.py). Critically also asserts the hook NEVER opens the
DB (it's in `no_db_commands`, #61 lock-safety): `open_db` is patched to raise.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import TjConfig
from tokenjam.core.savings_log import read_savings


@pytest.fixture
def runner():
    return CliRunner()


def _config(tmp_path) -> TjConfig:
    cfg = TjConfig(version="1")
    cfg.storage.path = str(tmp_path / "tj.duckdb")  # sink derives from this parent
    # The hook is DEFAULT-OFF opt-in (A/B gate failed — see CapOutputConfig).
    # These tests exercise the trim MECHANISM, so opt in explicitly; the
    # default-off policy itself is covered by test_config.py and the
    # disabled/passthrough tests below.
    cfg.hooks.output_cap.enabled = True
    return cfg


def _invoke(runner, cfg, stdin: str):
    # open_db raising proves the hook path never touches the DB.
    with patch("tokenjam.cli.main.load_config", return_value=cfg), \
         patch("tokenjam.cli.main.open_db",
               side_effect=AssertionError("hook must not open the DB")):
        return runner.invoke(cli, ["hook", "cap-output"], input=stdin)


def _bash_event(stdout: str, command="seq 1 600") -> str:
    return json.dumps({
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "tool_response": {
            "stdout": stdout, "stderr": "", "interrupted": False,
            "isImage": False, "noOutputExpected": False,
        },
        "session_id": "sess-1",
    })


def _big(nlines: int) -> str:
    return "\n".join(f"line {i:04d} " + "x" * 60 for i in range(nlines))


def test_big_bash_output_is_trimmed_with_shape_preserved(runner, tmp_path):
    cfg = _config(tmp_path)
    res = _invoke(runner, cfg, _bash_event(_big(600)))
    assert res.exit_code == 0
    out = json.loads(res.stdout)
    hs = out["hookSpecificOutput"]
    assert hs["hookEventName"] == "PostToolUse"
    updated = hs["updatedToolOutput"]
    # shape preserved — all original keys still present
    assert set(updated.keys()) == {
        "stdout", "stderr", "interrupted", "isImage", "noOutputExpected"
    }
    assert "tj cap-output" in updated["stdout"]
    assert len(updated["stdout"]) < len(_big(600))
    # first & last lines survive
    assert "line 0000" in updated["stdout"]
    assert "line 0599" in updated["stdout"]


def test_small_output_passes_through_no_stdout(runner, tmp_path):
    cfg = _config(tmp_path)
    res = _invoke(runner, cfg, _bash_event("hi\nthere", command="echo hi"))
    assert res.exit_code == 0
    assert res.stdout.strip() == ""


def test_malformed_stdin_fails_open(runner, tmp_path):
    cfg = _config(tmp_path)
    res = _invoke(runner, cfg, "not json {{{")
    assert res.exit_code == 0
    assert res.stdout.strip() == ""


def test_empty_stdin_fails_open(runner, tmp_path):
    cfg = _config(tmp_path)
    res = _invoke(runner, cfg, "")
    assert res.exit_code == 0
    assert res.stdout.strip() == ""


def test_ineligible_tool_passes_through(runner, tmp_path):
    cfg = _config(tmp_path)
    event = json.dumps({
        "tool_name": "Read",
        "tool_input": {}, "tool_response": {"stdout": _big(600)},
        "session_id": "s",
    })
    res = _invoke(runner, cfg, event)
    assert res.exit_code == 0
    assert res.stdout.strip() == ""


def test_disabled_config_passes_through(runner, tmp_path):
    cfg = _config(tmp_path)
    cfg.hooks.output_cap.enabled = False
    res = _invoke(runner, cfg, _bash_event(_big(600)))
    assert res.exit_code == 0
    assert res.stdout.strip() == ""


def test_killswitch_passes_through(runner, tmp_path):
    cfg = _config(tmp_path)
    cfg.hooks.output_cap.killswitch = True
    res = _invoke(runner, cfg, _bash_event(_big(600)))
    assert res.exit_code == 0
    assert res.stdout.strip() == ""


def test_trim_records_savings_event(runner, tmp_path):
    cfg = _config(tmp_path)
    res = _invoke(runner, cfg, _bash_event(_big(600)))
    assert res.exit_code == 0
    events = read_savings(cfg)
    assert len(events) == 1
    ev = events[0]
    assert ev["tool"] == "Bash"
    assert ev["session_id"] == "sess-1"
    assert ev["saved_tok_est"] > 0
    assert "ts" in ev


def test_string_tool_response_stays_string(runner, tmp_path):
    # A tool whose tool_response is a plain string → updatedToolOutput is a string.
    cfg = _config(tmp_path)
    event = json.dumps({
        "tool_name": "WebFetch",
        "tool_input": {}, "tool_response": _big(600),
        "session_id": "s",
    })
    res = _invoke(runner, cfg, event)
    assert res.exit_code == 0
    out = json.loads(res.stdout)
    updated = out["hookSpecificOutput"]["updatedToolOutput"]
    assert isinstance(updated, str)
    assert "tj cap-output" in updated
