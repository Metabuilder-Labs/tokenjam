"""Onboard config-writer hygiene (issues #5, #15)."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from tokenjam.cli.cmd_onboard import cmd_onboard


@pytest.fixture(autouse=True)
def _no_existing_config(monkeypatch):
    monkeypatch.setattr("tokenjam.cli.cmd_onboard.find_config_file", lambda: None)


def _run(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(cmd_onboard, ["--no-daemon", "--budget", "0"], obj={})
        cfg = Path(".tj/config.toml")
        return res, (cfg.read_text() if cfg.exists() else "")


def test_capture_block_has_all_four_toggles(tmp_path):
    # #15: onboard previously omitted tool_inputs (CLAUDE.md documents four).
    res, cfg = _run(tmp_path)
    assert res.exit_code == 0, res.output
    for key in ("prompts", "completions", "tool_inputs", "tool_outputs"):
        assert f"{key} = false" in cfg, f"[capture] missing {key}"


def test_no_stale_openclawwatch_url(tmp_path):
    # #5: stale Metabuilder-Labs/openclawwatch URL in config header + output.
    res, cfg = _run(tmp_path)
    assert "openclawwatch" not in cfg
    assert "openclawwatch" not in res.output
    assert "github.com/Metabuilder-Labs/tokenjam" in cfg
    assert "github.com/Metabuilder-Labs/tokenjam" in res.output
