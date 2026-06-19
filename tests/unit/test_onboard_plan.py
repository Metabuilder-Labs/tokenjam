"""Plain `tj onboard` honors --plan / writes the plan tier (issue #4)."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from tokenjam.cli.cmd_onboard import cmd_onboard


@pytest.fixture(autouse=True)
def _no_existing_config(monkeypatch):
    # Force the fresh-config path regardless of any ~/.config/tj on the machine.
    monkeypatch.setattr("tokenjam.cli.cmd_onboard.find_config_file", lambda: None)


def _onboard(tmp_path, *args):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(cmd_onboard, ["--no-daemon", "--budget", "0", *args], obj={})
        cfg = Path(".tj/config.toml")
        return res, (cfg.read_text() if cfg.exists() else "")


def test_plain_onboard_writes_anthropic_plan(tmp_path):
    res, cfg = _onboard(tmp_path, "--plan", "api")
    assert res.exit_code == 0, res.output
    assert "[budget.anthropic]" in cfg
    assert 'plan = "api"' in cfg


def test_plain_onboard_subscription_tier(tmp_path):
    res, cfg = _onboard(tmp_path, "--plan", "max_5x")
    assert res.exit_code == 0, res.output
    assert "[budget.anthropic]" in cfg
    assert 'plan = "max_5x"' in cfg


def test_plain_onboard_openai_tier_routes_to_openai(tmp_path):
    res, cfg = _onboard(tmp_path, "--plan", "plus")
    assert res.exit_code == 0, res.output
    assert "[budget.openai]" in cfg
    assert 'plan = "plus"' in cfg


def test_plain_onboard_no_plan_noninteractive_writes_no_plan(tmp_path):
    # No --plan + no TTY (CliRunner) → no prompt, no plan section (no hang, no
    # presumptuous default).
    res, cfg = _onboard(tmp_path)
    assert res.exit_code == 0, res.output
    assert "[budget." not in cfg
    assert "plan =" not in cfg


def test_plan_message_shows_section_name(tmp_path):
    # Regression for #157: the "written to [budget.<provider>]" confirmation must
    # NOT have its TOML section header eaten by Rich markup parsing (which left
    # the message reading "(written to )").
    res, _ = _onboard(tmp_path, "--plan", "api")
    assert res.exit_code == 0, res.output
    assert "[budget.anthropic]" in res.output, res.output
    assert "(written to )" not in res.output


def test_plain_onboard_writes_valid_toml(tmp_path):
    import sys
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    res, cfg = _onboard(tmp_path, "--plan", "max_20x")
    assert res.exit_code == 0, res.output
    parsed = tomllib.loads(cfg)
    assert parsed["budget"]["anthropic"]["plan"] == "max_20x"
