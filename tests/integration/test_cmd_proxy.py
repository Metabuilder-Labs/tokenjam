"""Lifecycle tests for `tj proxy` (#219) — enable/disable/status/killswitch.

`proxy` is in `no_db_commands`, so these patch `open_db` with an
`AssertionError` side effect to prove the command never opens the DB. Config and
the Claude Code settings env are written under a tmp HOME so nothing touches the
developer's real files.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli
from tokenjam.core.config import TjConfig, load_config


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated HOME + a project config path the proxy commands write to."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    cfg_path = tmp_path / "tj.toml"
    cfg_path.write_text('version = "1"\n')
    return {"home": home, "cfg_path": cfg_path}


def _invoke(runner, cfg_path, args):
    config = load_config(str(cfg_path))
    # proxy is a no_db_command — assert it never opens the DB.
    with patch("tokenjam.cli.main.load_config", return_value=config), \
         patch("tokenjam.cli.main.open_db",
               side_effect=AssertionError("proxy must not open the DB")):
        return runner.invoke(cli, args, obj=None)


def _read_proxy(cfg_path) -> TjConfig:
    return load_config(str(cfg_path)).proxy


def test_enable_sets_config_and_wires_env(runner, env):
    res = _invoke(runner, env["cfg_path"], ["proxy", "enable"])
    assert res.exit_code == 0, res.output
    proxy = _read_proxy(env["cfg_path"])
    assert proxy.enabled is True
    assert proxy.killswitch is False
    # Base-URL env wired into ~/.claude/settings.json.
    settings = json.loads((env["home"] / ".claude" / "settings.json").read_text())
    base = f"http://{proxy.host}:{proxy.port}"
    assert settings["env"]["ANTHROPIC_BASE_URL"] == base
    assert settings["env"]["OPENAI_BASE_URL"] == base


def test_disable_clears_config_and_unwires(runner, env):
    _invoke(runner, env["cfg_path"], ["proxy", "enable"])
    res = _invoke(runner, env["cfg_path"], ["proxy", "disable"])
    assert res.exit_code == 0, res.output
    assert _read_proxy(env["cfg_path"]).enabled is False
    settings = json.loads((env["home"] / ".claude" / "settings.json").read_text())
    assert "ANTHROPIC_BASE_URL" not in settings.get("env", {})
    assert "OPENAI_BASE_URL" not in settings.get("env", {})


def test_disable_leaves_user_custom_base_url_untouched(runner, env):
    # A user's own ANTHROPIC_BASE_URL (not our proxy) must survive disable.
    claude = env["home"] / ".claude"
    claude.mkdir(parents=True)
    (claude / "settings.json").write_text(json.dumps(
        {"env": {"ANTHROPIC_BASE_URL": "https://my-gateway.example"}}))
    _invoke(runner, env["cfg_path"], ["proxy", "disable"])
    settings = json.loads((claude / "settings.json").read_text())
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "https://my-gateway.example"


def test_killswitch_engage_and_release(runner, env):
    _invoke(runner, env["cfg_path"], ["proxy", "enable"])
    _invoke(runner, env["cfg_path"], ["proxy", "killswitch"])
    assert _read_proxy(env["cfg_path"]).killswitch is True
    _invoke(runner, env["cfg_path"], ["proxy", "killswitch", "--off"])
    assert _read_proxy(env["cfg_path"]).killswitch is False


def test_status_json_reports_state(runner, env):
    _invoke(runner, env["cfg_path"], ["proxy", "enable"])
    res = _invoke(runner, env["cfg_path"], ["--json", "proxy", "status"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["enabled"] is True
    assert payload["port"] == 7392
    assert payload["mode"] == "suggest"
    assert "ANTHROPIC_BASE_URL" in payload["wiring"]
    assert payload["orphaned_wiring"] == []


def test_doctor_flags_orphaned_wiring(runner, env, tmp_path):
    # Enable (wires env), then disable ONLY the config flag by hand so the env
    # wiring is left orphaned — doctor must flag it.
    _invoke(runner, env["cfg_path"], ["proxy", "enable"])
    # Re-disable just the config (simulating a hand-edit), keep the env wiring.
    from tokenjam.core.config import write_config
    cfg = load_config(str(env["cfg_path"]))
    cfg.proxy.enabled = False
    write_config(cfg, env["cfg_path"])

    from tokenjam.proxy.wiring import find_orphaned_wiring
    reloaded = load_config(str(env["cfg_path"]))
    assert find_orphaned_wiring(reloaded) == ["ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"]
