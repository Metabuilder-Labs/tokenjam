"""Non-interactive onboarding contract (#86).

`tj onboard` is an interactive wizard by default, but every prompt has a flag
that supplies the answer instead — `--plan`, `--budget`, `--project`
(`--claude-code` only), `--no-daemon`. These tests pin that contract for each
persona: with all prompts pre-answered via flags and no TTY attached, onboard
must run to completion with zero interactive prompts, exit 0, and produce the
expected config/settings writes. A companion idempotency test pins that
re-running with the same flags doesn't duplicate state or reintroduce a
prompt.

See docs/ci-setup.md for the user-facing version of this contract and
docs/internal/wizard-contract.md for the broader wizard etiquette these
tests help guard.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolate_real_world_side_effects(tmp_path_factory, monkeypatch):
    """Keep onboard off real user/daemon state (mirrors the fixture in
    tests/integration/test_cli.py — see its docstring for the full rationale:
    without this, onboard reaches out to ~/.claude/projects, shells out to the
    real `claude` CLI, and pokes the real `tj serve` daemon lock)."""
    iso = tmp_path_factory.mktemp("onboard-ci-iso")
    monkeypatch.setattr(
        "tokenjam.core.backfill.CLAUDE_CODE_PROJECTS_ROOT",
        iso / "no-such-claude-projects",
        raising=False,
    )
    monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, b"", b""),
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._stop_serve_for_db_write", lambda: False
    )
    monkeypatch.setenv("HOME", str(iso))


def _refuse_prompts():
    """Patch click.prompt/click.confirm in cmd_onboard to blow up if called.

    A "zero interactive prompts" assertion is only meaningful if the test
    would actually fail when a prompt is attempted — silently supplying
    `input=` (like the existing onboard tests do) can't tell a skipped prompt
    from an answered one. This makes any un-skipped prompt a hard test
    failure instead.
    """
    return (
        patch(
            "tokenjam.cli.cmd_onboard.click.prompt",
            side_effect=AssertionError("interactive prompt attempted: click.prompt"),
        ),
        patch(
            "tokenjam.cli.cmd_onboard.click.confirm",
            side_effect=AssertionError("interactive prompt attempted: click.confirm"),
        ),
    )


def test_onboard_bare_sdk_noninteractive_zero_prompts(runner, tmp_path):
    """Bare `tj onboard` with --plan/--budget/--no-daemon prompts for nothing."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    p1, p2 = _refuse_prompts()

    with runner.isolated_filesystem() as cwd, \
         patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         p1, p2:
        result = runner.invoke(cli, [
            "onboard", "--plan", "api", "--budget", "5.0", "--no-daemon",
        ])
        assert result.exit_code == 0, result.output

        config_path = _as_path(cwd) / ".tj" / "config.toml"
        assert config_path.exists()
        text = config_path.read_text()
        assert 'plan = "api"' in text
        assert "daily_usd = 5.0" in text


def test_onboard_claude_code_noninteractive_zero_prompts(runner, tmp_path):
    """--claude-code with --plan/--budget/--project/--no-daemon prompts for nothing
    and writes the global config + settings.json wiring."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    p1, p2 = _refuse_prompts()

    with runner.isolated_filesystem(), \
         patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         p1, p2:
        result = runner.invoke(cli, [
            "onboard", "--claude-code", "--no-daemon",
            "--plan", "max_20x", "--budget", "5.0", "--project", "aquanode",
        ])
        assert result.exit_code == 0, result.output

    from tokenjam.core.config import load_config
    cfg = load_config(str(fake_home / ".config" / "tj" / "config.toml"))
    agent_id = next(k for k in cfg.agents if k.startswith("claude-code-"))
    assert cfg.agents[agent_id].project == "aquanode"
    assert cfg.agents[agent_id].budget.daily_usd == 5.0
    assert cfg.budgets["anthropic"].plan == "max_20x"

    settings = json.loads((fake_home / ".claude" / "settings.json").read_text())
    env = settings["env"]
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" in env
    assert settings["statusLine"]["command"] == "tj statusline"


def test_onboard_codex_noninteractive_zero_prompts(runner, tmp_path):
    """--codex with --plan/--budget/--no-daemon prompts for nothing and writes
    the [otel] block to ~/.codex/config.toml (no --project for this persona —
    Codex hardcodes service.name=codex_exec so onboarding is project-agnostic)."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    p1, p2 = _refuse_prompts()

    with runner.isolated_filesystem(), \
         patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
         p1, p2:
        result = runner.invoke(cli, [
            "onboard", "--codex", "--no-daemon",
            "--plan", "enterprise", "--budget", "5.0",
        ])
        assert result.exit_code == 0, result.output

    from tokenjam.core.config import load_config
    cfg = load_config(str(fake_home / ".config" / "tj" / "config.toml"))
    assert cfg.budgets["openai"].plan == "enterprise"
    assert cfg.agents["codex_exec"].budget.daily_usd == 5.0

    codex_toml = (fake_home / ".codex" / "config.toml").read_text()
    assert "[otel]" in codex_toml


def test_onboard_claude_code_noninteractive_rerun_is_idempotent(runner, tmp_path):
    """Re-running --claude-code with the same flags is a no-op wrt duplication:
    no second agent entry, no duplicated env keys, no duplicated ~/.zshrc
    harness block — and still zero prompts on the second run."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    args = [
        "onboard", "--claude-code", "--no-daemon",
        "--plan", "max_20x", "--budget", "5.0", "--project", "aquanode",
    ]

    # Same cwd across both invocations — the (unlabeled) agent_id is derived
    # from the repo/directory name, not from --project (which only sets the
    # dashboard-grouping `project` field on that agent). A fresh
    # isolated_filesystem() per iteration would derive a different agent_id
    # each time and defeat the duplication check below.
    with runner.isolated_filesystem():
        for _ in range(2):
            p1, p2 = _refuse_prompts()
            with patch("tokenjam.cli.cmd_onboard.find_config_file", return_value=None), \
                 patch("tokenjam.cli.cmd_onboard.Path.home", return_value=fake_home), \
                 p1, p2:
                result = runner.invoke(cli, args)
                assert result.exit_code == 0, result.output

    from tokenjam.core.config import load_config
    cfg = load_config(str(fake_home / ".config" / "tj" / "config.toml"))
    cc_agents = [k for k in cfg.agents if k.startswith("claude-code-")]
    assert len(cc_agents) == 1, f"expected exactly one agent, got {cc_agents}"

    zshrc_text = (fake_home / ".zshrc").read_text()
    assert zshrc_text.count("# tj harness observability") == 1

    # The rerun rotates nothing (same flags), so the auth header must still be
    # a single "Authorization=Bearer <secret>" value, not something the
    # read-merge-write accidentally concatenated across two writes.
    settings = json.loads((fake_home / ".claude" / "settings.json").read_text())
    headers = settings["env"]["OTEL_EXPORTER_OTLP_HEADERS"]
    assert headers.count("Authorization=Bearer") == 1


def _as_path(cwd):
    """`isolated_filesystem()` yields a str cwd on some click versions and a
    Path on others; normalize once here instead of at every call site."""
    from pathlib import Path
    return Path(cwd)
