"""Prompt capture defaults on (E33): `cache-recommend` and `trim` were dark,
and `reuse` never reached its prompt-prefix mode, for every onboarded user
because `capture.prompts` stayed false even though `capture.tool_inputs` was
already flipped on. Covers:

- The plain-flow config template writes `prompts = true`.
- `--claude-code` / `--codex` fresh onboard inherit `prompts = true` from the
  `CaptureConfig` dataclass default (they build `TjConfig(...)` directly,
  with no literal template — see `core/config.py::CaptureConfig`).
- The onboarding disclosure line names the new local-only capture.
- Migration: re-running `--claude-code`/`--codex` with `--reconfigure` over
  an existing config that has a stale `prompts = false` picks up the new
  default; a plain re-run (no `--reconfigure`) leaves an explicit value
  alone.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from tokenjam.cli.cmd_onboard import cmd_onboard
from tokenjam.core.config import load_config

# --- Plain flow ---------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_existing_config(monkeypatch):
    monkeypatch.setattr("tokenjam.cli.cmd_onboard.find_config_file", lambda: None)


def _run_plain(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(cmd_onboard, ["--no-daemon", "--budget", "0"], obj={})
        cfg = Path(".tj/config.toml")
        return res, (cfg.read_text() if cfg.exists() else "")


def test_plain_onboard_writes_prompts_true(tmp_path):
    res, cfg_text = _run_plain(tmp_path)
    assert res.exit_code == 0, res.output
    assert "prompts = true" in cfg_text
    assert "tool_inputs = true" in cfg_text
    # completions/tool_outputs stay off — this fix is scoped to prompts only.
    assert "completions = false" in cfg_text
    assert "tool_outputs = false" in cfg_text


def test_plain_onboard_prints_capture_disclosure(tmp_path):
    res, _ = _run_plain(tmp_path)
    assert res.exit_code == 0, res.output
    assert "Prompt capture" in res.output
    assert "locally" in res.output
    assert "capture.prompts = false" in res.output  # the opt-out instruction


# --- --claude-code / --codex fresh onboard: dataclass default ----------
#
# Neither `_onboard_claude_code` nor `_onboard_codex` write a literal
# `[capture]` block on a fresh config — they build `TjConfig(...)` directly,
# so a fresh onboard here is exercising the `CaptureConfig` dataclass default,
# not a template string. If that default ever regresses to False, these are
# the tests that catch it (the plain-flow test above only covers the
# template).


@pytest.fixture
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._stop_serve_for_db_write", lambda: False,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._finish_onboard_serve", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._try_apply_declared_plans", lambda *a, **k: None,
    )
    monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _x: None)
    import tokenjam.core.backfill as backfill_mod
    monkeypatch.setattr(
        backfill_mod, "CLAUDE_CODE_PROJECTS_ROOT", tmp_path / "no-such-claude",
    )


def _invoke(tmp_path, *args, input_text: str = ""):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        res = runner.invoke(cmd_onboard, ["--no-daemon", *args], input=input_text, obj={})
        cfg_path = tmp_path / ".config" / "tj" / "config.toml"
        return res, cfg_path


def test_claude_code_fresh_onboard_captures_prompts_by_default(_isolated_home, tmp_path):
    res, cfg_path = _invoke(
        tmp_path, "--claude-code", "--project", "testproj", "--plan", "max_5x",
    )
    assert res.exit_code == 0, res.output
    config = load_config(str(cfg_path))
    assert config.capture.prompts is True
    assert "Prompt capture" in res.output


def test_codex_fresh_onboard_captures_prompts_by_default(_isolated_home, tmp_path):
    res, cfg_path = _invoke(tmp_path, "--codex", "--plan", "plus")
    assert res.exit_code == 0, res.output
    config = load_config(str(cfg_path))
    assert config.capture.prompts is True
    assert "Prompt capture" in res.output


# --- Migration: re-onboarding over a pre-existing stale config ---------


def _seed_stale_config(tmp_path, *, provider: str, plan: str):
    cfg_dir = tmp_path / ".config" / "tj"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        f'version = "1"\n'
        f"[security]\ningest_secret = \"{'a' * 64}\"\n"
        f"[budget.{provider}]\nplan = \"{plan}\"\ncycle_start_day = 1\n"
        f"[capture]\nprompts = false\ntool_inputs = true\n"
    )


def test_claude_code_reconfigure_upgrades_stale_prompts_false(_isolated_home, tmp_path):
    _seed_stale_config(tmp_path, provider="anthropic", plan="max_20x")
    res, cfg_path = _invoke(
        tmp_path, "--claude-code", "--project", "testproj",
        "--reconfigure", "--plan", "max_20x",
    )
    assert res.exit_code == 0, res.output
    config = load_config(str(cfg_path))
    assert config.capture.prompts is True
    assert "Prompt capture" in res.output


def test_codex_reconfigure_upgrades_stale_prompts_false(_isolated_home, tmp_path):
    _seed_stale_config(tmp_path, provider="openai", plan="team")
    res, cfg_path = _invoke(tmp_path, "--codex", "--reconfigure", "--plan", "team")
    assert res.exit_code == 0, res.output
    config = load_config(str(cfg_path))
    assert config.capture.prompts is True
    assert "Prompt capture" in res.output


def test_claude_code_plain_rerun_leaves_explicit_false_alone(_isolated_home, tmp_path):
    """No --reconfigure: an explicit `prompts = false` is left as-is rather
    than silently flipped — only the deliberate `--reconfigure` action picks
    up the new default (see the comment at the call site in
    `_onboard_claude_code`)."""
    _seed_stale_config(tmp_path, provider="anthropic", plan="max_20x")
    res, cfg_path = _invoke(tmp_path, "--claude-code", "--project", "testproj")
    assert res.exit_code == 0, res.output
    config = load_config(str(cfg_path))
    assert config.capture.prompts is False
    assert "Prompt capture" not in res.output
