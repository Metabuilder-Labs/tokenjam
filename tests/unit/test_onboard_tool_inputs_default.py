"""Tool-input capture was inverted across onboarding flows: the plain/SDK
flow wrote `[capture] tool_inputs = true` in its literal TOML template (whose
own comment mistakenly credited it to `tj context` / the statusline — both
Claude Code-only features the SDK/bare persona never reaches), while
`--claude-code` / `--codex` never set it at all and silently inherited the
`CaptureConfig.tool_inputs` dataclass default of `False`. That default is
exactly backwards: Claude Code's JSONL backfill is the persona that actually
populates `gen_ai.tool.input` (Read/Grep/Glob file paths and search queries),
which the `script` (workflow-restructure) and `verbosity` analyzers need for
argument-shape clustering instead of falling back to tool-names-only,
degraded mode.

This mirrors `test_onboard_capture_default.py`'s coverage of the earlier
`prompts` default-flip, applied to `tool_inputs`. Covers:

- The plain-flow config template still writes `tool_inputs = true` (unchanged
  value, corrected comment).
- `--claude-code` / `--codex` fresh onboard now inherit `tool_inputs = true`
  from the `CaptureConfig` dataclass default (they build `TjConfig(...)`
  directly, with no literal template — see `core/config.py::CaptureConfig`).
- The onboarding disclosure line names the newly-corrected local-only capture.
- Migration: re-running `--claude-code`/`--codex` with `--reconfigure` over
  an existing config that has a stale `tool_inputs = false` picks up the new
  default; a plain re-run (no `--reconfigure`) leaves an explicit value alone.
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


def test_plain_onboard_still_writes_tool_inputs_true(tmp_path):
    # Value is unchanged (it was already correct here); this guards against a
    # future edit accidentally flipping it while fixing the misleading comment.
    res, cfg_text = _run_plain(tmp_path)
    assert res.exit_code == 0, res.output
    assert "tool_inputs = true" in cfg_text


def test_plain_onboard_prints_tool_input_disclosure(tmp_path):
    res, _ = _run_plain(tmp_path)
    assert res.exit_code == 0, res.output
    assert "Tool-input capture" in res.output
    assert "locally" in res.output
    assert "capture.tool_inputs = false" in res.output  # the opt-out instruction


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


def test_claude_code_fresh_onboard_captures_tool_inputs_by_default(_isolated_home, tmp_path):
    res, cfg_path = _invoke(
        tmp_path, "--claude-code", "--project", "testproj", "--plan", "max_5x",
    )
    assert res.exit_code == 0, res.output
    config = load_config(str(cfg_path))
    assert config.capture.tool_inputs is True
    assert "Tool-input capture" in res.output


def test_codex_fresh_onboard_captures_tool_inputs_by_default(_isolated_home, tmp_path):
    res, cfg_path = _invoke(tmp_path, "--codex", "--plan", "plus")
    assert res.exit_code == 0, res.output
    config = load_config(str(cfg_path))
    assert config.capture.tool_inputs is True
    assert "Tool-input capture" in res.output


# --- Migration: re-onboarding over a pre-existing stale config ---------


def _seed_stale_config(tmp_path, *, provider: str, plan: str):
    cfg_dir = tmp_path / ".config" / "tj"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text(
        f'version = "1"\n'
        f"[security]\ningest_secret = \"{'a' * 64}\"\n"
        f"[budget.{provider}]\nplan = \"{plan}\"\ncycle_start_day = 1\n"
        f"[capture]\nprompts = true\ntool_inputs = false\n"
    )


def test_claude_code_reconfigure_upgrades_stale_tool_inputs_false(_isolated_home, tmp_path):
    _seed_stale_config(tmp_path, provider="anthropic", plan="max_20x")
    res, cfg_path = _invoke(
        tmp_path, "--claude-code", "--project", "testproj",
        "--reconfigure", "--plan", "max_20x",
    )
    assert res.exit_code == 0, res.output
    config = load_config(str(cfg_path))
    assert config.capture.tool_inputs is True
    assert "Tool-input capture" in res.output


def test_codex_reconfigure_upgrades_stale_tool_inputs_false(_isolated_home, tmp_path):
    _seed_stale_config(tmp_path, provider="openai", plan="team")
    res, cfg_path = _invoke(tmp_path, "--codex", "--reconfigure", "--plan", "team")
    assert res.exit_code == 0, res.output
    config = load_config(str(cfg_path))
    assert config.capture.tool_inputs is True
    assert "Tool-input capture" in res.output


def test_claude_code_plain_rerun_leaves_explicit_false_alone(_isolated_home, tmp_path):
    """No --reconfigure: an explicit `tool_inputs = false` is left as-is
    rather than silently flipped — only the deliberate `--reconfigure` action
    picks up the new default (see the comment at the call site in
    `_onboard_claude_code`)."""
    _seed_stale_config(tmp_path, provider="anthropic", plan="max_20x")
    res, cfg_path = _invoke(tmp_path, "--claude-code", "--project", "testproj")
    assert res.exit_code == 0, res.output
    config = load_config(str(cfg_path))
    assert config.capture.tool_inputs is False
    assert "Tool-input capture" not in res.output
