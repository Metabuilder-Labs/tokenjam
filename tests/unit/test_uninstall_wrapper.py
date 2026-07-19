"""Tests for #117: `tj uninstall` must remove the `claude()` shell wrapper
that `tj onboard --claude-code` installs (and clean up any legacy output-cap
PostToolUse hook entry left by a prior release alongside it).

Two layers:
  - `_unwire_claude_wrapper()` in cmd_onboard.py — the counterpart to
    `_install_claude_wrapper()` that `tj uninstall` was previously missing
    entirely, leaving a `claude()` function that calls `tj` (and errors)
    after the package is uninstalled.
  - the `cmd_uninstall` CLI path end-to-end: seed a fixture ~/.zshrc with
    both onboard-written blocks (harness observability + the wrapper), run
    uninstall, and assert no `tj ` references or `claude()` function remain.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tokenjam.cli import cmd_onboard as onboard_mod
from tokenjam.cli import cmd_uninstall as uninstall_mod
from tokenjam.cli.cmd_onboard import (
    _WRAPPER_END_MARKER,
    _WRAPPER_MARKER,
    _install_claude_wrapper,
    _unwire_claude_wrapper,
)
from tokenjam.cli.cmd_uninstall import cmd_uninstall


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(onboard_mod.Path, "home", lambda: home)
    return home


def test_unwire_removes_block_written_by_install(fake_home):
    _install_claude_wrapper()
    zshrc = fake_home / ".zshrc"
    assert _WRAPPER_MARKER in zshrc.read_text()

    removed = _unwire_claude_wrapper()
    assert str(zshrc) in removed

    cleaned = zshrc.read_text()
    assert _WRAPPER_MARKER not in cleaned
    assert _WRAPPER_END_MARKER not in cleaned
    assert "claude()" not in cleaned
    assert "tj " not in cleaned


def test_unwire_also_cleans_bashrc_when_present(fake_home):
    (fake_home / ".bashrc").write_text("# pre-existing bashrc content\n")
    _install_claude_wrapper()
    assert _WRAPPER_MARKER in (fake_home / ".bashrc").read_text()

    removed = _unwire_claude_wrapper()
    assert str(fake_home / ".bashrc") in removed
    cleaned = (fake_home / ".bashrc").read_text()
    assert _WRAPPER_MARKER not in cleaned
    assert "pre-existing bashrc content" in cleaned


def test_unwire_is_idempotent_noop_when_absent(fake_home):
    (fake_home / ".zshrc").write_text("echo hi\n")
    assert _unwire_claude_wrapper() == []
    assert (fake_home / ".zshrc").read_text() == "echo hi\n"


def test_unwire_strips_block_with_no_trailing_newline(fake_home):
    """A block that is the LAST line of the rc file, with no final newline,
    must still be stripped — the regex previously required a hard `\\n`
    after `_WRAPPER_END_MARKER`, silently no-opping here and leaving the
    wrapper (and its embedded `tj` calls) behind."""
    zshrc = fake_home / ".zshrc"
    _install_claude_wrapper()
    text = zshrc.read_text()
    assert text.endswith("\n")
    zshrc.write_text(text.rstrip("\n"))  # drop the final newline
    assert not zshrc.read_text().endswith("\n")

    removed = _unwire_claude_wrapper()
    assert str(zshrc) in removed

    cleaned = zshrc.read_text()
    assert _WRAPPER_MARKER not in cleaned
    assert _WRAPPER_END_MARKER not in cleaned
    assert "claude()" not in cleaned
    assert "tj " not in cleaned


def test_unwire_preserves_surrounding_content(fake_home):
    zshrc = fake_home / ".zshrc"
    zshrc.write_text("export FOO=bar\nalias ll='ls -la'\n")
    _install_claude_wrapper()
    with zshrc.open("a") as f:
        f.write("\nexport AFTER=1\n")

    _unwire_claude_wrapper()
    cleaned = zshrc.read_text()
    assert "export FOO=bar" in cleaned
    assert "alias ll='ls -la'" in cleaned
    assert "export AFTER=1" in cleaned
    assert _WRAPPER_MARKER not in cleaned


_OTEL_ZSHRC_BLOCK = (
    "\n# tj harness observability\n"
    "export CLAUDE_CODE_ENABLE_TELEMETRY=1\n"
    "export OTEL_LOGS_EXPORTER=otlp\n"
    "export OTEL_EXPORTER_OTLP_PROTOCOL=http/json\n"
    "export OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:7391\n"
    'export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer deadbeef"\n'
)


def test_uninstall_cli_removes_wrapper_and_cap_hook(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tokenjam.cli.cmd_stop.cmd_stop", MagicMock())
    # `_find_persistent_install()`'s dir-probe fallback reads these directly
    # (not via Path.home()) — unset so a real PIPX_HOME/XDG_DATA_HOME on the
    # dev/CI machine can't leak into this test's package-removal step.
    monkeypatch.delenv("PIPX_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)

    # Seed a fixture ~/.zshrc as `tj onboard --claude-code` would leave it:
    # the OTEL harness-observability block, then the claude() wrapper.
    (home / ".zshrc").write_text(_OTEL_ZSHRC_BLOCK)
    _install_claude_wrapper()

    # Simulate a legacy output-cap PostToolUse hook entry from a prior release
    # (the hook itself is removed — this only exercises `tj uninstall`'s
    # best-effort cleanup of an already-installed entry it might find).
    claude_dir = home / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    settings_path = claude_dir / "settings.json"
    settings: dict = {
        "hooks": {
            "PostToolUse": [{
                "matcher": "Bash|Grep|Glob|WebFetch",
                "hooks": [{"type": "command", "command": "tj hook cap-output"}],
            }],
        },
    }
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    zshrc_before = (home / ".zshrc").read_text()
    assert "claude()" in zshrc_before
    assert "tj " in zshrc_before

    runner = CliRunner()
    with patch.object(uninstall_mod.shutil, "which", return_value=None):
        result = runner.invoke(cmd_uninstall, ["--yes"])
    assert result.exit_code == 0, result.output

    zshrc_after = (home / ".zshrc").read_text()
    assert "claude()" not in zshrc_after
    assert "tj " not in zshrc_after

    settings_after = json.loads(settings_path.read_text())
    post = settings_after.get("hooks", {}).get("PostToolUse", [])
    assert not any(
        "hook cap-output" in str(h)
        for entry in post
        for h in entry.get("hooks", [])
    )
