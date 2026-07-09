"""Tests for `tj reset` (#442): the config-only counterpart to `tj uninstall`.

`tj uninstall` was redesigned to do a full removal by default — config/
daemon/wiring AND the tokenjam package itself, dropping the old `--purge`/
`--remove-package` flags. `tj reset` is the new command that runs ONLY the
shared `_teardown_side_effects()` helper (config/daemon/wiring cleanup) and
never touches the package, leaving the CLI ready for `tj onboard` again.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tokenjam.cli import cmd_uninstall as uninstall_mod
from tokenjam.cli.cmd_reset import cmd_reset


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Keep reset off real machine state: fake HOME/cwd, no real subprocess,
    no `claude` on PATH. Most tests below mock `_teardown_side_effects()`
    entirely, but this guards any future test that exercises it for real —
    mirrors the `_isolate` fixture in tests/integration/test_cli_uninstall.py
    (Greptile #443)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tokenjam.cli.cmd_stop.cmd_stop", MagicMock())
    monkeypatch.delenv("PIPX_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    with patch.object(uninstall_mod.shutil, "which", return_value=None):
        yield


def test_reset_calls_shared_teardown(runner):
    """`tj reset --yes` invokes the same `_teardown_side_effects()` helper
    that `tj uninstall` uses for its config/wiring cleanup."""
    with patch.object(uninstall_mod, "_teardown_side_effects") as teardown:
        result = runner.invoke(cmd_reset, ["--yes"])
    assert result.exit_code == 0, result.output
    teardown.assert_called_once()


def test_reset_never_touches_package_removal(runner):
    """`tj reset` must not invoke any package-removal path — no subprocess
    call, regardless of how tokenjam was installed."""
    with patch.object(uninstall_mod, "_teardown_side_effects"), \
         patch.object(uninstall_mod, "_remove_persistent_install") as remove_install, \
         patch.object(uninstall_mod, "_find_persistent_install") as find_installs, \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = runner.invoke(cmd_reset, ["--yes"])
    assert result.exit_code == 0, result.output
    remove_install.assert_not_called()
    find_installs.assert_not_called()
    run.assert_not_called()


def test_reset_prints_onboard_hint(runner):
    """After a successful reset, tell the user how to set back up."""
    with patch.object(uninstall_mod, "_teardown_side_effects"):
        result = runner.invoke(cmd_reset, ["--yes"])
    assert result.exit_code == 0, result.output
    assert "tj onboard" in result.output


def test_reset_prompt_confirms_before_wiping(runner):
    """Without --yes, `tj reset` asks first — declining is a no-op that never
    reaches the teardown helper."""
    with patch.object(uninstall_mod, "_teardown_side_effects") as teardown:
        result = runner.invoke(cmd_reset, [], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Cancelled" in result.output
    teardown.assert_not_called()


def test_reset_prompt_wording_keeps_package_installed(runner):
    """The confirm prompt must make clear the package stays installed — the
    distinguishing behavior from `tj uninstall`'s full-removal prompt."""
    with patch.object(uninstall_mod, "_teardown_side_effects"):
        result = runner.invoke(cmd_reset, [], input="y\n")
    assert result.exit_code == 0, result.output
    assert "tokenjam package itself stays installed" in result.output
