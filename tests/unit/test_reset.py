"""Tests for `tj reset` (#442): the config-only counterpart to `tj uninstall`.

`tj uninstall` was redesigned to do a full removal by default — config/
daemon/wiring AND the tokenjam package itself, dropping the old `--purge`/
`--remove-package` flags. `tj reset` is the new command that runs ONLY the
shared `_teardown_side_effects()` helper (config/daemon/wiring cleanup) and
never touches the package, leaving the CLI ready for `tj onboard` again.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from tokenjam.cli import cmd_uninstall as uninstall_mod
from tokenjam.cli.cmd_reset import cmd_reset


@pytest.fixture
def runner():
    return CliRunner()


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
         patch.object(uninstall_mod, "_remove_package") as remove_package, \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = runner.invoke(cmd_reset, ["--yes"])
    assert result.exit_code == 0, result.output
    remove_package.assert_not_called()
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
