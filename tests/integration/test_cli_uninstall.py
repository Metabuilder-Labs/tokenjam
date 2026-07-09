"""CLI tests for `tj uninstall` messaging + optional package removal.

These focus on the closing UX (issue: uninstall/reinstall confusion) — the
two-step "package remains; reinstall FRESH with upgrade/--force" messaging and
the optional package-removal prompt. Filesystem side effects are neutered by
pointing HOME at a tmp dir and mocking subprocess/shutil so nothing real is
touched.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tokenjam.cli import cmd_uninstall as uninstall_mod
from tokenjam.cli.cmd_uninstall import cmd_uninstall


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Keep uninstall off real machine state: fake HOME/cwd, no real subprocess,
    no `claude` on PATH so the MCP-deregister branch is skipped."""
    home = tmp_path / "home"
    home.mkdir()
    # Wide console so Rich doesn't wrap commands mid-token in assertions.
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.chdir(tmp_path)
    # cmd_stop is invoked first; stub it out so it doesn't poke the real daemon.
    monkeypatch.setattr("tokenjam.cli.cmd_stop.cmd_stop", MagicMock())
    with patch.object(uninstall_mod.shutil, "which", return_value=None):
        yield


def _run(runner, *args, **kwargs):
    return runner.invoke(cmd_uninstall, ["--yes", *args], **kwargs)


def test_messaging_states_package_remains(runner):
    """Default (no --remove-package, --yes): package left in place, two-step
    guidance printed, and the FRESH-reinstall command is upgrade/--force."""
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=True):
        result = _run(runner)
    assert result.exit_code == 0, result.output
    out = result.output
    assert "package itself is still installed" in out
    assert "pipx uninstall tokenjam" in out
    # FRESH reinstall must NOT be a bare `pipx install` (that no-ops).
    assert "pipx upgrade tokenjam" in out
    assert "pipx install --force tokenjam" in out
    assert "no-ops when tokenjam is already present" in out


def test_reinstall_hint_for_pip_install(runner):
    """Non-pipx install: hint is pip uninstall + pip install --upgrade."""
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=False):
        result = _run(runner)
    assert result.exit_code == 0, result.output
    assert "pip uninstall tokenjam" in result.output
    assert "pip install --upgrade tokenjam" in result.output


def test_remove_package_flag_runs_pipx(runner):
    """--remove-package on a pipx install runs `pipx uninstall tokenjam`."""
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=True), \
         patch.object(uninstall_mod.shutil, "which", side_effect=lambda c: "/usr/bin/pipx" if c == "pipx" else None), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake) as run:
        result = _run(runner, "--remove-package")
    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        ["pipx", "uninstall", "tokenjam"], capture_output=True, text=True
    )
    assert "tokenjam package removed" in result.output
    # After a SUCCESSFUL removal the venv is gone, so `pipx upgrade` would fail
    # ("not installed"). The reinstall hint must be a fresh `pipx install`, not
    # upgrade/--force (#430).
    assert "pipx install tokenjam" in result.output
    assert "pipx upgrade" not in result.output


def test_remove_package_flag_non_pipx_prints_command(runner):
    """--remove-package on a non-pipx install must NOT guess/run — it prints
    the right command instead."""
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=False), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = _run(runner, "--remove-package")
    assert result.exit_code == 0, result.output
    # Only cmd_stop (mocked) — no pipx/pip uninstall subprocess fired.
    run.assert_not_called()
    assert "not a pipx- or uv-tool-managed venv" in result.output
    assert "pip uninstall tokenjam" in result.output


def test_prompt_offered_when_interactive(runner):
    """Without --yes, the user is asked whether to also remove the package;
    answering No leaves it and prints the two-step guidance."""
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=True):
        # First prompt: confirm delete-all (y). Second: also remove package (n).
        result = runner.invoke(cmd_uninstall, [], input="y\nn\n")
    assert result.exit_code == 0, result.output
    assert "Also remove the tokenjam package now?" in result.output
    assert "package itself is still installed" in result.output


def test_prompt_wording_matches_install_type(runner):
    """The confirm prompt must describe what actually happens: pipx installs are
    auto-run, pip/venv installs only get the command printed (#430)."""
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=True):
        res_pipx = runner.invoke(cmd_uninstall, [], input="y\nn\n")
    assert "runs pipx uninstall tokenjam" in res_pipx.output

    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=False):
        res_pip = runner.invoke(cmd_uninstall, [], input="y\nn\n")
    assert "prints pip uninstall tokenjam" in res_pip.output


def test_remove_package_flag_runs_uv_tool(runner):
    """--purge (alias for --remove-package) on a uv-tool install runs
    `uv tool uninstall tokenjam` (#121)."""
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=False), \
         patch.object(uninstall_mod, "_installed_via_uv_tool", return_value=True), \
         patch.object(uninstall_mod.shutil, "which", side_effect=lambda c: "/usr/bin/uv" if c == "uv" else None), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake) as run:
        result = _run(runner, "--purge")
    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        ["uv", "tool", "uninstall", "tokenjam"], capture_output=True, text=True
    )
    assert "tokenjam package removed" in result.output
    assert "uv tool install tokenjam" in result.output
    assert "uv tool upgrade" not in result.output


def test_uv_tool_reinstall_hint(runner):
    """Non-pipx, uv-tool install: hint is `uv tool uninstall` + `uv tool
    upgrade` (#121)."""
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=False), \
         patch.object(uninstall_mod, "_installed_via_uv_tool", return_value=True):
        result = _run(runner)
    assert result.exit_code == 0, result.output
    assert "uv tool uninstall tokenjam" in result.output
    assert "uv tool upgrade tokenjam" in result.output


def test_purge_is_noop_on_ephemeral_runner(runner):
    """`--purge` on an ephemeral runner (uvx/pipx run — no persistent
    install) must not attempt any removal; it explains why and exits clean
    (#121)."""
    with patch.object(uninstall_mod, "_is_ephemeral_runner", return_value=True), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = _run(runner, "--purge")
    assert result.exit_code == 0, result.output
    run.assert_not_called()
    assert "no persistent" in result.output.lower() or "nothing persistent" in result.output.lower()
    assert "--purge is a no-op here" in result.output
    # The (inapplicable) "package still installed" two-step must not print.
    assert "package itself is still installed" not in result.output


def test_no_purge_prompt_on_ephemeral_runner(runner):
    """Without --purge/--yes, an ephemeral runner must not ask to remove a
    package that was never persistently installed (#121)."""
    with patch.object(uninstall_mod, "_is_ephemeral_runner", return_value=True):
        result = runner.invoke(cmd_uninstall, [], input="y\n")
    assert result.exit_code == 0, result.output
    assert "Also remove the tokenjam package now?" not in result.output
    assert "nothing persistent to remove" in result.output.lower()


def test_pipx_uninstall_failure_falls_back_to_manual(runner):
    """If pipx uninstall fails, we surface the error and the manual command
    rather than claiming success."""
    fake = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=True), \
         patch.object(uninstall_mod.shutil, "which", side_effect=lambda c: "/usr/bin/pipx" if c == "pipx" else None), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake):
        result = _run(runner, "--remove-package")
    assert result.exit_code == 0, result.output
    assert "Could not remove the package automatically" in result.output
    assert "pipx uninstall tokenjam" in result.output
