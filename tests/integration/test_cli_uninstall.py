"""CLI tests for `tj uninstall` — full removal by default (#442): config,
daemon, wiring, AND the tokenjam package itself, in one command. The
installer-aware package-removal step is unchanged from #121 — pipx/uv-tool
installs are auto-run, a plain pip/venv install only gets the exact command
printed (never guessed), and an ephemeral runner (uvx/pipx run) is a no-op
with an explanation. Filesystem side effects are neutered by pointing HOME at
a tmp dir and mocking subprocess/shutil so nothing real is touched.
"""
from __future__ import annotations

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


def test_default_removes_package_for_pipx_install(runner):
    """Default `tj uninstall --yes` now ALSO removes the package (previously
    gated behind --remove-package/--purge, which #442 dropped) — a pipx
    install auto-runs `pipx uninstall tokenjam`."""
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=True), \
         patch.object(uninstall_mod.shutil, "which", side_effect=lambda c: "/usr/bin/pipx" if c == "pipx" else None), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake) as run:
        result = _run(runner)
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


def test_default_removes_package_for_uv_tool_install(runner):
    """Default `tj uninstall --yes` on a uv-tool install auto-runs
    `uv tool uninstall tokenjam` (#121, now unconditional)."""
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=False), \
         patch.object(uninstall_mod, "_installed_via_uv_tool", return_value=True), \
         patch.object(uninstall_mod.shutil, "which", side_effect=lambda c: "/usr/bin/uv" if c == "uv" else None), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake) as run:
        result = _run(runner)
    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        ["uv", "tool", "uninstall", "tokenjam"], capture_output=True, text=True
    )
    assert "tokenjam package removed" in result.output
    assert "uv tool install tokenjam" in result.output
    assert "uv tool upgrade" not in result.output


def test_default_prints_command_for_pip_install(runner):
    """Default `tj uninstall --yes` on a non-pipx/non-uv-tool install must NOT
    guess/run the package removal — it prints the right command instead.

    Mocks `_installed_via_uv_tool` and `_is_ephemeral_runner` too (not just
    `_installed_via_pipx`) so this is hermetic — otherwise it silently
    depends on whatever venv the CI runner's `sys.executable` happens to
    live under (e.g. a uv-managed venv would flip the branch taken)."""
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=False), \
         patch.object(uninstall_mod, "_installed_via_uv_tool", return_value=False), \
         patch.object(uninstall_mod, "_is_ephemeral_runner", return_value=False), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = _run(runner)
    assert result.exit_code == 0, result.output
    # Only cmd_stop (mocked) — no pipx/pip uninstall subprocess fired.
    run.assert_not_called()
    assert "not a pipx- or uv-tool-managed venv" in result.output
    assert "pip uninstall tokenjam" in result.output


def test_confirm_prompt_mentions_config_and_package(runner):
    """Without --yes, the single confirmation prompt must make clear it
    removes BOTH config/wiring AND the package — there is no longer a second
    "also remove the package?" prompt (#442)."""
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=True), \
         patch.object(uninstall_mod.shutil, "which", side_effect=lambda c: "/usr/bin/pipx" if c == "pipx" else None), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake):
        result = runner.invoke(cmd_uninstall, [], input="y\n")
    assert result.exit_code == 0, result.output
    assert "AND remove the tokenjam package itself" in result.output
    assert "tokenjam package removed" in result.output


def test_declining_confirm_cancels_everything(runner):
    """Declining the single prompt cancels the whole operation — no teardown,
    no package removal attempted."""
    with patch.object(uninstall_mod.subprocess, "run") as run:
        result = runner.invoke(cmd_uninstall, [], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Cancelled" in result.output
    run.assert_not_called()


def test_ephemeral_runner_is_noop_for_package_removal(runner):
    """An ephemeral runner (uvx/pipx run — no persistent install) must not
    attempt any package removal; it explains why and exits clean (#121)."""
    with patch.object(uninstall_mod, "_is_ephemeral_runner", return_value=True), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = _run(runner)
    assert result.exit_code == 0, result.output
    run.assert_not_called()
    assert "nothing persistent to remove" in result.output.lower()


def test_pipx_uninstall_failure_falls_back_to_manual(runner):
    """If pipx uninstall fails, we surface the error and the manual command
    rather than claiming success."""
    fake = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=True), \
         patch.object(uninstall_mod.shutil, "which", side_effect=lambda c: "/usr/bin/pipx" if c == "pipx" else None), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake):
        result = _run(runner)
    assert result.exit_code == 0, result.output
    assert "Could not remove the package automatically" in result.output
    assert "pipx uninstall tokenjam" in result.output
