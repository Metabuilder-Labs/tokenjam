"""CLI tests for `tj uninstall` — full removal by default (#442), with
package detection now ENVIRONMENT-WIDE via `_find_persistent_install()`
rather than the current process's `sys.executable` (Greptile P1 on #443):
`npx tokenjam uninstall` always runs `tj` via an ephemeral `uvx`/`pipx run`
venv, so the old `sys.executable`-based detection always reported "nothing
to remove" even when a persistent pipx/uv-tool install existed elsewhere on
the machine — and the confirm prompt lied about what would actually happen.

These tests patch `_find_persistent_install()` directly to drive
`cmd_uninstall`'s prompt wording and execution; the detection function's own
probing logic (subprocess + dir-probe fallback) is unit-tested in
tests/unit/test_uninstall_persistent_install.py. Filesystem side effects are
neutered by pointing HOME at a tmp dir and mocking subprocess/shutil so
nothing real is touched.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tokenjam.cli import cmd_uninstall as uninstall_mod
from tokenjam.cli.cmd_uninstall import PersistentInstall, cmd_uninstall


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
    # `_find_persistent_install()`'s dir-probe fallback reads these directly
    # (not via Path.home()) — unset so a developer's/CI runner's real
    # PIPX_HOME/XDG_DATA_HOME can never leak a real tokenjam install into a
    # test that doesn't explicitly mock `_find_persistent_install()`.
    monkeypatch.delenv("PIPX_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    with patch.object(uninstall_mod.shutil, "which", return_value=None):
        yield


def _run(runner, *args, **kwargs):
    return runner.invoke(cmd_uninstall, ["--yes", *args], **kwargs)


_PIPX_INSTALL = PersistentInstall(
    manager="pipx", auto=True,
    argv=["pipx", "uninstall", "tokenjam"], display="pipx uninstall tokenjam",
)
_UV_TOOL_INSTALL = PersistentInstall(
    manager="uv-tool", auto=True,
    argv=["uv", "tool", "uninstall", "tokenjam"],
    display="uv tool uninstall tokenjam",
)
_PIP_INSTALL = PersistentInstall(
    manager="pip", auto=False, argv=None, display="pip uninstall tokenjam",
)


def test_default_removes_package_for_pipx_install(runner):
    """A detected persistent pipx install is auto-removed by default."""
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(uninstall_mod, "_find_persistent_install", return_value=[_PIPX_INSTALL]), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake) as run:
        result = _run(runner)
    assert result.exit_code == 0, result.output
    run.assert_called_once_with(
        ["pipx", "uninstall", "tokenjam"], capture_output=True, text=True
    )
    assert "tokenjam package removed" in result.output
    # After a SUCCESSFUL removal the venv is gone, so `pipx upgrade` would fail
    # ("not installed"). The reinstall hint must be a fresh `pipx install`, not
    # upgrade/--force (#430) — and must reflect the MANAGER that was actually
    # removed (pipx), not the current (often ephemeral) process's guess.
    assert "pipx install tokenjam" in result.output
    assert "pipx upgrade" not in result.output


def test_default_removes_package_for_uv_tool_install(runner):
    """A detected persistent uv-tool install is auto-removed by default."""
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(uninstall_mod, "_find_persistent_install", return_value=[_UV_TOOL_INSTALL]), \
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
    """A detected pip/editable install is never auto-run — only printed
    (guessing the wrong environment, or nuking a shared/live/editable-dev
    venv, is worse than a copy-paste)."""
    with patch.object(uninstall_mod, "_find_persistent_install", return_value=[_PIP_INSTALL]), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = _run(runner)
    assert result.exit_code == 0, result.output
    run.assert_not_called()
    assert "not a pipx- or uv-tool-managed venv" in result.output
    assert "pip uninstall tokenjam" in result.output


def test_default_noop_when_nothing_persistent_found(runner):
    """No persistent install anywhere → no package-removal attempt, clean exit."""
    with patch.object(uninstall_mod, "_find_persistent_install", return_value=[]), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = _run(runner)
    assert result.exit_code == 0, result.output
    run.assert_not_called()
    assert "no persistent tokenjam install found" in result.output.lower()


def test_both_pipx_and_uv_tool_present_removes_both(runner):
    """If BOTH a pipx and a uv-tool install exist on the machine, `tj
    uninstall` removes each auto-removable one, not just the first."""
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(
        uninstall_mod, "_find_persistent_install",
        return_value=[_PIPX_INSTALL, _UV_TOOL_INSTALL],
    ), patch.object(uninstall_mod.subprocess, "run", return_value=fake) as run:
        result = _run(runner)
    assert result.exit_code == 0, result.output
    assert run.call_count == 2
    run.assert_any_call(
        ["pipx", "uninstall", "tokenjam"], capture_output=True, text=True
    )
    run.assert_any_call(
        ["uv", "tool", "uninstall", "tokenjam"], capture_output=True, text=True
    )


# -- Honest confirm prompt (Greptile P1 on #443) -----------------------------
#
# Detection now runs BEFORE the prompt is built, and is environment-wide
# (`_find_persistent_install()`), not based on the current process. Before
# this fix, `npx tokenjam uninstall` — which always runs `tj` via an
# ephemeral uvx/pipx-run venv — would introspect ITS OWN `sys.executable`,
# always conclude "ephemeral, nothing to remove", and both the prompt and
# the outcome would silently leave a persistent pipx/uv-tool install behind.


def test_prompt_promises_removal_when_persistent_pipx_present(runner):
    """Without --yes: even when the CURRENT process is an ephemeral
    uvx/pipx-run runner, if `_find_persistent_install()` detects a
    persistent pipx install elsewhere on the machine, the prompt promises
    package removal — and answering yes actually removes it."""
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(uninstall_mod, "_is_ephemeral_runner", return_value=True), \
         patch.object(uninstall_mod, "_find_persistent_install", return_value=[_PIPX_INSTALL]), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake) as run:
        result = runner.invoke(cmd_uninstall, [], input="y\n")
    assert result.exit_code == 0, result.output
    assert "uninstall the tokenjam package (via pipx)" in result.output
    run.assert_called_once()
    assert "tokenjam package removed" in result.output


def test_prompt_does_not_mention_package_when_nothing_persistent(runner):
    """Without --yes: if nothing persistent is found anywhere, the prompt
    must NOT claim it will remove a package — only config/daemon/wiring."""
    with patch.object(uninstall_mod, "_find_persistent_install", return_value=[]), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = runner.invoke(cmd_uninstall, [], input="y\n")
    assert result.exit_code == 0, result.output
    assert "uninstall the tokenjam package" not in result.output
    assert "config, telemetry history" in result.output
    run.assert_not_called()


def test_prompt_wording_for_pip_install_defers_to_shown_command(runner):
    """Without --yes: a pip/editable install's prompt says the command will
    be shown after (never auto-run), instead of claiming an automatic
    removal."""
    with patch.object(uninstall_mod, "_find_persistent_install", return_value=[_PIP_INSTALL]), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = runner.invoke(cmd_uninstall, [], input="y\n")
    assert result.exit_code == 0, result.output
    assert "will need `pip uninstall tokenjam` (shown after)" in result.output
    run.assert_not_called()
    assert "pip uninstall tokenjam" in result.output


def test_declining_confirm_cancels_everything(runner):
    """Declining the single prompt cancels the whole operation — no teardown,
    no package removal attempted."""
    with patch.object(uninstall_mod, "_find_persistent_install", return_value=[_PIPX_INSTALL]), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        result = runner.invoke(cmd_uninstall, [], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Cancelled" in result.output
    run.assert_not_called()


def test_pipx_uninstall_failure_falls_back_to_manual(runner):
    """If pipx uninstall fails, we surface the error and the manual command
    rather than claiming success."""
    fake = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch.object(uninstall_mod, "_find_persistent_install", return_value=[_PIPX_INSTALL]), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake):
        result = _run(runner)
    assert result.exit_code == 0, result.output
    assert "Could not remove the package automatically" in result.output
    assert "pipx uninstall tokenjam" in result.output
