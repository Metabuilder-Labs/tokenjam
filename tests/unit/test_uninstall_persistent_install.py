"""Tests for `_find_persistent_install()` (#443, Greptile P1): `tj uninstall`
detects a persistent tokenjam install by probing the ENVIRONMENT, not the
current process's `sys.executable`.

Why this matters: `npx tokenjam uninstall` always runs `tj` via an ephemeral
`uvx`/`pipx run` venv (see npm-wrapper/bin/tj.js's runner preference). The
OLD `sys.executable`-based detection (`_installed_via_pipx()` etc.) always
reported "this process is ephemeral, nothing to remove" on that path — even
when the user had separately `pipx install`ed (or `uv tool install`ed)
tokenjam, leaving that install behind while `tj uninstall` claimed success.

These tests mock subprocess (`pipx list --json` / `uv tool list`) and the
filesystem dir-probe fallback directly — never touch a real pipx/uv install.
CLI-level behavior (prompt wording, execution) is covered in
tests/integration/test_cli_uninstall.py, which patches
`_find_persistent_install()` wholesale.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tokenjam.cli import cmd_uninstall as uninstall_mod


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Fake HOME so the dir-probe fallback never touches the real machine,
    and unset PIPX_HOME/XDG_DATA_HOME so a developer's/CI runner's real
    values can't leak into these tests."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.delenv("PIPX_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return home


# -- _pipx_has_tokenjam(): authoritative `pipx list --json`, dir-probe fallback


def test_pipx_has_tokenjam_true_via_json_list():
    fake = MagicMock(
        returncode=0,
        stdout='{"venvs": {"tokenjam": {"metadata": {}}}}',
        stderr="",
    )
    with patch.object(uninstall_mod.shutil, "which", return_value="/usr/bin/pipx"), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake) as run:
        assert uninstall_mod._pipx_has_tokenjam() is True
    run.assert_called_once_with(
        ["/usr/bin/pipx", "list", "--json"],
        capture_output=True, text=True, timeout=10,
    )


def test_pipx_has_tokenjam_false_via_json_list_when_absent():
    """`pipx list --json` succeeds but tokenjam isn't in it — trust that,
    don't fall back to the dir probe."""
    fake = MagicMock(returncode=0, stdout='{"venvs": {"other": {}}}', stderr="")
    with patch.object(uninstall_mod.shutil, "which", return_value="/usr/bin/pipx"), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake):
        assert uninstall_mod._pipx_has_tokenjam() is False


def test_pipx_has_tokenjam_falls_back_to_dir_probe_when_pipx_missing(tmp_path, _isolate):
    """No `pipx` binary on PATH (e.g. running via an ephemeral uvx venv that
    has no pipx of its own) — fall back to the dir probe."""
    venv_dir = _isolate / ".local" / "pipx" / "venvs" / "tokenjam"
    venv_dir.mkdir(parents=True)
    with patch.object(uninstall_mod.shutil, "which", return_value=None), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        assert uninstall_mod._pipx_has_tokenjam() is True
    run.assert_not_called()


def test_pipx_has_tokenjam_dir_probe_respects_pipx_home_override(tmp_path, monkeypatch, _isolate):
    custom = tmp_path / "custom-pipx"
    (custom / "venvs" / "tokenjam").mkdir(parents=True)
    monkeypatch.setenv("PIPX_HOME", str(custom))
    with patch.object(uninstall_mod.shutil, "which", return_value=None):
        assert uninstall_mod._pipx_has_tokenjam() is True


def test_pipx_has_tokenjam_falls_back_when_command_errors():
    """`pipx` is on PATH but the command exits non-zero — treat as "missing",
    fall back to the dir probe (which reports False since nothing was seeded)."""
    fake = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch.object(uninstall_mod.shutil, "which", return_value="/usr/bin/pipx"), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake):
        assert uninstall_mod._pipx_has_tokenjam() is False


# -- _uv_tool_has_tokenjam(): authoritative `uv tool list`, dir-probe fallback


def test_uv_tool_has_tokenjam_true_via_list():
    fake = MagicMock(
        returncode=0,
        stdout="tokenjam v0.5.4\n- tj\n- tokenjam\nother v1.0.0\n- other\n",
        stderr="",
    )
    with patch.object(uninstall_mod.shutil, "which", return_value="/usr/bin/uv"), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake) as run:
        assert uninstall_mod._uv_tool_has_tokenjam() is True
    run.assert_called_once_with(
        ["/usr/bin/uv", "tool", "list"],
        capture_output=True, text=True, timeout=10,
    )


def test_uv_tool_has_tokenjam_ignores_entrypoint_lines():
    """A tool literally named "tokenjam" as an ENTRYPOINT of some other
    package (indented `- tokenjam` line) must not false-positive."""
    fake = MagicMock(
        returncode=0,
        stdout="other-pkg v1.0.0\n- tokenjam\n- other-pkg\n",
        stderr="",
    )
    with patch.object(uninstall_mod.shutil, "which", return_value="/usr/bin/uv"), \
         patch.object(uninstall_mod.subprocess, "run", return_value=fake):
        assert uninstall_mod._uv_tool_has_tokenjam() is False


def test_uv_tool_has_tokenjam_falls_back_to_dir_probe_when_uv_missing(_isolate):
    tool_dir = _isolate / ".local" / "share" / "uv" / "tools" / "tokenjam"
    tool_dir.mkdir(parents=True)
    with patch.object(uninstall_mod.shutil, "which", return_value=None), \
         patch.object(uninstall_mod.subprocess, "run") as run:
        assert uninstall_mod._uv_tool_has_tokenjam() is True
    run.assert_not_called()


def test_uv_tool_has_tokenjam_dir_probe_respects_xdg_data_home_override(tmp_path, monkeypatch, _isolate):
    custom = tmp_path / "custom-xdg"
    (custom / "uv" / "tools" / "tokenjam").mkdir(parents=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(custom))
    with patch.object(uninstall_mod.shutil, "which", return_value=None):
        assert uninstall_mod._uv_tool_has_tokenjam() is True


# -- _pip_tj_on_path(): plain pip / editable-dev install on PATH


def test_pip_tj_on_path_returns_plain_install():
    with patch.object(uninstall_mod.shutil, "which", return_value="/usr/local/bin/tj"):
        assert uninstall_mod._pip_tj_on_path() == "/usr/local/bin/tj"


def test_pip_tj_on_path_none_when_missing():
    with patch.object(uninstall_mod.shutil, "which", return_value=None):
        assert uninstall_mod._pip_tj_on_path() is None


@pytest.mark.parametrize("path", [
    "/Users/x/.local/share/pipx/venvs/tokenjam/bin/tj",
    "/Users/x/.local/share/uv/tools/tokenjam/bin/tj",
    "/Users/x/.cache/uv/archive-v0/abc123/bin/tj",
    "/Users/x/.local/pipx/.cache/abc123/bin/tj",
])
def test_pip_tj_on_path_excludes_other_managers_and_ephemeral_shims(path):
    """Already covered by the pipx/uv-tool probes, or genuinely ephemeral —
    must not double-count or falsely report a "plain pip" install."""
    with patch.object(uninstall_mod.shutil, "which", return_value=path):
        assert uninstall_mod._pip_tj_on_path() is None


# -- _find_persistent_install(): the combined detection matrix


def test_find_persistent_install_pipx_only():
    with patch.object(uninstall_mod, "_pipx_has_tokenjam", return_value=True), \
         patch.object(uninstall_mod, "_uv_tool_has_tokenjam", return_value=False), \
         patch.object(uninstall_mod, "_pip_tj_on_path", return_value=None):
        installs = uninstall_mod._find_persistent_install()
    assert len(installs) == 1
    assert installs[0].manager == "pipx"
    assert installs[0].auto is True
    assert installs[0].argv == ["pipx", "uninstall", "tokenjam"]


def test_find_persistent_install_uv_tool_only():
    with patch.object(uninstall_mod, "_pipx_has_tokenjam", return_value=False), \
         patch.object(uninstall_mod, "_uv_tool_has_tokenjam", return_value=True), \
         patch.object(uninstall_mod, "_pip_tj_on_path", return_value=None):
        installs = uninstall_mod._find_persistent_install()
    assert len(installs) == 1
    assert installs[0].manager == "uv-tool"
    assert installs[0].auto is True
    assert installs[0].argv == ["uv", "tool", "uninstall", "tokenjam"]


def test_find_persistent_install_pip_editable_only():
    with patch.object(uninstall_mod, "_pipx_has_tokenjam", return_value=False), \
         patch.object(uninstall_mod, "_uv_tool_has_tokenjam", return_value=False), \
         patch.object(uninstall_mod, "_pip_tj_on_path", return_value="/usr/local/bin/tj"):
        installs = uninstall_mod._find_persistent_install()
    assert len(installs) == 1
    assert installs[0].manager == "pip"
    assert installs[0].auto is False
    assert installs[0].argv is None


def test_find_persistent_install_none_found():
    with patch.object(uninstall_mod, "_pipx_has_tokenjam", return_value=False), \
         patch.object(uninstall_mod, "_uv_tool_has_tokenjam", return_value=False), \
         patch.object(uninstall_mod, "_pip_tj_on_path", return_value=None):
        assert uninstall_mod._find_persistent_install() == []


def test_find_persistent_install_both_pipx_and_uv_tool():
    """Both a pipx AND a uv-tool install exist on the machine — both are
    returned, so `tj uninstall` removes each auto-removable one."""
    with patch.object(uninstall_mod, "_pipx_has_tokenjam", return_value=True), \
         patch.object(uninstall_mod, "_uv_tool_has_tokenjam", return_value=True), \
         patch.object(uninstall_mod, "_pip_tj_on_path", return_value=None):
        installs = uninstall_mod._find_persistent_install()
    assert {i.manager for i in installs} == {"pipx", "uv-tool"}
    assert all(i.auto for i in installs)


def test_find_persistent_install_ephemeral_current_process_still_detects_pipx():
    """The scenario this whole fix targets: the CURRENT process is an
    ephemeral uvx/pipx-run runner (as `npx tokenjam uninstall` always is),
    but a persistent pipx install exists elsewhere — it's still detected."""
    with patch.object(uninstall_mod, "_is_ephemeral_runner", return_value=True), \
         patch.object(uninstall_mod, "_pipx_has_tokenjam", return_value=True), \
         patch.object(uninstall_mod, "_uv_tool_has_tokenjam", return_value=False), \
         patch.object(uninstall_mod, "_pip_tj_on_path", return_value=None):
        installs = uninstall_mod._find_persistent_install()
    assert len(installs) == 1
    assert installs[0].manager == "pipx"
