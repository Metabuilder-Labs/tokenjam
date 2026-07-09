"""Runner detection -> package-removal command mapping for `tj uninstall`
(#121): pipx vs `uv tool` vs plain pip vs an ephemeral runner (uvx/pipx run)
with nothing persistent to remove.

Paths below were verified empirically against real `uv`/`pipx` installs:
  - `uv tool install <pkg>` venv:  .../uv/tools/<pkg>/bin/python
  - `uvx --from <pkg> ...` venv:   .../uv/archive-v0/<hash>/bin/python
  - `pipx install <pkg>` venv:     .../pipx/venvs/<pkg>/bin/python
  - `pipx run --spec <pkg> ...`:   delegates to uv when present (same as
    uvx above), else falls back to .../pipx/.cache/<hash>/bin/python
"""
from __future__ import annotations

from unittest.mock import patch

from tokenjam.cli import cmd_uninstall as uninstall_mod


def _exe(path: str):
    return patch.object(uninstall_mod.sys, "executable", path)


# --- persistent installs: detected, NOT ephemeral ---------------------------


def test_pipx_venv_detected_as_pipx_not_ephemeral():
    with _exe("/Users/x/.local/share/pipx/venvs/tokenjam/bin/python"):
        assert uninstall_mod._installed_via_pipx() is True
        assert uninstall_mod._installed_via_uv_tool() is False
        assert uninstall_mod._is_ephemeral_runner() is False


def test_uv_tool_venv_detected_as_uv_tool_not_ephemeral():
    with _exe("/Users/x/.local/share/uv/tools/tokenjam/bin/python"):
        assert uninstall_mod._installed_via_uv_tool() is True
        assert uninstall_mod._installed_via_pipx() is False
        assert uninstall_mod._is_ephemeral_runner() is False


def test_plain_pip_venv_is_neither_pipx_uv_tool_nor_ephemeral():
    with _exe("/usr/local/bin/python3"):
        assert uninstall_mod._installed_via_pipx() is False
        assert uninstall_mod._installed_via_uv_tool() is False
        assert uninstall_mod._is_ephemeral_runner() is False


def test_uv_managed_python_runtime_is_not_ephemeral():
    """A `pip install`-into-a-uv-managed-Python setup must NOT be classified
    as ephemeral — that previously caused `tj uninstall --purge` to silently
    no-op for these users (the `/uv/` substring check also matched uv's
    managed *Python runtime* paths, not just its throwaway cache venvs).
    """
    with _exe(
        "/Users/x/.local/share/uv/python/cpython-3.12.4-macos-aarch64-none/"
        "bin/python3"
    ):
        assert uninstall_mod._installed_via_uv_tool() is False
        assert uninstall_mod._is_ephemeral_runner() is False
        # Falls through to the plain pip/venv path.
        assert uninstall_mod._package_uninstall_hint() == "pip uninstall tokenjam"


# --- ephemeral runners: ephemeral, NOT a persistent install ------------------


def test_uvx_cache_venv_detected_as_ephemeral():
    with _exe("/Users/x/.cache/uv/archive-v0/abc123/bin/python"):
        assert uninstall_mod._is_ephemeral_runner() is True
        assert uninstall_mod._installed_via_pipx() is False
        assert uninstall_mod._installed_via_uv_tool() is False


def test_pipx_run_cache_venv_detected_as_ephemeral():
    with _exe("/Users/x/.local/pipx/.cache/abc123/bin/python"):
        assert uninstall_mod._is_ephemeral_runner() is True
        assert uninstall_mod._installed_via_pipx() is False


# --- command mapping: uninstall / reinstall / fresh-install hints -----------


def test_command_hints_for_pipx():
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=True), \
         patch.object(uninstall_mod, "_installed_via_uv_tool", return_value=False):
        assert uninstall_mod._package_uninstall_hint() == "pipx uninstall tokenjam"
        assert "pipx upgrade tokenjam" in uninstall_mod._package_reinstall_hint()
        assert uninstall_mod._package_fresh_install_hint() == "pipx install tokenjam"


def test_command_hints_for_uv_tool():
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=False), \
         patch.object(uninstall_mod, "_installed_via_uv_tool", return_value=True):
        assert uninstall_mod._package_uninstall_hint() == "uv tool uninstall tokenjam"
        assert "uv tool upgrade tokenjam" in uninstall_mod._package_reinstall_hint()
        assert uninstall_mod._package_fresh_install_hint() == "uv tool install tokenjam"


def test_command_hints_for_plain_pip():
    with patch.object(uninstall_mod, "_installed_via_pipx", return_value=False), \
         patch.object(uninstall_mod, "_installed_via_uv_tool", return_value=False):
        assert uninstall_mod._package_uninstall_hint() == "pip uninstall tokenjam"
        assert uninstall_mod._package_reinstall_hint() == "pip install --upgrade tokenjam"
        assert uninstall_mod._package_fresh_install_hint() == "pip install tokenjam"
