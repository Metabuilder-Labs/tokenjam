"""Console-script entrypoint for `tj` / `tokenjam`.

Why this file exists separately from `tokenjam.cli.main`: it is force-included
into the wheel (see `[tool.hatch.build.targets.wheel.force-include]` in
`pyproject.toml`), so it is physically copied into site-packages even for an
editable install. That matters because an editable install of `tokenjam`
otherwise resolves every submodule (including `tokenjam.cli.main`) through a
`.pth` entry that just points back at the dev checkout — if that checkout
(e.g. a git worktree) is later deleted, `tokenjam` degrades to an empty PEP
420 namespace package and every real submodule import raises
`ModuleNotFoundError`, with nothing left on disk to run a guard from.

This module is the one piece of code guaranteed to still be on disk in that
situation, so it is the only place that can catch the failure and print an
actionable hint instead of letting a bare traceback reach the user.
"""
from __future__ import annotations

import os
import sys

_DANGLING_EDITABLE_HINT = (
    "tj's install points at a source path that no longer exists. "
    "Reinstall with `pipx reinstall tokenjam` or `pip install -e <checkout>`."
)


def _dangling_editable_install() -> bool:
    """Best-effort detection of a broken editable install.

    Must never raise: any failure here just means "not detected" so the
    caller falls back to re-raising the original ModuleNotFoundError.
    """
    try:
        import tokenjam

        if getattr(tokenjam, "__file__", None) is not None:
            # Resolved as a normal module (not a namespace package) - a real
            # install, nothing dangling.
            return False

        # `__file__ is None` means `tokenjam` only resolved as a PEP 420
        # namespace package. That alone isn't proof the editable source is
        # gone (a namespace package is also valid packaging), so also check
        # whether any contributing path actually holds the real `cli`
        # subpackage on disk.
        paths = list(getattr(tokenjam, "__path__", None) or [])
        return not any(_has_real_cli_package(path) for path in paths)
    except Exception:
        return False


def _has_real_cli_package(path: str) -> bool:
    try:
        return os.path.isfile(os.path.join(path, "cli", "main.py"))
    except Exception:
        return False


def _load_cli():
    """Import the real CLI. Split out so tests can simulate the failure
    without needing an actual broken editable install on disk."""
    from tokenjam.cli.main import cli

    return cli


def main() -> None:
    """Entry point registered as the `tj` / `tokenjam` console scripts."""
    try:
        cli = _load_cli()
    except ModuleNotFoundError:
        if _dangling_editable_install():
            sys.stderr.write(_DANGLING_EDITABLE_HINT + "\n")
            raise SystemExit(1)
        raise
    raise SystemExit(cli())
