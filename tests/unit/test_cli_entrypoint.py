"""Tests for the `tj` / `tokenjam` console-script entrypoint.

Covers the dangling-editable-install case: an editable install whose source
checkout (e.g. a git worktree) has been deleted degrades `tokenjam` to an
empty PEP 420 namespace package (`tokenjam.__file__ is None`, no real `cli`
subpackage on any contributing path), so every real submodule import raises
ModuleNotFoundError. The entrypoint must detect that specific condition and
print a one-line repair hint instead of letting a bare traceback surface.
"""
from __future__ import annotations

import sys
import types

import pytest

from tokenjam.cli import entrypoint


def _namespace_module(path: str) -> types.ModuleType:
    """Build a fake module object shaped like a broken PEP 420 namespace
    package: no __file__, __path__ pointing at the given directory."""
    module = types.ModuleType("tokenjam")
    module.__file__ = None
    module.__path__ = [path]
    return module


def test_dangling_editable_install_true_when_source_gone(tmp_path, monkeypatch):
    # tmp_path has no `cli/main.py` - simulates the deleted-worktree case.
    monkeypatch.setitem(sys.modules, "tokenjam", _namespace_module(str(tmp_path)))

    assert entrypoint._dangling_editable_install() is True


def test_dangling_editable_install_false_when_cli_package_present(tmp_path, monkeypatch):
    # A namespace package whose path DOES still hold a real cli/main.py is
    # not the dangling-install case (defensive: don't misfire on a benign
    # namespace-package layout).
    cli_dir = tmp_path / "cli"
    cli_dir.mkdir()
    (cli_dir / "main.py").write_text("")
    monkeypatch.setitem(sys.modules, "tokenjam", _namespace_module(str(tmp_path)))

    assert entrypoint._dangling_editable_install() is False


def test_dangling_editable_install_false_for_real_module(monkeypatch):
    real_module = types.ModuleType("tokenjam")
    real_module.__file__ = "/some/real/install/tokenjam/__init__.py"
    monkeypatch.setitem(sys.modules, "tokenjam", real_module)

    assert entrypoint._dangling_editable_install() is False


def test_dangling_editable_install_never_raises_when_tokenjam_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "tokenjam", raising=False)

    def _boom(name, *args, **kwargs):
        raise ModuleNotFoundError("No module named 'tokenjam'")

    monkeypatch.setattr("builtins.__import__", _boom)

    assert entrypoint._dangling_editable_install() is False


def test_main_prints_repair_hint_and_exits_1_on_dangling_install(monkeypatch, capsys):
    def _fail_load_cli():
        raise ModuleNotFoundError("No module named 'tokenjam.cli'")

    monkeypatch.setattr(entrypoint, "_load_cli", _fail_load_cli)
    monkeypatch.setattr(entrypoint, "_dangling_editable_install", lambda: True)

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "no longer exists" in captured.err
    assert "pipx reinstall tokenjam" in captured.err
    assert "Traceback" not in captured.err


def test_main_reraises_unrelated_module_not_found_error(monkeypatch):
    # A ModuleNotFoundError from some OTHER cause (e.g. a genuinely missing
    # dependency) must not be swallowed as if it were the dangling-install
    # case - only the specific namespace-package signal should be caught.
    def _fail_load_cli():
        raise ModuleNotFoundError("No module named 'some_unrelated_dependency'")

    monkeypatch.setattr(entrypoint, "_load_cli", _fail_load_cli)
    monkeypatch.setattr(entrypoint, "_dangling_editable_install", lambda: False)

    with pytest.raises(ModuleNotFoundError, match="some_unrelated_dependency"):
        entrypoint.main()


def test_main_runs_cli_normally_when_import_succeeds(monkeypatch):
    monkeypatch.setattr(entrypoint, "_load_cli", lambda: (lambda: 0))

    with pytest.raises(SystemExit) as exc_info:
        entrypoint.main()

    assert exc_info.value.code == 0
