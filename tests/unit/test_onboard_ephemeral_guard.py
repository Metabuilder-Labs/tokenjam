"""Ephemeral-runner guard (#120).

`npx tokenjam onboard` delegates to `uvx --from tokenjam tj onboard` (or
`pipx run --spec tokenjam tj onboard`), both of which resolve `sys.executable`
into a throwaway venv that is not kept on PATH once the process exits. Onboard
wires a background daemon and a Claude Code statusline that both invoke `tj`
afterward — under an ephemeral runner those references go stale the moment the
session ends. `_maybe_guard_ephemeral_runner` detects that situation and offers
(or performs) a persistent install, re-execing onboard through it.
"""
from __future__ import annotations

import subprocess

import click
import pytest

from tokenjam.cli import cmd_onboard


def _ctx() -> click.Context:
    return click.Context(cmd_onboard.cmd_onboard)


# --- _is_ephemeral_runner: path-signature detection --------------------------


@pytest.mark.parametrize(
    "executable,expected",
    [
        ("/Users/x/.local/share/uv/tools/tokenjam/bin/python", False),
        ("/Users/x/.local/share/pipx/venvs/tokenjam/bin/python", False),
        ("/Users/x/.cache/uv/archive-v0/abc123/bin/python", True),
        ("/Users/x/.local/share/pipx/.cache/xyz/bin/python", True),
        ("/usr/bin/python3", False),
        ("/Users/x/project/.venv/bin/python", False),
    ],
)
def test_is_ephemeral_runner_by_executable_path(monkeypatch, executable, expected):
    monkeypatch.setattr(cmd_onboard.sys, "executable", executable)
    assert cmd_onboard._is_ephemeral_runner() is expected


# --- _maybe_guard_ephemeral_runner: no-op for the common case ---------------


def test_noop_when_not_ephemeral(monkeypatch, capsys):
    monkeypatch.setattr(cmd_onboard, "_is_ephemeral_runner", lambda: False)
    cmd_onboard._maybe_guard_ephemeral_runner(_ctx())
    assert capsys.readouterr().out == ""


# --- Non-interactive: warn and continue in-process --------------------------


def test_non_interactive_warns_and_continues(monkeypatch, capsys):
    monkeypatch.setattr(cmd_onboard, "_is_ephemeral_runner", lambda: True)
    monkeypatch.setattr(cmd_onboard, "_is_interactive", lambda: False)
    cmd_onboard._maybe_guard_ephemeral_runner(_ctx())
    out = capsys.readouterr().out
    assert "Heads up" in out
    assert "Non-interactive" in out


# --- Interactive: declined install continues in-process ---------------------


def test_interactive_decline_continues(monkeypatch, capsys):
    monkeypatch.setattr(cmd_onboard, "_is_ephemeral_runner", lambda: True)
    monkeypatch.setattr(cmd_onboard, "_is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *a, **k: False)
    cmd_onboard._maybe_guard_ephemeral_runner(_ctx())
    out = capsys.readouterr().out
    assert "Continuing without a persistent install" in out


# --- Interactive: install failure continues in-process -----------------------


def test_interactive_install_failure_continues(monkeypatch, capsys):
    monkeypatch.setattr(cmd_onboard, "_is_ephemeral_runner", lambda: True)
    monkeypatch.setattr(cmd_onboard, "_is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(cmd_onboard, "_install_tokenjam_persistently", lambda: None)
    cmd_onboard._maybe_guard_ephemeral_runner(_ctx())
    out = capsys.readouterr().out
    assert "Persistent install failed" in out


# --- Interactive: successful install re-execs and exits ---------------------


def test_interactive_success_reexecs_and_exits(monkeypatch, capsys):
    monkeypatch.setattr(cmd_onboard, "_is_ephemeral_runner", lambda: True)
    monkeypatch.setattr(cmd_onboard, "_is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(
        cmd_onboard, "_install_tokenjam_persistently",
        lambda: "/home/x/.local/bin/tj",
    )
    monkeypatch.setattr(cmd_onboard.sys, "argv", ["tj", "onboard", "--claude-code"])

    calls = {}

    def _fake_run(cmd):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cmd_onboard.subprocess, "run", _fake_run)

    with pytest.raises(click.exceptions.Exit) as exc_info:
        cmd_onboard._maybe_guard_ephemeral_runner(_ctx())

    assert exc_info.value.exit_code == 0
    assert calls["cmd"] == ["/home/x/.local/bin/tj", "onboard", "--claude-code"]
    out = capsys.readouterr().out
    assert "re-running onboard" in out


# --- _install_tokenjam_persistently: runner selection + idempotence ---------


def test_install_prefers_uv_over_pipx(monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_onboard, "_LOCAL_BIN_DIR", tmp_path)
    (tmp_path / "tj").write_text("#!/bin/sh\n")
    monkeypatch.setattr(
        cmd_onboard.shutil, "which",
        lambda bin_: f"/usr/bin/{bin_}" if bin_ in ("uv", "pipx") else None,
    )
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cmd_onboard.subprocess, "run", _fake_run)
    result = cmd_onboard._install_tokenjam_persistently()
    assert result == str(tmp_path / "tj")
    assert calls[0][0] == "uv"


def test_install_falls_back_to_pipx_when_uv_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_onboard, "_LOCAL_BIN_DIR", tmp_path)
    (tmp_path / "tj").write_text("#!/bin/sh\n")
    monkeypatch.setattr(
        cmd_onboard.shutil, "which",
        lambda bin_: "/usr/bin/pipx" if bin_ == "pipx" else None,
    )
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cmd_onboard.subprocess, "run", _fake_run)
    result = cmd_onboard._install_tokenjam_persistently()
    assert result == str(tmp_path / "tj")
    assert calls[0][0] == "pipx"


def test_install_falls_back_to_path_when_bin_dir_is_customized(monkeypatch, tmp_path):
    """`UV_TOOL_BIN_DIR` / `PIPX_BIN_DIR` (or a customized install prefix) can
    place `tj` somewhere other than the default `~/.local/bin` — the default
    dir check must not be the only signal, or a successful install there gets
    reported as failed and the guard falls through to the ephemeral path it
    exists to avoid (Greptile P1)."""
    monkeypatch.setattr(cmd_onboard, "_LOCAL_BIN_DIR", tmp_path)  # empty: no tj here
    custom_tj = "/opt/custom/bin/tj"
    monkeypatch.setattr(
        cmd_onboard.shutil, "which",
        lambda bin_: {"uv": "/usr/bin/uv", "tj": custom_tj}.get(bin_),
    )
    monkeypatch.setattr(
        cmd_onboard.subprocess, "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    result = cmd_onboard._install_tokenjam_persistently()
    assert result == custom_tj


def test_install_returns_none_when_no_runner_available(monkeypatch):
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda bin_: None)
    assert cmd_onboard._install_tokenjam_persistently() is None


def test_install_returns_none_when_binary_never_appears(monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_onboard, "_LOCAL_BIN_DIR", tmp_path)
    monkeypatch.setattr(
        cmd_onboard.shutil, "which",
        lambda bin_: "/usr/bin/uv" if bin_ == "uv" else None,
    )
    monkeypatch.setattr(
        cmd_onboard.subprocess, "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""),
    )
    assert cmd_onboard._install_tokenjam_persistently() is None
