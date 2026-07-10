"""PATH resolution guard for onboard's bare-`tj` artifacts.

Onboard installs a persistent `tj` and then writes artifacts (next-steps
nudge, statusline, the claude() shell wrapper) that invoke bare `tj` later,
in whatever shell the user happens to be in. That shell's PATH is not
guaranteed to resolve `tj` to the install onboard manages: it may have no
PATH entry for it at all, or an older `tj` earlier on PATH shadowing it
(confirmed in the wild: a VS Code integrated terminal ordered a stale pip
`tj` ahead of a freshly-installed uv-tool shim). `_probe_tj_path_resolution`
detects both cases; `_ensure_tj_on_path` best-effort fixes the "nothing
resolves" case; `_print_tj_path_warning` surfaces the "shadowed" case
explicitly since PATH reordering isn't something onboard should do silently.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from tokenjam.cli import cmd_onboard


# --- _current_tj_binary: PATH-independent resolution ------------------------


def test_current_tj_binary_prefers_interpreter_sibling(monkeypatch, tmp_path):
    sibling = tmp_path / "tj"
    sibling.write_text("#!/bin/sh\n")
    monkeypatch.setattr(cmd_onboard.sys, "executable", str(tmp_path / "python3"))
    # Even if PATH resolves to something else entirely, the sibling wins.
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: "/usr/bin/tj")
    assert cmd_onboard._current_tj_binary() == str(sibling)


def test_current_tj_binary_falls_back_to_which_when_no_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_onboard.sys, "executable", str(tmp_path / "python3"))
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: "/usr/bin/tj")
    assert cmd_onboard._current_tj_binary() == "/usr/bin/tj"


def test_current_tj_binary_falls_back_to_bare_when_nothing_resolves(monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_onboard.sys, "executable", str(tmp_path / "python3"))
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: None)
    assert cmd_onboard._current_tj_binary() == "tj"


# --- _resolve_tj_binary: daemon-install must share the same PATH-shadow ------
# --- immunity as _current_tj_binary, without going through it directly ------
#
# `_resolve_tj_binary` feeds `_install_launchd`/`_install_systemd`, which bake
# its result into a plist/unit file that persists indefinitely. It used to
# prefer `shutil.which("tj")` first, so an older/other `tj` shadowing the
# real one on PATH at the moment `tj onboard` installs the daemon got
# permanently pinned into the daemon's ProgramArguments — surviving even
# after the shadowing PATH entry was later removed. It must resolve the same
# way `_current_tj_binary` does (sibling first), but its last-resort
# fallback intentionally returns the *constructed* sibling path rather than
# a bare `"tj"`, so `_daemon_program_args`'s ephemeral uv/pipx-cache
# detection still has a real path shape to inspect.


def test_resolve_tj_binary_prefers_interpreter_sibling_over_a_shadowing_path_tj(monkeypatch, tmp_path):
    sibling = tmp_path / "tj"
    sibling.write_text("#!/bin/sh\n")
    monkeypatch.setattr(cmd_onboard.sys, "executable", str(tmp_path / "python3"))
    # Even if PATH resolves to a different (shadowing) tj, the sibling wins.
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: "/usr/bin/tj")
    assert cmd_onboard._resolve_tj_binary() == str(sibling)


def test_resolve_tj_binary_falls_back_to_which_when_no_sibling(monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_onboard.sys, "executable", str(tmp_path / "python3"))
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: "/usr/bin/tj")
    assert cmd_onboard._resolve_tj_binary() == "/usr/bin/tj"


def test_resolve_tj_binary_falls_back_to_constructed_sibling_when_nothing_resolves(monkeypatch, tmp_path):
    """Unlike `_current_tj_binary`, the final fallback is the constructed
    sibling path, NOT a bare `"tj"` — `_daemon_program_args` needs a path
    shape (e.g. containing `/uv/` or `/pipx/`) to detect an ephemeral cache
    install and route the daemon through a durable uvx/pipx shim instead."""
    sibling = tmp_path / "tj"
    monkeypatch.setattr(cmd_onboard.sys, "executable", str(tmp_path / "python3"))
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: None)
    assert cmd_onboard._resolve_tj_binary() == str(sibling)


# --- _probe_tj_path_resolution: the three states -----------------------------


def test_probe_ok_when_bare_tj_matches_expected(monkeypatch):
    monkeypatch.setattr(cmd_onboard, "_current_tj_binary", lambda: "/home/x/.local/bin/tj")
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: "/home/x/.local/bin/tj")
    status, expected, shadow = cmd_onboard._probe_tj_path_resolution()
    assert status == "ok"
    assert expected == "/home/x/.local/bin/tj"
    assert shadow is None


def test_probe_unresolved_when_nothing_on_path(monkeypatch):
    monkeypatch.setattr(cmd_onboard, "_current_tj_binary", lambda: "/home/x/.local/bin/tj")
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: None)
    status, expected, shadow = cmd_onboard._probe_tj_path_resolution()
    assert status == "unresolved"
    assert expected == "/home/x/.local/bin/tj"
    assert shadow is None


def test_probe_shadowed_when_bare_tj_resolves_elsewhere(monkeypatch):
    monkeypatch.setattr(cmd_onboard, "_current_tj_binary", lambda: "/home/x/.local/bin/tj")
    monkeypatch.setattr(
        cmd_onboard.shutil, "which",
        lambda _b: "/Library/Frameworks/Python.framework/Versions/3.13/bin/tj",
    )
    status, expected, shadow = cmd_onboard._probe_tj_path_resolution()
    assert status == "shadowed"
    assert expected == "/home/x/.local/bin/tj"
    assert shadow == "/Library/Frameworks/Python.framework/Versions/3.13/bin/tj"


# --- _ensure_tj_on_path: best-effort fix for the unresolved case ------------


def test_ensure_on_path_prefers_uv_update_shell(monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_onboard.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda b: "/usr/bin/uv" if b == "uv" else None)
    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cmd_onboard.subprocess, "run", _fake_run)
    status = cmd_onboard._ensure_tj_on_path("/home/x/.local/bin/tj")
    assert status == "ran-uv-update-shell"
    assert calls == [["uv", "tool", "update-shell"]]
    # uv covers shell profiles itself — no need to also hand-edit zshrc.
    assert not (tmp_path / ".zshrc").exists()


def test_ensure_on_path_falls_back_to_zshrc_block_when_uv_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_onboard.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: None)
    status = cmd_onboard._ensure_tj_on_path("/home/x/.local/bin/tj")
    assert status == "wrote-zshrc-block"
    text = (tmp_path / ".zshrc").read_text()
    assert cmd_onboard._ZSHRC_PATH_START in text
    assert '/home/x/.local/bin' in text


def test_ensure_on_path_falls_back_when_uv_update_shell_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_onboard.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda b: "/usr/bin/uv" if b == "uv" else None)
    monkeypatch.setattr(
        cmd_onboard.subprocess, "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"),
    )
    status = cmd_onboard._ensure_tj_on_path("/home/x/.local/bin/tj")
    assert status == "wrote-zshrc-block"
    assert cmd_onboard._ZSHRC_PATH_START in (tmp_path / ".zshrc").read_text()


def test_ensure_on_path_refuses_bare_command_name(monkeypatch, tmp_path):
    """The bare-"tj" fallback carries no directory: Path("tj").parent is "."
    and exporting `.:$PATH` would put the shell's CWD first on PATH — a
    privilege-escalation vector. Nothing may be written to zshrc."""
    monkeypatch.setattr(cmd_onboard.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: None)
    status = cmd_onboard._ensure_tj_on_path("tj")
    assert status == "no-absolute-path"
    assert not (tmp_path / ".zshrc").exists()


def test_ensure_on_path_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(cmd_onboard.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cmd_onboard.shutil, "which", lambda _b: None)
    first = cmd_onboard._ensure_tj_on_path("/home/x/.local/bin/tj")
    second = cmd_onboard._ensure_tj_on_path("/home/x/.local/bin/tj")
    assert first == "wrote-zshrc-block"
    assert second == "already-managed"
    text = (tmp_path / ".zshrc").read_text()
    assert text.count(cmd_onboard._ZSHRC_PATH_START) == 1


# --- _warn_if_tj_path_unresolved: wiring between probe, fix, and print -----


def test_warn_noop_when_ok(monkeypatch, capsys):
    monkeypatch.setattr(
        cmd_onboard, "_probe_tj_path_resolution",
        lambda: ("ok", "/home/x/.local/bin/tj", None),
    )
    cmd_onboard._warn_if_tj_path_unresolved()
    assert capsys.readouterr().out == ""


def test_warn_fixes_and_prints_when_unresolved(monkeypatch, capsys):
    monkeypatch.setattr(
        cmd_onboard, "_probe_tj_path_resolution",
        lambda: ("unresolved", "/home/x/.local/bin/tj", None),
    )
    fix_calls = []

    def _fake_ensure(expected):
        fix_calls.append(expected)
        return "wrote-zshrc-block"

    monkeypatch.setattr(cmd_onboard, "_ensure_tj_on_path", _fake_ensure)
    cmd_onboard._warn_if_tj_path_unresolved()
    assert fix_calls == ["/home/x/.local/bin/tj"]
    out = capsys.readouterr().out
    assert "Heads up" in out
    assert "/home/x/.local/bin/tj" in out
    assert "Fixed for next time" in out


def test_warn_does_not_fix_when_shadowed(monkeypatch, capsys):
    monkeypatch.setattr(
        cmd_onboard, "_probe_tj_path_resolution",
        lambda: (
            "shadowed",
            "/home/x/.local/bin/tj",
            "/Library/Frameworks/Python.framework/Versions/3.13/bin/tj",
        ),
    )
    fix_calls = []
    monkeypatch.setattr(cmd_onboard, "_ensure_tj_on_path", lambda expected: fix_calls.append(expected))
    monkeypatch.setattr(cmd_onboard, "_tj_binary_version", lambda _p: "tj, version 0.5.4")

    cmd_onboard._warn_if_tj_path_unresolved()

    assert fix_calls == []  # never auto-reorders PATH
    out = capsys.readouterr().out
    assert "Heads up" in out
    assert "tj, version 0.5.4" in out
    assert "/Library/Frameworks/Python.framework/Versions/3.13/bin/tj" in out
    assert "/home/x/.local/bin/tj" in out


# --- _claude_wrapper_block: absolute path, not bare `tj` --------------------


def test_wrapper_block_uses_absolute_tj_path(monkeypatch):
    monkeypatch.setattr(cmd_onboard, "_current_tj_binary", lambda: "/home/x/.local/bin/tj")
    block = cmd_onboard._claude_wrapper_block()
    assert '"/home/x/.local/bin/tj" otel-resource-attrs' in block
    assert '"/home/x/.local/bin/tj" session-end --instance' in block
    # Never leaves a bare, unqualified `tj ` invocation that a shadowing
    # install on PATH could hijack.
    assert " tj otel-resource-attrs" not in block
    assert " tj session-end" not in block


def test_wrapper_block_handles_path_with_spaces(monkeypatch):
    monkeypatch.setattr(cmd_onboard, "_current_tj_binary", lambda: "/Users/a b/.local/bin/tj")
    block = cmd_onboard._claude_wrapper_block()
    assert '"/Users/a b/.local/bin/tj"' in block


# --- _claude_wrapper_block: stderr silenced, no-leading-comma fallback ------
#
# A stale/shadowed/half-uninstalled `tj` must never flash a traceback into
# every `claude` launch, and a failed or empty `otel-resource-attrs` lookup
# must never produce a malformed `,service.instance.id=...` value.


def test_wrapper_block_silences_otel_attrs_stderr(monkeypatch):
    monkeypatch.setattr(cmd_onboard, "_current_tj_binary", lambda: "/home/x/.local/bin/tj")
    block = cmd_onboard._claude_wrapper_block()
    assert '"/home/x/.local/bin/tj" otel-resource-attrs 2>/dev/null' in block
    # The capture must be failure-tolerant (never aborts the wrapper / lets a
    # nonzero tj exit status propagate into the claude launch).
    assert '_tj_attrs="$("/home/x/.local/bin/tj" otel-resource-attrs 2>/dev/null)" || _tj_attrs=""' in block


def test_wrapper_block_falls_back_without_leading_comma(monkeypatch):
    monkeypatch.setattr(cmd_onboard, "_current_tj_binary", lambda: "/home/x/.local/bin/tj")
    block = cmd_onboard._claude_wrapper_block()
    # Old (buggy) shape: a single export directly concatenating the raw
    # command substitution, which produces a leading comma when it's empty.
    assert '),service.instance.id=$_tj_inst"' not in block
    # New shape: branch on whether the lookup produced anything.
    assert 'export OTEL_RESOURCE_ATTRIBUTES="$_tj_attrs,service.instance.id=$_tj_inst"' in block
    assert 'export OTEL_RESOURCE_ATTRIBUTES="service.instance.id=$_tj_inst"' in block


def test_wrapper_block_survives_broken_tj_end_to_end(tmp_path, monkeypatch):
    """Empirical: source the generated block in a real zsh with a
    deliberately broken `tj` first on PATH and confirm `claude` launches
    with zero extra output and a well-formed OTEL_RESOURCE_ATTRIBUTES."""
    import os
    import stat

    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh not available")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def _make_executable(path, content):
        path.write_text(content)
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    broken_tj = bin_dir / "tj"
    _make_executable(
        broken_tj,
        "#!/bin/sh\n"
        "echo 'Traceback (most recent call last):' >&2\n"
        "echo 'ModuleNotFoundError: no module named tokenjam' >&2\n"
        "exit 1\n",
    )
    fake_claude = bin_dir / "claude"
    _make_executable(fake_claude, "#!/bin/sh\necho claude-ran\n")

    monkeypatch.setattr(cmd_onboard, "_current_tj_binary", lambda: str(broken_tj))
    block = cmd_onboard._claude_wrapper_block()

    script = tmp_path / "rc.zsh"
    script.write_text(block + "\nclaude\necho \"ATTRS=$OTEL_RESOURCE_ATTRIBUTES\"\n")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    result = subprocess.run(
        [zsh, str(script)], capture_output=True, text=True, env=env, timeout=10,
    )

    assert result.stderr == ""  # the broken tj's traceback must never leak
    assert "Traceback" not in result.stdout
    assert "claude-ran" in result.stdout

    [attrs_line] = [line for line in result.stdout.splitlines() if line.startswith("ATTRS=")]
    attrs = attrs_line[len("ATTRS="):]
    assert not attrs.startswith(",")
    assert attrs == "service.instance.id=unknown"
