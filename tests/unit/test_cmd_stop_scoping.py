"""`tj stop` must only ever touch the `tj serve` daemon started under THIS
install's $HOME -- never another install/worktree's daemon that happens to
be running on the same machine.

The old `_find_serve_pid()` was a bare `pgrep -f "tokenjam\\.serve|tj
serve|uv run tj serve"` with no scoping at all: any process on the machine
whose command line matched that pattern got SIGTERM'd, regardless of which
install started it. `TestCrossInstallIsolation` below spawns a REAL,
detached child process (not a mock) with a command line that matches the
old pattern, so a regression back to machine-wide pgrep gets caught for
real, not just in a mock's argument list.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from tokenjam.cli.cmd_stop import stop_tj_serve
from tokenjam.core.server_state import find_own_serve_pid


def _spawn_orphan_serve() -> int:
    """Spawn a real, detached process -- reparented off of the test process
    the moment the wrapping shell exits, exactly like a real orphaned `tj
    serve` foreground process, so nothing here is responsible for reaping
    it (avoids the zombie-stays-"alive"-to-os.kill artifact of a direct
    subprocess.Popen child).

    Its command line contains the literal tokens "tj" "serve", so a bare
    `pgrep -f "tj serve"` still matches it the same way it would match a
    real `tj serve` invocation.
    """
    result = subprocess.run(
        [
            "/bin/sh", "-c",
            f"{sys.executable} -c 'import time; time.sleep(30)' tj serve "
            "</dev/null >/dev/null 2>&1 & echo $!",
        ],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _cleanup(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _write_state(home: Path, pid: int, port: int = 7391) -> None:
    state_path = home / ".local" / "share" / "tj" / "server.state"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "config_path": str(home / ".config" / "tj" / "config.toml"),
        "port": port,
        "pid": pid,
    }))


class TestCrossInstallIsolation:
    def test_stop_from_other_install_does_not_kill_this_install_daemon(
        self, tmp_path, monkeypatch,
    ):
        home_a = tmp_path / "home-a"
        home_b = tmp_path / "home-b"
        home_a.mkdir()
        home_b.mkdir()

        pid_a = _spawn_orphan_serve()
        try:
            _write_state(home_a, pid_a)
            # Install B has no server.state -- nothing running there.

            monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: home_b)
            stopped, stopped_via = stop_tj_serve(quiet=True)

            assert stopped is False
            assert stopped_via == []
            assert _is_alive(pid_a), (
                "tj stop from install B signaled install A's daemon -- "
                "this is the cross-install kill bug."
            )
        finally:
            _cleanup(pid_a)

    def test_stop_from_same_install_stops_its_own_daemon(self, tmp_path, monkeypatch):
        home_a = tmp_path / "home-a"
        home_a.mkdir()

        pid_a = _spawn_orphan_serve()
        try:
            _write_state(home_a, pid_a)
            monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: home_a)

            stopped, stopped_via = stop_tj_serve(quiet=True)

            assert stopped is True
            assert any(str(pid_a) in entry for entry in stopped_via)
            assert not _is_alive(pid_a)
        finally:
            _cleanup(pid_a)


class TestStaleStateFile:
    def test_dead_pid_in_state_file_is_treated_as_not_running(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        # A PID that (almost certainly) doesn't exist: spawn+wait, then reuse
        # its now-dead PID.
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait(timeout=5)

        _write_state(home, dead.pid)
        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: home)

        stopped, stopped_via = stop_tj_serve(quiet=True)

        assert stopped is False
        assert stopped_via == []

    def test_find_own_serve_pid_none_when_state_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tokenjam.core.server_state.Path.home", lambda: tmp_path / "no-state-here",
        )
        assert find_own_serve_pid() is None


class TestNoDaemonRunning:
    def test_reports_not_running_with_no_state_file_and_no_plist(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: home)

        stopped, stopped_via = stop_tj_serve(quiet=True)

        assert stopped is False
        assert stopped_via == []
