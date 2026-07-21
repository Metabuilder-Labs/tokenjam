"""`tj stop` must only ever touch the `tj serve` daemon started under THIS
install's $HOME -- never another install/worktree's daemon that happens to
be running on the same machine.

The old `_find_serve_pid()` was a bare `pgrep -f "tokenjam\\.serve|tj
serve|uv run tj serve"` with no scoping at all: any process on the machine
whose command line matched that pattern got SIGTERM'd, regardless of which
install started it. `TestCrossInstallIsolation` below spawns a REAL child
process (not a mock) with a command line that matches the old pattern, so a
regression back to machine-wide pgrep gets caught for real, not just in a
mock's argument list.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from tokenjam.cli.cmd_stop import stop_tj_serve
from tokenjam.core.server_state import find_own_serve_pid


def _spawn_serve_lookalike() -> subprocess.Popen[bytes]:
    """Spawn a real process (not a mock) whose command line contains the
    literal tokens "tj" "serve", so a bare `pgrep -f "tj serve"` still
    matches it the same way it would match a real `tj serve` invocation.

    Kept as a genuine, un-detached child of the test process -- NOT
    backgrounded off of a wrapping shell -- so the test can reap its exit
    deterministically instead of relying on the OS. A detached process
    reparents to whatever subreaper owns the job once its wrapping shell
    exits; on GitHub Actions that's the runner's own "clean up orphan
    processes" feature, which only sweeps orphans at job completion, not in
    real time. Until swept, a killed-but-unreaped orphan is a zombie, and a
    zombie's PID stays allocated -- `os.kill(pid, 0)` keeps reporting it
    "alive" long after it actually died, which is exactly what made this
    test flap to a false `stopped=False` in CI (macOS's launchd, by
    contrast, reaps orphans immediately, which is why a detached process
    never showed the bug locally).

    Being a direct child isn't enough on its own, though: Python only reaps
    a child's exit status when something calls `wait()`/`poll()` on it, and
    nothing does that during `stop_tj_serve`'s `os.kill(pid, 0)` polling
    loop. So a background thread blocks on `Popen.wait()` here -- a real
    `waitpid()` call completes the instant the process actually dies, with
    no polling delay, so by the next `os.kill(pid, 0)` check the PID is
    already fully freed.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "tj", "serve"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    threading.Thread(target=proc.wait, daemon=True).start()
    return proc


def _is_alive(proc: subprocess.Popen[bytes]) -> bool:
    return proc.poll() is None


def _cleanup(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


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

        proc_a = _spawn_serve_lookalike()
        try:
            _write_state(home_a, proc_a.pid)
            # Install B has no server.state -- nothing running there.

            monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: home_b)
            stopped, stopped_via = stop_tj_serve(quiet=True)

            assert stopped is False
            assert stopped_via == []
            assert _is_alive(proc_a), (
                "tj stop from install B signaled install A's daemon -- "
                "this is the cross-install kill bug."
            )
        finally:
            _cleanup(proc_a)

    def test_stop_from_same_install_stops_its_own_daemon(self, tmp_path, monkeypatch):
        home_a = tmp_path / "home-a"
        home_a.mkdir()

        proc_a = _spawn_serve_lookalike()
        try:
            _write_state(home_a, proc_a.pid)
            monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: home_a)

            stopped, stopped_via = stop_tj_serve(quiet=True)

            assert stopped is True
            assert any(str(proc_a.pid) in entry for entry in stopped_via)
            assert not _is_alive(proc_a)
        finally:
            _cleanup(proc_a)


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
