"""Unit tests for `tj stop` lifecycle behavior."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tokenjam.cli.cmd_stop import cmd_stop


class TestStopSweepsForegroundProcesses:
    """`tj stop` must reap orphan foreground `tj serve &` processes after
    a successful launchctl unload — otherwise port 7391 stays held and
    "tj stop didn't actually stop tj" (C6)."""

    def test_kills_foreground_after_launchd_unload(self, tmp_path, monkeypatch):
        # Pretend a plist exists so the launchd branch runs.
        plist = tmp_path / "Library" / "LaunchAgents" / "com.tokenjam.serve.plist"
        plist.parent.mkdir(parents=True)
        plist.write_text("<plist/>")
        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: tmp_path)

        # First call: launchctl unload (returncode=0). Subsequent calls:
        # _find_serve_pid is patched separately below (PID-file lookup).
        run_mock = MagicMock(return_value=MagicMock(returncode=0))
        kill_mock = MagicMock()
        # _find_serve_pid returns a PID once, then None to break the loop.
        find_pid_mock = MagicMock(side_effect=[12345, None])

        with patch("tokenjam.cli.cmd_stop.subprocess.run", run_mock), \
             patch("tokenjam.cli.cmd_stop.os.kill", kill_mock), \
             patch("tokenjam.cli.cmd_stop._find_serve_pid", find_pid_mock), \
             patch("tokenjam.cli.cmd_stop._wait_for_exit", return_value=True):
            result = CliRunner().invoke(cmd_stop, [], obj={})

        assert result.exit_code == 0, result.output
        # The foreground PID must have been signaled.
        kill_mock.assert_called_once_with(12345, 15)  # SIGTERM = 15
        # Output mentions both stop methods
        assert "launchd daemon unloaded" in result.output
        assert "PID 12345" in result.output

    def test_does_not_loop_on_slow_shutdown(self, tmp_path, monkeypatch):
        """SIGTERM is async — if the target process hasn't exited before the
        next lookup, the sweep must NOT re-signal the same PID. Otherwise a
        slow shutdown handler can make `tj stop` hang forever."""
        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: tmp_path)
        kill_mock = MagicMock()
        # The PID-file lookup keeps finding the same PID — simulates a
        # process whose SIGTERM handler hasn't completed yet.
        find_pid_mock = MagicMock(side_effect=[12345] * 50)

        with patch("tokenjam.cli.cmd_stop.os.kill", kill_mock), \
             patch("tokenjam.cli.cmd_stop._find_serve_pid", find_pid_mock), \
             patch("tokenjam.cli.cmd_stop._wait_for_exit", return_value=True):
            result = CliRunner().invoke(cmd_stop, [], obj={})

        assert result.exit_code == 0
        # SIGTERM sent exactly once even though the lookup saw the PID 50 times.
        assert kill_mock.call_count == 1
        kill_mock.assert_called_once_with(12345, 15)

    def test_reports_not_running_when_nothing_to_stop(self, tmp_path, monkeypatch):
        # No plist, no systemd unit, no foreground process.
        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: tmp_path)
        with patch("tokenjam.cli.cmd_stop._find_serve_pid", return_value=None):
            result = CliRunner().invoke(cmd_stop, [], obj={})
        assert result.exit_code == 0
        assert "not running" in result.output

    def test_does_not_claim_stop_when_plist_exists_but_not_loaded(
        self, tmp_path, monkeypatch,
    ):
        """A plist FILE existing on disk (e.g. left over from a prior
        install) must not be reported as "stopped" unless launchd actually
        had the label loaded — otherwise onboard's already-running skip
        (which keys off a truthful stop signal) is unreachable (regression
        for the daemon restarting on every onboard run)."""
        from tokenjam.cli.cmd_stop import stop_tj_serve

        plist = tmp_path / "Library" / "LaunchAgents" / "com.tokenjam.serve.plist"
        plist.parent.mkdir(parents=True)
        plist.write_text("<plist/>")
        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: tmp_path)

        # `launchctl list com.tokenjam.serve` fails (not loaded); no
        # `unload` call should ever be issued as a result.
        run_mock = MagicMock(return_value=MagicMock(returncode=1))

        with patch("tokenjam.cli.cmd_stop.subprocess.run", run_mock), \
             patch("tokenjam.cli.cmd_stop._find_serve_pid", return_value=None):
            stopped, stopped_via = stop_tj_serve(quiet=True)

        assert stopped is False
        assert stopped_via == []
        calls = [c.args[0] for c in run_mock.call_args_list]
        assert any("list" in c for c in calls), "must check load state first"
        assert not any("unload" in c for c in calls), (
            f"must not unload a plist that was never loaded; calls were {calls}"
        )


    def test_does_not_claim_stop_when_systemd_unit_exists_but_inactive(
        self, tmp_path, monkeypatch,
    ):
        """Same story as the launchd case, for the systemd (Linux) path:
        a unit file existing is not evidence the daemon was ever active."""
        from tokenjam.cli.cmd_stop import stop_tj_serve

        systemd_unit = tmp_path / ".config" / "systemd" / "user" / "tokenjam.service"
        systemd_unit.parent.mkdir(parents=True)
        systemd_unit.write_text("[Unit]\n")
        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: tmp_path)

        run_mock = MagicMock(return_value=MagicMock(returncode=3, stdout="inactive\n"))

        with patch("tokenjam.cli.cmd_stop.subprocess.run", run_mock), \
             patch("tokenjam.cli.cmd_stop._find_serve_pid", return_value=None):
            stopped, stopped_via = stop_tj_serve(quiet=True)

        assert stopped is False
        assert stopped_via == []
        calls = [c.args[0] for c in run_mock.call_args_list]
        assert any("is-active" in c for c in calls)
        assert not any("disable" in c for c in calls)

    def test_stops_systemd_unit_that_is_activating(self, tmp_path, monkeypatch):
        """`systemctl is-active` reports "activating" (and "reloading") for a
        unit that is genuinely live but mid-transition -- not settled
        "active" yet. Those states must still count as running, or
        `stop_tj_serve` declines to stop a daemon that's actually there and
        misreports it as not-running (the same class of bug as the
        plist-exists-but-never-loaded case, just inverted: false negative
        instead of false positive)."""
        from tokenjam.cli.cmd_stop import stop_tj_serve

        systemd_unit = tmp_path / ".config" / "systemd" / "user" / "tokenjam.service"
        systemd_unit.parent.mkdir(parents=True)
        systemd_unit.write_text("[Unit]\n")
        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: tmp_path)

        run_mock = MagicMock(side_effect=[
            MagicMock(returncode=3, stdout="activating\n"),  # is-active
            MagicMock(returncode=0, stdout=""),               # disable --now
        ])

        with patch("tokenjam.cli.cmd_stop.subprocess.run", run_mock), \
             patch("tokenjam.cli.cmd_stop._find_serve_pid", return_value=None):
            stopped, stopped_via = stop_tj_serve(quiet=True)

        assert stopped is True
        assert stopped_via == ["systemd service stopped"]
        calls = [c.args[0] for c in run_mock.call_args_list]
        assert any("is-active" in c for c in calls)
        assert any("disable" in c for c in calls), (
            f"an activating unit must still be disabled/stopped; calls were {calls}"
        )


class TestSystemdUnitActive:
    """Direct tests of the live-state classifier used to gate the systemd
    stop path."""

    def test_treats_activating_as_live(self, monkeypatch):
        from tokenjam.cli.cmd_stop import _systemd_unit_active
        result_mock = MagicMock(returncode=3, stdout="activating\n")
        with patch("tokenjam.cli.cmd_stop.subprocess.run", return_value=result_mock):
            assert _systemd_unit_active("tokenjam") is True

    def test_treats_reloading_as_live(self, monkeypatch):
        from tokenjam.cli.cmd_stop import _systemd_unit_active
        result_mock = MagicMock(returncode=0, stdout="reloading\n")
        with patch("tokenjam.cli.cmd_stop.subprocess.run", return_value=result_mock):
            assert _systemd_unit_active("tokenjam") is True

    def test_treats_active_as_live(self, monkeypatch):
        from tokenjam.cli.cmd_stop import _systemd_unit_active
        result_mock = MagicMock(returncode=0, stdout="active\n")
        with patch("tokenjam.cli.cmd_stop.subprocess.run", return_value=result_mock):
            assert _systemd_unit_active("tokenjam") is True

    def test_treats_inactive_as_not_live(self, monkeypatch):
        from tokenjam.cli.cmd_stop import _systemd_unit_active
        result_mock = MagicMock(returncode=3, stdout="inactive\n")
        with patch("tokenjam.cli.cmd_stop.subprocess.run", return_value=result_mock):
            assert _systemd_unit_active("tokenjam") is False

    def test_treats_failed_as_not_live(self, monkeypatch):
        from tokenjam.cli.cmd_stop import _systemd_unit_active
        result_mock = MagicMock(returncode=3, stdout="failed\n")
        with patch("tokenjam.cli.cmd_stop.subprocess.run", return_value=result_mock):
            assert _systemd_unit_active("tokenjam") is False


class TestStopVerifiesTermination:
    """`tj stop` must not claim success for a process that's still alive.
    It has to observe the exit (escalating to SIGKILL once) before it does."""

    def test_escalates_to_sigkill_when_sigterm_does_not_land(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: tmp_path)
        kill_mock = MagicMock()
        find_pid_mock = MagicMock(side_effect=[12345, None])
        # SIGTERM's wait fails (still alive); SIGKILL's wait succeeds.
        wait_mock = MagicMock(side_effect=[False, True])

        with patch("tokenjam.cli.cmd_stop.os.kill", kill_mock), \
             patch("tokenjam.cli.cmd_stop._find_serve_pid", find_pid_mock), \
             patch("tokenjam.cli.cmd_stop._wait_for_exit", wait_mock):
            result = CliRunner().invoke(cmd_stop, [], obj={})

        assert result.exit_code == 0, result.output
        assert kill_mock.call_count == 2
        kill_mock.assert_any_call(12345, 15)  # SIGTERM
        kill_mock.assert_any_call(12345, 9)   # SIGKILL
        assert "PID 12345 (SIGKILL)" in result.output
        assert "tj serve stopped" in result.output

    def test_reports_failure_honestly_when_process_survives_sigkill(
        self, tmp_path, monkeypatch,
    ):
        """A process that resists even SIGKILL (e.g. stuck in uninterruptible
        I/O) must never be reported as stopped."""
        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: tmp_path)
        kill_mock = MagicMock()
        find_pid_mock = MagicMock(side_effect=[12345, None])
        wait_mock = MagicMock(return_value=False)  # never confirmed dead

        with patch("tokenjam.cli.cmd_stop.os.kill", kill_mock), \
             patch("tokenjam.cli.cmd_stop._find_serve_pid", find_pid_mock), \
             patch("tokenjam.cli.cmd_stop._wait_for_exit", wait_mock):
            result = CliRunner().invoke(cmd_stop, [], obj={})

        assert result.exit_code == 0
        assert "tj serve did not stop" in result.output
        assert "PID 12345" in result.output
        assert "tj serve stopped" not in result.output

    def test_stop_tj_serve_returns_false_when_termination_unconfirmed(
        self, tmp_path, monkeypatch,
    ):
        from tokenjam.cli.cmd_stop import stop_tj_serve

        monkeypatch.setattr("tokenjam.cli.cmd_stop.Path.home", lambda: tmp_path)
        find_pid_mock = MagicMock(side_effect=[12345, None])

        with patch("tokenjam.cli.cmd_stop.os.kill", MagicMock()), \
             patch("tokenjam.cli.cmd_stop._find_serve_pid", find_pid_mock), \
             patch("tokenjam.cli.cmd_stop._wait_for_exit", return_value=False):
            stopped, stopped_via = stop_tj_serve(quiet=True)

        assert stopped is False
        assert stopped_via == []


class TestWaitForExit:
    """Direct tests of the polling primitive, with fake sleep/monotonic so
    nothing here waits real time."""

    def test_returns_true_as_soon_as_pid_is_gone(self):
        from tokenjam.cli.cmd_stop import _wait_for_exit

        alive_calls = MagicMock(side_effect=[True, True, False])
        sleep_mock = MagicMock()
        with patch("tokenjam.cli.cmd_stop._pid_alive", alive_calls):
            result = _wait_for_exit(
                12345, timeout_s=5.0, sleep=sleep_mock, monotonic=MagicMock(return_value=0.0),
            )

        assert result is True
        assert sleep_mock.call_count == 2

    def test_returns_false_once_timeout_elapses(self):
        from tokenjam.cli.cmd_stop import _wait_for_exit

        # Always alive; monotonic jumps straight past the timeout on the
        # second read, so this returns immediately without a real loop.
        monotonic_mock = MagicMock(side_effect=[0.0, 10.0])
        with patch("tokenjam.cli.cmd_stop._pid_alive", return_value=True):
            result = _wait_for_exit(
                12345, timeout_s=1.0, sleep=MagicMock(), monotonic=monotonic_mock,
            )

        assert result is False
