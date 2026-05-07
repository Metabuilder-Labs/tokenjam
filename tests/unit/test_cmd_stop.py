"""Unit tests for `ocw stop` lifecycle behavior."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tj.cli.cmd_stop import cmd_stop


class TestStopSweepsForegroundProcesses:
    """`ocw stop` must reap orphan foreground `ocw serve &` processes after
    a successful launchctl unload — otherwise port 7391 stays held and
    "ocw stop didn't actually stop ocw" (C6)."""

    def test_kills_foreground_after_launchd_unload(self, tmp_path, monkeypatch):
        # Pretend a plist exists so the launchd branch runs.
        plist = tmp_path / "Library" / "LaunchAgents" / "com.tokenjam.serve.plist"
        plist.parent.mkdir(parents=True)
        plist.write_text("<plist/>")
        monkeypatch.setattr("tj.cli.cmd_stop.Path.home", lambda: tmp_path)

        # First call: launchctl unload (returncode=0). Subsequent calls:
        # _find_serve_pid uses subprocess too — but it's gated behind
        # _find_serve_pid which we'll patch separately.
        run_mock = MagicMock(return_value=MagicMock(returncode=0))
        kill_mock = MagicMock()
        # _find_serve_pid returns a PID once, then None to break the loop.
        find_pid_mock = MagicMock(side_effect=[12345, None])

        with patch("tj.cli.cmd_stop.subprocess.run", run_mock), \
             patch("tj.cli.cmd_stop.os.kill", kill_mock), \
             patch("tj.cli.cmd_stop._find_serve_pid", find_pid_mock):
            result = CliRunner().invoke(cmd_stop, [], obj={})

        assert result.exit_code == 0, result.output
        # The foreground PID must have been signaled.
        kill_mock.assert_called_once_with(12345, 15)  # SIGTERM = 15
        # Output mentions both stop methods
        assert "launchd daemon unloaded" in result.output
        assert "PID 12345" in result.output

    def test_does_not_loop_on_slow_shutdown(self, tmp_path, monkeypatch):
        """SIGTERM is async — if the target process hasn't exited before the
        next pgrep, the sweep must NOT re-signal the same PID. Otherwise a
        slow shutdown handler can make `ocw stop` hang forever."""
        monkeypatch.setattr("tj.cli.cmd_stop.Path.home", lambda: tmp_path)
        kill_mock = MagicMock()
        # pgrep keeps returning the same PID — simulates a process whose
        # SIGTERM handler hasn't completed yet.
        find_pid_mock = MagicMock(side_effect=[12345] * 50)

        with patch("tj.cli.cmd_stop.os.kill", kill_mock), \
             patch("tj.cli.cmd_stop._find_serve_pid", find_pid_mock):
            result = CliRunner().invoke(cmd_stop, [], obj={})

        assert result.exit_code == 0
        # SIGTERM sent exactly once even though pgrep saw the PID 50 times.
        assert kill_mock.call_count == 1
        kill_mock.assert_called_once_with(12345, 15)

    def test_reports_not_running_when_nothing_to_stop(self, tmp_path, monkeypatch):
        # No plist, no systemd unit, no foreground process.
        monkeypatch.setattr("tj.cli.cmd_stop.Path.home", lambda: tmp_path)
        with patch("tj.cli.cmd_stop._find_serve_pid", return_value=None):
            result = CliRunner().invoke(cmd_stop, [], obj={})
        assert result.exit_code == 0
        assert "not running" in result.output
