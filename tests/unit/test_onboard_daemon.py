"""Unit tests for daemon detection logic in cmd_onboard."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from tj.cli.cmd_onboard import _daemon_already_running


class TestDaemonAlreadyRunning:
    def test_darwin_plist_exists_and_loaded(self, tmp_path, monkeypatch):
        """Returns True on macOS when plist exists and launchctl list succeeds."""
        monkeypatch.setattr("tj.cli.cmd_onboard.platform.system", lambda: "Darwin")
        plist = tmp_path / "Library" / "LaunchAgents" / "com.tokenjam.serve.plist"
        plist.parent.mkdir(parents=True)
        plist.write_text("<plist/>")
        monkeypatch.setattr("tj.cli.cmd_onboard.Path.home", lambda: tmp_path)

        result_mock = MagicMock(returncode=0)
        with patch("tj.cli.cmd_onboard.subprocess.run", return_value=result_mock) as run_mock:
            assert _daemon_already_running() is True
            run_mock.assert_called_once_with(
                ["launchctl", "list", "com.tokenjam.serve"],
                capture_output=True, text=True,
            )

    def test_darwin_plist_missing(self, tmp_path, monkeypatch):
        """Returns False on macOS when plist does not exist."""
        monkeypatch.setattr("tj.cli.cmd_onboard.platform.system", lambda: "Darwin")
        monkeypatch.setattr("tj.cli.cmd_onboard.Path.home", lambda: tmp_path)
        assert _daemon_already_running() is False

    def test_darwin_plist_exists_but_not_loaded(self, tmp_path, monkeypatch):
        """Returns False on macOS when plist exists but launchctl list fails."""
        monkeypatch.setattr("tj.cli.cmd_onboard.platform.system", lambda: "Darwin")
        plist = tmp_path / "Library" / "LaunchAgents" / "com.tokenjam.serve.plist"
        plist.parent.mkdir(parents=True)
        plist.write_text("<plist/>")
        monkeypatch.setattr("tj.cli.cmd_onboard.Path.home", lambda: tmp_path)

        result_mock = MagicMock(returncode=3)
        with patch("tj.cli.cmd_onboard.subprocess.run", return_value=result_mock):
            assert _daemon_already_running() is False

    def test_linux_active(self, monkeypatch):
        """Returns True on Linux when systemctl reports active."""
        monkeypatch.setattr("tj.cli.cmd_onboard.platform.system", lambda: "Linux")
        result_mock = MagicMock(returncode=0, stdout="active\n")
        with patch("tj.cli.cmd_onboard.subprocess.run", return_value=result_mock) as run_mock:
            assert _daemon_already_running() is True
            run_mock.assert_called_once_with(
                ["systemctl", "--user", "is-active", "tokenjam"],
                capture_output=True, text=True,
            )

    def test_linux_inactive(self, monkeypatch):
        """Returns False on Linux when systemctl reports inactive."""
        monkeypatch.setattr("tj.cli.cmd_onboard.platform.system", lambda: "Linux")
        result_mock = MagicMock(returncode=3, stdout="inactive\n")
        with patch("tj.cli.cmd_onboard.subprocess.run", return_value=result_mock):
            assert _daemon_already_running() is False

    def test_unsupported_platform(self, monkeypatch):
        """Returns False on unsupported platforms."""
        monkeypatch.setattr("tj.cli.cmd_onboard.platform.system", lambda: "Windows")
        assert _daemon_already_running() is False


class TestLaunchdInstallUsesWFlag:
    """`_install_launchd` must pass -w to both unload and load so it clears
    the Disabled=true flag that `ocw stop` writes (C1)."""

    def test_load_uses_w_flag(self, tmp_path, monkeypatch):
        from tj.cli.cmd_onboard import _install_launchd
        monkeypatch.setattr("tj.cli.cmd_onboard.Path.home", lambda: tmp_path)
        monkeypatch.setattr("tj.cli.cmd_onboard.shutil.which", lambda _: "/usr/bin/ocw")

        run_mock = MagicMock(return_value=MagicMock(returncode=0))
        with patch("tj.cli.cmd_onboard.subprocess.run", run_mock):
            _install_launchd("/tmp/cfg.toml")

        # Both unload and load must include -w
        calls = [c.args[0] for c in run_mock.call_args_list]
        assert any("unload" in c and "-w" in c for c in calls), (
            f"unload should use -w; calls were {calls}"
        )
        assert any("load" in c and "-w" in c for c in calls), (
            f"load should use -w; calls were {calls}"
        )
