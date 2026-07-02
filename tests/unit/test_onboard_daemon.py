"""Unit tests for daemon detection logic in cmd_onboard."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from tokenjam.cli.cmd_onboard import _daemon_already_running


class TestDaemonAlreadyRunning:
    def test_darwin_plist_exists_and_loaded(self, tmp_path, monkeypatch):
        """Returns True on macOS when plist exists and launchctl list succeeds."""
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.platform.system", lambda: "Darwin")
        plist = tmp_path / "Library" / "LaunchAgents" / "com.tokenjam.serve.plist"
        plist.parent.mkdir(parents=True)
        plist.write_text("<plist/>")
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)

        result_mock = MagicMock(returncode=0)
        with patch("tokenjam.cli.cmd_onboard.subprocess.run", return_value=result_mock) as run_mock:
            assert _daemon_already_running() is True
            run_mock.assert_called_once_with(
                ["launchctl", "list", "com.tokenjam.serve"],
                capture_output=True, text=True,
            )

    def test_darwin_plist_missing(self, tmp_path, monkeypatch):
        """Returns False on macOS when plist does not exist."""
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.platform.system", lambda: "Darwin")
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        assert _daemon_already_running() is False

    def test_darwin_plist_exists_but_not_loaded(self, tmp_path, monkeypatch):
        """Returns False on macOS when plist exists but launchctl list fails."""
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.platform.system", lambda: "Darwin")
        plist = tmp_path / "Library" / "LaunchAgents" / "com.tokenjam.serve.plist"
        plist.parent.mkdir(parents=True)
        plist.write_text("<plist/>")
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)

        result_mock = MagicMock(returncode=3)
        with patch("tokenjam.cli.cmd_onboard.subprocess.run", return_value=result_mock):
            assert _daemon_already_running() is False

    def test_linux_active(self, monkeypatch):
        """Returns True on Linux when systemctl reports active."""
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.platform.system", lambda: "Linux")
        result_mock = MagicMock(returncode=0, stdout="active\n")
        with patch("tokenjam.cli.cmd_onboard.subprocess.run", return_value=result_mock) as run_mock:
            assert _daemon_already_running() is True
            run_mock.assert_called_once_with(
                ["systemctl", "--user", "is-active", "tokenjam"],
                capture_output=True, text=True,
            )

    def test_linux_inactive(self, monkeypatch):
        """Returns False on Linux when systemctl reports inactive."""
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.platform.system", lambda: "Linux")
        result_mock = MagicMock(returncode=3, stdout="inactive\n")
        with patch("tokenjam.cli.cmd_onboard.subprocess.run", return_value=result_mock):
            assert _daemon_already_running() is False

    def test_unsupported_platform(self, monkeypatch):
        """Returns False on unsupported platforms."""
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.platform.system", lambda: "Windows")
        assert _daemon_already_running() is False


class TestLaunchdInstallUsesWFlag:
    """`_install_launchd` must pass -w to both unload and load so it clears
    the Disabled=true flag that `tj stop` writes (C1)."""

    def test_load_uses_w_flag(self, tmp_path, monkeypatch):
        from tokenjam.cli.cmd_onboard import _install_launchd
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: "/usr/bin/tj")

        run_mock = MagicMock(return_value=MagicMock(returncode=0))
        with patch("tokenjam.cli.cmd_onboard.subprocess.run", run_mock):
            _install_launchd("/tmp/cfg.toml")

        # Both unload and load must include -w
        calls = [c.args[0] for c in run_mock.call_args_list]
        assert any("unload" in c and "-w" in c for c in calls), (
            f"unload should use -w; calls were {calls}"
        )
        assert any("load" in c and "-w" in c for c in calls), (
            f"load should use -w; calls were {calls}"
        )


class TestTjBinaryResolution:
    """The daemon installers must point launchd/systemd at a real `tj` binary.

    Regression for #340: when `tj` is off PATH, the fallback derived the path
    with `sys.executable.replace("/python", "/tj")`, which rewrote a
    `python3`-named interpreter to a nonexistent `tj3` (because `/python`
    matches inside `/python3`). The unit is written pointing at a binary that
    doesn't exist; `launchctl load` still returns 0, so onboarding reports
    success while `tj serve` never launches.
    """

    def test_prefers_tj_on_path(self, monkeypatch):
        from tokenjam.cli.cmd_onboard import _resolve_tj_binary
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: "/usr/local/bin/tj")
        assert _resolve_tj_binary() == "/usr/local/bin/tj"

    def test_fallback_python3_resolves_to_sibling_tj(self, monkeypatch):
        """A `python3`-named interpreter must yield the sibling `tj`, not `tj3`."""
        from tokenjam.cli.cmd_onboard import _resolve_tj_binary
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: None)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.sys.executable", "/opt/venv/bin/python3"
        )
        assert _resolve_tj_binary() == "/opt/venv/bin/tj"

    def test_fallback_python311_resolves_to_sibling_tj(self, monkeypatch):
        """A versioned `python3.11` interpreter must also yield `tj`, not `tj3.11`."""
        from tokenjam.cli.cmd_onboard import _resolve_tj_binary
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: None)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.sys.executable", "/opt/venv/bin/python3.11"
        )
        assert _resolve_tj_binary() == "/opt/venv/bin/tj"

    def test_launchd_plist_never_points_at_tj3(self, tmp_path, monkeypatch):
        from tokenjam.cli.cmd_onboard import _install_launchd
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: None)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.sys.executable", "/opt/venv/bin/python3"
        )
        with patch(
            "tokenjam.cli.cmd_onboard.subprocess.run",
            MagicMock(return_value=MagicMock(returncode=0)),
        ):
            _install_launchd("/tmp/cfg.toml")

        plist = (tmp_path / "Library/LaunchAgents/com.tokenjam.serve.plist").read_text()
        assert "<string>/opt/venv/bin/tj</string>" in plist
        assert "/tj3" not in plist

    def test_systemd_unit_never_points_at_tj3(self, tmp_path, monkeypatch):
        from tokenjam.cli.cmd_onboard import _install_systemd
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: None)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.sys.executable", "/opt/venv/bin/python3.11"
        )
        with patch(
            "tokenjam.cli.cmd_onboard.subprocess.run",
            MagicMock(return_value=MagicMock(returncode=0)),
        ):
            _install_systemd("/tmp/cfg.toml")

        unit = (tmp_path / ".config/systemd/user/tokenjam.service").read_text()
        assert "ExecStart=/opt/venv/bin/tj --config /tmp/cfg.toml serve" in unit
        assert "/tj3" not in unit
