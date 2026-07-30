"""Unit tests for `tj upgrade` (cmd_upgrade.py).

Covers: install-manager detection per install layout, the ephemeral/
unmanaged refusal, the daemon-not-restarted-on-failed-install guard, the
launchd-vs-stop/serve restart branch, the no-daemon-running case, and the
post-restart version verification (success + failing-loudly-on-timeout).

Everything here mocks subprocess/launchctl/HTTP -- nothing in this file may
upgrade a real package, shell out to launchctl for real, or touch a real
daemon.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from tokenjam.cli import cmd_upgrade as upgrade_mod
from tokenjam.core.server_state import ServerState


# -- detect_upgrade_plan(): install-manager detection per layout


class TestDetectUpgradePlan:
    def test_pipx_install_produces_pipx_upgrade_plan(self):
        with patch.object(upgrade_mod, "_is_ephemeral_runner", return_value=False), \
             patch.object(upgrade_mod, "_installed_via_pipx", return_value=True), \
             patch.object(upgrade_mod, "_installed_via_uv_tool", return_value=False):
            plan = upgrade_mod.detect_upgrade_plan()
        assert plan is not None
        assert plan.manager == "pipx"
        assert plan.argv == ["pipx", "upgrade", "tokenjam"]

    def test_uv_tool_install_produces_uv_tool_upgrade_plan(self):
        with patch.object(upgrade_mod, "_is_ephemeral_runner", return_value=False), \
             patch.object(upgrade_mod, "_installed_via_pipx", return_value=False), \
             patch.object(upgrade_mod, "_installed_via_uv_tool", return_value=True):
            plan = upgrade_mod.detect_upgrade_plan()
        assert plan is not None
        assert plan.manager == "uv-tool"
        assert plan.argv == ["uv", "tool", "upgrade", "tokenjam"]

    def test_plain_writable_pip_install_produces_pip_upgrade_plan(self):
        with patch.object(upgrade_mod, "_is_ephemeral_runner", return_value=False), \
             patch.object(upgrade_mod, "_installed_via_pipx", return_value=False), \
             patch.object(upgrade_mod, "_installed_via_uv_tool", return_value=False), \
             patch.object(upgrade_mod, "_pip_target_writable", return_value=True):
            plan = upgrade_mod.detect_upgrade_plan()
        assert plan is not None
        assert plan.manager == "pip"
        assert plan.argv[-3:] == ["install", "--upgrade", "tokenjam"]

    def test_ephemeral_runner_refuses_to_upgrade(self):
        with patch.object(upgrade_mod, "_is_ephemeral_runner", return_value=True):
            plan = upgrade_mod.detect_upgrade_plan()
        assert plan is None

    def test_unwritable_pip_target_refuses_to_upgrade(self):
        """A system/read-only Python: never attempted, so it can't be left
        half-upgraded."""
        with patch.object(upgrade_mod, "_is_ephemeral_runner", return_value=False), \
             patch.object(upgrade_mod, "_installed_via_pipx", return_value=False), \
             patch.object(upgrade_mod, "_installed_via_uv_tool", return_value=False), \
             patch.object(upgrade_mod, "_pip_target_writable", return_value=False):
            plan = upgrade_mod.detect_upgrade_plan()
        assert plan is None


class TestCmdUpgradeRefusesEphemeralInstall(object):
    def test_cli_exits_nonzero_and_never_touches_daemon(self):
        restart_mock = MagicMock()
        with patch.object(upgrade_mod, "detect_upgrade_plan", return_value=None), \
             patch.object(upgrade_mod, "restart_daemon", restart_mock):
            result = CliRunner().invoke(upgrade_mod.cmd_upgrade, [], obj={})
        assert result.exit_code != 0
        assert "ephemeral" in result.output.lower() or "unmanaged" in result.output.lower()
        restart_mock.assert_not_called()


# -- daemon NOT restarted when install fails


class TestDaemonNotRestartedOnFailedInstall:
    def test_run_package_upgrade_reports_failure_on_nonzero_exit(self):
        plan = upgrade_mod.UpgradePlan(
            manager="pipx", argv=["pipx", "upgrade", "tokenjam"], display="pipx upgrade tokenjam",
        )
        fake = MagicMock(returncode=1, stdout="", stderr="not installed")
        with patch.object(upgrade_mod.subprocess, "run", return_value=fake):
            success, detail = upgrade_mod.run_package_upgrade(plan)
        assert success is False
        assert "not installed" in detail

    def test_cli_exits_nonzero_and_skips_restart_when_install_fails(self):
        plan = upgrade_mod.UpgradePlan(
            manager="pipx", argv=["pipx", "upgrade", "tokenjam"], display="pipx upgrade tokenjam",
        )
        restart_mock = MagicMock()
        with patch.object(upgrade_mod, "detect_upgrade_plan", return_value=plan), \
             patch.object(upgrade_mod, "run_package_upgrade", return_value=(False, "boom")), \
             patch.object(upgrade_mod, "restart_daemon", restart_mock):
            result = CliRunner().invoke(upgrade_mod.cmd_upgrade, [], obj={})
        assert result.exit_code != 0
        assert "Upgrade failed" in result.output
        restart_mock.assert_not_called()


# -- restart_daemon(): launchd path vs stop/serve fallback vs no-daemon-running


class TestRestartDaemon:
    def test_no_daemon_running_is_a_noop_not_a_failure(self):
        method, success, detail = upgrade_mod.restart_daemon(None)
        assert method == "none"
        assert success is True

    def test_no_daemon_running_when_pid_dead(self):
        state = ServerState(pid=99999, port=7391, config_path=None)
        with patch.object(upgrade_mod, "is_pid_alive", return_value=False):
            method, success, detail = upgrade_mod.restart_daemon(state)
        assert method == "none"
        assert success is True

    def test_no_daemon_running_when_pid_alive_but_not_a_serve_process(self):
        state = ServerState(pid=123, port=7391, config_path=None)
        with patch.object(upgrade_mod, "is_pid_alive", return_value=True), \
             patch.object(upgrade_mod, "is_serve_process", return_value=False):
            method, success, detail = upgrade_mod.restart_daemon(state)
        assert method == "none"
        assert success is True

    def test_uses_launchd_kickstart_when_launchd_supervises_it(self):
        state = ServerState(pid=123, port=7391, config_path=None)
        run_mock = MagicMock(return_value=MagicMock(returncode=0, stdout="", stderr=""))
        with patch.object(upgrade_mod, "is_pid_alive", return_value=True), \
             patch.object(upgrade_mod, "is_serve_process", return_value=True), \
             patch.object(upgrade_mod, "_launchd_loaded", return_value=True), \
             patch.object(upgrade_mod.subprocess, "run", run_mock), \
             patch.object(upgrade_mod.os, "getuid", return_value=501, create=True):
            method, success, detail = upgrade_mod.restart_daemon(state)
        assert method == "launchd"
        assert success is True
        called_argv = run_mock.call_args.args[0]
        assert called_argv[:2] == ["launchctl", "kickstart"]
        assert "-k" in called_argv
        assert any("com.tokenjam.serve" in arg for arg in called_argv)

    def test_launchd_kickstart_failure_is_reported_not_swallowed(self):
        state = ServerState(pid=123, port=7391, config_path=None)
        run_mock = MagicMock(return_value=MagicMock(returncode=1, stdout="", stderr="no such target"))
        with patch.object(upgrade_mod, "is_pid_alive", return_value=True), \
             patch.object(upgrade_mod, "is_serve_process", return_value=True), \
             patch.object(upgrade_mod, "_launchd_loaded", return_value=True), \
             patch.object(upgrade_mod.subprocess, "run", run_mock), \
             patch.object(upgrade_mod.os, "getuid", return_value=501, create=True):
            method, success, detail = upgrade_mod.restart_daemon(state)
        assert method == "launchd"
        assert success is False
        assert "no such target" in detail

    def test_falls_back_to_stop_serve_when_launchd_not_loaded(self):
        state = ServerState(pid=123, port=7391, config_path=None)
        fallback_mock = MagicMock(return_value=(True, "restarted via tj serve"))
        with patch.object(upgrade_mod, "is_pid_alive", return_value=True), \
             patch.object(upgrade_mod, "is_serve_process", return_value=True), \
             patch.object(upgrade_mod, "_launchd_loaded", return_value=False), \
             patch.object(upgrade_mod, "_restart_via_stop_serve", fallback_mock):
            method, success, detail = upgrade_mod.restart_daemon(state)
        assert method == "stop-serve"
        assert success is True
        fallback_mock.assert_called_once()

    def test_stop_serve_fallback_stops_then_relaunches_via_resolved_binary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(upgrade_mod.Path, "home", lambda: tmp_path)
        monkeypatch.setattr(upgrade_mod.time, "sleep", lambda *_: None)
        stop_mock = MagicMock(return_value=(True, ["PID 123"]))
        popen_mock = MagicMock(return_value=MagicMock(poll=MagicMock(return_value=None)))
        with patch.object(upgrade_mod, "stop_tj_serve", stop_mock), \
             patch("tokenjam.cli.cmd_onboard._resolve_tj_binary", return_value="/usr/local/bin/tj"), \
             patch.object(upgrade_mod.subprocess, "Popen", popen_mock):
            success, detail = upgrade_mod._restart_via_stop_serve(None)
        assert success is True
        stop_mock.assert_called_once_with(quiet=True)
        popen_args = popen_mock.call_args.args[0]
        assert popen_args == ["/usr/local/bin/tj", "serve"]
        assert popen_mock.call_args.kwargs.get("start_new_session") is True

    def test_stop_serve_fallback_passes_original_config_path_through(self, tmp_path, monkeypatch):
        """Regression guard: a daemon started with `tj --config <path> serve`
        (or `TJ_CONFIG`) must come back up against the SAME config, not
        whatever `tj serve` resolves by default -- otherwise the replacement
        can load a different database/address/port than the one the caller
        polls for version verification afterward."""
        monkeypatch.setattr(upgrade_mod.Path, "home", lambda: tmp_path)
        monkeypatch.setattr(upgrade_mod.time, "sleep", lambda *_: None)
        stop_mock = MagicMock(return_value=(True, ["PID 123"]))
        popen_mock = MagicMock(return_value=MagicMock(poll=MagicMock(return_value=None)))
        with patch.object(upgrade_mod, "stop_tj_serve", stop_mock), \
             patch("tokenjam.cli.cmd_onboard._resolve_tj_binary", return_value="/usr/local/bin/tj"), \
             patch.object(upgrade_mod.subprocess, "Popen", popen_mock):
            success, detail = upgrade_mod._restart_via_stop_serve("/home/user/.tj/config.toml")
        assert success is True
        popen_args = popen_mock.call_args.args[0]
        assert popen_args == ["/usr/local/bin/tj", "--config", "/home/user/.tj/config.toml", "serve"]

    def test_restart_daemon_passes_state_config_path_to_stop_serve_fallback(self):
        state = ServerState(pid=123, port=7391, config_path="/etc/tj/config.toml")
        fallback_mock = MagicMock(return_value=(True, "restarted"))
        with patch.object(upgrade_mod, "is_pid_alive", return_value=True), \
             patch.object(upgrade_mod, "is_serve_process", return_value=True), \
             patch.object(upgrade_mod, "_launchd_loaded", return_value=False), \
             patch.object(upgrade_mod, "_restart_via_stop_serve", fallback_mock):
            upgrade_mod.restart_daemon(state)
        fallback_mock.assert_called_once_with("/etc/tj/config.toml")

    def test_stop_serve_fallback_refuses_to_launch_a_second_daemon_when_stop_fails(self, tmp_path, monkeypatch):
        """If `stop_tj_serve` cannot confirm the old daemon actually exited,
        launching a replacement anyway would run two daemons against the
        same port/DB -- the fallback must refuse rather than double up."""
        monkeypatch.setattr(upgrade_mod.Path, "home", lambda: tmp_path)
        stop_mock = MagicMock(return_value=(False, []))
        popen_mock = MagicMock()
        with patch.object(upgrade_mod, "stop_tj_serve", stop_mock), \
             patch.object(upgrade_mod.subprocess, "Popen", popen_mock):
            success, detail = upgrade_mod._restart_via_stop_serve(None)
        assert success is False
        assert "could not" in detail.lower()
        popen_mock.assert_not_called()

    def test_stop_serve_fallback_reports_failure_when_relaunched_child_dies_immediately(self, tmp_path, monkeypatch):
        """A `Popen` call succeeding only means the OS accepted the exec, not
        that the process is still alive a moment later -- a bad config or a
        port already in use exits almost immediately, and that must surface
        as a restart failure, not a false 'restarted' report that later
        shows up as a misleading version mismatch."""
        monkeypatch.setattr(upgrade_mod.Path, "home", lambda: tmp_path)
        monkeypatch.setattr(upgrade_mod.time, "sleep", lambda *_: None)
        stop_mock = MagicMock(return_value=(True, ["PID 123"]))
        popen_mock = MagicMock(return_value=MagicMock(poll=MagicMock(return_value=1)))
        with patch.object(upgrade_mod, "stop_tj_serve", stop_mock), \
             patch("tokenjam.cli.cmd_onboard._resolve_tj_binary", return_value="/usr/local/bin/tj"), \
             patch.object(upgrade_mod.subprocess, "Popen", popen_mock):
            success, detail = upgrade_mod._restart_via_stop_serve(None)
        assert success is False
        assert "exited immediately" in detail


# -- post-restart version verification


class TestPollDaemonVersion:
    def test_verifies_once_new_version_appears(self):
        fetch_mock = MagicMock(side_effect=["0.6.0", "0.6.0", "0.6.1"])
        sleep_mock = MagicMock()
        monotonic_mock = MagicMock(side_effect=[0.0, 0.1, 0.2, 0.3])
        verified, seen = upgrade_mod.poll_daemon_version(
            7391, "0.6.1", timeout_s=5.0, interval_s=0.1,
            sleep=sleep_mock, monotonic=monotonic_mock, fetch=fetch_mock,
        )
        assert verified is True
        assert seen == "0.6.1"
        assert sleep_mock.call_count == 2

    def test_fails_loudly_when_daemon_never_reports_new_version(self):
        """Regression guard: must never claim success for a daemon still
        stuck on the old version."""
        fetch_mock = MagicMock(return_value="0.6.0")
        monotonic_mock = MagicMock(side_effect=[0.0, 10.0])
        verified, seen = upgrade_mod.poll_daemon_version(
            7391, "0.6.1", timeout_s=1.0, interval_s=0.1,
            sleep=MagicMock(), monotonic=monotonic_mock, fetch=fetch_mock,
        )
        assert verified is False
        assert seen == "0.6.0"

    def test_fetch_daemon_version_returns_none_on_connection_failure(self):
        import urllib.error

        with patch.object(
            upgrade_mod.urllib.request, "urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            assert upgrade_mod.fetch_daemon_version(7391) is None


# -- full CLI success + failure paths


class TestCmdUpgradeEndToEnd:
    def test_success_path_reports_verified_version(self):
        plan = upgrade_mod.UpgradePlan(
            manager="uv-tool", argv=["uv", "tool", "upgrade", "tokenjam"],
            display="uv tool upgrade tokenjam",
        )
        state = ServerState(pid=123, port=7391, config_path=None)
        with patch.object(upgrade_mod, "detect_upgrade_plan", return_value=plan), \
             patch.object(upgrade_mod, "run_package_upgrade", return_value=(True, "ok")), \
             patch.object(upgrade_mod, "detect_new_version", return_value="0.6.1"), \
             patch.object(upgrade_mod, "read_server_state", return_value=state), \
             patch.object(upgrade_mod, "restart_daemon", return_value=("launchd", True, "restarted")), \
             patch.object(upgrade_mod, "poll_daemon_version", return_value=(True, "0.6.1")):
            result = CliRunner().invoke(upgrade_mod.cmd_upgrade, [], obj={})
        assert result.exit_code == 0, result.output
        assert "verified running 0.6.1" in result.output

    def test_no_daemon_running_exits_zero(self):
        plan = upgrade_mod.UpgradePlan(
            manager="pipx", argv=["pipx", "upgrade", "tokenjam"], display="pipx upgrade tokenjam",
        )
        with patch.object(upgrade_mod, "detect_upgrade_plan", return_value=plan), \
             patch.object(upgrade_mod, "run_package_upgrade", return_value=(True, "ok")), \
             patch.object(upgrade_mod, "detect_new_version", return_value="0.6.1"), \
             patch.object(upgrade_mod, "read_server_state", return_value=None), \
             patch.object(upgrade_mod, "restart_daemon", return_value=("none", True, "no daemon running")):
            result = CliRunner().invoke(upgrade_mod.cmd_upgrade, [], obj={})
        assert result.exit_code == 0, result.output
        assert "not running" in result.output.lower()

    def test_restart_failure_exits_nonzero(self):
        plan = upgrade_mod.UpgradePlan(
            manager="pipx", argv=["pipx", "upgrade", "tokenjam"], display="pipx upgrade tokenjam",
        )
        state = ServerState(pid=123, port=7391, config_path=None)
        with patch.object(upgrade_mod, "detect_upgrade_plan", return_value=plan), \
             patch.object(upgrade_mod, "run_package_upgrade", return_value=(True, "ok")), \
             patch.object(upgrade_mod, "detect_new_version", return_value="0.6.1"), \
             patch.object(upgrade_mod, "read_server_state", return_value=state), \
             patch.object(upgrade_mod, "restart_daemon", return_value=("launchd", False, "kickstart failed")):
            result = CliRunner().invoke(upgrade_mod.cmd_upgrade, [], obj={})
        assert result.exit_code != 0
        assert "Daemon restart failed" in result.output

    def test_verification_timeout_exits_nonzero_and_does_not_claim_success(self):
        plan = upgrade_mod.UpgradePlan(
            manager="pipx", argv=["pipx", "upgrade", "tokenjam"], display="pipx upgrade tokenjam",
        )
        state = ServerState(pid=123, port=7391, config_path=None)
        with patch.object(upgrade_mod, "detect_upgrade_plan", return_value=plan), \
             patch.object(upgrade_mod, "run_package_upgrade", return_value=(True, "ok")), \
             patch.object(upgrade_mod, "detect_new_version", return_value="0.6.1"), \
             patch.object(upgrade_mod, "read_server_state", return_value=state), \
             patch.object(upgrade_mod, "restart_daemon", return_value=("stop-serve", True, "restarted")), \
             patch.object(upgrade_mod, "poll_daemon_version", return_value=(False, "0.6.0")):
            result = CliRunner().invoke(upgrade_mod.cmd_upgrade, [], obj={})
        assert result.exit_code != 0
        assert "still reports 0.6.0" in result.output
        assert "verified running" not in result.output


# -- _pip_target_writable()


class TestPipTargetWritable:
    def test_true_when_purelib_dir_is_writable(self, tmp_path):
        purelib = tmp_path / "site-packages"
        purelib.mkdir()
        with patch("sysconfig.get_paths", return_value={"purelib": str(purelib)}):
            assert upgrade_mod._pip_target_writable() is True

    def test_false_when_purelib_dir_is_unwritable(self, tmp_path):
        purelib = tmp_path / "site-packages"
        purelib.mkdir()
        purelib.chmod(0o555)
        try:
            with patch("sysconfig.get_paths", return_value={"purelib": str(purelib)}):
                assert upgrade_mod._pip_target_writable() is False
        finally:
            purelib.chmod(0o755)

    def test_false_when_sysconfig_raises(self):
        with patch("sysconfig.get_paths", side_effect=Exception("boom")):
            assert upgrade_mod._pip_target_writable() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
