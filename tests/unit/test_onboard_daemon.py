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


class TestRepeatOnboardDoesNotChurnDaemon:
    """Regression: `_stop_serve_for_db_write` used to call `stop_tj_serve`,
    which reported "stopped" whenever a plist FILE existed on disk, even if
    launchd never had it loaded. That made `stopped_for_db` True on every
    onboard run, which forced `need_restart` True in `_finish_onboard_serve`,
    which meant the "already running -> skip reinstall" branch was
    unreachable and the daemon reinstalled/restarted on every single
    onboard -- not just when something genuinely changed."""

    def _finish(self, config_path, **overrides):
        from tokenjam.cli.cmd_onboard import _finish_onboard_serve

        kwargs = dict(
            want_daemon=True,
            plan_changed=False,
            stopped_for_db=False,
            secret_rotated=False,
            no_daemon=False,
            force=False,
        )
        kwargs.update(overrides)
        return _finish_onboard_serve(config_path, **kwargs)

    def test_second_onboard_with_daemon_running_skips_reinstall(
        self, tmp_path, monkeypatch,
    ):
        """Daemon genuinely running, nothing changed: `stop_tj_serve` must
        report False (nothing was actually stopped), so the already-running
        skip branch is taken instead of reinstalling."""
        config = tmp_path / "config.toml"
        config.write_text("[budget]\n")
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)

        with patch("tokenjam.cli.cmd_onboard._daemon_already_running", return_value=True), \
             patch("tokenjam.cli.cmd_onboard._install_daemon") as install_mock, \
             patch("tokenjam.cli.cmd_stop.stop_tj_serve", return_value=(False, [])):
            from tokenjam.cli.cmd_onboard import _stop_serve_for_db_write
            stopped_for_db = _stop_serve_for_db_write()
            restart_msg = self._finish(str(config), stopped_for_db=stopped_for_db)

        assert stopped_for_db is False
        install_mock.assert_not_called()
        assert restart_msg == "daemon already running"

    def test_second_onboard_with_daemon_not_running_does_not_restart(
        self, tmp_path, monkeypatch,
    ):
        """Daemon not running at all (plist absent, or never loaded): no
        restart should be triggered just because onboard ran again."""
        config = tmp_path / "config.toml"
        config.write_text("[budget]\n")
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)

        with patch("tokenjam.cli.cmd_onboard._daemon_already_running", return_value=False), \
             patch("tokenjam.cli.cmd_onboard._install_daemon", return_value="installed") as install_mock, \
             patch("tokenjam.cli.cmd_stop.stop_tj_serve", return_value=(False, [])):
            from tokenjam.cli.cmd_onboard import _stop_serve_for_db_write
            stopped_for_db = _stop_serve_for_db_write()
            self._finish(str(config), stopped_for_db=stopped_for_db)

        assert stopped_for_db is False
        # Not "already running", so the normal install path runs once --
        # but crucially the restart path (`_restart_tj_server`) is not what
        # ran; `_install_daemon` is the plain (re)install, called exactly
        # once, not repeatedly forced by a false "stopped_for_db".
        install_mock.assert_called_once()

    def test_plan_change_still_forces_restart(self, tmp_path, monkeypatch):
        """A genuine plan change must still restart the daemon even though
        stop_tj_serve reports nothing was stopped."""
        config = tmp_path / "config.toml"
        config.write_text("[budget]\n")
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)

        with patch("tokenjam.cli.cmd_onboard._daemon_already_running", return_value=True), \
             patch("tokenjam.cli.cmd_onboard._restart_tj_server", return_value="restarted") as restart_mock, \
             patch("tokenjam.cli.cmd_stop.stop_tj_serve", return_value=(False, [])):
            from tokenjam.cli.cmd_onboard import _stop_serve_for_db_write
            stopped_for_db = _stop_serve_for_db_write()
            restart_msg = self._finish(
                str(config), stopped_for_db=stopped_for_db, plan_changed=True,
            )

        assert stopped_for_db is False
        restart_mock.assert_called_once()
        assert restart_msg == "restarted"

    def test_genuine_stop_for_db_write_still_forces_restart(self, tmp_path, monkeypatch):
        """When a daemon really was running and really got stopped to allow
        a DB write, the restart must still happen -- the fix must not
        swallow genuine stops, only false-positive ones."""
        config = tmp_path / "config.toml"
        config.write_text("[budget]\n")
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)

        with patch("tokenjam.cli.cmd_onboard._daemon_already_running", return_value=True), \
             patch("tokenjam.cli.cmd_onboard._restart_tj_server", return_value="restarted") as restart_mock, \
             patch("tokenjam.cli.cmd_stop.stop_tj_serve", return_value=(True, ["launchd daemon unloaded"])):
            from tokenjam.cli.cmd_onboard import _stop_serve_for_db_write
            stopped_for_db = _stop_serve_for_db_write()
            restart_msg = self._finish(str(config), stopped_for_db=stopped_for_db)

        assert stopped_for_db is True
        restart_mock.assert_called_once()
        assert restart_msg == "restarted"


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

    Also regression for a PATH-shadow bug: `_resolve_tj_binary` used to
    prefer `shutil.which("tj")` over the interpreter sibling, so an
    older/other `tj` earlier on PATH at the moment `tj onboard` installs the
    daemon got permanently baked into the launchd/systemd unit — surviving
    even after the shadowing PATH entry was later removed. It must now
    prefer the PATH-independent sibling next to the running interpreter,
    the same priority `_current_tj_binary` uses.
    """

    def test_prefers_interpreter_sibling_over_a_shadowing_path_tj(self, monkeypatch, tmp_path):
        """A `tj` earlier on PATH must never win over the sibling next to the
        interpreter that onboard is actually running as — that's the whole
        PATH-shadow bug this resolves."""
        from tokenjam.cli.cmd_onboard import _resolve_tj_binary
        sibling = tmp_path / "tj"
        sibling.write_text("#!/bin/sh\n")
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.sys.executable", str(tmp_path / "python3"))
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.shutil.which", lambda _: "/usr/local/bin/tj",
        )
        assert _resolve_tj_binary() == str(sibling)

    def test_falls_back_to_which_when_no_sibling_exists(self, monkeypatch, tmp_path):
        from tokenjam.cli.cmd_onboard import _resolve_tj_binary
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.sys.executable", str(tmp_path / "python3"))
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.shutil.which", lambda _: "/usr/local/bin/tj",
        )
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


class TestDaemonSurvivesUvCachePrune:
    """A daemon unit installed by uvx/pipx-driven onboard must not point at
    uv's ephemeral tool-archive cache (#155): `uv cache prune`/`uv cache
    clean` (routine maintenance, also run by some CI/cleanup tools) deletes
    that path outright, silently killing the daemon on next launchd/systemd
    load, and pins the daemon to whatever version was resolved at onboard
    time forever — independent of the wrapper's `--refresh` freshness logic
    (#111). When the only resolvable `tj` is an ephemeral cache path, the
    unit must instead invoke through the stable `uvx`/`pipx` shim so it keeps
    working (and self-updates) after a prune.
    """

    def test_program_args_prefers_direct_tj_when_not_ephemeral(self, monkeypatch, tmp_path):
        from tokenjam.cli.cmd_onboard import _daemon_program_args
        # No real sibling `tj` next to this fake interpreter path, so
        # resolution falls back to the (non-ephemeral) `which` result.
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.sys.executable", str(tmp_path / "python3"))
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.shutil.which",
            lambda b: "/usr/local/bin/tj" if b == "tj" else None,
        )
        assert _daemon_program_args("/tmp/cfg.toml") == [
            "/usr/local/bin/tj", "--config", "/tmp/cfg.toml", "serve",
        ]

    def test_program_args_falls_back_to_uvx_shim_when_tj_is_archive_cache(self, monkeypatch):
        from tokenjam.cli.cmd_onboard import _daemon_program_args
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.sys.executable",
            "/Users/x/.cache/uv/archive-v0/abc123/bin/python",
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.shutil.which",
            lambda b: "/Users/x/.local/bin/uvx" if b == "uvx" else None,
        )
        args = _daemon_program_args("/tmp/cfg.toml")
        assert args == [
            "/Users/x/.local/bin/uvx", "--from", "tokenjam", "tj",
            "--config", "/tmp/cfg.toml", "serve",
        ]
        assert not any("archive-v0" in a for a in args)

    def test_program_args_falls_back_to_pipx_shim_when_no_uvx(self, monkeypatch):
        from tokenjam.cli.cmd_onboard import _daemon_program_args
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.sys.executable",
            "/Users/x/.local/share/pipx/.cache/xyz/bin/python",
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.shutil.which",
            lambda b: "/usr/local/bin/pipx" if b == "pipx" else None,
        )
        args = _daemon_program_args("/tmp/cfg.toml")
        assert args == [
            "/usr/local/bin/pipx", "run", "--spec", "tokenjam", "tj",
            "--config", "/tmp/cfg.toml", "serve",
        ]

    def test_program_args_none_when_no_durable_entrypoint_exists(self, monkeypatch):
        from tokenjam.cli.cmd_onboard import _daemon_program_args
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: None)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.sys.executable",
            "/Users/x/.cache/uv/archive-v0/abc123/bin/python",
        )
        assert _daemon_program_args("/tmp/cfg.toml") is None

    def test_launchd_plist_never_contains_archive_v0_path(self, tmp_path, monkeypatch):
        from tokenjam.cli.cmd_onboard import _install_launchd
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.sys.executable",
            "/Users/x/.cache/uv/archive-v0/abc123/bin/python",
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.shutil.which",
            lambda b: "/Users/x/.local/bin/uvx" if b == "uvx" else None,
        )
        with patch(
            "tokenjam.cli.cmd_onboard.subprocess.run",
            MagicMock(return_value=MagicMock(returncode=0)),
        ):
            result = _install_launchd("/tmp/cfg.toml")

        assert result is not None
        plist = (tmp_path / "Library/LaunchAgents/com.tokenjam.serve.plist").read_text()
        assert "archive-v0" not in plist
        assert "<string>/Users/x/.local/bin/uvx</string>" in plist
        assert "<string>--from</string>" in plist
        assert "<string>tokenjam</string>" in plist

    def test_launchd_skips_install_when_no_durable_entrypoint(self, tmp_path, monkeypatch, capsys):
        from tokenjam.cli.cmd_onboard import _install_launchd
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: None)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.sys.executable",
            "/Users/x/.cache/uv/archive-v0/abc123/bin/python",
        )
        result = _install_launchd("/tmp/cfg.toml")
        assert result is None
        assert not (tmp_path / "Library/LaunchAgents/com.tokenjam.serve.plist").exists()
        assert "No durable" in capsys.readouterr().out

    def test_systemd_unit_never_contains_archive_v0_path(self, tmp_path, monkeypatch):
        from tokenjam.cli.cmd_onboard import _install_systemd
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.sys.executable",
            "/Users/x/.cache/uv/archive-v0/abc123/bin/python",
        )
        monkeypatch.setattr(
            "tokenjam.cli.cmd_onboard.shutil.which",
            lambda b: "/Users/x/.local/bin/uvx" if b == "uvx" else None,
        )
        with patch(
            "tokenjam.cli.cmd_onboard.subprocess.run",
            MagicMock(return_value=MagicMock(returncode=0)),
        ):
            result = _install_systemd("/tmp/cfg.toml")

        assert result is not None
        unit = (tmp_path / ".config/systemd/user/tokenjam.service").read_text()
        assert "archive-v0" not in unit
        assert "ExecStart=/Users/x/.local/bin/uvx --from tokenjam tj --config /tmp/cfg.toml serve" in unit
