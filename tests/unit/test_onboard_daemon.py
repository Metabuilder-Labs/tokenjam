"""Unit tests for daemon detection logic in cmd_onboard."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from tokenjam.cli.cmd_onboard import _daemon_already_running


def _print_stdout(program_args: list[str]) -> str:
    """A `launchctl print` text body reporting an active unit whose
    `arguments` array is exactly `program_args` — what `_install_launchd`'s
    post-install content check parses out and compares."""
    args_block = "\n".join(f"\t\t{a}" for a in program_args)
    return (
        "gui/501/com.tokenjam.serve = {\n"
        "\tactive count = 1\n"
        "\tstate = running\n\n"
        f"\tprogram = {program_args[0]}\n"
        "\targuments = {\n"
        f"{args_block}\n"
        "\t}\n"
        "}\n"
    )


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


class TestLaunchdInstallSurvivesRegistration:
    """`_install_launchd` must use the modern `bootstrap`/`enable`/`bootout`
    subcommands, never the legacy `load`/`unload` — the `launchctl` man page
    marks load/unload legacy and warns they "will only return a non-zero exit
    code due to improper usage. Otherwise, zero is always returned", so a
    silently-failed registration is invisible to a caller that only checks
    the exit code (the actual production failure mode this fix addresses:
    the plist was present, `RunAtLoad`/`KeepAlive` both true, no `Disabled`
    key, yet the unit was simply never re-loaded into launchd).

    Formerly `TestLaunchdInstallUsesWFlag` (C1), which pinned `load -w` /
    `unload -w` clearing the Disabled flag `tj stop` writes. `enable` is the
    modern equivalent of that disable-clearing half; this class re-asserts
    the same guarantee against the new mechanism (Critical Rule 23 in the
    project's `.claude/rules` — invert a superseded assertion rather than
    dropping the coverage) and adds the independent post-install
    verification the old mechanism never did.
    """

    def test_install_bootouts_enables_bootstraps_and_verifies(self, tmp_path, monkeypatch):
        from tokenjam.cli.cmd_onboard import _install_launchd
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        # No real `tj` sibling next to this fake interpreter, so
        # `_resolve_tj_binary` falls through to the mocked `which` below —
        # on a real venv (CI installs the package editable) a genuine `tj`
        # console-script sibling can otherwise outrank the mock and change
        # the actual program args this test hardcodes.
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.sys.executable", str(tmp_path / "python3"))
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: "/usr/bin/tj")
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.os.getuid", lambda: 501, raising=False)

        program_args = ["/usr/bin/tj", "--config", "/tmp/cfg.toml", "serve"]

        def _run(cmd, **kwargs):
            if cmd[1] == "print":
                return MagicMock(returncode=0, stdout=_print_stdout(program_args), stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        run_mock = MagicMock(side_effect=_run)
        with patch("tokenjam.cli.cmd_onboard.subprocess.run", run_mock):
            result = _install_launchd("/tmp/cfg.toml")

        assert result is not None
        calls = [c.args[0] for c in run_mock.call_args_list]
        subcommands = [c[1] for c in calls]
        assert "bootout" in subcommands
        assert "enable" in subcommands
        assert "bootstrap" in subcommands
        assert "print" in subcommands
        # Never the deprecated legacy calls this replaces.
        assert "load" not in subcommands
        assert "unload" not in subcommands

        target = "gui/501/com.tokenjam.serve"
        enable_call = next(c for c in calls if c[1] == "enable")
        assert enable_call[-1] == target
        bootout_call = next(c for c in calls if c[1] == "bootout")
        assert bootout_call[-1] == target
        print_call = next(c for c in calls if c[1] == "print")
        assert print_call[-1] == target
        bootstrap_call = next(c for c in calls if c[1] == "bootstrap")
        assert bootstrap_call[2] == "gui/501"

    def test_install_fails_when_verification_shows_not_registered(self, tmp_path, monkeypatch):
        """`bootstrap` can report success while the service never actually
        registers — the same "exit code lies" failure mode this fix exists
        for. The install must be reported as FAILED when the independent
        `print` verification can't find the label, not only when bootstrap
        itself errors."""
        from tokenjam.cli.cmd_onboard import _install_launchd
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.sys.executable", str(tmp_path / "python3"))
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: "/usr/bin/tj")

        def _run(cmd, **kwargs):
            if cmd[1] == "print":
                return MagicMock(returncode=3, stdout="", stderr="Could not find service")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("tokenjam.cli.cmd_onboard.subprocess.run", side_effect=_run):
            result = _install_launchd("/tmp/cfg.toml")

        assert result is None

    def test_install_tolerates_already_bootstrapped_when_content_matches(
        self, tmp_path, monkeypatch,
    ):
        """A re-onboard racing the `bootout` above (or one that no-op'd
        against a job still shutting down) must not be reported as a failed
        install, so long as the independent verification confirms the
        ACTIVE unit's program arguments match what was just written — not
        merely that some unit with that label exists."""
        from tokenjam.cli.cmd_onboard import _install_launchd
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.sys.executable", str(tmp_path / "python3"))
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: "/usr/bin/tj")

        program_args = ["/usr/bin/tj", "--config", "/tmp/cfg.toml", "serve"]

        def _run(cmd, **kwargs):
            if cmd[1] == "bootstrap":
                return MagicMock(returncode=1, stdout="", stderr="Service already bootstrapped")
            if cmd[1] == "print":
                return MagicMock(returncode=0, stdout=_print_stdout(program_args), stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("tokenjam.cli.cmd_onboard.subprocess.run", side_effect=_run):
            result = _install_launchd("/tmp/cfg.toml")

        assert result is not None

    def test_install_fails_when_already_bootstrapped_verifies_the_old_unit(
        self, tmp_path, monkeypatch,
    ):
        """The exact defect this fix closes: `bootout` didn't actually clear
        the previous registration, `bootstrap` reports "already bootstrapped"
        (meaning the OLD unit is still loaded), and `launchctl print` happily
        confirms that old label — with OLD program arguments (a stale
        executable/config path), not the new plist just written. A bare
        "label exists" check would pass this; the content check must fail
        it loudly instead of reporting a false success."""
        from tokenjam.cli.cmd_onboard import _install_launchd
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.Path.home", lambda: tmp_path)
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.sys.executable", str(tmp_path / "python3"))
        monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _: "/usr/bin/tj")

        old_program_args = ["/usr/bin/tj", "--config", "/tmp/OLD-cfg.toml", "serve"]

        def _run(cmd, **kwargs):
            if cmd[1] == "bootstrap":
                return MagicMock(returncode=1, stdout="", stderr="Service already bootstrapped")
            if cmd[1] == "print":
                # Always reports the OLD unit, even after the retry's
                # bootout/bootstrap — simulating a registration that never
                # actually gets replaced.
                return MagicMock(returncode=0, stdout=_print_stdout(old_program_args), stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("tokenjam.cli.cmd_onboard.subprocess.run", side_effect=_run):
            result = _install_launchd("/tmp/cfg.toml")

        assert result is None


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
        program_args = [
            "/Users/x/.local/bin/uvx", "--from", "tokenjam", "tj",
            "--config", "/tmp/cfg.toml", "serve",
        ]

        def _run(cmd, **kwargs):
            if cmd[1] == "print":
                return MagicMock(returncode=0, stdout=_print_stdout(program_args), stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("tokenjam.cli.cmd_onboard.subprocess.run", side_effect=_run):
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
