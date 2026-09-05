"""Unit tests for `_looks_like_serve`'s cmdline matcher.

The REAL installed daemon (see `_daemon_program_args()` in cmd_onboard.py)
launches as `<tj_path> --config <config_path> serve` -- `--config <path>`
sits BETWEEN `tj` and `serve`, so no contiguous `"tj serve"` substring ever
appears in its cmdline. A synthetic test argv that happens to contain the
literal substring `"tj serve"` (as used elsewhere, e.g.
test_cmd_stop_scoping.py's spawned process) would pass a naive substring
check without ever exercising this gap -- these tests use the REAL argv
shape so a regression back to substring matching is caught for real.
"""
from __future__ import annotations

import pytest

from tokenjam.core.server_state import _looks_like_serve


class TestMatchesRealDaemonInvocations:
    def test_matches_real_installed_daemon_argv(self):
        # Exactly what _daemon_program_args() writes into the launchd
        # plist / systemd unit for the common (non-ephemeral-path) case.
        cmdline = "/usr/local/bin/tj --config /home/user/.config/tj/config.toml serve"
        assert _looks_like_serve(cmdline) is True

    def test_matches_bare_tj_serve(self):
        assert _looks_like_serve("tj serve") is True

    def test_matches_module_form(self):
        assert _looks_like_serve("/usr/bin/python3 -m tokenjam.serve") is True

    def test_matches_uv_run_tj_serve(self):
        # Dev-invocation form -- must not regress.
        assert _looks_like_serve("uv run tj serve") is True

    def test_matches_uvx_wrapper_form(self):
        # _daemon_program_args()'s uvx fallback when tj resolves to an
        # ephemeral uv-cache path.
        cmdline = "/usr/local/bin/uvx --from tokenjam tj --config /home/user/.config/tj/config.toml serve"
        assert _looks_like_serve(cmdline) is True

    def test_matches_pipx_wrapper_form(self):
        # _daemon_program_args()'s pipx fallback.
        cmdline = "/usr/local/bin/pipx run --spec tokenjam tj --config /home/user/.config/tj/config.toml serve"
        assert _looks_like_serve(cmdline) is True


class TestDoesNotMatchUnrelatedProcesses:
    def test_does_not_match_serve_word_without_tj_token(self):
        assert _looks_like_serve("python manage.py serve") is False

    def test_does_not_match_tj_without_serve_subcommand(self):
        assert _looks_like_serve("/usr/local/bin/tj --version") is False

    def test_does_not_match_tj_as_substring_of_longer_token(self):
        # "tj" must be a bare token (or a path basename), not a substring
        # of some unrelated word.
        assert _looks_like_serve("/usr/bin/notjserve --serve") is False


class TestUnavailableProcessIdentity:
    @pytest.mark.parametrize("error", [FileNotFoundError, PermissionError, OSError])
    def test_unavailable_ps_does_not_identify_a_serve_process(self, monkeypatch, error):
        from unittest.mock import Mock

        from tokenjam.core import server_state

        monkeypatch.setattr(server_state.Path, "exists", lambda path: False)
        monkeypatch.setattr(server_state.subprocess, "run", Mock(side_effect=error))

        assert server_state.is_serve_process(42424242) is False

    def test_stop_does_not_signal_a_live_pid_without_identity(self, tmp_path, monkeypatch):
        import json
        from unittest.mock import Mock

        from tokenjam.cli import cmd_stop
        from tokenjam.core import server_state

        state_path = tmp_path / ".local/share/tj/server.state"
        state_path.parent.mkdir(parents=True)
        state_path.write_text(json.dumps({"pid": 42424242, "port": 7391, "config_path": None}))
        monkeypatch.setattr(server_state.Path, "home", lambda: tmp_path)
        monkeypatch.setattr(server_state.Path, "exists", lambda path: False)
        monkeypatch.setattr(server_state.subprocess, "run", Mock(side_effect=FileNotFoundError))
        monkeypatch.setattr(cmd_stop, "_DISCOVERY_MAX_MISSES", 1)
        monkeypatch.setattr(cmd_stop, "_DISCOVERY_RETRY_S", 0)
        probes = []

        def no_termination(pid, sig):
            probes.append((pid, sig))
            assert sig == 0, "An unverified PID must never receive a termination signal"

        monkeypatch.setattr(server_state.os, "kill", no_termination)

        assert server_state.find_own_serve_pid() is None
        assert cmd_stop.stop_tj_serve(quiet=True) == (False, [])
        assert probes and all(pid == 42424242 and sig == 0 for pid, sig in probes)

    @pytest.mark.parametrize("command, expected", [
        ("/usr/local/bin/tj --config /tmp/config.toml serve", True),
        ("/usr/bin/python worker.py", False),
        ("", False),
    ])
    def test_ps_fallback_requires_matching_identity(self, monkeypatch, command, expected):
        from subprocess import CompletedProcess
        from unittest.mock import Mock

        from tokenjam.core import server_state

        monkeypatch.setattr(server_state.Path, "exists", lambda path: False)
        probe = Mock(return_value=CompletedProcess([], 0, stdout=command))
        monkeypatch.setattr(server_state.subprocess, "run", probe)
        assert server_state.is_serve_process(42424242) is expected
        probe.assert_called_once_with(
            ["ps", "-ww", "-p", "42424242", "-o", "command="],
            capture_output=True, text=True,
        )

    def test_readable_proc_identity_does_not_require_ps(self, monkeypatch):
        from unittest.mock import Mock

        from tokenjam.core import server_state

        monkeypatch.setattr(server_state.Path, "exists", lambda path: True)
        monkeypatch.setattr(server_state.Path, "read_bytes", lambda path: b"tj\0serve\0")
        probe = Mock(side_effect=AssertionError("ps should not be needed"))
        monkeypatch.setattr(server_state.subprocess, "run", probe)
        assert server_state.is_serve_process(42424242) is True
        probe.assert_not_called()
