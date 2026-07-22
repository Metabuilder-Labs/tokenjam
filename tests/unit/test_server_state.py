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
