"""Unit tests for `tj serve`'s port-in-use pre-check (issue #509).

Before starting uvicorn, `tj serve` must detect an already-bound port and
fail fast with an actionable message, rather than printing "Application
startup complete" and THEN an EADDRINUSE bind error that reads like the
server booted and crashed.
"""
from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tokenjam.cli.cmd_serve import _port_in_use, cmd_serve
from tokenjam.core.config import TjConfig


def _serve_ctx_obj():
    """Minimal ctx.obj for cmd_serve — the pre-check runs before the db/pipeline
    are touched, so a bare config plus a placeholder db is enough."""
    return {"config": TjConfig(version="1"), "db": MagicMock()}


class TestPortInUsePreCheck:
    def test_fails_fast_with_clear_message_when_port_taken(self):
        """The pre-check message names the port and points at `tj stop` /
        `--port`, and uvicorn is never reached."""
        with patch("tokenjam.cli.cmd_serve._port_in_use", return_value=True), \
             patch("uvicorn.run") as run_mock:
            result = CliRunner().invoke(cmd_serve, [], obj=_serve_ctx_obj())

        assert result.exit_code == 1, result.output
        assert "already in use" in result.output
        assert "tj stop" in result.output
        assert "--port" in result.output
        # Never fell through to actually starting the server.
        run_mock.assert_not_called()

    def test_starts_when_port_free(self):
        """When the port is free the pre-check is a no-op and uvicorn starts."""
        with patch("tokenjam.cli.cmd_serve._port_in_use", return_value=False), \
             patch("uvicorn.run") as run_mock, \
             patch("tokenjam.core.ingest.build_default_pipeline",
                   return_value=MagicMock()), \
             patch("tokenjam.api.app.create_app", return_value=MagicMock()):
            result = CliRunner().invoke(cmd_serve, [], obj=_serve_ctx_obj())

        assert result.exit_code == 0, result.output
        run_mock.assert_called_once()


class TestPortInUseDetection:
    def test_detects_bound_port(self):
        """A held socket makes _port_in_use return True for that port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]
            assert _port_in_use("127.0.0.1", port) is True

    def test_free_port_is_not_in_use(self):
        """A port that nothing holds reads as free. We grab a free port,
        release it, then check — a benign race, adequate for this assertion."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        assert _port_in_use("127.0.0.1", port) is False

    def test_ipv6_host_free_port_is_not_in_use(self):
        """An IPv6 bind_host must not be misread as a conflict: the pre-check
        picks AF_INET6 by host, so a free port on `::1` reads as free rather
        than raising a bogus OSError against an IPv4 socket (Greptile P1)."""
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
            port = sock.getsockname()[1]
        assert _port_in_use("::1", port) is False

    def test_ipv6_host_detects_bound_port(self):
        """A held IPv6 socket reads as in-use for that host/port."""
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as held:
            held.bind(("::1", 0))
            held.listen(1)
            port = held.getsockname()[1]
            assert _port_in_use("::1", port) is True
