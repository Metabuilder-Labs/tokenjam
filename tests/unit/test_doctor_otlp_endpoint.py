"""`tj doctor`'s OTLP endpoint check: resolve, connect, authenticate.

The fault it exists to catch: onboarding wrote a container-only hostname into
a host user's ~/.zshrc, so every span an agent session emitted failed at DNS
resolution and was dropped. Nothing reported it. Token and cost figures still
appeared, because a separate transcript backfill recovers those in delayed
batches, which is exactly why the loss went unnoticed for so long: the event
types with no transcript equivalent (tool decisions, API errors) simply never
existed.

Each layer gets its own verdict, and the check reads the endpoint out of the
shell profile rather than the config, because the shell profile is what an
agent session actually inherits.
"""
from __future__ import annotations

import socket

import pytest

from tokenjam.cli.cmd_doctor import _check_otlp_endpoint
from tokenjam.core.config import ApiConfig, SecurityConfig, TjConfig

_BLOCK_SECRET = "block-secret-value"


def _config() -> TjConfig:
    return TjConfig(
        version="1",
        api=ApiConfig(host="127.0.0.1", port=7391),
        security=SecurityConfig(ingest_secret=_BLOCK_SECRET),
    )


def _write_block(home, endpoint: str, secret: str = _BLOCK_SECRET) -> None:
    (home / ".zshrc").write_text(
        "# my own env\n"
        "# >>> tokenjam OTEL (managed) >>>\n"
        "export CLAUDE_CODE_ENABLE_TELEMETRY=1\n"
        "export OTEL_LOGS_EXPORTER=otlp\n"
        "export OTEL_EXPORTER_OTLP_PROTOCOL=http/json\n"
        f"export OTEL_EXPORTER_OTLP_ENDPOINT={endpoint}\n"
        f'export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer {secret}"\n'
        "# <<< tokenjam OTEL <<<\n"
    )


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: h)
    return h


def test_no_managed_block_is_reported_as_unknown_not_as_healthy(home):
    """No block installed is not a passing endpoint. It is "nothing configured
    here", and it must not render as a green reachability verdict."""
    (home / ".zshrc").write_text("export FOO=bar\n")
    check = _check_otlp_endpoint(_config())
    assert check["level"] == "info"
    assert "No tj-managed OTel block" in check["message"]


def test_unresolvable_hostname_is_an_error(home, monkeypatch):
    """The reported bug, verbatim: a hostname with no address on this machine."""
    _write_block(home, "http://host.docker.internal:7391")

    def _no_such_host(*a, **k):
        raise socket.gaierror("nodename nor servname provided")

    monkeypatch.setattr(socket, "getaddrinfo", _no_such_host)
    check = _check_otlp_endpoint(_config())
    assert check["level"] == "error"
    assert "does not resolve" in check["message"]
    assert "host.docker.internal" in check["message"]


def test_nothing_listening_is_a_warning_and_names_the_port_mismatch(home, monkeypatch):
    _write_block(home, "http://127.0.0.1:7500")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [()])

    def _refused(*a, **k):
        raise ConnectionRefusedError(61, "Connection refused")

    monkeypatch.setattr(socket, "create_connection", _refused)
    check = _check_otlp_endpoint(_config())
    assert check["level"] == "warning"
    assert "nothing is listening" in check["message"]
    # The daemon binds 7391; the block names 7500. Say so.
    assert "7391" in check["message"] and "7500" in check["message"]
    assert "tj serve" in check["message"]


class _Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_reachable_and_authenticated_is_ok(home, monkeypatch):
    _write_block(home, "http://127.0.0.1:7391")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [()])
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _NullSocket())
    seen: dict = {}

    def _get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["auth"] = (headers or {}).get("Authorization")
        return _Resp(200)

    monkeypatch.setattr("httpx.get", _get)
    check = _check_otlp_endpoint(_config())
    assert check["level"] == "ok"
    assert seen["url"] == "http://127.0.0.1:7391/api/v1/status"
    # The probe authenticates with the secret the block itself signs with,
    # which is the only way to answer "would this session's spans be accepted".
    assert seen["auth"] == f"Bearer {_BLOCK_SECRET}"


def test_rejected_secret_is_an_error(home, monkeypatch):
    """Reachable but 401: spans arrive and are discarded, which looks
    identical to a healthy pipeline from the sending side."""
    _write_block(home, "http://127.0.0.1:7391", secret="a-different-secret")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [()])
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _NullSocket())
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp(401))

    check = _check_otlp_endpoint(_config())
    assert check["level"] == "error"
    assert "401" in check["message"]


def test_no_check_message_ever_carries_a_secret(home, monkeypatch):
    """A health report is copied into issues and pasted into chats."""
    _write_block(home, "http://127.0.0.1:7391")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [()])
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _NullSocket())
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp(401))

    check = _check_otlp_endpoint(_config())
    assert _BLOCK_SECRET not in check["message"]


class _NullSocket:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
