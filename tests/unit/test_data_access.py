"""ServeDataAccess must turn daemon transport / bad-payload failures into a
clean, actionable ClickException — never a raw httpx traceback.

A `tj serve` that crashes, restarts, or times out mid-command makes the
ApiBackend fetchers raise `httpx.ConnectError` / `httpx.ReadTimeout` (or
`HTTPStatusError` via `raise_for_status` on a 5xx); a version-skewed daemon can
return a malformed payload. The old bespoke serve path (`_render_via_serve`)
wrapped these; the DataAccess seam restores that guard at one boundary.
"""
from __future__ import annotations

import click
import httpx
import pytest

from tokenjam.cli.cmd_backfill import claude_code as cmd_backfill_claude_code
from tokenjam.cli.cmd_backfill import cmd_backfill
from tokenjam.cli.cmd_context import cmd_context
from tokenjam.cli.cmd_quota_audit import cmd_quota_audit
from tokenjam.cli.cmd_tokenmaxx import cmd_tokenmaxx
from tokenjam.cli.data_access import ServeDataAccess, resolve_data_access
from tokenjam.cli.tj_status import config_required_message, db_required_message
from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend


def _http_status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "http://daemon/api/v1/context")
    resp = httpx.Response(code, request=req)
    return httpx.HTTPStatusError(f"{code} Server Error", request=req, response=resp)


class _RaisingBackend:
    """An ApiBackend stand-in whose fetchers raise a chosen exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def fetch_context_diagnostic(self, **_kw):
        raise self._exc

    def fetch_opus_quota_audit(self, **_kw):
        raise self._exc


class _MalformedBackend:
    """Returns a non-dict payload (shape the reconstruction can't read)."""

    def fetch_context_diagnostic(self, **_kw):
        return ["not", "a", "dict"]

    def fetch_opus_quota_audit(self, **_kw):
        return ["not", "a", "dict"]


@pytest.mark.parametrize("method", ["context_diagnostic", "quota_audit"])
def test_serve_wraps_connect_error(method):
    sda = ServeDataAccess(_RaisingBackend(httpx.ConnectError("Connection refused")))
    with pytest.raises(click.ClickException) as ei:
        getattr(sda, method)(since="30d", agent_id=None)
    msg = str(ei.value)
    assert "tj serve" in msg
    assert "tj stop" in msg          # actionable: how to run directly
    assert "Connection refused" in msg  # the cause is surfaced


@pytest.mark.parametrize("method", ["context_diagnostic", "quota_audit"])
def test_serve_wraps_timeout(method):
    sda = ServeDataAccess(_RaisingBackend(httpx.ReadTimeout("timed out")))
    with pytest.raises(click.ClickException):
        getattr(sda, method)(since="30d", agent_id=None)


@pytest.mark.parametrize("method", ["context_diagnostic", "quota_audit"])
def test_serve_wraps_5xx_status(method):
    sda = ServeDataAccess(_RaisingBackend(_http_status_error(503)))
    with pytest.raises(click.ClickException) as ei:
        getattr(sda, method)(since="30d", agent_id=None)
    assert "tj serve" in str(ei.value)


@pytest.mark.parametrize("method", ["context_diagnostic", "quota_audit"])
def test_serve_wraps_malformed_payload(method):
    sda = ServeDataAccess(_MalformedBackend())
    with pytest.raises(click.ClickException) as ei:
        getattr(sda, method)(since="30d", agent_id=None)
    # Distinct copy for a shape/version mismatch vs a transport failure.
    assert "unreadable" in str(ei.value)


def _make_context_for(
    cmd: click.Command, info_name: str, parent: click.Context | None = None
) -> click.Context:
    return click.Context(cmd, info_name=info_name, parent=parent)


@pytest.mark.parametrize(
    ("expected_path", "build_ctx"),
    [
        ("tj context", lambda: _make_context_for(cmd_context, "tj context")),
        ("tj quota-audit", lambda: _make_context_for(cmd_quota_audit, "tj quota-audit")),
        ("tj tokenmaxx", lambda: _make_context_for(cmd_tokenmaxx, "tj tokenmaxx")),
        (
            "tj backfill claude-code",
            lambda: _make_context_for(
                cmd_backfill_claude_code,
                "claude-code",
                parent=_make_context_for(cmd_backfill, "tj backfill"),
            ),
        ),
    ],
)
def test_resolve_data_access_db_none_uses_shared_message_with_command_path(
    expected_path, build_ctx
):
    """Bypassing ``TjCommand.invoke``, ``resolve_data_access`` still emits the
    shared ``db_required_message`` naming ``ctx.command_path`` (#733)."""
    ctx = build_ctx()
    assert ctx.command_path == expected_path
    ctx.obj = {"db": None, "config": TjConfig(version="1")}

    with pytest.raises(click.ClickException) as ei:
        resolve_data_access(ctx)

    assert str(ei.value) == db_required_message(expected_path)
    assert "this command" not in str(ei.value)


def test_resolve_data_access_unusable_backend_uses_shared_message_with_command_path():
    class _UnusableBackend:
        pass

    ctx = _make_context_for(cmd_context, "tj context")
    ctx.obj = {"db": _UnusableBackend(), "config": TjConfig(version="1")}

    with pytest.raises(click.ClickException) as ei:
        resolve_data_access(ctx)

    assert str(ei.value) == db_required_message("tj context")
    assert "this command" not in str(ei.value)


@pytest.mark.parametrize(
    ("expected_path", "build_ctx"),
    [
        ("tj context", lambda: _make_context_for(cmd_context, "tj context")),
        ("tj quota-audit", lambda: _make_context_for(cmd_quota_audit, "tj quota-audit")),
        ("tj tokenmaxx", lambda: _make_context_for(cmd_tokenmaxx, "tj tokenmaxx")),
        (
            "tj backfill claude-code",
            lambda: _make_context_for(
                cmd_backfill_claude_code,
                "claude-code",
                parent=_make_context_for(cmd_backfill, "tj backfill"),
            ),
        ),
    ],
)
def test_resolve_data_access_config_none_reports_configuration_error(
    expected_path, build_ctx
):
    """``config is None`` must report a configuration error, not a missing DB (#733)."""
    db = InMemoryBackend()
    try:
        ctx = build_ctx()
        ctx.obj = {"db": db, "config": None}

        with pytest.raises(click.ClickException) as ei:
            resolve_data_access(ctx)

        msg = str(ei.value)
        assert msg == config_required_message(expected_path)
        assert "could not load configuration" in msg
        assert "database" not in msg.lower()
        assert "duckdb" not in msg.lower()
    finally:
        db.close()
