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

from tokenjam.cli.data_access import ServeDataAccess


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
