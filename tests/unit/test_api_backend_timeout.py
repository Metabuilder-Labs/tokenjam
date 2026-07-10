"""The ApiBackend HTTP client keeps a tight blanket timeout for cheap shim
reads but overrides it for the heavy *computed* endpoints.

Regression for the onboarding-killer bug: with `tj serve` holding the DuckDB
write lock, `tj quota-audit` routes through `GET /api/v1/quota-audit`, which the
daemon computes in ~13s on a large history (3,424 sessions / 150k turns). Under
the old blanket 10s client timeout that raised `httpx.ReadTimeout`, so the very
next step onboarding advertises failed deterministically for exactly the
large-history users tj most wants. The fix is a per-request read-timeout
override on the heavy computed fetchers, keeping the 10s default for the cheap
reads.

We assert on the timeout httpx actually attaches to each outgoing request
(`request.extensions["timeout"]`), so this pins the wire-level behavior, not an
internal constant.
"""
from __future__ import annotations

import httpx
import pytest

from tokenjam.core.api_backend import ApiBackend


def _backend_recording_timeouts() -> tuple[ApiBackend, dict[str, float | None]]:
    """An ApiBackend whose transport records the per-request *read* timeout.

    Preserves the real client's blanket default timeout so the cheap-read path
    is faithfully exercised; only the socket is swapped for a MockTransport.
    """
    read_timeouts: dict[str, float | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        timeout = request.extensions.get("timeout") or {}
        read_timeouts[request.url.path] = timeout.get("read")
        return httpx.Response(200, json={})

    api = ApiBackend("http://daemon")
    # Rebuild the client with the SAME default timeout the constructor uses,
    # over a MockTransport, so an un-overridden `_get` still sees the 10s ceiling.
    api.client = httpx.Client(
        base_url="http://daemon",
        timeout=ApiBackend._DEFAULT_TIMEOUT,
        transport=httpx.MockTransport(handler),
    )
    return api, read_timeouts


@pytest.mark.parametrize(
    "call, path",
    [
        (lambda api: api.fetch_opus_quota_audit(since="30d"), "/api/v1/quota-audit"),
        (lambda api: api.fetch_context_diagnostic(since="30d"), "/api/v1/context"),
        (lambda api: api.fetch_optimize_report(since="30d"), "/api/v1/optimize"),
        (lambda api: api.fetch_reuse_clusters(since="30d"), "/api/v1/reuse/clusters"),
        (lambda api: api.fetch_cost_compare(since="30d"), "/api/v1/cost/compare"),
    ],
)
def test_heavy_endpoints_use_extended_read_timeout(call, path):
    api, read_timeouts = _backend_recording_timeouts()
    call(api)
    # The heavy computed endpoint must ride the 60s override, not the 10s
    # cheap-read ceiling — otherwise a >10s server-side compute ReadTimeouts.
    assert read_timeouts[path] == ApiBackend._HEAVY_ENDPOINT_TIMEOUT
    assert read_timeouts[path] > ApiBackend._DEFAULT_TIMEOUT
    api.close()


def test_cheap_reads_keep_the_tight_default_timeout():
    from tokenjam.core.models import CostFilters, TraceFilters

    api, read_timeouts = _backend_recording_timeouts()
    api.get_traces(TraceFilters(limit=1, offset=0))
    api.get_cost_summary(CostFilters())
    # Cheap shim reads must NOT inherit the heavy override — a wedged daemon
    # should still fail fast for these near-instant lookups.
    assert read_timeouts["/api/v1/traces"] == ApiBackend._DEFAULT_TIMEOUT
    assert read_timeouts["/api/v1/cost"] == ApiBackend._DEFAULT_TIMEOUT
    api.close()
