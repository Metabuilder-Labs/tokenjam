"""A second `tj serve` that lost the DuckDB write-lock race runs as a thin
proxy: its ``app.state.db`` is an ``ApiBackend`` pointing at the primary daemon,
not a ``DuckDBBackend``. ``GET /api/v1/cost/compare`` used to call
``compute_cost_diff(db, ...)`` unconditionally, which reaches for
``db.get_window_cost_totals`` / ``db.get_cost_delta_by_group`` — methods only the
direct DuckDB backend has. Under the proxy that raised
``AttributeError: 'ApiBackend' object has no attribute 'get_window_cost_totals'``
→ a 500.

The route now detects the proxy case and forwards the whole comparison to the
primary daemon (which owns the direct connection) via
``ApiBackend.fetch_cost_compare``, returning the same schema a direct hit does.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.api_backend import ApiBackend
from tokenjam.core.config import ApiAuthConfig, ApiConfig, StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session

_COMPARE_URL = "/api/v1/cost/compare?since=7d&compare=previous"
# The DuckDBBackend methods compute_cost_diff needs but the proxy shim lacks.
_PROXY_ONLY_MISSING = "get_window_cost_totals"


def _config() -> TjConfig:
    # Auth off so the route's require_api_key passes without a bearer header;
    # the proxy scenario itself is independent of auth.
    return TjConfig(
        version="1",
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
    )


def _seed(db) -> None:
    """LLM spans in both the current (7d) and previous (7-14d) windows so the
    comparison has real numbers on each side of the diff."""
    now = utcnow()
    db.upsert_session(make_session(agent_id="a", session_id="cur"))
    db.upsert_session(make_session(agent_id="a", session_id="prev"))
    db.insert_span(make_llm_span(
        session_id="cur", agent_id="a", start_time=now - timedelta(days=1),
        input_tokens=1000, output_tokens=300, cost_usd=2.0,
    ))
    db.insert_span(make_llm_span(
        session_id="prev", agent_id="a", start_time=now - timedelta(days=10),
        input_tokens=800, output_tokens=200, cost_usd=1.0,
    ))


def _proxy_apibackend(primary_app, monkeypatch) -> ApiBackend:
    """An ApiBackend whose sync ``_get`` is bridged into the primary daemon's
    in-process ASGI app — exactly the object a second `tj serve` puts in
    ``app.state.db`` when the first daemon holds the write lock."""
    api = ApiBackend("http://primary")
    transport = httpx.ASGITransport(app=primary_app)

    def _sync_get(path, params=None, *, timeout=None):
        async def _call():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://primary",
            ) as c:
                r = await c.get(path, params=params)
                r.raise_for_status()
                return r.json()

        return asyncio.run(_call())

    monkeypatch.setattr(api, "_get", _sync_get)
    return api


def test_apibackend_lacks_direct_compare_methods():
    """Guards the premise: the proxy shim genuinely can't compute the diff
    locally, so the route MUST forward rather than call compute_cost_diff."""
    assert not hasattr(ApiBackend, _PROXY_ONLY_MISSING)
    assert not hasattr(ApiBackend, "get_cost_delta_by_group")
    # …but it CAN forward the whole comparison to the primary.
    assert hasattr(ApiBackend, "fetch_cost_compare")


@pytest.mark.asyncio
async def test_cost_compare_forwards_through_proxy(tmp_path, monkeypatch):
    """Hitting /cost/compare on the proxy daemon returns the real diff (200 +
    full schema), not the old AttributeError 500."""
    primary_db = DuckDBBackend(StorageConfig(path=str(tmp_path / "primary.duckdb")))
    config = _config()
    _seed(primary_db)
    primary_app = create_app(
        config=config, db=primary_db,
        ingest_pipeline=IngestPipeline(db=primary_db, config=config),
    )

    api = _proxy_apibackend(primary_app, monkeypatch)
    # The proxy daemon's state.db is the ApiBackend — no direct connection.
    assert getattr(api, "conn", None) is None

    proxy_app = create_app(
        config=config, db=api,
        ingest_pipeline=IngestPipeline(db=primary_db, config=config),
    )
    assert isinstance(proxy_app.state.db, ApiBackend)

    transport = httpx.ASGITransport(app=proxy_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
        resp = await c.get(_COMPARE_URL)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Full CostDiff schema forwarded verbatim from the primary.
    for key in (
        "current", "previous", "cost_delta_usd", "cost_delta_pct",
        "tokens_delta", "by_agent", "by_model", "framing",
    ):
        assert key in body, f"missing {key!r} in forwarded response"
    # Real numbers from the seeded windows (current $2 vs previous $1).
    assert body["current"]["total_cost_usd"] == pytest.approx(2.0)
    assert body["previous"]["total_cost_usd"] == pytest.approx(1.0)
    assert body["cost_delta_usd"] == pytest.approx(1.0)

    primary_db.close()


@pytest.mark.asyncio
async def test_cost_compare_proxy_degrades_on_upstream_failure(tmp_path, monkeypatch):
    """If the primary daemon is unreachable, the proxy returns a clean 502 —
    not a raw proxy traceback."""
    config = _config()
    # An ApiBackend whose forward always fails at the transport layer.
    api = ApiBackend("http://primary")

    def _boom(path, params=None, *, timeout=None):
        raise httpx.ConnectError("primary daemon down")

    monkeypatch.setattr(api, "_get", _boom)

    # A throwaway real db just to satisfy create_app / the pipeline; the route
    # never touches it because state.db is the ApiBackend.
    throwaway = DuckDBBackend(StorageConfig(path=str(tmp_path / "throwaway.duckdb")))
    proxy_app = create_app(
        config=config, db=api,
        ingest_pipeline=IngestPipeline(db=throwaway, config=config),
    )

    transport = httpx.ASGITransport(app=proxy_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
        resp = await c.get(_COMPARE_URL)

    assert resp.status_code == 502, resp.text
    assert "Upstream tj serve unavailable" in resp.json()["detail"]
    throwaway.close()
