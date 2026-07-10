"""`tj quota-audit` must render while `tj serve` is running, instead of refusing
with "stop the daemon" — the same daemon-up availability gap `tj context` (#63)
closed, now for the retroactive Opus quota audit.

The audit aggregates per-session Opus token/model metadata at a grain the
read-only API shim never exposed, and DuckDB won't open a concurrent read-only
connection alongside the serve write-lock. So the daemon computes the audit
server-side over ``GET /api/v1/quota-audit`` and the CLI renders the returned
payload, dispatched through the unified ``DataAccess`` seam (no backend
sniffing).

Two levels:
  * the endpoint computes the audit + a plan-tier ``framing`` block server-side;
  * ``tj quota-audit`` in API-shim mode (an ``ApiBackend`` with no ``.conn`` —
    the daemon-is-running state) fetches + renders it, without the old
    "needs a direct database connection" refusal.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from click.testing import CliRunner

from tokenjam.api.app import create_app
from tokenjam.cli.cmd_quota_audit import cmd_quota_audit
from tokenjam.core.api_backend import ApiBackend
from tokenjam.core.config import (
    ProviderBudget,
    StorageConfig,
    TjConfig,
)
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


_STOP_DAEMON_ERROR = "direct database connection"


def _seed(db) -> None:
    """Opus sessions whose spans are Sonnet-shaped (small in/out, no tools).

    `claude-opus-4-7` is premium-tier with a known cheaper alternative, so the
    shared tier predicate flags these sessions; the small token shape makes them
    reclaim candidates. This is
    exactly the per-session token/model aggregation the shim can't serve — the
    data path the endpoint has to cover.
    """
    now = utcnow()
    for i in range(3):
        sid = f"opus-{i}"
        db.upsert_session(make_session(session_id=sid, plan_tier="max_5x"))
        db.insert_span(make_llm_span(
            session_id=sid,
            model="claude-opus-4-7",
            start_time=now,
            input_tokens=500,
            output_tokens=120,
            cost_usd=2.0,
        ))


def _config() -> TjConfig:
    # max_5x budget so the framing block resolves to subscription mode.
    return TjConfig(
        version="1",
        budgets={"anthropic": ProviderBudget(plan="max_5x")},
    )


# ── endpoint-level: the daemon computes the audit server-side ───────────────

@pytest.mark.asyncio
async def test_quota_audit_endpoint_computes_audit_server_side(tmp_path):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = _config()
    _seed(db)

    app = create_app(
        config=config, db=db,
        ingest_pipeline=IngestPipeline(db=db, config=config),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/api/v1/quota-audit", params={"since": "30d"})

    assert resp.status_code == 200
    payload = resp.json()
    # Three Opus sessions, all Sonnet-shaped → fully reclaimable.
    assert payload["opus_sessions"] == 3
    assert payload["candidate_sessions"] == 3
    assert payload["percent_quota_reclaimable"] > 0
    # The mandatory honesty caveat rides through the payload.
    assert "spot-check" in payload["caveat"].lower()
    # Framing block present + subscription (max_5x budget).
    assert payload["framing"]["pricing_mode"] == "subscription"
    db.close()


# ── CLI-shim: `tj quota-audit` renders through a running daemon ─────────────

def _apibackend_wired_to(app, monkeypatch) -> ApiBackend:
    """An ApiBackend whose HTTP calls route into the in-process ASGI app.

    Faithfully drives the real route handler + real ApiBackend.fetch + real
    cmd_quota_audit shim path — only the socket is bridged.
    """
    api = ApiBackend("http://daemon")
    transport = httpx.ASGITransport(app=app)

    def _sync_get(path, params=None):
        async def _call():
            async with httpx.AsyncClient(
                transport=transport, base_url="http://daemon",
            ) as c:
                r = await c.get(path, params=params)
                r.raise_for_status()
                return r.json()
        return asyncio.run(_call())

    monkeypatch.setattr(api, "_get", _sync_get)
    return api


def test_quota_audit_cli_renders_through_serve(tmp_path, monkeypatch):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = _config()
    _seed(db)
    app = create_app(
        config=config, db=db,
        ingest_pipeline=IngestPipeline(db=db, config=config),
    )
    api = _apibackend_wired_to(app, monkeypatch)

    # `api` has no `.conn` — exactly the state the CLI is in when `tj serve`
    # holds the DuckDB write lock.
    assert getattr(api, "conn", None) is None

    result = CliRunner().invoke(
        cmd_quota_audit, ["--since", "30d"],
        obj={"db": api, "config": config, "agent": None},
    )

    assert result.exit_code == 0, result.output
    # The old refuse-to-run error must be gone…
    assert _STOP_DAEMON_ERROR not in result.output
    # …and the audit must actually render.
    assert "Premium quota audit" in result.output
    db.close()


def test_quota_audit_cli_json_through_serve(tmp_path, monkeypatch):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = _config()
    _seed(db)
    app = create_app(
        config=config, db=db,
        ingest_pipeline=IngestPipeline(db=db, config=config),
    )
    api = _apibackend_wired_to(app, monkeypatch)

    result = CliRunner().invoke(
        cmd_quota_audit, ["--since", "30d", "--json"],
        obj={"db": api, "config": config, "agent": None},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["opus_sessions"] == 3
    assert payload["candidate_sessions"] == 3
    assert payload["framing"]["pricing_mode"] == "subscription"
    db.close()


def test_quota_audit_cli_errors_cleanly_without_serve_or_conn():
    """A backend that is neither a live conn nor an ApiBackend → clean error,
    not a traceback (the DataAccess seam guards on ApiBackend)."""
    class _Dummy:  # no .conn, not an ApiBackend
        pass

    result = CliRunner().invoke(
        cmd_quota_audit, ["--since", "30d"],
        obj={"db": _Dummy(), "config": _config(), "agent": None},
    )
    assert result.exit_code != 0
    assert "tj serve" in result.output
