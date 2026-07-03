"""#63 — `tj context` (the launch-hero command) must render while `tj serve` is
running, instead of refusing with "stop the daemon".

The composition diagnostic reads the raw ``attributes`` column for
recurring-inclusion detection, which the API shim doesn't expose row-by-row, and
DuckDB won't open a concurrent read-only connection alongside the serve
write-lock (it raises an IOException — only one writer OR many readers across
processes). So the daemon now computes the diagnostic server-side over
``GET /api/v1/context`` and the CLI renders the returned payload.

Two levels:
  * the endpoint computes the diagnostic (incl. recurring inclusions, which need
    the raw ``attributes``) + a plan-tier ``framing`` block, server-side;
  * ``tj context`` in API-shim mode (an ``ApiBackend`` with no ``.conn`` — the
    daemon-is-running state) fetches + renders it, without ever hitting the old
    "needs a direct database connection" error.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json

import httpx
import pytest
from click.testing import CliRunner

from tokenjam.api.app import create_app
from tokenjam.cli.cmd_context import cmd_context
from tokenjam.core.api_backend import ApiBackend
from tokenjam.core.config import (
    CaptureConfig,
    ProviderBudget,
    StorageConfig,
    TjConfig,
)
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span


_STOP_DAEMON_ERROR = "needs a direct database connection"


def _seed(db) -> None:
    """Reread-heavy LLM turns + a file re-read across 4 sessions.

    The LLM turns give the composition headline; the repeated ``Read`` of the
    same ``file_path`` across ≥3 distinct sessions is a recurring inclusion —
    detectable ONLY from the raw ``attributes`` the shim can't serve row-by-row,
    so it's the exact data path the endpoint has to cover.
    """
    now = utcnow()
    for i in range(4):
        sid = f"sess-{i}"
        db.upsert_session(make_session(session_id=sid, plan_tier="max_5x"))
        # A reread-heavy turn: cache reads dwarf net-new input + output.
        db.insert_span(make_llm_span(
            session_id=sid,
            start_time=now,
            input_tokens=500,
            output_tokens=200,
            cache_tokens=120_000,
            cost_usd=1.0,
        ))
        # The same file re-read every session → recurring file inclusion.
        tool = make_tool_span(
            tool_name="Read",
            tool_input={"file_path": "/repo/CLAUDE.md"},
        )
        db.insert_span(dataclasses.replace(tool, session_id=sid, start_time=now))


def _config() -> TjConfig:
    # tool_inputs capture ON so recurring file-read detection is exercised; a
    # max_5x budget so the framing block resolves to subscription mode.
    return TjConfig(
        version="1",
        budgets={"anthropic": ProviderBudget(plan="max_5x")},
        capture=CaptureConfig(tool_inputs=True),
    )


# ── endpoint-level: the daemon computes the diagnostic server-side ──────────

@pytest.mark.asyncio
async def test_context_endpoint_computes_diagnostic_server_side(tmp_path):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = _config()
    _seed(db)

    app = create_app(
        config=config, db=db,
        ingest_pipeline=IngestPipeline(db=db, config=config),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/api/v1/context", params={"since": "30d"})

    assert resp.status_code == 200
    payload = resp.json()
    # Composition computed over the seeded turns.
    assert payload["turns"] == 4
    assert payload["sessions"] == 4
    assert payload["total_reread_tokens"] == 480_000
    assert payload["reread_share"] > 0.9
    # Recurring inclusion detected from the raw `attributes` — the shim gap.
    targets = [r["target"] for r in payload["recurring"]]
    assert "/repo/CLAUDE.md" in targets
    # Framing block present + subscription (max_5x budget).
    assert payload["framing"]["pricing_mode"] == "subscription"
    db.close()


# ── CLI-shim: `tj context` renders through a running daemon (the #63 bug) ───

def _apibackend_wired_to(app, monkeypatch) -> ApiBackend:
    """An ApiBackend whose HTTP calls route into the in-process ASGI app.

    Faithfully drives the real route handler + real ApiBackend.fetch +
    real cmd_context shim path — only the socket is bridged. ApiBackend is sync
    and ASGITransport is async, so `_get` is bridged via asyncio.run.
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


def test_context_cli_renders_through_serve(tmp_path, monkeypatch):
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
        cmd_context, ["--since", "30d"],
        obj={"db": api, "config": config, "agent": None},
    )

    assert result.exit_code == 0, result.output
    # The old refuse-to-run error must be gone…
    assert _STOP_DAEMON_ERROR not in result.output
    # …and the diagnostic must actually render, including the recurring
    # inclusion that only the raw `attributes` path can surface.
    assert "Context composition" in result.output
    assert "CLAUDE.md" in result.output
    db.close()


def test_context_cli_json_through_serve(tmp_path, monkeypatch):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = _config()
    _seed(db)
    app = create_app(
        config=config, db=db,
        ingest_pipeline=IngestPipeline(db=db, config=config),
    )
    api = _apibackend_wired_to(app, monkeypatch)

    result = CliRunner().invoke(
        cmd_context, ["--since", "30d", "--json"],
        obj={"db": api, "config": config, "agent": None},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["turns"] == 4
    assert payload["framing"]["pricing_mode"] == "subscription"
    db.close()


def test_context_cli_errors_cleanly_without_serve_or_conn():
    """A backend that is neither a live conn nor an ApiBackend → clean error,
    not a traceback (the shim path guards on ApiBackend)."""
    class _Dummy:  # no .conn, not an ApiBackend
        pass

    result = CliRunner().invoke(
        cmd_context, ["--since", "30d"],
        obj={"db": _Dummy(), "config": _config(), "agent": None},
    )
    assert result.exit_code != 0
    assert "running tj serve" in result.output
