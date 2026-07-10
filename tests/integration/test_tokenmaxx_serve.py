"""`tj tokenmaxx` must render while `tj serve` is running, instead of refusing
with "stop the daemon" — the same daemon-up availability gap `tj context` (#63)
closed, now for the shareable quota/efficiency card.

`tj tokenmaxx` reads the same context-composition diagnostic as `tj context`,
so it reuses the EXISTING ``GET /api/v1/context`` endpoint (no new endpoint) via
the unified ``DataAccess`` seam. When the daemon holds the DuckDB write lock the
compute is routed through it; the CLI reconstructs the diagnostic and renders
the card. No backend sniffing in the command.
"""
from __future__ import annotations

import asyncio
import json

import httpx
from click.testing import CliRunner

from tokenjam.api.app import create_app
from tokenjam.cli.cmd_tokenmaxx import cmd_tokenmaxx
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
from tests.factories import make_llm_span, make_session


def _seed(db) -> None:
    """Reread-heavy LLM turns across 4 sessions → a high-overhead card."""
    now = utcnow()
    for i in range(4):
        sid = f"sess-{i}"
        db.upsert_session(make_session(session_id=sid, plan_tier="max_5x"))
        db.insert_span(make_llm_span(
            session_id=sid,
            start_time=now,
            input_tokens=500,
            output_tokens=200,
            cache_tokens=120_000,
            cost_usd=1.0,
        ))


def _config() -> TjConfig:
    return TjConfig(
        version="1",
        budgets={"anthropic": ProviderBudget(plan="max_5x")},
        capture=CaptureConfig(tool_inputs=True),
    )


def _apibackend_wired_to(app, monkeypatch) -> ApiBackend:
    """An ApiBackend whose HTTP calls route into the in-process ASGI app."""
    api = ApiBackend("http://daemon")
    transport = httpx.ASGITransport(app=app)

    def _sync_get(path, params=None, *, timeout=None):
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


def test_tokenmaxx_cli_renders_through_serve(tmp_path, monkeypatch):
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
        cmd_tokenmaxx, ["--since", "30d"],
        obj={"db": api, "config": config, "agent": None},
    )

    assert result.exit_code == 0, result.output
    # The old refuse-to-run error must be gone…
    assert "direct database connection" not in result.output
    # …and the efficiency card must actually render.
    assert "Card" in result.output or "Recap" in result.output
    assert "overhead" in result.output.lower()
    db.close()


def test_tokenmaxx_cli_json_through_serve(tmp_path, monkeypatch):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = _config()
    _seed(db)
    app = create_app(
        config=config, db=db,
        ingest_pipeline=IngestPipeline(db=db, config=config),
    )
    api = _apibackend_wired_to(app, monkeypatch)

    result = CliRunner().invoke(
        cmd_tokenmaxx, ["--since", "30d", "--json"],
        obj={"db": api, "config": config, "agent": None},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sessions"] == 4
    assert payload["turns"] == 4
    # Reread-heavy seed → high overhead share.
    assert payload["overhead_share"] > 0.9
    assert payload["pricing_mode"] == "subscription"
    db.close()


def test_tokenmaxx_cli_errors_cleanly_without_serve_or_conn():
    """A backend that is neither a live conn nor an ApiBackend → clean error."""
    class _Dummy:  # no .conn, not an ApiBackend
        pass

    result = CliRunner().invoke(
        cmd_tokenmaxx, ["--since", "30d"],
        obj={"db": _Dummy(), "config": _config(), "agent": None},
    )
    assert result.exit_code != 0
    assert "tj serve" in result.output
