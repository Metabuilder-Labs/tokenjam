"""Parity guard for the two ``DataAccess`` implementations.

The CLI's ``DirectDataAccess`` (direct DuckDB) and ``ServeDataAccess`` (routed
through ``tj serve``) must return byte-identical computed results for the same
database — otherwise the daemon-up path silently drifts from the daemon-off
path, the exact class of bug the old ``hasattr``-sniffing seam produced (a shim
that quietly dropped fields). This runs both impls against ONE seeded DB (serve
side bridged into the in-process ASGI app) and asserts the serialized
diagnostic / audit + framing match.

Known, documented gap: ``diagnostic_to_dict`` carries ``heaviest_turns[].cost_usd``
for ``--json`` consumers, which ``diagnostic_from_dict`` rebuilds best-effort as
0.0 (never rendered — see its docstring). We compare everything the commands
actually render and exclude only that one best-effort field.
"""
from __future__ import annotations

import asyncio
import dataclasses

import httpx

from tokenjam.api.app import create_app
from tokenjam.cli.data_access import DirectDataAccess, ServeDataAccess
from tokenjam.core.api_backend import ApiBackend
from tokenjam.core.config import (
    CaptureConfig,
    ProviderBudget,
    StorageConfig,
    TjConfig,
)
from tokenjam.core.context_diagnostic import diagnostic_to_dict
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.optimize.types import audit_to_dict
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span


def _config() -> TjConfig:
    return TjConfig(
        version="1",
        budgets={"anthropic": ProviderBudget(plan="max_5x")},
        capture=CaptureConfig(tool_inputs=True),
    )


def _seed(db) -> None:
    """A mixed corpus: reread-heavy turns (context) + Opus sessions (audit)."""
    now = utcnow()
    # Reread-heavy turns + a recurring file read (context diagnostic surface).
    for i in range(4):
        sid = f"ctx-{i}"
        db.upsert_session(make_session(session_id=sid, plan_tier="max_5x"))
        db.insert_span(make_llm_span(
            session_id=sid, start_time=now,
            input_tokens=500, output_tokens=200, cache_tokens=120_000, cost_usd=1.0,
        ))
        tool = make_tool_span(tool_name="Read", tool_input={"file_path": "/repo/CLAUDE.md"})
        db.insert_span(dataclasses.replace(tool, session_id=sid, start_time=now))
    # Sonnet-shaped Opus sessions (quota-audit surface). Distinct token shapes
    # so the audit's token-sorted examples order deterministically — otherwise
    # tied sessions sort unstably between two independent query runs, which is
    # incidental to the seam parity this test guards.
    for i in range(3):
        sid = f"opus-{i}"
        db.upsert_session(make_session(session_id=sid, plan_tier="max_5x"))
        db.insert_span(make_llm_span(
            session_id=sid, model="claude-opus-4-7", start_time=now,
            input_tokens=500 + i * 100, output_tokens=120, cost_usd=2.0,
        ))


def _serve_access(app, monkeypatch) -> ServeDataAccess:
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
    return ServeDataAccess(api)


def _normalize_diag(payload: dict) -> dict:
    """Normalize a diagnostic dict for parity comparison.

    Drops (a) ``heaviest_turns[].cost_usd`` — the one documented best-effort
    field ``diagnostic_from_dict`` defaults to 0.0 — and (b) the ``since`` /
    ``until`` window-boundary timestamps, which each impl stamps from its own
    ``utcnow()`` call and so differ by microseconds of clock skew, not by any
    real drift in the computed analytics this guard protects.
    """
    out = {k: v for k, v in payload.items() if k not in ("since", "until")}
    out["heaviest_turns"] = [
        {k: v for k, v in t.items() if k != "cost_usd"}
        for t in payload.get("heaviest_turns", [])
    ]
    return out


def _normalize_audit(payload: dict) -> dict:
    """Round ``window_days`` — each impl computes it from its own ``utcnow()``,
    so it differs by microseconds of clock skew, not real drift."""
    out = dict(payload)
    out["window_days"] = round(float(payload.get("window_days", 0.0)), 3)
    return out


def test_context_diagnostic_parity(tmp_path, monkeypatch):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = _config()
    _seed(db)
    app = create_app(
        config=config, db=db,
        ingest_pipeline=IngestPipeline(db=db, config=config),
    )

    direct = DirectDataAccess(db, config)
    serve = _serve_access(app, monkeypatch)

    d_diag, d_framing = direct.context_diagnostic(since="30d", agent_id=None)
    s_diag, s_framing = serve.context_diagnostic(since="30d", agent_id=None)

    assert _normalize_diag(diagnostic_to_dict(d_diag)) == \
        _normalize_diag(diagnostic_to_dict(s_diag))
    assert d_framing.to_dict() == s_framing.to_dict()
    # Sanity: the seed actually produced the surface we care about (4 context +
    # 3 Opus sessions, each one LLM turn = 7 turns the diagnostic walks).
    assert d_diag.turns == 7
    assert d_framing.pricing_mode == "subscription"
    db.close()


def test_quota_audit_parity(tmp_path, monkeypatch):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = _config()
    _seed(db)
    app = create_app(
        config=config, db=db,
        ingest_pipeline=IngestPipeline(db=db, config=config),
    )

    direct = DirectDataAccess(db, config)
    serve = _serve_access(app, monkeypatch)

    d_audit, d_framing = direct.quota_audit(since="30d", agent_id=None)
    s_audit, s_framing = serve.quota_audit(since="30d", agent_id=None)

    # The audit round-trips fully (modulo window_days clock skew).
    assert _normalize_audit(audit_to_dict(d_audit)) == \
        _normalize_audit(audit_to_dict(s_audit))
    assert d_framing.to_dict() == s_framing.to_dict()
    assert d_audit.opus_sessions == 3
    assert d_framing.pricing_mode == "subscription"
    db.close()
