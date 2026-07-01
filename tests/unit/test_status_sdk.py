"""Tests for the /status SDK-services zone: per-minute sparkline series
(`sdk_service_series`) + the last-seen-keyed lifecycle (`_build_sdk_services`)
and the coding/sdk `kind` tag on the status route.
"""
from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.api.routes.status import _build_sdk_services
from tokenjam.core.config import ApiAuthConfig, ApiConfig, SecurityConfig, TjConfig
from tokenjam.core.db import InMemoryBackend, sdk_service_series
from tokenjam.core.ingest import IngestPipeline
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


# ── sdk_service_series (per-minute buckets + zero-fill) ─────────────────────

def test_series_buckets_and_zero_fills():
    db = InMemoryBackend()
    try:
        now = utcnow()
        # Two spans in the current minute: one ok, one error, $0.01 each.
        db.insert_span(make_llm_span(agent_id="svc-x", cost_usd=0.01,
                                     status="ok", start_time=now))
        db.insert_span(make_llm_span(agent_id="svc-x", cost_usd=0.01,
                                     status="error", start_time=now))
        window_start = now - timedelta(minutes=24)
        out = sdk_service_series(db.conn, ["svc-x"], window_start, now, slots=24)
        s = out["svc-x"]

        assert len(s["cost_per_min"]) == 24
        assert len(s["calls_per_min"]) == 24
        assert len(s["err_pct_per_min"]) == 24
        # The current minute is the last slot.
        assert s["calls_per_min"][-1] == 2
        assert s["cost_per_min"][-1] == pytest.approx(0.02)
        assert s["err_pct_per_min"][-1] == pytest.approx(50.0)
        # Earlier slots are a flatline.
        assert s["calls_per_min"][0] == 0
        assert s["err_pct_per_min"][0] == 0.0
        # Window totals.
        assert s["window_calls"] == 2
        assert s["window_errors"] == 1
        assert s["window_cost"] == pytest.approx(0.02)
        assert s["last_seen"] is not None
    finally:
        db.close()


def test_series_flatline_when_no_recent_spans():
    db = InMemoryBackend()
    try:
        now = utcnow()
        old = now - timedelta(days=2)
        db.insert_span(make_llm_span(agent_id="svc-old", cost_usd=0.5,
                                     start_time=old))
        window_start = now - timedelta(minutes=24)
        out = sdk_service_series(db.conn, ["svc-old"], window_start, now, slots=24)
        s = out["svc-old"]

        assert s["calls_per_min"] == [0] * 24
        assert s["err_pct_per_min"] == [0.0] * 24
        assert s["window_calls"] == 0
        # last_seen comes from an all-history query, not the sparkline window.
        assert s["last_seen"] is not None
    finally:
        db.close()


def test_series_empty_for_no_agents():
    db = InMemoryBackend()
    try:
        now = utcnow()
        assert sdk_service_series(db.conn, [], now - timedelta(minutes=24), now) == {}
    finally:
        db.close()


# ── _build_sdk_services (lifecycle state machine + kind) ────────────────────

def test_build_classifies_live_quiet_dormant_and_excludes_coding():
    db = InMemoryBackend()
    try:
        now = utcnow()
        db.insert_span(make_llm_span(agent_id="svc-live",
                                     start_time=now, cost_usd=0.01))
        db.insert_span(make_llm_span(agent_id="svc-quiet",
                                     start_time=now - timedelta(minutes=20),
                                     cost_usd=0.01))
        db.insert_span(make_llm_span(agent_id="svc-dormant",
                                     start_time=now - timedelta(minutes=45),
                                     cost_usd=0.01))
        # An interactive coding agent must NOT appear in the SDK zone.
        db.insert_span(make_llm_span(agent_id="claude-code-x",
                                     start_time=now, cost_usd=0.01))
        agent_ids = ["svc-live", "svc-quiet", "svc-dormant", "claude-code-x"]

        out = _build_sdk_services(db, None, agent_ids, now)
        by = {s["agent_id"]: s for s in out}

        assert "claude-code-x" not in by
        assert by["svc-live"]["state"] == "live"
        assert by["svc-quiet"]["state"] == "went_quiet"
        assert by["svc-dormant"]["state"] == "long_dormant"
        assert all(s["kind"] == "sdk" for s in out)
        # Ordering: live first, then went_quiet, then long_dormant.
        assert [s["state"] for s in out] == ["live", "went_quiet", "long_dormant"]
        # Series ride along.
        assert len(by["svc-live"]["cost_per_min"]) == 24
    finally:
        db.close()


def test_build_computes_err_rate_and_req_per_min():
    db = InMemoryBackend()
    try:
        now = utcnow()
        # 3 calls this minute, 1 an error -> err_rate 33.3%, req/min = 3/24.
        db.insert_span(make_llm_span(agent_id="svc-e", start_time=now,
                                     status="ok", cost_usd=0.01))
        db.insert_span(make_llm_span(agent_id="svc-e", start_time=now,
                                     status="ok", cost_usd=0.01))
        db.insert_span(make_llm_span(agent_id="svc-e", start_time=now,
                                     status="error", cost_usd=0.01))
        out = _build_sdk_services(db, None, ["svc-e"], now)
        svc = out[0]
        assert svc["err_rate"] == pytest.approx(100.0 / 3)
        assert svc["req_per_min"] == pytest.approx(3 / 24)
        assert svc["window_cost"] == pytest.approx(0.03)
    finally:
        db.close()


def test_build_excludes_beyond_discovery_window():
    db = InMemoryBackend()
    try:
        now = utcnow()
        db.insert_span(make_llm_span(agent_id="svc-ancient",
                                     start_time=now - timedelta(days=10),
                                     cost_usd=0.01))
        assert _build_sdk_services(db, None, ["svc-ancient"], now) == []
    finally:
        db.close()


# ── /status route end-to-end (kind + sdk_services) ─────────────────────────

@pytest.fixture
def _cfg():
    return TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="test-secret"),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
    )


@pytest.fixture
def _db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def _client(_cfg, _db):
    pipeline = IngestPipeline(db=_db, config=_cfg)
    app = create_app(config=_cfg, db=_db, ingest_pipeline=pipeline)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_status_route_returns_sdk_services_live(_db, _client):
    now = utcnow()
    _db.insert_span(make_llm_span(agent_id="svc-checkout",
                                  start_time=now, cost_usd=0.02))
    resp = await _client.get("/api/v1/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "sdk_services" in data
    svc = [s for s in data["sdk_services"] if s["agent_id"] == "svc-checkout"]
    assert svc, data["sdk_services"]
    assert svc[0]["kind"] == "sdk"
    assert svc[0]["state"] == "live"
    assert len(svc[0]["cost_per_min"]) == 24


async def test_status_route_tags_coding_kind_on_archive(_db, _client):
    now = utcnow()
    _db.upsert_session(make_session(
        agent_id="claude-code-tokenjam", session_id="sess-1", status="closed",
        input_tokens=1000, output_tokens=200, tool_call_count=5,
        started_at=now - timedelta(hours=1), ended_at=now - timedelta(minutes=30),
    ))
    resp = await _client.get("/api/v1/status")
    data = resp.json()
    arch = [a for a in data["archived"] if a["agent_id"] == "claude-code-tokenjam"]
    assert arch, data["archived"]
    assert arch[0]["kind"] == "coding"


async def test_status_archive_filters_zero_signal_zombies(_db, _client):
    # A terminal that opened and did nothing (0 tokens, 0 tool calls) carries no
    # method/cost, so it must not clutter the archive. A session with any signal
    # is kept.
    now = utcnow()
    _db.upsert_session(make_session(
        agent_id="claude-code-zombie", session_id="z-1", status="closed",
        input_tokens=0, output_tokens=0, tool_call_count=0,
        started_at=now - timedelta(hours=2), ended_at=now - timedelta(hours=1),
    ))
    _db.upsert_session(make_session(
        agent_id="claude-code-real", session_id="r-1", status="closed",
        input_tokens=500, output_tokens=100, tool_call_count=3,
        started_at=now - timedelta(hours=2), ended_at=now - timedelta(hours=1),
    ))
    resp = await _client.get("/api/v1/status")
    archived_ids = {a["agent_id"] for a in resp.json()["archived"]}
    assert "claude-code-real" in archived_ids
    assert "claude-code-zombie" not in archived_ids
