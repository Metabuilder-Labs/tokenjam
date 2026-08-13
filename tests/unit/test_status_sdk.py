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
    """The live panel is bounded by SDK_DISCOVERY_WINDOW, and that stays.

    DO NOT "fix" this by widening the window. `live` / `went_quiet` /
    `long_dormant` are all defined against it, so widening changes what those
    three words mean and only moves the cliff further out. An SDK agent that
    ages out of this panel must DEGRADE INTO HISTORY instead — see
    `test_an_sdk_agent_past_the_discovery_window_is_still_reachable_as_history`,
    which pins the other half. Half of this behaviour on its own is the bug:
    the SDK persona's only session surface rendered this panel and nothing
    else, so an agent quiet for longer than the window left the product.
    """
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


# ── lifecycle-status -> zone mapping (live tiles vs archive) ───────────────

async def test_status_archives_completed_sessions(_db, _client):
    # Backfilled Claude Code sessions land with status='completed'. It is a
    # terminal status, so the session belongs in the archive — not dropped from
    # both zones because the archive SQL only knew about 'closed'.
    now = utcnow()
    _db.upsert_session(make_session(
        agent_id="claude-code-done", session_id="c-1", status="completed",
        input_tokens=800, output_tokens=150, tool_call_count=4,
        started_at=now - timedelta(hours=2), ended_at=now - timedelta(hours=1),
    ))
    data = (await _client.get("/api/v1/status")).json()
    arch = [a for a in data["archived"] if a["session_id"] == "c-1"]
    assert arch, data["archived"]
    assert arch[0]["status"] == "completed"
    # Terminal sessions never become a live tile.
    assert not [a for a in data["agents"] if a["agent_id"] == "claude-code-done"]


async def test_archived_total_matches_list_length_when_under_limit(_db, _client):
    now = utcnow()
    for i in range(5):
        _db.upsert_session(make_session(
            agent_id="claude-code-few", session_id=f"few-{i}", status="closed",
            input_tokens=100, output_tokens=20, tool_call_count=1,
            started_at=now - timedelta(hours=2), ended_at=now - timedelta(hours=1),
        ))
    data = (await _client.get("/api/v1/status")).json()
    assert data["archived_total"] == len(data["archived"]) == 5


async def test_archived_total_is_zero_for_empty_archive(_db, _client):
    data = (await _client.get("/api/v1/status")).json()
    assert data["archived"] == []
    assert data["archived_total"] == 0


async def test_archived_total_is_not_clamped_to_archive_limit(_db, _client):
    # ARCHIVE_LIMIT caps the returned list at 50, but archived_total must
    # report the TRUE count so the UI can show "latest 50 of N" honestly
    # instead of silently presenting the capped page as the whole archive.
    now = utcnow()
    for i in range(60):
        _db.upsert_session(make_session(
            agent_id="claude-code-many", session_id=f"many-{i}", status="closed",
            input_tokens=100, output_tokens=20, tool_call_count=1,
            started_at=now - timedelta(hours=2),
            ended_at=now - timedelta(minutes=i + 1),
        ))
    data = (await _client.get("/api/v1/status")).json()
    assert len(data["archived"]) == 50
    assert data["archived_total"] == 60


async def test_archived_total_excludes_zero_signal_zombies(_db, _client):
    # archived_total must describe the SAME population as the list beside it —
    # a zero-signal terminal the list drops must not inflate the total either,
    # or the two figures would disagree about what "archived" means.
    now = utcnow()
    _db.upsert_session(make_session(
        agent_id="claude-code-zombie", session_id="zt-1", status="closed",
        input_tokens=0, output_tokens=0, tool_call_count=0,
        started_at=now - timedelta(hours=2), ended_at=now - timedelta(hours=1),
    ))
    _db.upsert_session(make_session(
        agent_id="claude-code-real", session_id="zt-2", status="closed",
        input_tokens=500, output_tokens=100, tool_call_count=3,
        started_at=now - timedelta(hours=2), ended_at=now - timedelta(hours=1),
    ))
    data = (await _client.get("/api/v1/status")).json()
    assert len(data["archived"]) == 1
    assert data["archived_total"] == 1


async def test_status_zone_split_by_lifecycle_status(_db, _client):
    # Pin the whole mapping: 'active' + recent -> live tile; 'closed' and
    # 'completed' -> archive only.
    now = utcnow()
    for sid, aid, status in (
        ("live-1", "claude-code-live", "active"),
        ("closed-1", "claude-code-closed", "closed"),
        ("completed-1", "claude-code-completed", "completed"),
    ):
        _db.upsert_session(make_session(
            agent_id=aid, session_id=sid, status=status,
            input_tokens=500, output_tokens=100, tool_call_count=2,
            started_at=now - timedelta(minutes=10), ended_at=now,
        ))
    data = (await _client.get("/api/v1/status")).json()
    tile_agents = {a["agent_id"] for a in data["agents"]}
    archived_ids = {a["session_id"] for a in data["archived"]}

    assert tile_agents == {"claude-code-live"}
    assert archived_ids == {"closed-1", "completed-1"}


# ── The SDK persona's HISTORY surface ───────────────────────────────────────
# The live panel above is bounded by SDK_DISCOVERY_WINDOW. This is the other
# half: what a reader sees once an agent falls outside it. Without this pair,
# the Sessions page for the SDK persona rendered one sentence ("No SDK services
# live right now") and nothing else, on the only session surface that persona
# has — so a workload that last ran twelve days ago read as never having been
# recorded at all.

async def test_an_sdk_agent_past_the_discovery_window_is_still_reachable_as_history(
    _db, _client,
):
    """The exact reported case, end to end.

    An SDK agent last seen well beyond SDK_DISCOVERY_WINDOW is correctly ABSENT
    from `/status`'s live discovery AND correctly PRESENT in the persona-scoped
    session list the history surface reads. Both halves in one test, because
    either alone is a state that shipped: the first alone is the defect, and the
    second alone would mean the live panel had started including dormant
    services.
    """
    now = utcnow()
    stale_at = now - timedelta(days=12)
    for i, agent in enumerate((
        "sdk-workload-tool-heavy-chain",
        "sdk-workload-oversized-model",
        "demo-surprise-cost",
    )):
        _db.upsert_session(make_session(
            agent_id=agent, session_id=f"sdkhist-{i}", status="closed",
            input_tokens=5000, output_tokens=900, tool_call_count=7,
            started_at=stale_at, ended_at=stale_at + timedelta(minutes=4),
        ))
        _db.insert_span(make_llm_span(
            agent_id=agent, session_id=f"sdkhist-{i}",
            start_time=stale_at, cost_usd=0.25,
        ))

    status = (await _client.get("/api/v1/status")).json()
    live_ids = {s["agent_id"] for s in status.get("sdk_services") or []}
    assert not (live_ids & {
        "sdk-workload-tool-heavy-chain", "sdk-workload-oversized-model",
        "demo-surprise-cost",
    }), "dormant SDK agents must stay out of the LIVE panel"

    history = (await _client.get("/api/v1/sessions?persona=sdk&limit=50")).json()
    by_agent = {s["agent_id"] for s in history["sessions"]}
    assert {
        "sdk-workload-tool-heavy-chain", "sdk-workload-oversized-model",
        "demo-surprise-cost",
    } <= by_agent, (
        "an SDK agent outside the live-discovery window must remain reachable "
        "as history"
    )
    # The rows carry what the history table renders, so a column cannot silently
    # become a dash for every row.
    row = next(s for s in history["sessions"] if s["agent_id"] == "demo-surprise-cost")
    for field in ("session_id", "started_at", "total_cost_usd", "input_tokens",
                  "output_tokens", "tool_call_count", "error_count"):
        assert field in row, field
    assert row["started_at"] is not None


async def test_a_coding_session_never_leaks_into_the_sdk_history(_db, _client):
    """The history list is persona-scoped by the SAME agent_id-prefix rule the
    analyzer gate uses. A coding session appearing here would put thousands of
    rows under an SDK heading."""
    now = utcnow()
    _db.upsert_session(make_session(
        agent_id="claude-code-proj", session_id="cc-leak", status="closed",
        started_at=now - timedelta(days=12), ended_at=now - timedelta(days=12),
    ))
    _db.upsert_session(make_session(
        agent_id="billing-service", session_id="sdk-keep", status="closed",
        started_at=now - timedelta(days=12), ended_at=now - timedelta(days=12),
    ))
    history = (await _client.get("/api/v1/sessions?persona=sdk&limit=50")).json()
    ids = {s["session_id"] for s in history["sessions"]}
    assert "sdk-keep" in ids
    assert "cc-leak" not in ids
