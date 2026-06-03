"""Integration tests for the REST API using httpx.AsyncClient + ASGITransport."""
from __future__ import annotations

import pytest
import httpx

from unittest.mock import patch

from tokenjam.api.app import create_app
from tokenjam.core.config import (
    AgentConfig,
    AlertsConfig,
    ApiAuthConfig,
    ApiConfig,
    BudgetConfig,
    TjConfig,
    SecurityConfig,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tests.factories import make_invoke_agent_span, make_llm_span, make_session, make_tool_span


INGEST_SECRET = "test-secret-token"


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config():
    return TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret=INGEST_SECRET),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
    )


@pytest.fixture
def config_with_api_auth():
    return TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret=INGEST_SECRET),
        api=ApiConfig(auth=ApiAuthConfig(enabled=True, api_key="my-api-key")),
    )


@pytest.fixture
def app(config, db):
    pipeline = IngestPipeline(db=db, config=config)
    return create_app(config=config, db=db, ingest_pipeline=pipeline)


@pytest.fixture
def app_with_auth(config_with_api_auth, db):
    pipeline = IngestPipeline(db=db, config=config_with_api_auth)
    return create_app(config=config_with_api_auth, db=db, ingest_pipeline=pipeline)


@pytest.fixture
def client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def auth_client(app_with_auth):
    transport = httpx.ASGITransport(app=app_with_auth)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _otlp_body(spans: list[dict] | None = None) -> dict:
    """Build a minimal OTLP JSON body."""
    if spans is None:
        spans = [_make_otlp_span()]
    return {
        "resourceSpans": [{
            "resource": {
                "attributes": [
                    {"key": "gen_ai.agent.id", "value": {"stringValue": "test-agent"}},
                    {"key": "gen_ai.provider.name", "value": {"stringValue": "anthropic"}},
                ],
            },
            "scopeSpans": [{"spans": spans}],
        }],
    }


def _make_otlp_span(
    span_id: str = "abc123def456",
    trace_id: str = "aabbccdd" * 4,
    name: str = "gen_ai.llm.call",
    status_code: int = 1,
    **extra_attrs: str,
) -> dict:
    """Build a single OTLP span dict."""
    attrs = [
        {"key": "gen_ai.request.model", "value": {"stringValue": "claude-haiku-4-5"}},
        {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "500"}},
        {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "100"}},
    ]
    for k, v in extra_attrs.items():
        attrs.append({"key": k, "value": {"stringValue": v}})
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": 3,  # CLIENT
        "startTimeUnixNano": "1711600000000000000",
        "endTimeUnixNano": "1711600001000000000",
        "status": {"code": status_code},
        "attributes": attrs,
    }


# ── Ingest auth ────────────────────────────────────────────────────────────

async def test_post_spans_without_auth_returns_401(client):
    resp = await client.post("/api/v1/spans", json=_otlp_body())
    assert resp.status_code == 401


async def test_post_spans_with_wrong_secret_returns_401(client):
    resp = await client.post(
        "/api/v1/spans",
        json=_otlp_body(),
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


async def test_post_spans_with_correct_auth_ingests_spans(client):
    resp = await client.post(
        "/api/v1/spans",
        json=_otlp_body(),
        headers={"Authorization": f"Bearer {INGEST_SECRET}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingested"] == 1
    assert data["rejected"] == 0


async def test_post_spans_invalid_json_returns_400(client):
    resp = await client.post(
        "/api/v1/spans",
        content=b"not json",
        headers={
            "Authorization": f"Bearer {INGEST_SECRET}",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 400


async def test_post_spans_missing_resource_spans_returns_400(client):
    resp = await client.post(
        "/api/v1/spans",
        json={"wrong_key": []},
        headers={"Authorization": f"Bearer {INGEST_SECRET}"},
    )
    assert resp.status_code == 400


async def test_post_spans_partial_rejection_returns_200(client, db, config):
    """Batch of 2 spans where 1 has oversized attributes — should partially succeed."""
    good_span = _make_otlp_span(span_id="good11111111")
    # Create a span with an attribute exceeding the default max_attribute_bytes (65536)
    big_value = "x" * 70000
    bad_span = _make_otlp_span(span_id="bad111111111")
    bad_span["attributes"].append(
        {"key": "huge_attr", "value": {"stringValue": big_value}}
    )
    body = _otlp_body(spans=[good_span, bad_span])
    resp = await client.post(
        "/api/v1/spans",
        json=body,
        headers={"Authorization": f"Bearer {INGEST_SECRET}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingested"] == 1
    assert data["rejected"] == 1
    assert len(data["rejections"]) == 1


# ── GET endpoints ──────────────────────────────────────────────────────────

async def _ingest_sample_span(client):
    """Helper: ingest one span so GET endpoints have data."""
    resp = await client.post(
        "/api/v1/spans",
        json=_otlp_body(),
        headers={"Authorization": f"Bearer {INGEST_SECRET}"},
    )
    assert resp.status_code == 200


async def test_get_traces_returns_list(client):
    await _ingest_sample_span(client)
    resp = await client.get("/api/v1/traces")
    assert resp.status_code == 200
    data = resp.json()
    assert "traces" in data
    assert len(data["traces"]) >= 1


async def test_get_traces_filter_by_agent_id(client):
    await _ingest_sample_span(client)
    resp = await client.get("/api/v1/traces", params={"agent_id": "test-agent"})
    assert resp.status_code == 200
    data = resp.json()
    for t in data["traces"]:
        assert t["agent_id"] == "test-agent"


async def test_get_trace_by_id_returns_span_waterfall(client):
    await _ingest_sample_span(client)
    trace_id = "aabbccdd" * 4
    resp = await client.get(f"/api/v1/traces/{trace_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["trace_id"] == trace_id
    assert "spans" in data
    assert len(data["spans"]) >= 1


async def test_get_cost_returns_aggregated_rows(client):
    await _ingest_sample_span(client)
    resp = await client.get("/api/v1/cost")
    assert resp.status_code == 200
    data = resp.json()
    assert "rows" in data
    assert "total_cost_usd" in data


async def test_get_alerts_returns_list(client):
    resp = await client.get("/api/v1/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)


async def test_get_tools_returns_list(client):
    resp = await client.get("/api/v1/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data


async def test_get_metrics_returns_prometheus_format(client):
    await _ingest_sample_span(client)
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "tj_cost_usd_total" in text
    assert "# HELP" in text
    assert "# TYPE" in text


async def test_get_drift_without_agent_id_returns_all(client):
    resp = await client.get("/api/v1/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data


# ── API key auth ───────────────────────────────────────────────────────────

async def test_get_endpoint_requires_api_key_when_auth_enabled(auth_client):
    resp = await auth_client.get("/api/v1/traces")
    assert resp.status_code in (401, 403)


async def test_get_endpoint_works_with_valid_api_key(auth_client):
    resp = await auth_client.get(
        "/api/v1/traces",
        headers={"Authorization": "Bearer my-api-key"},
    )
    assert resp.status_code == 200


# ── Docs endpoint ──────────────────────────────────────────────────────────

async def test_docs_endpoint_is_accessible(client):
    resp = await client.get("/docs")
    assert resp.status_code == 200


# ── agent_id normalization ─────────────────────────────────────────────────

async def test_span_without_agent_id_normalizes_to_unknown(client):
    body = {
        "resourceSpans": [{
            "resource": {"attributes": []},
            "scopeSpans": [{"spans": [_make_otlp_span()]}],
        }]
    }
    resp = await client.post(
        "/api/v1/spans", json=body,
        headers={"Authorization": f"Bearer {INGEST_SECRET}"},
    )
    assert resp.status_code == 200

    # Verify spans table (the actual fix) — traces must show "unknown", not None
    traces = await client.get("/api/v1/traces")
    trace_agents = [t["agent_id"] for t in traces.json()["traces"]]
    assert "unknown" in trace_agents

    # Verify sessions table agrees. The OTLP span carries an old timestamp, so
    # the session is stale and surfaces in the archive (not a live tile).
    status = await client.get("/api/v1/status")
    body_json = status.json()
    status_agents = (
        {a["agent_id"] for a in body_json["agents"]}
        | {s["agent_id"] for s in body_json["archived"]}
    )
    assert "unknown" in status_agents


async def test_status_and_traces_agree_on_agent_ids(client):
    # Post a span with NO agent_id — this is the scenario that diverged before the fix
    body = {
        "resourceSpans": [{
            "resource": {"attributes": []},
            "scopeSpans": [{"spans": [_make_otlp_span()]}],
        }]
    }
    resp = await client.post(
        "/api/v1/spans", json=body,
        headers={"Authorization": f"Bearer {INGEST_SECRET}"},
    )
    assert resp.status_code == 200

    status = await client.get("/api/v1/status")
    traces = await client.get("/api/v1/traces")

    sj = status.json()
    # An agent's sessions can be live (agents) or archived (closed/stale); the
    # full set of known agent ids is the union of both.
    status_agents = (
        {a["agent_id"] for a in sj["agents"]}
        | {s["agent_id"] for s in sj["archived"]}
    )
    trace_agents = {t["agent_id"] for t in traces.json()["traces"]}
    assert trace_agents == status_agents


# ── Budget ─────────────────────────────────────────────────────────────────

async def test_post_budget_zero_clears_limit(db):
    """Posting daily_usd=0 (empty field from UI) should set limit to None (no limit)."""
    cfg = TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret=INGEST_SECRET),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        agents={"my-agent": AgentConfig(budget=BudgetConfig(daily_usd=5.0))},
    )
    pipeline = IngestPipeline(db=db, config=cfg)
    app = create_app(config=cfg, db=db, ingest_pipeline=pipeline)
    transport = httpx.ASGITransport(app=app)

    with patch("tokenjam.api.routes.budget.find_config_file", return_value="/fake/tj.toml"), \
         patch("tokenjam.api.routes.budget.write_config"):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/v1/budget", json={"scope": "my-agent", "daily_usd": 0})

    assert resp.status_code == 200
    agent = resp.json()["agents"]["my-agent"]
    assert agent["configured"]["daily_usd"] is None  # limit was cleared


# ===========================================================================
# Status route: concurrent live sessions each get a tile, read as active
#
# Claude Code's logs path emits a zero-duration invoke_agent marker per user
# prompt and never an explicit end. Each live terminal is its own session;
# the status route surfaces one tile per recently-active session (not a single
# collapsed/"completed" tile). Uses DuckDBBackend because /status reads via
# db.conn.
# ===========================================================================

@pytest.mark.asyncio
async def test_status_shows_live_session_over_empty_marker(tmp_path):
    from tokenjam.core.db import DuckDBBackend
    from tokenjam.core.config import StorageConfig
    from tokenjam.core.models import AgentRecord
    from tokenjam.utils.time_parse import utcnow

    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    try:
        config = TjConfig(
            version="1",
            security=SecurityConfig(ingest_secret=INGEST_SECRET),
            api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        )
        pipeline = IngestPipeline(db=db, config=config)
        now = utcnow()
        db.upsert_agent(AgentRecord(agent_id="claude-code", first_seen=now, last_seen=now))

        # Live session: turn-start marker + real LLM activity.
        pipeline.process(make_invoke_agent_span(
            agent_id="claude-code", session_id="live", conversation_id="c1", duration_ms=0.0))
        pipeline.process(make_llm_span(
            agent_id="claude-code", session_id="live", conversation_id="c1",
            input_tokens=19562, output_tokens=1321, cost_usd=0.17))
        # A newer but EMPTY marker session (used to win the tile and show 0/0).
        pipeline.process(make_invoke_agent_span(
            agent_id="claude-code", session_id="empty", conversation_id="c2", duration_ms=0.0))

        app = create_app(config=config, db=db, ingest_pipeline=pipeline)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/status")

        assert resp.status_code == 200
        # Concurrent active sessions each get their own tile (one per terminal).
        cc_tiles = [a for a in resp.json()["agents"] if a["agent_id"] == "claude-code"]
        by_session = {a["session_id"]: a for a in cc_tiles}
        assert "live" in by_session and "empty" in by_session
        live = by_session["live"]
        assert live["status"] == "active"
        assert live["input_tokens"] == 19562
        assert live["output_tokens"] == 1321
        # The just-started marker session is its own tile with no work yet.
        assert by_session["empty"]["input_tokens"] == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_status_returns_service_namespace(tmp_path):
    """The status tile carries service.namespace so the dashboard groups by project."""
    from tokenjam.core.db import DuckDBBackend
    from tokenjam.core.config import StorageConfig
    from tokenjam.core.models import AgentRecord
    from tokenjam.utils.time_parse import utcnow

    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    try:
        config = TjConfig(
            version="1",
            security=SecurityConfig(ingest_secret=INGEST_SECRET),
            api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        )
        pipeline = IngestPipeline(db=db, config=config)
        now = utcnow()
        db.upsert_agent(AgentRecord(agent_id="claude-code-harness", first_seen=now, last_seen=now))
        pipeline.process(make_llm_span(
            agent_id="claude-code-harness", session_id="s1",
            service_namespace="aquanode"))

        app = create_app(config=config, db=db, ingest_pipeline=pipeline)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/status")

        assert resp.status_code == 200
        agents = {a["agent_id"]: a for a in resp.json()["agents"]}
        assert agents["claude-code-harness"]["namespace"] == "aquanode"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_status_namespace_falls_back_to_configured_project(tmp_path):
    """Archived sessions with no per-session namespace still group via
    [agents.<id>].project.

    Covers the already-running session case: no service.namespace ever arrived
    on the wire, but the server-side project mapping groups it anyway. A closed
    session is no longer a live tile, so the fallback must apply to the archive.
    """
    from tokenjam.core.db import DuckDBBackend
    from tokenjam.core.config import StorageConfig, AgentConfig
    from tokenjam.core.models import AgentRecord
    from tokenjam.utils.time_parse import utcnow

    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    try:
        config = TjConfig(
            version="1",
            security=SecurityConfig(ingest_secret=INGEST_SECRET),
            api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
            agents={"claude-code-harness": AgentConfig(project="aquanode")},
        )
        pipeline = IngestPipeline(db=db, config=config)
        now = utcnow()
        db.upsert_agent(AgentRecord(agent_id="claude-code-harness", first_seen=now, last_seen=now))
        # Pre-existing session with NO namespace (collected before the mapping).
        db.upsert_session(make_session(
            agent_id="claude-code-harness", session_id="s1", status="closed"))

        app = create_app(config=config, db=db, ingest_pipeline=pipeline)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/status")

        assert resp.status_code == 200
        # Closed session -> archive, not a live tile.
        assert resp.json()["agents"] == []
        archived = {s["session_id"]: s for s in resp.json()["archived"]}
        assert archived["s1"]["namespace"] == "aquanode"
        assert archived["s1"]["status"] == "closed"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_status_one_tile_per_concurrent_session(tmp_path):
    """Three concurrent terminals under one agent each get a tile, grouped by project."""
    from tokenjam.core.db import DuckDBBackend
    from tokenjam.core.config import StorageConfig, AgentConfig
    from tokenjam.core.models import AgentRecord
    from tokenjam.utils.time_parse import utcnow

    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    try:
        config = TjConfig(
            version="1",
            security=SecurityConfig(ingest_secret=INGEST_SECRET),
            api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
            agents={"claude-code-harness": AgentConfig(project="aquanode")},
        )
        pipeline = IngestPipeline(db=db, config=config)
        now = utcnow()
        db.upsert_agent(AgentRecord(agent_id="claude-code-harness", first_seen=now, last_seen=now))
        # Three live terminals = three session ids under one agent.
        for sid, intok in [("term-a", 100), ("term-b", 200), ("term-c", 300)]:
            pipeline.process(make_llm_span(
                agent_id="claude-code-harness", session_id=sid, conversation_id=sid,
                input_tokens=intok, output_tokens=10))

        app = create_app(config=config, db=db, ingest_pipeline=pipeline)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/status")

        assert resp.status_code == 200
        tiles = [a for a in resp.json()["agents"] if a["agent_id"] == "claude-code-harness"]
        assert len(tiles) == 3
        assert {t["session_id"] for t in tiles} == {"term-a", "term-b", "term-c"}
        assert all(t["namespace"] == "aquanode" for t in tiles)
        assert all(t["status"] == "active" for t in tiles)
        assert {t["session_id"]: t["input_tokens"] for t in tiles} == {
            "term-a": 100, "term-b": 200, "term-c": 300,
        }
    finally:
        db.close()


@pytest.mark.asyncio
async def test_status_session_labels(tmp_path):
    """Session label resolves: manual override > service.instance.id > short id."""
    from tokenjam.core.db import DuckDBBackend
    from tokenjam.core.config import StorageConfig, AgentConfig
    from tokenjam.core.models import AgentRecord
    from tokenjam.utils.time_parse import utcnow

    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    try:
        config = TjConfig(
            version="1",
            security=SecurityConfig(ingest_secret=INGEST_SECRET),
            api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
            agents={"claude-code-harness": AgentConfig(project="aquanode")},
            # prefix override for a current terminal; "ovr" overrides instance.id
            session_labels={"manual": "harness", "ovr": "config-wins"},
        )
        pipeline = IngestPipeline(db=db, config=config)
        now = utcnow()
        db.upsert_agent(AgentRecord(agent_id="claude-code-harness", first_seen=now, last_seen=now))
        # instance.id on the wire -> durable label
        pipeline.process(make_llm_span(
            agent_id="claude-code-harness", session_id="wired", conversation_id="w",
            service_instance_id="founder-os"))
        # manual config prefix label only
        pipeline.process(make_llm_span(
            agent_id="claude-code-harness", session_id="manual-123", conversation_id="m"))
        # both present -> manual override wins
        pipeline.process(make_llm_span(
            agent_id="claude-code-harness", session_id="ovr-9", conversation_id="o",
            service_instance_id="wire-name"))
        # neither -> label is None (UI falls back to short id)
        pipeline.process(make_llm_span(
            agent_id="claude-code-harness", session_id="plain", conversation_id="p"))

        app = create_app(config=config, db=db, ingest_pipeline=pipeline)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/status")

        assert resp.status_code == 200
        by = {a["session_id"]: a for a in resp.json()["agents"]
              if a["agent_id"] == "claude-code-harness"}
        assert by["wired"]["label"] == "founder-os"      # from instance.id
        assert by["manual-123"]["label"] == "harness"    # from config prefix
        assert by["ovr-9"]["label"] == "config-wins"     # config beats instance.id
        assert by["plain"]["label"] is None              # no label
    finally:
        db.close()


# ===========================================================================
# Session lifecycle: active/idle tiers as tiles, closed/stale archived, cap.
# ===========================================================================

def _lifecycle_app(tmp_path, agents=None):
    """A DuckDB-backed app for session-lifecycle tests (status reads db.conn)."""
    from tokenjam.core.db import DuckDBBackend
    from tokenjam.core.config import StorageConfig

    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    config = TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret=INGEST_SECRET),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        agents=agents or {},
    )
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    return db, app


@pytest.mark.asyncio
async def test_status_active_and_idle_become_tiles_stale_archived(tmp_path):
    from datetime import timedelta
    from tokenjam.utils.time_parse import utcnow

    db, app = _lifecycle_app(tmp_path)
    try:
        now = utcnow()
        db.upsert_session(make_session(
            agent_id="cc", session_id="act", status="active",
            started_at=now - timedelta(minutes=2), ended_at=now - timedelta(minutes=1)))
        db.upsert_session(make_session(
            agent_id="cc", session_id="idl", status="active",
            started_at=now - timedelta(minutes=40), ended_at=now - timedelta(minutes=30)))
        db.upsert_session(make_session(
            agent_id="cc", session_id="stl", status="active",
            started_at=now - timedelta(hours=6), ended_at=now - timedelta(hours=5)))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/status")

        assert resp.status_code == 200
        sj = resp.json()
        tiles = {a["session_id"]: a for a in sj["agents"]}
        assert set(tiles) == {"act", "idl"}            # only active + idle
        assert tiles["act"]["status"] == "active"
        assert tiles["idl"]["status"] == "idle"
        archived = {s["session_id"]: s for s in sj["archived"]}
        assert "stl" in archived and archived["stl"]["status"] == "stale"
        assert "act" not in archived and "idl" not in archived
    finally:
        db.close()


@pytest.mark.asyncio
async def test_status_no_fallback_tile_for_completed_or_closed(tmp_path):
    db, app = _lifecycle_app(tmp_path)
    try:
        db.upsert_session(make_session(
            agent_id="cc", session_id="done", status="completed"))
        db.upsert_session(make_session(
            agent_id="cc", session_id="shut", status="closed"))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/status")

        sj = resp.json()
        # No active/idle session for this agent -> no current tile at all.
        assert sj["agents"] == []
        archived = {s["session_id"]: s["status"] for s in sj["archived"]}
        # Closed lands in the archive; completed is neither current nor archived.
        assert archived.get("shut") == "closed"
        assert "done" not in archived
    finally:
        db.close()


@pytest.mark.asyncio
async def test_status_caps_tiles_per_agent_and_reports_overflow(tmp_path):
    from datetime import timedelta
    from tokenjam.utils.time_parse import utcnow

    db, app = _lifecycle_app(tmp_path)
    try:
        now = utcnow()
        for i in range(9):
            db.upsert_session(make_session(
                agent_id="cc", session_id=f"s{i}", status="active",
                started_at=now - timedelta(seconds=30 + i),
                ended_at=now - timedelta(seconds=i)))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/status")

        tiles = [a for a in resp.json()["agents"] if a["agent_id"] == "cc"]
        assert len(tiles) == 6                 # MAX_SESSION_TILES
        assert all(t["overflow"] == 3 for t in tiles)  # 9 - 6 surfaced, not dropped
    finally:
        db.close()


@pytest.mark.asyncio
async def test_close_sessions_by_instance_marks_closed_idempotent(tmp_path):
    db, app = _lifecycle_app(tmp_path)
    try:
        from tokenjam.utils.time_parse import utcnow
        now = utcnow()
        for sid in ("s1", "s2"):
            db.upsert_session(make_session(
                agent_id="cc", session_id=sid, status="active",
                service_instance_id="term-x", ended_at=now))
        db.upsert_session(make_session(
            agent_id="cc", session_id="s3", status="active",
            service_instance_id="other", ended_at=now))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/sessions/close", json={"instance_id": "term-x"},
                headers={"Authorization": f"Bearer {INGEST_SECRET}"})
            assert resp.status_code == 200
            assert resp.json()["closed"] == 2
            # Re-closing is a no-op (idempotent).
            resp2 = await c.post(
                "/api/v1/sessions/close", json={"instance_id": "term-x"},
                headers={"Authorization": f"Bearer {INGEST_SECRET}"})
            assert resp2.json()["closed"] == 0

        assert db.get_session("s1").status == "closed"
        assert db.get_session("s2").status == "closed"
        assert db.get_session("s3").status == "active"   # different instance
    finally:
        db.close()


@pytest.mark.asyncio
async def test_close_session_by_id(tmp_path):
    db, app = _lifecycle_app(tmp_path)
    try:
        db.upsert_session(make_session(
            agent_id="cc", session_id="s1", status="active"))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/sessions/close", json={"session_id": "s1"},
                headers={"Authorization": f"Bearer {INGEST_SECRET}"})
        assert resp.json()["closed"] == 1
        assert db.get_session("s1").status == "closed"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_close_sessions_requires_ingest_secret(tmp_path):
    db, app = _lifecycle_app(tmp_path)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/sessions/close", json={"instance_id": "term-x"})
        assert resp.status_code == 401
    finally:
        db.close()


@pytest.mark.asyncio
async def test_close_sessions_requires_an_id(tmp_path):
    db, app = _lifecycle_app(tmp_path)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/sessions/close", json={},
                headers={"Authorization": f"Bearer {INGEST_SECRET}"})
        assert resp.status_code == 400
    finally:
        db.close()


# ===========================================================================
# GET /api/v1/sessions/{session_id} — Session Detail view (Layer 1).
# ===========================================================================

@pytest.mark.asyncio
async def test_session_detail_returns_rollup_tools_and_traces(tmp_path):
    """A known session returns its rollup + per-tool breakdown + its traces."""
    from tokenjam.core.models import Alert, AlertType, Severity
    from tokenjam.utils.time_parse import utcnow

    db, app = _lifecycle_app(tmp_path)
    try:
        sid = "sess-1"
        trace_a = "trace-a"
        db.upsert_session(make_session(
            agent_id="cc", session_id=sid, conversation_id="conv-1",
            input_tokens=5000, output_tokens=800, tool_call_count=3,
            error_count=1, total_cost_usd=0.42, status="active",
            plan_tier="api"))
        # Two LLM spans (one with a tool_name, one a failing tool) on one trace.
        db.insert_span(make_llm_span(
            agent_id="cc", session_id=sid, conversation_id="conv-1",
            trace_id=trace_a, input_tokens=2500, output_tokens=400,
            cost_usd=0.21))
        db.insert_span(make_llm_span(
            agent_id="cc", session_id=sid, conversation_id="conv-1",
            trace_id=trace_a, tool_name="Read", input_tokens=0,
            output_tokens=0, cost_usd=0.0))
        db.insert_span(make_llm_span(
            agent_id="cc", session_id=sid, conversation_id="conv-2",
            trace_id="trace-b", tool_name="Bash", status="error",
            input_tokens=2500, output_tokens=400, cost_usd=0.21))
        # An active alert attributed to this session, plus one to another.
        now = utcnow()
        db.insert_alert(Alert(
            alert_id="al-1", fired_at=now, type=AlertType.RETRY_LOOP,
            severity=Severity.WARNING, title="Retry loop detected",
            detail={}, agent_id="cc", session_id=sid))
        db.insert_alert(Alert(
            alert_id="al-2", fired_at=now, type=AlertType.FAILURE_RATE,
            severity=Severity.INFO, title="Other session", detail={},
            agent_id="cc", session_id="someone-else"))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/api/v1/sessions/{sid}")

        assert resp.status_code == 200
        body = resp.json()
        sess = body["session"]
        assert sess["session_id"] == sid
        assert sess["agent_id"] == "cc"
        assert sess["plan_tier"] == "api"
        assert sess["pricing_mode"] == "api"
        assert sess["input_tokens"] == 5000
        assert sess["output_tokens"] == 800
        assert sess["total_cost_usd"] == 0.42
        # Two distinct conversation_ids across the session's spans.
        assert sess["conversation_count"] == 2
        assert sess["active_alerts"] == 1

        # Tool breakdown: Read (ok) + Bash (1 failure).
        tools = {t["tool_name"]: t for t in body["tools"]}
        assert tools["Read"]["count"] == 1 and tools["Read"]["error_count"] == 0
        assert tools["Bash"]["count"] == 1 and tools["Bash"]["error_count"] == 1

        # Only this session's alert appears.
        assert [a["title"] for a in body["alerts"]] == ["Retry loop detected"]

        # Traces: both the session's traces, newest first; status rolls up.
        trace_ids = {t["trace_id"] for t in body["traces"]}
        assert trace_ids == {"trace-a", "trace-b"}
        by_trace = {t["trace_id"]: t for t in body["traces"]}
        assert by_trace["trace-a"]["status_code"] == "ok"
        assert by_trace["trace-b"]["status_code"] == "error"
        assert by_trace["trace-a"]["span_count"] == 2

        # No drift baseline recorded for this agent.
        assert body["drift"] is None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_session_detail_unknown_returns_404(tmp_path):
    db, app = _lifecycle_app(tmp_path)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/sessions/does-not-exist")
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_session_detail_subscription_plan_tier(tmp_path):
    """A subscription session reports pricing_mode='subscription' so the UI
    renders the implied-API-value framing (not a dollar 'spend' claim)."""
    db, app = _lifecycle_app(tmp_path)
    try:
        sid = "sub-sess"
        db.upsert_session(make_session(
            agent_id="cc", session_id=sid, plan_tier="max_20x",
            input_tokens=1000, output_tokens=200, total_cost_usd=1.50,
            status="active"))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/api/v1/sessions/{sid}")

        assert resp.status_code == 200
        sess = resp.json()["session"]
        assert sess["plan_tier"] == "max_20x"
        assert sess["pricing_mode"] == "subscription"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_session_detail_includes_drift_baseline(tmp_path):
    """When the agent has a drift baseline, the detail view surfaces its
    summary; otherwise it is null."""
    from tokenjam.core.models import DriftBaseline
    from tokenjam.utils.time_parse import utcnow

    db, app = _lifecycle_app(tmp_path)
    try:
        sid = "drift-sess"
        db.upsert_session(make_session(
            agent_id="cc", session_id=sid, status="active"))
        db.upsert_baseline(DriftBaseline(
            agent_id="cc", sessions_sampled=12, computed_at=utcnow(),
            avg_input_tokens=4200.0, avg_output_tokens=600.0,
            avg_tool_call_count=3.5, avg_session_duration_s=120.0))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/api/v1/sessions/{sid}")

        assert resp.status_code == 200
        drift = resp.json()["drift"]
        assert drift is not None
        assert drift["sessions_sampled"] == 12
        assert drift["avg_input_tokens"] == 4200.0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_session_detail_requires_api_key_when_enabled(tmp_path):
    """The GET is guarded by require_api_key like other GET endpoints."""
    from tokenjam.core.db import DuckDBBackend
    from tokenjam.core.config import StorageConfig

    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "auth.duckdb")))
    try:
        config = TjConfig(
            version="1",
            security=SecurityConfig(ingest_secret=INGEST_SECRET),
            api=ApiConfig(auth=ApiAuthConfig(enabled=True, api_key="my-api-key")),
        )
        pipeline = IngestPipeline(db=db, config=config)
        app = create_app(config=config, db=db, ingest_pipeline=pipeline)
        db.upsert_session(make_session(
            agent_id="cc", session_id="s1", status="active"))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            unauth = await c.get("/api/v1/sessions/s1")
            assert unauth.status_code == 401
            ok = await c.get(
                "/api/v1/sessions/s1",
                headers={"Authorization": "Bearer my-api-key"})
            assert ok.status_code == 200
    finally:
        db.close()


@pytest.mark.asyncio
async def test_session_detail_model_mix_and_turn_count(tmp_path):
    """Multi-model session: model_mix aggregates per model (calls + summed
    tokens + cost), ordered by calls desc; turn_count == llm.call span count."""
    from datetime import timedelta
    from tokenjam.utils.time_parse import utcnow

    db, app = _lifecycle_app(tmp_path)
    try:
        sid = "mm-sess"
        db.upsert_session(make_session(
            agent_id="cc", session_id=sid, status="active", plan_tier="api"))
        base = utcnow()
        # opus x3, sonnet x2, haiku x1 -> ordered opus, sonnet, haiku.
        specs = (
            [("claude-opus-4-8", 1000, 100, 50, 0.30)] * 3
            + [("claude-sonnet-4-6", 500, 80, 10, 0.05)] * 2
            + [("claude-haiku-4-5", 200, 40, 0, 0.01)] * 1
        )
        for i, (model, inp, out, cache, cost) in enumerate(specs):
            db.insert_span(make_llm_span(
                agent_id="cc", session_id=sid, model=model,
                input_tokens=inp, output_tokens=out, cache_tokens=cache,
                cost_usd=cost, start_time=base + timedelta(seconds=i)))
        # A tool span must not be counted as a turn or appear in model_mix.
        db.insert_span(make_tool_span(agent_id="cc", tool_name="Read"))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/api/v1/sessions/{sid}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["turn_count"] == 6  # llm.call spans only

        mix = body["model_mix"]
        assert [m["model"] for m in mix] == [
            "claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5",
        ]
        by_model = {m["model"]: m for m in mix}
        assert by_model["claude-opus-4-8"]["calls"] == 3
        assert by_model["claude-opus-4-8"]["input_tokens"] == 3000
        assert by_model["claude-opus-4-8"]["output_tokens"] == 300
        assert by_model["claude-opus-4-8"]["cache_tokens"] == 150
        assert by_model["claude-opus-4-8"]["cost_usd"] == pytest.approx(0.90)
        assert by_model["claude-sonnet-4-6"]["calls"] == 2
        assert by_model["claude-sonnet-4-6"]["input_tokens"] == 1000
        assert by_model["claude-haiku-4-5"]["calls"] == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_session_context_series_ordered_and_complete(tmp_path):
    """A small session emits one context point per llm.call, time-ordered,
    each carrying that turn's real input_tokens."""
    from datetime import timedelta
    from tokenjam.utils.time_parse import utcnow

    db, app = _lifecycle_app(tmp_path)
    try:
        sid = "ctx-sess"
        db.upsert_session(make_session(
            agent_id="cc", session_id=sid, status="active"))
        base = utcnow()
        inputs = [100, 500, 1200, 800, 300]
        # Insert out of chronological order to prove SQL ORDER BY start_time.
        for i in [2, 0, 4, 1, 3]:
            db.insert_span(make_llm_span(
                agent_id="cc", session_id=sid, model="claude-haiku-4-5",
                input_tokens=inputs[i], output_tokens=10,
                start_time=base + timedelta(seconds=i)))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/api/v1/sessions/{sid}")

        assert resp.status_code == 200
        series = resp.json()["context_series"]
        assert len(series) == 5
        # Time-ordered ascending.
        ts = [p["t"] for p in series]
        assert ts == sorted(ts)
        # Each point keeps its turn's real input_tokens, in chronological order.
        assert [p["input_tokens"] for p in series] == inputs
    finally:
        db.close()


@pytest.mark.asyncio
async def test_session_context_series_downsampled_preserves_first_last(tmp_path):
    """A session with > MAX_CONTEXT_POINTS llm.calls is downsampled to <= 120
    points, keeping the first and last turns."""
    from datetime import timedelta
    from tokenjam.utils.time_parse import utcnow
    from tokenjam.api.routes.sessions import MAX_CONTEXT_POINTS

    db, app = _lifecycle_app(tmp_path)
    try:
        sid = "big-sess"
        db.upsert_session(make_session(
            agent_id="cc", session_id=sid, status="active"))
        base = utcnow()
        n = 300
        first_input = 11
        last_input = 9999
        for i in range(n):
            if i == 0:
                inp = first_input
            elif i == n - 1:
                inp = last_input
            else:
                inp = 100 + i
            db.insert_span(make_llm_span(
                agent_id="cc", session_id=sid, model="claude-haiku-4-5",
                input_tokens=inp, output_tokens=10,
                start_time=base + timedelta(seconds=i)))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get(f"/api/v1/sessions/{sid}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["turn_count"] == n  # turn_count is the true count, undownsampled
        series = body["context_series"]
        assert 0 < len(series) <= MAX_CONTEXT_POINTS
        # First and last turns are preserved.
        assert series[0]["input_tokens"] == first_input
        assert series[-1]["input_tokens"] == last_input
        # Still time-ordered.
        ts = [p["t"] for p in series]
        assert ts == sorted(ts)
    finally:
        db.close()


# ===========================================================================
# Cross-session run grouping (Layer 3): tokenjam.run_id resource attribute,
# GET /api/v1/sessions/{id} exposing the fields, and GET /api/v1/runs/{run_id}.
# ===========================================================================

@pytest.mark.asyncio
async def test_ingest_logs_captures_run_id_from_resource_attrs(tmp_path):
    """A Claude Code OTLP-logs batch with tokenjam.run_id in the resource attrs
    produces a session carrying that run_id (and parent_session_id)."""
    from tests.factories import (
        make_claude_code_api_request_log,
        make_otlp_logs_body,
    )

    db, app = _lifecycle_app(tmp_path)
    try:
        body = make_otlp_logs_body(
            [make_claude_code_api_request_log(session_id="cc-run-sess")],
            resource_attributes={
                "tokenjam.run_id": "run-42",
                "tokenjam.parent_session_id": "parent-sess",
            },
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/v1/logs", json=body,
                headers={"Authorization": f"Bearer {INGEST_SECRET}"})
        assert resp.status_code == 200

        sess = db.get_session("cc-run-sess")
        assert sess is not None
        assert sess.run_id == "run-42"
        assert sess.parent_session_id == "parent-sess"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_ingest_spans_captures_run_id_from_resource_attrs(tmp_path):
    """The live spans path extracts tokenjam.run_id from resource attrs too
    (shared otlp_parsing.py path)."""
    db, app = _lifecycle_app(tmp_path)
    try:
        body = {
            "resourceSpans": [{
                "resource": {"attributes": [
                    {"key": "service.name", "value": {"stringValue": "worker-a"}},
                    {"key": "session.id", "value": {"stringValue": "span-run-sess"}},
                    {"key": "tokenjam.run_id", "value": {"stringValue": "run-7"}},
                ]},
                "scopeSpans": [{"spans": [_make_otlp_span()]}],
            }]
        }
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post(
                "/api/v1/spans", json=body,
                headers={"Authorization": f"Bearer {INGEST_SECRET}"})
        assert resp.status_code == 200

        sess = db.get_session("span-run-sess")
        assert sess is not None and sess.run_id == "run-7"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_session_detail_exposes_run_fields(tmp_path):
    """GET /api/v1/sessions/{id} surfaces run_id + parent_session_id."""
    db, app = _lifecycle_app(tmp_path)
    try:
        db.upsert_session(make_session(
            agent_id="cc", session_id="s-run", status="active",
            run_id="run-9", parent_session_id="root-sess"))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/sessions/s-run")
        assert resp.status_code == 200
        sess = resp.json()["session"]
        assert sess["run_id"] == "run-9"
        assert sess["parent_session_id"] == "root-sess"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_run_detail_groups_sessions_aggregates_and_tree(tmp_path):
    """Three sessions sharing a run_id (one a child of another) group into one
    run with aggregated totals and a parent-edge tree."""
    db, app = _lifecycle_app(tmp_path)
    try:
        # Root + one child + one leaf, all in run-100.
        db.upsert_session(make_session(
            agent_id="cc", session_id="root", run_id="run-100",
            input_tokens=1000, output_tokens=200, total_cost_usd=0.10,
            tool_call_count=2, status="active", plan_tier="api"))
        db.upsert_session(make_session(
            agent_id="cc", session_id="child", run_id="run-100",
            parent_session_id="root", input_tokens=500, output_tokens=100,
            total_cost_usd=0.05, tool_call_count=1, status="active",
            plan_tier="api"))
        db.upsert_session(make_session(
            agent_id="cc", session_id="leaf", run_id="run-100",
            input_tokens=300, output_tokens=50, total_cost_usd=0.02,
            status="active", plan_tier="api"))
        # A session in a different run must NOT leak in.
        db.upsert_session(make_session(
            agent_id="cc", session_id="other", run_id="run-999",
            total_cost_usd=9.0, status="active"))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/runs/run-100")

        assert resp.status_code == 200
        body = resp.json()
        run = body["run"]
        assert run["run_id"] == "run-100"
        assert run["session_count"] == 3
        # Totals aggregate over the 3 members only (not the other run).
        assert round(run["total_cost_usd"], 2) == 0.17
        assert run["input_tokens"] == 1800
        assert run["output_tokens"] == 350
        assert run["tool_call_count"] == 3
        assert run["pricing_mode"] == "api"

        member_ids = {s["session_id"] for s in body["sessions"]}
        assert member_ids == {"root", "child", "leaf"}

        # Tree: root and leaf are roots; child nests under root.
        tree = body["tree"]
        root_ids = {n["session_id"] for n in tree}
        assert root_ids == {"root", "leaf"}
        root_node = next(n for n in tree if n["session_id"] == "root")
        assert [c["session_id"] for c in root_node["children"]] == ["child"]
        leaf_node = next(n for n in tree if n["session_id"] == "leaf")
        assert leaf_node["children"] == []
    finally:
        db.close()


@pytest.mark.asyncio
async def test_run_detail_unknown_returns_404(tmp_path):
    db, app = _lifecycle_app(tmp_path)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/runs/nope")
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"]
    finally:
        db.close()


@pytest.mark.asyncio
async def test_run_detail_mixed_pricing_mode(tmp_path):
    """A run whose members span api + subscription reports pricing_mode='mixed'
    so the dashboard avoids a single dollar 'spend' claim."""
    db, app = _lifecycle_app(tmp_path)
    try:
        db.upsert_session(make_session(
            agent_id="cc", session_id="api-s", run_id="run-mix",
            plan_tier="api", status="active"))
        db.upsert_session(make_session(
            agent_id="cc", session_id="sub-s", run_id="run-mix",
            plan_tier="max_20x", status="active"))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/runs/run-mix")
        assert resp.status_code == 200
        assert resp.json()["run"]["pricing_mode"] == "mixed"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_runs_index_lists_runs_newest_first(tmp_path):
    """GET /api/v1/runs lists each run once with totals; sessions with no
    run_id are excluded."""
    from datetime import timedelta
    from tokenjam.utils.time_parse import utcnow

    db, app = _lifecycle_app(tmp_path)
    try:
        now = utcnow()
        db.upsert_session(make_session(
            agent_id="cc", session_id="old", run_id="run-old",
            status="active", ended_at=now - timedelta(hours=2)))
        db.upsert_session(make_session(
            agent_id="cc", session_id="new", run_id="run-new",
            status="active", ended_at=now))
        db.upsert_session(make_session(
            agent_id="cc", session_id="norun", status="active"))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/api/v1/runs")
        assert resp.status_code == 200
        runs = resp.json()["runs"]
        run_ids = [r["run_id"] for r in runs]
        # Only runs with a run_id; newest activity first.
        assert run_ids == ["run-new", "run-old"]
        assert all(r["session_count"] == 1 for r in runs)
    finally:
        db.close()


# --- Session Story endpoint --------------------------------------------------

def _write_story_transcript(projects_root, session_id: str) -> None:
    """Write a minimal Claude Code JSONL transcript for the Story endpoint."""
    import json as _json

    project_dir = projects_root / "-Users-test-project"
    project_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {"type": "user", "message": {"role": "user", "content": "Build the thing."}},
        {
            "type": "assistant",
            "timestamp": "2026-06-15T09:11:36.133Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [
                    {"type": "text", "text": "Reading the file."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Read",
                        "input": {"file_path": "src/app.py"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "..."}
                ],
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-06-15T09:12:00.000Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "Done — it works."}],
            },
        },
    ]
    (project_dir / f"{session_id}.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in records), encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_get_session_story_available(config, db, tmp_path):
    _write_story_transcript(tmp_path, "story-sess")
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    app.state.claude_projects_root = tmp_path

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/v1/sessions/story-sess/story")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["task"] == "Build the thing."
    assert body["outcome"] == "Done — it works."
    assert body["step_count"] == 2
    assert body["steps"][0]["tools"][0]["name"] == "Read"
    assert body["steps"][0]["tools"][0]["label"] == "src/app.py"
    assert body["steps"][0]["tools"][0]["status"] == "ok"


@pytest.mark.asyncio
async def test_get_session_story_unavailable(config, db, tmp_path):
    # No transcript written -> available:false with HTTP 200.
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    app.state.claude_projects_root = tmp_path

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/v1/sessions/unknown-sess/story")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert "reason" in body


def _write_story_with_subagent(projects_root, session_id: str) -> None:
    """Parent transcript that spawns a subagent via a Task tool.

    The child agentId lives in the Task tool_result; the child transcript lives
    flat under ``<session_id>/subagents/agent-<id>.jsonl``.
    """
    import json as _json

    child_id = "abcabcabcabcabc12"
    project_dir = projects_root / "-Users-test-project"
    project_dir.mkdir(parents=True, exist_ok=True)
    parent = [
        {"type": "user", "message": {"role": "user", "content": "Orchestrate."}},
        {
            "type": "assistant",
            "timestamp": "2026-06-15T09:11:36.133Z",
            "message": {"role": "assistant", "model": "claude-opus-4-8", "content": [
                {"type": "text", "text": "Spawning a worker."},
                {"type": "tool_use", "id": "tt1", "name": "Task",
                 "input": {"description": "build-it",
                           "subagent_type": "general-purpose"}},
            ]},
        },
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tt1",
             "content": f"Agent (agentId: {child_id}) finished."}]}},
        {"type": "assistant", "timestamp": "2026-06-15T09:12:00.000Z",
         "message": {"role": "assistant", "model": "claude-opus-4-8",
                     "content": [{"type": "text", "text": "All done."}]}},
    ]
    (project_dir / f"{session_id}.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in parent), encoding="utf-8"
    )
    subdir = project_dir / session_id / "subagents"
    subdir.mkdir(parents=True, exist_ok=True)
    child = [
        {"type": "user", "message": {"role": "user", "content": "Build it."}},
        {"type": "assistant", "timestamp": "2026-06-15T09:11:40.000Z",
         "message": {"role": "assistant", "model": "claude-opus-4-8",
                     "content": [{"type": "text", "text": "Worker done."}]}},
    ]
    (subdir / f"agent-{child_id}.jsonl").write_text(
        "\n".join(_json.dumps(r) for r in child), encoding="utf-8"
    )
    (subdir / f"agent-{child_id}.meta.json").write_text(
        _json.dumps({"agentType": "general-purpose", "name": "build-it",
                     "toolUseId": "tt1"}), encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_get_session_story_nested_subagent(config, db, tmp_path):
    _write_story_with_subagent(tmp_path, "story-parent")
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    app.state.claude_projects_root = tmp_path

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/v1/sessions/story-parent/story")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    task_step = body["steps"][0]
    assert task_step["tools"][0]["name"] == "Task"
    sub = task_step["subagent"]
    assert sub["name"] == "build-it"
    assert sub["task"] == "Build it."
    assert sub["outcome"] == "Worker done."
    assert len(sub["steps"]) == 1


@pytest.mark.asyncio
async def test_get_session_story_subagents_false_is_flat(config, db, tmp_path):
    _write_story_with_subagent(tmp_path, "story-flat")
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)
    app.state.claude_projects_root = tmp_path

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/v1/sessions/story-flat/story?subagents=false")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert "subagent" not in body["steps"][0]
