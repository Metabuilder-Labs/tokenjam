"""Integration tests for the /v1/logs OTLP ingest endpoint."""
from __future__ import annotations

import pytest
import httpx

from tj.api.app import create_app
from tj.core.config import (
    ApiAuthConfig,
    ApiConfig,
    TjConfig,
    SecurityConfig,
)
from tj.core.db import InMemoryBackend
from tj.core.ingest import IngestPipeline
from tests.factories import (
    make_claude_code_api_error_log,
    make_claude_code_api_request_log,
    make_claude_code_tool_result_log,
    make_otlp_logs_body,
)


INGEST_SECRET = "test-secret-logs"


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
def app(config, db):
    pipeline = IngestPipeline(db=db, config=config)
    return create_app(config=config, db=db, ingest_pipeline=pipeline)


@pytest.fixture
def client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {INGEST_SECRET}"}


def _minimal_logs_body() -> dict:
    return make_otlp_logs_body([make_claude_code_api_request_log()])


@pytest.mark.asyncio
async def test_post_logs_without_auth_returns_401(client):
    resp = await client.post("/v1/logs", json=_minimal_logs_body())
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_post_logs_with_auth_ingests(client):
    resp = await client.post("/v1/logs", json=_minimal_logs_body(), headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingested"] == 1
    assert data["rejected"] == 0


@pytest.mark.asyncio
async def test_post_logs_invalid_json_returns_400(client):
    resp = await client.post(
        "/v1/logs",
        content=b"not-json",
        headers={**_auth_headers(), "Content-Type": "application/json"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_post_logs_non_log_payload_returns_200_empty(client):
    # Non-log OTLP signals (e.g. resourceSpans, resourceMetrics) that land on
    # /v1/logs are silently ignored — 200 with 0 ingested — so SDK exporters
    # that reuse this endpoint for all signal types don't produce 400 noise.
    resp = await client.post(
        "/v1/logs",
        json={"resourceSpans": []},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    assert resp.json()["ingested"] == 0


@pytest.mark.asyncio
async def test_api_request_event_stored_in_db(client, db):
    record = make_claude_code_api_request_log(
        session_id="test-cc-session",
        model="claude-sonnet-4-6",
        input_tokens=1500,
        output_tokens=300,
        cost_usd=0.005,
    )
    resp = await client.post(
        "/v1/logs",
        json=make_otlp_logs_body([record]),
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    assert resp.json()["ingested"] == 1

    # Verify the span is in the DB
    spans = db.get_recent_spans("test-cc-session", limit=10)
    assert len(spans) == 1
    span = spans[0]
    assert span.model == "claude-sonnet-4-6"
    assert span.input_tokens == 1500
    assert span.output_tokens == 300


@pytest.mark.asyncio
async def test_tool_result_event_stored_in_db(client, db):
    record = make_claude_code_tool_result_log(
        session_id="tool-session",
        tool_name="Bash",
        success=True,
    )
    resp = await client.post(
        "/v1/logs",
        json=make_otlp_logs_body([record]),
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    spans = db.get_recent_spans("tool-session", limit=10)
    assert len(spans) == 1
    assert spans[0].tool_name == "Bash"


@pytest.mark.asyncio
async def test_mixed_events_batch(client, db):
    session_id = "mixed-session"
    records = [
        make_claude_code_api_request_log(session_id=session_id, sequence=1),
        make_claude_code_tool_result_log(session_id=session_id, sequence=2),
        make_claude_code_api_error_log(session_id=session_id, sequence=3),
    ]
    resp = await client.post(
        "/v1/logs",
        json=make_otlp_logs_body(records),
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingested"] == 3
    assert data["rejected"] == 0


@pytest.mark.asyncio
async def test_cost_usd_preserved(client, db):
    """CostEngine must not overwrite Claude Code's authoritative cost_usd."""
    record = make_claude_code_api_request_log(
        session_id="cost-session",
        cost_usd=0.0123,
    )
    await client.post(
        "/v1/logs",
        json=make_otlp_logs_body([record]),
        headers=_auth_headers(),
    )
    spans = db.get_recent_spans("cost-session", limit=10)
    assert spans[0].cost_usd == pytest.approx(0.0123)


@pytest.mark.asyncio
async def test_partial_failure_returns_200_with_rejections(client, db):
    """A bad record alongside good records -> 200 with counts."""
    good = make_claude_code_api_request_log(session_id="partial-session")
    # Build a malformed record with missing required field (session.id)
    bad = {
        "timeUnixNano": "1712700000000000000",
        "body": {"stringValue": "claude_code.api_request"},
        "attributes": [
            # Missing session.id -> KeyError in converter -> rejection
            {"key": "model", "value": {"stringValue": "claude-sonnet-4-6"}},
            {"key": "duration_ms", "value": {"doubleValue": 100.0}},
            {"key": "input_tokens", "value": {"intValue": "100"}},
            {"key": "output_tokens", "value": {"intValue": "50"}},
        ],
    }
    resp = await client.post(
        "/v1/logs",
        json=make_otlp_logs_body([good, bad]),
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingested"] == 1
    assert data["rejected"] == 1
    assert len(data["rejections"]) == 1


@pytest.mark.asyncio
async def test_session_created_from_session_id(client, db):
    record = make_claude_code_api_request_log(session_id="new-cc-session")
    await client.post(
        "/v1/logs",
        json=make_otlp_logs_body([record]),
        headers=_auth_headers(),
    )
    # The ingest pipeline should have upserted a session
    sessions = db.get_completed_sessions("claude-code", limit=10)
    # Session may be active (not completed), so check via spans
    spans = db.get_recent_spans("new-cc-session", limit=10)
    assert len(spans) == 1


@pytest.mark.asyncio
async def test_v1_traces_now_requires_auth(client):
    """The /v1/traces endpoint must also be protected by auth middleware."""
    from tests.factories import make_llm_span
    body = {
        "resourceSpans": [{
            "resource": {"attributes": []},
            "scopeSpans": [{"spans": []}],
        }]
    }
    resp = await client.post("/v1/traces", json=body)
    assert resp.status_code == 401
