"""Tests for OpenClaw OTLP ingestion — route aliases and attribute mapping."""
from __future__ import annotations

import pytest
import httpx

from tokenjam.api.app import create_app
from tokenjam.core.config import ApiAuthConfig, ApiConfig, TjConfig, SecurityConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline


INGEST_SECRET = "test-secret"


def _get_spans(db):
    """Query all spans from the in-memory DB."""
    rows = db.conn.execute("SELECT * FROM spans ORDER BY start_time").fetchall()
    cols = [d[0] for d in db.conn.description]
    from tokenjam.core.db import _row_to_span
    return [_row_to_span(r, cols) for r in rows]


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _otlp_body(
    spans: list[dict],
    resource_attrs: list[dict] | None = None,
) -> dict:
    if resource_attrs is None:
        resource_attrs = []
    return {
        "resourceSpans": [{
            "resource": {"attributes": resource_attrs},
            "scopeSpans": [{"spans": spans}],
        }],
    }


def _make_span(
    name: str,
    span_id: str = "abc123",
    trace_id: str = "trace001",
    attrs: list[dict] | None = None,
) -> dict:
    return {
        "spanId": span_id,
        "traceId": trace_id,
        "name": name,
        "kind": 1,
        "startTimeUnixNano": "1700000000000000000",
        "endTimeUnixNano": "1700000001000000000",
        "attributes": attrs or [],
        "status": {"code": 1},
        "events": [],
    }


# ---------------------------------------------------------------------------
# Route alias tests
# ---------------------------------------------------------------------------

class TestOtlpRouteAliases:

    @pytest.mark.asyncio
    async def test_v1_traces_accepts_otlp_json(self, client):
        span = _make_span("test.span", attrs=[
            {"key": "gen_ai.agent.id", "value": {"stringValue": "my-agent"}},
        ])
        body = _otlp_body([span])
        resp = await client.post(
            "/v1/traces",
            json=body,
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ingested"] == 1
        assert data["rejected"] == 0

    @pytest.mark.asyncio
    async def test_v1_metrics_stub_returns_200(self, client):
        resp = await client.post(
            "/v1/metrics",
            json={"resourceMetrics": []},
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_v1_logs_stub_returns_200(self, client):
        resp = await client.post(
            "/v1/logs",
            json={"resourceLogs": []},
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Attribute mapping tests
# ---------------------------------------------------------------------------

class TestOpenClawAttributeMapping:

    @pytest.mark.asyncio
    async def test_service_name_fallback_as_agent_id(self, client, db):
        """When gen_ai.agent.id is absent, service.name is used."""
        span = _make_span("openclaw.request")
        body = _otlp_body(
            [span],
            resource_attrs=[
                {"key": "service.name", "value": {"stringValue": "my-openclaw-agent"}},
            ],
        )
        resp = await client.post(
            "/v1/traces",
            json=body,
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200
        spans = _get_spans(db)
        assert len(spans) == 1
        assert spans[0].agent_id == "my-openclaw-agent"

    @pytest.mark.asyncio
    async def test_explicit_agent_id_takes_precedence(self, client, db):
        """gen_ai.agent.id wins over service.name."""
        span = _make_span("openclaw.request", attrs=[
            {"key": "gen_ai.agent.id", "value": {"stringValue": "explicit-id"}},
        ])
        body = _otlp_body(
            [span],
            resource_attrs=[
                {"key": "service.name", "value": {"stringValue": "service-name"}},
            ],
        )
        resp = await client.post(
            "/v1/traces",
            json=body,
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200
        spans = _get_spans(db)
        assert len(spans) == 1
        assert spans[0].agent_id == "explicit-id"

    @pytest.mark.asyncio
    async def test_tool_name_extracted_from_span_name(self, client, db):
        """Span named 'tool.Read' extracts tool_name='Read'."""
        span = _make_span("tool.Read")
        body = _otlp_body([span])
        resp = await client.post(
            "/v1/traces",
            json=body,
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200
        spans = _get_spans(db)
        assert len(spans) == 1
        assert spans[0].tool_name == "Read"

    @pytest.mark.asyncio
    async def test_tool_exec_span_name(self, client, db):
        """Span named 'tool.exec' extracts tool_name='exec'."""
        span = _make_span("tool.exec", span_id="exec01")
        body = _otlp_body([span])
        resp = await client.post(
            "/v1/traces",
            json=body,
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200
        spans = _get_spans(db)
        assert len(spans) == 1
        assert spans[0].tool_name == "exec"

    @pytest.mark.asyncio
    async def test_explicit_tool_name_attr_takes_precedence(self, client, db):
        """gen_ai.tool.name wins over span name parsing."""
        span = _make_span("tool.Read", span_id="toolattr01", attrs=[
            {"key": "gen_ai.tool.name", "value": {"stringValue": "custom_tool"}},
        ])
        body = _otlp_body([span])
        resp = await client.post(
            "/v1/traces",
            json=body,
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200
        spans = _get_spans(db)
        assert len(spans) == 1
        assert spans[0].tool_name == "custom_tool"

    @pytest.mark.asyncio
    async def test_model_usage_span_extracts_tokens(self, client, db):
        """openclaw.model.usage spans carry token counts."""
        span = _make_span("openclaw.model.usage", span_id="usage01", attrs=[
            {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "500"}},
            {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "200"}},
            {"key": "gen_ai.request.model", "value": {"stringValue": "claude-sonnet-4-20250514"}},
        ])
        body = _otlp_body(
            [span],
            resource_attrs=[
                {"key": "service.name", "value": {"stringValue": "my-agent"}},
            ],
        )
        resp = await client.post(
            "/v1/traces",
            json=body,
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200
        spans = _get_spans(db)
        assert len(spans) == 1
        assert spans[0].input_tokens == 500
        assert spans[0].output_tokens == 200
        assert spans[0].model == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_gen_ai_system_fallback_for_provider(self, client, db):
        """gen_ai.system is used as provider when gen_ai.provider.name is absent."""
        span = _make_span("openclaw.model.usage", span_id="prov01", attrs=[
            {"key": "gen_ai.system", "value": {"stringValue": "anthropic"}},
        ])
        body = _otlp_body([span])
        resp = await client.post(
            "/v1/traces",
            json=body,
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200
        spans = _get_spans(db)
        assert len(spans) == 1
        assert spans[0].provider == "anthropic"

    @pytest.mark.asyncio
    async def test_realistic_openclaw_payload(self, client, db):
        """Full OpenClaw trace with request, agent turn, tools, and model usage."""
        resource_attrs = [
            {"key": "service.name", "value": {"stringValue": "my-openclaw"}},
        ]
        spans = [
            _make_span("openclaw.request", span_id="req001", trace_id="t001"),
            _make_span("openclaw.agent.turn", span_id="turn001", trace_id="t001"),
            _make_span("tool.Read", span_id="read001", trace_id="t001"),
            _make_span("tool.exec", span_id="exec001", trace_id="t001"),
            _make_span("tool.Write", span_id="write001", trace_id="t001"),
            _make_span("openclaw.model.usage", span_id="usage001", trace_id="t001", attrs=[
                {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "1000"}},
                {"key": "gen_ai.usage.output_tokens", "value": {"intValue": "300"}},
                {"key": "gen_ai.request.model", "value": {"stringValue": "claude-sonnet-4-20250514"}},
                {"key": "gen_ai.system", "value": {"stringValue": "anthropic"}},
            ]),
        ]
        body = _otlp_body(spans, resource_attrs=resource_attrs)
        resp = await client.post(
            "/v1/traces",
            json=body,
            headers={"Authorization": f"Bearer {INGEST_SECRET}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ingested"] == 6
        assert data["rejected"] == 0

        all_spans = _get_spans(db)
        assert len(all_spans) == 6

        # All spans should inherit agent_id from service.name
        for s in all_spans:
            assert s.agent_id == "my-openclaw"

        # Tool spans should have tool_name extracted
        tool_spans = [s for s in all_spans if s.tool_name]
        tool_names = sorted(s.tool_name for s in tool_spans)
        assert tool_names == ["Read", "Write", "exec"]

        # Model usage span should have tokens
        usage_spans = [s for s in all_spans if s.name == "openclaw.model.usage"]
        assert len(usage_spans) == 1
        assert usage_spans[0].input_tokens == 1000
        assert usage_spans[0].output_tokens == 300
        assert usage_spans[0].provider == "anthropic"


# ---------------------------------------------------------------------------
# GET /api/v1/agents
# ---------------------------------------------------------------------------

class TestAgentsEndpoint:

    @pytest.mark.asyncio
    async def test_list_agents_returns_registered_agents(self, client, db):
        from tokenjam.core.models import AgentRecord
        from tokenjam.utils.time_parse import utcnow

        t = utcnow()
        db.upsert_agent(AgentRecord(agent_id="agent-a", first_seen=t, last_seen=t))
        db.upsert_agent(AgentRecord(agent_id="agent-b", first_seen=t, last_seen=t))

        from tests.factories import make_llm_span
        db.insert_span(make_llm_span(agent_id="agent-a", cost_usd=1.50))
        db.insert_span(make_llm_span(agent_id="agent-a", cost_usd=0.50))

        resp = await client.get("/api/v1/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data

        by_id = {a["agent_id"]: a for a in data["agents"]}
        assert "agent-a" in by_id
        assert "agent-b" in by_id

        a = by_id["agent-a"]
        assert abs(a["lifetime_cost_usd"] - 2.0) < 0.01
        assert a["first_seen"] is not None
        assert a["last_seen"] is not None

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, client):
        resp = await client.get("/api/v1/agents")
        assert resp.status_code == 200
        assert resp.json() == {"agents": []}
