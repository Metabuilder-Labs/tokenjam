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
from tests.factories import make_llm_span, make_tool_span


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


async def test_cost_includes_daily_series_for_chart(client):
    """/api/v1/cost carries a daily-bucketed series (per date+agent+model) the
    Cost chart consumes — finer than the grouped rows (#113)."""
    await _ingest_sample_span(client)
    resp = await client.get("/api/v1/cost", params={"since": "7d", "group_by": "day"})
    assert resp.status_code == 200
    data = resp.json()
    assert "series" in data and isinstance(data["series"], list)
    if data["series"]:
        item = data["series"][0]
        assert {"date", "agent_id", "model", "cost_usd",
                "input_tokens", "output_tokens"} <= set(item)


_FRAMING_KEYS = {
    "pricing_mode", "plan_tier", "plan_label", "plan_monthly_usd",
    "subscription_share_pct", "api_share_pct", "display_rule", "qualifier_text",
}


async def test_cost_response_includes_framing_block(client):
    await _ingest_sample_span(client)
    resp = await client.get("/api/v1/cost")
    assert resp.status_code == 200
    framing = resp.json()["framing"]
    assert _FRAMING_KEYS <= set(framing)


async def test_optimize_response_includes_framing_block(client):
    await _ingest_sample_span(client)
    resp = await client.get("/api/v1/optimize?since=30d")
    assert resp.status_code == 200
    data = resp.json()
    if data.get("error") == "no_data":
        pytest.skip("no spans landed for optimize in this fixture")
    assert _FRAMING_KEYS <= set(data["framing"])


async def test_root_serves_lens_title(client):
    """Brand pass (#114): the served dashboard <title> is 'TokenJam Lens'."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "<title>TokenJam Lens</title>" in resp.text


async def test_optimize_chain_framing_and_recoverable_fields(client):
    """The framing block (#110) + per-finding recoverable fields (#111) are
    both present on /api/v1/optimize — validates the chain Overview relies on."""
    await _ingest_sample_span(client)
    resp = await client.get("/api/v1/optimize?since=30d")
    data = resp.json()
    if data.get("error") == "no_data":
        pytest.skip("no spans landed for optimize in this fixture")
    assert _FRAMING_KEYS <= set(data["framing"])
    # The savings analyzers carry the recoverable contract fields (#111).
    # cache-recommend / budget-projection intentionally do not.
    findings = data.get("findings") or {}
    for name in ("cache", "script", "trim"):
        if name in findings:
            assert "estimated_recoverable_usd" in findings[name]
            assert "estimate_basis" in findings[name]


async def test_optimize_fast_skips_trim(client):
    """fast=true skips the expensive Trim analyzer and reports it (#114)."""
    await _ingest_sample_span(client)
    resp = await client.get("/api/v1/optimize?since=30d&fast=true")
    assert resp.status_code == 200
    data = resp.json()
    if data.get("error") == "no_data":
        pytest.skip("no spans landed for optimize in this fixture")
    assert "trim" in data.get("skipped_analyzers", [])
    assert "trim" not in (data.get("findings") or {})


async def test_budget_framing_reflects_configured_subscription_plan(db):
    """The budget surface has no window, so framing falls back to the
    declared plan in config (#110)."""
    from tokenjam.core.config import ProviderBudget

    cfg = TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret=INGEST_SECRET),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
    )
    cfg.budgets["anthropic"] = ProviderBudget(plan="max_5x")
    pipeline = IngestPipeline(db=db, config=cfg)
    app = create_app(config=cfg, db=db, ingest_pipeline=pipeline)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/api/v1/budget")
    assert resp.status_code == 200
    framing = resp.json()["framing"]
    assert framing["pricing_mode"] == "subscription"
    assert framing["plan_tier"] == "max_5x"
    assert framing["plan_label"] == "Max 5x plan"
    assert framing["display_rule"] == "suppress_dollars_for_subscription_share"


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

    # Verify sessions table agrees
    status = await client.get("/api/v1/status")
    status_agents = [a["agent_id"] for a in status.json()["agents"]]
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

    status_agents = {a["agent_id"] for a in status.json()["agents"]}
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
