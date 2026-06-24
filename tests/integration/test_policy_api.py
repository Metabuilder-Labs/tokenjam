"""Integration tests for GET /api/v1/policy/* (#223).

These power the MCP policy tools when `tj serve` holds the DB lock. Verifies the
three routes return the suggest-mode / unvalidated views and that the savings
route never presents realized savings.
"""
from __future__ import annotations

import json

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import (
    ApiAuthConfig,
    ApiConfig,
    PolicyConfig,
    ProviderBudget,
    SecurityConfig,
    TjConfig,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.proxy.audit import AuditSink
from tokenjam.proxy.engine import (
    ACTION_WOULD_BLOCK,
    PolicyEnvelope,
    PolicyEvaluation,
)
from tokenjam.proxy.gate import POLICY, GateDecision
from tokenjam.proxy.observer import ProxyObserver
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def config():
    return TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="x"),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        budgets={"openai": ProviderBudget(plan="api")},
        policies=[PolicyConfig(name="cap", kind="budget_cap")],
    )


def _seed(db):
    sess = make_session(agent_id="codegen", total_cost_usd=10.0)
    db.upsert_session(sess)
    db.insert_span(make_llm_span(agent_id="codegen", provider="openai",
                                 model="gpt-4o", cost_usd=10.0,
                                 session_id=sess.session_id))
    obs = ProxyObserver(sink=AuditSink(db))
    obs.record(method="POST", path="/v1/chat/completions",
               decision=GateDecision(provider="openai", plan_tier="api",
                                     pricing_mode="api", path=POLICY, reason="api"),
               envelope=PolicyEnvelope(
                   ts=utcnow().isoformat(), provider="openai",
                   path="/v1/chat/completions", agent=None, gate_path=POLICY,
                   overall_action=ACTION_WOULD_BLOCK,
                   evaluations=[PolicyEvaluation(
                       policy_name="cap", kind="budget_cap", mode="suggest",
                       would_action=ACTION_WOULD_BLOCK, reason="over",
                       enforcement_gated=False,
                       details={"estimated_recoverable_usd": 2.0})]))


@pytest.fixture
def client(config, db):
    _seed(db)
    app = create_app(config=config, db=db,
                     ingest_pipeline=IngestPipeline(db=db, config=config))
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_policy_status_route(client):
    async with client:
        resp = await client.get("/api/v1/policy/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["suggest_mode"] is True
    assert data["label"] == "unvalidated"
    assert data["policies"][0]["name"] == "cap"
    assert len(data["recent_decisions"]) == 1


@pytest.mark.asyncio
async def test_policy_savings_route_never_realized(client):
    async with client:
        resp = await client.get("/api/v1/policy/savings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["estimated_recoverable_usd"] == 2.0
    assert data["actual_spend_usd"] == 10.0
    assert data["realized"] is False
    assert data["label"] == "unvalidated"
    non_disclaimer = {k: v for k, v in data.items() if k != "disclaimer"}
    assert "saved" not in json.dumps(non_disclaimer).lower()


@pytest.mark.asyncio
async def test_policy_suggestions_route(client):
    async with client:
        resp = await client.get("/api/v1/policy/suggestions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["label"] == "unvalidated"
    assert data["suggestions"][0]["kind"] == "budget_cap"
    assert data["suggestions"][0]["provider"] == "openai"
    assert "not validated" in data["note"].lower()
