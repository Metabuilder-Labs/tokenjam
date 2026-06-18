"""Integration tests for the policy engine wired into the proxy app (#220).

Verifies, through the real Starlette app (httpx MockTransport upstream):
- the POLICY path runs the engine, records the envelope, and STILL forwards
  the request UNMODIFIED (suggest mode enforces nothing),
- the OBSERVE_ONLY path NEVER builds a policy envelope (the engine never runs),
- the `/__tj/policy/decisions` + `/__tj/policy/policies` read surfaces,
- the unvalidated label rides the recorded envelope.
"""
from __future__ import annotations

import httpx
import pytest

from tokenjam.core.config import PolicyConfig, ProviderBudget, TjConfig
from tokenjam.proxy.app import build_proxy_app
from tokenjam.proxy.engine import ACTION_WOULD_BLOCK, UNVALIDATED_LABEL, PolicyOutcome, register_policy
from tokenjam.proxy.observer import ProxyObserver


@register_policy("itest_block")
def _itest_block(policy, request):
    return PolicyOutcome(would_action=ACTION_WOULD_BLOCK, reason="itest: would block")


def _config(plans: dict[str, str], policies=None) -> TjConfig:
    return TjConfig(
        version="1",
        budgets={p: ProviderBudget(plan=plan) for p, plan in plans.items()},
        policies=policies or [],
    )


class _Upstream:
    def __init__(self):
        self.last_body: bytes | None = None
        self.last_request: httpx.Request | None = None

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.last_body = await request.aread()
        self.last_request = request

        async def _agen():
            yield b'{"ok":true}'
        return httpx.Response(200, headers={"content-type": "application/json"}, content=_agen())


def _client(upstream):
    return httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))


async def _post(app, path, body=b"{}", headers=None):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
        return await c.post(path, content=body, headers=headers or {})


async def _get(app, path):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
        return await c.get(path)


@pytest.mark.asyncio
async def test_policy_path_evaluates_records_and_forwards_unmodified():
    cfg = _config({"openai": "api"}, [PolicyConfig(name="blocker", kind="itest_block")])
    upstream = _Upstream()
    observer = ProxyObserver()
    app = build_proxy_app(cfg, observer=observer, client=_client(upstream))

    body = b'{"model":"gpt-4o","messages":[]}'
    resp = await _post(app, "/v1/chat/completions", body=body,
                       headers={"authorization": "Bearer sk-caller"})

    assert resp.status_code == 200
    # Suggest mode: the request was forwarded UNMODIFIED despite the would_block.
    assert upstream.last_body == body
    # The envelope was recorded with what the policy WOULD do + the label.
    obs = observer.observations[0]
    assert obs.decision == "policy"
    assert obs.policy is not None
    assert obs.policy["overall_action"] == ACTION_WOULD_BLOCK
    assert obs.policy["enforced"] is False
    assert obs.policy["label"] == UNVALIDATED_LABEL
    assert obs.policy["evaluations"][0]["policy_name"] == "blocker"


@pytest.mark.asyncio
async def test_observe_only_path_never_builds_an_envelope():
    # Subscription traffic is observe-only — the engine must never run, so no
    # policy envelope is attached even though a policy is defined.
    cfg = _config({"anthropic": "max_5x"}, [PolicyConfig(name="blocker", kind="itest_block")])
    upstream = _Upstream()
    observer = ProxyObserver()
    app = build_proxy_app(cfg, observer=observer, client=_client(upstream))

    await _post(app, "/v1/messages", body=b"{}", headers={"x-api-key": "sk"})

    obs = observer.observations[0]
    assert obs.decision == "observe_only"
    assert obs.policy is None  # engine never evaluated observe-only traffic


@pytest.mark.asyncio
async def test_policies_read_endpoint_lists_defined_policies():
    cfg = _config({"openai": "api"},
                  [PolicyConfig(name="blocker", kind="itest_block", target_provider="openai")])
    app = build_proxy_app(cfg, observer=ProxyObserver(), client=_client(_Upstream()))

    resp = await _get(app, "/__tj/policy/policies")
    data = resp.json()
    assert data["label"] == "unvalidated"
    assert data["policies"][0]["name"] == "blocker"
    assert data["policies"][0]["target_provider"] == "openai"


@pytest.mark.asyncio
async def test_decisions_read_endpoint_returns_recorded_envelopes():
    cfg = _config({"openai": "api"}, [PolicyConfig(name="blocker", kind="itest_block")])
    observer = ProxyObserver()
    app = build_proxy_app(cfg, observer=observer, client=_client(_Upstream()))

    # Generate one policy decision + one observe-only (no envelope).
    await _post(app, "/v1/chat/completions", body=b"{}")

    resp = await _get(app, "/__tj/policy/decisions")
    data = resp.json()
    assert data["label"] == "unvalidated"
    assert len(data["decisions"]) == 1  # only the POLICY-path one carries an envelope
    assert data["decisions"][0]["policy"]["overall_action"] == ACTION_WOULD_BLOCK
