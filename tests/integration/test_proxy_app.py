"""Integration tests for the proxy forwarding app (#219) — suggest mode.

Uses ``httpx.MockTransport`` as the upstream provider (no real network) and
``httpx.ASGITransport`` to drive the proxy app. Verifies:
- subscription/unknown → forwarded UNMODIFIED + recorded observe-only,
- api/usage → recorded on the policy path (still forwarded in suggest mode),
- pass-through is sacred: a classification error still forwards,
- the proxy holds no keys: the caller's credentials pass through untouched,
- streaming responses are forwarded as streams.
"""
from __future__ import annotations

import httpx
import pytest

from tokenjam.core.config import ProviderBudget, TjConfig
from tokenjam.proxy import app as proxy_app_module
from tokenjam.proxy.app import build_proxy_app
from tokenjam.proxy.observer import ProxyObserver


def _config(provider_plans: dict[str, str]) -> TjConfig:
    return TjConfig(
        version="1",
        budgets={p: ProviderBudget(plan=plan) for p, plan in provider_plans.items()},
    )


class _Upstream:
    """Records the last forwarded request and returns a canned response.

    The handler is async and returns an async-streamed body so the full
    forward path exercises streaming (``client.send(stream=True)`` →
    ``aiter_raw()``), matching how a real provider responds.
    """

    def __init__(self, status: int = 200, body: bytes = b'{"ok":true}',
                 headers: dict | None = None, chunks: list[bytes] | None = None):
        self.last_request: httpx.Request | None = None
        self.last_body: bytes | None = None
        self._status = status
        self._headers = headers or {"content-type": "application/json"}
        self._chunks = chunks if chunks is not None else [body]

    async def handler(self, request: httpx.Request) -> httpx.Response:
        # Read the body so we can assert it was forwarded unmodified.
        self.last_body = await request.aread()
        self.last_request = request
        chunks = list(self._chunks)

        async def _agen():
            for c in chunks:
                yield c

        return httpx.Response(self._status, headers=self._headers, content=_agen())


def _make_client(upstream: _Upstream) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))


async def _call(app, method, path, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        return await client.request(method, path, **kwargs)


@pytest.mark.asyncio
async def test_subscription_forwarded_unmodified_and_observed():
    cfg = _config({"anthropic": "max_5x"})
    upstream = _Upstream()
    observer = ProxyObserver()
    app = build_proxy_app(cfg, observer=observer, client=_make_client(upstream))

    body = b'{"model":"claude-3","messages":[]}'
    resp = await _call(app, "POST", "/v1/messages", content=body,
                       headers={"x-api-key": "sk-caller", "anthropic-version": "2023-06-01"})

    assert resp.status_code == 200
    # Forwarded UNMODIFIED to the real provider.
    assert upstream.last_request is not None
    assert str(upstream.last_request.url) == "https://api.anthropic.com/v1/messages"
    assert upstream.last_body == body
    # Observed only — never a policy decision.
    obs = observer.observations
    assert len(obs) == 1
    assert obs[0].provider == "anthropic"
    assert obs[0].pricing_mode == "subscription"
    assert obs[0].decision == "observe_only"
    assert obs[0].forwarded is True


@pytest.mark.asyncio
async def test_api_reaches_policy_path_but_still_forwards():
    cfg = _config({"openai": "api"})
    upstream = _Upstream()
    observer = ProxyObserver()
    app = build_proxy_app(cfg, observer=observer, client=_make_client(upstream))

    resp = await _call(app, "POST", "/v1/chat/completions", content=b"{}",
                       headers={"authorization": "Bearer sk-caller"})

    assert resp.status_code == 200
    assert str(upstream.last_request.url) == "https://api.openai.com/v1/chat/completions"
    # api/usage-billed is the ONLY traffic on the policy path; suggest mode still
    # forwards it unmodified (enforces nothing).
    assert observer.observations[0].decision == "policy"
    assert observer.observations[0].forwarded is True


@pytest.mark.asyncio
async def test_holds_no_keys_client_credentials_pass_through():
    cfg = _config({"anthropic": "api"})
    upstream = _Upstream()
    app = build_proxy_app(cfg, observer=ProxyObserver(), client=_make_client(upstream))

    await _call(app, "POST", "/v1/messages", content=b"{}",
                headers={"x-api-key": "sk-secret-caller-key"})

    # The proxy holds no keys — the caller's credential reaches upstream as-is,
    # and the proxy never injected one of its own.
    assert upstream.last_request.headers.get("x-api-key") == "sk-secret-caller-key"


@pytest.mark.asyncio
async def test_pass_through_on_classification_error(monkeypatch):
    cfg = _config({"anthropic": "api"})
    upstream = _Upstream()
    observer = ProxyObserver()

    def _boom(*a, **k):
        raise RuntimeError("classifier exploded")

    # Pass-through is sacred: even if classification raises, the request must
    # still be forwarded unmodified.
    monkeypatch.setattr(proxy_app_module, "classify", _boom)
    app = build_proxy_app(cfg, observer=observer, client=_make_client(upstream))

    resp = await _call(app, "POST", "/v1/messages", content=b'{"x":1}')

    assert resp.status_code == 200
    assert upstream.last_request is not None            # forwarded despite the error
    assert upstream.last_body == b'{"x":1}'
    # A fail-safe observe-only record is still produced.
    assert observer.observations[0].decision == "observe_only"
    assert observer.observations[0].reason == "classification_error_failsafe"


@pytest.mark.asyncio
async def test_streaming_response_forwarded_as_stream():
    cfg = _config({"anthropic": "api"})
    # An SSE-style streamed upstream body, forwarded chunk-by-chunk.
    chunks = [b"data: a\n\n", b"data: b\n\n", b"data: [DONE]\n\n"]
    upstream = _Upstream(headers={"content-type": "text/event-stream"}, chunks=chunks)
    app = build_proxy_app(cfg, observer=ProxyObserver(), client=_make_client(upstream))

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        async with client.stream("POST", "/v1/messages", content=b"{}") as resp:
            assert resp.status_code == 200
            assert resp.headers.get("content-type") == "text/event-stream"
            received = b"".join([part async for part in resp.aiter_raw()])

    assert received == b"".join(chunks)


@pytest.mark.asyncio
async def test_unrecognised_path_returns_404_no_forward():
    cfg = _config({"anthropic": "api"})
    upstream = _Upstream()
    app = build_proxy_app(cfg, observer=ProxyObserver(), client=_make_client(upstream))

    resp = await _call(app, "GET", "/v1/unknown-endpoint")
    assert resp.status_code == 404
    assert upstream.last_request is None  # nothing forwarded for an unknown upstream
