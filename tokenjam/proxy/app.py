"""The proxy's HTTP forwarding app (#219) — suggest mode only.

A small Starlette app that speaks the Anthropic (``/v1/messages``) and OpenAI
(``/v1/chat/completions``) request shapes, runs the pricing-mode gate, and
forwards the request to the real provider **unmodified**, streaming the response
back. It holds no keys — the caller's credentials pass straight through.

Safety doctrine, enforced structurally:
- The gate runs first, but in suggest mode the *action* is identical for both
  paths (forward unmodified). The gate decision is only **recorded**.
- Pass-through is sacred: any error in classification / recording is swallowed
  and the request is forwarded anyway.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from tokenjam.proxy.gate import classify
from tokenjam.proxy.observer import ProxyObserver

logger = logging.getLogger("tokenjam.proxy")

# Per-route provider mapping. The path itself identifies the provider's API
# shape; the catch-all resolves to "unknown" → observe-only (fail-safe).
_ROUTE_PROVIDER = {
    "/v1/messages": "anthropic",
    "/v1/chat/completions": "openai",
}

# Hop-by-hop headers must not be forwarded (RFC 7230 §6.1) plus host/length,
# which httpx / the ASGI server recompute for the new connection.
_HOP_BY_HOP = frozenset({
    "host", "content-length", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade",
})


def _provider_for(path: str) -> str:
    return _ROUTE_PROVIDER.get(path, "unknown")


def _base_url_for(config: Any, provider: str) -> str | None:
    if provider == "anthropic":
        return config.proxy.anthropic_base_url.rstrip("/")
    if provider == "openai":
        return config.proxy.openai_base_url.rstrip("/")
    return None


def _forward_headers(request: Request) -> list[tuple[bytes, bytes]]:
    """Client headers minus hop-by-hop — credentials pass through untouched."""
    return [
        (k, v) for k, v in request.headers.raw
        if k.decode("latin-1").lower() not in _HOP_BY_HOP
    ]


def _response_headers(upstream: httpx.Response) -> list[tuple[bytes, bytes]]:
    return [
        (k, v) for k, v in upstream.headers.raw
        if k.decode("latin-1").lower() not in _HOP_BY_HOP
    ]


def build_proxy_app(config: Any, observer: ProxyObserver | None = None,
                    client: httpx.AsyncClient | None = None) -> Starlette:
    """Build the Starlette proxy app.

    ``observer`` and ``client`` are injectable for testing — pass an
    ``httpx.AsyncClient`` backed by ``httpx.MockTransport`` to avoid real
    network calls. When ``client`` is None a real client is created on startup
    and closed on shutdown.
    """
    observer = observer or ProxyObserver()
    state: dict[str, Any] = {"client": client, "owns_client": client is None}

    async def _get_client() -> httpx.AsyncClient:
        if state["client"] is None:
            state["client"] = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        return state["client"]

    async def _forward(request: Request, base_url: str) -> StreamingResponse:
        client_ = await _get_client()
        url = base_url + request.url.path
        if request.url.query:
            url += "?" + request.url.query
        body = await request.body()
        upstream_req = client_.build_request(
            request.method, url, headers=_forward_headers(request), content=body,
        )
        upstream = await client_.send(upstream_req, stream=True)
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers={k.decode("latin-1"): v.decode("latin-1")
                     for k, v in _response_headers(upstream)},
            background=BackgroundTask(upstream.aclose),
        )

    async def _handle(request: Request) -> Any:
        provider = _provider_for(request.url.path)
        base_url = _base_url_for(config, provider)

        # The pricing-mode gate runs FIRST. Pass-through is sacred: if anything
        # in classification raises, we forward unmodified anyway.
        decision = None
        try:
            decision = classify(
                config, provider, killswitch=bool(config.proxy.killswitch),
            )
        except Exception:  # noqa: BLE001 — never let our logic break traffic
            logger.exception("proxy gate classification failed; forwarding unmodified")

        # No upstream known (unrecognised path) → nothing to forward to.
        if base_url is None:
            _safe_record(observer, request, decision, forwarded=False)
            return JSONResponse(
                {"error": "tj proxy: unrecognised path; no upstream configured",
                 "path": request.url.path},
                status_code=404,
            )

        # Forward UNMODIFIED (suggest mode enforces nothing on either path).
        response = await _forward(request, base_url)
        _safe_record(observer, request, decision, forwarded=True)
        return response

    routes = [
        Route("/v1/messages", _handle, methods=["POST"]),
        Route("/v1/chat/completions", _handle, methods=["POST"]),
        Route("/{path:path}", _handle, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
    ]

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(_app: Starlette):
        try:
            yield
        finally:
            if state["owns_client"] and state["client"] is not None:
                await state["client"].aclose()

    app = Starlette(routes=routes, lifespan=_lifespan)
    app.state.observer = observer
    app.state.tj_config = config
    return app


def _safe_record(observer: ProxyObserver, request: Request, decision: Any,
                 *, forwarded: bool) -> None:
    """Record the observation; recording must never break pass-through."""
    try:
        if decision is None:
            # Classification failed — synthesise a fail-safe observe-only record
            # so the gate failure is still visible without re-running classify.
            from tokenjam.proxy.gate import OBSERVE_ONLY, GateDecision
            decision = GateDecision(
                provider=_provider_for(request.url.path), plan_tier=None,
                pricing_mode="unknown", path=OBSERVE_ONLY,
                reason="classification_error_failsafe",
            )
        observer.record(
            method=request.method, path=request.url.path,
            decision=decision, forwarded=forwarded,
        )
    except Exception:  # noqa: BLE001
        logger.exception("proxy observation recording failed (ignored)")
