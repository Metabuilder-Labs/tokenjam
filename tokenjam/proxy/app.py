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

from tokenjam.proxy.engine import POLICY, PolicyEngine, PolicyRequest
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


def _policy_request(request: Request, body: bytes) -> PolicyRequest:
    """Build the pure (HTTP-free) policy-evaluation context from a request.

    The body is parsed best-effort for kind-specific evaluators to read (e.g. a
    future budget_cap reading the model); a malformed body is simply None.
    """
    import json
    parsed: dict | None = None
    try:
        loaded = json.loads(body or b"{}")
        if isinstance(loaded, dict):
            parsed = loaded
    except Exception:  # noqa: BLE001 — body parsing never breaks the request
        parsed = None
    return PolicyRequest(
        provider=_provider_for(request.url.path),
        path=request.url.path,
        agent=None,  # the proxy does not resolve the agent at request time (MVP)
        body=parsed,
    )


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
                    client: httpx.AsyncClient | None = None,
                    engine: PolicyEngine | None = None,
                    db: Any = None) -> Starlette:
    """Build the Starlette proxy app.

    ``observer``, ``client``, ``engine`` and ``db`` are injectable for testing —
    pass an ``httpx.AsyncClient`` backed by ``httpx.MockTransport`` to avoid real
    network calls. When ``client`` is None a real client is created on startup
    and closed on shutdown; when ``engine`` is None it is built from ``config``,
    threading ``db`` (the shared tj-serve DuckDB) so in-process policies like
    budget_cap (#222) can read current-cycle spend.
    """
    observer = observer or ProxyObserver()
    engine = engine or PolicyEngine.from_config(config, db=db)
    state: dict[str, Any] = {"client": client, "owns_client": client is None}

    async def _get_client() -> httpx.AsyncClient:
        if state["client"] is None:
            state["client"] = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        return state["client"]

    async def _forward(request: Request, base_url: str, body: bytes) -> StreamingResponse:
        client_ = await _get_client()
        url = base_url + request.url.path
        if request.url.query:
            url += "?" + request.url.query
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

        # Read the body once — needed both for forwarding and for policy context.
        body = await request.body()

        # POLICY-path branch (#220): ONLY api/usage-billed traffic reaches the
        # engine — the api-only guard inside the engine is belt-and-suspenders
        # with this check. Observe-only traffic NEVER builds an envelope. Pass-
        # through is sacred: any engine error forwards unmodified anyway.
        envelope = None
        if decision is not None and decision.path == POLICY:
            try:
                envelope = engine.evaluate(decision, _policy_request(request, body))
            except Exception:  # noqa: BLE001 — engine never breaks traffic
                logger.exception("policy engine failed; forwarding unmodified")

        # No upstream known (unrecognised path) → nothing to forward to.
        if base_url is None:
            _safe_record(observer, request, decision, forwarded=False, envelope=envelope)
            return JSONResponse(
                {"error": "tj proxy: unrecognised path; no upstream configured",
                 "path": request.url.path},
                status_code=404,
            )

        # Forward UNMODIFIED. Suggest mode enforces nothing on EITHER path — the
        # envelope only records what a policy WOULD do; the request is untouched.
        response = await _forward(request, base_url, body)
        _safe_record(observer, request, decision, forwarded=True, envelope=envelope)
        return response

    async def _policies(_request: Request) -> JSONResponse:
        """Read-only: the defined policies the engine loaded (#220 tj policy)."""
        return JSONResponse({
            "policies": [
                {"name": getattr(p, "name", ""), "kind": getattr(p, "kind", ""),
                 "mode": getattr(p, "mode", "suggest"),
                 "target_provider": getattr(p, "target_provider", None),
                 "target_agent": getattr(p, "target_agent", None),
                 "enabled": getattr(p, "enabled", True)}
                for p in engine.policies
            ],
            "label": "unvalidated",
            "note": "Suggest mode only — policies record what they WOULD do; "
                    "nothing is enforced. OSS policies run unvalidated.",
        })

    async def _decisions(_request: Request) -> JSONResponse:
        """Read-only: recent policy decisions from the observer ring (#220)."""
        policy_obs = [o for o in observer.as_dicts() if o.get("policy") is not None]
        return JSONResponse({"decisions": policy_obs, "label": "unvalidated"})

    routes = [
        Route("/v1/messages", _handle, methods=["POST"]),
        Route("/v1/chat/completions", _handle, methods=["POST"]),
        # tj-internal read surfaces (namespaced so they never collide with a
        # provider path). Consumed by `tj policy policies|decisions`.
        Route("/__tj/policy/policies", _policies, methods=["GET"]),
        Route("/__tj/policy/decisions", _decisions, methods=["GET"]),
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
                 *, forwarded: bool, envelope: Any = None) -> None:
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
            decision=decision, forwarded=forwarded, envelope=envelope,
        )
    except Exception:  # noqa: BLE001
        logger.exception("proxy observation recording failed (ignored)")
