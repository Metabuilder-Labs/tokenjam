"""Public HTTP client for shipping spans into a running ``tj serve``.

This is the entry point used by external integrations that cannot rely on the
in-process OTel TracerProvider — most notably, the upstream LiteLLM named
callback (``litellm.success_callback = ["tokenjam"]``) which lives in the
LiteLLM repo and only depends on this public surface.

For in-process use inside a tokenjam-aware app, prefer ``patch_litellm()`` —
it produces the same spans via the OTel pipeline.
"""
from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from tokenjam.core.pricing import provider_for_model
from tokenjam.otel.semconv import GenAIAttributes, TjAttributes

logger = logging.getLogger("tokenjam.sdk")

_TIMEOUT_SECS = 5.0


class TokenJamClient:
    """Thin HTTP client that POSTs a single LiteLLM call as an OTLP span.

    Designed to be embedded in foreign codebases (e.g. LiteLLM's named-callback
    machinery), so it has no dependency on the rest of the tokenjam SDK at
    construction time and never raises from its public methods — failures are
    logged at ``debug`` and the event is dropped.
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:7391",
        ingest_secret: Optional[str] = None,
    ) -> None:
        # Accept either the server base URL or the full /api/v1/spans path.
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/api/v1/spans"):
            endpoint = f"{endpoint}/api/v1/spans"
        self._endpoint = endpoint
        self._secret = ingest_secret

    def emit_litellm_span(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
        success: bool,
    ) -> None:
        """Translate a LiteLLM success/failure callback payload into an OTLP
        span and POST it to ``tj serve``.

        Non-blocking: all errors are swallowed and logged at debug.
        """
        try:
            span = _build_litellm_span(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                success=success,
            )
            payload = {
                "resourceSpans": [{
                    "resource": {"attributes": [
                        {"key": "service.name",
                         "value": {"stringValue": "litellm"}},
                    ]},
                    "scopeSpans": [{"spans": [span]}],
                }],
            }
            headers = {"Content-Type": "application/json"}
            if self._secret:
                headers["Authorization"] = f"Bearer {self._secret}"
            resp = httpx.post(
                self._endpoint,
                json=payload,
                headers=headers,
                timeout=_TIMEOUT_SECS,
            )
            if resp.status_code >= 300:
                logger.debug(
                    "tj serve returned %d on emit_litellm_span",
                    resp.status_code,
                )
        except Exception as exc:  # noqa: BLE001 — non-blocking by design
            logger.debug("emit_litellm_span failed: %s", exc)


# ---------------------------------------------------------------------------
# Payload construction (pure functions, exposed for unit testing)
# ---------------------------------------------------------------------------


def _build_litellm_span(
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: datetime,
    end_time: datetime,
    success: bool,
) -> dict[str, Any]:
    """Build an OTLP JSON span dict from a LiteLLM callback payload."""
    raw_model = str(kwargs.get("model") or "unknown")
    model = _strip_provider_prefix(raw_model)
    provider = _parse_provider(raw_model, response_obj)
    metadata = kwargs.get("metadata") or {}

    attrs: dict[str, Any] = {
        GenAIAttributes.REQUEST_MODEL: model,
        GenAIAttributes.PROVIDER_NAME: provider,
    }

    # Token usage from response.usage (OpenAI-style dict or pydantic model)
    usage = _get_usage(response_obj)
    if usage:
        prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
        completion_tokens = _coerce_int(usage.get("completion_tokens"))
        if prompt_tokens is not None:
            attrs[GenAIAttributes.INPUT_TOKENS] = prompt_tokens
        if completion_tokens is not None:
            attrs[GenAIAttributes.OUTPUT_TOKENS] = completion_tokens
        # Anthropic-style cache token fields, if present
        cache_read = _coerce_int(usage.get("cache_read_input_tokens"))
        cache_create = _coerce_int(usage.get("cache_creation_input_tokens"))
        if cache_read is not None:
            attrs[GenAIAttributes.CACHE_READ_TOKENS] = cache_read
        if cache_create is not None:
            attrs[GenAIAttributes.CACHE_CREATE_TOKENS] = cache_create

    # LiteLLM puts a precomputed cost on kwargs / response.hidden_params
    cost = _get_response_cost(kwargs, response_obj)
    if cost is not None:
        attrs[TjAttributes.COST_USD] = cost

    # Per-call agent + session tags supplied via metadata
    if isinstance(metadata, dict):
        agent_id = metadata.get("tj_agent_id")
        if agent_id:
            attrs[GenAIAttributes.AGENT_ID] = str(agent_id)
        session_id = metadata.get("tj_session_id")
        if session_id:
            attrs[GenAIAttributes.CONVERSATION_ID] = str(session_id)

    span: dict[str, Any] = {
        "traceId": _new_trace_id(),
        "spanId": _new_span_id(),
        "name": GenAIAttributes.SPAN_LLM_CALL,
        "kind": 3,  # CLIENT
        "startTimeUnixNano": str(_to_unix_nanos(start_time)),
        "endTimeUnixNano": str(_to_unix_nanos(end_time)),
        "attributes": [
            {"key": k, "value": _to_otlp_value(v)} for k, v in attrs.items()
        ],
        "status": {"code": 1 if success else 2},
        "events": [],
    }
    if not success:
        msg = _extract_error_message(response_obj)
        if msg:
            span["status"]["message"] = msg
    return span


def _strip_provider_prefix(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def _parse_provider(model: str, response: Any) -> str:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        p = hidden.get("custom_llm_provider")
        if p:
            return str(p)
    if isinstance(response, dict):
        hidden = response.get("_hidden_params")
        if isinstance(hidden, dict) and hidden.get("custom_llm_provider"):
            return str(hidden["custom_llm_provider"])
    if "/" in model:
        return model.split("/", 1)[0]
    # Infer from the bare model name; never write the bogus "litellm" (#194).
    return provider_for_model(model) or "unknown"


def _get_usage(response: Any) -> Optional[dict[str, Any]]:
    """Normalize response.usage to a dict, regardless of pydantic/dict shape."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    # pydantic v2 model or plain object with attributes
    out: dict[str, Any] = {}
    for k in (
        "prompt_tokens", "completion_tokens",
        "cache_read_input_tokens", "cache_creation_input_tokens",
    ):
        v = getattr(usage, k, None)
        if v is not None:
            out[k] = v
    return out or None


def _get_response_cost(kwargs: dict[str, Any], response: Any) -> Optional[float]:
    # LiteLLM exposes response_cost on the callback kwargs
    cost = kwargs.get("response_cost")
    if cost is None:
        hidden = getattr(response, "_hidden_params", None)
        if isinstance(hidden, dict):
            cost = hidden.get("response_cost")
    if cost is None:
        return None
    try:
        return float(cost)
    except (TypeError, ValueError):
        return None


def _extract_error_message(response: Any) -> Optional[str]:
    if isinstance(response, Exception):
        return str(response)
    if isinstance(response, dict):
        err = response.get("error") or response.get("message")
        if err:
            return str(err)
    return None


def _coerce_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _to_unix_nanos(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _new_trace_id() -> str:
    return secrets.token_hex(16)


def _new_span_id() -> str:
    return secrets.token_hex(8)


def _to_otlp_value(v: Any) -> dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}
