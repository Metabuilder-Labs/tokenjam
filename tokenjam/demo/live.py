"""
Live-replay sink for `tj demo --live`.

By default a demo scenario injects its synthetic SDK-shaped spans into a
throwaway ``InMemoryBackend`` (see :mod:`tokenjam.demo.env`), so the output
never reaches the on-disk DuckDB that ``tj serve`` reads — the live dashboard
can't render it. ``--live`` routes the same spans through the REAL ingest path:
each ``NormalizedSpan`` is serialized to OTLP JSON and POSTed to a running
``tj serve``'s ``/api/v1/spans`` endpoint, exactly as the SDK's
``TjHttpExporter`` does. The demo then doubles as an SDK-zone dogfood tool.

``_span_to_otlp`` is the inverse of ``otlp_parsing.parse_otlp_span``: it maps a
``NormalizedSpan``'s indexed fields back onto the ``gen_ai.*`` / ``tokenjam.*``
attribute keys the receive path reads, so the round-trip reproduces an
equivalent span (and therefore the same alerts) on the server side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx

from tokenjam.core.models import NormalizedSpan, SpanKind, SpanStatus
from tokenjam.otel.semconv import (
    GenAIAttributes,
    ResourceAttributes,
    TjAttributes,
)
from tokenjam.utils.ids import new_span_id

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig


# How long a single replay POST may take before we give up. Generous: a local
# daemon answers in milliseconds, but a scenario can carry ~30 spans and the
# server runs cost + drift hooks synchronously on ingest.
_REPLAY_TIMEOUT_S = 10.0
# The serve health probe should be snappy — a missing daemon is the common case.
_HEALTH_TIMEOUT_S = 2.0

_KIND_TO_OTLP = {
    SpanKind.INTERNAL: 1,
    SpanKind.SERVER: 2,
    SpanKind.CLIENT: 3,
    SpanKind.PRODUCER: 4,
    SpanKind.CONSUMER: 5,
}
_STATUS_TO_OTLP = {
    SpanStatus.UNSET: 0,
    SpanStatus.OK: 1,
    SpanStatus.ERROR: 2,
}


class LiveReplayError(RuntimeError):
    """Raised when a demo scenario cannot be replayed into a live tj serve."""


@dataclass
class LiveResult:
    """Outcome of flushing a scenario's spans to a live tj serve."""
    endpoint: str
    sent: int
    ingested: int
    rejected: int
    rejections: list[dict] = field(default_factory=list)


def _to_otlp_value(v: object) -> dict:
    """Convert a Python value to an OTLP AttributeValue dict (mirrors the SDK)."""
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_to_otlp_value(item) for item in v]}}
    if isinstance(v, dict):
        return {"kvlistValue": {"values": [
            {"key": str(k), "value": _to_otlp_value(val)} for k, val in v.items()
        ]}}
    return {"stringValue": str(v)}


def _to_nanos(dt: datetime | None) -> str:
    """OTLP timestamps are unix-nanoseconds as decimal strings."""
    if dt is None:
        return "0"
    return str(int(dt.timestamp() * 1e9))


def _span_to_otlp(span: NormalizedSpan) -> dict:
    """Serialize a NormalizedSpan to an OTLP JSON span dict.

    Inverse of ``otlp_parsing.parse_otlp_span``. Raw ``span.attributes`` pass
    through first (so e.g. ``gen_ai.tool.input`` survives), then the promoted
    indexed fields are written onto their semconv keys — explicit fields win on
    key conflict, matching how the parser prefers the dedicated columns.
    """
    attr_map: dict[str, Any] = {}
    for key, value in (span.attributes or {}).items():
        if value is not None:
            attr_map[key] = value

    def put(key: str, value: object) -> None:
        if value is not None:
            attr_map[key] = value

    put(GenAIAttributes.AGENT_ID, span.agent_id)
    put(GenAIAttributes.PROVIDER_NAME, span.provider)
    put(GenAIAttributes.REQUEST_MODEL, span.model)
    put(GenAIAttributes.TOOL_NAME, span.tool_name)
    put(GenAIAttributes.INPUT_TOKENS, span.input_tokens)
    put(GenAIAttributes.OUTPUT_TOKENS, span.output_tokens)
    put(GenAIAttributes.CACHE_READ_TOKENS, span.cache_tokens)
    put(GenAIAttributes.CACHE_CREATE_TOKENS, span.cache_write_tokens)
    put(GenAIAttributes.REQUEST_TYPE, span.request_type)
    put(GenAIAttributes.CONVERSATION_ID, span.conversation_id)
    put(TjAttributes.SESSION_ID, span.session_id)  # "session.id"
    put(TjAttributes.BILLING_ACCOUNT, span.billing_account)
    put(ResourceAttributes.SERVICE_NAMESPACE, span.service_namespace)
    put(ResourceAttributes.SERVICE_INSTANCE_ID, span.service_instance_id)
    put(TjAttributes.RUN_ID, span.run_id)
    put(TjAttributes.PARENT_SESSION_ID, span.parent_session_id)

    otlp: dict[str, Any] = {
        "traceId": span.trace_id or "",
        "spanId": span.span_id or new_span_id(),
        "name": span.name,
        "kind": _KIND_TO_OTLP.get(span.kind, 1),
        "startTimeUnixNano": _to_nanos(span.start_time),
        "endTimeUnixNano": _to_nanos(span.end_time),
        "attributes": [
            {"key": k, "value": _to_otlp_value(v)} for k, v in attr_map.items()
        ],
        "status": {"code": _STATUS_TO_OTLP.get(span.status_code, 0)},
    }
    if span.status_message:
        otlp["status"]["message"] = span.status_message
    if span.parent_span_id:
        otlp["parentSpanId"] = span.parent_span_id
    return otlp


def _build_payload(otlp_spans: list[dict]) -> dict:
    """Wrap OTLP span dicts in the resourceSpans envelope /api/v1/spans expects."""
    return {
        "resourceSpans": [{
            "resource": {"attributes": [
                {"key": "service.name", "value": {"stringValue": "tokenjam"}},
            ]},
            "scopeSpans": [{"spans": otlp_spans}],
        }],
    }


class LiveSink:
    """Buffers scenario spans and POSTs them to a running tj serve in one batch."""

    def __init__(self, endpoint: str, ingest_secret: str | None) -> None:
        self.endpoint = endpoint
        self._headers = {"Content-Type": "application/json"}
        # Only attach the Bearer header when a secret is actually configured —
        # sending "Authorization: Bearer " (empty/whitespace secret) is an
        # illegal header value that httpx rejects before the request goes out
        # (same guard as TjHttpExporter, #431). Strip once and reuse the
        # stripped value so a whitespace-padded config value doesn't leak
        # into the header and get rejected by the server as a mismatch.
        stripped_secret = (ingest_secret or "").strip()
        if stripped_secret:
            self._headers["Authorization"] = f"Bearer {stripped_secret}"
        self._buffer: list[dict] = []

    def record(self, span: NormalizedSpan) -> None:
        """Snapshot a span to OTLP JSON and buffer it for the next flush."""
        self._buffer.append(_span_to_otlp(span))

    @property
    def pending(self) -> int:
        return len(self._buffer)

    def flush(self) -> LiveResult:
        """POST all buffered spans to tj serve, clear the buffer, return the result.

        Raises LiveReplayError on transport failure, auth rejection (401), or any
        non-2xx status — the caller surfaces it and exits non-zero.
        """
        if not self._buffer:
            return LiveResult(self.endpoint, sent=0, ingested=0, rejected=0)

        payload = _build_payload(self._buffer)
        sent = len(self._buffer)
        self._buffer = []
        try:
            resp = httpx.post(
                self.endpoint, json=payload, headers=self._headers,
                timeout=_REPLAY_TIMEOUT_S,
            )
        except httpx.RequestError as exc:
            raise LiveReplayError(
                f"could not reach tj serve at {self.endpoint}: {exc}"
            ) from exc

        if resp.status_code == 401:
            raise LiveReplayError(
                "tj serve rejected the replay with 401 (ingest secret mismatch). "
                "Ensure the CLI and the running daemon share the same "
                ".tj/config.toml, or restart the daemon after rotating the secret."
            )
        if resp.status_code >= 300:
            raise LiveReplayError(
                f"tj serve returned {resp.status_code} on span replay: "
                f"{resp.text[:200]}"
            )

        body = resp.json() if resp.content else {}
        return LiveResult(
            endpoint=self.endpoint,
            sent=sent,
            ingested=int(body.get("ingested", 0)),
            rejected=int(body.get("rejected", 0)),
            rejections=list(body.get("rejections", [])),
        )


def serve_base_url(config: "TjConfig") -> str:
    return f"http://{config.api.host}:{config.api.port}"


def spans_endpoint(config: "TjConfig") -> str:
    return f"{serve_base_url(config)}/api/v1/spans"


def check_serve_alive(config: "TjConfig") -> bool:
    """True if a tj serve daemon answers the status probe (200 or 401)."""
    try:
        resp = httpx.get(
            f"{serve_base_url(config)}/api/v1/status", timeout=_HEALTH_TIMEOUT_S
        )
    except httpx.RequestError:
        return False
    return resp.status_code in (200, 401)


def build_sink(config: "TjConfig") -> LiveSink:
    return LiveSink(spans_endpoint(config), config.security.ingest_secret)


def sink_from_context() -> "LiveSink | None":
    """Return the LiveSink stashed on the current click context, if any.

    ``tj demo --live`` builds the sink and stores it on ``ctx.obj`` so scenarios
    — which construct their own ``DemoEnvironment`` with no wiring — pick it up
    transparently. Outside a CLI invocation (unit tests, library use) there is no
    context and this returns None, preserving the throwaway-backend behaviour.
    """
    import click

    ctx = click.get_current_context(silent=True)
    if ctx is None or ctx.obj is None:
        return None
    return ctx.obj.get("demo_live_sink")
