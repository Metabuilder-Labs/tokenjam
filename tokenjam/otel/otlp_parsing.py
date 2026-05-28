"""
Shared OTLP JSON → NormalizedSpan parsing.

Two callers:
  - `tokenjam/api/routes/spans.py` — live `POST /api/v1/spans` endpoint
  - `tokenjam/core/ingest_adapters/otlp.py` — `tj backfill otlp` adapter

Both need the same record-shape mapping. Keeping one implementation here
avoids drift between the live receive path and the backfill adapter.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tokenjam.core.models import NormalizedSpan, SpanKind, SpanStatus
from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tokenjam.utils.ids import new_span_id


def otlp_value(v: dict) -> Any:
    """Extract a value from an OTLP AttributeValue wrapper."""
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return v["boolValue"]
    if "arrayValue" in v:
        return [otlp_value(item) for item in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {
            kv["key"]: otlp_value(kv["value"])
            for kv in v["kvlistValue"].get("values", [])
        }
    return None


def safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def provider_to_billing_account(provider: str | None) -> str | None:
    """Map gen_ai.provider.name (or gen_ai.system) to a tj billing_account."""
    if not provider:
        return None
    p = str(provider).lower()
    if p in {"anthropic", "openai", "google", "bedrock", "local.ollama"}:
        return p
    if p in {"aws.bedrock", "aws_bedrock"}:
        return "bedrock"
    if p in {"google.gemini", "gemini", "vertex", "vertex_ai"}:
        return "google"
    if p in {"ollama"}:
        return "local.ollama"
    if p in {"azure", "azure.openai", "azure_openai"}:
        return "openai"
    return None


def extract_resource_attrs(resource_span: dict) -> dict[str, Any]:
    """Pull flat attributes from the resource section."""
    resource = resource_span.get("resource", {})
    attrs: dict[str, Any] = {}
    for attr in resource.get("attributes", []):
        key = attr.get("key", "")
        value = otlp_value(attr.get("value", {}))
        if key and value is not None:
            attrs[key] = value
    return attrs


def parse_otlp_span(raw: dict, resource_attrs: dict[str, Any]) -> NormalizedSpan:
    """Convert an OTLP JSON span dict into a NormalizedSpan."""
    # Merge resource + span attributes (span attrs win on key conflict)
    attrs: dict[str, Any] = dict(resource_attrs)
    for attr in raw.get("attributes", []):
        key = attr.get("key", "")
        value = otlp_value(attr.get("value", {}))
        if key and value is not None:
            attrs[key] = value

    # Parse timestamps (OTLP uses nanoseconds as strings)
    start_ns = int(raw.get("startTimeUnixNano", 0))
    end_ns = int(raw.get("endTimeUnixNano", 0))
    start_time = (
        datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
        if start_ns else datetime.now(tz=timezone.utc)
    )
    end_time = (
        datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc)
        if end_ns else None
    )

    duration_ms = None
    if start_ns and end_ns:
        duration_ms = (end_ns - start_ns) / 1e6

    # Parse status
    status_raw = raw.get("status", {})
    status_code_int = status_raw.get("code", 0)
    status_map = {0: SpanStatus.UNSET, 1: SpanStatus.OK, 2: SpanStatus.ERROR}
    status_code = status_map.get(status_code_int, SpanStatus.UNSET)

    # Parse kind
    kind_int = raw.get("kind", 1)
    kind_map = {
        1: SpanKind.INTERNAL, 2: SpanKind.SERVER, 3: SpanKind.CLIENT,
        4: SpanKind.PRODUCER, 5: SpanKind.CONSUMER,
    }
    kind = kind_map.get(kind_int, SpanKind.INTERNAL)

    # Indexed-field extraction
    span_name = raw.get("name", "")
    agent_id = attrs.get(GenAIAttributes.AGENT_ID)
    tool_name = attrs.get(GenAIAttributes.TOOL_NAME)
    provider = attrs.get(GenAIAttributes.PROVIDER_NAME)

    # Fall back to service.name for agent_id (OpenClaw sets service.name).
    if not agent_id:
        agent_id = attrs.get("service.name") or None

    # Fall back to gen_ai.system for provider (OpenClaw uses this).
    if not provider:
        provider = attrs.get("gen_ai.system") or None

    # Extract tool_name from span names like "tool.Read", "tool.exec".
    if not tool_name and span_name.startswith("tool."):
        tool_name = span_name[5:]

    return NormalizedSpan(
        span_id=raw.get("spanId", new_span_id()),
        trace_id=raw.get("traceId", ""),
        name=span_name,
        kind=kind,
        status_code=status_code,
        status_message=status_raw.get("message"),
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        parent_span_id=raw.get("parentSpanId"),
        attributes=attrs,
        events=[
            {
                "name": e.get("name", ""),
                "time": e.get("timeUnixNano"),
                "attributes": {
                    a.get("key", ""): otlp_value(a.get("value", {}))
                    for a in e.get("attributes", [])
                },
            }
            for e in raw.get("events", [])
        ],
        agent_id=agent_id,
        provider=provider,
        model=attrs.get(GenAIAttributes.REQUEST_MODEL),
        tool_name=tool_name,
        input_tokens=safe_int(attrs.get(GenAIAttributes.INPUT_TOKENS)),
        output_tokens=safe_int(attrs.get(GenAIAttributes.OUTPUT_TOKENS)),
        cache_tokens=safe_int(attrs.get(GenAIAttributes.CACHE_READ_TOKENS)),
        cost_usd=safe_float(attrs.get(TjAttributes.COST_USD)),
        request_type=attrs.get(GenAIAttributes.REQUEST_TYPE),
        conversation_id=attrs.get(GenAIAttributes.CONVERSATION_ID),
        session_id=attrs.get("session.id"),
        billing_account=(
            attrs.get(TjAttributes.BILLING_ACCOUNT)
            or provider_to_billing_account(provider)
        ),
    )


def iter_otlp_spans(payload: dict) -> list[tuple[dict, dict]]:
    """
    Yield (raw_span_dict, resource_attrs) for every span in an OTLP JSON
    payload. Returns an empty list if the payload doesn't carry
    resourceSpans (lets callers reuse the iterator without crashing on
    unexpected shapes).
    """
    if not isinstance(payload, dict):
        return []
    out: list[tuple[dict, dict]] = []
    for resource_span in payload.get("resourceSpans", []) or []:
        resource_attrs = extract_resource_attrs(resource_span)
        for scope_span in resource_span.get("scopeSpans", []) or []:
            for raw_span in scope_span.get("spans", []) or []:
                out.append((raw_span, resource_attrs))
    return out
