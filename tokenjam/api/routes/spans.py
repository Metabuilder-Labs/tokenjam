"""POST /api/v1/spans — OTLP JSON span ingest endpoint."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from tokenjam.core.ingest import SpanRejectedError
from tokenjam.core.models import NormalizedSpan, SpanKind, SpanStatus
from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tokenjam.utils.ids import new_span_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/spans")
async def ingest_spans(request: Request) -> JSONResponse:
    """
    Accept a batch of spans in OTLP JSON format.
    Auth is enforced by IngestAuthMiddleware.
    Returns 200 even on partial rejection; 400 only if body is entirely malformed.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body"},
        )

    if not isinstance(body, dict) or "resourceSpans" not in body:
        return JSONResponse(
            status_code=400,
            content={"error": "Expected OTLP JSON with 'resourceSpans' key"},
        )

    pipeline = request.app.state.pipeline
    ingested = 0
    rejections: list[dict[str, str]] = []

    for resource_span in body.get("resourceSpans", []):
        resource_attrs = _extract_resource_attrs(resource_span)
        for scope_span in resource_span.get("scopeSpans", []):
            for raw_span in scope_span.get("spans", []):
                span_id = raw_span.get("spanId", new_span_id())
                try:
                    span = _parse_span(raw_span, resource_attrs)
                    pipeline.process(span)
                    ingested += 1
                except SpanRejectedError as exc:
                    rejections.append({"span_id": span_id, "reason": str(exc)})
                except Exception as exc:
                    logger.warning("Failed to process span %s: %s", span_id, exc)
                    rejections.append({"span_id": span_id, "reason": str(exc)})

    return JSONResponse(
        status_code=200,
        content={
            "ingested": ingested,
            "rejected": len(rejections),
            "rejections": rejections,
        },
    )


def _extract_resource_attrs(resource_span: dict) -> dict[str, Any]:
    """Pull flat attributes from the resource section."""
    resource = resource_span.get("resource", {})
    attrs: dict[str, Any] = {}
    for attr in resource.get("attributes", []):
        key = attr.get("key", "")
        value = _otlp_value(attr.get("value", {}))
        if key and value is not None:
            attrs[key] = value
    return attrs


def _parse_span(raw: dict, resource_attrs: dict[str, Any]) -> NormalizedSpan:
    """Convert an OTLP JSON span dict into a NormalizedSpan."""
    # Merge resource + span attributes
    attrs: dict[str, Any] = dict(resource_attrs)
    for attr in raw.get("attributes", []):
        key = attr.get("key", "")
        value = _otlp_value(attr.get("value", {}))
        if key and value is not None:
            attrs[key] = value

    # Parse timestamps (OTLP uses nanoseconds as strings)
    start_ns = int(raw.get("startTimeUnixNano", 0))
    end_ns = int(raw.get("endTimeUnixNano", 0))
    start_time = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc) if start_ns else datetime.now(tz=timezone.utc)
    end_time = datetime.fromtimestamp(end_ns / 1e9, tz=timezone.utc) if end_ns else None

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

    # --- OpenClaw / generic OTLP attribute enrichment ---
    span_name = raw.get("name", "")
    agent_id = attrs.get(GenAIAttributes.AGENT_ID)
    tool_name = attrs.get(GenAIAttributes.TOOL_NAME)
    provider = attrs.get(GenAIAttributes.PROVIDER_NAME)

    # Fall back to service.name for agent_id (OpenClaw sets service.name)
    if not agent_id:
        agent_id = attrs.get("service.name") or None

    # Fall back to gen_ai.system for provider (OpenClaw uses this)
    if not provider:
        provider = attrs.get("gen_ai.system") or None

    # Extract tool_name from span names like "tool.Read", "tool.exec"
    if not tool_name and span_name.startswith("tool."):
        tool_name = span_name[5:]  # strip "tool." prefix

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
            {"name": e.get("name", ""), "time": e.get("timeUnixNano"),
             "attributes": {a.get("key", ""): _otlp_value(a.get("value", {})) for a in e.get("attributes", [])}}
            for e in raw.get("events", [])
        ],
        # Extract indexed fields from merged attributes
        agent_id=agent_id,
        provider=provider,
        model=attrs.get(GenAIAttributes.REQUEST_MODEL),
        tool_name=tool_name,
        input_tokens=_safe_int(attrs.get(GenAIAttributes.INPUT_TOKENS)),
        output_tokens=_safe_int(attrs.get(GenAIAttributes.OUTPUT_TOKENS)),
        cache_tokens=_safe_int(attrs.get(GenAIAttributes.CACHE_READ_TOKENS)),
        cost_usd=_safe_float(attrs.get(TjAttributes.COST_USD)),
        request_type=attrs.get(GenAIAttributes.REQUEST_TYPE),
        conversation_id=attrs.get(GenAIAttributes.CONVERSATION_ID),
        session_id=attrs.get("session.id"),
    )


def _otlp_value(v: dict) -> Any:
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
        return [_otlp_value(item) for item in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {
            kv["key"]: _otlp_value(kv["value"])
            for kv in v["kvlistValue"].get("values", [])
        }
    return None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
