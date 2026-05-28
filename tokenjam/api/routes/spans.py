"""POST /api/v1/spans — OTLP JSON span ingest endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from tokenjam.core.ingest import SpanRejectedError
from tokenjam.otel.otlp_parsing import (
    extract_resource_attrs,
    otlp_value as _otlp_value,    # noqa: F401  (re-exported for logs.py + tests)
    parse_otlp_span,
    safe_float as _safe_float,    # noqa: F401  (re-exported)
    safe_int as _safe_int,        # noqa: F401  (re-exported)
)
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
        resource_attrs = extract_resource_attrs(resource_span)
        for scope_span in resource_span.get("scopeSpans", []):
            for raw_span in scope_span.get("spans", []):
                span_id = raw_span.get("spanId", new_span_id())
                try:
                    span = parse_otlp_span(raw_span, resource_attrs)
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
