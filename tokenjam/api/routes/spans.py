"""POST /api/v1/spans — OTLP JSON span ingest endpoint."""
from __future__ import annotations

import gzip
import json
import logging
from typing import Any

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


async def _read_otlp_body(request: Request) -> dict[str, Any]:
    """Return the parsed JSON body, decompressing gzip if needed.

    Decompresses when Content-Encoding: gzip is set, or when the body starts
    with the gzip magic bytes (\\x1f\\x8b) as a fallback for exporters that
    compress without setting the header. Raises ValueError on any failure.
    """
    raw = await request.body()

    # 1. Honour explicit Content-Encoding header
    if request.headers.get("content-encoding", "").lower() == "gzip":
        try:
            raw = gzip.decompress(raw)
        except Exception as exc:
            # gzip.decompress can raise OSError, EOFError, or zlib.error
            raise ValueError(f"gzip decompression failed: {exc}") from exc
    # 2. Sniff fallback — gzip magic bytes present but no Content-Encoding header
    elif raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except Exception as exc:
            # gzip.decompress can raise OSError, EOFError, or zlib.error
            raise ValueError(
                f"body appears gzip-compressed but decompression failed: {exc}"
            ) from exc

    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except Exception as exc:
        raise ValueError(f"JSON decode failed: {exc}") from exc


@router.post("/spans")
async def ingest_spans(request: Request) -> JSONResponse:
    """
    Accept a batch of spans in OTLP JSON format.
    Auth is enforced by IngestAuthMiddleware.
    Returns 200 even on partial rejection; 400 only if the body is entirely unparseable.
    """
    try:
        body = await _read_otlp_body(request)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": f"could not parse OTLP body: {exc}"},
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
