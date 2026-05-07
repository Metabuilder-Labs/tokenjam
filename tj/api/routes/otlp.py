"""Standard OTLP/HTTP route aliases.

POST /v1/traces — forwards to the same OTLP JSON ingest logic as /api/v1/spans.
POST /v1/metrics — stub (200 OK, silently discards).
POST /v1/logs — primary ingest path for Claude Code telemetry; converts OTLP log
    events to NormalizedSpan objects via parse_log_records() in logs.py.

These exist so that OTel exporters configured with a bare endpoint
(e.g. ``http://127.0.0.1:7391``) work out of the box — OpenClaw's
diagnostics-otel plugin uses this convention.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from tj.api.routes.spans import ingest_spans

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/traces")
async def otlp_traces(request: Request) -> JSONResponse:
    """Accept OTLP JSON traces — same handler as /api/v1/spans."""
    return await ingest_spans(request)


@router.post("/v1/metrics")
async def otlp_metrics(request: Request) -> JSONResponse:
    """Stub — accept and discard OTLP metrics to avoid noisy client warnings."""
    return JSONResponse(status_code=200, content={"status": "ok"})


@router.post("/v1/logs")
async def otlp_logs(request: Request) -> JSONResponse:
    """Accept OTLP JSON logs — primary ingest path for Claude Code telemetry."""
    from tj.api.routes.logs import parse_log_records

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    if not isinstance(body, dict) or "resourceLogs" not in body:
        # Non-log OTLP signals (resourceSpans, resourceMetrics) routed here
        # when an SDK uses this endpoint as its base — silently ignore.
        return JSONResponse(status_code=200, content={"ingested": 0, "rejected": 0, "rejections": []})

    pipeline = request.app.state.pipeline
    ingested, rejections = parse_log_records(body, pipeline)

    return JSONResponse(
        status_code=200,
        content={
            "ingested": ingested,
            "rejected": len(rejections),
            "rejections": rejections,
        },
    )
