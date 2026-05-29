"""
OTel SpanExporter that POSTs OTLP JSON to tj serve's /api/v1/spans endpoint.
Used when tj serve is running and holds the DuckDB lock.
"""
from __future__ import annotations

import logging
from typing import Sequence

import httpx
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan

logger = logging.getLogger("tokenjam.sdk")


class TjHttpExporter(SpanExporter):
    """Exports spans via HTTP POST to tj serve."""

    def __init__(self, endpoint: str, ingest_secret: str) -> None:
        self._endpoint = endpoint
        self._ingest_secret = ingest_secret
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ingest_secret}",
        }
        # Cumulative count of spans dropped on auth failure. Exposed so
        # `tj doctor` can surface this in a future enhancement (#68 §2).
        self.dropped_auth_failures = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        otlp_spans = [_span_to_otlp(s) for s in spans]
        payload = {
            "resourceSpans": [{
                "resource": {"attributes": [
                    {"key": "service.name", "value": {"stringValue": "tokenjam"}},
                ]},
                "scopeSpans": [{"spans": otlp_spans}],
            }],
        }
        try:
            resp = httpx.post(
                self._endpoint, json=payload, headers=self._headers, timeout=5.0,
            )
            if resp.status_code < 300:
                return SpanExportResult.SUCCESS
            if resp.status_code == 401:
                # 401 won't change on retry. Log loudly at ERROR level with
                # the secret fingerprint so the user can spot the mismatch
                # without grepping debug logs (#68 §2). The OTel
                # BatchSpanProcessor treats FAILURE as drop-after-retries
                # — by the time we get here the spans are effectively lost.
                self.dropped_auth_failures += len(spans)
                secret_fp = (
                    f"{self._ingest_secret[:8]}..." if self._ingest_secret
                    else "(no ingest_secret configured)"
                )
                logger.error(
                    "tj serve rejected span export with 401 — your SDK is "
                    "using ingest_secret=%s but the running daemon expects "
                    "a different secret. Spans are being DROPPED. Check "
                    "that .tj/config.toml and ~/.config/tj/config.toml "
                    "agree (or run `tj stop && tj serve &` after rotating "
                    "the secret). Dropped %d span(s) so far.",
                    secret_fp,
                    self.dropped_auth_failures,
                )
            else:
                logger.warning(
                    "tj serve returned %d on span export", resp.status_code,
                )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            logger.warning("Failed to export spans to tj serve: %s", exc)
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _span_to_otlp(span: ReadableSpan) -> dict:
    """Convert an OTel ReadableSpan to OTLP JSON dict."""
    ctx = span.context
    attrs = []
    for k, v in (span.attributes or {}).items():
        attrs.append({"key": k, "value": _to_otlp_value(v)})

    result: dict = {
        "traceId": format(ctx.trace_id, "032x") if ctx else "",
        "spanId": format(ctx.span_id, "016x") if ctx else "",
        "name": span.name,
        "kind": (span.kind.value + 1) if span.kind is not None else 1,
        "startTimeUnixNano": str(span.start_time or 0),
        "endTimeUnixNano": str(span.end_time or 0),
        "attributes": attrs,
        "status": {},
        "events": [],
    }

    if span.parent and span.parent.span_id:
        result["parentSpanId"] = format(span.parent.span_id, "016x")

    if span.status:
        code_map = {0: 0, 1: 1, 2: 2}  # UNSET, OK, ERROR
        result["status"] = {
            "code": code_map.get(span.status.status_code.value, 0),
        }
        if span.status.description:
            result["status"]["message"] = span.status.description

    for event in span.events or []:
        evt_attrs = [
            {"key": k, "value": _to_otlp_value(v)}
            for k, v in (event.attributes or {}).items()
        ]
        result["events"].append({
            "name": event.name,
            "timeUnixNano": str(event.timestamp or 0),
            "attributes": evt_attrs,
        })

    return result


def _to_otlp_value(v: object) -> dict:
    """Convert a Python value to an OTLP AttributeValue dict."""
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_to_otlp_value(item) for item in v]}}
    return {"stringValue": str(v)}
