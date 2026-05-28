"""
Raw OTLP → TokenJam ingest adapter.

Generic OTLP JSON ingestion. The "works with anything" adapter that
covers sources Langfuse / Helicone don't handle: OTel SDKs writing
JSON files, observability tools emitting OTLP-shaped exports, OTLP
HTTP collectors with a /v1/traces endpoint.

Two input modes:
  - --source-url <url>  : POSTs to a live OTLP HTTP endpoint or GETs
                          a JSON dump from a URL. (For live OTLP
                          ingestion this is uncommon — most OTLP servers
                          expect POST-based push. The URL mode is here
                          mainly for fetching JSON dumps hosted at HTTPS.)
  - --source-file <path>: read an OTLP JSON file from disk.

Idempotent: span_id comes from the OTLP payload itself (each span has a
unique trace_id+span_id). The DB's PRIMARY KEY on spans.span_id ensures
re-runs skip rows already present.

Reuses tokenjam.otel.otlp_parsing — the same parser used by the live
`POST /api/v1/spans` endpoint. One implementation, two callers.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from tokenjam.otel.otlp_parsing import iter_otlp_spans


def _load_file(path: Path) -> Any:
    """Load JSON file. Accepts a top-level OTLP envelope or NDJSON."""
    text = path.read_text().strip()
    if not text:
        return {}
    # Single-document JSON (the common case for OTLP file dumps).
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # NDJSON: one OTLP envelope per line. Merge into a single envelope.
    merged: dict[str, list] = {"resourceSpans": []}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if isinstance(record, dict) and "resourceSpans" in record:
            merged["resourceSpans"].extend(record["resourceSpans"])
    return merged


def _fetch_url(url: str) -> Any:
    """Fetch an OTLP JSON dump via HTTP GET. Live push endpoints don't use this."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "Live OTLP URL ingestion requires httpx. Install with "
            "`pip install httpx` or use --source-file for offline ingestion."
        ) from exc

    with httpx.Client(timeout=60.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()


def ingest_otlp(
    db,
    *,
    source_url: str | None = None,
    source_file: str | None = None,
    since: datetime | None = None,
) -> dict[str, int]:
    """
    Ingest spans from an OTLP JSON dump into the local DB.

    Exactly one of source_url or source_file must be provided. Returns a
    summary dict: {"spans_seen", "spans_written", "spans_skipped",
    "spans_rejected"}.
    """
    if (source_url is None) == (source_file is None):
        raise ValueError("Provide exactly one of source_url or source_file.")

    if source_file:
        payload = _load_file(Path(source_file).expanduser())
    else:
        assert source_url is not None
        payload = _fetch_url(source_url)

    spans_seen = 0
    spans_written = 0
    spans_skipped = 0
    spans_rejected = 0

    conn = getattr(db, "conn", None)

    for raw_span, resource_attrs in iter_otlp_spans(payload):
        spans_seen += 1
        try:
            span = parse_otlp_span_safe(raw_span, resource_attrs)
        except Exception:
            spans_rejected += 1
            continue
        if span is None:
            spans_rejected += 1
            continue
        if since is not None and span.start_time and span.start_time < since:
            continue
        if conn is not None:
            existing = conn.execute(
                "SELECT 1 FROM spans WHERE span_id = $1", [span.span_id]
            ).fetchone()
            if existing:
                spans_skipped += 1
                continue
        try:
            db.insert_span(span)
            spans_written += 1
        except Exception:
            spans_skipped += 1

    return {
        "spans_seen": spans_seen,
        "spans_written": spans_written,
        "spans_skipped": spans_skipped,
        "spans_rejected": spans_rejected,
    }


def parse_otlp_span_safe(raw_span, resource_attrs):
    """
    Wrap parse_otlp_span with a defensive guard that returns None for
    spans that lack the minimum fields (no spanId, no traceId).
    """
    from tokenjam.otel.otlp_parsing import parse_otlp_span
    if not raw_span.get("spanId") or not raw_span.get("traceId"):
        return None
    return parse_otlp_span(raw_span, resource_attrs)
