"""
Helicone → TokenJam ingest adapter.

Reads Helicone request records (from a live API endpoint or a local
JSON file) and writes equivalent NormalizedSpan rows to the local DB.

Idempotent: span_id is a deterministic hash of the Helicone request_id,
so re-runs skip rows already present.

Two input modes:
  - --source-url <url>  : POST /v1/request/query (or /v1/requests/query)
                          against a live Helicone instance with --api-key
                          Bearer auth.
  - --source-file <path>: read a JSON dump from disk (testing, offline use)

Helicone request envelope (relevant fields, normalised across v1/v2):
  request.id, request.user_id, request.model, request.provider,
  request.created_at, request.prompt_tokens (or
  prompt_tokens_with_response),
  response.id, response.completion_tokens, response.delay_ms,
  cost_usd / costUSD, session_id (Helicone-Property-Session header),
  properties (custom attribution tags).

Top-level API responses use {"data": [...]} on /v1/request/query.
The adapter also accepts bare lists and NDJSON for flexibility.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tokenjam.core.models import NormalizedSpan, SpanKind, SpanStatus


def _det_id(prefix: str, *parts: str, length: int = 16) -> str:
    """Deterministic hex ID derived from the named parts."""
    h = hashlib.sha256(":".join((prefix, *parts)).encode()).hexdigest()
    return h[:length]


def _parse_ts(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp; returns None when missing or unparseable."""
    if not value:
        return None
    try:
        s = value.replace("Z", "+00:00") if value.endswith("Z") else value
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _provider_to_billing_account(provider: str | None) -> tuple[str | None, str | None]:
    """
    Helicone reports provider as an uppercase enum (OPENAI, ANTHROPIC, ...).
    Map to (tj provider name, tj billing_account).
    """
    if not provider:
        return None, None
    p = str(provider).lower()
    if p in {"anthropic"}:
        return "anthropic", "anthropic"
    if p in {"openai", "azure", "azure_openai"}:
        return "openai", "openai"
    if p in {"google", "gemini", "vertex", "vertex_ai", "google_ai"}:
        return "google", "google"
    if p in {"bedrock", "aws_bedrock", "aws"}:
        return "bedrock", "bedrock"
    if p in {"ollama"}:
        return "local.ollama", "local.ollama"
    return None, None


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    """Best-effort cast to a dict — handle JSON strings."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _record_to_span(record: dict[str, Any]) -> NormalizedSpan | None:
    """
    Convert one Helicone request record to a NormalizedSpan.

    Returns None when the record lacks enough data to produce a meaningful
    span (missing request id, missing timestamps).
    """
    # Helicone's response shape evolves over versions. Tolerate both
    # nested (record["request"]["id"]) and flat (record["request_id"]).
    request = _coerce_dict(record.get("request")) or record
    response = _coerce_dict(record.get("response")) or {}

    req_id = request.get("id") or record.get("request_id") or record.get("id")
    if not req_id:
        return None

    start_time = _parse_ts(
        request.get("created_at")
        or record.get("request_created_at")
        or record.get("created_at")
    )
    if start_time is None:
        return None

    delay_ms = (
        response.get("delay_ms")
        or record.get("delay_ms")
        or record.get("latency")
        or record.get("response_delay_ms")
    )
    duration_ms = float(delay_ms) if delay_ms is not None else None
    end_time = (
        start_time + timedelta_from_ms(duration_ms)
        if duration_ms is not None else None
    )

    model = request.get("model") or record.get("model") or record.get("response_model")
    provider_raw = (
        request.get("provider")
        or record.get("provider")
        or record.get("request_provider")
    )
    provider, billing_account = _provider_to_billing_account(provider_raw)

    # Token usage. Helicone reports prompt + completion separately in newer
    # API versions; older versions report totals on the request.
    input_tokens = (
        request.get("prompt_tokens")
        or record.get("prompt_tokens")
        or record.get("total_prompt_tokens")
    )
    output_tokens = (
        response.get("completion_tokens")
        or record.get("completion_tokens")
        or record.get("total_completion_tokens")
    )
    cache_read = (
        response.get("cache_read_input_tokens")
        or record.get("cache_read_input_tokens")
    )
    # Anthropic-via-Helicone surfaces cache-creation tokens on the response
    # object alongside cache-read. Threading the count through so cache-write
    # cost reporting matches the live OTLP path (issue #93).
    cache_write = (
        response.get("cache_creation_input_tokens")
        or record.get("cache_creation_input_tokens")
    )

    cost = (
        record.get("cost_usd")
        or record.get("costUSD")
        or record.get("cost")
    )

    status_raw = (
        response.get("status")
        or record.get("status")
        or record.get("response_status")
    )
    status_code = SpanStatus.OK
    if status_raw and int(status_raw) >= 400:
        status_code = SpanStatus.ERROR
    error_msg = record.get("error") or response.get("error")

    user_id = (
        request.get("user_id")
        or record.get("user_id")
        or "helicone"
    )
    # Helicone's session attribution comes via Helicone-Property-Session.
    properties = _coerce_dict(
        record.get("properties") or request.get("properties")
    ) or {}
    session_label = (
        record.get("session_id")
        or properties.get("Helicone-Property-Session")
        or properties.get("session_id")
    )

    span_id = _det_id("helicone-req", str(req_id), length=16)
    trace_id = _det_id("helicone-trace", str(session_label or req_id), length=32)

    conversation_id = str(session_label or req_id)

    return NormalizedSpan(
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=None,
        name="gen_ai.llm.call",
        kind=SpanKind.CLIENT,
        status_code=status_code,
        status_message=str(error_msg) if error_msg else None,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        agent_id=str(user_id),
        session_id=None,  # let IngestPipeline resolve via conversation_id
        provider=provider,
        model=str(model) if model else None,
        tool_name=None,
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        cache_tokens=int(cache_read) if cache_read is not None else None,
        cache_write_tokens=int(cache_write) if cache_write is not None else None,
        cost_usd=float(cost) if cost is not None else None,
        request_type="completion",
        conversation_id=conversation_id,
        attributes={"source": "ingest.helicone"},
        billing_account=billing_account,
    )


def timedelta_from_ms(ms: float):
    from datetime import timedelta as _td
    return _td(milliseconds=ms)


def _iter_records(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield records from {data: [...]}, bare list, or NDJSON shapes."""
    if isinstance(payload, list):
        yield from (r for r in payload if isinstance(r, dict))
        return
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            yield from (r for r in payload["data"] if isinstance(r, dict))
            return
        # Some dumps put a single record at top level.
        if any(k in payload for k in ("request", "request_id", "id")):
            yield payload


def _load_file(path: Path) -> Any:
    """Load JSON file. Accepts {data: [...]}, bare list, or NDJSON."""
    text = path.read_text().strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    records: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _fetch_url(url: str, api_key: str | None, since: datetime | None) -> Any:
    """Fetch from a live Helicone instance via POST /v1/request/query."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "Live Helicone ingestion requires httpx. Install with "
            "`pip install httpx` or use --source-file for offline ingestion."
        ) from exc

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    base = url.rstrip("/")
    if "/v1/request/query" not in base and "/v1/requests/query" not in base:
        base = f"{base}/v1/request/query"

    filter_clause: dict[str, Any] = {}
    if since is not None:
        filter_clause = {
            "request": {
                "created_at": {"gte": since.isoformat()}
            }
        }

    all_records: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    with httpx.Client(timeout=30.0) as client:
        while True:
            body = {
                "filter": filter_clause or "all",
                "offset": offset,
                "limit": page_size,
                "sort": {"created_at": "desc"},
            }
            resp = client.post(base, headers=headers, json=body)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else payload
            if not data:
                break
            all_records.extend(data)
            if len(data) < page_size:
                break
            offset += page_size
    return all_records


def ingest_helicone(
    db,
    *,
    source_url: str | None = None,
    source_file: str | None = None,
    api_key: str | None = None,
    since: datetime | None = None,
) -> dict[str, int]:
    """
    Ingest Helicone records into the local DB.

    Exactly one of source_url or source_file must be provided. Returns a
    summary dict: {"records_read", "spans_written", "spans_skipped"}.
    """
    if (source_url is None) == (source_file is None):
        raise ValueError("Provide exactly one of source_url or source_file.")

    if source_file:
        payload = _load_file(Path(source_file).expanduser())
    else:
        assert source_url is not None
        payload = _fetch_url(source_url, api_key=api_key, since=since)

    records_read = 0
    spans_written = 0
    spans_skipped = 0

    conn = getattr(db, "conn", None)

    for record in _iter_records(payload):
        records_read += 1
        span = _record_to_span(record)
        if span is None:
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
        "records_read": records_read,
        "spans_written": spans_written,
        "spans_skipped": spans_skipped,
    }
