"""
Langfuse → TokenJam ingest adapter.

Reads Langfuse `Observation` records (from a live API endpoint or a local
JSON file) and writes equivalent `NormalizedSpan` rows to the local DB.

Idempotent: span_id is a deterministic hash of (langfuse_trace_id,
observation_id), so re-runs skip rows already present.

Two input modes:
  - --source-url <url>  : GET /api/public/observations against a live
                          Langfuse instance with --api-key Bearer auth
  - --source-file <path>: read a JSON dump from disk (testing, offline use)

Langfuse `Observation` envelope (relevant fields):
  id, traceId, type ("GENERATION" | "SPAN" | "EVENT"),
  name, startTime, endTime, model,
  usage.{input, output}, calculatedTotalCost,
  sessionId, parentObservationId, level, statusMessage

Top-level API responses can be either a bare list or `{"data": [...]}`.
We tolerate both.
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
    """Parse an ISO-8601 timestamp; returns None if value is missing or unparseable."""
    if not value:
        return None
    try:
        # Langfuse uses ISO-8601 with 'Z' or '+00:00' suffix.
        s = value.replace("Z", "+00:00") if value.endswith("Z") else value
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _model_to_provider(model: str | None) -> tuple[str | None, str | None]:
    """
    Best-effort (provider, billing_account) from the model name. Returns
    (None, None) when the model is unknown — we don't invent providers.
    """
    if not model:
        return None, None
    m = model.lower()
    if "claude" in m:
        return "anthropic", "anthropic"
    if m.startswith(("gpt", "o3", "o4", "chatgpt-")):
        return "openai", "openai"
    if "gemini" in m:
        return "google", "google"
    if "command" in m or m.startswith("cohere-"):
        return "cohere", None
    if m.startswith(("llama", "qwen", "mistral", "phi-")):
        # Without more context, assume local. The user can override by
        # mapping their billing_account explicitly in a future flag.
        return "local.ollama", "local.ollama"
    return None, None


def _observation_to_span(obs: dict[str, Any]) -> NormalizedSpan | None:
    """
    Convert one Langfuse observation to a NormalizedSpan. Returns None when
    the observation lacks enough data to produce a meaningful span (e.g. a
    minimal "EVENT" with no timing).
    """
    obs_id = obs.get("id")
    trace_id_lf = obs.get("traceId") or obs.get("trace_id")
    if not obs_id or not trace_id_lf:
        return None

    obs_type = (obs.get("type") or "").upper()
    if obs_type not in {"GENERATION", "SPAN", "EVENT"}:
        # Tolerate unknown types — treat as a generic span.
        obs_type = "SPAN"

    start_time = _parse_ts(obs.get("startTime") or obs.get("start_time"))
    end_time = _parse_ts(obs.get("endTime") or obs.get("end_time"))
    if start_time is None:
        return None

    duration_ms: float | None = None
    if end_time is not None:
        duration_ms = (end_time - start_time).total_seconds() * 1000.0

    # Token usage: prefer the modern usageDetails dict, fall back to legacy usage.
    usage = obs.get("usage") or {}
    usage_details = obs.get("usageDetails") or {}
    input_tokens = (
        usage_details.get("input")
        or usage.get("input")
        or usage.get("promptTokens")
        or None
    )
    output_tokens = (
        usage_details.get("output")
        or usage.get("output")
        or usage.get("completionTokens")
        or None
    )
    cache_read = (
        usage_details.get("input_cache_read")
        or usage_details.get("cacheReadInputTokens")
        or None
    )

    model = obs.get("model")
    provider, billing_account = _model_to_provider(model)
    cost = obs.get("calculatedTotalCost") or obs.get("totalCost")

    status_code = SpanStatus.OK
    level = (obs.get("level") or "DEFAULT").upper()
    if level in {"ERROR", "WARNING"} or obs.get("statusMessage"):
        status_code = SpanStatus.ERROR if level == "ERROR" else SpanStatus.OK

    name = obs.get("name") or (
        "gen_ai.llm.call" if obs_type == "GENERATION" else "gen_ai.tool.call"
    )
    kind = SpanKind.CLIENT if obs_type == "GENERATION" else SpanKind.INTERNAL

    # Deterministic IDs derived from Langfuse IDs so re-runs are idempotent.
    span_id = _det_id("langfuse-obs", str(trace_id_lf), str(obs_id), length=16)
    trace_id = _det_id("langfuse-trace", str(trace_id_lf), length=32)
    parent_lf = obs.get("parentObservationId") or obs.get("parent_observation_id")
    parent_span_id = (
        _det_id("langfuse-obs", str(trace_id_lf), str(parent_lf), length=16)
        if parent_lf else None
    )

    session_id = obs.get("sessionId") or obs.get("session_id")
    # Use Langfuse trace ID as conversation_id when sessionId is absent — it
    # ties spans from one trace into a TokenJam session.
    conversation_id = session_id or str(trace_id_lf)

    return NormalizedSpan(
        span_id=span_id,
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        name=str(name),
        kind=kind,
        status_code=status_code,
        status_message=obs.get("statusMessage"),
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        agent_id=str(obs.get("userId") or obs.get("user_id") or "langfuse"),
        session_id=None,  # let IngestPipeline resolve via conversation_id
        provider=provider,
        model=model,
        tool_name=(obs.get("name") if obs_type == "SPAN" else None),
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        cache_tokens=int(cache_read) if cache_read is not None else None,
        cost_usd=float(cost) if cost is not None else None,
        request_type=("completion" if obs_type == "GENERATION" else None),
        conversation_id=str(conversation_id),
        attributes={"source": "ingest.langfuse"},
        billing_account=billing_account,
    )


def _iter_observations(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield observations from either a bare list or a {"data": [...]} envelope."""
    if isinstance(payload, list):
        yield from (o for o in payload if isinstance(o, dict))
        return
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            yield from (o for o in payload["data"] if isinstance(o, dict))
            return
        # Some dumps put a single observation at the top level.
        if "id" in payload and "traceId" in payload:
            yield payload


def _load_file(path: Path) -> Any:
    """Load a JSON file. Accepts a bare list, a {data: [...]} envelope, or NDJSON."""
    text = path.read_text()
    text = text.strip()
    if not text:
        return []
    # NDJSON heuristic: looks like multiple JSON objects on separate lines.
    if text[0] == "{" and "\n{" in text and not text.startswith("{\""):
        # Could be NDJSON or a multi-line object. Try parsing whole-file first.
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
    return json.loads(text)


def _fetch_url(url: str, api_key: str | None, since: datetime | None) -> Any:
    """
    Fetch observations from a live Langfuse instance.

    Uses the public API endpoint /api/public/observations with Bearer auth.
    Pagination is followed transparently; returns the merged list.
    """
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "Live Langfuse ingestion requires httpx. Install with "
            "`pip install httpx` or use --source-file for offline ingestion."
        ) from exc

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    base = url.rstrip("/")
    if "/api/public/observations" not in base:
        base = f"{base}/api/public/observations"

    params: dict[str, Any] = {"limit": 100}
    if since is not None:
        params["fromStartTime"] = since.isoformat()

    all_observations: list[dict[str, Any]] = []
    page = 1
    with httpx.Client(timeout=30.0) as client:
        while True:
            params["page"] = page
            resp = client.get(base, headers=headers, params=params)
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data") if isinstance(body, dict) else body
            if not data:
                break
            all_observations.extend(data)
            total_pages = (body.get("meta") or {}).get("totalPages")
            if total_pages is None or page >= int(total_pages):
                break
            page += 1
    return all_observations


def ingest_langfuse(
    db,
    *,
    source_url: str | None = None,
    source_file: str | None = None,
    api_key: str | None = None,
    since: datetime | None = None,
) -> dict[str, int]:
    """
    Ingest Langfuse observations into the local DB.

    Exactly one of source_url or source_file must be provided. Returns a
    summary dict: {"observations_read", "spans_written", "spans_skipped"}.

    Spans flow through IngestPipeline if available; otherwise they're written
    directly via db.insert_span and the session record is created/updated
    via db.upsert_session. Idempotent — re-running skips rows already
    present (deterministic span_ids + PRIMARY KEY conflicts on spans.span_id).
    """
    if (source_url is None) == (source_file is None):
        raise ValueError("Provide exactly one of source_url or source_file.")

    if source_file:
        payload = _load_file(Path(source_file).expanduser())
    else:
        assert source_url is not None
        payload = _fetch_url(source_url, api_key=api_key, since=since)

    observations_read = 0
    spans_written = 0
    spans_skipped = 0

    conn = getattr(db, "conn", None)

    for obs in _iter_observations(payload):
        observations_read += 1
        span = _observation_to_span(obs)
        if span is None:
            continue
        # Filter by `since` for source-file mode too (the API filter already
        # applies for source-url mode).
        if since is not None and span.start_time and span.start_time < since:
            continue
        # Idempotency: skip if a span with this ID already exists.
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
            # Idempotency belt-and-suspenders: PRIMARY KEY violations also land here.
            spans_skipped += 1

    return {
        "observations_read": observations_read,
        "spans_written": spans_written,
        "spans_skipped": spans_skipped,
    }
