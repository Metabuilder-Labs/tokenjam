"""Log-to-span converter for Claude Code OTLP log events."""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from tj.core.ingest import IngestPipeline, SpanRejectedError
from tj.core.models import NormalizedSpan, SpanKind, SpanStatus
from tj.otel.semconv import ClaudeCodeEvents, CodexEvents, GenAIAttributes
from tj.utils.ids import new_span_id
from tj.api.routes.spans import _otlp_value, _safe_int

logger = logging.getLogger(__name__)


def _trace_id_from_session(session_id: str) -> str:
    """Deterministic 32-hex-char trace ID from session.id."""
    return hashlib.md5(session_id.encode()).hexdigest()


def _span_id_from_prompt(prompt_id: str) -> str:
    """Deterministic 16-hex-char span ID from prompt.id.
    Used as parent_span_id for tool/api spans within a turn,
    and as span_id for the user_prompt span itself."""
    return hashlib.md5(prompt_id.encode()).hexdigest()[:16]


def _parse_attrs(raw_attrs: list[dict]) -> dict[str, Any]:
    """Convert OTLP attribute list to a flat dict."""
    attrs: dict[str, Any] = {}
    for attr in raw_attrs:
        key = attr.get("key", "")
        value = _otlp_value(attr.get("value", {}))
        if key and value is not None:
            attrs[key] = value
    return attrs


def _ts_to_datetime(timestamp_ns: int) -> datetime:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc)


def _api_request_to_span(
    attrs: dict[str, Any],
    resource_attrs: dict[str, Any],
    timestamp_ns: int,
) -> NormalizedSpan:
    session_id = str(attrs[ClaudeCodeEvents.SESSION_ID])
    prompt_id = attrs.get(ClaudeCodeEvents.PROMPT_ID)
    duration_ms = float(attrs[ClaudeCodeEvents.DURATION_MS])
    start_time = _ts_to_datetime(timestamp_ns)
    end_time = start_time + timedelta(milliseconds=duration_ms)

    extra_attrs: dict[str, Any] = {}
    for key in (
        ClaudeCodeEvents.SPEED,
        ClaudeCodeEvents.CACHE_CREATION_TOKENS,
        ClaudeCodeEvents.EVENT_SEQUENCE,
    ):
        if key in attrs:
            extra_attrs[key] = attrs[key]

    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=_trace_id_from_session(session_id),
        name=GenAIAttributes.SPAN_LLM_CALL,
        kind=SpanKind.CLIENT,
        status_code=SpanStatus.OK,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        agent_id=resource_attrs.get("service.name", "claude-code"),
        session_id=session_id,
        conversation_id=prompt_id,
        parent_span_id=_span_id_from_prompt(prompt_id) if prompt_id else None,
        provider="anthropic",
        model=str(attrs["model"]) if "model" in attrs else None,
        input_tokens=_safe_int(attrs.get(ClaudeCodeEvents.INPUT_TOKENS)),
        output_tokens=_safe_int(attrs.get(ClaudeCodeEvents.OUTPUT_TOKENS)),
        cache_tokens=_safe_int(attrs.get(ClaudeCodeEvents.CACHE_READ_TOKENS, 0)),
        cost_usd=float(attrs[ClaudeCodeEvents.COST_USD]) if ClaudeCodeEvents.COST_USD in attrs else None,
        attributes=extra_attrs,
    )


def _tool_result_to_span(
    attrs: dict[str, Any],
    resource_attrs: dict[str, Any],
    timestamp_ns: int,
) -> NormalizedSpan:
    session_id = str(attrs[ClaudeCodeEvents.SESSION_ID])
    prompt_id = attrs.get(ClaudeCodeEvents.PROMPT_ID)
    duration_ms = float(attrs[ClaudeCodeEvents.DURATION_MS])
    start_time = _ts_to_datetime(timestamp_ns)
    end_time = start_time + timedelta(milliseconds=duration_ms)

    success_val = attrs.get(ClaudeCodeEvents.SUCCESS)
    # Claude Code sends success as a boolean or the string "true"
    if isinstance(success_val, bool):
        ok = success_val
    else:
        ok = str(success_val).lower() == "true"

    status_code = SpanStatus.OK if ok else SpanStatus.ERROR
    status_message = attrs.get(ClaudeCodeEvents.ERROR) if not ok else None

    extra_attrs: dict[str, Any] = {}
    for key in (
        ClaudeCodeEvents.TOOL_PARAMETERS,
        ClaudeCodeEvents.TOOL_INPUT,
        ClaudeCodeEvents.DECISION_TYPE,
        ClaudeCodeEvents.TOOL_RESULT_SIZE,
        ClaudeCodeEvents.EVENT_SEQUENCE,
    ):
        if key in attrs:
            extra_attrs[key] = attrs[key]

    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=_trace_id_from_session(session_id),
        name=GenAIAttributes.SPAN_TOOL_CALL,
        kind=SpanKind.INTERNAL,
        status_code=status_code,
        status_message=status_message,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        agent_id=resource_attrs.get("service.name", "claude-code"),
        session_id=session_id,
        conversation_id=prompt_id,
        parent_span_id=_span_id_from_prompt(prompt_id) if prompt_id else None,
        tool_name=str(attrs[ClaudeCodeEvents.TOOL_NAME]),
        attributes=extra_attrs,
    )


def _api_error_to_span(
    attrs: dict[str, Any],
    resource_attrs: dict[str, Any],
    timestamp_ns: int,
) -> NormalizedSpan:
    session_id = str(attrs[ClaudeCodeEvents.SESSION_ID])
    prompt_id = attrs.get(ClaudeCodeEvents.PROMPT_ID)
    duration_ms = float(attrs[ClaudeCodeEvents.DURATION_MS])
    start_time = _ts_to_datetime(timestamp_ns)
    end_time = start_time + timedelta(milliseconds=duration_ms)

    extra_attrs: dict[str, Any] = {}
    for key in (
        ClaudeCodeEvents.STATUS_CODE_HTTP,
        ClaudeCodeEvents.ATTEMPT,
        ClaudeCodeEvents.EVENT_SEQUENCE,
    ):
        if key in attrs:
            extra_attrs[key] = attrs[key]

    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=_trace_id_from_session(session_id),
        name=GenAIAttributes.SPAN_LLM_CALL,
        kind=SpanKind.CLIENT,
        status_code=SpanStatus.ERROR,
        status_message=str(attrs[ClaudeCodeEvents.ERROR]),
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        agent_id=resource_attrs.get("service.name", "claude-code"),
        session_id=session_id,
        conversation_id=prompt_id,
        parent_span_id=_span_id_from_prompt(prompt_id) if prompt_id else None,
        provider="anthropic",
        model=str(attrs["model"]) if "model" in attrs else None,
        attributes=extra_attrs,
    )


def _user_prompt_to_span(
    attrs: dict[str, Any],
    resource_attrs: dict[str, Any],
    timestamp_ns: int,
) -> NormalizedSpan:
    session_id = str(attrs[ClaudeCodeEvents.SESSION_ID])
    prompt_id = attrs.get(ClaudeCodeEvents.PROMPT_ID)
    start_time = _ts_to_datetime(timestamp_ns)

    extra_attrs: dict[str, Any] = {}
    for key in ("prompt_length", ClaudeCodeEvents.EVENT_SEQUENCE):
        if key in attrs:
            extra_attrs[key] = attrs[key]

    return NormalizedSpan(
        span_id=_span_id_from_prompt(prompt_id) if prompt_id else new_span_id(),
        trace_id=_trace_id_from_session(session_id),
        name=GenAIAttributes.SPAN_INVOKE_AGENT,
        kind=SpanKind.SERVER,
        status_code=SpanStatus.OK,
        start_time=start_time,
        end_time=start_time,
        agent_id=resource_attrs.get("service.name", "claude-code"),
        session_id=session_id,
        conversation_id=prompt_id,
        attributes=extra_attrs,
    )


def _tool_decision_to_span(
    attrs: dict[str, Any],
    resource_attrs: dict[str, Any],
    timestamp_ns: int,
) -> NormalizedSpan:
    session_id = str(attrs[ClaudeCodeEvents.SESSION_ID])
    prompt_id = attrs.get(ClaudeCodeEvents.PROMPT_ID)
    start_time = _ts_to_datetime(timestamp_ns)

    extra_attrs: dict[str, Any] = {}
    for key in (
        ClaudeCodeEvents.DECISION,
        ClaudeCodeEvents.DECISION_SOURCE,
        ClaudeCodeEvents.EVENT_SEQUENCE,
    ):
        if key in attrs:
            extra_attrs[key] = attrs[key]

    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=_trace_id_from_session(session_id),
        name="tool_decision",
        kind=SpanKind.INTERNAL,
        status_code=SpanStatus.OK,
        start_time=start_time,
        agent_id=resource_attrs.get("service.name", "claude-code"),
        session_id=session_id,
        conversation_id=prompt_id,
        tool_name=str(attrs[ClaudeCodeEvents.TOOL_NAME]),
        attributes=extra_attrs,
    )


def _codex_api_request_to_span(
    attrs: dict[str, Any],
    resource_attrs: dict[str, Any],
    timestamp_ns: int,
) -> "NormalizedSpan | None":
    """Only convert api_request events that carry an error; skip successful ones.

    Token counts live on codex.sse_event (kind=completion), so successful
    api_request events are redundant — _codex_sse_event_to_span captures them.
    """
    error = attrs.get(CodexEvents.ERROR_MESSAGE)
    if not error:
        return None

    conversation_id = str(attrs.get(CodexEvents.CONVERSATION_ID, "unknown"))
    duration_ms = float(attrs.get(CodexEvents.DURATION_MS, 0))
    start_time = _ts_to_datetime(timestamp_ns)
    end_time = start_time + timedelta(milliseconds=duration_ms)

    extra_attrs: dict[str, Any] = {}
    for key in (CodexEvents.HTTP_STATUS, CodexEvents.ATTEMPT):
        if key in attrs:
            extra_attrs[key] = attrs[key]

    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=_trace_id_from_session(conversation_id),
        name=GenAIAttributes.SPAN_LLM_CALL,
        kind=SpanKind.CLIENT,
        status_code=SpanStatus.ERROR,
        status_message=error,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        agent_id=resource_attrs.get("service.name", "codex-cli"),
        session_id=conversation_id,
        conversation_id=conversation_id,
        provider="openai",
        model=str(attrs["model"]) if "model" in attrs else None,
        attributes=extra_attrs,
    )


def _codex_sse_event_to_span(
    attrs: dict[str, Any],
    resource_attrs: dict[str, Any],
    timestamp_ns: int,
) -> "NormalizedSpan | None":
    """Convert SSE completion events to LLM call spans.

    Codex emits one sse_event per SSE chunk; only the final chunk has
    event.kind == "completion" and carries token counts.  All other
    kinds (e.g. "content_block_delta") are skipped.
    """
    if attrs.get(CodexEvents.EVENT_KIND) != "response.completed":
        return None

    conversation_id = str(attrs.get(CodexEvents.CONVERSATION_ID, "unknown"))
    duration_ms = float(attrs.get(CodexEvents.DURATION_MS, 0))
    start_time = _ts_to_datetime(timestamp_ns)
    end_time = start_time + timedelta(milliseconds=duration_ms)

    extra_attrs: dict[str, Any] = {}
    for key in (CodexEvents.REASONING_TOKEN_COUNT, CodexEvents.TOOL_TOKEN_COUNT):
        if key in attrs:
            extra_attrs[key] = attrs[key]

    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=_trace_id_from_session(conversation_id),
        name=GenAIAttributes.SPAN_LLM_CALL,
        kind=SpanKind.CLIENT,
        status_code=SpanStatus.OK,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        agent_id=resource_attrs.get("service.name", "codex-cli"),
        session_id=conversation_id,
        conversation_id=conversation_id,
        provider="openai",
        model=str(attrs["model"]) if "model" in attrs else None,
        input_tokens=_safe_int(attrs.get(CodexEvents.INPUT_TOKEN_COUNT)),
        output_tokens=_safe_int(attrs.get(CodexEvents.OUTPUT_TOKEN_COUNT)),
        cache_tokens=_safe_int(attrs.get(CodexEvents.CACHED_TOKEN_COUNT, 0)),
        attributes=extra_attrs,
    )


def _codex_user_prompt_to_span(
    attrs: dict[str, Any],
    resource_attrs: dict[str, Any],
    timestamp_ns: int,
) -> NormalizedSpan:
    conversation_id = str(attrs.get(CodexEvents.CONVERSATION_ID, "unknown"))
    start_time = _ts_to_datetime(timestamp_ns)

    extra_attrs: dict[str, Any] = {}
    for key in (CodexEvents.PROMPT_LENGTH, CodexEvents.PROMPT):
        if key in attrs:
            extra_attrs[key] = attrs[key]

    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=_trace_id_from_session(conversation_id),
        name=GenAIAttributes.SPAN_INVOKE_AGENT,
        kind=SpanKind.SERVER,
        status_code=SpanStatus.OK,
        start_time=start_time,
        end_time=start_time,
        agent_id=resource_attrs.get("service.name", "codex-cli"),
        session_id=conversation_id,
        conversation_id=conversation_id,
        attributes=extra_attrs,
    )


def _codex_tool_decision_to_span(
    attrs: dict[str, Any],
    resource_attrs: dict[str, Any],
    timestamp_ns: int,
) -> NormalizedSpan:
    conversation_id = str(attrs.get(CodexEvents.CONVERSATION_ID, "unknown"))
    start_time = _ts_to_datetime(timestamp_ns)

    extra_attrs: dict[str, Any] = {}
    for key in (CodexEvents.DECISION, CodexEvents.DECISION_SOURCE, CodexEvents.CALL_ID):
        if key in attrs:
            extra_attrs[key] = attrs[key]

    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=_trace_id_from_session(conversation_id),
        name="tool_decision",
        kind=SpanKind.INTERNAL,
        status_code=SpanStatus.OK,
        start_time=start_time,
        agent_id=resource_attrs.get("service.name", "codex-cli"),
        session_id=conversation_id,
        conversation_id=conversation_id,
        tool_name=str(attrs.get(CodexEvents.TOOL_NAME, "")),
        attributes=extra_attrs,
    )


def _codex_tool_result_to_span(
    attrs: dict[str, Any],
    resource_attrs: dict[str, Any],
    timestamp_ns: int,
) -> NormalizedSpan:
    conversation_id = str(attrs.get(CodexEvents.CONVERSATION_ID, "unknown"))
    duration_ms = float(attrs.get(CodexEvents.DURATION_MS, 0))
    start_time = _ts_to_datetime(timestamp_ns)
    end_time = start_time + timedelta(milliseconds=duration_ms)

    success_val = attrs.get(CodexEvents.SUCCESS)
    if isinstance(success_val, bool):
        ok = success_val
    else:
        ok = str(success_val).lower() == "true"

    status_code = SpanStatus.OK if ok else SpanStatus.ERROR
    status_message = attrs.get(CodexEvents.ERROR_MESSAGE) if not ok else None

    extra_attrs: dict[str, Any] = {}
    for key in (CodexEvents.ARGUMENTS, CodexEvents.CALL_ID):
        if key in attrs:
            extra_attrs[key] = attrs[key]

    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=_trace_id_from_session(conversation_id),
        name=GenAIAttributes.SPAN_TOOL_CALL,
        kind=SpanKind.INTERNAL,
        status_code=status_code,
        status_message=status_message,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        agent_id=resource_attrs.get("service.name", "codex-cli"),
        session_id=conversation_id,
        conversation_id=conversation_id,
        tool_name=str(attrs.get(CodexEvents.TOOL_NAME, "")),
        attributes=extra_attrs,
    )


_CONVERTERS = {
    ClaudeCodeEvents.API_REQUEST:   _api_request_to_span,
    ClaudeCodeEvents.TOOL_RESULT:   _tool_result_to_span,
    ClaudeCodeEvents.API_ERROR:     _api_error_to_span,
    ClaudeCodeEvents.USER_PROMPT:   _user_prompt_to_span,
    ClaudeCodeEvents.TOOL_DECISION: _tool_decision_to_span,
    # Codex CLI events
    CodexEvents.API_REQUEST:   _codex_api_request_to_span,
    CodexEvents.SSE_EVENT:     _codex_sse_event_to_span,
    CodexEvents.USER_PROMPT:   _codex_user_prompt_to_span,
    CodexEvents.TOOL_DECISION: _codex_tool_decision_to_span,
    CodexEvents.TOOL_RESULT:   _codex_tool_result_to_span,
}


def parse_log_records(
    body: dict,
    pipeline: IngestPipeline,
) -> tuple[int, list[dict[str, str]]]:
    """
    Walk resourceLogs -> scopeLogs -> logRecords.
    Dispatch each record by event name to the appropriate converter.
    Call pipeline.process() for each resulting NormalizedSpan.
    Returns (ingested_count, rejections_list).

    Same error-tolerance as spans.py: individual failures are logged and
    collected in rejections, never propagated. Batch continues processing.
    """
    ingested = 0
    rejections: list[dict[str, str]] = []

    for resource_log in body.get("resourceLogs", []):
        # Extract resource-level attributes (e.g. service.name)
        resource = resource_log.get("resource", {})
        resource_attrs = _parse_attrs(resource.get("attributes", []))

        for scope_log in resource_log.get("scopeLogs", []):
            for record in scope_log.get("logRecords", []):
                timestamp_ns = int(record.get("timeUnixNano", 0))
                body_val = record.get("body", {})
                event_name = _otlp_value(body_val) if isinstance(body_val, dict) else body_val

                # Parse attributes here — needed both for the Codex event.name
                # fallback and for converters that follow.
                attrs = _parse_attrs(record.get("attributes", []))

                # Codex CLI puts the event name in attrs["event.name"] rather
                # than the log record body; fall back to that when body is empty.
                if not isinstance(event_name, str):
                    event_name = attrs.get("event.name")

                if not isinstance(event_name, str):
                    continue

                # Codex CLI sets timeUnixNano=0 and puts the real timestamp in
                # attrs["event.timestamp"] as an ISO-8601 UTC string.
                if timestamp_ns == 0:
                    ts_str = attrs.get(CodexEvents.EVENT_TIMESTAMP)
                    if ts_str:
                        try:
                            dt = datetime.fromisoformat(ts_str.rstrip("Z") + "+00:00")
                            timestamp_ns = int(dt.timestamp() * 1e9)
                        except ValueError:
                            pass

                converter = _CONVERTERS.get(event_name)
                if converter is None:
                    # Unknown event — skip silently
                    continue

                record_id = f"{event_name}:{timestamp_ns}"

                try:
                    span = converter(attrs, resource_attrs, timestamp_ns)
                    if span is None:
                        continue
                    pipeline.process(span)
                    ingested += 1
                except SpanRejectedError as exc:
                    rejections.append({"record_id": record_id, "reason": str(exc)})
                except Exception as exc:
                    logger.warning("Failed to process log record %s: %s", record_id, exc)
                    rejections.append({"record_id": record_id, "reason": str(exc)})

    return ingested, rejections
