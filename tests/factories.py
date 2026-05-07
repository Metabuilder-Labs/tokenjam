"""
Span factory for tests. Never construct NormalizedSpan directly in tests --
use these factory functions. This ensures consistent defaults and readable tests.
"""
from __future__ import annotations
from datetime import timedelta
from tj.core.models import (
    NormalizedSpan, SessionRecord,
    SpanStatus, SpanKind,
)
from tj.utils.ids import new_uuid, new_trace_id, new_span_id
from tj.utils.time_parse import utcnow


def make_llm_span(
    agent_id: str = "test-agent",
    model: str = "claude-haiku-4-5",
    provider: str = "anthropic",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_tokens: int = 0,
    cost_usd: float | None = None,
    tool_name: str | None = None,
    status: str = "ok",
    duration_ms: float = 800.0,
    start_time=None,
    conversation_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    session_id: str | None = None,
    extra_attributes: dict | None = None,
) -> NormalizedSpan:
    """Create a NormalizedSpan representing a single LLM call."""
    now = start_time or utcnow()
    end = now + timedelta(milliseconds=duration_ms)
    attrs = extra_attributes.copy() if extra_attributes else {}

    return NormalizedSpan(
        span_id=span_id or new_span_id(),
        trace_id=trace_id or new_trace_id(),
        name="gen_ai.llm.call",
        kind=SpanKind.CLIENT,
        status_code=SpanStatus(status),
        start_time=now,
        end_time=end,
        duration_ms=duration_ms,
        agent_id=agent_id,
        session_id=session_id,
        provider=provider,
        model=model,
        tool_name=tool_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_tokens=cache_tokens,
        cost_usd=cost_usd,
        conversation_id=conversation_id,
        attributes=attrs,
    )


def make_tool_span(
    agent_id: str = "test-agent",
    tool_name: str = "test_tool",
    status: str = "ok",
    duration_ms: float = 100.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    conversation_id: str | None = None,
    trace_id: str | None = None,
) -> NormalizedSpan:
    """Create a NormalizedSpan representing a single tool call."""
    now = utcnow()
    end = now + timedelta(milliseconds=duration_ms)

    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=trace_id or new_trace_id(),
        name="gen_ai.tool.call",
        kind=SpanKind.INTERNAL,
        status_code=SpanStatus(status),
        start_time=now,
        end_time=end,
        duration_ms=duration_ms,
        agent_id=agent_id,
        tool_name=tool_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        conversation_id=conversation_id,
    )


def make_session(
    agent_id: str = "test-agent",
    session_id: str | None = None,
    conversation_id: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    tool_call_count: int = 0,
    error_count: int = 0,
    total_cost_usd: float | None = None,
    status: str = "completed",
    duration_seconds: float = 60.0,
) -> SessionRecord:
    """Create a SessionRecord with sensible defaults."""
    now = utcnow()
    started = now - timedelta(seconds=duration_seconds)

    return SessionRecord(
        session_id=session_id or new_uuid(),
        agent_id=agent_id,
        started_at=started,
        ended_at=now if status == "completed" else None,
        conversation_id=conversation_id or new_uuid(),
        status=status,
        total_cost_usd=total_cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_call_count=tool_call_count,
        error_count=error_count,
    )


def make_session_with_spans(
    agent_id: str = "test-agent",
    span_count: int = 5,
    model: str = "claude-haiku-4-5",
    input_tokens_per_span: int = 1000,
    output_tokens_per_span: int = 200,
) -> tuple[SessionRecord, list[NormalizedSpan]]:
    """Create a session and a matching list of spans sharing a conversation_id."""
    conv_id = new_uuid()
    session_id = new_uuid()
    trace_id = new_trace_id()

    total_input = input_tokens_per_span * span_count
    total_output = output_tokens_per_span * span_count

    session = make_session(
        agent_id=agent_id,
        session_id=session_id,
        conversation_id=conv_id,
        input_tokens=total_input,
        output_tokens=total_output,
        tool_call_count=0,
        duration_seconds=span_count * 1.0,
    )

    spans = []
    for i in range(span_count):
        span = make_llm_span(
            agent_id=agent_id,
            model=model,
            input_tokens=input_tokens_per_span,
            output_tokens=output_tokens_per_span,
            trace_id=trace_id,
            session_id=session_id,
            conversation_id=conv_id,
            duration_ms=800.0,
        )
        spans.append(span)

    return session, spans


# -- Claude Code OTLP log factories --

def make_claude_code_api_request_log(
    session_id: str = "cc-session-1",
    prompt_id: str | None = None,
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    cost_usd: float = 0.003,
    duration_ms: float = 1200.0,
    sequence: int = 1,
    timestamp_ns: int | None = None,
) -> dict:
    """Build an OTLP logRecord dict for a Claude Code api_request event."""
    if timestamp_ns is None:
        import time
        timestamp_ns = int(time.time() * 1e9)
    if prompt_id is None:
        prompt_id = new_uuid()
    attributes = [
        {"key": "session.id", "value": {"stringValue": session_id}},
        {"key": "prompt.id", "value": {"stringValue": prompt_id}},
        {"key": "event.sequence", "value": {"intValue": str(sequence)}},
        {"key": "model", "value": {"stringValue": model}},
        {"key": "input_tokens", "value": {"intValue": str(input_tokens)}},
        {"key": "output_tokens", "value": {"intValue": str(output_tokens)}},
        {"key": "cache_read_tokens", "value": {"intValue": str(cache_read_tokens)}},
        {"key": "cache_creation_tokens", "value": {"intValue": str(cache_creation_tokens)}},
        {"key": "cost_usd", "value": {"doubleValue": cost_usd}},
        {"key": "duration_ms", "value": {"doubleValue": duration_ms}},
    ]
    return {
        "timeUnixNano": str(timestamp_ns),
        "body": {"stringValue": "claude_code.api_request"},
        "attributes": attributes,
    }


def make_claude_code_tool_result_log(
    session_id: str = "cc-session-1",
    prompt_id: str | None = None,
    tool_name: str = "Read",
    success: bool = True,
    duration_ms: float = 50.0,
    error: str | None = None,
    sequence: int = 2,
    timestamp_ns: int | None = None,
) -> dict:
    """Build an OTLP logRecord dict for a Claude Code tool_result event."""
    if timestamp_ns is None:
        import time
        timestamp_ns = int(time.time() * 1e9)
    if prompt_id is None:
        prompt_id = new_uuid()
    attributes = [
        {"key": "session.id", "value": {"stringValue": session_id}},
        {"key": "prompt.id", "value": {"stringValue": prompt_id}},
        {"key": "event.sequence", "value": {"intValue": str(sequence)}},
        {"key": "tool_name", "value": {"stringValue": tool_name}},
        {"key": "success", "value": {"boolValue": success}},
        {"key": "duration_ms", "value": {"doubleValue": duration_ms}},
    ]
    if error:
        attributes.append({"key": "error", "value": {"stringValue": error}})
    return {
        "timeUnixNano": str(timestamp_ns),
        "body": {"stringValue": "claude_code.tool_result"},
        "attributes": attributes,
    }


def make_claude_code_api_error_log(
    session_id: str = "cc-session-1",
    model: str = "claude-sonnet-4-6",
    error: str = "rate_limit_exceeded",
    status_code: int = 429,
    attempt: int = 1,
    duration_ms: float = 100.0,
    sequence: int = 3,
    timestamp_ns: int | None = None,
) -> dict:
    """Build an OTLP logRecord dict for a Claude Code api_error event."""
    if timestamp_ns is None:
        import time
        timestamp_ns = int(time.time() * 1e9)
    return {
        "timeUnixNano": str(timestamp_ns),
        "body": {"stringValue": "claude_code.api_error"},
        "attributes": [
            {"key": "session.id", "value": {"stringValue": session_id}},
            {"key": "event.sequence", "value": {"intValue": str(sequence)}},
            {"key": "model", "value": {"stringValue": model}},
            {"key": "error", "value": {"stringValue": error}},
            {"key": "status_code", "value": {"intValue": str(status_code)}},
            {"key": "attempt", "value": {"intValue": str(attempt)}},
            {"key": "duration_ms", "value": {"doubleValue": duration_ms}},
        ],
    }


def make_otlp_logs_body(
    log_records: list[dict],
    service_name: str = "claude-code",
) -> dict:
    """Wrap log records in the full resourceLogs envelope."""
    return {
        "resourceLogs": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": service_name}},
                ],
            },
            "scopeLogs": [{"logRecords": log_records}],
        }],
    }
