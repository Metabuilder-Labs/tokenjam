"""
Span factory for tests. Never construct NormalizedSpan directly in tests --
use these factory functions. This ensures consistent defaults and readable tests.
"""
from __future__ import annotations
from datetime import timedelta
from tokenjam.core.models import (
    NormalizedSpan, SessionRecord,
    SpanStatus, SpanKind,
)
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.utils.ids import new_uuid, new_trace_id, new_span_id
from tokenjam.utils.time_parse import utcnow


def make_invoke_agent_span(
    agent_id: str = "test-agent",
    session_id: str | None = None,
    conversation_id: str | None = None,
    duration_ms: float = 0.0,
    start_time=None,
    trace_id: str | None = None,
    service_namespace: str | None = None,
) -> NormalizedSpan:
    """Create an ``invoke_agent`` span.

    ``duration_ms == 0`` -> a zero-duration turn-start marker, exactly what the
    Claude Code / Codex logs path emits for every ``user_prompt`` event
    (``end_time == start_time``). These mark the *start* of a turn and must NOT
    complete the session.

    ``duration_ms > 0`` -> a real session-wrapping span, what the SDK
    ``@watch()`` path emits to bracket a whole agent run. This DOES complete the
    session.
    """
    now = start_time or utcnow()
    end = now + timedelta(milliseconds=duration_ms)
    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=trace_id or new_trace_id(),
        name=GenAIAttributes.SPAN_INVOKE_AGENT,
        kind=SpanKind.SERVER,
        status_code=SpanStatus.OK,
        start_time=now,
        end_time=end,
        duration_ms=duration_ms if duration_ms else None,
        agent_id=agent_id,
        session_id=session_id,
        conversation_id=conversation_id,
        service_namespace=service_namespace,
    )


def make_llm_span(
    agent_id: str = "test-agent",
    model: str = "claude-haiku-4-5",
    provider: str = "anthropic",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cache_tokens: int = 0,
    cache_write_tokens: int = 0,
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
    billing_account: str | None = "anthropic",
    service_namespace: str | None = None,
    service_instance_id: str | None = None,
) -> NormalizedSpan:
    """
    Create a NormalizedSpan representing a single LLM call.

    `billing_account` defaults to "anthropic" so existing tests using the
    default `provider="anthropic"` get a sensible value. Tests exercising
    OpenAI/Google/Bedrock/local paths should pass it explicitly.
    """
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
        cache_write_tokens=cache_write_tokens,
        cost_usd=cost_usd,
        conversation_id=conversation_id,
        attributes=attrs,
        billing_account=billing_account,
        service_namespace=service_namespace,
        service_instance_id=service_instance_id,
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
    plan_tier: str = "api",
    started_at=None,
    ended_at=None,
    service_namespace: str | None = None,
    service_instance_id: str | None = None,
    run_id: str | None = None,
    parent_session_id: str | None = None,
) -> SessionRecord:
    """
    Create a SessionRecord with sensible defaults.

    `plan_tier` defaults to "api" so existing tests see dollar figures
    rendered normally (least-disruption). Tests for subscription / local /
    unknown rendering paths should pass it explicitly.

    Pass `started_at` / `ended_at` to control the timeline explicitly (e.g.
    to test ordering by last activity, or the idle/stale lifecycle tiers);
    otherwise they derive from `duration_seconds` ending at "now".

    `service_instance_id` sets the per-terminal label (used by close-by-instance
    and session-lifecycle tests); `service_namespace` sets the project grouping.

    `run_id` ties this session to a fan-out harness run (cross-session run
    grouping); `parent_session_id` is the optional spawning-session id for the
    run tree.
    """
    now = utcnow()
    started = started_at or (now - timedelta(seconds=duration_seconds))
    if ended_at is not None:
        ended = ended_at
    else:
        ended = now if status == "completed" else None

    return SessionRecord(
        session_id=session_id or new_uuid(),
        agent_id=agent_id,
        started_at=started,
        ended_at=ended,
        conversation_id=conversation_id or new_uuid(),
        status=status,
        total_cost_usd=total_cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_call_count=tool_call_count,
        error_count=error_count,
        plan_tier=plan_tier,
        service_namespace=service_namespace,
        service_instance_id=service_instance_id,
        run_id=run_id,
        parent_session_id=parent_session_id,
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
    resource_attributes: dict[str, str] | None = None,
) -> dict:
    """Wrap log records in the full resourceLogs envelope.

    `resource_attributes` adds extra string-valued resource attributes (e.g.
    ``{"tokenjam.run_id": "run-1"}``) so tests can exercise resource-attr
    capture (run grouping, service.namespace) on the logs path.
    """
    attributes = [
        {"key": "service.name", "value": {"stringValue": service_name}},
    ]
    for key, value in (resource_attributes or {}).items():
        attributes.append({"key": key, "value": {"stringValue": value}})
    return {
        "resourceLogs": [{
            "resource": {"attributes": attributes},
            "scopeLogs": [{"logRecords": log_records}],
        }],
    }
