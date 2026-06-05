"""Unit tests for the Claude Code and Codex log-to-span converters."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from tokenjam.api.routes.logs import (
    _api_error_to_span,
    _api_request_to_span,
    _codex_api_request_to_span,
    _codex_sse_event_to_span,
    _codex_tool_decision_to_span,
    _codex_tool_result_to_span,
    _codex_user_prompt_to_span,
    _span_id_from_prompt,
    _tool_decision_to_span,
    _tool_result_to_span,
    _trace_id_from_session,
    _user_prompt_to_span,
    parse_log_records,
)
from tokenjam.core.models import SpanKind, SpanStatus
from tokenjam.otel.semconv import GenAIAttributes
from tests.factories import (
    make_claude_code_api_error_log,
    make_claude_code_api_request_log,
    make_claude_code_tool_result_log,
)


NOW_NS = int(time.time() * 1e9)
SESSION_ID = "test-session-abc"
PROMPT_ID = "test-prompt-xyz"
RESOURCE = {"service.name": "claude-code"}


def _req_attrs(**overrides) -> dict:
    base = {
        "session.id": SESSION_ID,
        "prompt.id": PROMPT_ID,
        "model": "claude-sonnet-4-6",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_read_tokens": 500,
        "cache_creation_tokens": 100,
        "cost_usd": 0.003,
        "duration_ms": 1200.0,
        "event.sequence": 1,
    }
    base.update(overrides)
    return base


def test_api_request_produces_llm_span():
    span = _api_request_to_span(_req_attrs(), RESOURCE, NOW_NS)
    assert span.name == GenAIAttributes.SPAN_LLM_CALL
    assert span.kind == SpanKind.CLIENT
    assert span.status_code == SpanStatus.OK
    assert span.model == "claude-sonnet-4-6"
    assert span.provider == "anthropic"
    assert span.input_tokens == 1000
    assert span.output_tokens == 200
    assert span.cost_usd == pytest.approx(0.003)
    assert span.duration_ms == pytest.approx(1200.0)
    assert span.agent_id == "claude-code"
    assert span.session_id == SESSION_ID


def test_api_request_cache_tokens():
    span = _api_request_to_span(_req_attrs(), RESOURCE, NOW_NS)
    # cache_read_tokens -> cache_tokens field (cache reads only)
    assert span.cache_tokens == 500
    # cache_creation_tokens is now its own indexed field, not stuffed in attrs
    assert span.cache_creation_tokens == 100
    assert "cache_creation_tokens" not in span.attributes


def test_tool_result_success_produces_ok_span():
    attrs = {
        "session.id": SESSION_ID,
        "prompt.id": PROMPT_ID,
        "tool_name": "Read",
        "success": True,
        "duration_ms": 50.0,
        "event.sequence": 2,
    }
    span = _tool_result_to_span(attrs, RESOURCE, NOW_NS)
    assert span.name == GenAIAttributes.SPAN_TOOL_CALL
    assert span.kind == SpanKind.INTERNAL
    assert span.status_code == SpanStatus.OK
    assert span.tool_name == "Read"
    assert span.duration_ms == pytest.approx(50.0)
    assert span.status_message is None


def test_tool_result_failure_produces_error_span():
    attrs = {
        "session.id": SESSION_ID,
        "prompt.id": PROMPT_ID,
        "tool_name": "Bash",
        "success": False,
        "error": "command not found",
        "duration_ms": 10.0,
        "event.sequence": 3,
    }
    span = _tool_result_to_span(attrs, RESOURCE, NOW_NS)
    assert span.status_code == SpanStatus.ERROR
    assert span.status_message == "command not found"


def test_api_error_produces_error_llm_span():
    attrs = {
        "session.id": SESSION_ID,
        "model": "claude-sonnet-4-6",
        "error": "rate_limit_exceeded",
        "status_code": 429,
        "attempt": 1,
        "duration_ms": 100.0,
        "event.sequence": 4,
    }
    span = _api_error_to_span(attrs, RESOURCE, NOW_NS)
    assert span.name == GenAIAttributes.SPAN_LLM_CALL
    assert span.status_code == SpanStatus.ERROR
    assert span.status_message == "rate_limit_exceeded"
    assert span.attributes.get("status_code") == 429


def test_user_prompt_produces_invoke_agent_span():
    attrs = {
        "session.id": SESSION_ID,
        "prompt.id": PROMPT_ID,
        "prompt_length": 42,
        "event.sequence": 1,
    }
    span = _user_prompt_to_span(attrs, RESOURCE, NOW_NS)
    assert span.name == GenAIAttributes.SPAN_INVOKE_AGENT
    assert span.kind == SpanKind.SERVER
    assert span.status_code == SpanStatus.OK
    assert span.attributes.get("prompt_length") == 42


def test_tool_decision_produces_internal_span():
    attrs = {
        "session.id": SESSION_ID,
        "prompt.id": PROMPT_ID,
        "tool_name": "Bash",
        "decision": "allow",
        "source": "rules",
        "event.sequence": 5,
    }
    span = _tool_decision_to_span(attrs, RESOURCE, NOW_NS)
    assert span.name == "tool_decision"
    assert span.kind == SpanKind.INTERNAL
    assert span.attributes.get("decision") == "allow"
    assert span.attributes.get("source") == "rules"


def test_trace_id_deterministic_from_session():
    tid1 = _trace_id_from_session("session-abc")
    tid2 = _trace_id_from_session("session-abc")
    assert tid1 == tid2
    assert len(tid1) == 32


def test_trace_id_differs_across_sessions():
    tid1 = _trace_id_from_session("session-abc")
    tid2 = _trace_id_from_session("session-xyz")
    assert tid1 != tid2


def test_parent_span_from_prompt_id():
    attrs_tool = {
        "session.id": SESSION_ID, "prompt.id": PROMPT_ID,
        "tool_name": "Read", "success": True, "duration_ms": 10.0,
    }
    attrs_api = _req_attrs()

    tool_span = _tool_result_to_span(attrs_tool, RESOURCE, NOW_NS)
    api_span = _api_request_to_span(attrs_api, RESOURCE, NOW_NS)

    assert tool_span.parent_span_id == _span_id_from_prompt(PROMPT_ID)
    assert api_span.parent_span_id == _span_id_from_prompt(PROMPT_ID)
    assert tool_span.parent_span_id == api_span.parent_span_id


def test_session_id_extracted_from_attrs():
    span = _api_request_to_span(_req_attrs(**{"session.id": "my-session"}), RESOURCE, NOW_NS)
    assert span.session_id == "my-session"


def test_conversation_id_from_prompt_id():
    span = _api_request_to_span(_req_attrs(**{"prompt.id": "my-prompt"}), RESOURCE, NOW_NS)
    assert span.conversation_id == "my-prompt"


def test_event_sequence_in_attributes():
    span = _api_request_to_span(_req_attrs(**{"event.sequence": 99}), RESOURCE, NOW_NS)
    assert span.attributes.get("event.sequence") == 99


def test_cache_creation_tokens_indexed_field():
    span = _api_request_to_span(_req_attrs(**{"cache_creation_tokens": 256}), RESOURCE, NOW_NS)
    # cache_creation_tokens is captured as its own indexed field (not dropped
    # into the generic attributes blob like it used to be)
    assert span.cache_creation_tokens == 256
    assert "cache_creation_tokens" not in span.attributes
    # cache_tokens field stays reads-only
    assert span.cache_tokens == 500  # from default _req_attrs


def test_missing_optional_fields_handled():
    # Minimal attrs — no cost_usd, no cache tokens, no prompt.id
    attrs = {
        "session.id": SESSION_ID,
        "model": "claude-sonnet-4-6",
        "input_tokens": 100,
        "output_tokens": 50,
        "duration_ms": 500.0,
    }
    span = _api_request_to_span(attrs, RESOURCE, NOW_NS)
    assert span.cost_usd is None
    assert span.conversation_id is None
    assert span.parent_span_id is None


def test_user_prompt_span_id_matches_parent():
    """The user_prompt span_id should equal the parent_span_id of its children."""
    prompt_attrs = {
        "session.id": SESSION_ID,
        "prompt.id": PROMPT_ID,
        "prompt_length": 10,
    }
    tool_attrs = {
        "session.id": SESSION_ID, "prompt.id": PROMPT_ID,
        "tool_name": "Read", "success": True, "duration_ms": 10.0,
    }
    prompt_span = _user_prompt_to_span(prompt_attrs, RESOURCE, NOW_NS)
    tool_span = _tool_result_to_span(tool_attrs, RESOURCE, NOW_NS)

    assert prompt_span.span_id == _span_id_from_prompt(PROMPT_ID)
    assert tool_span.parent_span_id == prompt_span.span_id


# ── Missing required fields ──────────────────────────────────────────────


def test_api_request_missing_session_id_raises():
    attrs = _req_attrs()
    del attrs["session.id"]
    with pytest.raises(KeyError):
        _api_request_to_span(attrs, RESOURCE, NOW_NS)


def test_api_request_missing_duration_ms_raises():
    attrs = _req_attrs()
    del attrs["duration_ms"]
    with pytest.raises(KeyError):
        _api_request_to_span(attrs, RESOURCE, NOW_NS)


def test_tool_result_missing_tool_name_raises():
    attrs = {
        "session.id": SESSION_ID,
        "prompt.id": PROMPT_ID,
        "success": True,
        "duration_ms": 10.0,
    }
    with pytest.raises(KeyError):
        _tool_result_to_span(attrs, RESOURCE, NOW_NS)


def test_api_error_missing_error_raises():
    attrs = {
        "session.id": SESSION_ID,
        "model": "claude-sonnet-4-6",
        "duration_ms": 100.0,
    }
    with pytest.raises(KeyError):
        _api_error_to_span(attrs, RESOURCE, NOW_NS)


# ── Codex converter tests ────────────────────────────────────────────────

CODEX_CONV_ID = "conv-codex-abc"
CODEX_RESOURCE = {"service.name": "codex-tokenjam"}


def _codex_sse_attrs(**overrides) -> dict:
    base = {
        "conversation.id": CODEX_CONV_ID,
        "model": "gpt-5.4",
        "event.kind": "response.completed",
        "input_token_count": 800,
        "output_token_count": 150,
        "cached_token_count": 200,
        "duration_ms": 900.0,
    }
    base.update(overrides)
    return base


def test_codex_sse_completion_produces_llm_span():
    span = _codex_sse_event_to_span(_codex_sse_attrs(), CODEX_RESOURCE, NOW_NS)
    assert span is not None
    assert span.name == GenAIAttributes.SPAN_LLM_CALL
    assert span.kind == SpanKind.CLIENT
    assert span.status_code == SpanStatus.OK
    assert span.provider == "openai"
    assert span.model == "gpt-5.4"
    assert span.input_tokens == 800
    assert span.output_tokens == 150
    assert span.cache_tokens == 200
    assert span.session_id == CODEX_CONV_ID
    assert span.agent_id == "codex-tokenjam"


def test_codex_sse_non_completion_returns_none():
    span = _codex_sse_event_to_span(
        _codex_sse_attrs(**{"event.kind": "content_block_delta"}), CODEX_RESOURCE, NOW_NS
    )
    assert span is None


def test_codex_sse_reasoning_tokens_in_attributes():
    span = _codex_sse_event_to_span(
        _codex_sse_attrs(**{"reasoning_token_count": 50}), CODEX_RESOURCE, NOW_NS
    )
    assert span is not None
    assert span.attributes.get("reasoning_token_count") == 50


def test_codex_user_prompt_produces_invoke_agent_span():
    attrs = {
        "conversation.id": CODEX_CONV_ID,
        "prompt_length": 25,
        "prompt": "echo hello",
    }
    span = _codex_user_prompt_to_span(attrs, CODEX_RESOURCE, NOW_NS)
    assert span.name == GenAIAttributes.SPAN_INVOKE_AGENT
    assert span.kind == SpanKind.SERVER
    assert span.status_code == SpanStatus.OK
    assert span.session_id == CODEX_CONV_ID
    assert span.attributes.get("prompt_length") == 25
    assert span.attributes.get("prompt") == "echo hello"


def test_codex_tool_decision_produces_internal_span():
    attrs = {
        "conversation.id": CODEX_CONV_ID,
        "tool_name": "shell",
        "call_id": "call-1",
        "decision": "allow",
        "source": "auto",
    }
    span = _codex_tool_decision_to_span(attrs, CODEX_RESOURCE, NOW_NS)
    assert span.name == "tool_decision"
    assert span.kind == SpanKind.INTERNAL
    assert span.tool_name == "shell"
    assert span.attributes.get("decision") == "allow"
    assert span.attributes.get("source") == "auto"
    assert span.attributes.get("call_id") == "call-1"


def test_codex_tool_result_success_produces_ok_span():
    attrs = {
        "conversation.id": CODEX_CONV_ID,
        "tool_name": "shell",
        "success": True,
        "duration_ms": 120.0,
        "arguments": '{"command": "echo hello"}',
        "call_id": "call-1",
    }
    span = _codex_tool_result_to_span(attrs, CODEX_RESOURCE, NOW_NS)
    assert span.name == GenAIAttributes.SPAN_TOOL_CALL
    assert span.kind == SpanKind.INTERNAL
    assert span.status_code == SpanStatus.OK
    assert span.tool_name == "shell"
    assert span.duration_ms == pytest.approx(120.0)
    assert span.status_message is None
    assert span.attributes.get("call_id") == "call-1"


def test_codex_tool_result_failure_produces_error_span():
    attrs = {
        "conversation.id": CODEX_CONV_ID,
        "tool_name": "shell",
        "success": False,
        "error.message": "permission denied",
        "duration_ms": 5.0,
    }
    span = _codex_tool_result_to_span(attrs, CODEX_RESOURCE, NOW_NS)
    assert span.status_code == SpanStatus.ERROR
    assert span.status_message == "permission denied"


def test_codex_api_request_with_error_produces_error_span():
    attrs = {
        "conversation.id": CODEX_CONV_ID,
        "model": "gpt-5.4",
        "error.message": "rate_limit_exceeded",
        "http.response.status_code": 429,
        "attempt": 1,
        "duration_ms": 50.0,
    }
    span = _codex_api_request_to_span(attrs, CODEX_RESOURCE, NOW_NS)
    assert span is not None
    assert span.name == GenAIAttributes.SPAN_LLM_CALL
    assert span.status_code == SpanStatus.ERROR
    assert span.status_message == "rate_limit_exceeded"
    assert span.provider == "openai"
    assert span.attributes.get("http.response.status_code") == 429


def test_codex_api_request_without_error_returns_none():
    attrs = {
        "conversation.id": CODEX_CONV_ID,
        "model": "gpt-5.4",
        "duration_ms": 900.0,
    }
    span = _codex_api_request_to_span(attrs, CODEX_RESOURCE, NOW_NS)
    assert span is None


def test_codex_trace_id_from_conversation_id():
    span = _codex_sse_event_to_span(_codex_sse_attrs(), CODEX_RESOURCE, NOW_NS)
    assert span is not None
    assert span.trace_id == _trace_id_from_session(CODEX_CONV_ID)


def test_codex_missing_conversation_id_defaults_to_unknown():
    attrs = _codex_sse_attrs()
    del attrs["conversation.id"]
    span = _codex_sse_event_to_span(attrs, CODEX_RESOURCE, NOW_NS)
    assert span is not None
    assert span.session_id == "unknown"


# ── parse_log_records: Codex-specific behavior ───────────────────────────

def _make_codex_log_body(event_name: str, attrs_list: list[dict], ts_iso: str) -> dict:
    """Build a minimal OTLP resourceLogs payload as Codex sends it."""
    all_attrs = [
        {"key": "event.name", "value": {"stringValue": event_name}},
        {"key": "event.timestamp", "value": {"stringValue": ts_iso}},
    ] + attrs_list
    return {
        "resourceLogs": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "codex-test"}},
                ],
            },
            "scopeLogs": [{
                "logRecords": [{
                    "timeUnixNano": "0",
                    "body": {},
                    "attributes": all_attrs,
                }],
            }],
        }],
    }


def test_parse_log_records_codex_event_name_from_attrs():
    """Event name in attrs["event.name"] is picked up when body is empty."""
    attrs_list = [
        {"key": "conversation.id", "value": {"stringValue": "conv-xyz"}},
        {"key": "event.kind", "value": {"stringValue": "response.completed"}},
        {"key": "model", "value": {"stringValue": "gpt-5.4"}},
        {"key": "input_token_count", "value": {"intValue": "100"}},
        {"key": "output_token_count", "value": {"intValue": "20"}},
        {"key": "cached_token_count", "value": {"intValue": "0"}},
        {"key": "duration_ms", "value": {"doubleValue": 500.0}},
    ]
    body = _make_codex_log_body("codex.sse_event", attrs_list, "2026-04-24T10:00:00Z")
    pipeline = MagicMock()
    ingested, rejections = parse_log_records(body, pipeline)
    assert ingested == 1
    assert rejections == []
    pipeline.process.assert_called_once()
    span = pipeline.process.call_args[0][0]
    assert span.name == GenAIAttributes.SPAN_LLM_CALL
    assert span.input_tokens == 100


def test_parse_log_records_codex_epoch_timestamp_fallback():
    """timeUnixNano=0 falls back to event.timestamp ISO-8601 attribute."""
    attrs_list = [
        {"key": "conversation.id", "value": {"stringValue": "conv-xyz"}},
        {"key": "event.kind", "value": {"stringValue": "response.completed"}},
        {"key": "model", "value": {"stringValue": "gpt-5.4"}},
        {"key": "input_token_count", "value": {"intValue": "50"}},
        {"key": "output_token_count", "value": {"intValue": "10"}},
        {"key": "cached_token_count", "value": {"intValue": "0"}},
        {"key": "duration_ms", "value": {"doubleValue": 200.0}},
    ]
    body = _make_codex_log_body("codex.sse_event", attrs_list, "2026-04-24T15:30:00Z")
    pipeline = MagicMock()
    parse_log_records(body, pipeline)
    span = pipeline.process.call_args[0][0]
    # Start time must not be epoch; must be 2026-04-24
    assert span.start_time.year == 2026
    assert span.start_time.month == 4
    assert span.start_time.day == 24


def test_parse_log_records_unknown_event_skipped():
    """Unknown event names produce no span and no rejection."""
    body = _make_codex_log_body("codex.unknown_event", [], "2026-04-24T10:00:00Z")
    pipeline = MagicMock()
    ingested, rejections = parse_log_records(body, pipeline)
    assert ingested == 0
    assert rejections == []
    pipeline.process.assert_not_called()


def test_parse_log_records_extracts_service_namespace():
    """service.namespace on the resource is stamped onto every synthesized span."""
    record = make_claude_code_api_request_log(session_id="s-ns")
    body = {
        "resourceLogs": [{
            "resource": {"attributes": [
                {"key": "service.name", "value": {"stringValue": "claude-code-harness"}},
                {"key": "service.namespace", "value": {"stringValue": "aquanode"}},
            ]},
            "scopeLogs": [{"logRecords": [record]}],
        }],
    }
    pipeline = MagicMock()
    parse_log_records(body, pipeline)

    pipeline.process.assert_called_once()
    span = pipeline.process.call_args[0][0]
    assert span.service_namespace == "aquanode"
    assert span.agent_id == "claude-code-harness"
