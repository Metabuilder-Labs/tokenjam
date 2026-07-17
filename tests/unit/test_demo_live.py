"""Unit tests for the `tj demo --live` replay sink (tokenjam.demo.live)."""
from __future__ import annotations

import json

from tokenjam.core.models import NormalizedSpan, SpanKind, SpanStatus
from tokenjam.demo.live import (
    LiveReplayError,
    LiveSink,
    _build_payload,
    _span_to_otlp,
    build_sink,
    check_serve_alive,
)
from tokenjam.otel.otlp_parsing import parse_otlp_span
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.utils.ids import new_span_id, new_trace_id, new_uuid
from tokenjam.utils.time_parse import utcnow


def _tool_span() -> NormalizedSpan:
    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=new_trace_id(),
        name="gen_ai.tool.call",
        kind=SpanKind.INTERNAL,
        status_code=SpanStatus.ERROR,
        start_time=utcnow(),
        agent_id="demo-retry-loop",
        conversation_id=new_uuid(),
        tool_name="search_knowledge_base",
        status_message="connection timeout",
        attributes={GenAIAttributes.TOOL_INPUT: json.dumps({"query": "x"})},
    )


def _llm_span() -> NormalizedSpan:
    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=new_trace_id(),
        name="gen_ai.llm.call",
        kind=SpanKind.CLIENT,
        status_code=SpanStatus.OK,
        start_time=utcnow(),
        agent_id="demo-surprise-cost",
        conversation_id=new_uuid(),
        provider="anthropic",
        model="claude-opus-4-6",
        input_tokens=40_000,
        output_tokens=6_000,
    )


def test_span_to_otlp_round_trips_llm_fields():
    span = _llm_span()
    parsed = parse_otlp_span(_span_to_otlp(span), {})
    assert parsed.agent_id == "demo-surprise-cost"
    assert parsed.provider == "anthropic"
    assert parsed.model == "claude-opus-4-6"
    assert parsed.input_tokens == 40_000
    assert parsed.output_tokens == 6_000
    assert parsed.kind == SpanKind.CLIENT
    assert parsed.status_code == SpanStatus.OK
    assert parsed.conversation_id == span.conversation_id
    assert parsed.trace_id == span.trace_id
    assert parsed.span_id == span.span_id


def test_span_to_otlp_preserves_tool_input_and_error():
    span = _tool_span()
    parsed = parse_otlp_span(_span_to_otlp(span), {})
    assert parsed.tool_name == "search_knowledge_base"
    assert parsed.status_code == SpanStatus.ERROR
    assert parsed.status_message == "connection timeout"
    # The raw tool input must survive — it's what makes retry-loop detection fire.
    assert parsed.attributes.get(GenAIAttributes.TOOL_INPUT) == json.dumps({"query": "x"})


def test_build_payload_shape():
    payload = _build_payload([_span_to_otlp(_llm_span())])
    assert "resourceSpans" in payload
    spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1
    assert spans[0]["name"] == "gen_ai.llm.call"


def test_live_sink_attaches_bearer_only_with_secret():
    with_secret = LiveSink("http://x/api/v1/spans", "s3cret")
    assert with_secret._headers.get("Authorization") == "Bearer s3cret"
    # Empty / whitespace secret must NOT produce an illegal "Bearer " header.
    for blank in ("", "   ", None):
        assert "Authorization" not in LiveSink("http://x/api/v1/spans", blank)._headers


def test_live_sink_buffers_and_flushes(monkeypatch):
    captured = {}

    class _Resp:
        status_code = 200
        content = b"{}"

        def json(self):
            return {"ingested": 2, "rejected": 0, "rejections": []}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr("tokenjam.demo.live.httpx.post", fake_post)

    sink = LiveSink("http://serve/api/v1/spans", "")
    sink.record(_llm_span())
    sink.record(_tool_span())
    assert sink.pending == 2

    result = sink.flush()
    assert result.sent == 2
    assert result.ingested == 2
    assert sink.pending == 0  # buffer cleared after flush
    assert captured["url"] == "http://serve/api/v1/spans"
    assert len(captured["payload"]["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 2


def test_live_sink_flush_raises_on_401(monkeypatch):
    class _Resp:
        status_code = 401
        content = b""
        text = ""

    monkeypatch.setattr(
        "tokenjam.demo.live.httpx.post",
        lambda url, json=None, headers=None, timeout=None: _Resp(),
    )
    sink = LiveSink("http://serve/api/v1/spans", "wrong")
    sink.record(_llm_span())
    try:
        sink.flush()
        assert False, "expected LiveReplayError"
    except LiveReplayError as exc:
        assert "401" in str(exc)


def test_live_sink_flush_raises_on_connect_error(monkeypatch):
    import httpx

    def boom(url, json=None, headers=None, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("tokenjam.demo.live.httpx.post", boom)
    sink = LiveSink("http://serve/api/v1/spans", "")
    sink.record(_llm_span())
    try:
        sink.flush()
        assert False, "expected LiveReplayError"
    except LiveReplayError as exc:
        assert "could not reach" in str(exc)


def test_empty_flush_is_noop():
    sink = LiveSink("http://serve/api/v1/spans", "")
    result = sink.flush()
    assert result.sent == 0
    assert result.ingested == 0


def test_check_serve_alive_true_on_200_and_401(monkeypatch):
    from tokenjam.core.config import TjConfig

    config = TjConfig(version="1")

    class _Resp:
        def __init__(self, code):
            self.status_code = code

    for code in (200, 401):
        monkeypatch.setattr(
            "tokenjam.demo.live.httpx.get",
            lambda url, timeout=None, _c=code: _Resp(_c),
        )
        assert check_serve_alive(config) is True

    import httpx

    def boom(url, timeout=None):
        raise httpx.ConnectError("no daemon")

    monkeypatch.setattr("tokenjam.demo.live.httpx.get", boom)
    assert check_serve_alive(config) is False


def test_build_sink_uses_config_endpoint_and_secret():
    from tokenjam.core.config import TjConfig, SecurityConfig

    config = TjConfig(version="1", security=SecurityConfig(ingest_secret="abc"))
    sink = build_sink(config)
    assert sink.endpoint.endswith("/api/v1/spans")
    assert sink._headers.get("Authorization") == "Bearer abc"
