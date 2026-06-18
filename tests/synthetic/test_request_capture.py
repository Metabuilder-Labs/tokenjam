"""Full-request capture tests (issue #209).

Covers the four acceptance criteria:
  1. LLM spans capture sampling params + the tools/tool_choice payload under the
     existing [capture] gating (honored on/off like message content; no behavior
     change when off).
  2. The provider-patch / LiteLLM capture helper maps call kwargs to the right
     semconv attributes.
  3. The captured fields round-trip NormalizedSpan -> DB -> back.
  4. Spans are built via tests/factories.

The capture flow is: integration sets OTel attributes (record_full_request) ->
strip_captured_content gates them at ingest -> extract_request_capture projects
the surviving attributes into the structured request_params / request_tools
fields -> DuckDB JSON columns round-trip them back.
"""
from __future__ import annotations

import json

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import (
    IngestPipeline,
    extract_request_capture,
    strip_captured_content,
)
from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tokenjam.sdk.integrations._request_capture import (
    record_full_request,
    record_full_request_bedrock,
    record_full_request_gemini,
)
from tests.factories import make_llm_span, make_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSpan:
    """Minimal OTel-span stand-in that records set_attribute calls."""

    def __init__(self) -> None:
        self.attrs: dict = {}

    def set_attribute(self, key, value) -> None:
        self.attrs[key] = value


def _pipeline(capture: CaptureConfig) -> tuple[IngestPipeline, InMemoryBackend]:
    db = InMemoryBackend()
    config = TjConfig(version="1", capture=capture)
    return IngestPipeline(db=db, config=config), db


def _request_span(session_id: str = "s1"):
    """A factory span whose attributes carry the OTel request-capture keys, as a
    provider patch would set them before ingest."""
    return make_llm_span(
        session_id=session_id,
        extra_attributes={
            GenAIAttributes.REQUEST_TEMPERATURE: 0.7,
            GenAIAttributes.REQUEST_MAX_TOKENS: 1024,
            GenAIAttributes.REQUEST_STOP_SEQUENCES: ["STOP"],
            TjAttributes.REQUEST_TOOLS: json.dumps(
                {"tools": [{"name": "get_weather"}], "tool_choice": "auto"}
            ),
        },
    )


# ===========================================================================
# strip_captured_content — the single gate (acceptance #1)
# ===========================================================================

class TestStripGate:

    def test_sampling_params_stripped_when_prompts_off(self):
        attrs = {
            GenAIAttributes.REQUEST_TEMPERATURE: 0.7,
            GenAIAttributes.REQUEST_MAX_TOKENS: 1024,
        }
        stripped = strip_captured_content(attrs, CaptureConfig(prompts=False))
        assert GenAIAttributes.REQUEST_TEMPERATURE not in stripped
        assert GenAIAttributes.REQUEST_MAX_TOKENS not in stripped

    def test_sampling_params_kept_when_prompts_on(self):
        attrs = {GenAIAttributes.REQUEST_TEMPERATURE: 0.7}
        stripped = strip_captured_content(attrs, CaptureConfig(prompts=True))
        assert stripped[GenAIAttributes.REQUEST_TEMPERATURE] == 0.7

    def test_tools_stripped_when_tool_inputs_off(self):
        attrs = {TjAttributes.REQUEST_TOOLS: json.dumps({"tools": []})}
        stripped = strip_captured_content(attrs, CaptureConfig(tool_inputs=False))
        assert TjAttributes.REQUEST_TOOLS not in stripped

    def test_tools_kept_when_tool_inputs_on(self):
        payload = json.dumps({"tools": [{"name": "t"}]})
        attrs = {TjAttributes.REQUEST_TOOLS: payload}
        stripped = strip_captured_content(attrs, CaptureConfig(tool_inputs=True))
        assert stripped[TjAttributes.REQUEST_TOOLS] == payload

    def test_params_and_tools_gate_independently(self):
        # prompts on but tool_inputs off -> params kept, tools dropped.
        attrs = {
            GenAIAttributes.REQUEST_TEMPERATURE: 0.7,
            TjAttributes.REQUEST_TOOLS: json.dumps({"tools": []}),
        }
        stripped = strip_captured_content(
            attrs, CaptureConfig(prompts=True, tool_inputs=False)
        )
        assert GenAIAttributes.REQUEST_TEMPERATURE in stripped
        assert TjAttributes.REQUEST_TOOLS not in stripped


# ===========================================================================
# extract_request_capture — projection into structured fields
# ===========================================================================

class TestExtractRequestCapture:

    def test_projects_params_and_tools(self):
        span = _request_span()
        extract_request_capture(span)
        assert span.request_params == {
            "temperature": 0.7,
            "max_tokens": 1024,
            "stop_sequences": ["STOP"],
        }
        assert span.request_tools == {
            "tools": [{"name": "get_weather"}],
            "tool_choice": "auto",
        }
        # The keys are popped out of the attributes blob (single home).
        assert GenAIAttributes.REQUEST_TEMPERATURE not in span.attributes
        assert TjAttributes.REQUEST_TOOLS not in span.attributes

    def test_no_request_attrs_leaves_fields_none(self):
        span = make_llm_span()
        extract_request_capture(span)
        assert span.request_params is None
        assert span.request_tools is None


# ===========================================================================
# Pipeline gating end-to-end + DB round-trip (acceptance #1 + #3)
# ===========================================================================

class TestPipelineGating:

    def test_captured_when_toggles_on(self):
        pipeline, db = _pipeline(CaptureConfig(prompts=True, tool_inputs=True))
        pipeline.process(_request_span())

        stored = db.get_recent_spans("s1", 1)[0]
        assert stored.request_params == {
            "temperature": 0.7,
            "max_tokens": 1024,
            "stop_sequences": ["STOP"],
        }
        assert stored.request_tools == {
            "tools": [{"name": "get_weather"}],
            "tool_choice": "auto",
        }
        db.close()

    def test_no_behavior_change_when_capture_off(self):
        # Acceptance #1: default capture (all off) -> nothing captured, and the
        # raw attributes are gone, exactly like message content.
        pipeline, db = _pipeline(CaptureConfig())
        pipeline.process(_request_span())

        stored = db.get_recent_spans("s1", 1)[0]
        assert stored.request_params is None
        assert stored.request_tools is None
        assert GenAIAttributes.REQUEST_TEMPERATURE not in stored.attributes
        assert TjAttributes.REQUEST_TOOLS not in stored.attributes
        db.close()

    def test_params_captured_but_tools_gated_off(self):
        pipeline, db = _pipeline(CaptureConfig(prompts=True, tool_inputs=False))
        pipeline.process(_request_span())

        stored = db.get_recent_spans("s1", 1)[0]
        assert stored.request_params is not None
        assert stored.request_tools is None
        db.close()


# ===========================================================================
# DB round-trip via factory fields (acceptance #3 + #4)
# ===========================================================================

class TestDbRoundTrip:

    def test_request_capture_roundtrips(self):
        db = InMemoryBackend()
        db.upsert_session(make_session(session_id="s1"))
        params = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 2048, "stop_sequences": ["END"]}
        tools = {
            "tools": [{"name": "calc", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto"},
        }
        db.insert_span(make_llm_span(session_id="s1", request_params=params, request_tools=tools))

        got = db.get_recent_spans("s1", 1)[0]
        assert got.request_params == params
        assert got.request_tools == tools
        db.close()

    def test_null_request_capture_roundtrips_as_none(self):
        db = InMemoryBackend()
        db.upsert_session(make_session(session_id="s1"))
        db.insert_span(make_llm_span(session_id="s1"))

        got = db.get_recent_spans("s1", 1)[0]
        assert got.request_params is None
        assert got.request_tools is None
        db.close()


# ===========================================================================
# Integration-side capture helper (acceptance #2)
# ===========================================================================

class TestRecordFullRequest:

    def test_maps_openai_style_kwargs(self):
        span = _FakeSpan()
        record_full_request(span, {
            "model": "gpt-4o",
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 512,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "seed": 42,
            "stop": ["X"],
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "tool_choice": "auto",
        })
        assert span.attrs[GenAIAttributes.REQUEST_TEMPERATURE] == 0.5
        assert span.attrs[GenAIAttributes.REQUEST_TOP_P] == 0.9
        assert span.attrs[GenAIAttributes.REQUEST_MAX_TOKENS] == 512
        assert span.attrs[GenAIAttributes.REQUEST_FREQUENCY_PENALTY] == 0.1
        assert span.attrs[GenAIAttributes.REQUEST_PRESENCE_PENALTY] == 0.2
        assert span.attrs[GenAIAttributes.REQUEST_SEED] == 42
        assert span.attrs[GenAIAttributes.REQUEST_STOP_SEQUENCES] == ["X"]
        tools = json.loads(span.attrs[TjAttributes.REQUEST_TOOLS])
        assert tools["tool_choice"] == "auto"
        assert tools["tools"][0]["function"]["name"] == "t"

    def test_max_completion_tokens_alias(self):
        span = _FakeSpan()
        record_full_request(span, {"max_completion_tokens": 256})
        assert span.attrs[GenAIAttributes.REQUEST_MAX_TOKENS] == 256

    def test_nothing_set_without_request_data(self):
        span = _FakeSpan()
        record_full_request(span, {"model": "gpt-4o", "messages": []})
        assert span.attrs == {}

    def test_tool_choice_only_still_captured(self):
        span = _FakeSpan()
        record_full_request(span, {"tool_choice": "none"})
        assert json.loads(span.attrs[TjAttributes.REQUEST_TOOLS]) == {"tool_choice": "none"}

    def test_gemini_generation_config_dict_flattened(self):
        span = _FakeSpan()
        record_full_request_gemini(span, {
            "generation_config": {"temperature": 0.3, "top_k": 40, "max_output_tokens": 800},
            "tools": [{"name": "search"}],
        })
        assert span.attrs[GenAIAttributes.REQUEST_TEMPERATURE] == 0.3
        assert span.attrs[GenAIAttributes.REQUEST_TOP_K] == 40
        assert span.attrs[GenAIAttributes.REQUEST_MAX_TOKENS] == 800
        assert json.loads(span.attrs[TjAttributes.REQUEST_TOOLS])["tools"] == [{"name": "search"}]

    def test_gemini_generation_config_object_flattened(self):
        class _GenConfig:
            temperature = 0.9
            top_p = 0.8
            max_output_tokens = 100

        span = _FakeSpan()
        record_full_request_gemini(span, {"generation_config": _GenConfig()})
        assert span.attrs[GenAIAttributes.REQUEST_TEMPERATURE] == 0.9
        assert span.attrs[GenAIAttributes.REQUEST_TOP_P] == 0.8
        assert span.attrs[GenAIAttributes.REQUEST_MAX_TOKENS] == 100

    def test_bedrock_json_body_parsed(self):
        span = _FakeSpan()
        body = json.dumps({
            "max_tokens": 1000,
            "temperature": 0.4,
            "top_p": 0.7,
            "stop_sequences": ["\n\nHuman:"],
            "tools": [{"name": "lookup"}],
        })
        record_full_request_bedrock(span, {"modelId": "anthropic.claude-3", "body": body})
        assert span.attrs[GenAIAttributes.REQUEST_MAX_TOKENS] == 1000
        assert span.attrs[GenAIAttributes.REQUEST_TEMPERATURE] == 0.4
        assert span.attrs[GenAIAttributes.REQUEST_STOP_SEQUENCES] == ["\n\nHuman:"]
        assert json.loads(span.attrs[TjAttributes.REQUEST_TOOLS])["tools"] == [{"name": "lookup"}]

    def test_bedrock_no_body_is_noop(self):
        span = _FakeSpan()
        record_full_request_bedrock(span, {"modelId": "x"})
        assert span.attrs == {}

    def test_bool_value_preserved_not_coerced_to_int(self):
        # A bool must not be silently treated as an int by the coercion path.
        span = _FakeSpan()
        record_full_request(span, {"seed": True})
        assert span.attrs[GenAIAttributes.REQUEST_SEED] is True
