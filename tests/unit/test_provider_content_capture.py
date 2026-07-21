"""Prompt + completion content capture on the provider monkey-patches (#320).

Before this, only litellm.py and sdk/agent.py set gen_ai.prompt.content /
gen_ai.completion.content; the anthropic/openai/gemini/bedrock provider patches
captured #209 request_params but NOT the messages or the completion text — so
their spans weren't self-contained enough to replay.

These tests verify, per provider:
  - the LLM span carries PROMPT_CONTENT == json.dumps(messages) and the
    COMPLETION_CONTENT, set UNCONDITIONALLY by the patch;
  - with [capture] prompts/completions OFF, both are stripped at the single
    ingest gate (strip_captured_content) — same path litellm uses.

anthropic + openai are exercised through the real installed patch (with a fake
upstream method); gemini + bedrock are exercised through the exact capture calls
their patches make (their SDKs aren't installed in CI).
"""
from __future__ import annotations

import json
import types

import pytest

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.sdk.integrations._request_capture import (
    extract_anthropic_completion,
    extract_gemini_completion,
    extract_openai_completion,
    record_completion_content,
    record_prompt_content,
)
from tests.factories import make_llm_span, make_session


# ---------------------------------------------------------------------------
# A recording fake span (mirrors the litellm test's mock tracer span).
# ---------------------------------------------------------------------------

class _FakeSpan:
    """Minimal OTel-span stand-in recording set_attribute; the rest are no-ops
    so a full patch closure (which also calls set_status/end) can run."""

    def __init__(self) -> None:
        self.attributes: dict = {}

    def set_attribute(self, key, value) -> None:
        self.attributes[key] = value

    def set_status(self, *_a, **_k) -> None:
        pass

    def end(self, *_a, **_k) -> None:
        pass

    def is_recording(self) -> bool:
        return True


def _recording_tracer():
    """A tracer whose start_span returns a _FakeSpan recording set_attribute."""
    spans: list[_FakeSpan] = []

    class _Tracer:
        def start_span(self, _name):
            span = _FakeSpan()
            spans.append(span)
            return span

    return _Tracer(), spans


# ===========================================================================
# Shared helpers — the single serialization point (#320)
# ===========================================================================

class TestSharedHelpers:

    def test_prompt_content_is_json_dumps_messages(self):
        # CRITICAL: must match litellm's shape so a replay harness / backfill
        # reading gen_ai.prompt.content gets ONE consistent serialization.
        span = _FakeSpan()
        messages = [{"role": "user", "content": "hi"}]
        record_prompt_content(span, messages)
        assert span.attributes[GenAIAttributes.PROMPT_CONTENT] == json.dumps(messages)

    def test_prompt_content_none_is_noop(self):
        span = _FakeSpan()
        record_prompt_content(span, None)
        assert GenAIAttributes.PROMPT_CONTENT not in span.attributes

    def test_completion_content_is_the_text(self):
        span = _FakeSpan()
        record_completion_content(span, "the answer")
        assert span.attributes[GenAIAttributes.COMPLETION_CONTENT] == "the answer"

    def test_completion_content_none_is_noop(self):
        span = _FakeSpan()
        record_completion_content(span, None)
        assert GenAIAttributes.COMPLETION_CONTENT not in span.attributes

    def test_prompt_content_non_serialisable_falls_back_to_str(self):
        span = _FakeSpan()

        class _Weird:
            def __repr__(self) -> str:
                return "<weird>"

        # default=str keeps json.dumps from raising on an exotic object.
        record_prompt_content(span, [{"x": _Weird()}])
        assert GenAIAttributes.PROMPT_CONTENT in span.attributes


# ===========================================================================
# Per-provider completion-text extraction (SDK-independent)
# ===========================================================================

class TestCompletionExtraction:

    def test_anthropic_joins_text_blocks_and_skips_tool_use(self):
        resp = types.SimpleNamespace(content=[
            types.SimpleNamespace(type="text", text="Hello "),
            types.SimpleNamespace(type="tool_use", id="t1"),  # no .text
            types.SimpleNamespace(type="text", text="world"),
        ])
        assert extract_anthropic_completion(resp) == "Hello world"

    def test_anthropic_tool_only_response_is_none(self):
        resp = types.SimpleNamespace(content=[types.SimpleNamespace(type="tool_use")])
        assert extract_anthropic_completion(resp) is None

    def test_openai_reads_first_choice_message_content(self):
        resp = types.SimpleNamespace(choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content="the answer")),
        ])
        assert extract_openai_completion(resp) == "the answer"

    def test_openai_empty_choices_is_none(self):
        assert extract_openai_completion(types.SimpleNamespace(choices=[])) is None

    def test_gemini_reads_text(self):
        assert extract_gemini_completion(types.SimpleNamespace(text="gen text")) == "gen text"


# ===========================================================================
# Anthropic patch — full closure with a fake upstream (#320)
# ===========================================================================

class TestAnthropicPatch:

    def test_patch_captures_prompt_and_completion(self):
        pytest.importorskip("anthropic")
        from anthropic.resources import Messages

        from tokenjam.sdk.integrations.anthropic import AnthropicIntegration

        tracer, spans = _recording_tracer()
        integ = AnthropicIntegration()
        integ.install(tracer)
        try:
            messages = [{"role": "user", "content": "what is 2+2?"}]
            fake_resp = types.SimpleNamespace(
                usage=types.SimpleNamespace(input_tokens=10, output_tokens=5),
                content=[types.SimpleNamespace(type="text", text="4")],
            )
            integ._original_create = lambda _self, *a, **kw: fake_resp
            Messages.create("self", model="claude-haiku-4-5", messages=messages)

            attrs = spans[0].attributes
            assert attrs[GenAIAttributes.PROMPT_CONTENT] == json.dumps(messages)
            assert attrs[GenAIAttributes.COMPLETION_CONTENT] == "4"
        finally:
            integ.uninstall()


# ===========================================================================
# OpenAI patch — full closure with a fake upstream (#320)
# ===========================================================================

class TestOpenAIPatch:

    def test_patch_captures_prompt_and_completion(self):
        pytest.importorskip("openai")
        from openai.resources.chat.completions import Completions

        from tokenjam.sdk.integrations.openai import OpenAIIntegration

        tracer, spans = _recording_tracer()
        integ = OpenAIIntegration()
        integ.install(tracer)
        try:
            messages = [{"role": "user", "content": "ping"}]
            fake_resp = types.SimpleNamespace(
                usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=1),
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="pong"))],
            )
            integ._original_create = lambda _self, *a, **kw: fake_resp
            Completions.create("self", model="gpt-4o-mini", messages=messages)

            attrs = spans[0].attributes
            assert attrs[GenAIAttributes.PROMPT_CONTENT] == json.dumps(messages)
            assert attrs[GenAIAttributes.COMPLETION_CONTENT] == "pong"
        finally:
            integ.uninstall()


# ===========================================================================
# Gemini patch — the exact capture calls the patch makes (SDK not in CI)
# ===========================================================================

class TestGeminiCapture:

    def test_contents_and_completion_captured(self):
        from tokenjam.sdk.integrations.gemini import _gemini_contents

        span = _FakeSpan()
        contents = [{"role": "user", "parts": ["hi"]}]
        # As the patched_generate does: prompt from contents, completion from .text.
        record_prompt_content(span, _gemini_contents((contents,), {}))
        record_completion_content(
            span, extract_gemini_completion(types.SimpleNamespace(text="hi back")),
        )
        assert span.attributes[GenAIAttributes.PROMPT_CONTENT] == json.dumps(contents)
        assert span.attributes[GenAIAttributes.COMPLETION_CONTENT] == "hi back"

    def test_contents_from_kwarg(self):
        from tokenjam.sdk.integrations.gemini import _gemini_contents

        contents = "just a string prompt"
        assert _gemini_contents((), {"contents": contents}) == contents


# ===========================================================================
# Bedrock patch — request from body, completion via _extract_bedrock_usage
# ===========================================================================

class TestBedrockCapture:

    def test_request_messages_and_completion_from_body(self):
        from tokenjam.sdk.integrations.bedrock import (
            _bedrock_request_messages,
            _extract_bedrock_usage,
        )

        messages = [{"role": "user", "content": "hello"}]
        body = json.dumps({"messages": messages, "max_tokens": 100})
        span = _FakeSpan()
        # Prompt, exactly as the patch does it.
        record_prompt_content(span, _bedrock_request_messages({"body": body}))
        assert span.attributes[GenAIAttributes.PROMPT_CONTENT] == json.dumps(messages)

        # Completion: _extract_bedrock_usage parses the response body once and
        # sets both usage and completion (Anthropic-on-Bedrock content blocks).
        response = {"body": json.dumps({
            "content": [{"type": "text", "text": "hi from bedrock"}],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        })}
        _extract_bedrock_usage(response, span)
        assert span.attributes[GenAIAttributes.COMPLETION_CONTENT] == "hi from bedrock"
        assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 7

    def test_completion_text_scalar_schema(self):
        from tokenjam.sdk.integrations.bedrock import _bedrock_completion_text

        assert _bedrock_completion_text({"completion": "older claude"}) == "older claude"
        assert _bedrock_completion_text(
            {"results": [{"outputText": "titan text"}]}) == "titan text"


# ===========================================================================
# Ingest gate — both stripped when capture off, kept when on (#320)
# ===========================================================================

class TestIngestGate:

    def _span_with_content(self, session_id="s1"):
        messages = [{"role": "user", "content": "secret prompt"}]
        return make_llm_span(
            session_id=session_id,
            extra_attributes={
                GenAIAttributes.PROMPT_CONTENT: json.dumps(messages),
                GenAIAttributes.COMPLETION_CONTENT: "secret completion",
            },
        )

    def _pipeline(self, capture):
        db = InMemoryBackend()
        db.upsert_session(make_session(session_id="s1"))
        return IngestPipeline(db=db, config=TjConfig(version="1", capture=capture)), db

    def test_stripped_when_capture_off(self):
        # Every toggle explicitly off — content must be gone from the stored
        # span, exactly like the litellm path.
        pipeline, db = self._pipeline(CaptureConfig(prompts=False))
        pipeline.process(self._span_with_content())
        stored = db.get_recent_spans("s1", 1)[0]
        assert GenAIAttributes.PROMPT_CONTENT not in stored.attributes
        assert GenAIAttributes.COMPLETION_CONTENT not in stored.attributes
        db.close()

    def test_default_capture_keeps_prompt_strips_completion(self):
        # `prompts` defaults on (E33); `completions` stays off.
        pipeline, db = self._pipeline(CaptureConfig())
        pipeline.process(self._span_with_content())
        stored = db.get_recent_spans("s1", 1)[0]
        assert "secret prompt" in stored.attributes[GenAIAttributes.PROMPT_CONTENT]
        assert GenAIAttributes.COMPLETION_CONTENT not in stored.attributes
        db.close()

    def test_kept_when_capture_on(self):
        pipeline, db = self._pipeline(CaptureConfig(prompts=True, completions=True))
        pipeline.process(self._span_with_content())
        stored = db.get_recent_spans("s1", 1)[0]
        assert "secret prompt" in stored.attributes[GenAIAttributes.PROMPT_CONTENT]
        assert stored.attributes[GenAIAttributes.COMPLETION_CONTENT] == "secret completion"
        db.close()

    def test_gate_independently_prompts_on_completions_off(self):
        pipeline, db = self._pipeline(CaptureConfig(prompts=True, completions=False))
        pipeline.process(self._span_with_content())
        stored = db.get_recent_spans("s1", 1)[0]
        assert GenAIAttributes.PROMPT_CONTENT in stored.attributes
        assert GenAIAttributes.COMPLETION_CONTENT not in stored.attributes
        db.close()
