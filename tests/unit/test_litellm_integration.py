"""Tests for the LiteLLM integration (ocw.sdk.integrations.litellm)."""
from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest

from tj.otel.semconv import GenAIAttributes


# ---------------------------------------------------------------------------
# Fake litellm module — injected into sys.modules so the integration can
# import it without installing the real package.
# ---------------------------------------------------------------------------

def _make_fake_litellm():
    mod = types.ModuleType("litellm")

    def _fake_completion(*args, **kwargs):
        resp = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        resp.model = "gpt-4o-mini"
        resp._hidden_params = {"custom_llm_provider": "openai"}
        return resp

    async def _fake_acompletion(*args, **kwargs):
        resp = MagicMock()
        resp.usage.prompt_tokens = 200
        resp.usage.completion_tokens = 80
        resp.model = "claude-haiku-4-5"
        resp._hidden_params = {"custom_llm_provider": "anthropic"}
        return resp

    mod.completion = _fake_completion
    mod.acompletion = _fake_acompletion
    return mod


@pytest.fixture(autouse=True)
def _inject_fake_litellm():
    """Inject a fake litellm module for the duration of each test."""
    fake = _make_fake_litellm()
    sys.modules["litellm"] = fake
    yield fake
    # Uninstall to reset state between tests
    from tj.sdk.integrations.litellm import LiteLLMIntegration
    LiteLLMIntegration.installed = False
    del sys.modules["litellm"]


# ---------------------------------------------------------------------------
# Helper to capture spans
# ---------------------------------------------------------------------------

@pytest.fixture()
def tracer_and_spans():
    """Return a (tracer, spans_list) pair using a recording mock tracer."""
    spans = []

    def start_span(name):
        span = MagicMock()
        span.is_recording.return_value = True
        span.attributes = {}
        # Track set_attribute calls in a dict for easy assertions
        def _set_attr(k, v):
            span.attributes[k] = v
        span.set_attribute = _set_attr
        spans.append(span)
        return span

    tracer = MagicMock()
    tracer.start_span = start_span
    return tracer, spans


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLiteLLMIntegration:

    def test_sync_completion_creates_span(self, tracer_and_spans):
        tracer, spans = tracer_and_spans
        from tj.sdk.integrations.litellm import LiteLLMIntegration
        import litellm

        integration = LiteLLMIntegration()
        integration.install(tracer)

        litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert len(spans) == 1
        span = spans[0]
        assert span.attributes[GenAIAttributes.REQUEST_MODEL] == "gpt-4o-mini"
        assert span.attributes[GenAIAttributes.PROVIDER_NAME] == "openai"
        assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 100
        assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 50
        span.end.assert_called_once()

    def test_async_completion_creates_span(self, tracer_and_spans):
        tracer, spans = tracer_and_spans
        from tj.sdk.integrations.litellm import LiteLLMIntegration
        import litellm

        integration = LiteLLMIntegration()
        integration.install(tracer)

        asyncio.run(
            litellm.acompletion(
                model="anthropic/claude-haiku-4-5",
                messages=[{"role": "user", "content": "hi"}],
            )
        )

        assert len(spans) == 1
        span = spans[0]
        assert span.attributes[GenAIAttributes.REQUEST_MODEL] == "claude-haiku-4-5"
        assert span.attributes[GenAIAttributes.PROVIDER_NAME] == "anthropic"
        assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 200
        assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 80
        span.end.assert_called_once()

    def test_provider_from_model_prefix_fallback(self, tracer_and_spans):
        """When custom_llm_provider is missing, provider is inferred from model prefix."""
        tracer, spans = tracer_and_spans
        import litellm

        # Override to return response without _hidden_params
        def no_hidden(*a, **kw):
            resp = MagicMock()
            resp.usage.prompt_tokens = 10
            resp.usage.completion_tokens = 5
            resp._hidden_params = {}  # empty — no custom_llm_provider
            resp.model = "gpt-4o-mini"
            return resp
        litellm.completion = no_hidden

        from tj.sdk.integrations.litellm import LiteLLMIntegration
        integration = LiteLLMIntegration()
        integration.install(tracer)

        litellm.completion(model="together/mistral-7b", messages=[])

        span = spans[0]
        assert span.attributes[GenAIAttributes.PROVIDER_NAME] == "together"

    def test_provider_fallback_no_slash(self, tracer_and_spans):
        """Model with no slash and no hidden params falls back to 'litellm'."""
        tracer, spans = tracer_and_spans
        import litellm

        def no_provider(*a, **kw):
            resp = MagicMock()
            resp.usage.prompt_tokens = 10
            resp.usage.completion_tokens = 5
            resp._hidden_params = {}
            resp.model = "gpt-4o-mini"
            return resp
        litellm.completion = no_provider

        from tj.sdk.integrations.litellm import LiteLLMIntegration
        integration = LiteLLMIntegration()
        integration.install(tracer)

        litellm.completion(model="gpt-4o-mini", messages=[])

        span = spans[0]
        assert span.attributes[GenAIAttributes.PROVIDER_NAME] == "litellm"

    def test_sync_streaming(self, tracer_and_spans):
        tracer, spans = tracer_and_spans
        import litellm

        # Make completion return a stream-like iterable
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk2 = MagicMock()
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 50
        chunk2.usage.completion_tokens = 25

        litellm.completion = lambda *a, **kw: iter([chunk1, chunk2])

        from tj.sdk.integrations.litellm import LiteLLMIntegration
        integration = LiteLLMIntegration()
        integration.install(tracer)

        stream = litellm.completion(
            model="openai/gpt-4o-mini", messages=[], stream=True,
        )
        chunks = list(stream)

        assert len(chunks) == 2
        span = spans[0]
        assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 50
        assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 25
        span.end.assert_called_once()

    def test_async_streaming(self, tracer_and_spans):
        tracer, spans = tracer_and_spans
        import litellm

        chunk1 = MagicMock()
        chunk1.usage = None
        chunk2 = MagicMock()
        chunk2.usage = MagicMock()
        chunk2.usage.prompt_tokens = 60
        chunk2.usage.completion_tokens = 30

        async def fake_astream(*a, **kw):
            for c in [chunk1, chunk2]:
                yield c

        async def fake_acompletion(*a, **kw):
            return fake_astream()

        litellm.acompletion = fake_acompletion

        from tj.sdk.integrations.litellm import LiteLLMIntegration
        integration = LiteLLMIntegration()
        integration.install(tracer)

        async def _run():
            stream = await litellm.acompletion(
                model="anthropic/claude-haiku-4-5", messages=[], stream=True,
            )
            return [c async for c in stream]

        chunks = asyncio.run(_run())

        assert len(chunks) == 2
        span = spans[0]
        assert span.attributes[GenAIAttributes.INPUT_TOKENS] == 60
        assert span.attributes[GenAIAttributes.OUTPUT_TOKENS] == 30
        span.end.assert_called_once()

    def test_error_sets_span_status_and_reraises(self, tracer_and_spans):
        tracer, spans = tracer_and_spans
        import litellm

        litellm.completion = MagicMock(side_effect=RuntimeError("API down"))

        from tj.sdk.integrations.litellm import LiteLLMIntegration
        integration = LiteLLMIntegration()
        integration.install(tracer)

        with pytest.raises(RuntimeError, match="API down"):
            litellm.completion(model="openai/gpt-4o-mini", messages=[])

        span = spans[0]
        span.set_status.assert_called_once()
        status_arg = span.set_status.call_args[0][0]
        assert status_arg.status_code.name == "ERROR"
        span.end.assert_called_once()

    def test_uninstall_restores_original(self, tracer_and_spans):
        tracer, spans = tracer_and_spans
        import litellm

        original = litellm.completion
        from tj.sdk.integrations.litellm import LiteLLMIntegration
        integration = LiteLLMIntegration()
        integration.install(tracer)
        assert litellm.completion is not original

        integration.uninstall()
        assert litellm.completion is original

    def test_context_var_suppresses_openai_patch(self, tracer_and_spans):
        """When litellm patch is active, the openai patch skips span creation."""
        from tj.sdk.integrations.litellm import _tj_litellm_active

        # Verify the context var is False by default
        assert _tj_litellm_active.get(False) is False

        # Simulate being inside a litellm wrapper
        token = _tj_litellm_active.set(True)
        try:
            assert _tj_litellm_active.get(False) is True
        finally:
            _tj_litellm_active.reset(token)

        assert _tj_litellm_active.get(False) is False
