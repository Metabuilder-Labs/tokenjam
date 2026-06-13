"""Tests for the LiteLLM integration (tokenjam.sdk.integrations.litellm)."""
from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest

from tokenjam.otel.semconv import GenAIAttributes


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
    from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
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
        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
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
        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
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

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
        integration = LiteLLMIntegration()
        integration.install(tracer)

        litellm.completion(model="together/mistral-7b", messages=[])

        span = spans[0]
        assert span.attributes[GenAIAttributes.PROVIDER_NAME] == "together"

    def test_bare_model_no_hidden_resolves_provider(self, tracer_and_spans):
        """Bare model + no custom_llm_provider is resolved from the model name (#194).

        Aider hits exactly this: LiteLLM >= 1.75 returns custom_llm_provider=None
        and the model is bare (no ``anthropic/`` prefix). The provider must
        resolve to ``anthropic`` (never the bogus ``"litellm"``) so pricing and
        billing_account are correct.
        """
        tracer, spans = tracer_and_spans
        import litellm

        def no_provider(*a, **kw):
            resp = MagicMock()
            resp.usage.prompt_tokens = 10
            resp.usage.completion_tokens = 5
            resp._hidden_params = {"custom_llm_provider": None}
            resp.model = "claude-haiku-4-5-20251001"
            return resp
        litellm.completion = no_provider

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
        integration = LiteLLMIntegration()
        integration.install(tracer)

        litellm.completion(model="claude-haiku-4-5", messages=[])

        span = spans[0]
        assert span.attributes[GenAIAttributes.PROVIDER_NAME] == "anthropic"

    def test_unresolvable_bare_model_is_unknown_not_litellm(self, tracer_and_spans):
        """An unattributable bare model yields 'unknown', never 'litellm' (#194)."""
        tracer, spans = tracer_and_spans
        import litellm

        def no_provider(*a, **kw):
            resp = MagicMock()
            resp.usage.prompt_tokens = 10
            resp.usage.completion_tokens = 5
            resp._hidden_params = {}
            resp.model = "some-internal-model-x"
            return resp
        litellm.completion = no_provider

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
        integration = LiteLLMIntegration()
        integration.install(tracer)

        litellm.completion(model="some-internal-model-x", messages=[])

        span = spans[0]
        assert span.attributes[GenAIAttributes.PROVIDER_NAME] == "unknown"
        assert span.attributes[GenAIAttributes.PROVIDER_NAME] != "litellm"

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

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
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

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
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

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
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
        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
        integration = LiteLLMIntegration()
        integration.install(tracer)
        assert litellm.completion is not original

        integration.uninstall()
        assert litellm.completion is original

    def test_context_var_suppresses_openai_patch(self, tracer_and_spans):
        """When litellm patch is active, the openai patch skips span creation."""
        from tokenjam.sdk.integrations.litellm import _tj_litellm_active

        # Verify the context var is False by default
        assert _tj_litellm_active.get(False) is False

        # Simulate being inside a litellm wrapper
        token = _tj_litellm_active.set(True)
        try:
            assert _tj_litellm_active.get(False) is True
        finally:
            _tj_litellm_active.reset(token)

        assert _tj_litellm_active.get(False) is False

    # -- content + cache-token capture (#195) ------------------------------- #

    def test_captures_prompt_and_completion_content(self, tracer_and_spans):
        """Prompt (from request messages) + completion (from response) land on
        the span as PROMPT_CONTENT / COMPLETION_CONTENT (#195). The ingest gate
        strips them later when [capture] is off — capture is unconditional here."""
        tracer, spans = tracer_and_spans
        import litellm

        def fake(*a, **kw):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="the answer"))],
                usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                _hidden_params={"custom_llm_provider": "openai"},
                model="gpt-4o-mini",
            )
        litellm.completion = fake

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
        LiteLLMIntegration().install(tracer)

        litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "what is 2+2?"}],
        )

        attrs = spans[0].attributes
        assert "what is 2+2?" in attrs[GenAIAttributes.PROMPT_CONTENT]
        assert attrs[GenAIAttributes.COMPLETION_CONTENT] == "the answer"

    def test_captures_cache_tokens_anthropic_style(self, tracer_and_spans):
        """Anthropic-style usage exposes cache_read/creation_input_tokens directly."""
        tracer, spans = tracer_and_spans
        import litellm

        def fake(*a, **kw):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="ok"))],
                usage=types.SimpleNamespace(
                    prompt_tokens=100, completion_tokens=50,
                    cache_read_input_tokens=300,
                    cache_creation_input_tokens=120,
                ),
                _hidden_params={"custom_llm_provider": "anthropic"},
                model="claude-haiku-4-5",
            )
        litellm.completion = fake

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
        LiteLLMIntegration().install(tracer)
        litellm.completion(model="anthropic/claude-haiku-4-5", messages=[])

        attrs = spans[0].attributes
        assert attrs[GenAIAttributes.CACHE_READ_TOKENS] == 300
        assert attrs[GenAIAttributes.CACHE_CREATE_TOKENS] == 120

    def test_captures_cache_tokens_openai_style(self, tracer_and_spans):
        """OpenAI-style usage nests the cached read count under
        prompt_tokens_details.cached_tokens (no creation count)."""
        tracer, spans = tracer_and_spans
        import litellm

        def fake(*a, **kw):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="ok"))],
                usage=types.SimpleNamespace(
                    prompt_tokens=100, completion_tokens=50,
                    prompt_tokens_details=types.SimpleNamespace(cached_tokens=200),
                ),
                _hidden_params={"custom_llm_provider": "openai"},
                model="gpt-4o-mini",
            )
        litellm.completion = fake

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
        LiteLLMIntegration().install(tracer)
        litellm.completion(model="openai/gpt-4o-mini", messages=[])

        attrs = spans[0].attributes
        assert attrs[GenAIAttributes.CACHE_READ_TOKENS] == 200
        assert GenAIAttributes.CACHE_CREATE_TOKENS not in attrs

    def test_no_cache_fields_means_no_cache_attrs(self, tracer_and_spans):
        """A usage object with no cache fields leaves cache attributes unset, so
        no-cache spans stay clean (and don't carry MagicMock-truthy garbage)."""
        tracer, spans = tracer_and_spans
        import litellm

        def fake(*a, **kw):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content="ok"))],
                usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                _hidden_params={"custom_llm_provider": "openai"},
                model="gpt-4o-mini",
            )
        litellm.completion = fake

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
        LiteLLMIntegration().install(tracer)
        litellm.completion(model="openai/gpt-4o-mini", messages=[])

        attrs = spans[0].attributes
        assert GenAIAttributes.CACHE_READ_TOKENS not in attrs
        assert GenAIAttributes.CACHE_CREATE_TOKENS not in attrs

    def test_streaming_captures_completion_content_and_cache(self, tracer_and_spans):
        """Sync streaming accumulates delta content + reads cache from the final
        usage chunk."""
        tracer, spans = tracer_and_spans
        import litellm

        def _chunk(text=None, usage=None):
            delta = types.SimpleNamespace(content=text)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=delta)], usage=usage,
            )
        chunks = [
            _chunk("Hel"),
            _chunk("lo!"),
            _chunk(None, types.SimpleNamespace(
                prompt_tokens=50, completion_tokens=25,
                cache_read_input_tokens=10)),
        ]
        litellm.completion = lambda *a, **kw: iter(chunks)

        from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
        LiteLLMIntegration().install(tracer)
        list(litellm.completion(
            model="openai/gpt-4o-mini", messages=[], stream=True,
        ))

        attrs = spans[0].attributes
        assert attrs[GenAIAttributes.COMPLETION_CONTENT] == "Hello!"
        assert attrs[GenAIAttributes.CACHE_READ_TOKENS] == 10
        assert attrs[GenAIAttributes.INPUT_TOKENS] == 50
