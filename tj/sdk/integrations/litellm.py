"""
LiteLLM provider integration.

Wraps litellm.completion and litellm.acompletion to automatically
create OTel spans with token usage and provider attribution.

LiteLLM provides a unified interface across 100+ LLM providers. A single
patch_litellm() call gives tj coverage across all of them.

When both patch_litellm() and patch_openai()/patch_anthropic() are active,
the LiteLLM patch wins — a context variable prevents inner provider patches
from creating duplicate spans.
"""
from __future__ import annotations

import contextvars
import functools
import logging

from opentelemetry import trace

from tj.otel.semconv import GenAIAttributes

logger = logging.getLogger(__name__)

# Context variable used to suppress inner provider patches (openai, anthropic)
# when a call originates from litellm.completion/acompletion.
_tj_litellm_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_tj_litellm_active", default=False,
)


def _parse_provider(model: str, response: object) -> str:
    """Extract provider name from response or model string prefix."""
    # Prefer the actual provider from LiteLLM's hidden params
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        provider = hidden.get("custom_llm_provider")
        if provider:
            return str(provider)
    # Fallback: infer from model string prefix (e.g. "anthropic/claude-...")
    if "/" in model:
        return model.split("/", 1)[0]
    return "litellm"


def _strip_provider_prefix(model: str) -> str:
    """Strip provider prefix from LiteLLM model string.

    'openai/gpt-4o-mini' -> 'gpt-4o-mini', consistent with patch_openai.
    """
    if "/" in model:
        return model.split("/", 1)[1]
    return model


class LiteLLMIntegration:
    name = "litellm"
    installed = False

    def __init__(self) -> None:
        self._original_completion = None
        self._original_acompletion = None
        self._tracer = None

    def install(self, tracer) -> None:
        """Patch litellm.completion and litellm.acompletion."""
        if self.installed:
            return
        self._tracer = tracer
        try:
            import litellm
        except ImportError:
            logger.warning("litellm package not installed — skipping patch")
            return

        self._original_completion = litellm.completion
        self._original_acompletion = litellm.acompletion
        integration = self

        @functools.wraps(self._original_completion)
        def patched_completion(*args, **kwargs):
            raw_model = str(args[0] if args else kwargs.get("model", "unknown"))
            model = _strip_provider_prefix(raw_model)
            is_stream = kwargs.get("stream", False)

            span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
            span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)

            # Inherit agent_id / conversation_id from parent span
            parent_span = trace.get_current_span()
            if parent_span and parent_span.is_recording():
                agent_id = parent_span.attributes.get(GenAIAttributes.AGENT_ID)
                if agent_id:
                    span.set_attribute(GenAIAttributes.AGENT_ID, agent_id)
                conv_id = parent_span.attributes.get(
                    GenAIAttributes.CONVERSATION_ID,
                )
                if conv_id:
                    span.set_attribute(
                        GenAIAttributes.CONVERSATION_ID, conv_id,
                    )

            token = _tj_litellm_active.set(True)
            try:
                response = integration._original_completion(*args, **kwargs)
                if is_stream:
                    return _SyncStreamWrapper(response, span, raw_model, token)
                _record_usage(response, span, raw_model)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return response
            except Exception as exc:
                span.set_status(
                    trace.Status(trace.StatusCode.ERROR, str(exc)),
                )
                raise
            finally:
                if not is_stream:
                    span.end()
                    _tj_litellm_active.reset(token)

        litellm.completion = patched_completion

        @functools.wraps(self._original_acompletion)
        async def patched_acompletion(*args, **kwargs):
            raw_model = str(args[0] if args else kwargs.get("model", "unknown"))
            model = _strip_provider_prefix(raw_model)
            is_stream = kwargs.get("stream", False)

            span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
            span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)

            parent_span = trace.get_current_span()
            if parent_span and parent_span.is_recording():
                agent_id = parent_span.attributes.get(GenAIAttributes.AGENT_ID)
                if agent_id:
                    span.set_attribute(GenAIAttributes.AGENT_ID, agent_id)
                conv_id = parent_span.attributes.get(
                    GenAIAttributes.CONVERSATION_ID,
                )
                if conv_id:
                    span.set_attribute(
                        GenAIAttributes.CONVERSATION_ID, conv_id,
                    )

            token = _tj_litellm_active.set(True)
            try:
                response = await integration._original_acompletion(
                    *args, **kwargs,
                )
                if is_stream:
                    return _AsyncStreamWrapper(response, span, raw_model, token)
                _record_usage(response, span, raw_model)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return response
            except Exception as exc:
                span.set_status(
                    trace.Status(trace.StatusCode.ERROR, str(exc)),
                )
                raise
            finally:
                if not is_stream:
                    span.end()
                    _tj_litellm_active.reset(token)

        litellm.acompletion = patched_acompletion

        self.installed = True
        logger.debug("LiteLLM integration installed")

    def uninstall(self) -> None:
        if not self.installed:
            return
        try:
            import litellm
            if self._original_completion:
                litellm.completion = self._original_completion
            if self._original_acompletion:
                litellm.acompletion = self._original_acompletion
        except ImportError:
            pass
        self.installed = False


def _record_usage(response: object, span, model: str) -> None:
    """Extract provider and token usage from a LiteLLM ModelResponse."""
    provider = _parse_provider(model, response)
    span.set_attribute(GenAIAttributes.PROVIDER_NAME, provider)
    # Update model attribute to strip provider prefix for consistent pricing
    span.set_attribute(GenAIAttributes.REQUEST_MODEL, _strip_provider_prefix(model))

    usage = getattr(response, "usage", None)
    if usage:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is not None:
            span.set_attribute(GenAIAttributes.INPUT_TOKENS, prompt_tokens)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if completion_tokens is not None:
            span.set_attribute(
                GenAIAttributes.OUTPUT_TOKENS, completion_tokens,
            )


class _SyncStreamWrapper:
    """Wraps a LiteLLM sync stream to capture usage and end the span."""

    def __init__(self, stream, span, model: str, token):
        self._stream = stream
        self._span = span
        self._model = model
        self._token = token
        self._usage = None
        self._last_chunk = None

    def __iter__(self):
        _ok = False
        try:
            for chunk in self._stream:
                usage = getattr(chunk, "usage", None)
                if usage:
                    self._usage = usage
                self._last_chunk = chunk
                yield chunk
            _ok = True
        except Exception as exc:
            self._span.set_status(
                trace.Status(trace.StatusCode.ERROR, str(exc)),
            )
            raise
        finally:
            provider = _parse_provider(self._model, self._last_chunk)
            self._span.set_attribute(GenAIAttributes.PROVIDER_NAME, provider)
            self._span.set_attribute(
                GenAIAttributes.REQUEST_MODEL,
                _strip_provider_prefix(self._model),
            )
            if self._usage:
                prompt_tokens = getattr(
                    self._usage, "prompt_tokens", None,
                )
                if prompt_tokens is not None:
                    self._span.set_attribute(
                        GenAIAttributes.INPUT_TOKENS, prompt_tokens,
                    )
                completion_tokens = getattr(
                    self._usage, "completion_tokens", None,
                )
                if completion_tokens is not None:
                    self._span.set_attribute(
                        GenAIAttributes.OUTPUT_TOKENS, completion_tokens,
                    )
            if _ok:
                self._span.set_status(trace.Status(trace.StatusCode.OK))
            self._span.end()
            _tj_litellm_active.reset(self._token)

    def __next__(self):
        return self._stream.__next__()


class _AsyncStreamWrapper:
    """Wraps a LiteLLM async stream to capture usage and end the span."""

    def __init__(self, stream, span, model: str, token):
        self._stream = stream
        self._span = span
        self._model = model
        self._token = token
        self._usage = None
        self._last_chunk = None

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        _ok = False
        try:
            async for chunk in self._stream:
                usage = getattr(chunk, "usage", None)
                if usage:
                    self._usage = usage
                self._last_chunk = chunk
                yield chunk
            _ok = True
        except Exception as exc:
            self._span.set_status(
                trace.Status(trace.StatusCode.ERROR, str(exc)),
            )
            raise
        finally:
            provider = _parse_provider(self._model, self._last_chunk)
            self._span.set_attribute(GenAIAttributes.PROVIDER_NAME, provider)
            self._span.set_attribute(
                GenAIAttributes.REQUEST_MODEL,
                _strip_provider_prefix(self._model),
            )
            if self._usage:
                prompt_tokens = getattr(
                    self._usage, "prompt_tokens", None,
                )
                if prompt_tokens is not None:
                    self._span.set_attribute(
                        GenAIAttributes.INPUT_TOKENS, prompt_tokens,
                    )
                completion_tokens = getattr(
                    self._usage, "completion_tokens", None,
                )
                if completion_tokens is not None:
                    self._span.set_attribute(
                        GenAIAttributes.OUTPUT_TOKENS, completion_tokens,
                    )
            if _ok:
                self._span.set_status(trace.Status(trace.StatusCode.OK))
            self._span.end()
            _tj_litellm_active.reset(self._token)


def patch_litellm() -> None:
    """Convenience function. Instantiates and installs LiteLLMIntegration."""
    from tj.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = LiteLLMIntegration()
    integration.install(trace.get_tracer("tj.sdk"))
