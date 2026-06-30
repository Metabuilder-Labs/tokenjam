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

from tokenjam.core.pricing import provider_for_model
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.sdk.integrations._request_capture import (
    record_completion_content,
    record_full_request,
    record_prompt_content,
)

logger = logging.getLogger(__name__)

# Context variable used to suppress inner provider patches (openai, anthropic)
# when a call originates from litellm.completion/acompletion.
_tj_litellm_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_tj_litellm_active", default=False,
)


def _parse_provider(model: str, response: object) -> str:
    """Resolve the real provider for a LiteLLM call.

    Resolution order:
      1. LiteLLM's ``custom_llm_provider`` from the response hidden params.
      2. A ``provider/`` prefix on the model string (``anthropic/claude-...``).
      3. Inference from the bare model name (``claude-*`` -> ``anthropic`` etc.).

    Never returns the literal ``"litellm"`` — that is not a real provider and
    misses both pricing and billing_account, undercounting cost and suppressing
    plan-tier framing (#194). An unresolvable provider yields ``"unknown"``,
    which has no pricing table (so it isn't billed as a real provider) and maps
    to a NULL billing_account downstream.
    """
    # 1. Prefer the actual provider from LiteLLM's hidden params.
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        provider = hidden.get("custom_llm_provider")
        if provider:
            return str(provider)
    # 2. A "provider/model" prefix (e.g. "anthropic/claude-...").
    if "/" in model:
        return model.split("/", 1)[0]
    # 3. Infer from the bare model name; never fall back to "litellm".
    return provider_for_model(model) or "unknown"


def _strip_provider_prefix(model: str) -> str:
    """Strip provider prefix from LiteLLM model string.

    'openai/gpt-4o-mini' -> 'gpt-4o-mini', consistent with patch_openai.
    """
    if "/" in model:
        return model.split("/", 1)[1]
    return model


# --- content + cache-token enrichment (#195) ------------------------------- #
# These set attributes unconditionally; the IngestPipeline's
# strip_captured_content() gate drops prompt/completion content when the
# matching [capture] toggle is off, so there is no behavior change when content
# capture is disabled. Without them, cache-recommend / Reuse text / trim get no
# content and the cache analyzer always reads 0% efficacy for LiteLLM spans.

def _record_prompt_content(span, args: tuple, kwargs: dict) -> None:
    """Set PROMPT_CONTENT on the span (stripped at ingest if capture is off).

    Delegates the serialization to the shared ``record_prompt_content`` helper
    (#320) so every capture path (provider patches + litellm) agrees on the
    ``json.dumps(messages)`` shape a replay harness / backfill expects. LiteLLM's
    extra ``messages``-positional / text-``prompt`` fallbacks are resolved here
    before handing the resolved request object to the shared helper.
    """
    messages = kwargs.get("messages")
    if messages is None and len(args) > 1:
        messages = args[1]
    if messages is not None:
        record_prompt_content(span, messages)
        return
    prompt = kwargs.get("prompt")
    if prompt is not None:
        record_prompt_content(span, prompt if isinstance(prompt, str) else str(prompt))


def _extract_completion_text(response: object) -> str | None:
    """Pull the assistant text from a non-streaming ModelResponse."""
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    try:
        choice = choices[0]
    except (IndexError, TypeError):
        return None
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return content if isinstance(content, str) else str(content)
    text = getattr(choice, "text", None)
    if text is not None:
        return str(text)
    return None


def _chunk_delta_text(chunk: object) -> str | None:
    """Extract the incremental assistant text from one streaming chunk."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    try:
        delta = getattr(choices[0], "delta", None)
    except (IndexError, TypeError):
        return None
    if delta is None:
        return None
    content = getattr(delta, "content", None)
    return content if isinstance(content, str) else None


def _extract_cache_tokens(usage: object) -> tuple[int | None, int | None]:
    """Return ``(cache_read, cache_write)`` from a LiteLLM Usage object.

    LiteLLM normalizes provider usage inconsistently: Anthropic-style prompt
    caching surfaces ``cache_read_input_tokens`` / ``cache_creation_input_tokens``
    directly on the usage object, while OpenAI-style caching nests the read
    count under ``prompt_tokens_details.cached_tokens``. Check both shapes.
    """
    cache_read = getattr(usage, "cache_read_input_tokens", None)
    if not cache_read:
        details = getattr(usage, "prompt_tokens_details", None)
        if isinstance(details, dict):
            cache_read = details.get("cached_tokens")
        elif details is not None:
            cache_read = getattr(details, "cached_tokens", None)
    cache_write = getattr(usage, "cache_creation_input_tokens", None)
    return cache_read, cache_write


def _set_usage_attributes(usage: object, span) -> None:
    """Set input/output + cache-read/write token attributes from a Usage object.

    Cache fields (#195) were previously dropped, so the ``cache`` analyzer
    always read 0% efficacy for LiteLLM spans. Mirrors patch_anthropic: only
    set cache attributes when non-zero so no-cache spans stay clean.
    """
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is not None:
        span.set_attribute(GenAIAttributes.INPUT_TOKENS, prompt_tokens)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is not None:
        span.set_attribute(GenAIAttributes.OUTPUT_TOKENS, completion_tokens)
    cache_read, cache_write = _extract_cache_tokens(usage)
    if cache_read:
        span.set_attribute(GenAIAttributes.CACHE_READ_TOKENS, cache_read)
    if cache_write:
        span.set_attribute(GenAIAttributes.CACHE_CREATE_TOKENS, cache_write)


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

            _record_prompt_content(span, args, kwargs)
            record_full_request(span, kwargs)

            token = _tj_litellm_active.set(True)
            handed_off = False
            try:
                response = integration._original_completion(*args, **kwargs)
                if is_stream:
                    wrapper = _SyncStreamWrapper(response, span, raw_model, token)
                    handed_off = True
                    return wrapper
                _record_usage(response, span, raw_model)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return response
            except Exception as exc:
                span.set_status(
                    trace.Status(trace.StatusCode.ERROR, str(exc)),
                )
                raise
            finally:
                # Clean up unless the stream wrapper took ownership; this also
                # covers a streaming call that raised before the wrapper was
                # built, which would otherwise leak the contextvar + span (#48).
                if not handed_off:
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

            _record_prompt_content(span, args, kwargs)
            record_full_request(span, kwargs)

            token = _tj_litellm_active.set(True)
            handed_off = False
            try:
                response = await integration._original_acompletion(
                    *args, **kwargs,
                )
                if is_stream:
                    wrapper = _AsyncStreamWrapper(
                        response, span, raw_model, token,
                    )
                    handed_off = True
                    return wrapper
                _record_usage(response, span, raw_model)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return response
            except Exception as exc:
                span.set_status(
                    trace.Status(trace.StatusCode.ERROR, str(exc)),
                )
                raise
            finally:
                # Clean up unless the stream wrapper took ownership; this also
                # covers a streaming call that raised before the wrapper was
                # built, which would otherwise leak the contextvar + span (#48).
                if not handed_off:
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
        _set_usage_attributes(usage, span)

    record_completion_content(span, _extract_completion_text(response))


class _SyncStreamWrapper:
    """Wraps a LiteLLM sync stream to capture usage and end the span."""

    def __init__(self, stream, span, model: str, token):
        self._stream = stream
        self._span = span
        self._model = model
        self._token = token
        self._usage = None
        self._last_chunk = None
        self._completion_parts: list[str] = []

    def __iter__(self):
        _ok = False
        try:
            for chunk in self._stream:
                usage = getattr(chunk, "usage", None)
                if usage:
                    self._usage = usage
                delta = _chunk_delta_text(chunk)
                if delta:
                    self._completion_parts.append(delta)
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
                _set_usage_attributes(self._usage, self._span)
            if self._completion_parts:
                self._span.set_attribute(
                    GenAIAttributes.COMPLETION_CONTENT,
                    "".join(self._completion_parts),
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
        self._completion_parts: list[str] = []

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        _ok = False
        try:
            async for chunk in self._stream:
                usage = getattr(chunk, "usage", None)
                if usage:
                    self._usage = usage
                delta = _chunk_delta_text(chunk)
                if delta:
                    self._completion_parts.append(delta)
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
                _set_usage_attributes(self._usage, self._span)
            if self._completion_parts:
                self._span.set_attribute(
                    GenAIAttributes.COMPLETION_CONTENT,
                    "".join(self._completion_parts),
                )
            if _ok:
                self._span.set_status(trace.Status(trace.StatusCode.OK))
            self._span.end()
            _tj_litellm_active.reset(self._token)


def patch_litellm() -> None:
    """Convenience function. Instantiates and installs LiteLLMIntegration."""
    from tokenjam.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = LiteLLMIntegration()
    integration.install(trace.get_tracer("tokenjam.sdk"))
