"""
Anthropic provider integration.

Wraps anthropic.resources.Messages.create and .stream to automatically
create OTel spans with token usage and model attributes.
"""
from __future__ import annotations

import functools
import logging
from typing import Any

from opentelemetry import trace

from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.sdk.integrations._request_capture import (
    extract_anthropic_completion,
    record_completion_content,
    record_full_request,
    record_prompt_content,
)

logger = logging.getLogger(__name__)


class AnthropicIntegration:
    name = "anthropic"
    installed = False

    def __init__(self) -> None:
        self._original_create: Any = None
        self._original_stream: Any = None
        self._tracer = None

    def install(self, tracer) -> None:
        """Patch anthropic.resources.Messages.create and .stream."""
        if self.installed:
            return
        self._tracer = tracer
        try:
            from anthropic.resources import Messages
        except ImportError:
            logger.warning("anthropic package not installed — skipping patch")
            return

        self._original_create = Messages.create
        self._original_stream = getattr(Messages, "stream", None)

        integration = self

        @functools.wraps(self._original_create)
        def patched_create(self_msg, *args, **kwargs):
            # Skip span if call originates from litellm (avoids double-counting)
            from tokenjam.sdk.integrations.litellm import _tj_litellm_active
            if _tj_litellm_active.get(False):
                return integration._original_create(self_msg, *args, **kwargs)
            span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, "anthropic")
            span.set_attribute(
                GenAIAttributes.REQUEST_MODEL,
                kwargs.get("model", "unknown"),
            )
            record_full_request(span, kwargs)
            # Prompt content (#320). Set unconditionally; stripped at ingest
            # unless [capture] prompts is on. Same serialization as litellm.
            record_prompt_content(span, kwargs.get("messages"))
            # Inherit agent_id from parent span (set by @watch())
            parent_span = trace.get_current_span()
            if parent_span and parent_span.is_recording():
                agent_id = parent_span.attributes.get(GenAIAttributes.AGENT_ID)
                if agent_id:
                    span.set_attribute(GenAIAttributes.AGENT_ID, agent_id)
                conv_id = parent_span.attributes.get(GenAIAttributes.CONVERSATION_ID)
                if conv_id:
                    span.set_attribute(GenAIAttributes.CONVERSATION_ID, conv_id)
            try:
                response = integration._original_create(self_msg, *args, **kwargs)
                if hasattr(response, "usage"):
                    span.set_attribute(
                        GenAIAttributes.INPUT_TOKENS,
                        response.usage.input_tokens,
                    )
                    span.set_attribute(
                        GenAIAttributes.OUTPUT_TOKENS,
                        response.usage.output_tokens,
                    )
                    cache_read = getattr(response.usage, "cache_read_input_tokens", None)
                    if cache_read:
                        span.set_attribute(GenAIAttributes.CACHE_READ_TOKENS, cache_read)
                    cache_create = getattr(response.usage, "cache_creation_input_tokens", None)
                    if cache_create:
                        span.set_attribute(GenAIAttributes.CACHE_CREATE_TOKENS, cache_create)
                # Completion content (#320). Stripped at ingest unless
                # [capture] completions is on.
                record_completion_content(span, extract_anthropic_completion(response))
                span.set_status(trace.Status(trace.StatusCode.OK))
                return response
            except TypeError as exc:
                if "api_key" in str(exc) or "auth" in str(exc).lower():
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                    import sys
                    print(
                        "\n\033[1;31mError: Anthropic API key not found.\033[0m\n"
                        "\n"
                        "  Set it in your environment:\n"
                        "\n"
                        "    export ANTHROPIC_API_KEY='sk-ant-...'\n"
                        "\n"
                        "  Or pass it directly:\n"
                        "\n"
                        "    anthropic.Anthropic(api_key='...')\n",
                        file=sys.stderr,
                    )
                    raise SystemExit(1)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end()

        Messages.create = patched_create

        if self._original_stream is not None:
            @functools.wraps(self._original_stream)
            def patched_stream(self_msg, *args, **kwargs):
                from tokenjam.sdk.integrations.litellm import _tj_litellm_active
                if _tj_litellm_active.get(False):
                    return integration._original_stream(self_msg, *args, **kwargs)
                span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
                span.set_attribute(GenAIAttributes.PROVIDER_NAME, "anthropic")
                span.set_attribute(
                    GenAIAttributes.REQUEST_MODEL,
                    kwargs.get("model", "unknown"),
                )
                record_full_request(span, kwargs)
                # Prompt content (#320). Completion content for the streaming
                # path would need buffering the stream (the wrapper doesn't
                # aggregate text) — out of scope; the request is captured here.
                record_prompt_content(span, kwargs.get("messages"))
                parent_span = trace.get_current_span()
                if parent_span and parent_span.is_recording():
                    agent_id = parent_span.attributes.get(GenAIAttributes.AGENT_ID)
                    if agent_id:
                        span.set_attribute(GenAIAttributes.AGENT_ID, agent_id)
                    conv_id = parent_span.attributes.get(GenAIAttributes.CONVERSATION_ID)
                    if conv_id:
                        span.set_attribute(GenAIAttributes.CONVERSATION_ID, conv_id)
                try:
                    stream = integration._original_stream(self_msg, *args, **kwargs)
                    return _StreamWrapper(stream, span)
                except Exception as exc:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                    span.end()
                    raise

            Messages.stream = patched_stream

        self.installed = True
        logger.debug("Anthropic integration installed")

    def uninstall(self) -> None:
        if not self.installed:
            return
        try:
            from anthropic.resources import Messages
            if self._original_create:
                Messages.create = self._original_create
            if self._original_stream:
                Messages.stream = self._original_stream
        except ImportError:
            pass
        self.installed = False


class _StreamWrapper:
    """Wraps an Anthropic stream to capture final usage and end the span."""

    def __init__(self, stream, span):
        self._stream = stream
        self._span = span

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self._stream.__exit__(exc_type, exc_val, exc_tb)
        final_message = getattr(self._stream, "get_final_message", lambda: None)()
        if final_message and hasattr(final_message, "usage"):
            self._span.set_attribute(
                GenAIAttributes.INPUT_TOKENS,
                final_message.usage.input_tokens,
            )
            self._span.set_attribute(
                GenAIAttributes.OUTPUT_TOKENS,
                final_message.usage.output_tokens,
            )
        if exc_type is None:
            self._span.set_status(trace.Status(trace.StatusCode.OK))
        else:
            self._span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
        self._span.end()
        return result

    def __iter__(self):
        return iter(self._stream)

    def __next__(self):
        return next(self._stream)


def patch_anthropic() -> None:
    """Convenience function. Instantiates and installs AnthropicIntegration."""
    from tokenjam.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = AnthropicIntegration()
    integration.install(trace.get_tracer("tokenjam.sdk"))
