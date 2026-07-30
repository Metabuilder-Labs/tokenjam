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

from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tokenjam.sdk.attribution import stamp_span_attribution
from tokenjam.sdk.integrations._request_capture import (
    extract_anthropic_completion,
    record_completion_content,
    record_full_request,
    record_prompt_content,
)

logger = logging.getLogger(__name__)


def _record_response_id(span, response: Any) -> None:
    """Stamp the provider's own id for this response onto the span.

    The id (`msg_...`) names the API CALL rather than this observation of it,
    so a second observer of the same call — a transcript backfill, a sibling
    exporter — can be recognised as a restatement instead of counted again.
    See `core.optimize.accounting`. Best-effort: a response object without an
    id just leaves the span unstamped, which is what every span carried before.
    """
    response_id = getattr(response, "id", None)
    if isinstance(response_id, str) and response_id:
        span.set_attribute(GenAIAttributes.RESPONSE_ID, response_id)


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
            # Cost-attribution dimensions from the ambient sdk.attribution
            # context, if a caller declared one (#SDK dashboard shape) — this
            # patched client call has no per-call kwarg for tenant_id/feature.
            stamp_span_attribution(span)
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
                _record_response_id(span, response)
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
                stamp_span_attribution(span)
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


def _event_carries_content(event: Any) -> bool:
    """True when a streamed event delivered generated output to the caller.

    Structural and defensive — an unrecognised event shape counts as no content
    rather than raising into the caller's iteration.
    """
    delta = getattr(event, "delta", None)
    if delta is None:
        return False
    return bool(getattr(delta, "text", None) or getattr(delta, "partial_json", None))


class _StreamWrapper:
    """Wraps an Anthropic stream to capture final usage and end the span.

    Also records the streaming data-quality signature (see
    ``TjAttributes.STREAMING``). Anthropic reports output tokens only in the
    trailing ``message_delta`` / ``message_stop`` pair, which
    ``get_final_message()`` reads — so a caller that breaks out of the loop or
    is disconnected mid-response leaves this span with no token counts, which
    is indistinguishable from a free call unless the span says so itself.
    """

    def __init__(self, stream, span):
        self._stream = stream
        self._span = span
        self._content_events = 0

    def __enter__(self):
        self._stream.__enter__()
        return self

    def _final_message(self) -> Any:
        """The stream's final message, or None when the stream never finished.

        ``get_final_message()`` raises when the stream was not consumed to
        completion, which is exactly the case being detected — so the raise is
        an answer, not an error, and must not escape ``__exit__``.
        """
        getter = getattr(self._stream, "get_final_message", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self._stream.__exit__(exc_type, exc_val, exc_tb)
        # One message serves two purposes: it names the API call (so a second
        # observer of the same call is recognised as a restatement rather than
        # counted again) and it carries the trailing usage. Absent it, both
        # facts are simply unknown — which is itself recorded below.
        final_message = self._final_message()
        if final_message is not None:
            _record_response_id(self._span, final_message)
        usage = getattr(final_message, "usage", None)
        if usage is not None:
            self._span.set_attribute(
                GenAIAttributes.INPUT_TOKENS,
                usage.input_tokens,
            )
            self._span.set_attribute(
                GenAIAttributes.OUTPUT_TOKENS,
                usage.output_tokens,
            )
        # Stamped unconditionally, including on the happy path: the analyzer
        # needs the complete streams as the peer baseline it estimates the
        # missing ones against, so "usage reported" is as load-bearing a fact
        # as "usage missing".
        self._span.set_attribute(TjAttributes.STREAMING, True)
        self._span.set_attribute(TjAttributes.STREAM_USAGE_REPORTED, usage is not None)
        self._span.set_attribute(
            TjAttributes.STREAM_CONTENT_CHUNKS, self._content_events,
        )
        if exc_type is None:
            self._span.set_status(trace.Status(trace.StatusCode.OK))
        else:
            self._span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc_val)))
        self._span.end()
        return result

    def __iter__(self):
        for event in self._stream:
            if _event_carries_content(event):
                self._content_events += 1
            yield event

    def __next__(self):
        event = next(self._stream)
        if _event_carries_content(event):
            self._content_events += 1
        return event


def patch_anthropic() -> None:
    """Convenience function. Instantiates and installs AnthropicIntegration."""
    from tokenjam.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = AnthropicIntegration()
    integration.install(trace.get_tracer("tokenjam.sdk"))
