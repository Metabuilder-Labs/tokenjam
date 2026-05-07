"""
OpenAI provider integration.

Wraps openai.resources.chat.completions.Completions.create to automatically
create OTel spans with token usage and model attributes.

Also works for OpenAI-compatible providers (Groq, Together, Fireworks, xAI,
Azure OpenAI) — pass the provider's base_url and set provider name from it.
"""
from __future__ import annotations

import functools
import logging

from opentelemetry import trace

from tokenjam.otel.semconv import GenAIAttributes

logger = logging.getLogger(__name__)


class OpenAIIntegration:
    name = "openai"
    installed = False

    def __init__(self, provider_name: str = "openai") -> None:
        self._original_create = None
        self._tracer = None
        self._provider_name = provider_name

    def install(self, tracer) -> None:
        """Patch openai.resources.chat.completions.Completions.create."""
        if self.installed:
            return
        self._tracer = tracer
        try:
            from openai.resources.chat.completions import Completions
        except ImportError:
            logger.warning("openai package not installed — skipping patch")
            return

        self._original_create = Completions.create
        integration = self

        @functools.wraps(self._original_create)
        def patched_create(self_comp, *args, **kwargs):
            # Skip span if call originates from litellm (avoids double-counting)
            from tokenjam.sdk.integrations.litellm import _tj_litellm_active
            if _tj_litellm_active.get(False):
                return integration._original_create(self_comp, *args, **kwargs)
            span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, integration._provider_name)
            span.set_attribute(
                GenAIAttributes.REQUEST_MODEL,
                kwargs.get("model", "unknown"),
            )
            is_stream = kwargs.get("stream", False)
            try:
                response = integration._original_create(self_comp, *args, **kwargs)
                if is_stream:
                    return _StreamWrapper(response, span)
                if hasattr(response, "usage") and response.usage:
                    span.set_attribute(
                        GenAIAttributes.INPUT_TOKENS,
                        response.usage.prompt_tokens,
                    )
                    span.set_attribute(
                        GenAIAttributes.OUTPUT_TOKENS,
                        response.usage.completion_tokens,
                    )
                span.set_status(trace.Status(trace.StatusCode.OK))
                span.end()
                return response
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                span.end()
                raise

        Completions.create = patched_create
        self.installed = True
        logger.debug("OpenAI integration installed (provider=%s)", self._provider_name)

    def uninstall(self) -> None:
        if not self.installed:
            return
        try:
            from openai.resources.chat.completions import Completions
            if self._original_create:
                Completions.create = self._original_create
        except ImportError:
            pass
        self.installed = False


class _StreamWrapper:
    """Wraps an OpenAI stream to capture final usage chunk and end the span."""

    def __init__(self, stream, span):
        self._stream = stream
        self._span = span
        self._usage = None

    def __iter__(self):
        _ok = False
        try:
            for chunk in self._stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    self._usage = chunk.usage
                yield chunk
            _ok = True
        except Exception as exc:
            self._span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
            raise
        finally:
            if self._usage:
                self._span.set_attribute(
                    GenAIAttributes.INPUT_TOKENS,
                    self._usage.prompt_tokens,
                )
                self._span.set_attribute(
                    GenAIAttributes.OUTPUT_TOKENS,
                    self._usage.completion_tokens,
                )
            if _ok:
                self._span.set_status(trace.Status(trace.StatusCode.OK))
            self._span.end()

    def __next__(self):
        return self._stream.__next__()


def patch_openai(base_url: str | None = None) -> None:
    """
    Wraps the OpenAI client.
    Also works for OpenAI-compatible providers (Groq, Together, Fireworks, xAI,
    Azure OpenAI) — pass the provider's base_url and set provider name from it.
    """
    from tokenjam.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    provider = "openai"
    if base_url:
        # Infer provider name from base_url domain
        from urllib.parse import urlparse
        domain = urlparse(base_url).hostname or ""
        if "groq" in domain:
            provider = "groq"
        elif "together" in domain:
            provider = "together"
        elif "fireworks" in domain:
            provider = "fireworks"
        elif "xai" in domain or "x.ai" in domain:
            provider = "xai"
        elif "azure" in domain:
            provider = "azure.openai"
        else:
            provider = domain.split(".")[0] if domain else "openai-compatible"

    integration = OpenAIIntegration(provider_name=provider)
    integration.install(trace.get_tracer("tokenjam.sdk"))
