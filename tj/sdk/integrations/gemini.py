"""
Google Gemini provider integration.

Wraps google.generativeai.GenerativeModel.generate_content and
generate_content_async to create OTel spans with token usage.
"""
from __future__ import annotations

import functools
import logging

from opentelemetry import trace

from tj.otel.semconv import GenAIAttributes

logger = logging.getLogger(__name__)


class GeminiIntegration:
    name = "google"
    installed = False

    def __init__(self) -> None:
        self._original_generate = None
        self._original_generate_async = None
        self._tracer = None

    def install(self, tracer) -> None:
        if self.installed:
            return
        self._tracer = tracer
        try:
            import google.generativeai as genai
            model_cls = genai.GenerativeModel
        except ImportError:
            logger.warning("google-generativeai package not installed — skipping patch")
            return

        self._original_generate = model_cls.generate_content
        self._original_generate_async = getattr(
            model_cls, "generate_content_async", None,
        )
        integration = self

        @functools.wraps(self._original_generate)
        def patched_generate(self_model, *args, **kwargs):
            span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, "google")
            span.set_attribute(
                GenAIAttributes.REQUEST_MODEL,
                getattr(self_model, "model_name", "unknown"),
            )
            try:
                response = integration._original_generate(self_model, *args, **kwargs)
                meta = getattr(response, "usage_metadata", None)
                if meta:
                    span.set_attribute(
                        GenAIAttributes.INPUT_TOKENS,
                        getattr(meta, "prompt_token_count", 0),
                    )
                    span.set_attribute(
                        GenAIAttributes.OUTPUT_TOKENS,
                        getattr(meta, "candidates_token_count", 0),
                    )
                span.set_status(trace.Status(trace.StatusCode.OK))
                return response
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end()

        model_cls.generate_content = patched_generate

        if self._original_generate_async is not None:
            @functools.wraps(self._original_generate_async)
            async def patched_generate_async(self_model, *args, **kwargs):
                span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
                span.set_attribute(GenAIAttributes.PROVIDER_NAME, "google")
                span.set_attribute(
                    GenAIAttributes.REQUEST_MODEL,
                    getattr(self_model, "model_name", "unknown"),
                )
                try:
                    response = await integration._original_generate_async(
                        self_model, *args, **kwargs,
                    )
                    meta = getattr(response, "usage_metadata", None)
                    if meta:
                        span.set_attribute(
                            GenAIAttributes.INPUT_TOKENS,
                            getattr(meta, "prompt_token_count", 0),
                        )
                        span.set_attribute(
                            GenAIAttributes.OUTPUT_TOKENS,
                            getattr(meta, "candidates_token_count", 0),
                        )
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return response
                except Exception as exc:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                    raise
                finally:
                    span.end()

            model_cls.generate_content_async = patched_generate_async

        self.installed = True
        logger.debug("Gemini integration installed")

    def uninstall(self) -> None:
        if not self.installed:
            return
        try:
            import google.generativeai as genai
            model_cls = genai.GenerativeModel
            if self._original_generate:
                model_cls.generate_content = self._original_generate
            if self._original_generate_async:
                model_cls.generate_content_async = self._original_generate_async
        except ImportError:
            pass
        self.installed = False


def patch_gemini() -> None:
    """Convenience function. Instantiates and installs GeminiIntegration."""
    from tj.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = GeminiIntegration()
    integration.install(trace.get_tracer("tj.sdk"))
