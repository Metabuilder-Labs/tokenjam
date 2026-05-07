"""
LangChain framework integration.

Patches BaseLLM.generate and BaseTool.run (both sync and async variants)
to create OTel spans for LLM calls and tool invocations.
"""
from __future__ import annotations

import functools
import logging

from opentelemetry import trace

from tj.otel.semconv import GenAIAttributes

logger = logging.getLogger(__name__)


class LangChainIntegration:
    name = "langchain"
    installed = False

    def __init__(self) -> None:
        self._original_generate = None
        self._original_agenerate = None
        self._original_tool_run = None
        self._original_tool_arun = None
        self._tracer = None

    def install(self, tracer) -> None:
        if self.installed:
            return
        self._tracer = tracer
        try:
            from langchain_core.language_models import BaseLLM
            from langchain_core.tools import BaseTool
        except ImportError:
            logger.warning("langchain-core not installed — skipping patch")
            return

        integration = self

        # Patch BaseLLM.generate
        self._original_generate = BaseLLM.generate

        @functools.wraps(self._original_generate)
        def patched_generate(self_llm, prompts, *args, **kwargs):
            span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, "langchain")
            model = getattr(self_llm, "model_name", None) or getattr(self_llm, "model", "unknown")
            span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)
            try:
                result = integration._original_generate(self_llm, prompts, *args, **kwargs)
                if hasattr(result, "llm_output") and result.llm_output:
                    usage = result.llm_output.get("token_usage", {})
                    if "prompt_tokens" in usage:
                        span.set_attribute(GenAIAttributes.INPUT_TOKENS, usage["prompt_tokens"])
                    if "completion_tokens" in usage:
                        span.set_attribute(GenAIAttributes.OUTPUT_TOKENS, usage["completion_tokens"])
                span.set_status(trace.Status(trace.StatusCode.OK))
                return result
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end()

        BaseLLM.generate = patched_generate

        # Patch BaseLLM.agenerate
        self._original_agenerate = getattr(BaseLLM, "agenerate", None)
        if self._original_agenerate:
            @functools.wraps(self._original_agenerate)
            async def patched_agenerate(self_llm, prompts, *args, **kwargs):
                span = integration._tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
                span.set_attribute(GenAIAttributes.PROVIDER_NAME, "langchain")
                model = getattr(self_llm, "model_name", None) or getattr(self_llm, "model", "unknown")
                span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)
                try:
                    result = await integration._original_agenerate(self_llm, prompts, *args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as exc:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                    raise
                finally:
                    span.end()

            BaseLLM.agenerate = patched_agenerate

        # Patch BaseTool.run
        self._original_tool_run = BaseTool.run

        @functools.wraps(self._original_tool_run)
        def patched_tool_run(self_tool, *args, **kwargs):
            span = integration._tracer.start_span(GenAIAttributes.SPAN_TOOL_CALL)
            span.set_attribute(GenAIAttributes.TOOL_NAME, self_tool.name)
            try:
                result = integration._original_tool_run(self_tool, *args, **kwargs)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return result
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end()

        BaseTool.run = patched_tool_run

        # Patch BaseTool.arun
        self._original_tool_arun = getattr(BaseTool, "arun", None)
        if self._original_tool_arun:
            @functools.wraps(self._original_tool_arun)
            async def patched_tool_arun(self_tool, *args, **kwargs):
                span = integration._tracer.start_span(GenAIAttributes.SPAN_TOOL_CALL)
                span.set_attribute(GenAIAttributes.TOOL_NAME, self_tool.name)
                try:
                    result = await integration._original_tool_arun(self_tool, *args, **kwargs)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result
                except Exception as exc:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                    raise
                finally:
                    span.end()

            BaseTool.arun = patched_tool_arun

        self.installed = True
        logger.debug("LangChain integration installed")

    def uninstall(self) -> None:
        if not self.installed:
            return
        try:
            from langchain_core.language_models import BaseLLM
            from langchain_core.tools import BaseTool
            if self._original_generate:
                BaseLLM.generate = self._original_generate
            if self._original_agenerate:
                BaseLLM.agenerate = self._original_agenerate
            if self._original_tool_run:
                BaseTool.run = self._original_tool_run
            if self._original_tool_arun:
                BaseTool.arun = self._original_tool_arun
        except ImportError:
            pass
        self.installed = False


def patch_langchain() -> None:
    """Convenience function. Instantiates and installs LangChainIntegration."""
    from tj.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = LangChainIntegration()
    integration.install(trace.get_tracer("tj.sdk"))
