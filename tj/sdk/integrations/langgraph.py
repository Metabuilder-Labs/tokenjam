"""
LangGraph framework integration.

Patches CompiledGraph.invoke and CompiledGraph.astream to capture
graph execution as OTel spans.
"""
from __future__ import annotations

import functools
import logging

from opentelemetry import trace

from tj.otel.semconv import GenAIAttributes

logger = logging.getLogger(__name__)


class LangGraphIntegration:
    name = "langgraph"
    installed = False

    def __init__(self) -> None:
        self._original_invoke = None
        self._original_astream = None
        self._tracer = None

    def install(self, tracer) -> None:
        if self.installed:
            return
        self._tracer = tracer
        try:
            from langgraph.graph.state import CompiledGraph
        except ImportError:
            logger.warning("langgraph not installed — skipping patch")
            return

        integration = self
        self._original_invoke = CompiledGraph.invoke

        @functools.wraps(self._original_invoke)
        def patched_invoke(self_graph, *args, **kwargs):
            span = integration._tracer.start_span("langgraph.invoke")
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, "langgraph")
            graph_name = getattr(self_graph, "name", None) or "graph"
            span.set_attribute("langgraph.graph_name", graph_name)
            try:
                result = integration._original_invoke(self_graph, *args, **kwargs)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return result
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end()

        CompiledGraph.invoke = patched_invoke

        self._original_astream = getattr(CompiledGraph, "astream", None)
        if self._original_astream:
            @functools.wraps(self._original_astream)
            async def patched_astream(self_graph, *args, **kwargs):
                span = integration._tracer.start_span("langgraph.astream")
                span.set_attribute(GenAIAttributes.PROVIDER_NAME, "langgraph")
                graph_name = getattr(self_graph, "name", None) or "graph"
                span.set_attribute("langgraph.graph_name", graph_name)
                try:
                    async for chunk in integration._original_astream(self_graph, *args, **kwargs):
                        yield chunk
                    span.set_status(trace.Status(trace.StatusCode.OK))
                except Exception as exc:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                    raise
                finally:
                    span.end()

            CompiledGraph.astream = patched_astream

        self.installed = True
        logger.debug("LangGraph integration installed")

    def uninstall(self) -> None:
        if not self.installed:
            return
        try:
            from langgraph.graph.state import CompiledGraph
            if self._original_invoke:
                CompiledGraph.invoke = self._original_invoke
            if self._original_astream:
                CompiledGraph.astream = self._original_astream
        except ImportError:
            pass
        self.installed = False


def patch_langgraph() -> None:
    """Convenience function. Instantiates and installs LangGraphIntegration."""
    from tj.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = LangGraphIntegration()
    integration.install(trace.get_tracer("tj.sdk"))
