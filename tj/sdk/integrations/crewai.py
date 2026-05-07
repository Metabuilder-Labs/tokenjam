"""
CrewAI framework integration.

Patches Task.execute and Agent.execute_task to create OTel spans.
"""
from __future__ import annotations

import functools
import logging

from opentelemetry import trace

from tj.otel.semconv import GenAIAttributes

logger = logging.getLogger(__name__)


class CrewAIIntegration:
    name = "crewai"
    installed = False

    def __init__(self) -> None:
        self._original_task_execute = None
        self._original_agent_execute = None
        self._tracer = None

    def install(self, tracer) -> None:
        if self.installed:
            return
        self._tracer = tracer
        try:
            from crewai import Task, Agent
        except ImportError:
            logger.warning("crewai not installed — skipping patch")
            return

        integration = self

        self._original_task_execute = Task.execute
        @functools.wraps(self._original_task_execute)
        def patched_task_execute(self_task, *args, **kwargs):
            span = integration._tracer.start_span("crewai.task.execute")
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, "crewai")
            span.set_attribute("crewai.task_description", str(getattr(self_task, "description", ""))[:200])
            try:
                result = integration._original_task_execute(self_task, *args, **kwargs)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return result
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end()

        Task.execute = patched_task_execute

        self._original_agent_execute = Agent.execute_task
        @functools.wraps(self._original_agent_execute)
        def patched_agent_execute(self_agent, *args, **kwargs):
            span = integration._tracer.start_span("crewai.agent.execute_task")
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, "crewai")
            span.set_attribute("crewai.agent_role", str(getattr(self_agent, "role", ""))[:200])
            try:
                result = integration._original_agent_execute(self_agent, *args, **kwargs)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return result
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end()

        Agent.execute_task = patched_agent_execute
        self.installed = True
        logger.debug("CrewAI integration installed")

    def uninstall(self) -> None:
        if not self.installed:
            return
        try:
            from crewai import Task, Agent
            if self._original_task_execute:
                Task.execute = self._original_task_execute
            if self._original_agent_execute:
                Agent.execute_task = self._original_agent_execute
        except ImportError:
            pass
        self.installed = False


def patch_crewai() -> None:
    """Convenience function. Instantiates and installs CrewAIIntegration."""
    from tj.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = CrewAIIntegration()
    integration.install(trace.get_tracer("tj.sdk"))
