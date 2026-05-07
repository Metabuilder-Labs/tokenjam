"""
AutoGen framework integration.

Patches ConversableAgent.generate_reply and ConversableAgent.initiate_chat
to create OTel spans.
"""
from __future__ import annotations

import functools
import logging

from opentelemetry import trace

from tj.otel.semconv import GenAIAttributes

logger = logging.getLogger(__name__)


class AutoGenIntegration:
    name = "autogen"
    installed = False

    def __init__(self) -> None:
        self._original_generate_reply = None
        self._original_initiate_chat = None
        self._tracer = None

    def install(self, tracer) -> None:
        if self.installed:
            return
        self._tracer = tracer
        try:
            from autogen import ConversableAgent
        except ImportError:
            logger.warning("pyautogen not installed — skipping patch")
            return

        integration = self

        self._original_generate_reply = ConversableAgent.generate_reply
        @functools.wraps(self._original_generate_reply)
        def patched_generate_reply(self_agent, *args, **kwargs):
            span = integration._tracer.start_span("autogen.generate_reply")
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, "autogen")
            span.set_attribute("autogen.agent_name", getattr(self_agent, "name", "unknown"))
            try:
                result = integration._original_generate_reply(self_agent, *args, **kwargs)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return result
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end()

        ConversableAgent.generate_reply = patched_generate_reply

        self._original_initiate_chat = ConversableAgent.initiate_chat
        @functools.wraps(self._original_initiate_chat)
        def patched_initiate_chat(self_agent, *args, **kwargs):
            span = integration._tracer.start_span("autogen.initiate_chat")
            span.set_attribute(GenAIAttributes.PROVIDER_NAME, "autogen")
            span.set_attribute("autogen.agent_name", getattr(self_agent, "name", "unknown"))
            try:
                result = integration._original_initiate_chat(self_agent, *args, **kwargs)
                span.set_status(trace.Status(trace.StatusCode.OK))
                return result
            except Exception as exc:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(exc)))
                raise
            finally:
                span.end()

        ConversableAgent.initiate_chat = patched_initiate_chat
        self.installed = True
        logger.debug("AutoGen integration installed")

    def uninstall(self) -> None:
        if not self.installed:
            return
        try:
            from autogen import ConversableAgent
            if self._original_generate_reply:
                ConversableAgent.generate_reply = self._original_generate_reply
            if self._original_initiate_chat:
                ConversableAgent.initiate_chat = self._original_initiate_chat
        except ImportError:
            pass
        self.installed = False


def patch_autogen() -> None:
    """Convenience function. Instantiates and installs AutoGenIntegration."""
    from tj.sdk.bootstrap import ensure_initialised
    ensure_initialised()
    integration = AutoGenIntegration()
    integration.install(trace.get_tracer("tj.sdk"))
