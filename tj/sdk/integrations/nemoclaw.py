"""
NemoClaw/OpenShell Gateway WebSocket observer.

Connects to the OpenShell Gateway as an observer client, receives sandbox
events, and translates them into OTel spans fed into the ingest pipeline.

NOTICE: NemoClaw is licensed under Apache License 2.0.
This integration module acknowledges the upstream Apache 2.0 license.
See https://github.com/NVIDIA/NemoClaw/blob/main/LICENSE
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING


from tj.core.models import NormalizedSpan, SpanKind, SpanStatus
from tj.otel.semconv import TjAttributes
from tj.utils.ids import new_span_id, new_trace_id
from tj.utils.time_parse import utcnow

if TYPE_CHECKING:
    from tj.core.config import TjConfig

logger = logging.getLogger(__name__)

SANDBOX_EVENT_MAP = {
    "network_blocked": "network_blocked",
    "fs_access_denied": "fs_denied",
    "syscall_blocked": "syscall_denied",
    "inference_reroute": "inference_rerouted",
}


class NemoClawGatewayObserver:
    """
    Observes a NemoClaw OpenShell sandbox by connecting to the
    OpenShell Gateway WebSocket as an observer client.

    All inference calls, blocked network requests, filesystem denials,
    and syscall blocks are translated into OTel spans and fed into the
    standard tj ingest pipeline.

    Usage:
        observer = NemoClawGatewayObserver(ingest_pipeline)
        asyncio.run(observer.connect())  # runs until cancelled
    """

    def __init__(self, ingest_pipeline, gateway_url: str = "ws://127.0.0.1:18789"):
        self.pipeline = ingest_pipeline
        self.gateway_url = gateway_url

    async def connect(self) -> None:
        """Connect and observe. Reconnects with backoff on disconnect."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed — cannot observe NemoClaw")
            return

        backoff = 1.0
        while True:
            try:
                async with websockets.connect(self.gateway_url) as ws:
                    logger.info("Connected to NemoClaw gateway at %s", self.gateway_url)
                    backoff = 1.0
                    async for message in ws:
                        try:
                            event = json.loads(message)
                            span = self._translate_event(event)
                            if span:
                                self.pipeline.process(span)
                        except (json.JSONDecodeError, KeyError) as exc:
                            logger.debug("Skipping malformed gateway event: %s", exc)
            except Exception as exc:
                logger.warning(
                    "NemoClaw gateway disconnected: %s — reconnecting in %.0fs",
                    exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    def _translate_event(self, event: dict) -> NormalizedSpan | None:
        """Convert an OpenShell gateway event to a NormalizedSpan."""
        event_type = event.get("type", "")
        ocw_event = SANDBOX_EVENT_MAP.get(event_type)
        if not ocw_event:
            return None

        now = utcnow()
        attrs: dict = {
            TjAttributes.SANDBOX_EVENT: ocw_event,
        }

        if ocw_event == "network_blocked":
            attrs[TjAttributes.EGRESS_HOST] = event.get("host", "unknown")
            attrs[TjAttributes.EGRESS_PORT] = event.get("port")
        elif ocw_event == "fs_denied":
            attrs[TjAttributes.FILESYSTEM_PATH] = event.get("path", "unknown")
        elif ocw_event == "syscall_denied":
            attrs[TjAttributes.SYSCALL_NAME] = event.get("syscall", "unknown")

        return NormalizedSpan(
            span_id=new_span_id(),
            trace_id=new_trace_id(),
            name=f"sandbox.{ocw_event}",
            kind=SpanKind.INTERNAL,
            status_code=SpanStatus.ERROR,
            start_time=now,
            end_time=now,
            duration_ms=0.0,
            agent_id=event.get("agent_id"),
            attributes=attrs,
        )


def watch_nemoclaw(
    gateway_url: str = "ws://127.0.0.1:18789",
    config: TjConfig | None = None,
) -> NemoClawGatewayObserver:
    """
    Convenience function. Creates an observer instance.
    Call observer.connect() in an asyncio task to start observing.

    Note: You must pass an ingest_pipeline instance. If config is None,
    a default config is loaded.
    """
    from tj.core.config import load_config
    from tj.sdk.bootstrap import ensure_initialised, _pipeline
    if config is None:
        config = load_config()
    ensure_initialised()
    logger.info("NemoClaw observer created for %s", gateway_url)
    return NemoClawGatewayObserver(
        ingest_pipeline=_pipeline,
        gateway_url=gateway_url,
    )
