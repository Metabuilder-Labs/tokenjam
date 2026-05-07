"""
OpenAI Agents SDK integration.

OpenAI Agents SDK has native OTel support. This module configures it
to export traces to tj serve via the SDK's official set_trace_processors() API.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tokenjam.core.config import load_config

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig

logger = logging.getLogger(__name__)


def patch_openai_agents(config: TjConfig | None = None) -> None:
    """
    Configure OpenAI Agents SDK's built-in OTel support to export to tj serve.
    Uses the SDK's official set_trace_processors() API.
    """
    if config is None:
        config = load_config()

    try:
        from agents import set_trace_processors
        from agents.tracing.processors import BatchTraceProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        logger.warning(
            "openai-agents SDK not installed — "
            "run: pip install openai-agents"
        )
        return

    endpoint = f"http://{config.api.host}:{config.api.port}/api/v1/spans"
    headers: dict[str, str] = {}
    if config.security.ingest_secret:
        headers["Authorization"] = f"Bearer {config.security.ingest_secret}"

    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    processor = BatchTraceProcessor(exporter)
    set_trace_processors([processor])
    logger.debug("OpenAI Agents SDK integration installed (via native OTel)")
