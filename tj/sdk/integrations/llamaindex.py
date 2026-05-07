"""
LlamaIndex framework integration.

LlamaIndex has native OTel support. This module is a thin convenience wrapper
that configures LlamaIndex's built-in OTel instrumentation to point at tj serve.

Does NOT monkey-patch LlamaIndex internals — uses its official instrumentation API.

Requires: pip install opentelemetry-instrumentation-llama-index
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tj.core.config import load_config

if TYPE_CHECKING:
    from tj.core.config import TjConfig

logger = logging.getLogger(__name__)


def patch_llamaindex(config: TjConfig | None = None) -> None:
    """
    Configure LlamaIndex's built-in OTel support to export to tj serve.
    """
    if config is None:
        config = load_config()

    try:
        from opentelemetry.instrumentation.llama_index import LlamaIndexInstrumentor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-llama-index not installed — "
            "run: pip install opentelemetry-instrumentation-llama-index"
        )
        return

    endpoint = f"http://{config.api.host}:{config.api.port}/api/v1/spans"
    headers: dict[str, str] = {}
    if config.security.ingest_secret:
        headers["Authorization"] = f"Bearer {config.security.ingest_secret}"

    exporter = OTLPSpanExporter(endpoint=endpoint, headers=headers)
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    LlamaIndexInstrumentor().instrument(tracer_provider=provider)
    logger.debug("LlamaIndex integration installed (via native OTel)")
