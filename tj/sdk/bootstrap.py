"""
Auto-bootstrap: lazily initialise the OCW TracerProvider + IngestPipeline
the first time @watch() or a provider patch creates a span.

This ensures that SDK users don't need to manually wire up the pipeline.
"""
from __future__ import annotations

import atexit
import logging
import threading


logger = logging.getLogger("tj.sdk")

_lock = threading.Lock()
_initialised = False
_provider = None
_pipeline = None


def ensure_initialised() -> None:
    """
    Idempotent bootstrap. Safe to call multiple times / from multiple threads.
    Sets up: config -> DuckDB -> IngestPipeline -> TjSpanExporter -> TracerProvider.
    """
    global _initialised, _provider, _pipeline
    if _initialised:
        return

    with _lock:
        if _initialised:
            return

        try:
            from tj.core.config import load_config
            from tj.otel.provider import build_tracer_provider

            config = load_config()

            # Check if tj serve is running — use HTTP exporter if so
            if _try_http_mode(config):
                _initialised = True
                atexit.register(_shutdown)
                return

            # Direct DuckDB mode
            from tj.core.db import open_db
            from tj.core.ingest import build_default_pipeline

            db = open_db(config.storage)
            pipeline = build_default_pipeline(db, config)
            _pipeline = pipeline
            _provider = build_tracer_provider(config, pipeline)
            _initialised = True

            # Ensure spans are flushed on exit
            atexit.register(_shutdown)

            logger.debug("OCW: writing spans to local DuckDB (%s)", config.storage.path)

        except Exception as exc:
            # DuckDB lock error — try HTTP fallback
            err_msg = str(exc).lower()
            if "lock" in err_msg or "i/o error" in err_msg:
                try:
                    config = load_config()
                    if _try_http_mode(config):
                        _initialised = True
                        atexit.register(_shutdown)
                        return
                except Exception:
                    pass
            logger.warning("OCW bootstrap failed — spans will not be recorded: %s", exc)
            _initialised = True  # Don't retry on every call


def _try_http_mode(config) -> bool:
    """Try to connect to tj serve and set up HTTP exporter. Returns True on success."""
    global _provider
    import httpx
    base_url = f"http://{config.api.host}:{config.api.port}"
    try:
        resp = httpx.get(f"{base_url}/api/v1/status", timeout=2)
        if resp.status_code not in (200, 401):
            return False
    except (httpx.ConnectError, httpx.TimeoutException):
        return False

    from tj.sdk.http_exporter import TjHttpExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry import trace as trace_api
    import tj

    endpoint = f"{base_url}/api/v1/spans"
    exporter = TjHttpExporter(endpoint, config.security.ingest_secret)

    resource = Resource.create({
        "service.name": "tokenjam",
        "service.version": tj.__version__,
    })
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace_api.set_tracer_provider(provider)
    _provider = provider

    logger.info("TJ: sending spans to tj serve at %s", base_url)
    return True


def _shutdown() -> None:
    """Flush pending spans on interpreter exit."""
    if _provider is not None:
        try:
            _provider.force_flush(timeout_millis=5000)
            _provider.shutdown()
        except Exception:
            pass
