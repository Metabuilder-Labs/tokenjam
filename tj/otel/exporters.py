from __future__ import annotations

import logging

from tj.core.config import PrometheusConfig

logger = logging.getLogger("tj.otel")


def build_prometheus_exporter(config: PrometheusConfig):
    """
    Start the Prometheus metrics endpoint on config.port at config.path.
    Returns the exporter instance.
    """
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider

    reader = PrometheusMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])

    logger.info(
        "Prometheus metrics available on port %d at %s",
        config.port,
        config.path,
    )
    return reader, meter_provider
