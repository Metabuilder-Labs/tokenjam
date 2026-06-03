"""convert_otel_span resource-attribute extraction (the in-process SDK path).

The cross-session run markers (tokenjam.run_id / tokenjam.parent_session_id) are
OTel *resource* attributes. They must be extracted on every ingest path; this
covers the in-process Python SDK path (TjSpanExporter -> convert_otel_span),
alongside the HTTP/OTLP (otlp_parsing) and Claude Code logs (logs.py) paths.
"""
from __future__ import annotations

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

from tokenjam.otel.provider import convert_otel_span
from tokenjam.otel.semconv import GenAIAttributes


def _span_with_resource(resource_attrs: dict) -> object:
    """Build a real ended OTel ReadableSpan carrying the given resource attrs."""
    provider = TracerProvider(resource=Resource.create(resource_attrs))
    tracer = provider.get_tracer("test")
    span = tracer.start_span(GenAIAttributes.SPAN_LLM_CALL)
    span.end()
    return span


def test_convert_otel_span_extracts_run_markers_from_resource():
    span = _span_with_resource({
        "tokenjam.run_id": "run-xyz",
        "tokenjam.parent_session_id": "parent-abc",
        "service.namespace": "aquanode",
        "service.instance.id": "founder-os",
    })

    ns = convert_otel_span(span)

    assert ns.run_id == "run-xyz"
    assert ns.parent_session_id == "parent-abc"
    # Sanity: the sibling resource attrs still resolve (regression guard for the
    # whole resource-extraction block, not just the new markers).
    assert ns.service_namespace == "aquanode"
    assert ns.service_instance_id == "founder-os"


def test_convert_otel_span_run_markers_absent_is_none():
    span = _span_with_resource({"service.namespace": "aquanode"})

    ns = convert_otel_span(span)

    assert ns.run_id is None
    assert ns.parent_session_id is None
