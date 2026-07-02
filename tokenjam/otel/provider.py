from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import Resource
from opentelemetry import trace

from opentelemetry.trace import StatusCode as OtelStatusCode
from opentelemetry.trace import SpanKind as OtelSpanKind

from tokenjam.core.models import NormalizedSpan, SpanStatus, SpanKind
from tokenjam.core.config import TjConfig
from tokenjam.otel.semconv import GenAIAttributes, ResourceAttributes, TjAttributes

if TYPE_CHECKING:
    from tokenjam.core.ingest import IngestPipeline

logger = logging.getLogger("tokenjam.otel")

# Mapping from OTel SpanKind enum to our SpanKind enum
_OTEL_KIND_MAP = {
    OtelSpanKind.INTERNAL: SpanKind.INTERNAL,
    OtelSpanKind.SERVER:   SpanKind.SERVER,
    OtelSpanKind.CLIENT:   SpanKind.CLIENT,
    OtelSpanKind.PRODUCER: SpanKind.PRODUCER,
    OtelSpanKind.CONSUMER: SpanKind.CONSUMER,
}

_OTEL_STATUS_MAP = {
    OtelStatusCode.UNSET: SpanStatus.UNSET,
    OtelStatusCode.OK:    SpanStatus.OK,
    OtelStatusCode.ERROR: SpanStatus.ERROR,
}


class TjSpanExporter(SpanExporter):
    """
    Custom OTel SpanExporter that feeds spans into the IngestPipeline.
    Used when the Python SDK instruments code in-process.
    """

    def __init__(self, ingest_pipeline: IngestPipeline):
        self.pipeline = ingest_pipeline

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        failures = 0
        for otel_span in spans:
            try:
                normalised = convert_otel_span(otel_span)
                self.pipeline.process(normalised)
            except Exception as exc:
                logger.warning("Span export failed: %s", exc)
                failures += 1
        return SpanExportResult.FAILURE if failures > 0 else SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def convert_otel_span(otel_span: ReadableSpan) -> NormalizedSpan:
    """
    Convert an opentelemetry-sdk ReadableSpan to NormalizedSpan.
    Extract all indexed attributes (provider, model, tool_name, tokens, etc.)
    from the span's attribute dict using GenAIAttributes constants.
    """
    attrs = dict(otel_span.attributes or {})

    # Extract indexed fields from attributes
    provider = attrs.pop(GenAIAttributes.PROVIDER_NAME, None)
    model = attrs.pop(GenAIAttributes.REQUEST_MODEL, None)
    tool_name = attrs.pop(GenAIAttributes.TOOL_NAME, None)
    input_tokens = attrs.pop(GenAIAttributes.INPUT_TOKENS, None)
    output_tokens = attrs.pop(GenAIAttributes.OUTPUT_TOKENS, None)
    cache_tokens = attrs.pop(GenAIAttributes.CACHE_READ_TOKENS, None)
    cache_write_tokens = attrs.pop(GenAIAttributes.CACHE_CREATE_TOKENS, None)
    conversation_id = attrs.pop(GenAIAttributes.CONVERSATION_ID, None)
    request_type = attrs.pop(GenAIAttributes.REQUEST_TYPE, None)
    agent_id = attrs.pop(GenAIAttributes.AGENT_ID, None)
    cost_usd = attrs.pop(TjAttributes.COST_USD, None)
    session_id = attrs.pop(TjAttributes.SESSION_ID, None)
    # billing_account: explicit attribute wins, otherwise derived from provider.
    # Map gen_ai.provider.name values to billing_account (provider-only).
    billing_account = attrs.pop(TjAttributes.BILLING_ACCOUNT, None) or _provider_to_billing_account(provider)

    # service.namespace is a resource attribute (set on the TracerProvider /
    # OTEL_RESOURCE_ATTRIBUTES), not a span attribute.
    resource = getattr(otel_span, "resource", None)
    resource_attrs = dict(resource.attributes) if resource and resource.attributes else {}
    service_namespace = resource_attrs.get(ResourceAttributes.SERVICE_NAMESPACE)
    service_instance_id = resource_attrs.get(ResourceAttributes.SERVICE_INSTANCE_ID)

    # Convert int tokens to int (OTel may store as strings)
    if input_tokens is not None:
        input_tokens = int(input_tokens)
    if output_tokens is not None:
        output_tokens = int(output_tokens)
    if cache_tokens is not None:
        cache_tokens = int(cache_tokens)
    if cache_write_tokens is not None:
        cache_write_tokens = int(cache_write_tokens)
    if cost_usd is not None:
        cost_usd = float(cost_usd)

    # Convert timestamps (OTel uses nanoseconds since epoch)
    start_time = _ns_to_datetime(otel_span.start_time) if otel_span.start_time else None
    end_time = _ns_to_datetime(otel_span.end_time) if otel_span.end_time else None

    duration_ms = None
    if start_time and end_time:
        duration_ms = (end_time - start_time).total_seconds() * 1000.0

    # Map OTel kind and status
    kind = _OTEL_KIND_MAP.get(otel_span.kind, SpanKind.INTERNAL)
    status_code = SpanStatus.UNSET
    status_message = None
    if otel_span.status:
        status_code = _OTEL_STATUS_MAP.get(otel_span.status.status_code, SpanStatus.UNSET)
        status_message = otel_span.status.description

    # Format span and trace IDs as hex strings
    ctx = otel_span.context
    span_id = format(ctx.span_id, "016x") if ctx else ""
    trace_id = format(ctx.trace_id, "032x") if ctx else ""

    parent_span_id = None
    if otel_span.parent and otel_span.parent.span_id:
        parent_span_id = format(otel_span.parent.span_id, "016x")

    # Convert events
    events = []
    for event in otel_span.events or []:
        events.append({
            "name": event.name,
            "timestamp": _ns_to_datetime(event.timestamp).isoformat() if event.timestamp else None,
            "attributes": dict(event.attributes) if event.attributes else {},
        })

    return NormalizedSpan(
        span_id=span_id,
        trace_id=trace_id,
        name=otel_span.name,
        kind=kind,
        status_code=status_code,
        status_message=status_message,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        parent_span_id=parent_span_id,
        agent_id=agent_id,
        session_id=session_id,
        provider=provider,
        model=model,
        tool_name=tool_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_tokens=cache_tokens,
        cache_write_tokens=cache_write_tokens,
        cost_usd=cost_usd,
        request_type=request_type,
        conversation_id=conversation_id,
        attributes=attrs,
        events=events,
        billing_account=billing_account,
        service_namespace=service_namespace,
        service_instance_id=service_instance_id,
    )


def _provider_to_billing_account(provider: str | None) -> str | None:
    """
    Map gen_ai.provider.name to a tj billing_account (provider-only identifier).

    Returns None for non-LLM spans or unrecognised providers — the analyzers
    skip rows with a NULL billing_account when scoping per-provider spend.
    """
    if not provider:
        return None
    p = str(provider).lower()
    # Direct hits
    if p in {"anthropic", "openai", "google", "bedrock", "local.ollama"}:
        return p
    # Aliases
    if p in {"aws.bedrock", "aws_bedrock"}:
        return "bedrock"
    if p in {"google.gemini", "gemini", "vertex", "vertex_ai"}:
        return "google"
    if p in {"ollama"}:
        return "local.ollama"
    if p in {"azure", "azure.openai", "azure_openai"}:
        return "openai"  # Azure OpenAI billed as openai for plan-tier purposes
    return None


def _ns_to_datetime(ns: int) -> datetime:
    """Convert nanoseconds since epoch to timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def build_tracer_provider(config: TjConfig, ingest_pipeline: IngestPipeline) -> TracerProvider:
    """
    Build and configure the global TracerProvider.
    Attaches TjSpanExporter (always) and OTLP exporters (if configured).
    Sets as the global tracer provider.
    """
    import tokenjam

    resource = Resource.create({
        "service.name": "tokenjam",
        "service.version": tokenjam.__version__,
    })
    provider = TracerProvider(resource=resource)

    # Always attach the local exporter
    provider.add_span_processor(
        BatchSpanProcessor(TjSpanExporter(ingest_pipeline))
    )

    # Optionally attach OTLP exporter
    if config.export.otlp.enabled:
        otlp_exporter = _build_otlp_exporter(config)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    trace.set_tracer_provider(provider)
    return provider


def _build_otlp_exporter(config: TjConfig) -> SpanExporter:
    """Build OTLP/HTTP or OTLP/gRPC exporter based on config.protocol."""
    otlp = config.export.otlp
    if otlp.protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter(
            endpoint=otlp.endpoint,
            headers=otlp.headers or {},
            insecure=otlp.insecure,
        )
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        return OTLPSpanExporter(
            endpoint=f"{otlp.endpoint}/v1/traces",
            headers=otlp.headers or {},
        )
