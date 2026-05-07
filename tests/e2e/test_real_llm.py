"""
End-to-end tests that make real LLM API calls.

Requires: TJ_ANTHROPIC_API_KEY environment variable.
Auto-skipped without it (see conftest.py).

These tests verify the full path: real API call -> provider patch ->
OTel spans -> IngestPipeline -> DB with cost/alert hooks.
"""
from __future__ import annotations

import os
import threading
from typing import Sequence

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from tj.core.config import CaptureConfig, TjConfig, SecurityConfig
from tj.core.db import InMemoryBackend
from tj.core.ingest import IngestPipeline
from tj.core.cost import CostEngine
from tj.otel.provider import convert_otel_span
from tj.otel.semconv import GenAIAttributes

pytestmark = pytest.mark.skipif(
    not os.environ.get("TJ_ANTHROPIC_API_KEY"),
    reason="TJ_ANTHROPIC_API_KEY not set",
)


class _CollectingExporter(SpanExporter):
    def __init__(self) -> None:
        self._spans: list[ReadableSpan] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            self._spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def get_finished_spans(self) -> list[ReadableSpan]:
        with self._lock:
            return list(self._spans)


@pytest.fixture
def otel_setup():
    """Set up a fresh TracerProvider with collecting exporter."""
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Store old provider and set new one
    old_provider = trace.get_tracer_provider()
    trace.set_tracer_provider(provider)

    yield exporter

    # Clean up
    provider.shutdown()


def test_real_anthropic_call_produces_spans(otel_setup):
    """
    Make a real Anthropic API call via the SDK and verify spans are produced.
    """
    try:
        import anthropic
    except ImportError:
        pytest.skip("anthropic package not installed")

    from tj.sdk import watch, patch_anthropic
    import tj.sdk.agent as agent_mod

    # Re-bind tracer to use our test provider
    agent_mod._tracer = trace.get_tracer("tj.sdk")

    patch_anthropic()

    @watch(agent_id="e2e-test-agent")
    def call_claude():
        client = anthropic.Anthropic(api_key=os.environ["TJ_ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )
        return response.content[0].text

    result = call_claude()
    assert isinstance(result, str)
    assert len(result) > 0

    spans = otel_setup.get_finished_spans()
    assert len(spans) >= 2  # At least session + LLM call

    session_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_INVOKE_AGENT]
    llm_spans = [s for s in spans if s.name == GenAIAttributes.SPAN_LLM_CALL]
    assert len(session_spans) == 1
    assert len(llm_spans) >= 1

    # Verify the LLM span has token counts
    llm = llm_spans[0]
    attrs = dict(llm.attributes or {})
    assert GenAIAttributes.INPUT_TOKENS in attrs
    assert GenAIAttributes.OUTPUT_TOKENS in attrs
    assert int(attrs[GenAIAttributes.INPUT_TOKENS]) > 0
    assert int(attrs[GenAIAttributes.OUTPUT_TOKENS]) > 0


def test_real_span_converts_to_normalized_span(otel_setup):
    """Verify convert_otel_span works on real OTel spans."""
    from tj.sdk import watch, record_llm_call
    import tj.sdk.agent as agent_mod

    agent_mod._tracer = trace.get_tracer("tj.sdk")

    @watch(agent_id="e2e-convert-agent")
    def my_agent():
        record_llm_call("claude-haiku-4-5", "anthropic", 100, 20)

    my_agent()

    raw_spans = otel_setup.get_finished_spans()
    assert len(raw_spans) >= 1

    for raw in raw_spans:
        normalized = convert_otel_span(raw)
        assert normalized.span_id != ""
        assert normalized.trace_id != ""
        assert normalized.start_time is not None


def test_real_span_flows_through_pipeline(otel_setup):
    """Verify a real OTel span can be ingested through the full pipeline."""
    from tj.sdk import watch, record_llm_call
    import tj.sdk.agent as agent_mod

    agent_mod._tracer = trace.get_tracer("tj.sdk")

    @watch(agent_id="e2e-pipeline-agent")
    def my_agent():
        record_llm_call("claude-haiku-4-5", "anthropic", 500, 100)

    my_agent()

    db = InMemoryBackend()
    config = TjConfig(
        version="1",
        capture=CaptureConfig(prompts=True, completions=True),
    )
    cost_engine = CostEngine(db=db)
    pipeline = IngestPipeline(db=db, config=config, cost_engine=cost_engine)

    raw_spans = otel_setup.get_finished_spans()
    for raw in raw_spans:
        normalized = convert_otel_span(raw)
        pipeline.process(normalized)

    assert len(db.spans) >= 1
    assert len(db.sessions) >= 1
    db.close()
