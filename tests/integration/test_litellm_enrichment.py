"""End-to-end: the LiteLLM integration enriches spans with content + cache
tokens (#195).

Wires the real LiteLLM integration through a local TracerProvider →
TjSpanExporter → IngestPipeline → InMemoryBackend, so the genuine path is
exercised: `span.set_attribute` → `convert_otel_span` → `strip_captured_content`
gate → `NormalizedSpan` field mapping → the `cache` analyzer. A local provider
(passed explicitly to `install()`) avoids the process-global TracerProvider, so
the test is deterministic regardless of order.
"""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.optimize.analyzers.cache_efficacy import _compute_rows
from tokenjam.otel.provider import TjSpanExporter
from tokenjam.otel.semconv import GenAIAttributes


def _fake_litellm_module():
    """A litellm whose completion returns Anthropic-style usage with cache reads
    + creation tokens and a real assistant message."""
    mod = types.ModuleType("litellm")

    def completion(*args, **kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content="the cached answer"))],
            usage=types.SimpleNamespace(
                prompt_tokens=10_000, completion_tokens=200,
                cache_read_input_tokens=8_000,
                cache_creation_input_tokens=1_500,
            ),
            _hidden_params={"custom_llm_provider": "anthropic"},
            model="claude-sonnet-4-6",
        )

    async def acompletion(*args, **kwargs):
        return completion()

    mod.completion = completion
    mod.acompletion = acompletion
    return mod


@pytest.fixture
def fake_litellm():
    fake = _fake_litellm_module()
    sys.modules["litellm"] = fake
    yield fake
    from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
    LiteLLMIntegration.installed = False
    del sys.modules["litellm"]


def _run_litellm_call(db, capture):
    """Install the integration against a local provider feeding `db`, make one
    (fake) litellm.completion call, return the stored span as a column dict
    (``attributes`` parsed from JSON)."""
    config = TjConfig(version="1", capture=capture)
    pipeline = IngestPipeline(db=db, config=config)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(TjSpanExporter(pipeline)))

    from tokenjam.sdk.integrations.litellm import LiteLLMIntegration
    import litellm
    LiteLLMIntegration().install(provider.get_tracer("test-195"))
    litellm.completion(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "summarize the doc"}],
    )
    provider.force_flush()

    cur = db.conn.execute("SELECT * FROM spans ORDER BY start_time DESC LIMIT 1")
    row = dict(zip([d[0] for d in cur.description], cur.fetchone()))
    row["attributes"] = json.loads(row["attributes"]) if row.get("attributes") else {}
    return row


def test_content_kept_when_capture_on(fake_litellm):
    db = InMemoryBackend()
    row = _run_litellm_call(db, CaptureConfig(prompts=True, completions=True))
    assert "summarize the doc" in row["attributes"][GenAIAttributes.PROMPT_CONTENT]
    assert row["attributes"][GenAIAttributes.COMPLETION_CONTENT] == "the cached answer"


def test_content_stripped_when_capture_off(fake_litellm):
    db = InMemoryBackend()
    row = _run_litellm_call(db, CaptureConfig(prompts=False, completions=False))
    assert GenAIAttributes.PROMPT_CONTENT not in row["attributes"]
    assert GenAIAttributes.COMPLETION_CONTENT not in row["attributes"]
    # Cache tokens are NOT content — they survive regardless of capture toggles.
    assert row["cache_tokens"] == 8_000
    assert row["cache_write_tokens"] == 1_500


def test_cache_tokens_populate_normalized_span(fake_litellm):
    db = InMemoryBackend()
    row = _run_litellm_call(db, CaptureConfig())
    assert row["cache_tokens"] == 8_000
    assert row["cache_write_tokens"] == 1_500


def test_cache_tokens_drive_cache_efficacy(fake_litellm):
    """The cache analyzer now sees real cache reads on LiteLLM spans, so efficacy
    is non-zero — it always read 0% before #195 (no cache tokens captured)."""
    db = InMemoryBackend()
    _run_litellm_call(db, CaptureConfig())
    since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    until = datetime(2100, 1, 1, tzinfo=timezone.utc)
    rows = _compute_rows(db.conn, since, until, agent_id=None)
    assert len(rows) == 1
    row = rows[0]
    assert row.provider == "anthropic"
    assert row.cache_tokens == 8_000
    assert row.efficacy > 0  # 8000 / (10000 + 8000) ≈ 0.44
