"""Tests for tokenjam.core.ingest — sanitizer, pipeline, session resolution, capture stripping."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from tokenjam.core.config import TjConfig, SecurityConfig, CaptureConfig, AgentConfig
from tokenjam.core.ingest import (
    IngestPipeline,
    SpanRejectedError,
    SpanSanitizer,
    strip_captured_content,
)
from tokenjam.core.models import NormalizedSpan, SessionRecord, SpanStatus
from tokenjam.otel.semconv import GenAIAttributes
from tests.factories import (
    make_invoke_agent_span,
    make_llm_span,
    make_session,
    make_tool_span,
)


# ---------------------------------------------------------------------------
# Minimal in-memory storage stub (task 01 provides the real implementation)
# ---------------------------------------------------------------------------

class InMemoryBackend:
    """Stub StorageBackend that stores everything in dicts/lists."""

    def __init__(self) -> None:
        self.spans: list[NormalizedSpan] = []
        self.sessions: dict[str, SessionRecord] = {}

    def insert_span(self, span: NormalizedSpan) -> None:
        self.spans.append(span)

    def upsert_session(self, session: SessionRecord) -> None:
        self.sessions[session.session_id] = session

    def get_session(self, session_id: str) -> SessionRecord | None:
        return self.sessions.get(session_id)

    def get_session_by_conversation(self, conversation_id: str) -> SessionRecord | None:
        for s in self.sessions.values():
            if s.conversation_id == conversation_id:
                return s
        return None


# ---------------------------------------------------------------------------
# No-op hook stubs
# ---------------------------------------------------------------------------

class NoopCostEngine:
    def process_span(self, span: NormalizedSpan) -> None:
        pass


class NoopAlertEngine:
    def evaluate(self, span: NormalizedSpan) -> None:
        pass


class NoopSchemaValidator:
    def validate(self, span: NormalizedSpan) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(
    *,
    security: SecurityConfig | None = None,
    capture: CaptureConfig | None = None,
    db: InMemoryBackend | None = None,
    agents: dict | None = None,
) -> tuple[IngestPipeline, InMemoryBackend]:
    """Create an IngestPipeline with sensible defaults for testing."""
    db = db or InMemoryBackend()
    config = TjConfig(
        version="1",
        security=security or SecurityConfig(),
        capture=capture or CaptureConfig(),
        agents=agents or {},
    )
    pipeline = IngestPipeline(
        db=db,
        config=config,
        cost_engine=NoopCostEngine(),
        alert_engine=NoopAlertEngine(),
        schema_validator=NoopSchemaValidator(),
    )
    return pipeline, db


# ===========================================================================
# SpanSanitizer tests
# ===========================================================================

class TestSpanSanitizer:

    def test_passes_valid_span(self):
        sanitizer = SpanSanitizer(SecurityConfig())
        # Should not raise
        sanitizer.validate({"key": "value", "count": 42})

    def test_rejects_too_many_attributes(self):
        config = SecurityConfig(max_attributes_per_span=5)
        sanitizer = SpanSanitizer(config)
        attrs = {f"key_{i}": i for i in range(10)}
        with pytest.raises(SpanRejectedError, match="10 attributes"):
            sanitizer.validate(attrs)

    def test_rejects_oversized_attribute(self):
        config = SecurityConfig(max_attribute_bytes=100)
        sanitizer = SpanSanitizer(config)
        attrs = {"big": "x" * 200}
        with pytest.raises(SpanRejectedError, match="bytes"):
            sanitizer.validate(attrs)

    def test_rejects_deeply_nested_attributes(self):
        config = SecurityConfig(max_attribute_depth=3)
        sanitizer = SpanSanitizer(config)
        # Build nesting: {"a": {"b": {"c": {"d": 1}}}} = depth 4
        nested: dict = {"d": 1}
        for key in ["c", "b", "a"]:
            nested = {key: nested}
        with pytest.raises(SpanRejectedError, match="nesting depth"):
            sanitizer.validate(nested)

    def test_passes_at_exact_depth_limit(self):
        config = SecurityConfig(max_attribute_depth=3)
        sanitizer = SpanSanitizer(config)
        # depth 3: {"a": {"b": {"c": 1}}}
        attrs = {"a": {"b": {"c": 1}}}
        sanitizer.validate(attrs)  # Should not raise

    def test_empty_attributes_pass(self):
        sanitizer = SpanSanitizer(SecurityConfig())
        sanitizer.validate({})  # Should not raise


# ===========================================================================
# Session resolution tests
# ===========================================================================

class TestSessionResolution:

    def test_conversation_id_resolves_to_existing_session(self):
        pipeline, db = _make_pipeline()

        # Pre-create a session with a known conversation_id
        existing_session = make_session(
            session_id="sess-original",
            conversation_id="conv-1",
        )
        db.upsert_session(existing_session)

        # Ingest a span with the same conversation_id
        span = make_llm_span(conversation_id="conv-1")
        pipeline.process(span)

        # The span should have been assigned to the existing session
        assert db.spans[-1].session_id == "sess-original"

    def test_new_conversation_id_creates_new_session(self):
        pipeline, db = _make_pipeline()

        span = make_llm_span(conversation_id="conv-new")
        pipeline.process(span)

        # A new session should have been created
        assert len(db.sessions) == 1
        session = list(db.sessions.values())[0]
        assert session.conversation_id == "conv-new"

    def test_span_with_existing_session_id_keeps_it(self):
        pipeline, db = _make_pipeline()

        span = make_llm_span(session_id="my-session")
        pipeline.process(span)

        assert db.spans[-1].session_id == "my-session"

    def test_cache_write_tokens_aggregate_separately_from_reads(self):
        # Cache reads accumulate into cache_tokens; cache writes/creation into
        # cache_write_tokens. The two must never be conflated.
        pipeline, db = _make_pipeline()

        pipeline.process(make_llm_span(
            conversation_id="conv-cache", cache_tokens=100, cache_write_tokens=40,
        ))
        pipeline.process(make_llm_span(
            conversation_id="conv-cache", cache_tokens=200, cache_write_tokens=10,
        ))

        session = db.get_session_by_conversation("conv-cache")
        assert session is not None
        assert session.cache_tokens == 300            # reads only
        assert session.cache_write_tokens == 50       # writes only

    def test_span_without_session_or_conversation_gets_new_session(self):
        pipeline, db = _make_pipeline()

        span = make_llm_span(conversation_id=None)
        span.session_id = None
        pipeline.process(span)

        assert db.spans[-1].session_id is not None
        assert len(db.sessions) == 1


# ===========================================================================
# Capture content stripping tests
# ===========================================================================

class TestCaptureStripping:

    def test_prompt_content_stripped_when_capture_off(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(prompts=False))

        span = make_llm_span(extra_attributes={
            GenAIAttributes.PROMPT_CONTENT: "secret prompt",
        })
        pipeline.process(span)

        stored_span = db.spans[-1]
        assert GenAIAttributes.PROMPT_CONTENT not in stored_span.attributes

    def test_prompt_content_kept_when_capture_on(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(prompts=True))

        span = make_llm_span(extra_attributes={
            GenAIAttributes.PROMPT_CONTENT: "kept prompt",
        })
        pipeline.process(span)

        stored_span = db.spans[-1]
        assert stored_span.attributes[GenAIAttributes.PROMPT_CONTENT] == "kept prompt"

    def test_tool_output_stripped_when_capture_off(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(tool_outputs=False))

        span = make_llm_span(extra_attributes={
            GenAIAttributes.TOOL_OUTPUT: "secret output",
        })
        pipeline.process(span)

        stored_span = db.spans[-1]
        assert GenAIAttributes.TOOL_OUTPUT not in stored_span.attributes

    def test_completion_content_stripped_when_capture_off(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(completions=False))

        span = make_llm_span(extra_attributes={
            GenAIAttributes.COMPLETION_CONTENT: "secret completion",
        })
        pipeline.process(span)

        assert GenAIAttributes.COMPLETION_CONTENT not in db.spans[-1].attributes

    def test_tool_input_stripped_when_capture_off(self):
        pipeline, db = _make_pipeline(capture=CaptureConfig(tool_inputs=False))

        span = make_llm_span(extra_attributes={
            GenAIAttributes.TOOL_INPUT: "secret input",
        })
        pipeline.process(span)

        assert GenAIAttributes.TOOL_INPUT not in db.spans[-1].attributes


# ===========================================================================
# Session totals tests
# ===========================================================================

class TestSessionTotals:

    def test_session_totals_updated_after_multiple_spans(self):
        pipeline, db = _make_pipeline()

        conv_id = "conv-totals"
        for _ in range(3):
            span = make_llm_span(
                input_tokens=100,
                output_tokens=50,
                conversation_id=conv_id,
            )
            span.session_id = None  # Force session resolution
            pipeline.process(span)

        # All spans should share the same session
        session_ids = {s.session_id for s in db.spans}
        assert len(session_ids) == 1

        session = list(db.sessions.values())[0]
        assert session.input_tokens == 300
        assert session.output_tokens == 150

    def test_error_span_increments_error_count(self):
        pipeline, db = _make_pipeline()

        span = make_llm_span(status="error", conversation_id="conv-err")
        pipeline.process(span)

        session = list(db.sessions.values())[0]
        assert session.error_count == 1

    def test_tool_span_increments_tool_call_count(self):
        pipeline, db = _make_pipeline()

        span = make_tool_span(tool_name="my_tool", conversation_id="conv-tool")
        span.session_id = None
        pipeline.process(span)

        session = list(db.sessions.values())[0]
        assert session.tool_call_count == 1

    def test_cost_accumulated_in_session(self):
        pipeline, db = _make_pipeline()

        conv_id = "conv-cost"
        for _ in range(2):
            span = make_llm_span(cost_usd=0.05, conversation_id=conv_id)
            span.session_id = None
            pipeline.process(span)

        session = list(db.sessions.values())[0]
        assert session.total_cost_usd == pytest.approx(0.10)


# ===========================================================================
# Error handling tests
# ===========================================================================

class TestErrorHandling:

    def test_span_rejected_error_not_written_to_db(self):
        security = SecurityConfig(max_attributes_per_span=2)
        pipeline, db = _make_pipeline(security=security)

        span = make_llm_span(extra_attributes={
            "a": 1, "b": 2, "c": 3, "d": 4, "e": 5,
        })
        with pytest.raises(SpanRejectedError):
            pipeline.process(span)

        assert len(db.spans) == 0

    def test_hook_failure_does_not_crash_pipeline(self):
        """Post-ingest hook errors are logged, not propagated."""
        db = InMemoryBackend()
        config = TjConfig(version="1")

        class FailingCostEngine:
            def process_span(self, span: NormalizedSpan) -> None:
                raise RuntimeError("cost engine broke")

        pipeline = IngestPipeline(
            db=db,
            config=config,
            cost_engine=FailingCostEngine(),
            alert_engine=NoopAlertEngine(),
            schema_validator=NoopSchemaValidator(),
        )

        span = make_llm_span()
        # Should NOT raise even though cost engine fails
        pipeline.process(span)
        assert len(db.spans) == 1


# ===========================================================================
# Session lifecycle tests
#
# Regression coverage for the Claude Code / Codex logs path, where each
# user_prompt event is mapped to a zero-duration invoke_agent span. Treating
# those turn-start markers as session completions force-completed every live
# session on its first prompt — the dashboard showed active work as
# "completed" with 0 duration, and the drift/alert session-end hooks fired on
# every turn.
# ===========================================================================

class TestSessionLifecycle:

    def test_zero_duration_invoke_agent_marker_keeps_session_active(self):
        # Claude Code maps each user_prompt to a zero-duration invoke_agent
        # span (end_time == start_time). It marks the START of a turn.
        pipeline, db = _make_pipeline()
        marker = make_invoke_agent_span(session_id="s1", duration_ms=0.0)

        pipeline.process(marker)

        session = db.get_session("s1")
        assert session is not None
        assert session.status == "active"

    def test_streaming_activity_keeps_session_active(self):
        # A marker followed by real LLM activity is still an ongoing session.
        pipeline, db = _make_pipeline()
        pipeline.process(make_invoke_agent_span(session_id="s1", duration_ms=0.0))
        pipeline.process(make_llm_span(session_id="s1"))

        assert db.get_session("s1").status == "active"

    def test_real_invoke_agent_span_completes_session(self):
        # The SDK @watch() path emits one invoke_agent span that brackets the
        # whole run (end_time strictly after start_time). That DOES complete it.
        pipeline, db = _make_pipeline()
        end_span = make_invoke_agent_span(session_id="s1", duration_ms=5000.0)

        pipeline.process(end_span)

        assert db.get_session("s1").status == "completed"

    def test_activity_reactivates_mistakenly_completed_session(self):
        # An in-flight session left "completed" (e.g. by the old bug, or a
        # prior restart) must self-heal when new activity arrives.
        pipeline, db = _make_pipeline()
        db.upsert_session(make_session(session_id="s1", status="completed"))

        pipeline.process(make_llm_span(session_id="s1"))

        assert db.get_session("s1").status == "active"


class TestServiceNamespace:
    """service.namespace (project grouping) capture on the session."""

    def test_session_captures_service_namespace(self):
        pipeline, db = _make_pipeline()
        pipeline.process(make_llm_span(session_id="s1", service_namespace="aquanode"))

        assert db.get_session("s1").service_namespace == "aquanode"

    def test_namespace_late_resolves_from_later_span(self):
        # A tool span with no namespace creates the session; a later LLM span
        # that carries the namespace backfills it.
        pipeline, db = _make_pipeline()
        pipeline.process(make_invoke_agent_span(session_id="s1", service_namespace=None))
        assert db.get_session("s1").service_namespace is None

        pipeline.process(make_llm_span(session_id="s1", service_namespace="aquanode"))
        assert db.get_session("s1").service_namespace == "aquanode"

    def test_namespace_absent_stays_none(self):
        pipeline, db = _make_pipeline()
        pipeline.process(make_llm_span(session_id="s1"))

        assert db.get_session("s1").service_namespace is None

    def test_namespace_falls_back_to_configured_project(self):
        # An already-running agent never sends service.namespace; the agent's
        # configured project supplies it server-side (no restart needed).
        pipeline, db = _make_pipeline(
            agents={"claude-code-harness": AgentConfig(project="aquanode")},
        )
        pipeline.process(make_llm_span(agent_id="claude-code-harness", session_id="s1"))

        assert db.get_session("s1").service_namespace == "aquanode"

    def test_wire_namespace_wins_over_configured_project(self):
        pipeline, db = _make_pipeline(
            agents={"claude-code-harness": AgentConfig(project="aquanode")},
        )
        pipeline.process(make_llm_span(
            agent_id="claude-code-harness", session_id="s1",
            service_namespace="explicit-ns"))

        assert db.get_session("s1").service_namespace == "explicit-ns"

    def test_session_captures_service_instance_id(self):
        # The per-terminal instance id (e.g. "founder-os") is persisted on the
        # session for use as its display label.
        pipeline, db = _make_pipeline()
        pipeline.process(make_llm_span(session_id="s1", service_instance_id="founder-os"))

        assert db.get_session("s1").service_instance_id == "founder-os"
