"""Tests for ocw.core.schema_validator — schema validation, inference, skipping."""
from __future__ import annotations

import json
from typing import Any

import pytest

from tokenjam.core.config import TjConfig, AgentConfig
from tokenjam.core.models import (
    AlertType,
    DriftBaseline,
    NormalizedSpan,
    SchemaValidationResult,
    Severity,
)
from tokenjam.core.schema_validator import SchemaValidator
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_tool_span


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class InMemoryBackend:
    """Minimal stub for StorageBackend."""

    def __init__(self) -> None:
        self.validations: list[SchemaValidationResult] = []
        self.baselines: dict[str, DriftBaseline] = {}

    def insert_validation(self, result: SchemaValidationResult) -> None:
        self.validations.append(result)

    def get_baseline(self, agent_id: str) -> DriftBaseline | None:
        return self.baselines.get(agent_id)


class RecordingAlertEngine:
    """Stub AlertEngine that records fired alerts."""

    def __init__(self) -> None:
        self.fired: list[dict] = []

    def fire(
        self,
        alert_type: AlertType,
        span_or_session: Any,
        detail: dict,
        severity: Severity | None = None,
    ) -> None:
        self.fired.append({
            "alert_type": alert_type,
            "detail": detail,
            "severity": severity,
        })

    def evaluate(self, span: NormalizedSpan) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {"type": "string"},
        "score": {"type": "number"},
    },
    "required": ["result"],
}


def _make_validator(
    *,
    schema: dict | None = None,
    agent_id: str = "test-agent",
    baseline_schema: dict | None = None,
    schema_file: str | None = None,
) -> tuple[SchemaValidator, InMemoryBackend, RecordingAlertEngine]:
    db = InMemoryBackend()
    alert_engine = RecordingAlertEngine()

    agents = {}
    if schema_file:
        agents[agent_id] = AgentConfig(output_schema=schema_file)

    config = TjConfig(version="1", agents=agents)

    if baseline_schema:
        db.baselines[agent_id] = DriftBaseline(
            agent_id=agent_id,
            sessions_sampled=10,
            computed_at=utcnow(),
            output_schema_inferred=baseline_schema,
        )

    validator = SchemaValidator(db=db, alert_engine=alert_engine, config=config)

    # If a direct schema is passed, inject it into the cache
    if schema:
        validator._schema_cache[agent_id] = schema

    return validator, db, alert_engine


def _tool_span_with_output(output: Any, agent_id: str = "test-agent") -> NormalizedSpan:
    """Create a tool span with gen_ai.tool.output in attributes."""
    span = make_tool_span(agent_id=agent_id, tool_name="test_tool")
    span.attributes[GenAIAttributes.TOOL_OUTPUT] = output
    return span


# ===========================================================================
# Validation pass/fail tests
# ===========================================================================

class TestSchemaValidation:

    def test_valid_output_passes(self):
        validator, db, alerts = _make_validator(schema=_SIMPLE_SCHEMA)
        span = _tool_span_with_output({"result": "success", "score": 0.95})

        validator.validate(span)

        assert len(db.validations) == 1
        assert db.validations[0].passed is True
        assert db.validations[0].errors == []
        assert len(alerts.fired) == 0

    def test_invalid_output_fires_alert(self):
        validator, db, alerts = _make_validator(schema=_SIMPLE_SCHEMA)
        # Missing required "result" field
        span = _tool_span_with_output({"score": 0.5})

        validator.validate(span)

        assert len(alerts.fired) == 1
        assert alerts.fired[0]["alert_type"] == AlertType.SCHEMA_VIOLATION
        assert alerts.fired[0]["severity"] == Severity.WARNING
        assert "result" in alerts.fired[0]["detail"]["errors"][0].lower()

    def test_invalid_output_persists_validation_result(self):
        validator, db, alerts = _make_validator(schema=_SIMPLE_SCHEMA)
        span = _tool_span_with_output({"score": "not a number"})

        validator.validate(span)

        assert len(db.validations) == 1
        result = db.validations[0]
        assert result.passed is False
        assert len(result.errors) > 0
        assert result.span_id == span.span_id
        assert result.agent_id == span.agent_id

    def test_string_output_parsed_as_json(self):
        validator, db, alerts = _make_validator(schema=_SIMPLE_SCHEMA)
        span = _tool_span_with_output(json.dumps({"result": "ok"}))

        validator.validate(span)

        assert len(db.validations) == 1
        assert db.validations[0].passed is True


# ===========================================================================
# Skipping conditions tests
# ===========================================================================

class TestSchemaSkipping:

    def test_skipped_when_tool_output_not_captured(self):
        """If gen_ai.tool.output is not in attributes, validation is skipped."""
        validator, db, alerts = _make_validator(schema=_SIMPLE_SCHEMA)
        span = make_tool_span(tool_name="test_tool")
        # No TOOL_OUTPUT in attributes

        validator.validate(span)

        assert len(db.validations) == 0
        assert len(alerts.fired) == 0

    def test_skipped_when_no_schema_available(self):
        """If no schema is declared or inferred, validation is skipped."""
        validator, db, alerts = _make_validator()  # No schema
        span = _tool_span_with_output({"anything": "goes"})

        validator.validate(span)

        assert len(db.validations) == 0
        assert len(alerts.fired) == 0

    def test_skipped_for_non_tool_spans(self):
        """LLM call spans should not trigger schema validation."""
        validator, db, alerts = _make_validator(schema=_SIMPLE_SCHEMA)
        span = make_llm_span()  # name is "gen_ai.llm.call", not "gen_ai.tool.call"

        validator.validate(span)

        assert len(db.validations) == 0


# ===========================================================================
# Schema source tests
# ===========================================================================

class TestSchemaSource:

    def test_uses_baseline_inferred_schema(self):
        """When no declared schema, falls back to baseline's inferred schema."""
        inferred = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        validator, db, alerts = _make_validator(baseline_schema=inferred)
        span = _tool_span_with_output({"value": 42})

        validator.validate(span)

        assert len(db.validations) == 1
        assert db.validations[0].passed is True

    def test_baseline_schema_catches_invalid(self):
        inferred = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        validator, db, alerts = _make_validator(baseline_schema=inferred)
        span = _tool_span_with_output({"value": "not an int"})

        validator.validate(span)

        assert len(db.validations) == 1
        assert db.validations[0].passed is False
        assert len(alerts.fired) == 1

    def test_schema_cached_after_first_lookup(self):
        """Schema should be loaded once and cached."""
        inferred = {"type": "object"}
        validator, db, alerts = _make_validator(baseline_schema=inferred)
        span1 = _tool_span_with_output({"a": 1})
        span2 = _tool_span_with_output({"b": 2})

        validator.validate(span1)
        validator.validate(span2)

        assert "test-agent" in validator._schema_cache


# ===========================================================================
# Schema inference tests
# ===========================================================================

class TestSchemaInference:

    def test_infer_schema_produces_valid_schema(self):
        validator, _, _ = _make_validator()
        outputs = [
            {"result": "ok", "score": 0.9},
            {"result": "fail", "score": 0.1},
        ]

        schema = validator.infer_schema_from_outputs(outputs)

        assert schema is not None
        assert schema.get("type") == "object"
        assert "result" in schema.get("properties", {})
        assert "score" in schema.get("properties", {})

    def test_infer_schema_returns_none_when_no_outputs(self):
        validator, _, _ = _make_validator()
        schema = validator.infer_schema_from_outputs([])
        assert schema is None

    def test_infer_schema_handles_string_json_outputs(self):
        validator, _, _ = _make_validator()
        outputs = [
            json.dumps({"key": "val1"}),
            json.dumps({"key": "val2"}),
        ]

        schema = validator.infer_schema_from_outputs(outputs)

        assert schema is not None
        assert "key" in schema.get("properties", {})
