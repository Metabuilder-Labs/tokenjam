from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonschema
from genson import SchemaBuilder

from tokenjam.core.models import SchemaValidationResult, AlertType, Severity
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.utils.ids import new_uuid
from tokenjam.utils.time_parse import utcnow

if TYPE_CHECKING:
    from tokenjam.core.db import StorageBackend
    from tokenjam.core.models import NormalizedSpan
    from tokenjam.core.alerts import AlertEngine
    from tokenjam.core.config import TjConfig

logger = logging.getLogger("tokenjam.schema")


class SchemaValidator:
    """
    Post-ingest hook. Called by IngestPipeline after each span.

    Only runs when ALL of these are true:
    1. span.name == "gen_ai.tool.call"
    2. span has gen_ai.tool.output in its attributes (capture.tool_outputs = true)
    3. The agent has an output_schema configured, OR an inferred schema exists in baseline

    If output_schema is a file path in config, load the JSON Schema from that file.
    If no schema is declared, use inferred schema from drift baseline.
    If neither exists yet, silently skip.
    """

    def __init__(self, db: StorageBackend, alert_engine: AlertEngine, config: TjConfig):
        self.db = db
        self.alert_engine = alert_engine
        self.config = config
        self._schema_cache: dict[str, dict] = {}

    def validate(self, span: NormalizedSpan) -> None:
        """
        Validate tool output against schema if applicable.
        Persists a SchemaValidationResult to DB regardless of pass/fail.
        Fires SCHEMA_VIOLATION alert on failure.
        """
        if span.name != GenAIAttributes.SPAN_TOOL_CALL:
            return

        tool_output = span.attributes.get(GenAIAttributes.TOOL_OUTPUT)
        if tool_output is None:
            return

        schema = self._get_schema(span.agent_id)
        if schema is None:
            return

        # Parse tool_output if it's a string
        parsed_output = tool_output
        if isinstance(parsed_output, str):
            try:
                parsed_output = json.loads(parsed_output)
            except (json.JSONDecodeError, TypeError):
                pass  # Validate the raw string against the schema

        errors = list(jsonschema.Draft7Validator(schema).iter_errors(parsed_output))

        result = SchemaValidationResult(
            validation_id=new_uuid(),
            span_id=span.span_id,
            agent_id=span.agent_id,
            validated_at=utcnow(),
            passed=len(errors) == 0,
            errors=[e.message for e in errors],
        )
        self.db.insert_validation(result)

        if not result.passed:
            self.alert_engine.fire(
                alert_type=AlertType.SCHEMA_VIOLATION,
                span_or_session=span,
                detail={"errors": result.errors, "tool_name": span.tool_name},
                severity=Severity.WARNING,
            )

    def _get_schema(self, agent_id: str | None) -> dict | None:
        """
        Return the JSON Schema for this agent, or None if unavailable.
        Priority: 1) declared schema file in config, 2) inferred schema in baseline.
        Caches loaded schemas in-memory.
        """
        if agent_id is None:
            return None

        if agent_id in self._schema_cache:
            return self._schema_cache[agent_id]

        # 1. Check declared schema in agent config
        agent_cfg = self.config.agents.get(agent_id)
        if agent_cfg and agent_cfg.output_schema:
            schema = self._load_schema_file(agent_cfg.output_schema)
            if schema is not None:
                self._schema_cache[agent_id] = schema
                return schema

        # 2. Fall back to inferred schema from drift baseline
        baseline = self.db.get_baseline(agent_id)
        if baseline and baseline.output_schema_inferred:
            self._schema_cache[agent_id] = baseline.output_schema_inferred
            return baseline.output_schema_inferred

        return None

    def _load_schema_file(self, path_str: str) -> dict | None:
        """Load a JSON Schema from a file path.

        Relative paths are resolved relative to the config file's parent directory
        (config.config_path). Falls back to CWD-relative resolution when no config
        file path is available (e.g. in tests with a synthetic TjConfig).
        """
        path = Path(path_str)
        if not path.is_absolute():
            config_file_path = getattr(self.config, "config_path", None)
            if config_file_path is not None:
                path = Path(config_file_path).parent / path_str
        if not path.exists():
            logger.warning("Schema file not found: %s", path)
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load schema file %s: %s", path, exc)
            return None

    def infer_schema_from_outputs(self, tool_outputs: list[Any]) -> dict | None:
        """
        Use genson to infer a JSON Schema from a list of tool outputs.
        Returns None if the list is empty.
        """
        if not tool_outputs:
            return None

        builder = SchemaBuilder()
        for output in tool_outputs:
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except (json.JSONDecodeError, TypeError):
                    pass
            builder.add_object(output)
        return builder.to_schema()
