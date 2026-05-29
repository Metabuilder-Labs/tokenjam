from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from tokenjam.core.models import NormalizedSpan, SessionRecord, SpanStatus
from tokenjam.core.config import TjConfig, SecurityConfig, CaptureConfig
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.utils.ids import new_uuid

if TYPE_CHECKING:
    from tokenjam.core.db import StorageBackend
    from tokenjam.core.cost import CostEngine
    from tokenjam.core.alerts import AlertEngine
    from tokenjam.core.drift import DriftDetector
    from tokenjam.core.schema_validator import SchemaValidator

logger = logging.getLogger("tokenjam.ingest")


class SpanRejectedError(Exception):
    """Raised when a span fails sanitization. The span is not written to DB."""


class SpanSanitizer:
    """
    Validates spans before they are written to the database.
    Rejects — never silently truncates — spans that violate limits.
    """

    def __init__(self, config: SecurityConfig):
        self.config = config

    def validate(self, raw_attributes: dict, source: str = "unknown") -> None:
        """
        Raises SpanRejectedError if:
        - The number of attributes exceeds max_attributes_per_span
        - Any attribute value serialises to more than max_attribute_bytes bytes
        - The JSON nesting depth exceeds max_attribute_depth
        """
        if len(raw_attributes) > self.config.max_attributes_per_span:
            raise SpanRejectedError(
                f"Span from {source} has {len(raw_attributes)} attributes "
                f"(max {self.config.max_attributes_per_span})"
            )
        for key, value in raw_attributes.items():
            serialised = json.dumps(value).encode()
            if len(serialised) > self.config.max_attribute_bytes:
                raise SpanRejectedError(
                    f"Attribute '{key}' in span from {source} is "
                    f"{len(serialised)} bytes (max {self.config.max_attribute_bytes})"
                )
        depth = _json_depth(raw_attributes)
        if depth > self.config.max_attribute_depth:
            raise SpanRejectedError(
                f"Span from {source} has attribute nesting depth {depth} "
                f"(max {self.config.max_attribute_depth})"
            )


def _json_depth(obj: object, current: int = 0) -> int:
    """Return the maximum nesting depth of a JSON-serialisable object."""
    if isinstance(obj, dict):
        if not obj:
            return current
        return max(_json_depth(v, current + 1) for v in obj.values())
    if isinstance(obj, list):
        if not obj:
            return current
        return max(_json_depth(item, current + 1) for item in obj)
    return current


def strip_captured_content(attributes: dict, capture: CaptureConfig) -> dict:
    """Remove prompt/completion/tool content from attributes based on capture config."""
    stripped = dict(attributes)
    if not capture.prompts:
        stripped.pop(GenAIAttributes.PROMPT_CONTENT, None)
    if not capture.completions:
        stripped.pop(GenAIAttributes.COMPLETION_CONTENT, None)
    if not capture.tool_inputs:
        stripped.pop(GenAIAttributes.TOOL_INPUT, None)
    if not capture.tool_outputs:
        stripped.pop(GenAIAttributes.TOOL_OUTPUT, None)
    return stripped


class IngestPipeline:
    """
    Central ingest hub. All spans — whether from the Python SDK's TjSpanExporter
    or from the REST API — flow through here.

    Post-ingest hooks run synchronously after the span is written to DB:
      1. CostEngine.process_span() — calculates and records cost
      2. AlertEngine.evaluate() — checks all alert rules
      3. SchemaValidator.validate() — checks tool outputs against schema
    """

    def __init__(
        self,
        db: StorageBackend,
        config: TjConfig,
        cost_engine: CostEngine | None = None,
        alert_engine: AlertEngine | None = None,
        schema_validator: SchemaValidator | None = None,
        drift_detector: DriftDetector | None = None,
    ):
        self.db = db
        self.config = config
        self.sanitizer = SpanSanitizer(config.security)
        self.cost_engine = cost_engine
        self.alert_engine = alert_engine
        self.schema_validator = schema_validator
        self.drift_detector = drift_detector

    def process(self, span: NormalizedSpan) -> None:
        """
        Full ingest pipeline for one span:
        1. Strip captured content per config
        2. Sanitize attributes
        3. Resolve or create session (using conversation_id if present)
        4. Write span to DB
        5. Upsert session totals
        6. Run post-ingest hooks
        """
        # 1. Strip captured content before sanitization/storage
        span.attributes = strip_captured_content(span.attributes, self.config.capture)

        # 2. Sanitize
        self.sanitizer.validate(span.attributes, source=span.agent_id or "unknown")

        # 3. Session resolution
        span = self._resolve_session(span)

        # Normalize agent_id so spans and sessions always agree
        span.agent_id = span.agent_id or "unknown"

        # 4. Write span
        self.db.insert_span(span)

        # 5. Session upsert (update running totals)
        session = self._build_or_update_session(span)

        # 5b. Session lifecycle.
        #
        # A session is "completed" only when we see a *real* session-wrapping
        # invoke_agent span — one whose end_time is strictly after its
        # start_time. The SDK `@watch()` path emits exactly that: one span that
        # brackets the whole agent run.
        #
        # The Claude Code / Codex logs path instead maps each `user_prompt`
        # event to a zero-duration invoke_agent span (end_time == start_time)
        # that marks the *start* of a turn, not the end of the session.
        # Treating those markers as completions was the bug: every live session
        # was force-completed on its first prompt (so the dashboard showed
        # active work as "completed" with 0 duration), and the drift/alert
        # session-end hooks fired on every single turn.
        if self._is_session_end(span):
            session.status = "completed"
            self.db.upsert_session(session)
            if self.drift_detector and span.agent_id:
                try:
                    self.drift_detector.on_session_end(span.agent_id, session)
                except Exception as exc:
                    logger.warning("DriftDetector hook failed: %s", exc)
            if self.alert_engine:
                try:
                    self.alert_engine.evaluate_session_end(session)
                except Exception as exc:
                    logger.warning("AlertEngine session-end hook failed: %s", exc)
        else:
            # Any other span is ongoing activity. Streaming telemetry (the logs
            # path) never sends an explicit end event, so a session that keeps
            # receiving spans is still alive — re-activate a record that was
            # previously (mis)marked completed. Genuinely idle sessions are
            # surfaced as "stale" at read time via
            # SessionRecord.effective_status (SESSION_STALE_THRESHOLD).
            if session.status != "active":
                session.status = "active"
            self.db.upsert_session(session)

        # 6. Post-ingest hooks (never let hook errors kill the pipeline)
        self._run_hooks(span)

    @staticmethod
    def _is_session_end(span: NormalizedSpan) -> bool:
        """True when the span is a real session-wrapping invoke_agent span.

        Distinguishes the SDK `@watch()` session span (real duration) from the
        zero-duration invoke_agent markers the Claude Code / Codex logs path
        emits at the start of every turn (end_time == start_time).
        """
        return (
            span.name == GenAIAttributes.SPAN_INVOKE_AGENT
            and span.end_time is not None
            and span.start_time is not None
            and span.end_time > span.start_time
        )

    def _resolve_session(self, span: NormalizedSpan) -> NormalizedSpan:
        """
        If the span has a conversation_id and a matching session exists,
        use that session_id. Otherwise create a new session_id.
        """
        if span.session_id:
            return span

        if span.conversation_id:
            existing = self.db.get_session_by_conversation(span.conversation_id)
            if existing is not None:
                span.session_id = existing.session_id
                return span

        # No existing session found — create a new session_id
        span.session_id = new_uuid()
        return span

    def _build_or_update_session(self, span: NormalizedSpan) -> SessionRecord:
        """
        Fetch the current session record and update its running totals
        from this span's token counts, cost, error status, etc.

        plan_tier resolution: derived from ProviderBudget.plan for the
        session's billing_account. Set at session creation; subsequent spans
        only update plan_tier if it's currently 'unknown' (e.g. tool spans
        arrived before an LLM span on a fresh session).
        """
        assert span.session_id is not None

        existing = self.db.get_session(span.session_id)
        if existing is not None:
            existing.input_tokens += span.input_tokens or 0
            existing.output_tokens += span.output_tokens or 0
            existing.cache_tokens += span.cache_tokens or 0
            if span.cost_usd is not None:
                existing.total_cost_usd = (existing.total_cost_usd or 0.0) + span.cost_usd
            if span.tool_name:
                existing.tool_call_count += 1
            if span.status_code == SpanStatus.ERROR:
                existing.error_count += 1
            # Update end time to track session duration
            if span.end_time and (existing.ended_at is None or span.end_time > existing.ended_at):
                existing.ended_at = span.end_time
            # Late-resolve plan_tier if this span finally carries a known
            # billing_account and the session was previously unknown.
            if existing.plan_tier == "unknown" and span.billing_account:
                resolved = self._resolve_plan_tier(span.billing_account)
                if resolved != "unknown":
                    existing.plan_tier = resolved
            # Late-resolve service_namespace: from the span if it now carries
            # one, otherwise from the agent's configured project (server-side
            # fallback for agents that never send service.namespace).
            if existing.service_namespace is None:
                resolved_ns = span.service_namespace or self._resolve_project(span.agent_id)
                if resolved_ns:
                    existing.service_namespace = resolved_ns
            # Late-resolve the per-terminal instance id (display label).
            if existing.service_instance_id is None and span.service_instance_id:
                existing.service_instance_id = span.service_instance_id
            return existing

        # New session
        plan_tier = self._resolve_plan_tier(span.billing_account)
        return SessionRecord(
            session_id=span.session_id,
            agent_id=span.agent_id or "unknown",
            started_at=span.start_time,
            ended_at=span.end_time,
            conversation_id=span.conversation_id,
            status="active",
            total_cost_usd=span.cost_usd,
            input_tokens=span.input_tokens or 0,
            output_tokens=span.output_tokens or 0,
            cache_tokens=span.cache_tokens or 0,
            tool_call_count=1 if span.tool_name else 0,
            error_count=1 if span.status_code == SpanStatus.ERROR else 0,
            plan_tier=plan_tier,
            service_namespace=span.service_namespace or self._resolve_project(span.agent_id),
            service_instance_id=span.service_instance_id,
        )

    def _resolve_project(self, agent_id: str | None) -> str | None:
        """Project name configured for this agent (``[agents.<id>].project``).

        Server-side fallback for service.namespace so sessions group by project
        even when the agent never sends service.namespace on the wire (e.g. an
        already-running Claude Code session whose env was fixed at startup).
        """
        if not agent_id:
            return None
        agent_cfg = self.config.agents.get(agent_id)
        return agent_cfg.project if agent_cfg else None

    def _resolve_plan_tier(self, billing_account: str | None) -> str:
        """
        Look up ProviderBudget.plan for the given billing_account.

        Returns 'unknown' when billing_account is None (e.g. tool spans),
        when no ProviderBudget is configured for the provider, or when the
        ProviderBudget exists but has no plan set. Onboarding writes the
        plan field; `tj optimize` suppresses dollar figures for unknown.

        Special case: billing_account 'local.ollama' always resolves to
        'local' regardless of config — local inference has no plan tier.
        """
        if not billing_account:
            return "unknown"
        if billing_account == "local.ollama":
            return "local"
        bcfg = self.config.budgets.get(billing_account)
        if bcfg is None or not bcfg.plan:
            return "unknown"
        return bcfg.plan

    def _run_hooks(self, span: NormalizedSpan) -> None:
        """Run post-ingest hooks. Errors are logged, never propagated."""
        if self.cost_engine is not None:
            try:
                self.cost_engine.process_span(span)
            except Exception as exc:
                logger.warning("CostEngine hook failed: %s", exc)

        if self.alert_engine is not None:
            try:
                self.alert_engine.evaluate(span)
            except Exception as exc:
                logger.warning("AlertEngine hook failed: %s", exc)

        if self.schema_validator is not None:
            try:
                self.schema_validator.validate(span)
            except Exception as exc:
                logger.warning("SchemaValidator hook failed: %s", exc)


def build_default_pipeline(db: "StorageBackend", config: TjConfig) -> "IngestPipeline":
    """Construct an IngestPipeline with all standard post-ingest hooks wired up.

    Used by both `tj serve` and the SDK auto-bootstrap so alerts, drift detection,
    and schema validation work uniformly regardless of how spans enter the system.
    """
    from tokenjam.core.alerts import AlertEngine
    from tokenjam.core.cost import CostEngine
    from tokenjam.core.drift import DriftDetector
    from tokenjam.core.schema_validator import SchemaValidator

    cost_engine = CostEngine(db)
    alert_engine = AlertEngine(db=db, config=config)
    drift_detector = DriftDetector(db=db, alert_engine=alert_engine, config=config)
    schema_validator = SchemaValidator(db=db, alert_engine=alert_engine, config=config)

    return IngestPipeline(
        db=db,
        config=config,
        cost_engine=cost_engine,
        alert_engine=alert_engine,
        drift_detector=drift_detector,
        schema_validator=schema_validator,
    )
