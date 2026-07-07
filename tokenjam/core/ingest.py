from __future__ import annotations

import json
import logging
import queue
import threading
from typing import TYPE_CHECKING, Any

from tokenjam.core.models import NormalizedSpan, SessionRecord, SpanStatus
from tokenjam.core.config import TjConfig, SecurityConfig, CaptureConfig
from tokenjam.otel.semconv import GenAIAttributes, TjAttributes
from tokenjam.utils.ids import new_uuid
from tokenjam.utils.signatures import tool_arg_signature

if TYPE_CHECKING:
    from tokenjam.core.db import StorageBackend
    from tokenjam.core.cost import CostEngine
    from tokenjam.core.alerts import AlertEngine
    from tokenjam.core.drift import DriftDetector
    from tokenjam.core.schema_validator import SchemaValidator

logger = logging.getLogger("tokenjam.ingest")

# Bound on the async-hooks background queue. Chosen to absorb bursty ingest
# while a slow hook catches up, without letting memory grow unbounded. When the
# queue is full, _enqueue_hook drops the OLDEST queued span (see its docstring)
# and logs the drop — post-ingest hooks are advisory, so newest telemetry wins.
HOOK_QUEUE_MAXSIZE = 10_000


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


# Sampling-parameter attributes captured for full-request replay (#209). These
# describe HOW generation was requested (not message content); they ride with
# the request prompt and so are gated by the [capture] `prompts` toggle.
_REQUEST_PARAM_ATTRS: tuple[str, ...] = (
    GenAIAttributes.REQUEST_TEMPERATURE,
    GenAIAttributes.REQUEST_TOP_P,
    GenAIAttributes.REQUEST_TOP_K,
    GenAIAttributes.REQUEST_MAX_TOKENS,
    GenAIAttributes.REQUEST_STOP_SEQUENCES,
    GenAIAttributes.REQUEST_FREQUENCY_PENALTY,
    GenAIAttributes.REQUEST_PRESENCE_PENALTY,
    GenAIAttributes.REQUEST_SEED,
)


def strip_captured_content(attributes: dict, capture: CaptureConfig) -> dict:
    """Remove prompt/completion/tool content from attributes based on capture config.

    This is the single ingest gate (Critical Rule 5 / issue #209). Full-request
    capture rides through the same gate: sampling parameters are gated with the
    request `prompts` toggle, and the tools/tool_choice payload (tool-definition
    content) is gated with `tool_inputs`.

    Before dropping the raw tool input, derive a privacy-safe argument signature
    (``tokenjam.tool_arg_sig``) and keep it — this is what retry-loop detection
    uses to tell an identical repeated call from normal repeated tool use, so the
    raw (possibly sensitive) input never has to be retained.
    """
    stripped = dict(attributes)
    # Compute the arg signature from whatever input is present, regardless of the
    # capture toggle, and persist it (the hash is non-sensitive).
    sig = tool_arg_signature(stripped.get(GenAIAttributes.TOOL_INPUT))
    if sig is not None:
        stripped[TjAttributes.TOOL_ARG_SIG] = sig
    if not capture.prompts:
        stripped.pop(GenAIAttributes.PROMPT_CONTENT, None)
        for key in _REQUEST_PARAM_ATTRS:
            stripped.pop(key, None)
    if not capture.completions:
        stripped.pop(GenAIAttributes.COMPLETION_CONTENT, None)
    if not capture.tool_inputs:
        stripped.pop(GenAIAttributes.TOOL_INPUT, None)
        stripped.pop(TjAttributes.REQUEST_TOOLS, None)
    if not capture.tool_outputs:
        stripped.pop(GenAIAttributes.TOOL_OUTPUT, None)
    return stripped


def extract_request_capture(span: NormalizedSpan) -> None:
    """Project full-request capture from span attributes into structured fields.

    Runs AFTER strip_captured_content, so it only ever sees attributes the
    [capture] config permits — when a toggle is off the keys are already gone
    and the corresponding structured field stays None (no behavior change when
    capture is off, acceptance criterion #1). The keys are popped out of the
    attributes blob so the data lives in exactly one place: the structured
    request_params / request_tools columns.
    """
    params: dict[str, Any] = {}
    for key in _REQUEST_PARAM_ATTRS:
        if key in span.attributes:
            # Store under the short param name (strip the gen_ai.request. prefix).
            params[key.rsplit(".", 1)[-1]] = span.attributes.pop(key)
    if params:
        span.request_params = params

    tools = span.attributes.pop(TjAttributes.REQUEST_TOOLS, None)
    if tools is not None:
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except (ValueError, TypeError):
                tools = {"raw": tools}
        span.request_tools = tools


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

        # Background worker thread & queue for async hooks (opt-in via
        # [alerts] async_hooks = true; default OFF — see _run_hooks).
        self._hook_queue: queue.Queue | None = None
        self._hook_thread: threading.Thread | None = None
        self._hook_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._hook_dropped = 0  # count of spans dropped on queue overflow

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

        # 1b. Project the (now gated) full-request capture into structured
        # fields. Runs after the strip so it can only ever see attributes the
        # capture config permits (#209).
        extract_request_capture(span)

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
        Resolve or create a session_id for the span.

        Resolution order:
        1. Span already carries a session_id — keep it.
        2. Span carries a conversation_id that matches an existing session — use that.
        3. Span carries a trace_id that matches spans already written for a known
           session — attach to that session (#326).
        4. None of the above — mint a fresh session_id.
        """
        if span.session_id:
            return span

        if span.conversation_id:
            existing = self.db.get_session_by_conversation(span.conversation_id)
            if existing is not None:
                span.session_id = existing.session_id
                return span

        # Look up session via trace_id sibling.
        if span.trace_id:
            for sibling in self.db.get_trace_spans(span.trace_id):
                if sibling.session_id:
                    span.session_id = sibling.session_id
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
            existing.cache_write_tokens += span.cache_write_tokens or 0
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
            # Self-heal cross-session run grouping: a session created before its
            # telemetry carried the run markers (e.g. an early tool span) gets
            # them backfilled when a later span declares them. Never overwrite a
            # value already on the session.
            if existing.run_id is None and span.run_id:
                existing.run_id = span.run_id
            if existing.parent_session_id is None and span.parent_session_id:
                existing.parent_session_id = span.parent_session_id
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
            cache_write_tokens=span.cache_write_tokens or 0,
            tool_call_count=1 if span.tool_name else 0,
            error_count=1 if span.status_code == SpanStatus.ERROR else 0,
            plan_tier=plan_tier,
            service_namespace=span.service_namespace or self._resolve_project(span.agent_id),
            service_instance_id=span.service_instance_id,
            run_id=span.run_id,
            parent_session_id=span.parent_session_id,
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

        # Default OFF: a config without async_hooks set runs hooks inline on the
        # ingest thread, identical to the pre-async synchronous behavior.
        if self.config.alerts.async_hooks:
            self._enqueue_hook(span)
        else:
            self._run_deferred_hooks(span)

    def _run_deferred_hooks(self, span: NormalizedSpan) -> None:
        """Run AlertEngine and SchemaValidator hooks."""
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

    def _enqueue_hook(self, span: NormalizedSpan) -> None:
        if self._hook_queue is None:
            self._start_background_worker()
        q = self._hook_queue
        assert q is not None  # _start_background_worker guarantees this
        # Bounded queue with a drop-oldest overflow policy: under a slow hook +
        # high span volume, memory must not grow without bound. When the queue is
        # full we evict the oldest queued span to make room for the newest, and
        # log the drop (never silently) so operators can see hooks are shedding
        # load. Post-ingest hooks are advisory (alerts / schema validation), so
        # favoring the freshest telemetry is the right trade-off.
        try:
            q.put_nowait(span)
        except queue.Full:
            try:
                dropped = q.get_nowait()
                q.task_done()
                self._hook_dropped += 1
                logger.warning(
                    "async hook queue full (maxsize=%d); dropped oldest queued "
                    "span %s to enqueue newest (%d dropped total). Hooks are "
                    "falling behind span ingest.",
                    HOOK_QUEUE_MAXSIZE,
                    getattr(dropped, "span_id", "<unknown>"),
                    self._hook_dropped,
                )
            except queue.Empty:
                pass
            try:
                q.put_nowait(span)
            except queue.Full:
                # Consumer refilled the queue between our get and put; drop the
                # newest span rather than block the ingest path. Still logged.
                self._hook_dropped += 1
                logger.warning(
                    "async hook queue still full; dropped newest span %s "
                    "(%d dropped total).",
                    getattr(span, "span_id", "<unknown>"),
                    self._hook_dropped,
                )

    def _start_background_worker(self) -> None:
        with self._hook_lock:
            if self._hook_queue is None:
                self._hook_queue = queue.Queue(maxsize=HOOK_QUEUE_MAXSIZE)
                self._shutdown_event.clear()
                self._hook_thread = threading.Thread(
                    target=self._worker_loop,
                    name="TjHookWorker",
                    daemon=True,
                )
                self._hook_thread.start()

    def _worker_loop(self) -> None:
        # NOTE: the loop deliberately does NOT check _shutdown_event at the top of
        # each iteration. On shutdown, close() sets the event AND enqueues a None
        # sentinel; the worker keeps draining until it reaches the sentinel, so
        # every span already queued gets its hooks run before the thread exits.
        # This is what makes flush()+close() lossless (blocker: shutdown dropped
        # queued alerts).
        q = self._hook_queue
        assert q is not None  # only started after the queue exists
        while True:
            try:
                span = q.get(timeout=0.1)
            except queue.Empty:
                # No work pending. Only exit on the empty queue once shutdown was
                # requested — otherwise keep waiting for new spans.
                if self._shutdown_event.is_set():
                    break
                continue

            if span is None:  # Shutdown sentinel — queue is drained, stop.
                q.task_done()
                break

            try:
                self._run_deferred_hooks(span)
            finally:
                q.task_done()

    def flush(self) -> None:
        """Block until all currently-queued hooks have been processed.

        Guarded so it can never hang forever: if the worker thread has already
        exited (e.g. close() ran first, or it was never started), there is no
        live consumer to drain the queue, so joining would block indefinitely.
        In that case we return immediately.
        """
        q = self._hook_queue
        t = self._hook_thread
        if q is None:
            return
        if t is None or not t.is_alive():
            return
        q.join()

    def close(self) -> None:
        """Flush queued hooks, then shut down the background worker thread.

        Drains the queue before exiting (via the sentinel + non-top-of-loop
        shutdown check in _worker_loop), so no queued alert is lost on exit.
        Idempotent and safe to call when async hooks were never enabled.
        """
        thread = self._hook_thread
        q = self._hook_queue
        if thread is None or q is None:
            return
        # Drain everything already queued before signalling stop.
        self.flush()
        self._shutdown_event.set()
        try:
            q.put_nowait(None)  # Sentinel; queue has room after the flush.
        except queue.Full:
            # Extremely unlikely (queue was just drained), but never block: the
            # top-of-empty shutdown check will still let the worker exit.
            pass
        thread.join(timeout=10.0)
        self._hook_queue = None
        self._hook_thread = None


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
