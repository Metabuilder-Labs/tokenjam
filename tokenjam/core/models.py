from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Sessions with no spans for this long are no longer "active" (live terminal).
SESSION_STALE_THRESHOLD = timedelta(minutes=5)
# Default idle window: active sessions quieter than the stale threshold but
# within this window are "idle" (paused, likely to resume); beyond it they are
# "stale" (zombie). Overridable per-install via [sessions] idle_minutes, applied
# at the status route — effective_status itself stays config-free.
SESSION_IDLE_THRESHOLD = timedelta(hours=4)


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING  = "warning"
    INFO     = "info"


class AlertType(str, Enum):
    COST_BUDGET_DAILY        = "cost_budget_daily"
    COST_BUDGET_SESSION      = "cost_budget_session"
    SENSITIVE_ACTION         = "sensitive_action"
    RETRY_LOOP               = "retry_loop"
    TOKEN_ANOMALY            = "token_anomaly"
    SESSION_DURATION         = "session_duration"
    SCHEMA_VIOLATION         = "schema_violation"
    DRIFT_DETECTED           = "drift_detected"
    FAILURE_RATE             = "failure_rate"
    NETWORK_EGRESS_BLOCKED   = "network_egress_blocked"
    FILESYSTEM_ACCESS_DENIED = "filesystem_access_denied"
    SYSCALL_DENIED           = "syscall_denied"
    INFERENCE_REROUTED       = "inference_rerouted"


class SpanStatus(str, Enum):
    OK    = "ok"
    ERROR = "error"
    UNSET = "unset"


class SpanKind(str, Enum):
    INTERNAL  = "internal"
    CLIENT    = "client"
    SERVER    = "server"
    PRODUCER  = "producer"
    CONSUMER  = "consumer"


@dataclass
class NormalizedSpan:
    span_id:        str
    trace_id:       str
    name:           str
    kind:           SpanKind
    status_code:    SpanStatus
    start_time:     datetime
    parent_span_id: str | None     = None
    session_id:     str | None     = None
    agent_id:       str | None     = None
    # Claude Code subagent (Task-tool / sidechain) identity. Set during backfill
    # from a record's top-level `agentId` when `isSidechain` is true; None for
    # main-thread spans and non-Claude-Code telemetry. Lets a session's cost be
    # broken down per subagent.
    sub_agent_id:   str | None     = None
    end_time:       datetime | None = None
    duration_ms:    float | None   = None
    status_message: str | None     = None
    attributes:     dict[str, Any] = field(default_factory=dict)
    events:         list[dict]     = field(default_factory=list)
    # Extracted indexed fields
    provider:       str | None     = None
    model:          str | None     = None
    tool_name:      str | None     = None
    input_tokens:   int | None     = None
    output_tokens:  int | None     = None
    cache_tokens:   int | None     = None
    # cache_tokens counts cache-READ tokens; cache_write_tokens counts
    # cache-CREATION tokens, which are priced at a higher rate. They are kept
    # separate so cost can charge each at its own rate (see calculate_cost).
    cache_write_tokens: int | None = None
    cost_usd:       float | None   = None
    request_type:   str | None     = None
    conversation_id: str | None    = None
    # Provider-only billing identifier (anthropic | openai | google | bedrock
    # | local.ollama). Plan tier lives on SessionRecord, not here.
    billing_account: str | None    = None
    # Full-request capture (issue #209) — makes a span self-contained enough to
    # replay. `request_params` holds sampling parameters (temperature, top_p,
    # max_tokens, stop_sequences, …); `request_tools` holds the tools /
    # tool_choice payload as {"tools": [...], "tool_choice": ...}. Both are
    # populated only when the corresponding [capture] toggle is on (sampling
    # params gate with `prompts`, tools gate with `tool_inputs`) — see
    # strip_captured_content() in core/ingest.py. Round-trip via JSON columns.
    request_params: dict[str, Any] | None = None
    request_tools:  dict[str, Any] | None = None
    # OTel service.namespace — the logical "project" this service rolls up
    # under (e.g. all Aquanodeio/* repos -> "aquanode"). Transient on the span;
    # persisted on the session it creates so the dashboard can group by it.
    service_namespace: str | None  = None
    # OTel service.instance.id — the per-terminal/process label (e.g.
    # "founder-os"). Persisted on the session for use as its display name.
    service_instance_id: str | None = None
    # Cross-session run grouping (tokenjam.run_id resource attribute). One id
    # per fan-out harness run, shared by all its workers. Transient on the
    # span; persisted on the session it creates so the dashboard can group runs.
    run_id:           str | None    = None
    # Optional spawning-session id (tokenjam.parent_session_id) for nested
    # spawns. Transient on the span; persisted on the session for the run tree.
    parent_session_id: str | None   = None


@dataclass
class SessionRecord:
    session_id:      str
    agent_id:        str
    started_at:      datetime
    conversation_id: str | None   = None
    ended_at:        datetime | None = None
    status:          str          = "active"
    total_cost_usd:  float | None = None
    input_tokens:    int          = 0
    output_tokens:   int          = 0
    # cache_tokens = cache reads (reused); cache_write_tokens = cache writes/
    # creation. Kept separate; the dashboard's "Cache tokens" shows their sum.
    cache_tokens:    int          = 0
    cache_write_tokens: int       = 0
    tool_call_count: int          = 0
    error_count:     int          = 0
    # Canonical plan-tier identifier for the user's billing relationship with
    # this session's provider. Set at session creation by reading
    # ProviderBudget.plan for the matching billing_account. Backfilled sessions
    # default to "unknown" — `tj optimize` suppresses dollar figures for those.
    # Valid values: see VALID_PLAN_TIERS in tokenjam.otel.semconv.
    plan_tier:       str          = "unknown"
    # OTel service.namespace — the logical "project" this session's service
    # rolls up under (e.g. repo `Aquanodeio/harness` -> namespace "aquanode").
    # Drives dashboard grouping. None when the telemetry carried no namespace.
    service_namespace: str | None = None
    # OTel service.instance.id — the per-terminal label (e.g. "founder-os").
    # Used as the session's display name when set; None otherwise.
    service_instance_id: str | None = None
    # Cross-session run grouping. `run_id` ties this session to all the other
    # sessions a fan-out harness spawned in the same run (declared via the
    # tokenjam.run_id resource attribute, not inferred). `parent_session_id`
    # is the optional spawning-session id for nested spawns; the run view uses
    # it to render a parent tree (flat list when no parent edges exist).
    run_id:           str | None    = None
    parent_session_id: str | None   = None

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    @property
    def effective_status(self) -> str:
        """Lifecycle tier for the dashboard (pure: uses module-default windows).

        Tiers:
          closed     -> explicitly ended (status='closed')
          completed  -> wrapped by a real session span (status='completed')
          active     -> last activity within SESSION_STALE_THRESHOLD (5 min)
          idle       -> within SESSION_IDLE_THRESHOLD (default 4h)
          stale      -> older than the idle window (zombie)

        The status route honours the configurable [sessions] idle_minutes via
        status_at(); this property stays config-free for use everywhere else.
        """
        return self.status_at(SESSION_IDLE_THRESHOLD)

    def status_at(self, idle_threshold: timedelta = SESSION_IDLE_THRESHOLD) -> str:
        """effective_status with a caller-supplied idle window.

        Separate from the property so the status route can apply a config-driven
        idle window while effective_status itself remains pure.
        """
        if self.status != "active":
            # closed / completed (or any other terminal state) pass through.
            return self.status
        from tokenjam.utils.time_parse import utcnow
        last_activity = self.ended_at or self.started_at
        if not last_activity:
            return "active"
        gap = utcnow() - last_activity
        if gap <= SESSION_STALE_THRESHOLD:
            return "active"
        if gap <= idle_threshold:
            return "idle"
        return "stale"

    def status_with_transcript_mtime(
        self,
        transcript_mtime: datetime | None,
        idle_threshold: timedelta = SESSION_IDLE_THRESHOLD,
    ) -> str:
        """status_at, but rescued to 'active' by a fresh transcript mtime.

        Claude Code spans are backfilled periodically, so a *live* CC session
        can drift to ``idle``/``stale`` once its last backfilled span ages past
        the stale threshold — even though its on-disk transcript is still being
        appended to. When the deterministic status is idle/stale and the
        transcript was modified within ``SESSION_STALE_THRESHOLD``, report
        ``active`` instead. Terminal states (closed/completed) and genuinely
        active sessions pass through untouched, so non-CC sessions (which have
        no transcript: ``transcript_mtime is None``) are unaffected.
        """
        base = self.status_at(idle_threshold)
        if base not in ("idle", "stale") or transcript_mtime is None:
            return base
        from tokenjam.utils.time_parse import utcnow
        if utcnow() - transcript_mtime <= SESSION_STALE_THRESHOLD:
            return "active"
        return base

    @property
    def pricing_mode(self) -> str:
        """
        Derived pricing mode: 'local' | 'subscription' | 'api' | 'unknown'.

        Branches evaluated top-to-bottom; first match wins:
          1. local if plan_tier == 'local'
          2. subscription if plan_tier is in SUBSCRIPTION_PLAN_TIERS
          3. api if plan_tier == 'api'
          4. unknown otherwise

        Note: 'local' here keys off plan_tier (set at session creation from the
        billing_account 'local.ollama' -> plan 'local'). This avoids reading
        the underlying span's billing_account every time pricing_mode is needed.
        """
        from tokenjam.otel.semconv import SUBSCRIPTION_PLAN_TIERS
        pt = self.plan_tier
        if pt == "local":
            return "local"
        if pt in SUBSCRIPTION_PLAN_TIERS:
            return "subscription"
        if pt == "api":
            return "api"
        return "unknown"


@dataclass
class AgentRecord:
    agent_id:   str
    first_seen: datetime
    last_seen:  datetime
    name:       str | None    = None
    version:    str | None    = None
    provider:   str | None    = None


@dataclass
class Alert:
    alert_id:   str
    fired_at:   datetime
    type:       AlertType
    severity:   Severity
    title:      str
    detail:     dict[str, Any]
    agent_id:   str | None = None
    session_id: str | None = None
    span_id:    str | None = None
    acknowledged: bool     = False
    suppressed:   bool     = False


@dataclass
class DriftBaseline:
    agent_id:               str
    sessions_sampled:       int
    computed_at:            datetime
    avg_input_tokens:       float | None = None
    stddev_input_tokens:    float | None = None
    avg_output_tokens:      float | None = None
    stddev_output_tokens:   float | None = None
    avg_session_duration_s: float | None = None
    stddev_session_duration: float | None = None
    avg_tool_call_count:    float | None = None
    stddev_tool_call_count: float | None = None
    common_tool_sequences:  list | None  = None
    output_schema_inferred: dict | None  = None


@dataclass
class DriftViolation:
    dimension: str
    z_score:   float | None   = None
    expected:  str | None     = None
    observed:  str | None     = None
    detail:    str | None     = None


@dataclass
class DriftResult:
    violations: list[DriftViolation]
    drifted:    bool


@dataclass
class SchemaValidationResult:
    validation_id: str
    span_id:       str
    validated_at:  datetime
    passed:        bool
    errors:        list[str]  = field(default_factory=list)
    agent_id:      str | None = None


@dataclass
class TraceRecord:
    trace_id:   str
    agent_id:   str | None
    name:       str
    start_time: datetime
    duration_ms: float | None = None
    cost_usd:   float | None  = None
    status_code: str          = "ok"
    span_count:  int          = 0
    # Per-trace token totals so the UI can render per-row cost as TOKENS for
    # subscription/local users (#249) — "% of cycle" is a window-level aggregate
    # and is nonsensical at per-trace granularity. Summed server-side (single
    # compute path) rather than re-aggregated in JS.
    input_tokens:  int        = 0
    output_tokens: int        = 0


@dataclass
class CostRow:
    group:        str
    agent_id:     str | None  = None
    model:        str | None  = None
    input_tokens: int         = 0
    output_tokens: int        = 0
    cache_tokens: int         = 0   # cache-READ tokens
    cache_write_tokens: int   = 0   # cache-CREATE tokens (the hidden cost driver, #17)
    cost_usd:     float       = 0.0


# -- Enforcement-plane audit log + savings meter (#221) --

@dataclass
class PolicyDecisionRecord:
    """One persisted proxy observation — a row in the append-only audit log.

    Records both the POLICY path and observe-only traffic. ``gate_decision`` +
    ``passthrough_tos`` distinguish "we chose not to act" (policy path,
    would_action='noop') from "we were not permitted to act" (subscription TOS).
    ``label`` ('unvalidated') rides through from the envelope.
    """
    decision_id:     str
    ts:              datetime
    provider:        str | None
    pricing_mode:    str
    gate_decision:   str            # observe_only | policy
    path:            str
    would_action:    str            # overall action; 'noop' on observe-only
    policy_name:     str | None = None
    policy_kind:     str | None = None
    passthrough_tos: bool = False   # observe-only due to provider TOS (subscription)
    label:           str = "unvalidated"
    suggest_only:    bool = True
    envelope:        dict | None = None  # full round-trippable envelope (None observe-only)


@dataclass
class SavingsLedgerEntry:
    """What ONE policy decision WOULD have recovered — never a realized figure.

    Suggest mode enforces nothing, so ``realized`` is always False and the
    amounts are ESTIMATED-RECOVERABLE / would-have-saved (Critical Rule 14).
    Dollar figures are api-only; subscription/local accrue token-quota framing.
    """
    ledger_id:                    str
    decision_id:                  str
    ts:                           datetime
    provider:                     str | None
    pricing_mode:                 str
    would_action:                 str
    policy_name:                  str | None = None
    estimated_recoverable_usd:    float = 0.0
    estimated_recoverable_tokens: int = 0
    estimate_basis:               str = ""
    billing_period:               str = ""       # YYYY-MM
    label:                        str = "unvalidated"
    realized:                     bool = False    # suggest mode: NEVER realized


@dataclass
class PolicyDecisionFilters:
    since:    datetime | None = None
    until:    datetime | None = None
    provider: str | None = None
    limit:    int = 100


# -- Filter dataclasses used by StorageBackend --

@dataclass
class TraceFilters:
    agent_id:   str | None   = None
    since:      datetime | None = None
    until:      datetime | None = None
    span_name:  str | None   = None
    status:     str | None   = None
    limit:      int          = 50
    offset:     int          = 0


@dataclass
class CostFilters:
    agent_id:  str | None   = None
    since:     datetime | None = None
    until:     datetime | None = None
    group_by:  str          = "day"   # agent | model | day | tool


@dataclass
class AlertFilters:
    agent_id:  str | None   = None
    since:     datetime | None = None
    severity:  Severity | None = None
    type:      AlertType | None = None
    unread:    bool          = False
    limit:     int           = 100
