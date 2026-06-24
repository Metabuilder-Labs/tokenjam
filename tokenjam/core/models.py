from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Sessions with no spans for this long are considered stale (zombie).
SESSION_STALE_THRESHOLD = timedelta(minutes=5)


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
    cache_tokens:    int          = 0
    tool_call_count: int          = 0
    error_count:     int          = 0
    # Canonical plan-tier identifier for the user's billing relationship with
    # this session's provider. Set at session creation by reading
    # ProviderBudget.plan for the matching billing_account. Backfilled sessions
    # default to "unknown" — `tj optimize` suppresses dollar figures for those.
    # Valid values: see VALID_PLAN_TIERS in tokenjam.otel.semconv.
    plan_tier:       str          = "unknown"

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    @property
    def effective_status(self) -> str:
        """Return 'stale' for zombie sessions whose process was killed."""
        if self.status != "active":
            return self.status
        from tokenjam.utils.time_parse import utcnow
        last_activity = self.ended_at or self.started_at
        if last_activity and (utcnow() - last_activity) > SESSION_STALE_THRESHOLD:
            return "stale"
        return "active"

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
