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
    cost_usd:       float | None   = None
    request_type:   str | None     = None
    conversation_id: str | None    = None


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
        from tj.utils.time_parse import utcnow
        last_activity = self.ended_at or self.started_at
        if last_activity and (utcnow() - last_activity) > SESSION_STALE_THRESHOLD:
            return "stale"
        return "active"


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


@dataclass
class CostRow:
    group:        str
    agent_id:     str | None  = None
    model:        str | None  = None
    input_tokens: int         = 0
    output_tokens: int        = 0
    cost_usd:     float       = 0.0


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
