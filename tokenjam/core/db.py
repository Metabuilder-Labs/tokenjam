"""
Database layer: StorageBackend protocol, DuckDB implementation, InMemoryBackend for tests,
and migration runner. DuckDB only — never import sqlite3.
"""
from __future__ import annotations

import json
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import duckdb

from tokenjam.core.config import StorageConfig
from tokenjam.core.models import (
    AgentRecord,
    Alert,
    AlertFilters,
    CostFilters,
    CostRow,
    DriftBaseline,
    NormalizedSpan,
    PolicyDecisionFilters,
    PolicyDecisionRecord,
    SavingsLedgerEntry,
    SchemaValidationResult,
    SessionRecord,
    SpanKind,
    SpanStatus,
    TraceFilters,
    TraceRecord,
)
from tokenjam.utils.time_parse import utcnow


# ---------------------------------------------------------------------------
# StorageBackend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StorageBackend(Protocol):
    def insert_span(self, span: NormalizedSpan) -> None: ...
    def insert_alert(self, alert: Alert) -> None: ...
    def insert_validation(self, result: SchemaValidationResult) -> None: ...
    def insert_policy_decision(self, decision: PolicyDecisionRecord) -> None: ...
    def insert_savings_entry(self, entry: SavingsLedgerEntry) -> None: ...
    def get_policy_decisions(
        self, filters: PolicyDecisionFilters,
    ) -> list[PolicyDecisionRecord]: ...
    def get_savings_entries(
        self, filters: PolicyDecisionFilters,
    ) -> list[SavingsLedgerEntry]: ...
    def upsert_session(self, session: SessionRecord) -> None: ...
    def upsert_agent(self, agent: AgentRecord) -> None: ...
    def upsert_baseline(self, baseline: DriftBaseline) -> None: ...
    def get_session(self, session_id: str) -> SessionRecord | None: ...
    def get_session_by_conversation(self, conversation_id: str) -> SessionRecord | None: ...
    def close_sessions_by_instance(self, instance_id: str) -> int: ...
    def close_session_by_id(self, session_id: str) -> int: ...
    def get_traces(self, filters: TraceFilters) -> list[TraceRecord]: ...
    def count_traces(self, filters: TraceFilters) -> int: ...
    def get_trace_spans(self, trace_id: str) -> list[NormalizedSpan]: ...
    def get_cost_summary(self, filters: CostFilters) -> list[CostRow]: ...
    def get_alerts(self, filters: AlertFilters) -> list[Alert]: ...
    def get_baseline(self, agent_id: str) -> DriftBaseline | None: ...
    def get_completed_sessions(self, agent_id: str, limit: int) -> list[SessionRecord]: ...
    def get_completed_session_count(self, agent_id: str) -> int: ...
    def get_tool_calls(
        self, agent_id: str | None, since: datetime | None, tool_name: str | None,
    ) -> list[dict]: ...
    def get_daily_cost(self, agent_id: str, date: date) -> float: ...
    def get_session_cost(self, session_id: str) -> float: ...
    def get_recent_spans(self, session_id: str, limit: int) -> list[NormalizedSpan]: ...
    # Issue #309: methods that callers (CostEngine, cmd_status, cost compare)
    # used to satisfy by reaching into `db.conn` directly. Having them on the
    # protocol keeps those paths behind the abstraction and lets InMemoryBackend
    # exercise them in unit tests.
    def update_span_cost(self, span_id: str, cost_usd: float) -> None: ...
    def increment_session_cost(self, session_id: str, delta_usd: float) -> None: ...
    def get_distinct_agent_ids(self) -> list[str]: ...
    def get_active_session(self, agent_id: str) -> SessionRecord | None: ...
    def get_session_active_seconds(self, session_id: str) -> float | None: ...
    def count_unknown_plan_tier_sessions(self) -> int: ...
    def get_window_cost_totals(
        self, since: datetime, until: datetime, agent_id: str | None = None,
    ) -> tuple[int, int, int, int, float]: ...
    def get_cost_delta_by_group(
        self, group_col: str, current_since: datetime, current_until: datetime,
        prev_since: datetime, prev_until: datetime, top_n: int,
    ) -> list[dict]: ...
    def delete_spans_before(self, cutoff: datetime) -> int: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Schema & migrations
# ---------------------------------------------------------------------------

# Canonical spans table DDL. Single-sourced here so the repair path
# (`repair_spans_stats`) can rebuild a table that is schema-identical to a
# freshly-migrated one — PRIMARY KEY, NOT NULL constraints and all. Referenced
# by INITIAL_SCHEMA_SQL below; do not inline a second copy.
SPANS_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS spans (
    span_id             TEXT PRIMARY KEY,
    trace_id            TEXT NOT NULL,
    parent_span_id      TEXT,
    session_id          TEXT,
    agent_id            TEXT,
    name                TEXT NOT NULL,
    kind                TEXT NOT NULL,
    status_code         TEXT NOT NULL,
    status_message      TEXT,
    start_time          TIMESTAMPTZ NOT NULL,
    end_time            TIMESTAMPTZ,
    duration_ms         DOUBLE,
    attributes          JSON NOT NULL DEFAULT '{}',
    provider            TEXT,
    model               TEXT,
    tool_name           TEXT,
    input_tokens        BIGINT,
    output_tokens       BIGINT,
    cache_tokens        BIGINT,
    cost_usd            DOUBLE,
    request_type        TEXT,
    conversation_id     TEXT,
    events              JSON DEFAULT '[]',
    -- cache_write_tokens (cache-CREATION tokens) added by migration 5.
    -- Kept separate from cache_tokens (cache-read) because they bill at
    -- different rates. See models.py::NormalizedSpan for the read/write split.
    cache_write_tokens  BIGINT
);
"""

# Secondary indexes on spans. Single-sourced so migration 3 and the repair path
# create the same set; keep in sync with the DROPs in migration 2.
SPANS_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_spans_trace_id    ON spans(trace_id);\n"
    "CREATE INDEX IF NOT EXISTS idx_spans_agent_id    ON spans(agent_id);\n"
    "CREATE INDEX IF NOT EXISTS idx_spans_start_time  ON spans(start_time);\n"
    "CREATE INDEX IF NOT EXISTS idx_spans_tool_name   ON spans(tool_name);\n"
    "CREATE INDEX IF NOT EXISTS idx_spans_conv_id     ON spans(conversation_id)"
)

INITIAL_SCHEMA_SQL = (
    """\
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id    TEXT PRIMARY KEY,
    name        TEXT,
    version     TEXT,
    provider    TEXT,
    first_seen  TIMESTAMPTZ NOT NULL,
    last_seen   TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    agent_id            TEXT NOT NULL,
    conversation_id     TEXT,
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ,
    status              TEXT NOT NULL DEFAULT 'active',
    total_cost_usd      DOUBLE,
    input_tokens        BIGINT DEFAULT 0,
    output_tokens       BIGINT DEFAULT 0,
    cache_tokens        BIGINT DEFAULT 0,
    -- cache_write_tokens (cache-CREATION tokens) added by migration 12. Kept
    -- separate from cache_tokens (cache-read) because they bill at a higher rate.
    cache_write_tokens  BIGINT DEFAULT 0,
    tool_call_count     INTEGER DEFAULT 0,
    error_count         INTEGER DEFAULT 0
);

"""
    + SPANS_TABLE_SQL
    + """\

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        TEXT PRIMARY KEY,
    agent_id        TEXT,
    session_id      TEXT,
    span_id         TEXT,
    fired_at        TIMESTAMPTZ NOT NULL,
    type            TEXT NOT NULL,
    severity        TEXT NOT NULL,
    title           TEXT NOT NULL,
    detail          JSON NOT NULL,
    acknowledged    BOOLEAN DEFAULT false,
    suppressed      BOOLEAN DEFAULT false
);

CREATE TABLE IF NOT EXISTS drift_baselines (
    agent_id                TEXT PRIMARY KEY,
    sessions_sampled        INTEGER NOT NULL,
    computed_at             TIMESTAMPTZ NOT NULL,
    avg_input_tokens        DOUBLE,
    stddev_input_tokens     DOUBLE,
    avg_output_tokens       DOUBLE,
    stddev_output_tokens    DOUBLE,
    avg_session_duration_s  DOUBLE,
    stddev_session_duration DOUBLE,
    avg_tool_call_count     DOUBLE,
    stddev_tool_call_count  DOUBLE,
    common_tool_sequences   JSON,
    output_schema_inferred  JSON
);

CREATE TABLE IF NOT EXISTS schema_validations (
    validation_id   TEXT PRIMARY KEY,
    span_id         TEXT NOT NULL,
    agent_id        TEXT,
    validated_at    TIMESTAMPTZ NOT NULL,
    passed          BOOLEAN NOT NULL,
    errors          JSON DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_sessions_agent_id  ON sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_conv_id   ON sessions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_alerts_agent_id    ON alerts(agent_id);
CREATE INDEX IF NOT EXISTS idx_alerts_fired_at    ON alerts(fired_at);
"""
)

MIGRATIONS: list[tuple[int, str]] = [
    (1, INITIAL_SCHEMA_SQL),
    (2, (
        "DROP INDEX IF EXISTS idx_spans_trace_id;\n"
        "DROP INDEX IF EXISTS idx_spans_agent_id;\n"
        "DROP INDEX IF EXISTS idx_spans_start_time;\n"
        "DROP INDEX IF EXISTS idx_spans_tool_name;\n"
        "DROP INDEX IF EXISTS idx_spans_conv_id"
    )),
    (3, SPANS_INDEX_SQL),
    # Migration 4: billing_account on spans, plan_tier on sessions.
    # `billing_account` is provider-only (anthropic | openai | google |
    # bedrock | local.ollama). Plan tier lives on sessions, not spans.
    # `plan_tier` defaults to 'unknown' for backfilled rows; new sessions
    # get it set at creation time from ProviderBudget.plan.
    (4, (
        # DuckDB ALTER TABLE doesn't support NOT NULL on added columns, so
        # plan_tier is nullable in the schema. Application code defaults
        # NULL to 'unknown' on read (see _row_to_session).
        "ALTER TABLE spans    ADD COLUMN IF NOT EXISTS billing_account TEXT;\n"
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS plan_tier       TEXT DEFAULT 'unknown'"
    )),
    # Migration 5: cache_write_tokens on spans. Issue #94.
    # NormalizedSpan and the cost engine started threading cache-write
    # tokens through in PR #92 (live OTLP path) but the count was never
    # persisted — only the resulting cost_usd landed. This column makes
    # per-token-class reporting possible. NULL on backfilled rows; the
    # _row_to_span helper coerces NULL -> None and ingest writes 0 for
    # spans that don't carry the count.
    (5, "ALTER TABLE spans ADD COLUMN IF NOT EXISTS cache_write_tokens BIGINT"),
    # Migration 6: enforcement-plane audit log + savings meter (#221).
    # `policy_decisions` is the append-only audit log — one row per recorded
    # proxy observation (both the POLICY path and observe-only). `gate_decision`
    # + `passthrough_tos` let the log distinguish "we CHOSE not to act" (policy
    # path, action=noop) from "we were NOT PERMITTED to act" (subscription TOS).
    # `savings_ledger` records what each policy decision WOULD have recovered —
    # SUGGEST MODE ENFORCES NOTHING, so `realized` is always FALSE and the
    # figures are estimated-recoverable / would-have-saved, NEVER realized
    # savings (Critical Rule 14). The `label` ('unvalidated') rides through from
    # the envelope on both tables.
    (6, (
        "CREATE TABLE IF NOT EXISTS policy_decisions (\n"
        "    decision_id     TEXT PRIMARY KEY,\n"
        "    ts              TIMESTAMPTZ NOT NULL,\n"
        "    provider        TEXT,\n"
        "    pricing_mode    TEXT,\n"
        "    gate_decision   TEXT,\n"
        "    path            TEXT,\n"
        "    policy_name     TEXT,\n"
        "    policy_kind     TEXT,\n"
        "    would_action    TEXT,\n"
        "    passthrough_tos BOOLEAN DEFAULT FALSE,\n"
        "    label           TEXT,\n"
        "    suggest_only    BOOLEAN DEFAULT TRUE,\n"
        "    envelope        JSON\n"
        ");\n"
        "CREATE TABLE IF NOT EXISTS savings_ledger (\n"
        "    ledger_id                    TEXT PRIMARY KEY,\n"
        "    decision_id                  TEXT NOT NULL,\n"
        "    ts                           TIMESTAMPTZ NOT NULL,\n"
        "    provider                     TEXT,\n"
        "    pricing_mode                 TEXT,\n"
        "    policy_name                  TEXT,\n"
        "    would_action                 TEXT,\n"
        "    estimated_recoverable_usd    DOUBLE DEFAULT 0.0,\n"
        "    estimated_recoverable_tokens BIGINT DEFAULT 0,\n"
        "    estimate_basis               TEXT,\n"
        "    billing_period               TEXT,\n"
        "    label                        TEXT,\n"
        "    realized                     BOOLEAN DEFAULT FALSE\n"
        ");\n"
        "CREATE INDEX IF NOT EXISTS idx_policy_decisions_ts ON policy_decisions(ts);\n"
        "CREATE INDEX IF NOT EXISTS idx_savings_ledger_ts   ON savings_ledger(ts)"
    )),
    # Migration 7: full-request capture on spans (#209). `request_params` holds
    # sampling parameters (temperature, top_p, max_tokens, stop_sequences, …);
    # `request_tools` holds the tools / tool_choice payload. Both are JSON,
    # NULL on rows captured before this migration (and whenever the relevant
    # [capture] toggle is off). _row_to_span coerces NULL -> None.
    (7, (
        "ALTER TABLE spans ADD COLUMN IF NOT EXISTS request_params JSON;\n"
        "ALTER TABLE spans ADD COLUMN IF NOT EXISTS request_tools  JSON"
    )),
    # Migration 8: service_namespace on sessions — the OTel service.namespace
    # the session's service rolls up under (the dashboard's "project" grouping
    # key). Nullable; sessions whose telemetry carried no namespace stay NULL.
    (8, "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS service_namespace TEXT"),
    # Migration 9: service_instance_id on sessions — the per-terminal label
    # (OTel service.instance.id) used as the session's display name.
    (9, "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS service_instance_id TEXT"),
    # Migration 10: repair ended_at on already-closed sessions. A prior bug in
    # close_session(s) advanced ended_at to the close time, so a session closed
    # days after its last span showed a "Last seen" of the close moment instead
    # of its real last activity. Recompute ended_at from the session's actual
    # spans (max of end_time / start_time), but only LOWER it — never touch
    # sessions whose ended_at already matches or precedes their last span.
    # Idempotent: re-running finds nothing left to correct.
    (10, (
        "UPDATE sessions AS s "
        "SET ended_at = sub.max_ts "
        "FROM (SELECT session_id, MAX(COALESCE(end_time, start_time)) AS max_ts "
        "      FROM spans GROUP BY session_id) AS sub "
        "WHERE s.session_id = sub.session_id "
        "  AND s.status = 'closed' "
        "  AND sub.max_ts IS NOT NULL "
        "  AND (s.ended_at IS NULL OR s.ended_at > sub.max_ts)"
    )),
    # Migration 11: session_labels — a user-supplied display name for a session,
    # set from the dashboard by right-clicking a session card (POST
    # /api/v1/sessions/{id}/label). One row per session (session_id PRIMARY KEY);
    # the /status route overlays these onto the tile/archive label, taking
    # precedence over the OTel service.instance.id but NOT over a config
    # [session_labels] entry (see status._session_label). Persisting to the DB
    # (rather than editing the config TOML) keeps renames a runtime dashboard
    # action that survives restarts without a config write.
    (11, (
        "CREATE TABLE IF NOT EXISTS session_labels (\n"
        "    session_id  TEXT PRIMARY KEY,\n"
        "    label       TEXT NOT NULL,\n"
        "    updated_at  TIMESTAMPTZ NOT NULL\n"
        ")"
    )),
    # Migration 12: cache_write_tokens on sessions. spans.cache_write_tokens
    # already exists (migration 5); the per-session aggregate column was still
    # missing, so cache-*write*/creation tokens never rolled up to the session
    # row. Now tracked so the dashboard can show total cache activity
    # (reads + writes) per session and the cost engine can price writes at the
    # higher cache-write rate. Nullable; existing rows default 0. The spans line
    # is a defensive no-op (the column is already present from migration 5).
    (12, (
        "ALTER TABLE spans    ADD COLUMN IF NOT EXISTS cache_write_tokens BIGINT DEFAULT 0;\n"
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS cache_write_tokens BIGINT DEFAULT 0"
    )),
    # Migration 13: run_id + parent_session_id on sessions — cross-session run
    # grouping declared by a fan-out harness (tokenjam.run_id /
    # tokenjam.parent_session_id resource attributes). Both nullable; existing
    # sessions stay NULL on upgrade.
    (13, (
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS run_id            TEXT;\n"
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS parent_session_id TEXT"
    )),
    # Migration 14: sub_agent_id on spans — the Claude Code subagent (Task-tool
    # / sidechain) that issued the span. NULL for main-thread spans and all
    # non-Claude-Code telemetry. A single research session can spawn 12-20
    # subagents whose spans all fold under the parent session_id; this column
    # keeps them attributable per subagent. Populated by the backfill parser
    # from each record's top-level agentId when isSidechain is true.
    (14, "ALTER TABLE spans ADD COLUMN IF NOT EXISTS sub_agent_id TEXT"),
    # Migration 15: session_story — a persisted snapshot of a session's
    # reconstructed Story (the recursive method/narration + subagent subtree,
    # core/transcript.py). /story and /workmap recompute that Story from the
    # on-disk Claude Code JSONL transcript on every request and never store it;
    # Claude Code PRUNES those transcripts, so a killed ephemeral agent's method
    # dies with the file. This table captures it at session close (M1,
    # core/method_capture.py) so it outlives the prune and can serve as a
    # read-through fallback. One row per session (session_id PRIMARY KEY);
    # `source` records provenance ('live-transcript' | 'backfill') and
    # `schema_version` the snapshot payload shape. story_json is the full
    # snapshot ({"story": ..., "asks": ...}); the depth_capped/budget_capped/
    # cycle markers ride through it unchanged.
    (15, (
        "CREATE TABLE IF NOT EXISTS session_story (\n"
        "    session_id     TEXT PRIMARY KEY,\n"
        "    story_json     JSON NOT NULL,\n"
        "    captured_at    TIMESTAMPTZ NOT NULL,\n"
        "    source         TEXT NOT NULL,\n"
        "    schema_version INTEGER NOT NULL DEFAULT 1\n"
        ")"
    )),
]


def run_migrations(conn: duckdb.DuckDBPyConnection) -> None:
    """Apply unapplied migrations. Idempotent."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ)"
    )
    applied = {
        row[0]
        for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }
    for version, sql in MIGRATIONS:
        if version not in applied:
            for statement in sql.split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations VALUES ($1, $2)",
                [version, utcnow()],
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_span(row: tuple, columns: list[str]) -> NormalizedSpan:
    d = dict(zip(columns, row))
    attrs = d.get("attributes") or {}
    if isinstance(attrs, str):
        attrs = json.loads(attrs)
    events = d.get("events") or []
    if isinstance(events, str):
        events = json.loads(events)
    request_params = d.get("request_params")
    if isinstance(request_params, str):
        request_params = json.loads(request_params)
    request_tools = d.get("request_tools")
    if isinstance(request_tools, str):
        request_tools = json.loads(request_tools)
    return NormalizedSpan(
        span_id=d["span_id"],
        trace_id=d["trace_id"],
        name=d["name"],
        kind=SpanKind(d["kind"]),
        status_code=SpanStatus(d["status_code"]),
        start_time=d["start_time"],
        parent_span_id=d.get("parent_span_id"),
        session_id=d.get("session_id"),
        agent_id=d.get("agent_id"),
        sub_agent_id=d.get("sub_agent_id"),
        end_time=d.get("end_time"),
        duration_ms=d.get("duration_ms"),
        status_message=d.get("status_message"),
        attributes=attrs,
        events=events,
        provider=d.get("provider"),
        model=d.get("model"),
        tool_name=d.get("tool_name"),
        input_tokens=_int_or_none(d.get("input_tokens")),
        output_tokens=_int_or_none(d.get("output_tokens")),
        cache_tokens=_int_or_none(d.get("cache_tokens")),
        cache_write_tokens=_int_or_none(d.get("cache_write_tokens")),
        cost_usd=d.get("cost_usd"),
        request_type=d.get("request_type"),
        conversation_id=d.get("conversation_id"),
        billing_account=d.get("billing_account"),
        request_params=request_params,
        request_tools=request_tools,
    )


def _row_to_session(row: tuple, columns: list[str]) -> SessionRecord:
    d = dict(zip(columns, row))
    return SessionRecord(
        session_id=d["session_id"],
        agent_id=d["agent_id"],
        started_at=d["started_at"],
        conversation_id=d.get("conversation_id"),
        ended_at=d.get("ended_at"),
        status=d.get("status", "active"),
        total_cost_usd=d.get("total_cost_usd"),
        input_tokens=d.get("input_tokens") or 0,
        output_tokens=d.get("output_tokens") or 0,
        cache_tokens=d.get("cache_tokens") or 0,
        cache_write_tokens=d.get("cache_write_tokens") or 0,
        tool_call_count=d.get("tool_call_count") or 0,
        error_count=d.get("error_count") or 0,
        plan_tier=d.get("plan_tier") or "unknown",
        service_namespace=d.get("service_namespace"),
        service_instance_id=d.get("service_instance_id"),
        run_id=d.get("run_id"),
        parent_session_id=d.get("parent_session_id"),
    )


def session_active_seconds(conn, session_id: str) -> float | None:
    """
    Active (compute) time for a session: the sum of its span durations, in
    seconds. Distinct from `SessionRecord.duration_seconds`, which is wall-clock
    (`ended_at - started_at`) and can span days for resumed Claude Code sessions.

    Returns None when the session has no spans with a recorded duration (so
    callers can omit the field rather than show a misleading 0).
    """
    if session_id is None:
        return None
    row = conn.execute(
        "SELECT SUM(duration_ms) FROM spans WHERE session_id = $1",
        [session_id],
    ).fetchone()
    if not row or row[0] is None:
        return None
    return float(row[0]) / 1000.0


# Token/cost rollup for a single session, joining cost spans onto the session via
# the trace(s) the session's own spans carry. The keys mirror the denormalized
# `sessions` aggregate columns so callers can splice the result straight in.
_SESSION_ROLLUP_KEYS = (
    "input_tokens", "output_tokens", "cache_tokens", "cache_write_tokens",
    "total_cost_usd", "tool_call_count",
)


def session_token_cost_rollup(conn, session_id: str) -> dict | None:
    """True per-session token/cost rollup, joining trace-keyed cost spans (#18).

    The denormalized `sessions` aggregate columns are accumulated per span keyed
    by `span.session_id` (see `IngestPipeline._build_or_update_session`). That is
    correct for telemetry that stamps the session id on its cost spans (the
    `/v1/logs` Claude Code/Codex path, and the on-disk backfill). But a fan-out
    harness posting raw OTLP to `/api/v1/spans` can emit the zero-cost
    `invoke_agent` marker span WITH `session.id` while its cost-bearing
    `gen_ai.llm.call` spans carry only `agent_id` + `traceId` (no `session.id`).
    Those cost spans never accumulate onto the marker's session row, so the
    per-session rollup reads 0 even though the spend is real and surfaces fine on
    Cost/Traces (which key by agent_id / trace, never session_id).

    The defensible association is the **trace**: the marker span (which carries
    the session_id) and the cost spans share a `trace_id`, exactly the join
    `/traces` already uses to attribute the same spans. So this rolls up every
    span whose `session_id` matches OR whose `trace_id` appears on a span that
    carries this session_id. Each span is counted once (a span matching both
    predicates isn't double-counted — the WHERE is a disjunction over distinct
    rows, not a self-join). When the cost spans DO carry the session_id (the
    common case), the trace clause is redundant and the result equals the plain
    `session_id` sum, so this is a strict superset that never under- or
    over-counts the already-correct paths.

    Returns a dict keyed by `_SESSION_ROLLUP_KEYS`, or None when the session has
    no spans at all (caller keeps the stored row's values).
    """
    if conn is None or session_id is None:
        return None
    row = conn.execute(
        """
        SELECT
            COALESCE(SUM(input_tokens), 0),
            COALESCE(SUM(output_tokens), 0),
            COALESCE(SUM(cache_tokens), 0),
            COALESCE(SUM(cache_write_tokens), 0),
            COALESCE(SUM(cost_usd), 0.0),
            COUNT(*) FILTER (WHERE tool_name IS NOT NULL),
            COUNT(*)
        FROM spans
        WHERE session_id = $1
           OR trace_id IN (
                SELECT DISTINCT trace_id FROM spans
                WHERE session_id = $1 AND trace_id IS NOT NULL
           )
        """,
        [session_id],
    ).fetchone()
    if not row or row[6] == 0:
        return None
    return {
        "input_tokens": int(row[0] or 0),
        "output_tokens": int(row[1] or 0),
        "cache_tokens": int(row[2] or 0),
        "cache_write_tokens": int(row[3] or 0),
        "total_cost_usd": float(row[4] or 0.0),
        "tool_call_count": int(row[5] or 0),
    }


def _resolve_conn(db_or_conn):
    """Return the underlying cursor for a backend, or the conn passed as-is.

    ``set_session_label`` / ``delete_session_label`` accept either a backend
    (whose per-thread ``.conn`` cursor is used) or a raw DuckDB connection (which
    has no ``.conn`` attr, so it passes through unchanged).
    """
    return getattr(db_or_conn, "conn", db_or_conn)


def set_session_label(db_or_conn, session_id: str, label: str) -> None:
    """Upsert a user-supplied display name for a session (migration 11).

    DuckDB has no portable UPSERT here, so this DELETEs any prior row then
    INSERTs the fresh one — idempotent (a re-label overwrites). ``updated_at`` is
    stamped via ``utcnow()`` (Critical Rule 9). Parameterised SQL only (Critical
    Rule 7: no f-string SQL).
    """
    conn = _resolve_conn(db_or_conn)
    now = utcnow()
    conn.execute("DELETE FROM session_labels WHERE session_id = $1", [session_id])
    conn.execute(
        "INSERT INTO session_labels (session_id, label, updated_at) "
        "VALUES ($1, $2, $3)",
        [session_id, label, now],
    )


def delete_session_label(db_or_conn, session_id: str) -> None:
    """Remove a session's user-supplied display name (migration 11). Idempotent."""
    conn = _resolve_conn(db_or_conn)
    conn.execute("DELETE FROM session_labels WHERE session_id = $1", [session_id])


def get_session_labels(conn) -> dict[str, str]:
    """All session -> user label overlays as a dict (migration 11).

    One SELECT so the /status route fetches every override in a single query.
    Guards a ``None`` conn (a non-DB backend) -> ``{}``.
    """
    if conn is None:
        return {}
    rows = conn.execute("SELECT session_id, label FROM session_labels").fetchall()
    return {r[0]: r[1] for r in rows if r[0] is not None and r[1] is not None}


def sdk_service_series(
    conn, agent_ids: list[str], window_start, now, *, slots: int = 24
) -> dict[str, dict]:
    """Per-minute cost / calls / error% series + last_seen for the given agents.

    Buckets `spans` by minute over the last `slots` minutes ending at `now`,
    zero-filled to a fixed-length grid so every agent yields exactly `slots`
    points (a flatline for services that emitted nothing recently). Also returns
    window totals (for req/min + err-rate) and `last_seen` across ALL history —
    a long-dormant service last emitted days ago, outside the sparkline window.

    Powers the /status SDK-services zone (Prometheus-style sparklines). Returns
    {} when `conn` is None or no agents are given. Each agent maps to:
        {cost_per_min, calls_per_min, err_pct_per_min: [slots],
         window_cost, window_calls, window_errors, last_seen}
    """
    if conn is None or not agent_ids:
        return {}

    # Fixed minute grid: slot i covers [grid[i], grid[i] + 60s); grid[-1] is the
    # minute containing `now`. Epoch-second keys match the SQL bucket below.
    base = int(now.timestamp() // 60) * 60
    grid = [base - (slots - 1 - i) * 60 for i in range(slots)]
    index = {ts: i for i, ts in enumerate(grid)}

    result: dict[str, dict] = {
        aid: {
            "cost_per_min": [0.0] * slots,
            "calls_per_min": [0] * slots,
            "err_pct_per_min": [0.0] * slots,
            "window_cost": 0.0,
            "window_calls": 0,
            "window_errors": 0,
            "last_seen": None,
        }
        for aid in agent_ids
    }

    # IN (…) with per-id placeholders — a controlled small list; all values bound
    # (never interpolated), matching the codebase's dynamic-placeholder style.
    ph = ", ".join(f"${i + 2}" for i in range(len(agent_ids)))
    rows = conn.execute(
        f"""
        SELECT agent_id,
               CAST(epoch(date_trunc('minute', start_time AT TIME ZONE 'UTC')) AS BIGINT) AS b,
               COALESCE(SUM(cost_usd), 0.0)                  AS cost,
               COUNT(*) FILTER (WHERE status_code = 'error') AS errors,
               COUNT(*)                                      AS calls
        FROM spans
        WHERE start_time >= $1 AND agent_id IN ({ph})
        GROUP BY agent_id, b
        """,
        [window_start, *agent_ids],
    ).fetchall()
    for aid, b, cost, errors, calls in rows:
        r = result[aid]
        r["window_cost"] += float(cost or 0.0)
        r["window_calls"] += int(calls or 0)
        r["window_errors"] += int(errors or 0)
        slot = index.get(int(b))
        if slot is None:
            continue
        r["cost_per_min"][slot] = float(cost or 0.0)
        r["calls_per_min"][slot] = int(calls or 0)
        r["err_pct_per_min"][slot] = (
            float(errors) / calls * 100.0 if calls else 0.0
        )

    ph1 = ", ".join(f"${i + 1}" for i in range(len(agent_ids)))
    seen_rows = conn.execute(
        f"SELECT agent_id, MAX(COALESCE(end_time, start_time)) "
        f"FROM spans WHERE agent_id IN ({ph1}) GROUP BY agent_id",
        [*agent_ids],
    ).fetchall()
    for aid, last_seen in seen_rows:
        if aid in result:
            result[aid]["last_seen"] = last_seen

    return result


def _int_or_none(val: object) -> int | None:
    if val is None:
        return None
    return int(val)


# ---------------------------------------------------------------------------
# Column-statistics corruption check & repair (DuckDB v1.5.x bug)
# ---------------------------------------------------------------------------
# Under some write patterns, DuckDB's per-row-group min/max statistics for the
# spans table get out of sync with the actual data. The equality fast-path then
# skips every row group, so `WHERE trace_id = X` returns 0 rows even when the
# data is clearly there. `WHERE trace_id LIKE X || '%'` works because it forces
# a full scan that bypasses the bad stats.
#
# Detection: pick a known trace_id (via wildcard-LIKE), then verify that the
# `=` predicate finds it too. If they disagree the table's stats are corrupt.
#
# Repair: copy the table to a fresh one and rename. CHECKPOINT alone does not
# rebuild stats; only a full table copy does.
#
# See issue #56.


def check_spans_stats_corruption(conn: duckdb.DuckDBPyConnection) -> bool:
    """Return True if the spans table's column-equality fast-path is broken.

    Samples up to 3 distinct trace_ids and compares `=` vs `LIKE col || '%'`
    counts. Any mismatch indicates corrupt column statistics. Returns False
    on an empty spans table (nothing to check, so nothing to fix).
    """
    try:
        sample = conn.execute(
            "SELECT DISTINCT trace_id FROM spans LIMIT 3"
        ).fetchall()
    except duckdb.Error:
        return False
    if not sample:
        return False
    for (tid,) in sample:
        if tid is None:
            continue
        try:
            eq_row = conn.execute(
                "SELECT COUNT(*) FROM spans WHERE trace_id = $1", [tid]
            ).fetchone()
            like_row = conn.execute(
                "SELECT COUNT(*) FROM spans WHERE trace_id LIKE $1 || '%'", [tid]
            ).fetchone()
        except duckdb.Error:
            return False
        # COUNT(*) always returns one row, but mypy doesn't know that.
        eq = eq_row[0] if eq_row else 0
        like = like_row[0] if like_row else 0
        if eq == 0 and like > 0:
            return True
    return False


def repair_spans_stats(conn: duckdb.DuckDBPyConnection) -> None:
    """Rebuild the spans table to refresh column statistics.

    Idempotent — safe to call when the table is healthy. Data is preserved.
    Caller is responsible for ensuring no other process holds a write lock on
    the database file (DuckDB enforces exclusive write access).

    The rebuild recreates the table from the canonical DDL rather than a bare
    `CREATE TABLE … AS SELECT` (#38). A CTAS copies DATA ONLY — it would drop the
    `span_id` PRIMARY KEY, the NOT NULL constraints, and every `idx_spans_*`
    index, permanently (migrations are already marked applied, so nothing
    recreates them). Instead we move the live rows aside, recreate `spans` with
    its full schema, copy the rows back, and re-issue the indexes — leaving the
    repaired table schema-identical to a freshly-migrated one.
    """
    # Stash the live rows + their column layout in a constraint-free holder, then
    # drop spans (which also drops its dependent indexes) so it can be rebuilt.
    conn.execute("CREATE TABLE _spans_repair AS SELECT * FROM spans")
    live_cols = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = '_spans_repair' ORDER BY ordinal_position"
    ).fetchall()
    conn.execute("DROP TABLE spans")
    # Recreate the constraint-bearing base table from the canonical DDL.
    conn.execute(SPANS_TABLE_SQL)
    # Re-add any columns later migrations appended to spans (e.g. billing_account,
    # request_params, request_tools), reading their definitions from the holder
    # so the rebuild tracks the live schema without duplicating the list.
    base_cols = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'spans'"
        ).fetchall()
    }
    for name, data_type in live_cols:
        if name not in base_cols:
            conn.execute(f'ALTER TABLE spans ADD COLUMN "{name}" {data_type}')
    # Column sets now match; BY NAME copy is order-independent.
    conn.execute("INSERT INTO spans BY NAME SELECT * FROM _spans_repair")
    conn.execute("DROP TABLE _spans_repair")
    for statement in SPANS_INDEX_SQL.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)
    conn.execute("CHECKPOINT")


# ---------------------------------------------------------------------------
# DuckDBBackend
# ---------------------------------------------------------------------------

class DuckDBBackend:
    """Concrete DuckDB implementation of StorageBackend."""

    def __init__(self, config: StorageConfig) -> None:
        db_path = Path(config.path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(db_path))
        run_migrations(self._conn)
        self._local = threading.local()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """A per-thread DuckDB cursor over the shared database (#124).

        The daemon's sync (`def`) read routes (`/optimize`, `/cost/compare`)
        run in Starlette's threadpool, so concurrent requests can reach the DB
        from several threads at once. A single DuckDB *connection object* is NOT
        safe for concurrent use — overlapping `execute()` calls abort the
        process (SIGABRT). Cursors created via `connect().cursor()` are
        independent connections over the *same* database that ARE safe to use
        concurrently from different threads (the DuckDB-recommended pattern), so
        each thread lazily gets and reuses its own cursor. All cursors share one
        database, so a write on one thread is visible to reads on another.

        Single-threaded callers (tests, the CLI) always see the same cursor, so
        behavior is unchanged for them.
        """
        cur = getattr(self._local, "cursor", None)
        if cur is None:
            cur = self._conn.cursor()
            self._local.cursor = cur
        return cur

    # -- writes --

    def insert_span(self, span: NormalizedSpan) -> None:
        # Named-column INSERT so future migrations adding columns don't break
        # positional-arg ordering (migration 4 added billing_account at the
        # end of the table, but we don't want to silently rely on that).
        self.conn.execute(
            "INSERT INTO spans ("
            "span_id, trace_id, parent_span_id, session_id, agent_id, "
            "name, kind, status_code, status_message, start_time, end_time, "
            "duration_ms, attributes, provider, model, tool_name, "
            "input_tokens, output_tokens, cache_tokens, cost_usd, "
            "request_type, conversation_id, events, billing_account, "
            "cache_write_tokens, request_params, request_tools, sub_agent_id"
            ") VALUES "
            "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28)",
            [
                span.span_id, span.trace_id, span.parent_span_id, span.session_id,
                span.agent_id, span.name, span.kind.value, span.status_code.value,
                span.status_message, span.start_time, span.end_time, span.duration_ms,
                json.dumps(span.attributes), span.provider, span.model, span.tool_name,
                span.input_tokens, span.output_tokens, span.cache_tokens, span.cost_usd,
                span.request_type, span.conversation_id, json.dumps(span.events),
                span.billing_account, span.cache_write_tokens,
                json.dumps(span.request_params) if span.request_params is not None else None,
                json.dumps(span.request_tools) if span.request_tools is not None else None,
                span.sub_agent_id,
            ],
        )

    def insert_alert(self, alert: Alert) -> None:
        self.conn.execute(
            "INSERT INTO alerts VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            [
                alert.alert_id, alert.agent_id, alert.session_id, alert.span_id,
                alert.fired_at, alert.type.value, alert.severity.value, alert.title,
                json.dumps(alert.detail), alert.acknowledged, alert.suppressed,
            ],
        )

    def insert_validation(self, result: SchemaValidationResult) -> None:
        self.conn.execute(
            "INSERT INTO schema_validations VALUES ($1,$2,$3,$4,$5,$6)",
            [
                result.validation_id, result.span_id, result.agent_id,
                result.validated_at, result.passed, json.dumps(result.errors),
            ],
        )

    def insert_policy_decision(self, decision: PolicyDecisionRecord) -> None:
        # Append-only audit log (#221). Named columns so future migrations stay safe.
        self.conn.execute(
            "INSERT INTO policy_decisions ("
            "decision_id, ts, provider, pricing_mode, gate_decision, path, "
            "policy_name, policy_kind, would_action, passthrough_tos, label, "
            "suggest_only, envelope"
            ") VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)",
            [
                decision.decision_id, decision.ts, decision.provider,
                decision.pricing_mode, decision.gate_decision, decision.path,
                decision.policy_name, decision.policy_kind, decision.would_action,
                decision.passthrough_tos, decision.label, decision.suggest_only,
                json.dumps(decision.envelope) if decision.envelope is not None else None,
            ],
        )

    def insert_savings_entry(self, entry: SavingsLedgerEntry) -> None:
        self.conn.execute(
            "INSERT INTO savings_ledger ("
            "ledger_id, decision_id, ts, provider, pricing_mode, policy_name, "
            "would_action, estimated_recoverable_usd, estimated_recoverable_tokens, "
            "estimate_basis, billing_period, label, realized"
            ") VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)",
            [
                entry.ledger_id, entry.decision_id, entry.ts, entry.provider,
                entry.pricing_mode, entry.policy_name, entry.would_action,
                entry.estimated_recoverable_usd, entry.estimated_recoverable_tokens,
                entry.estimate_basis, entry.billing_period, entry.label,
                entry.realized,
            ],
        )

    def _decision_where(self, filters: PolicyDecisionFilters) -> tuple[str, list]:
        clauses: list[str] = ["1=1"]
        params: list[object] = []
        idx = 1
        if filters.since:
            clauses.append(f"ts >= ${idx}")
            params.append(filters.since)
            idx += 1
        if filters.until:
            clauses.append(f"ts <= ${idx}")
            params.append(filters.until)
            idx += 1
        if filters.provider:
            clauses.append(f"provider = ${idx}")
            params.append(filters.provider)
            idx += 1
        return " AND ".join(clauses), params

    def get_policy_decisions(
        self, filters: PolicyDecisionFilters,
    ) -> list[PolicyDecisionRecord]:
        where, params = self._decision_where(filters)
        rows = self.conn.execute(
            "SELECT decision_id, ts, provider, pricing_mode, gate_decision, path, "
            "policy_name, policy_kind, would_action, passthrough_tos, label, "
            "suggest_only, envelope "
            f"FROM policy_decisions WHERE {where} ORDER BY ts DESC LIMIT ${len(params)+1}",
            [*params, filters.limit],
        ).fetchall()
        out: list[PolicyDecisionRecord] = []
        for r in rows:
            env = r[12]
            if isinstance(env, str):
                env = json.loads(env)
            out.append(PolicyDecisionRecord(
                decision_id=r[0], ts=r[1], provider=r[2], pricing_mode=r[3],
                gate_decision=r[4], path=r[5], policy_name=r[6], policy_kind=r[7],
                would_action=r[8], passthrough_tos=bool(r[9]), label=r[10],
                suggest_only=bool(r[11]), envelope=env,
            ))
        return out

    def get_savings_entries(
        self, filters: PolicyDecisionFilters,
    ) -> list[SavingsLedgerEntry]:
        where, params = self._decision_where(filters)
        rows = self.conn.execute(
            "SELECT ledger_id, decision_id, ts, provider, pricing_mode, policy_name, "
            "would_action, estimated_recoverable_usd, estimated_recoverable_tokens, "
            "estimate_basis, billing_period, label, realized "
            f"FROM savings_ledger WHERE {where} ORDER BY ts DESC LIMIT ${len(params)+1}",
            [*params, filters.limit],
        ).fetchall()
        return [
            SavingsLedgerEntry(
                ledger_id=r[0], decision_id=r[1], ts=r[2], provider=r[3],
                pricing_mode=r[4], policy_name=r[5], would_action=r[6],
                estimated_recoverable_usd=float(r[7] or 0.0),
                estimated_recoverable_tokens=int(r[8] or 0),
                estimate_basis=r[9] or "", billing_period=r[10] or "",
                label=r[11], realized=bool(r[12]),
            )
            for r in rows
        ]

    def upsert_session(self, session: SessionRecord) -> None:
        # plan_tier: promote unknown → known on conflict; never overwrite a
        # session that already has a known tier (backfill re-runs must not
        # clobber historical tiers when config plan changes).
        self.conn.execute(
            """
            INSERT INTO sessions (
                session_id, agent_id, conversation_id, started_at, ended_at,
                status, total_cost_usd, input_tokens, output_tokens, cache_tokens,
                tool_call_count, error_count, plan_tier, service_namespace,
                service_instance_id, cache_write_tokens, run_id, parent_session_id
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
            ON CONFLICT (session_id) DO UPDATE SET
                ended_at = COALESCE(EXCLUDED.ended_at, sessions.ended_at),
                status = EXCLUDED.status,
                total_cost_usd = EXCLUDED.total_cost_usd,
                input_tokens = EXCLUDED.input_tokens,
                output_tokens = EXCLUDED.output_tokens,
                cache_tokens = EXCLUDED.cache_tokens,
                cache_write_tokens = EXCLUDED.cache_write_tokens,
                tool_call_count = EXCLUDED.tool_call_count,
                error_count = EXCLUDED.error_count,
                plan_tier = CASE
                    WHEN COALESCE(sessions.plan_tier, 'unknown') != 'unknown'
                    THEN sessions.plan_tier
                    ELSE EXCLUDED.plan_tier
                END,
                service_namespace = COALESCE(EXCLUDED.service_namespace, sessions.service_namespace),
                service_instance_id = COALESCE(EXCLUDED.service_instance_id, sessions.service_instance_id),
                run_id = COALESCE(EXCLUDED.run_id, sessions.run_id),
                parent_session_id = COALESCE(EXCLUDED.parent_session_id, sessions.parent_session_id)
            """,
            [
                session.session_id, session.agent_id, session.conversation_id,
                session.started_at, session.ended_at, session.status,
                session.total_cost_usd, session.input_tokens, session.output_tokens,
                session.cache_tokens, session.tool_call_count, session.error_count,
                session.plan_tier, session.service_namespace,
                session.service_instance_id, session.cache_write_tokens,
                session.run_id, session.parent_session_id,
            ],
        )

    def recompute_session_totals_from_spans(self, session_ids: list[str]) -> None:
        """Reconcile the given session rows' token + cost aggregates to the SUM
        of their spans (the source of truth).

        Backfill upserts a session row once per on-disk file, but a Claude Code
        session is split across files that share one session_id (the main-thread
        transcript plus each subagents/agent-<id>.jsonl). Because upsert_session
        uses replace semantics, the per-file upserts would otherwise leave the
        row holding only the last-processed file's totals. Scoped to the given
        ids so it never touches live-ingested sessions. Idempotent.
        """
        if not session_ids:
            return
        self.conn.execute(
            """
            UPDATE sessions AS s SET
                input_tokens       = agg.input_tokens,
                output_tokens      = agg.output_tokens,
                cache_tokens       = agg.cache_tokens,
                cache_write_tokens = agg.cache_write_tokens,
                total_cost_usd     = agg.total_cost_usd,
                tool_call_count    = agg.tool_call_count
            FROM (
                SELECT session_id,
                       COALESCE(SUM(input_tokens), 0)       AS input_tokens,
                       COALESCE(SUM(output_tokens), 0)      AS output_tokens,
                       COALESCE(SUM(cache_tokens), 0)       AS cache_tokens,
                       COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                       COALESCE(SUM(cost_usd), 0.0)         AS total_cost_usd,
                       COUNT(*) FILTER (WHERE tool_name IS NOT NULL) AS tool_call_count
                FROM spans
                WHERE session_id IN (SELECT unnest($1))
                GROUP BY session_id
            ) AS agg
            WHERE s.session_id = agg.session_id
            """,
            [list(session_ids)],
        )

    def upsert_agent(self, agent: AgentRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO agents VALUES ($1,$2,$3,$4,$5,$6)
            ON CONFLICT (agent_id) DO UPDATE SET
                name = COALESCE(EXCLUDED.name, agents.name),
                version = COALESCE(EXCLUDED.version, agents.version),
                provider = COALESCE(EXCLUDED.provider, agents.provider),
                last_seen = EXCLUDED.last_seen
            """,
            [
                agent.agent_id, agent.name, agent.version, agent.provider,
                agent.first_seen, agent.last_seen,
            ],
        )

    def upsert_baseline(self, baseline: DriftBaseline) -> None:
        self.conn.execute(
            """
            INSERT INTO drift_baselines VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (agent_id) DO UPDATE SET
                sessions_sampled = EXCLUDED.sessions_sampled,
                computed_at = EXCLUDED.computed_at,
                avg_input_tokens = EXCLUDED.avg_input_tokens,
                stddev_input_tokens = EXCLUDED.stddev_input_tokens,
                avg_output_tokens = EXCLUDED.avg_output_tokens,
                stddev_output_tokens = EXCLUDED.stddev_output_tokens,
                avg_session_duration_s = EXCLUDED.avg_session_duration_s,
                stddev_session_duration = EXCLUDED.stddev_session_duration,
                avg_tool_call_count = EXCLUDED.avg_tool_call_count,
                stddev_tool_call_count = EXCLUDED.stddev_tool_call_count,
                common_tool_sequences = EXCLUDED.common_tool_sequences,
                output_schema_inferred = EXCLUDED.output_schema_inferred
            """,
            [
                baseline.agent_id, baseline.sessions_sampled, baseline.computed_at,
                baseline.avg_input_tokens, baseline.stddev_input_tokens,
                baseline.avg_output_tokens, baseline.stddev_output_tokens,
                baseline.avg_session_duration_s, baseline.stddev_session_duration,
                baseline.avg_tool_call_count, baseline.stddev_tool_call_count,
                json.dumps(baseline.common_tool_sequences),
                json.dumps(baseline.output_schema_inferred),
            ],
        )

    # -- reads --

    def get_session(self, session_id: str) -> SessionRecord | None:
        cur = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = $1", [session_id]
        )
        rows = cur.fetchall()
        if not rows:
            return None
        cols = [d[0] for d in cur.description]
        return _row_to_session(rows[0], cols)

    def get_session_by_conversation(self, conversation_id: str) -> SessionRecord | None:
        cur = self.conn.execute(
            "SELECT * FROM sessions WHERE conversation_id = $1 "
            "ORDER BY started_at DESC LIMIT 1",
            [conversation_id],
        )
        rows = cur.fetchall()
        if not rows:
            return None
        cols = [d[0] for d in cur.description]
        return _row_to_session(rows[0], cols)

    def close_sessions_by_instance(self, instance_id: str) -> int:
        """Mark all currently-active sessions for a terminal as 'closed'.

        Returns the number closed. Idempotent: already-closed/completed rows are
        not matched (status='active' filter), so re-closing is a no-op (0).
        ended_at is the session's last-activity time ("Last seen" in the UI), so
        closing must NOT advance it — a session closed long after its last span
        still last had telemetry at that span. Only stamp ended_at when it's
        NULL (a session that never recorded an end gets the close time).
        """
        now = utcnow()
        count_row = self.conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE service_instance_id = $1 AND status = 'active'",
            [instance_id],
        ).fetchone()
        count = count_row[0] if count_row else 0
        if count:
            self.conn.execute(
                "UPDATE sessions SET status = 'closed', "
                "ended_at = COALESCE(ended_at, $2) "
                "WHERE service_instance_id = $1 AND status = 'active'",
                [instance_id, now],
            )
        return count

    def close_session_by_id(self, session_id: str) -> int:
        """Mark a single active session as 'closed'. Idempotent (see above).

        Preserves ended_at (last-activity / "Last seen"); only stamps it when
        NULL. Closing is not telemetry, so it must not advance last-seen.
        """
        now = utcnow()
        count_row = self.conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE session_id = $1 AND status = 'active'",
            [session_id],
        ).fetchone()
        count = count_row[0] if count_row else 0
        if count:
            self.conn.execute(
                "UPDATE sessions SET status = 'closed', "
                "ended_at = COALESCE(ended_at, $2) "
                "WHERE session_id = $1 AND status = 'active'",
                [session_id, now],
            )
        return count

    def _trace_filter_where(self, filters: TraceFilters) -> tuple[str, list[object], int]:
        clauses: list[str] = []
        params: list[object] = []
        idx = 1
        if filters.agent_id:
            clauses.append(f"agent_id = ${idx}")
            params.append(filters.agent_id)
            idx += 1
        if filters.since:
            clauses.append(f"start_time >= ${idx}")
            params.append(filters.since)
            idx += 1
        if filters.until:
            clauses.append(f"start_time <= ${idx}")
            params.append(filters.until)
            idx += 1
        if filters.span_name:
            clauses.append(f"name = ${idx}")
            params.append(filters.span_name)
            idx += 1
        if filters.status:
            clauses.append(f"status_code = ${idx}")
            params.append(filters.status)
            idx += 1
        where = " AND ".join(clauses) if clauses else "1=1"
        return where, params, idx

    def get_traces(self, filters: TraceFilters) -> list[TraceRecord]:
        where, params, idx = self._trace_filter_where(filters)
        # Use FIRST(name ORDER BY start_time) to pick the root span name —
        # the previous correlated-subquery variant returned NULL for most
        # rows in DuckDB, leaving the TYPE column blank in `tj traces` (U2).
        sql = (
            f"SELECT trace_id, MAX(agent_id) AS agent_id, "
            f"FIRST(name ORDER BY start_time) AS name, "
            f"MIN(start_time) AS start_time, "
            f"SUM(duration_ms) AS duration_ms, "
            f"SUM(cost_usd) AS cost_usd, "
            f"CASE WHEN SUM(CASE WHEN status_code='error' THEN 1 ELSE 0 END) > 0 THEN 'error' "
            f"     WHEN SUM(CASE WHEN status_code='ok' THEN 1 ELSE 0 END) > 0 THEN 'ok' "
            f"     ELSE 'unset' END AS status_code, "
            f"COUNT(*) AS span_count, "
            f"SUM(input_tokens) AS input_tokens, "
            f"SUM(output_tokens) AS output_tokens "
            f"FROM spans WHERE {where} "
            f"GROUP BY trace_id "
            f"ORDER BY start_time DESC "
            f"LIMIT ${idx} OFFSET ${idx + 1}"
        )
        params.extend([filters.limit, filters.offset])
        rows = self.conn.execute(sql, params).fetchall()
        return [
            TraceRecord(
                trace_id=r[0], agent_id=r[1], name=r[2], start_time=r[3],
                duration_ms=r[4], cost_usd=r[5], status_code=r[6],
                span_count=r[7],
                input_tokens=int(r[8] or 0), output_tokens=int(r[9] or 0),
            )
            for r in rows
        ]

    def count_traces(self, filters: TraceFilters) -> int:
        where, params, _ = self._trace_filter_where(filters)
        row = self.conn.execute(f"SELECT COUNT(DISTINCT trace_id) FROM spans WHERE {where}", params).fetchone()
        return int(row[0] or 0) if row else 0

    def get_trace_spans(self, trace_id: str) -> list[NormalizedSpan]:
        cur = self.conn.execute(
            "SELECT * FROM spans WHERE trace_id = $1 ORDER BY start_time", [trace_id]
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [_row_to_span(r, cols) for r in rows]

    def get_cost_summary(self, filters: CostFilters) -> list[CostRow]:
        group_col_map = {
            "day": "CAST(start_time AS DATE)",
            "agent": "agent_id",
            "model": "model",
            "tool": "tool_name",
        }
        group_expr = group_col_map.get(filters.group_by, "CAST(start_time AS DATE)")

        clauses: list[str] = ["model IS NOT NULL"]
        params: list[object] = []
        idx = 1
        if filters.agent_id:
            clauses.append(f"agent_id = ${idx}")
            params.append(filters.agent_id)
            idx += 1
        if filters.since:
            clauses.append(f"start_time >= ${idx}")
            params.append(filters.since)
            idx += 1
        if filters.until:
            clauses.append(f"start_time <= ${idx}")
            params.append(filters.until)
            idx += 1
        where = " AND ".join(clauses)

        # Cache-read + cache-write are summed alongside in/out so callers can
        # show the full token picture (cache-write is often the dominant cost
        # driver yet was invisible above the DB — issue #17).
        token_cols = (
            "COALESCE(SUM(input_tokens), 0), "
            "COALESCE(SUM(output_tokens), 0), "
            "COALESCE(SUM(cache_tokens), 0), "
            "COALESCE(SUM(cache_write_tokens), 0), "
            "COALESCE(SUM(cost_usd), 0.0) "
        )
        if filters.group_by in ("agent", "model"):
            sql = (
                f"SELECT {group_expr} AS grp, agent_id, model, " + token_cols
                + f"FROM spans WHERE {where} "
                f"GROUP BY grp, agent_id, model "
                f"ORDER BY grp DESC"
            )
        else:
            # day / tool: group only by the primary expression to avoid cross-product
            sql = (
                f"SELECT {group_expr} AS grp, NULL AS agent_id, NULL AS model, " + token_cols
                + f"FROM spans WHERE {where} "
                f"GROUP BY grp "
                f"ORDER BY grp DESC"
            )
        rows = self.conn.execute(sql, params).fetchall()
        return [
            CostRow(
                group=str(r[0]), agent_id=r[1], model=r[2],
                input_tokens=r[3] or 0, output_tokens=r[4] or 0,
                cache_tokens=r[5] or 0, cache_write_tokens=r[6] or 0,
                cost_usd=r[7] or 0.0,
            )
            for r in rows
        ]

    def get_alerts(self, filters: AlertFilters) -> list[Alert]:
        from tokenjam.core.models import AlertType, Severity

        clauses: list[str] = []
        params: list[object] = []
        idx = 1
        if filters.agent_id:
            clauses.append(f"agent_id = ${idx}")
            params.append(filters.agent_id)
            idx += 1
        if filters.since:
            clauses.append(f"fired_at >= ${idx}")
            params.append(filters.since)
            idx += 1
        if filters.severity:
            clauses.append(f"severity = ${idx}")
            params.append(filters.severity.value)
            idx += 1
        if filters.type:
            clauses.append(f"type = ${idx}")
            params.append(filters.type.value)
            idx += 1
        if filters.unread:
            clauses.append("acknowledged = false")
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = (
            f"SELECT * FROM alerts WHERE {where} "
            f"ORDER BY fired_at DESC LIMIT ${idx}"
        )
        params.append(filters.limit)
        cur = self.conn.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        results = []
        for row in rows:
            d = dict(zip(cols, row))
            detail = d.get("detail") or {}
            if isinstance(detail, str):
                detail = json.loads(detail)
            results.append(Alert(
                alert_id=d["alert_id"],
                fired_at=d["fired_at"],
                type=AlertType(d["type"]),
                severity=Severity(d["severity"]),
                title=d["title"],
                detail=detail,
                agent_id=d.get("agent_id"),
                session_id=d.get("session_id"),
                span_id=d.get("span_id"),
                acknowledged=d.get("acknowledged", False),
                suppressed=d.get("suppressed", False),
            ))
        return results

    def get_baseline(self, agent_id: str) -> DriftBaseline | None:
        cur = self.conn.execute(
            "SELECT * FROM drift_baselines WHERE agent_id = $1", [agent_id]
        )
        rows = cur.fetchall()
        if not rows:
            return None
        cols = [d[0] for d in cur.description]
        d = dict(zip(cols, rows[0]))
        cts = d.get("common_tool_sequences")
        if isinstance(cts, str):
            cts = json.loads(cts)
        osi = d.get("output_schema_inferred")
        if isinstance(osi, str):
            osi = json.loads(osi)
        return DriftBaseline(
            agent_id=d["agent_id"],
            sessions_sampled=d["sessions_sampled"],
            computed_at=d["computed_at"],
            avg_input_tokens=d.get("avg_input_tokens"),
            stddev_input_tokens=d.get("stddev_input_tokens"),
            avg_output_tokens=d.get("avg_output_tokens"),
            stddev_output_tokens=d.get("stddev_output_tokens"),
            avg_session_duration_s=d.get("avg_session_duration_s"),
            stddev_session_duration=d.get("stddev_session_duration"),
            avg_tool_call_count=d.get("avg_tool_call_count"),
            stddev_tool_call_count=d.get("stddev_tool_call_count"),
            common_tool_sequences=cts,
            output_schema_inferred=osi,
        )

    def get_completed_sessions(self, agent_id: str, limit: int) -> list[SessionRecord]:
        # Order by last activity (ended_at), not start time. A short fragment
        # that *started* later must not hide a long-running session that was
        # still active afterwards — otherwise the status tile shows a 40s blip
        # instead of the real multi-hour session. Falls back to started_at when
        # ended_at is NULL.
        cur = self.conn.execute(
            "SELECT * FROM sessions WHERE agent_id = $1 AND status = 'completed' "
            "ORDER BY COALESCE(ended_at, started_at) DESC LIMIT $2",
            [agent_id, limit],
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [_row_to_session(r, cols) for r in rows]

    def get_completed_session_count(self, agent_id: str) -> int:
        result = self.conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE agent_id = $1 AND status = 'completed'",
            [agent_id],
        ).fetchone()
        return result[0] if result else 0

    def get_tool_calls(
        self, agent_id: str | None, since: datetime | None, tool_name: str | None,
    ) -> list[dict]:
        clauses = ["tool_name IS NOT NULL"]
        params: list[object] = []
        idx = 1
        if agent_id:
            clauses.append(f"agent_id = ${idx}")
            params.append(agent_id)
            idx += 1
        if since:
            clauses.append(f"start_time >= ${idx}")
            params.append(since)
            idx += 1
        if tool_name:
            clauses.append(f"tool_name = ${idx}")
            params.append(tool_name)
            idx += 1
        where = " AND ".join(clauses)
        rows = self.conn.execute(
            f"SELECT tool_name, agent_id, COUNT(*) AS call_count, "
            f"COALESCE(SUM(duration_ms), 0) AS total_duration_ms "
            f"FROM spans WHERE {where} "
            f"GROUP BY tool_name, agent_id ORDER BY call_count DESC",
            params,
        ).fetchall()
        return [
            {"tool_name": r[0], "agent_id": r[1], "call_count": r[2], "total_duration_ms": r[3]}
            for r in rows
        ]

    def get_daily_cost(self, agent_id: str, date: date) -> float:
        result = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spans "
            "WHERE agent_id = $1 AND CAST(start_time AT TIME ZONE 'UTC' AS DATE) = $2",
            [agent_id, date],
        ).fetchone()
        return float(result[0]) if result else 0.0

    def get_session_cost(self, session_id: str) -> float:
        result = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spans WHERE session_id = $1",
            [session_id],
        ).fetchone()
        return float(result[0]) if result else 0.0

    def get_recent_spans(self, session_id: str, limit: int) -> list[NormalizedSpan]:
        cur = self.conn.execute(
            "SELECT * FROM spans WHERE session_id = $1 ORDER BY start_time DESC LIMIT $2",
            [session_id, limit],
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [_row_to_span(r, cols) for r in rows]

    # -- issue #309: queries moved off direct db.conn access in callers --

    def update_span_cost(self, span_id: str, cost_usd: float) -> None:
        self.conn.execute(
            "UPDATE spans SET cost_usd = $1 WHERE span_id = $2",
            [cost_usd, span_id],
        )

    def increment_session_cost(self, session_id: str, delta_usd: float) -> None:
        self.conn.execute(
            "UPDATE sessions SET total_cost_usd = COALESCE(total_cost_usd, 0) + $1 "
            "WHERE session_id = $2",
            [delta_usd, session_id],
        )

    def get_distinct_agent_ids(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT agent_id FROM sessions WHERE agent_id IS NOT NULL "
            "ORDER BY agent_id"
        ).fetchall()
        return [r[0] for r in rows]

    def get_active_session(self, agent_id: str) -> SessionRecord | None:
        cur = self.conn.execute(
            "SELECT * FROM sessions WHERE agent_id = $1 AND status = 'active' "
            "ORDER BY started_at DESC LIMIT 1",
            [agent_id],
        )
        rows = cur.fetchall()
        if not rows:
            return None
        cols = [d[0] for d in cur.description]
        return _row_to_session(rows[0], cols)

    def get_session_active_seconds(self, session_id: str) -> float | None:
        return session_active_seconds(self.conn, session_id)

    def count_unknown_plan_tier_sessions(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM sessions "
            "WHERE plan_tier IS NULL OR plan_tier = 'unknown'"
        ).fetchone()
        return int(row[0]) if row else 0

    def get_window_cost_totals(
        self, since: datetime, until: datetime, agent_id: str | None = None,
    ) -> tuple[int, int, int, int, float]:
        clauses = ["start_time >= $1", "start_time < $2"]
        params: list = [since, until]
        if agent_id:
            clauses.append(f"agent_id = ${len(params) + 1}")
            params.append(agent_id)
        where = " AND ".join(clauses)
        row = self.conn.execute(
            f"SELECT COUNT(DISTINCT session_id) AS sessions, "
            f"COALESCE(SUM(input_tokens), 0)   AS in_tok, "
            f"COALESCE(SUM(output_tokens), 0)  AS out_tok, "
            f"COALESCE(SUM(cache_tokens), 0)   AS cache_tok, "
            f"COALESCE(SUM(cost_usd), 0.0)     AS cost "
            f"FROM spans WHERE {where}",
            params,
        ).fetchone()
        if row is None:  # COALESCE aggregate always returns a row; guard for typing
            return (0, 0, 0, 0, 0.0)
        return (
            int(row[0] or 0), int(row[1] or 0), int(row[2] or 0),
            int(row[3] or 0), float(row[4] or 0.0),
        )

    def get_cost_delta_by_group(
        self, group_col: str, current_since: datetime, current_until: datetime,
        prev_since: datetime, prev_until: datetime, top_n: int,
    ) -> list[dict]:
        # group_col is an internal, fixed identifier (never user input); the
        # allow-list keeps it that way so the interpolation below stays safe.
        if group_col not in ("agent_id", "model"):
            raise ValueError(f"Unsupported group_col {group_col!r}")
        sql = f"""
            SELECT {group_col} AS grp,
                   COALESCE(SUM(CASE WHEN start_time >= $1 AND start_time < $2
                                     THEN cost_usd ELSE 0 END), 0.0) AS cur_cost,
                   COALESCE(SUM(CASE WHEN start_time >= $3 AND start_time < $4
                                     THEN cost_usd ELSE 0 END), 0.0) AS prev_cost
            FROM spans
            WHERE (start_time >= $3 AND start_time < $2)
              AND {group_col} IS NOT NULL
            GROUP BY {group_col}
            HAVING ABS(cur_cost - prev_cost) > 0.0001
            ORDER BY ABS(cur_cost - prev_cost) DESC
            LIMIT $5
        """
        rows = self.conn.execute(
            sql, [current_since, current_until, prev_since, prev_until, top_n],
        ).fetchall()
        return [
            {"group": r[0], "current_cost": float(r[1]), "previous_cost": float(r[2]),
             "delta": float(r[1]) - float(r[2])}
            for r in rows
        ]

    def delete_spans_before(self, cutoff: datetime) -> int:
        result = self.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE start_time < $1", [cutoff]
        ).fetchone()
        count = result[0] if result else 0
        self.conn.execute("DELETE FROM spans WHERE start_time < $1", [cutoff])
        return count

    def close(self) -> None:
        # Closing the root connection tears down the database and all cursors.
        self._conn.close()


# ---------------------------------------------------------------------------
# InMemoryBackend (for tests)
# ---------------------------------------------------------------------------

class InMemoryBackend(DuckDBBackend):
    """In-memory DuckDB backend for tests. Same implementation, no disk I/O."""

    def __init__(self) -> None:
        # Bypass DuckDBBackend.__init__ to use :memory:. Cursors of an in-memory
        # connection share the same in-memory database, so the per-thread cursor
        # property (#124) works identically here — including cross-thread
        # visibility, which the threadpool-backed integration tests rely on.
        self._conn = duckdb.connect(":memory:")
        run_migrations(self._conn)
        self._local = threading.local()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def open_db(config: StorageConfig) -> DuckDBBackend:
    """Open the database and return a backend instance."""
    return DuckDBBackend(config)
