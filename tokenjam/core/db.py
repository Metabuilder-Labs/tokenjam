"""
Database layer: StorageBackend protocol, DuckDB implementation, InMemoryBackend for tests,
and migration runner. DuckDB only — never import sqlite3.
"""
from __future__ import annotations

import json
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
    def upsert_session(self, session: SessionRecord) -> None: ...
    def upsert_agent(self, agent: AgentRecord) -> None: ...
    def upsert_baseline(self, baseline: DriftBaseline) -> None: ...
    def get_session(self, session_id: str) -> SessionRecord | None: ...
    def get_session_by_conversation(self, conversation_id: str) -> SessionRecord | None: ...
    def get_traces(self, filters: TraceFilters) -> list[TraceRecord]: ...
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
    def delete_spans_before(self, cutoff: datetime) -> int: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Schema & migrations
# ---------------------------------------------------------------------------

INITIAL_SCHEMA_SQL = """\
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
    tool_call_count     INTEGER DEFAULT 0,
    error_count         INTEGER DEFAULT 0
);

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
    events              JSON DEFAULT '[]'
);

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

MIGRATIONS: list[tuple[int, str]] = [
    (1, INITIAL_SCHEMA_SQL),
    (2, (
        "DROP INDEX IF EXISTS idx_spans_trace_id;\n"
        "DROP INDEX IF EXISTS idx_spans_agent_id;\n"
        "DROP INDEX IF EXISTS idx_spans_start_time;\n"
        "DROP INDEX IF EXISTS idx_spans_tool_name;\n"
        "DROP INDEX IF EXISTS idx_spans_conv_id"
    )),
    (3, (
        "CREATE INDEX IF NOT EXISTS idx_spans_trace_id    ON spans(trace_id);\n"
        "CREATE INDEX IF NOT EXISTS idx_spans_agent_id    ON spans(agent_id);\n"
        "CREATE INDEX IF NOT EXISTS idx_spans_start_time  ON spans(start_time);\n"
        "CREATE INDEX IF NOT EXISTS idx_spans_tool_name   ON spans(tool_name);\n"
        "CREATE INDEX IF NOT EXISTS idx_spans_conv_id     ON spans(conversation_id)"
    )),
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
    # Migration 5: service_namespace on sessions — the OTel service.namespace
    # the session's service rolls up under (the dashboard's "project" grouping
    # key). Nullable; sessions whose telemetry carried no namespace stay NULL.
    (5, "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS service_namespace TEXT"),
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
        cost_usd=d.get("cost_usd"),
        request_type=d.get("request_type"),
        conversation_id=d.get("conversation_id"),
        billing_account=d.get("billing_account"),
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
        tool_call_count=d.get("tool_call_count") or 0,
        error_count=d.get("error_count") or 0,
        plan_tier=d.get("plan_tier") or "unknown",
        service_namespace=d.get("service_namespace"),
    )


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
    """
    conn.execute("CREATE TABLE _spans_repair AS SELECT * FROM spans")
    conn.execute("DROP TABLE spans")
    conn.execute("ALTER TABLE _spans_repair RENAME TO spans")
    conn.execute("CHECKPOINT")


# ---------------------------------------------------------------------------
# DuckDBBackend
# ---------------------------------------------------------------------------

class DuckDBBackend:
    """Concrete DuckDB implementation of StorageBackend."""

    def __init__(self, config: StorageConfig) -> None:
        db_path = Path(config.path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(db_path))
        run_migrations(self.conn)

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
            "request_type, conversation_id, events, billing_account"
            ") VALUES "
            "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24)",
            [
                span.span_id, span.trace_id, span.parent_span_id, span.session_id,
                span.agent_id, span.name, span.kind.value, span.status_code.value,
                span.status_message, span.start_time, span.end_time, span.duration_ms,
                json.dumps(span.attributes), span.provider, span.model, span.tool_name,
                span.input_tokens, span.output_tokens, span.cache_tokens, span.cost_usd,
                span.request_type, span.conversation_id, json.dumps(span.events),
                span.billing_account,
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

    def upsert_session(self, session: SessionRecord) -> None:
        # Named-column INSERT for migration safety (plan_tier added in migration 4).
        # plan_tier is updated on conflict because IngestPipeline late-resolves
        # it: a session that started with billing_account=None (e.g. tool span
        # first) can be promoted from 'unknown' to a real plan_tier when a
        # later LLM span carries billing_account. The Python side never demotes
        # a known plan_tier back to 'unknown', so always copying EXCLUDED is safe.
        self.conn.execute(
            """
            INSERT INTO sessions (
                session_id, agent_id, conversation_id, started_at, ended_at,
                status, total_cost_usd, input_tokens, output_tokens, cache_tokens,
                tool_call_count, error_count, plan_tier, service_namespace
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
            ON CONFLICT (session_id) DO UPDATE SET
                ended_at = COALESCE(EXCLUDED.ended_at, sessions.ended_at),
                status = EXCLUDED.status,
                total_cost_usd = EXCLUDED.total_cost_usd,
                input_tokens = EXCLUDED.input_tokens,
                output_tokens = EXCLUDED.output_tokens,
                cache_tokens = EXCLUDED.cache_tokens,
                tool_call_count = EXCLUDED.tool_call_count,
                error_count = EXCLUDED.error_count,
                plan_tier = EXCLUDED.plan_tier,
                service_namespace = COALESCE(EXCLUDED.service_namespace, sessions.service_namespace)
            """,
            [
                session.session_id, session.agent_id, session.conversation_id,
                session.started_at, session.ended_at, session.status,
                session.total_cost_usd, session.input_tokens, session.output_tokens,
                session.cache_tokens, session.tool_call_count, session.error_count,
                session.plan_tier, session.service_namespace,
            ],
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

    def get_traces(self, filters: TraceFilters) -> list[TraceRecord]:
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
            f"COUNT(*) AS span_count "
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
            )
            for r in rows
        ]

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

        if filters.group_by in ("agent", "model"):
            sql = (
                f"SELECT {group_expr} AS grp, agent_id, model, "
                f"COALESCE(SUM(input_tokens), 0), "
                f"COALESCE(SUM(output_tokens), 0), "
                f"COALESCE(SUM(cost_usd), 0.0) "
                f"FROM spans WHERE {where} "
                f"GROUP BY grp, agent_id, model "
                f"ORDER BY grp DESC"
            )
        else:
            # day / tool: group only by the primary expression to avoid cross-product
            sql = (
                f"SELECT {group_expr} AS grp, NULL AS agent_id, NULL AS model, "
                f"COALESCE(SUM(input_tokens), 0), "
                f"COALESCE(SUM(output_tokens), 0), "
                f"COALESCE(SUM(cost_usd), 0.0) "
                f"FROM spans WHERE {where} "
                f"GROUP BY grp "
                f"ORDER BY grp DESC"
            )
        rows = self.conn.execute(sql, params).fetchall()
        return [
            CostRow(
                group=str(r[0]), agent_id=r[1], model=r[2],
                input_tokens=r[3] or 0, output_tokens=r[4] or 0, cost_usd=r[5] or 0.0,
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
        cur = self.conn.execute(
            "SELECT * FROM sessions WHERE agent_id = $1 AND status = 'completed' "
            "ORDER BY started_at DESC LIMIT $2",
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

    def delete_spans_before(self, cutoff: datetime) -> int:
        result = self.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE start_time < $1", [cutoff]
        ).fetchone()
        count = result[0] if result else 0
        self.conn.execute("DELETE FROM spans WHERE start_time < $1", [cutoff])
        return count

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# InMemoryBackend (for tests)
# ---------------------------------------------------------------------------

class InMemoryBackend(DuckDBBackend):
    """In-memory DuckDB backend for tests. Same implementation, no disk I/O."""

    def __init__(self) -> None:
        # Bypass DuckDBBackend.__init__ to use :memory:
        self.conn = duckdb.connect(":memory:")
        run_migrations(self.conn)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def open_db(config: StorageConfig) -> DuckDBBackend:
    """Open the database and return a backend instance."""
    return DuckDBBackend(config)
