"""
Database layer: StorageBackend protocol, DuckDB implementation, InMemoryBackend for tests,
and migration runner. DuckDB only — never import sqlite3.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import tempfile
import threading
import uuid
import weakref
from contextlib import ExitStack
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, Sequence, cast, runtime_checkable

import duckdb

from tokenjam.core.config import StorageConfig
from tokenjam.core.data_span import MIN_PLAUSIBLE_YEAR
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
    TERMINAL_STATUSES,
    TraceCostStats,
    TraceFilters,
    TraceRecord,
)
from tokenjam.core.persona_scope import add_persona_clause, persona_agent_clause
from tokenjam.utils.time_parse import utcnow

logger = logging.getLogger("tokenjam.db")


def _is_cost_outlier(
    cost_usd: float | None,
    q1: float | None,
    q3: float | None,
    priced_count: int,
    min_sample: int,
) -> bool:
    """Tukey's-fence outlier check shared by get_traces / get_trace_cost_stats.

    False whenever there isn't enough priced-trace history to trust the
    quartiles (`priced_count < min_sample`), or this trace itself has no
    positive cost — a $0 trace is never an "outlier," it's just unpriced.
    """
    if cost_usd is None or cost_usd <= 0:
        return False
    if q1 is None or q3 is None or priced_count < min_sample:
        return False
    fence = q3 + 1.5 * (q3 - q1)
    return cost_usd > fence


# ---------------------------------------------------------------------------
# StorageBackend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StorageBackend(Protocol):
    def insert_span(self, span: NormalizedSpan) -> None: ...
    def bulk_insert_spans(self, spans: Sequence[NormalizedSpan]) -> None: ...
    def bulk_overlay_span_attrs(
        self, updates: Sequence[tuple[str, str | None, str | None, dict | None]],
    ) -> int: ...
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
    def upsert_session(
        self, session: SessionRecord, *, accumulate_totals: bool = False,
    ) -> None: ...
    def upsert_agent(self, agent: AgentRecord) -> None: ...
    def upsert_baseline(self, baseline: DriftBaseline) -> None: ...
    def get_session(self, session_id: str) -> SessionRecord | None: ...
    def get_session_by_conversation(self, conversation_id: str) -> SessionRecord | None: ...
    def close_sessions_by_instance(self, instance_id: str) -> int: ...
    def close_session_by_id(self, session_id: str) -> int: ...
    def mark_sessions_completed(self, session_ids: list[str]) -> None: ...
    def get_traces(self, filters: TraceFilters) -> list[TraceRecord]: ...
    def count_traces(self, filters: TraceFilters) -> int: ...
    def get_trace_spans(self, trace_id: str) -> list[NormalizedSpan]: ...
    def get_span(self, trace_id: str, span_id: str) -> NormalizedSpan | None: ...
    def get_trace_cost_stats(self, filters: TraceFilters) -> TraceCostStats: ...
    def get_session_ids_for_trace(self, trace_id: str) -> list[str]: ...
    def get_marker_session_ids_for_trace(self, trace_id: str) -> list[str]: ...
    def reconcile_trace_session_attribution(self, trace_id: str) -> None: ...
    def get_session_id_for_trace(self, trace_id: str) -> str | None: ...
    def get_cost_summary(self, filters: CostFilters) -> list[CostRow]: ...
    def get_alerts(self, filters: AlertFilters) -> list[Alert]: ...
    def get_baseline(self, agent_id: str) -> DriftBaseline | None: ...
    def get_completed_sessions(self, agent_id: str, limit: int) -> list[SessionRecord]: ...
    def get_completed_session_count(self, agent_id: str) -> int: ...
    def get_tool_calls(
        self, agent_id: str | None, since: datetime | None, tool_name: str | None,
    ) -> list[dict]: ...
    def get_daily_cost(self, agent_id: str, date: date) -> float: ...
    def get_daily_cost_for_agents(self, agent_ids: list[str], date: date) -> float: ...
    def get_session_cost(self, session_id: str) -> float: ...
    def get_recent_spans(self, session_id: str, limit: int) -> list[NormalizedSpan]: ...
    # Issue #309: methods that callers (CostEngine, cmd_status, cost compare)
    # used to satisfy by reaching into `db.conn` directly. Having them on the
    # protocol keeps those paths behind the abstraction and lets InMemoryBackend
    # exercise them in unit tests.
    def update_span_cost(
        self, span_id: str, cost_usd: float, pricing_source: str | None = None,
    ) -> None: ...
    def increment_session_cost(self, session_id: str, delta_usd: float) -> None: ...
    def get_distinct_agent_ids(self) -> list[str]: ...
    def get_active_session(self, agent_id: str) -> SessionRecord | None: ...
    def get_session_active_seconds(self, session_id: str) -> float | None: ...
    def count_unknown_plan_tier_sessions(self) -> int: ...
    def get_window_cost_totals(
        self, since: datetime, until: datetime, agent_id: str | None = None,
        persona: str | None = None,
    ) -> tuple[int, int, int, int, int, float]: ...
    def get_cost_delta_by_group(
        self, group_col: str, current_since: datetime, current_until: datetime,
        prev_since: datetime, prev_until: datetime, top_n: int,
        persona: str | None = None,
    ) -> list[dict]: ...
    def delete_spans_before(
        self,
        cutoff: datetime,
        *,
        retention_days: int | None = None,
        analysis_span_days: int | None = None,
    ) -> tuple[int, int]: ...
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
    -- NOT NULL, and deliberately so. A row with no observed time is REJECTED at
    -- ingest rather than stored (see api/routes/logs.py) — the alternative,
    -- a nullable column, moves the problem into ~25 `ORDER BY start_time` sites
    -- whose null placement is a DuckDB session setting rather than a property of
    -- the query, and buys only the ability to keep rows that can never time
    -- anything. What must never appear here is an epoch SENTINEL: a 1970 stamp
    -- participates in MIN() and drags a span measure back by decades.
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

# Secondary indexes on spans, as (index name, indexed column). Single-sourced so
# migration 3, the repair path, and the integrity check all speak about the same
# set; keep in sync with the DROPs in migration 2. The index-corruption check
# below probes each entry individually, so a name added here is checked without
# any further edit — the point being that the check cannot fall behind the
# schema the way a hand-kept second list would.
SPANS_INDEXES: tuple[tuple[str, str], ...] = (
    ("idx_spans_trace_id",   "trace_id"),
    ("idx_spans_agent_id",   "agent_id"),
    ("idx_spans_start_time", "start_time"),
    ("idx_spans_tool_name",  "tool_name"),
    ("idx_spans_conv_id",    "conversation_id"),
    ("idx_spans_session_id", "session_id"),
)

# Secondary indexes on sessions. Single-sourced for the same reason as the spans
# set above: DuckDB refuses to ALTER a column on a table carrying ART indexes, so
# migration 21 has to drop and re-issue these, and a second copy of the DDL there
# would be free to drift from the one the initial schema creates.
SESSIONS_INDEXES: tuple[tuple[str, str], ...] = (
    ("idx_sessions_agent_id", "agent_id"),
    ("idx_sessions_conv_id",  "conversation_id"),
)

SESSIONS_INDEX_SQL = ";\n".join(
    f"CREATE INDEX IF NOT EXISTS {name} ON sessions({column})"
    for name, column in SESSIONS_INDEXES
)

SPANS_INDEX_SQL = ";\n".join(
    f"CREATE INDEX IF NOT EXISTS {name} ON spans({column})"
    for name, column in SPANS_INDEXES
)

# ---------------------------------------------------------------------------
# Columnar bulk-append of spans (backfill hot path)
# ---------------------------------------------------------------------------
#
# The per-row `insert_span` (and `executemany`) marshals every span across the
# Python<->DuckDB boundary one row at a time — measured ~350x slower than the
# path below and the dominant cost of a full-history backfill. `bulk_insert_spans`
# instead writes the whole batch once as newline-delimited JSON and lets DuckDB's
# native `read_json` scan it in a single vectorized INSERT..SELECT. This is
# dependency-free (no pandas/pyarrow — DuckDB reads JSON natively), so the lean
# base install is unchanged.
#
# `_SPAN_BULK_COLUMNS` mirrors `insert_span`'s named-column list exactly (same 28
# columns, same order). `_SPAN_BULK_READ_TYPES` gives `read_json` an explicit
# schema so key order / type inference can never drift: JSON columns stay JSON,
# timestamps arrive as strings and are cast to TIMESTAMPTZ in the SELECT, numerics
# are pinned. Keep all three in lock-step with the table + `insert_span`.
_SPAN_BULK_COLUMNS: tuple[str, ...] = (
    "span_id", "trace_id", "parent_span_id", "session_id", "agent_id",
    "name", "kind", "status_code", "status_message", "start_time", "end_time",
    "duration_ms", "attributes", "provider", "model", "tool_name",
    "input_tokens", "output_tokens", "cache_tokens", "cost_usd",
    "request_type", "conversation_id", "events", "billing_account",
    "cache_write_tokens", "request_params", "request_tools", "sub_agent_id",
    "tenant_id", "feature", "environment", "service_version", "commit_sha",
    "prompt_template_id", "prompt_template_version", "pricing_source",
    "sub_agent_type",
)

# read_json column -> type. Timestamps are read as VARCHAR and cast to TIMESTAMPTZ
# in the projection (matches how binding a tz-aware datetime lands the same UTC
# instant). JSON columns stay JSON so nested objects/arrays are stored as JSON
# values — NOT as double-encoded strings.
_SPAN_BULK_READ_TYPES: dict[str, str] = {
    "span_id": "VARCHAR", "trace_id": "VARCHAR", "parent_span_id": "VARCHAR",
    "session_id": "VARCHAR", "agent_id": "VARCHAR", "name": "VARCHAR",
    "kind": "VARCHAR", "status_code": "VARCHAR", "status_message": "VARCHAR",
    "start_time": "VARCHAR", "end_time": "VARCHAR", "duration_ms": "DOUBLE",
    "attributes": "JSON", "provider": "VARCHAR", "model": "VARCHAR",
    "tool_name": "VARCHAR", "input_tokens": "BIGINT", "output_tokens": "BIGINT",
    "cache_tokens": "BIGINT", "cost_usd": "DOUBLE", "request_type": "VARCHAR",
    "conversation_id": "VARCHAR", "events": "JSON", "billing_account": "VARCHAR",
    "cache_write_tokens": "BIGINT", "request_params": "JSON",
    "request_tools": "JSON", "sub_agent_id": "VARCHAR",
    "tenant_id": "VARCHAR", "feature": "VARCHAR", "environment": "VARCHAR",
    "service_version": "VARCHAR", "commit_sha": "VARCHAR",
    "prompt_template_id": "VARCHAR", "prompt_template_version": "VARCHAR",
    "pricing_source": "VARCHAR", "sub_agent_type": "VARCHAR",
}

# Columns that need a cast in the SELECT (read as VARCHAR, stored as TIMESTAMPTZ).
_SPAN_BULK_CAST = {"start_time": "TIMESTAMPTZ", "end_time": "TIMESTAMPTZ"}

# read_json objects are tiny per span, but a captured-content span can be large;
# give read_json generous headroom so a fat prompt never trips the default cap.
_SPAN_BULK_MAX_OBJECT_BYTES = 256 * 1024 * 1024


def _build_bulk_span_insert_sql() -> str:
    cols = ", ".join(_SPAN_BULK_COLUMNS)
    projection = ", ".join(
        f"{c}::{_SPAN_BULK_CAST[c]}" if c in _SPAN_BULK_CAST else c
        for c in _SPAN_BULK_COLUMNS
    )
    read_cols = ", ".join(
        f"'{c}': '{_SPAN_BULK_READ_TYPES[c]}'" for c in _SPAN_BULK_COLUMNS
    )
    return (
        f"INSERT INTO spans ({cols})\n"
        f"SELECT {projection} FROM read_json(\n"
        f"    ?, format='newline_delimited', records='true',\n"
        f"    columns={{{read_cols}}},\n"
        f"    maximum_object_size={_SPAN_BULK_MAX_OBJECT_BYTES}\n"
        f") AS src\n"
        # Idempotent anti-join: skip any span_id already present so a span another
        # writer (or an earlier backfill) inserted between the caller's dedup pass
        # and this call is silently skipped rather than raising a PK conflict.
        f"WHERE NOT EXISTS (SELECT 1 FROM spans t WHERE t.span_id = src.span_id)"
    )


_BULK_SPAN_INSERT_SQL = _build_bulk_span_insert_sql()

# ---------------------------------------------------------------------------
# Columnar overlay of sub_agent_id / sub_agent_type / attributes onto
# EXISTING spans
# ---------------------------------------------------------------------------
#
# A normal (non-`--reingest`) backfill only ever INSERTs spans whose span_id
# is not yet in the store — `_BULK_SPAN_INSERT_SQL` above never touches a row
# that already exists. That is correct for the row's money/token fields (they
# must never move under an existing span), but it means anything added AFTER a
# span was first ingested can never fill on a row inserted before it existed —
# the row is never "new" again. Two independent cases share this shape:
#   - `sub_agent_type` (migration 19) is simply absent on older rows.
#   - `attributes` carries content keys (`gen_ai.prompt.content` etc.) only
#     when `[capture]` was ON at parse time — a row backfilled before the user
#     enabled it never gets the content, even though the SAME transcript can
#     supply it on a later re-parse.
# This is the set-based sibling of `_insert_session_idempotent`'s per-row
# `--reingest` UPDATE, sized for the full-history case: one vectorized
# `UPDATE ... FROM read_json` instead of a Python loop per span. Both cases
# go through this ONE primitive rather than two — see `_dedup_new_spans`'s
# `overlay_candidates` for how a span qualifies for either half.
#
# Strictly additive by construction:
#   - `COALESCE(spans.col, src.col)` for the two scalar columns only ever
#     replaces a NULL, so a value the row already carries (from any source,
#     live or backfill) can never be overwritten with a different one.
#   - `json_merge_patch(src.attributes, spans.attributes)` for the JSON
#     column — NOT the other argument order. RFC 7396 merge-patch semantics
#     make the SECOND argument win on any key both sides carry, so putting
#     the STORED attributes second means a key the row already has (e.g. from
#     live ingest, or a previous capture-content overlay) is never replaced;
#     only keys present in the freshly-parsed `src.attributes` and ABSENT
#     from the stored row get added. (Merge-patch also treats an explicit
#     JSON `null` value as "delete this key" — never a concern here, since
#     nothing in this codebase ever stores a literal null attribute value.)
# Re-running either overlay over an unchanged transcript is therefore always
# a no-op (the parsed value is stable), and this is what makes the AUTOMATIC,
# unattended catch-up loop safe to run this against on every pass.
#
# Reports the changed count via a separate COUNT query sharing the exact same
# JOIN + WHERE as the UPDATE, run first under the same write-lock hold — NOT
# `UPDATE ... RETURNING`, which crashed with an internal DuckDB fatal error
# ("Failed to append to PRIMARY_spans_0", a PRIMARY KEY constraint violation
# despite the source batch holding no duplicate span_id — reproduced on
# DuckDB 1.5.1 against the real `spans` table at ~25k rows/batch; a synthetic
# minimal table did NOT reproduce it, so it's specific to this table's shape,
# not a general RETURNING bug) the one time this ran against a real ~700k-span
# corpus. Two passes over the same batch costs one extra vectorized scan,
# negligible next to the crash it avoids.
_SUBAGENT_OVERLAY_COLUMNS: tuple[str, ...] = (
    "span_id", "sub_agent_id", "sub_agent_type", "attributes",
)
_SUBAGENT_OVERLAY_READ_TYPES: dict[str, str] = {
    "span_id": "VARCHAR", "sub_agent_id": "VARCHAR", "sub_agent_type": "VARCHAR",
    "attributes": "JSON",
}

# Match when a fill will actually HAPPEN on at least one column — a null slot
# paired with a non-null offered value for the two scalars, or a genuinely
# different (superset) attributes JSON for the content case — not merely
# "some column somewhere is null/could differ and something is offered",
# which would also match rows where no pairing does anything and inflate the
# count with no-op rows.
_SUBAGENT_OVERLAY_MATCH_PREDICATE = (
    "spans.span_id = src.span_id\n"
    "  AND (\n"
    "    (spans.sub_agent_id IS NULL AND src.sub_agent_id IS NOT NULL)\n"
    "    OR (spans.sub_agent_type IS NULL AND src.sub_agent_type IS NOT NULL)\n"
    "    OR (\n"
    "      src.attributes IS NOT NULL\n"
    "      AND json_merge_patch(src.attributes, spans.attributes) != spans.attributes\n"
    "    )\n"
    "  )"
)


def _subagent_overlay_read_json_clause() -> str:
    read_cols = ", ".join(
        f"'{c}': '{_SUBAGENT_OVERLAY_READ_TYPES[c]}'" for c in _SUBAGENT_OVERLAY_COLUMNS
    )
    return (
        f"read_json(\n"
        f"    ?, format='newline_delimited', records='true',\n"
        f"    columns={{{read_cols}}},\n"
        f"    maximum_object_size={_SPAN_BULK_MAX_OBJECT_BYTES}\n"
        f") AS src"
    )


_BULK_SUBAGENT_OVERLAY_COUNT_SQL = (
    f"SELECT COUNT(*) FROM spans, {_subagent_overlay_read_json_clause()}\n"
    f"WHERE {_SUBAGENT_OVERLAY_MATCH_PREDICATE}"
)

_BULK_SUBAGENT_OVERLAY_UPDATE_SQL = (
    "UPDATE spans SET "
    "sub_agent_id = COALESCE(spans.sub_agent_id, src.sub_agent_id), "
    "sub_agent_type = COALESCE(spans.sub_agent_type, src.sub_agent_type), "
    "attributes = CASE WHEN src.attributes IS NOT NULL "
    "THEN json_merge_patch(src.attributes, spans.attributes) "
    "ELSE spans.attributes END\n"
    f"FROM {_subagent_overlay_read_json_clause()}\n"
    f"WHERE {_SUBAGENT_OVERLAY_MATCH_PREDICATE}"
)


def _span_to_json_obj(span: NormalizedSpan) -> dict:
    """Serialize a span to the JSON object `read_json` expects — one key per
    bulk column. Enums are unwrapped to their `.value`; datetimes to ISO-8601
    (tz-aware, so the offset survives); `attributes`/`events`/`request_params`/
    `request_tools` stay NESTED (objects/arrays, never pre-stringified) so DuckDB
    stores them as JSON values identical to what `insert_span` binds.
    """
    return {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "session_id": span.session_id,
        "agent_id": span.agent_id,
        "name": span.name,
        "kind": span.kind.value,
        "status_code": span.status_code.value,
        "status_message": span.status_message,
        "start_time": span.start_time.isoformat() if span.start_time else None,
        "end_time": span.end_time.isoformat() if span.end_time else None,
        "duration_ms": span.duration_ms,
        "attributes": span.attributes,
        "provider": span.provider,
        "model": span.model,
        "tool_name": span.tool_name,
        "input_tokens": span.input_tokens,
        "output_tokens": span.output_tokens,
        "cache_tokens": span.cache_tokens,
        "cost_usd": span.cost_usd,
        "request_type": span.request_type,
        "conversation_id": span.conversation_id,
        "events": span.events,
        "billing_account": span.billing_account,
        "cache_write_tokens": span.cache_write_tokens,
        "request_params": span.request_params,
        "request_tools": span.request_tools,
        "sub_agent_id": span.sub_agent_id,
        "tenant_id": span.tenant_id,
        "feature": span.feature,
        "environment": span.environment,
        "service_version": span.service_version,
        "commit_sha": span.commit_sha,
        "prompt_template_id": span.prompt_template_id,
        "prompt_template_version": span.prompt_template_version,
        "pricing_source": span.pricing_source,
        "sub_agent_type": span.sub_agent_type,
    }


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

CREATE INDEX IF NOT EXISTS idx_alerts_agent_id    ON alerts(agent_id);
CREATE INDEX IF NOT EXISTS idx_alerts_fired_at    ON alerts(fired_at);
"""
    + SESSIONS_INDEX_SQL
)

# The retention ledger's DDL, single-sourced so migration 20 and the
# `EXPECTED_TABLES` self-heal below create the same table.
RETENTION_EVENTS_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS retention_events (\n"
    "    event_id            TEXT PRIMARY KEY,\n"
    "    ran_at              TIMESTAMPTZ NOT NULL,\n"
    "    cutoff              TIMESTAMPTZ NOT NULL,\n"
    "    retention_days      INTEGER,\n"
    "    analysis_span_days  INTEGER,\n"
    "    spans_deleted       BIGINT NOT NULL DEFAULT 0,\n"
    "    sessions_deleted    BIGINT NOT NULL DEFAULT 0,\n"
    "    oldest_kept         TIMESTAMPTZ\n"
    ")"
)

# The ingested agent-config surface, single-sourced so migration 22 and the
# `EXPECTED_TABLES` self-heal create the same table. See `core/agent_config.py`
# for what each column answers and why the measurement columns are separate
# from the size ones: `tokens` is what the file's own text costs, while
# `measured_tokens` is what an MCP server's tool schemas were MEASURED to inject
# and is NULL until something actually measured them. A NULL there is load-
# bearing — no consumer may substitute a default for it, because "we have not
# measured this server" and "this server injects nothing" are different answers.
AGENT_CONFIG_FILES_TABLE_SQL = (
    "CREATE TABLE IF NOT EXISTS agent_config_files (\n"
    "    config_id       TEXT PRIMARY KEY,\n"
    "    kind            TEXT NOT NULL,\n"
    "    scope           TEXT NOT NULL,\n"
    "    root            TEXT,\n"
    "    name            TEXT,\n"
    "    path            TEXT NOT NULL,\n"
    "    size_bytes      BIGINT NOT NULL DEFAULT 0,\n"
    "    tokens          BIGINT NOT NULL DEFAULT 0,\n"
    "    content_hash    TEXT,\n"
    "    last_seen       TIMESTAMPTZ NOT NULL,\n"
    "    subkind         TEXT,\n"
    "    detail          JSON,\n"
    "    measured_tokens BIGINT,\n"
    "    measured_at     TIMESTAMPTZ,\n"
    "    measure_status  TEXT,\n"
    "    seq             BIGINT NOT NULL DEFAULT 0\n"
    ")"
)

# The table's secondary indexes, single-sourced beside the DDL for the same
# reason `SPANS_INDEX_SQL` is: a fresh install and the `EXPECTED_TABLES`
# self-heal must create the SAME set, or a database quietly ends up missing an
# index that nothing will ever put back (migrations are already recorded
# applied). `repair_explicit_indexes` does not read this constant -- it
# re-issues each index from its own catalogue DDL, so it repairs indexes this
# module has never heard of.
AGENT_CONFIG_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_agent_config_kind "
    "ON agent_config_files(kind);\n"
    "CREATE INDEX IF NOT EXISTS idx_agent_config_last_seen "
    "ON agent_config_files(last_seen)"
)


MIGRATIONS: list[tuple[int, str]] = [
    (1, INITIAL_SCHEMA_SQL),
    (2, (
        "DROP INDEX IF EXISTS idx_spans_trace_id;\n"
        "DROP INDEX IF EXISTS idx_spans_agent_id;\n"
        "DROP INDEX IF EXISTS idx_spans_start_time;\n"
        "DROP INDEX IF EXISTS idx_spans_tool_name;\n"
        "DROP INDEX IF EXISTS idx_spans_conv_id;\n"
        "DROP INDEX IF EXISTS idx_spans_session_id"
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
    # Migration 16: close-the-loop tables (#53) — local-first annotations,
    # expectations, and a fix-history ledger. This is the capture half's missing
    # other half: going from "something weird happened" to "this is now a
    # repeatable check", entirely offline (no eval-platform / cloud dependency).
    # See core/loop.py + docs/internal/close-the-loop.md for the product bet.
    #
    #  * `run_annotations` — human note + verdict AFTER the fact on a run
    #    (session). MANY rows per session (an append-only log, not an upsert like
    #    `session_labels` which is a single display-name rename); `annotation_id`
    #    PK. `verdict` is one of good/bad/mixed/unknown (nullable = note only).
    #  * `expectations` — a labeled run promoted into a stored expectation/case.
    #    `origin_session_id` is the run it was promoted FROM (nullable — an
    #    expectation can be authored free-standing); `agent_id` scopes it.
    #  * `expectation_runs` — the fix-history ledger keyed to an expectation: one
    #    row per rerun recorded against it, `outcome` in pass/regress/unknown, so
    #    a user sees whether a change fixed or regressed the case over time.
    #
    # All additive (backward-compatible): old code ignores the new tables.
    (16, (
        "CREATE TABLE IF NOT EXISTS run_annotations (\n"
        "    annotation_id TEXT PRIMARY KEY,\n"
        "    session_id    TEXT NOT NULL,\n"
        "    verdict       TEXT,\n"
        "    note          TEXT,\n"
        "    created_at    TIMESTAMPTZ NOT NULL\n"
        ");\n"
        "CREATE INDEX IF NOT EXISTS idx_run_annotations_session "
        "ON run_annotations(session_id);\n"
        "CREATE TABLE IF NOT EXISTS expectations (\n"
        "    expectation_id    TEXT PRIMARY KEY,\n"
        "    origin_session_id TEXT,\n"
        "    agent_id          TEXT,\n"
        "    name              TEXT NOT NULL,\n"
        "    description       TEXT,\n"
        "    created_at        TIMESTAMPTZ NOT NULL\n"
        ");\n"
        "CREATE INDEX IF NOT EXISTS idx_expectations_origin "
        "ON expectations(origin_session_id);\n"
        "CREATE TABLE IF NOT EXISTS expectation_runs (\n"
        "    run_ledger_id  TEXT PRIMARY KEY,\n"
        "    expectation_id TEXT NOT NULL,\n"
        "    session_id     TEXT,\n"
        "    outcome        TEXT NOT NULL,\n"
        "    note           TEXT,\n"
        "    created_at     TIMESTAMPTZ NOT NULL\n"
        ");\n"
        "CREATE INDEX IF NOT EXISTS idx_expectation_runs_exp "
        "ON expectation_runs(expectation_id)"
    )),
    # Migration 17: SDK cost-attribution dimensions on spans — the multi-tenant
    # cost breakdown (which CUSTOMER/TENANT, FEATURE, ENVIRONMENT, and PROMPT
    # VERSION is spending the money). All nullable; existing spans and every
    # producer that doesn't set them are unaffected. tenant_id/feature are
    # tj-specific extensions (no OTel convention exists for a billing tenant or
    # an app "feature" label); environment/service_version/commit_sha carry
    # standard OTel semantic-convention values (deployment.environment.name /
    # service.version / vcs.ref.head.revision — see otel/semconv.py). See
    # core/models.py::NormalizedSpan for the full field-by-field rationale.
    (17, (
        "ALTER TABLE spans ADD COLUMN IF NOT EXISTS tenant_id                TEXT;\n"
        "ALTER TABLE spans ADD COLUMN IF NOT EXISTS feature                  TEXT;\n"
        "ALTER TABLE spans ADD COLUMN IF NOT EXISTS environment              TEXT;\n"
        "ALTER TABLE spans ADD COLUMN IF NOT EXISTS service_version          TEXT;\n"
        "ALTER TABLE spans ADD COLUMN IF NOT EXISTS commit_sha               TEXT;\n"
        "ALTER TABLE spans ADD COLUMN IF NOT EXISTS prompt_template_id       TEXT;\n"
        "ALTER TABLE spans ADD COLUMN IF NOT EXISTS prompt_template_version  TEXT"
    )),
    # Migration 18: pricing_source on spans — provenance for cost_usd (HOW the
    # rate resolved: exact / date_stripped / context_tag / override /
    # default_fallback — see pricing.classify_pricing_source). Nullable;
    # existing spans stay NULL on upgrade (their provenance was never
    # recorded and can't be reconstructed after the fact). Populated going
    # forward by CostEngine.process_span at ingest. Root-caused by an unpriced
    # model (no models.toml row) silently pricing its cache tokens at zero via
    # calculate_cost's fallback — the fallback figure and a real rate were
    # otherwise indistinguishable once only cost_usd remained.
    (18, "ALTER TABLE spans ADD COLUMN IF NOT EXISTS pricing_source TEXT"),
    # Migration 19: sub_agent_type on spans — the STABLE identity of a Claude
    # Code subagent dispatch, alongside the per-dispatch `sub_agent_id` from
    # migration 14. `sub_agent_id` is Claude Code's `agentId`, which is minted
    # fresh per Task dispatch, so each value belongs to exactly ONE session by
    # construction and no per-subagent cohort can ever be formed from it —
    # leaving a substantial share of spans (all subagent work) unclusterable.
    # This column carries the dispatched agent TYPE instead (the
    # spawning Task/Agent call's `subagent_type` argument), which recurs across
    # sessions and is the name that resolves to a `.claude/agents/<name>.md`
    # definition file. Nullable; NULL for main-thread spans, non-Claude-Code
    # telemetry, and dispatches whose type is a per-dispatch instance label
    # rather than a reusable definition (see backfill._subagent_type_for).
    # Populated by the backfill parser from the `agent-<id>.meta.json` sidecar;
    # `tj backfill --reingest` re-tags pre-column history.
    (19, "ALTER TABLE spans ADD COLUMN IF NOT EXISTS sub_agent_type TEXT"),
    # Migration 20: the retention ledger. Deleting a user's own history and
    # leaving no account of it is the fault this closes — the only way to learn
    # that eight weeks of the oldest history had gone was to measure the store
    # twice, days apart, and diff the two answers. One row per run of the
    # retention job, written in the same transaction as the delete, so a
    # deletion that happened always has a record and a record that exists always
    # describes a deletion. `tj doctor` reads it; nothing else writes it.
    (20, RETENTION_EVENTS_TABLE_SQL),
    # Migration 21: session provenance + task identity.
    #
    # `source` records what PRODUCED a session — 'claude-code' (matches
    # `agent_kind.CODING_AGENT_GROUPS`'s spelling exactly) / 'codex' / 'sdk' —
    # at the point of ingest, from the strongest signal available at
    # that ingest path (a literal constant for the two dedicated backfill
    # adapters, which parse ONLY that tool's transcripts; `agent_kind
    # .classify_agent_kind` for the live path, which sees a mix — its exact/
    # prefix rules are grounded in verified id-minting behavior, not a guess).
    # Before this, nothing recorded provenance at write time at all; every
    # reader re-derived a "coding vs SDK" answer from `agent_id` naming
    # conventions, and two independently-maintained predicates
    # (`core.agent_kind` vs `core.alerts.is_interactive_coding_agent`)
    # disagree BY DESIGN on Codex prefix-vs-exact matching (see
    # `agent_kind`'s module docstring). This column does NOT merge them —
    # both predicates, and their five existing call sites plus the pinned
    # margin-case test, are UNCHANGED; this is a new, more precise field a
    # caller can additionally choose to read.
    #
    # `task_statement_hash` / `dominant_model` support "did this session
    # repeat prior work" without storing raw prompt text: the only
    # high-confidence "same task" signal is the first user prompt, which
    # lived only in the on-disk transcript (rotated ~30 days by Claude Code)
    # and was discarded at ingest — unrecoverable once gone. Masked (hashed),
    # never the raw prompt. `dominant_model` records the model that actually
    # ran the bulk of the session, alongside the hash, for the same
    # correlate-across-sessions use case (`core.optimize.repeat_task`).
    (21,
     "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS source TEXT;"
     "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS task_statement_hash TEXT;"
     "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS dominant_model TEXT"),
    # Migration 22: the ingested agent-config surface.
    #
    # Three analyzers each walked the filesystem themselves at analysis time to
    # answer "what config does this user have" — `deadweight` re-read
    # `~/.claude.json` and every project's `.mcp.json`, `core/summarize/
    # candidates` re-globbed the prompt-file catalog, and `prompt_bloat` globbed
    # it a second time through its own helper. Nothing about any of it was
    # stored, so the same tree was walked three times per run and no question
    # about it could be answered without touching disk.
    #
    # This table is what those analyzers now read; the walk is only how it gets
    # populated. One row per instruction file, hook, or (MCP server, declaring
    # config file), carrying presence, size, token count, content hash and when
    # it was last seen. `measured_tokens` is the separate, independently taken
    # measurement of what an MCP server's tool schemas actually inject — cached
    # here precisely because taking it means STARTING the server, so it must
    # survive between analysis runs and be invalidated by the spec hash rather
    # than re-taken on a schedule.
    (22, AGENT_CONFIG_FILES_TABLE_SQL + ";\n" + AGENT_CONFIG_INDEX_SQL),
    # Migration 23: session-scoped span reconciliation index. The ingest-time
    # trace repair groups affected rows by session_id; keep that lookup bounded
    # on stores with a large cross-trace history. Additive and idempotent.
    (23, "CREATE INDEX IF NOT EXISTS idx_spans_session_id ON spans(session_id)"),
]


# Additive columns that later migrations append to the base tables via
# `ALTER TABLE ... ADD COLUMN`. Single-sourced here as the schema the *code*
# depends on, so `ensure_expected_columns` can reconcile a live DB to it
# regardless of what `schema_migrations` claims is applied (#55). Each entry is
# (table, column, column_def) where column_def is the SQL that follows the
# column name — it MUST be additive/backward-compatible (nullable or DEFAULTed)
# so re-applying it is always a safe no-op. Keep in sync with the ADD COLUMN
# statements in MIGRATIONS above; a column that a migration adds and the code
# reads/writes belongs here.
EXPECTED_ADDITIVE_COLUMNS: list[tuple[str, str, str]] = [
    ("spans",    "billing_account",    "TEXT"),                    # migration 4
    ("spans",    "cache_write_tokens", "BIGINT"),                  # migration 5
    ("spans",    "request_params",     "JSON"),                    # migration 7
    ("spans",    "request_tools",      "JSON"),                    # migration 7
    ("spans",    "sub_agent_id",       "TEXT"),                    # migration 14
    ("sessions", "plan_tier",          "TEXT DEFAULT 'unknown'"),  # migration 4
    ("sessions", "service_namespace",  "TEXT"),                    # migration 8
    ("sessions", "service_instance_id", "TEXT"),                   # migration 9
    ("sessions", "cache_write_tokens", "BIGINT DEFAULT 0"),        # migration 12
    ("sessions", "run_id",             "TEXT"),                    # migration 13
    ("sessions", "parent_session_id",  "TEXT"),                    # migration 13
    ("spans",    "tenant_id",               "TEXT"),               # migration 17
    ("spans",    "feature",                 "TEXT"),               # migration 17
    ("spans",    "environment",              "TEXT"),              # migration 17
    ("spans",    "service_version",         "TEXT"),               # migration 17
    ("spans",    "commit_sha",              "TEXT"),               # migration 17
    ("spans",    "prompt_template_id",      "TEXT"),               # migration 17
    ("spans",    "prompt_template_version", "TEXT"),               # migration 17
    ("spans",    "pricing_source",          "TEXT"),               # migration 18
    ("spans",    "sub_agent_type",          "TEXT"),               # migration 19
    ("sessions", "source",                  "TEXT"),               # migration 21
    ("sessions", "task_statement_hash",     "TEXT"),               # migration 21
    ("sessions", "dominant_model",          "TEXT"),               # migration 21
]


# Tables that a later `CREATE TABLE` migration adds and the code then reads/writes
# on a non-ingest path (proxy/policy audit, session labels, session-story capture,
# the close-the-loop feature). Single-sourced here as the schema the *code* depends
# on so `ensure_expected_tables` can reconcile a live DB to it regardless of what
# `schema_migrations` claims is applied (#382, follow-up to #55/#381). Each value is
# a self-contained `CREATE TABLE IF NOT EXISTS` DDL — additive/idempotent, so
# re-issuing it on every open is always a safe no-op. Keep in sync with the
# CREATE TABLE statements in MIGRATIONS above; a table a migration creates and the
# code depends on belongs here. (Indexes are optional performance hints, not a
# correctness dependency, so they are intentionally out of scope here.)
EXPECTED_TABLES: dict[str, str] = {
    # migration 6
    "policy_decisions": (
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
        ")"
    ),
    # migration 6
    "savings_ledger": (
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
        ")"
    ),
    # migration 11
    "session_labels": (
        "CREATE TABLE IF NOT EXISTS session_labels (\n"
        "    session_id  TEXT PRIMARY KEY,\n"
        "    label       TEXT NOT NULL,\n"
        "    updated_at  TIMESTAMPTZ NOT NULL\n"
        ")"
    ),
    # migration 15
    "session_story": (
        "CREATE TABLE IF NOT EXISTS session_story (\n"
        "    session_id     TEXT PRIMARY KEY,\n"
        "    story_json     JSON NOT NULL,\n"
        "    captured_at    TIMESTAMPTZ NOT NULL,\n"
        "    source         TEXT NOT NULL,\n"
        "    schema_version INTEGER NOT NULL DEFAULT 1\n"
        ")"
    ),
    # migration 16
    "run_annotations": (
        "CREATE TABLE IF NOT EXISTS run_annotations (\n"
        "    annotation_id TEXT PRIMARY KEY,\n"
        "    session_id    TEXT NOT NULL,\n"
        "    verdict       TEXT,\n"
        "    note          TEXT,\n"
        "    created_at    TIMESTAMPTZ NOT NULL\n"
        ")"
    ),
    # migration 16
    "expectations": (
        "CREATE TABLE IF NOT EXISTS expectations (\n"
        "    expectation_id    TEXT PRIMARY KEY,\n"
        "    origin_session_id TEXT,\n"
        "    agent_id          TEXT,\n"
        "    name              TEXT NOT NULL,\n"
        "    description       TEXT,\n"
        "    created_at        TIMESTAMPTZ NOT NULL\n"
        ")"
    ),
    # migration 16
    "expectation_runs": (
        "CREATE TABLE IF NOT EXISTS expectation_runs (\n"
        "    run_ledger_id  TEXT PRIMARY KEY,\n"
        "    expectation_id TEXT NOT NULL,\n"
        "    session_id     TEXT,\n"
        "    outcome        TEXT NOT NULL,\n"
        "    note           TEXT,\n"
        "    created_at     TIMESTAMPTZ NOT NULL\n"
        ")"
    ),
    # migration 20
    "retention_events": RETENTION_EVENTS_TABLE_SQL,
    # migration 22
    "agent_config_files": AGENT_CONFIG_FILES_TABLE_SQL,
}


def missing_expected_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Return the ``EXPECTED_TABLES`` absent from the live schema (#382).

    Read-only counterpart to :func:`ensure_expected_tables`; powers the
    ``tj doctor`` schema-integrity check without mutating the DB. Unlike
    :func:`missing_expected_columns` there is no "pre-migration empty DB" escape
    hatch — a table that a code path depends on is missing whether or not the base
    schema exists, and ``run_migrations`` recreates all of them idempotently, so a
    fresh DB simply reports the same set and is healed on the same open.
    """
    # Restrict to base tables in the main schema: information_schema.tables
    # also lists views, so a view sharing an expected table's name would
    # otherwise be read as "table present" and suppress the heal, leaving
    # writes to the real base table still failing.
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = 'main' AND table_type = 'BASE TABLE'"
        ).fetchall()
    }
    return [name for name in EXPECTED_TABLES if name not in existing]


def ensure_expected_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Idempotently reconcile the live schema to ``EXPECTED_TABLES`` (#382).

    ``run_migrations`` keys purely on the version INTEGER, so a version recorded
    applied under an older or renumbered definition (this repo renumbered
    migrations during a merge — see PR #306) never re-runs, and its
    ``CREATE TABLE`` silently never lands. Code that later writes to the missing
    table then raises a DuckDB error on a peripheral path (proxy/policy audit,
    session labels, session-story capture, the close-the-loop feature). This is
    the ``CREATE TABLE`` counterpart to :func:`ensure_expected_columns` (#55/#381).

    Re-issues ``CREATE TABLE IF NOT EXISTS`` for every code-depended table,
    independent of the recorded version set, so a DB with a recorded-but-unlanded
    migration self-heals on next open. Idempotent: a no-op on an already-correct DB
    (every statement is guarded by ``IF NOT EXISTS`` *and* a pre-check). Returns the
    table names it had to create (empty when healthy) so callers can log/report the
    repair. DDL comes from the hardcoded ``EXPECTED_TABLES`` constant, never user
    input, so there is no injection surface (Critical Rule 7 targets user SQL).
    """
    created: list[str] = []
    for name in missing_expected_tables(conn):
        conn.execute(EXPECTED_TABLES[name])
        created.append(name)
    return created


def missing_expected_columns(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Return the ``EXPECTED_ADDITIVE_COLUMNS`` absent from the live schema (#55).

    Read-only counterpart to :func:`ensure_expected_columns`; powers the
    ``tj doctor`` schema-integrity check without mutating the DB. Each entry is
    formatted ``table.column``. Tables that don't exist yet are skipped (their
    columns simply read as absent from ``information_schema``); a pre-migration
    empty DB therefore reports nothing rather than the whole list.
    """
    by_table: dict[str, list[str]] = {}
    for table, column, _ in EXPECTED_ADDITIVE_COLUMNS:
        by_table.setdefault(table, []).append(column)
    missing: list[str] = []
    for table, columns in by_table.items():
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = $1",
                [table],
            ).fetchall()
        }
        if not existing:
            # Table absent entirely — a fresh/pre-migration DB. run_migrations
            # creates it; nothing to reconcile here.
            continue
        missing.extend(f"{table}.{column}" for column in columns if column not in existing)
    return missing


def ensure_expected_columns(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Idempotently reconcile the live schema to ``EXPECTED_ADDITIVE_COLUMNS`` (#55).

    ``run_migrations`` keys purely on the version INTEGER, so if a version was
    recorded-applied under an older or renumbered definition (this repo
    renumbered migrations during a merge — see PR #306), the *current* SQL for
    that version never runs and its ``ADD COLUMN`` silently never lands. Every
    ingest that writes the missing column then hits a DuckDB Binder Error and is
    dropped, surfacing to the user as a blank/stale Status page.

    This closes that gap by re-issuing ``ADD COLUMN IF NOT EXISTS`` for the full
    set of additive columns the code depends on, independent of the recorded
    version set — so a DB with a recorded-but-unlanded migration self-heals on
    next open. Idempotent: a no-op on an already-correct DB (every add is guarded
    by ``IF NOT EXISTS`` *and* a pre-check). Returns the ``table.column`` list it
    had to add (empty when healthy) so callers can log/report the repair.

    Identifiers come from the hardcoded ``EXPECTED_ADDITIVE_COLUMNS`` constant,
    never user input, so the f-string DDL carries no injection surface (same
    pattern as ``repair_spans_stats``; Critical Rule 7 targets user-derived SQL).
    """
    defs = {(table, column): column_def for table, column, column_def in EXPECTED_ADDITIVE_COLUMNS}
    added: list[str] = []
    for key in missing_expected_columns(conn):
        table, column = key.split(".", 1)
        conn.execute(
            f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "{column}" {defs[(table, column)]}'
        )
        added.append(key)
    return added


# A session whose stored total differs from its spans' sum by less than this is
# treated as agreeing. Both sides are float sums over per-span figures rounded to
# 8dp, so a long session accumulates real floating-point residue; a tenth of a
# cent is far below anything a surface renders and far above that residue.
SESSION_COST_DRIFT_TOLERANCE_USD = 0.001


def session_cost_drift(
    conn: duckdb.DuckDBPyConnection,
    tolerance_usd: float = SESSION_COST_DRIFT_TOLERANCE_USD,
    limit: int = 20,
) -> tuple[int, float, list[tuple[str, float, float]]]:
    """Find sessions whose stored cost disagrees with canonical span cost.

    ``recompute_session_totals_from_spans`` uses the deduplicated logical-call
    total as the source of truth, so any gap is a stale session row — written by
    a path that moved one side without the other. Cross-source restatements are
    excluded using the same winner rule as recomputation; same-source repeats
    remain real calls. Two figures the UI can show side by side then differ,
    which is the defect: a published total that excludes rows it should include.

    Returns ``(session_count, total_abs_drift_usd, worst)`` where ``worst`` is up
    to ``limit`` ``(session_id, stored_usd, span_sum_usd)`` triples ordered by
    absolute drift, largest first.

    A NULL ``total_cost_usd`` is NOT drift when the session's spans carry no cost
    either: sessions whose spans are all tool/marker spans (or LLM calls with no
    usage attached) genuinely have nothing to price. ``COALESCE`` on both sides
    makes the comparison treat NULL and 0.0 as the same "no priced spans"
    statement, which is also how recomputation writes it.
    """
    redundant_sql = _duplicate_observation_sql("obs.span_id")
    rows = conn.execute(
        f"""
        WITH redundant AS (
            {redundant_sql}
        ), canonical AS (
            SELECT session_id, cost_usd
            FROM spans
            WHERE session_id IS NOT NULL
              AND span_id NOT IN (SELECT span_id FROM redundant)
        ),
        agg AS (
            SELECT session_id, SUM(cost_usd) AS span_cost
            FROM canonical
            GROUP BY session_id
        )
        SELECT s.session_id,
               COALESCE(s.total_cost_usd, 0.0)  AS stored,
               COALESCE(agg.span_cost, 0.0)     AS span_sum
        FROM sessions AS s
        LEFT JOIN agg ON agg.session_id = s.session_id
        WHERE ABS(COALESCE(agg.span_cost, 0.0) - COALESCE(s.total_cost_usd, 0.0)) > $1
        ORDER BY ABS(COALESCE(agg.span_cost, 0.0) - COALESCE(s.total_cost_usd, 0.0)) DESC
        """,
        [tolerance_usd],
    ).fetchall()
    total = sum(abs(float(r[2]) - float(r[1])) for r in rows)
    worst = [(str(r[0]), float(r[1]), float(r[2])) for r in rows[:limit]]
    return len(rows), total, worst


# --- Duplicate call observations --------------------------------------------
#
# One LLM call can reach the store twice: the live receive path observes it as
# it happens and a later transcript backfill observes it again, each minting its
# own span_id, so span_id-keyed idempotency never sees the overlap and every
# raw SUM prices the call twice. The two observations are recognised by their
# billed shape (accounting.call_fingerprint) and are only ever treated as one
# call when they came from DIFFERENT ingest sources — see that module for why
# a fingerprint may not collapse two rows from one observer.

#: An LLM call span: priced work, as opposed to a tool or marker span.
_LLM_SPAN_PREDICATE = "model IS NOT NULL AND tool_name IS NULL"

#: Columns making up a call's billed shape, in `call_fingerprint` order.
_FINGERPRINT_COLUMNS = (
    "session_id", "model",
    "COALESCE(input_tokens, 0)", "COALESCE(output_tokens, 0)",
    "COALESCE(cache_tokens, 0)", "COALESCE(cache_write_tokens, 0)",
)


@functools.lru_cache(maxsize=1)
def _ingest_source_sql() -> str:
    """SQL reading a row's ingest source, defaulting to the live receive path.

    Built from `accounting`'s constants so the SQL and the Python helpers can
    never name the attribute differently — neither half is user data. Resolved
    lazily because `tokenjam.core.optimize` pulls in every analyzer at import
    time, and `core.db` is on the import path of every CLI command.
    """
    from tokenjam.core.optimize import accounting
    return (
        f"COALESCE(json_extract_string(attributes, "
        f"'$.{accounting.INGEST_SOURCE_ATTRIBUTE}'), "
        f"'{accounting.LIVE_INGEST_SOURCE}')"
    )


def has_spans_from_another_source(
    conn: duckdb.DuckDBPyConnection, own_source: str,
) -> bool:
    """Could a second observer's restatement exist here at all?

    A duplicate needs two ingest sources. On a machine that has only ever
    backfilled, or only ever received live telemetry, the answer is no and
    every per-call lookup is wasted work — this asks once and lets the caller
    skip them all. Stops at the first match, so it is cheap exactly when the
    answer is yes; the full scan is paid only when there is nothing to find.
    """
    row = conn.execute(
        f"SELECT 1 FROM spans WHERE {_LLM_SPAN_PREDICATE} "
        f"AND {_ingest_source_sql()} <> $1 LIMIT 1",
        [own_source],
    ).fetchone()
    return row is not None


def stored_observations_of_call(
    conn: duckdb.DuckDBPyConnection,
    session_id: str,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_tokens: int,
    cache_write_tokens: int,
) -> dict[str, int]:
    """Per-ingest-source count of stored observations of ONE call.

    The live path's question, asked once per incoming LLM span: has another
    observer already recorded this call? Scoped to a single session and an
    exact token shape so it stays a narrow lookup rather than a scan.
    """
    if not session_id:
        return {}
    rows = conn.execute(
        f"SELECT {_ingest_source_sql()} AS src, COUNT(*) FROM spans "
        f"WHERE session_id = $1 AND model IS NOT DISTINCT FROM $2 "
        f"  AND COALESCE(input_tokens, 0) = $3 "
        f"  AND COALESCE(output_tokens, 0) = $4 "
        f"  AND COALESCE(cache_tokens, 0) = $5 "
        f"  AND COALESCE(cache_write_tokens, 0) = $6 "
        f"  AND tool_name IS NULL "
        f"GROUP BY 1",
        [session_id, model, int(input_tokens or 0), int(output_tokens or 0),
         int(cache_tokens or 0), int(cache_write_tokens or 0)],
    ).fetchall()
    return {str(r[0]): int(r[1]) for r in rows}


def stored_observations_by_call(
    conn: duckdb.DuckDBPyConnection, session_id: str,
) -> dict[str, dict[str, int]]:
    """Every stored LLM call in one session, as fingerprint -> {source: count}.

    The backfill path's question, asked once per session rather than once per
    span: which of the calls this file describes has another observer already
    recorded, and how many times?
    """
    if not session_id:
        return {}
    rows = conn.execute(
        f"SELECT session_id, model, COALESCE(input_tokens, 0), "
        f"COALESCE(output_tokens, 0), COALESCE(cache_tokens, 0), "
        f"COALESCE(cache_write_tokens, 0), {_ingest_source_sql()} AS src, COUNT(*) "
        f"FROM spans WHERE session_id = $1 AND {_LLM_SPAN_PREDICATE} "
        f"GROUP BY 1, 2, 3, 4, 5, 6, 7",
        [session_id],
    ).fetchall()
    from tokenjam.core.optimize import accounting

    by_call: dict[str, dict[str, int]] = {}
    for r in rows:
        key = accounting.call_fingerprint(*r[:6])
        by_call.setdefault(key, {})[str(r[6])] = int(r[7])
    return by_call


def unresolved_subagent_type_stats(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[int, float]:
    """Count + spend of Claude Code subagent spans with no resolved TYPE.

    `sub_agent_id IS NOT NULL` scopes to spans backfill has already tagged as
    a real Task-tool subagent dispatch (see Critical Rule 34 in CLAUDE.md);
    `sub_agent_type IS NULL` among those is either a row inserted before
    migration 19 landed (fixable — `tj backfill claude-code` re-derives it from
    the on-disk `agent-<id>.meta.json` sidecar and overlays it), a dispatch
    whose sidecar/transcript Claude Code has since pruned (unfixable — the
    source is gone), or a deliberate `_PER_DISPATCH_TASK_KINDS` carve-out
    (correct, not a gap). This count cannot distinguish the three from stored
    columns alone — see `_subagent_type_for` in `core/backfill.py`.
    """
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0.0) FROM spans "
        "WHERE sub_agent_id IS NOT NULL AND sub_agent_type IS NULL"
    ).fetchone()
    if not row:
        return 0, 0.0
    return int(row[0] or 0), float(row[1] or 0.0)


#: The backfill provenance tag every Claude Code backfill LLM/tool span
#: carries in `attributes.source` (mirrors `backfill._CLAUDE_CODE_SOURCE`,
#: not imported directly to keep `core/db.py` free of a `core/backfill.py`
#: import — the string is a stable, long-standing wire value, not an
#: implementation detail either side is free to change independently).
_BACKFILL_CLAUDE_CODE_SOURCE_TAG = "backfill.claude_code"


def missing_captured_content_stats(
    conn: duckdb.DuckDBPyConnection, *, prompts: bool, completions: bool,
) -> int:
    """Count backfill-sourced `gen_ai.llm.call` spans missing content
    `[capture]` says should be there — the linkage `bulk_overlay_span_attrs`'s
    content half exists to close: a row backfilled before the user turned
    `[capture]` on never gets the content on its own, even though a later
    `tj backfill claude-code` re-run CAN supply it from the same transcript.

    Counts a span missing `gen_ai.prompt.content` (when `prompts`) OR
    `gen_ai.completion.content` (when `completions`) — either is enough to
    flag it if both toggles are on. See `missing_captured_tool_input_stats`
    for `tool_inputs`, kept separate since it scopes to `gen_ai.tool.call`
    spans, a disjoint set from the LLM spans this counts.
    """
    if not (prompts or completions):
        return 0
    clauses = []
    if prompts:
        clauses.append("(attributes -> '$.\"gen_ai.prompt.content\"') IS NULL")
    if completions:
        clauses.append("(attributes -> '$.\"gen_ai.completion.content\"') IS NULL")
    where = " OR ".join(clauses)
    row = conn.execute(
        "SELECT COUNT(*) FROM spans "
        "WHERE name = 'gen_ai.llm.call' "
        "AND (attributes ->> '$.source') = $1 "
        f"AND ({where})",
        [_BACKFILL_CLAUDE_CODE_SOURCE_TAG],
    ).fetchone()
    return int(row[0]) if row else 0


def missing_captured_tool_input_stats(conn: duckdb.DuckDBPyConnection) -> int:
    """Count backfill-sourced tool spans missing `gen_ai.tool.input` — the
    `tool_inputs` half of `missing_captured_content_stats`, kept separate
    because it scopes to `gen_ai.tool.call` spans, not the LLM spans that
    function counts."""
    row = conn.execute(
        "SELECT COUNT(*) FROM spans "
        "WHERE name = 'gen_ai.tool.call' "
        "AND (attributes ->> '$.source') = $1 "
        "AND (attributes -> '$.\"gen_ai.tool.input\"') IS NULL",
        [_BACKFILL_CLAUDE_CODE_SOURCE_TAG],
    ).fetchone()
    return int(row[0]) if row else 0


def duplicate_call_observations(
    conn: duckdb.DuckDBPyConnection, limit: int = 20,
) -> tuple[int, float, list[tuple[str, int, float]]]:
    """Find calls a second ingest source restated, in a DB written before
    ingest-side suppression existed.

    Prevention lives at both ingest paths now, so a DB filled by a current
    build has nothing here. A DB filled by an older one carries a live and a
    backfill observation of the same call and prices it twice; this names the
    redundant rows so `tj doctor` can report them and `--repair` can drop them.

    Returns ``(span_count, redundant_cost_usd, worst)`` where ``worst`` is up to
    ``limit`` ``(session_id, span_count, redundant_cost_usd)`` triples ordered
    by redundant cost, largest first.
    """
    rows = conn.execute(
        _duplicate_observation_sql(
            "obs.session_id, COUNT(*), COALESCE(SUM(obs.cost_usd), 0.0)"
        ) + " GROUP BY obs.session_id ORDER BY 3 DESC"
    ).fetchall()
    total_spans = sum(int(r[1]) for r in rows)
    total_cost = sum(float(r[2] or 0.0) for r in rows)
    worst = [(str(r[0]), int(r[1]), float(r[2] or 0.0)) for r in rows[:limit]]
    return total_spans, total_cost, worst


def _duplicate_observation_sql(
    select_list: str, *, scope_sql: str | None = None,
) -> str:
    """Rows that are a second observer's restatement of an already-observed call.

    For each call, the number of times it really happened is the count the most
    complete observer recorded; every other source's rows for that call are
    restatements. Ties keep the live observation — it saw the request itself,
    and carries the request-side attributes a transcript never had.
    """
    from tokenjam.core.optimize import accounting

    fingerprint = ", ".join(_FINGERPRINT_COLUMNS)
    scope = f" AND ({scope_sql})" if scope_sql else ""
    return f"""
        WITH obs AS (
            SELECT span_id, session_id, cost_usd,
                   {_ingest_source_sql()} AS src,
                   MD5(CONCAT_WS('|', {fingerprint})) AS call_key
            FROM spans
            WHERE {_LLM_SPAN_PREDICATE} AND session_id IS NOT NULL{scope}
        ),
        per_source AS (
            SELECT call_key, src, COUNT(*) AS n FROM obs GROUP BY 1, 2
        ),
        contested AS (
            SELECT call_key FROM per_source GROUP BY call_key HAVING COUNT(*) > 1
        ),
        winner AS (
            SELECT call_key, src FROM per_source
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY call_key
                ORDER BY n DESC, (src = '{accounting.LIVE_INGEST_SOURCE}') DESC, src
            ) = 1
        )
        SELECT {select_list} FROM obs
        JOIN contested USING (call_key)
        JOIN winner USING (call_key)
        WHERE obs.src <> winner.src
    """


def purge_duplicate_call_observations(conn: duckdb.DuckDBPyConnection) -> tuple[int, list[str]]:
    """Delete the redundant observations `duplicate_call_observations` names.

    Returns ``(deleted_rows, touched_session_ids)`` — the caller reconciles
    those sessions' totals afterwards, since the delete moves `SUM(spans)`.
    Idempotent: a second run finds nothing left to collapse.
    """
    rows = conn.execute(
        _duplicate_observation_sql("obs.span_id, obs.session_id")
    ).fetchall()
    if not rows:
        return 0, []
    span_ids = [str(r[0]) for r in rows]
    sessions = sorted({str(r[1]) for r in rows})
    # Same ART-index workaround `reconcile_backfill_spans` documents: DuckDB
    # can raise a FATAL "Failed to delete all rows from index" — invalidating
    # the connection — when deleting indexed span rows. Drop the secondary
    # indexes, delete, recreate in a `finally` so a mid-delete error cannot
    # leave the table permanently unindexed.
    conn.execute(
        "DROP INDEX IF EXISTS idx_spans_trace_id;\n"
        "DROP INDEX IF EXISTS idx_spans_agent_id;\n"
        "DROP INDEX IF EXISTS idx_spans_start_time;\n"
        "DROP INDEX IF EXISTS idx_spans_tool_name;\n"
        "DROP INDEX IF EXISTS idx_spans_conv_id;\n"
        "DROP INDEX IF EXISTS idx_spans_session_id"
    )
    try:
        chunk = 5000
        for start in range(0, len(span_ids), chunk):
            batch = span_ids[start:start + chunk]
            placeholders = ",".join(f"${i + 1}" for i in range(len(batch)))
            conn.execute(
                f"DELETE FROM spans WHERE span_id IN ({placeholders})", batch,
            )
    finally:
        conn.execute(SPANS_INDEX_SQL)
    return len(span_ids), sessions


def run_migrations(conn: duckdb.DuckDBPyConnection) -> None:
    """Apply unapplied migrations, then reconcile the schema. Idempotent."""
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

    # Self-heal (#55): the version-keyed loop above trusts schema_migrations, so
    # a version recorded-applied under an older/renumbered definition leaves its
    # ADD COLUMN unlanded. Reconcile the additive columns the code depends on
    # regardless of recorded versions, so a mismatched DB is repaired on open
    # instead of silently dropping ingest on a Binder Error.
    healed = ensure_expected_columns(conn)
    if healed:
        logger.warning(
            "Schema self-heal: added missing column(s) %s recorded as migrated "
            "but never landed. Ingest of affected rows would otherwise fail "
            "with a DuckDB Binder Error and be silently dropped.",
            ", ".join(healed),
        )

    # Self-heal (#382): the same version-keyed trust gap leaves a recorded-but-
    # unlanded CREATE TABLE migration's table absent. Reconcile the code-depended
    # tables regardless of recorded versions, so a mismatched DB is repaired on
    # open instead of raising on a peripheral write (policy audit, session labels,
    # session-story capture, the loop feature).
    healed_tables = ensure_expected_tables(conn)
    if healed_tables:
        logger.warning(
            "Schema self-heal: created missing table(s) %s recorded as migrated "
            "but never landed. Code writing to them would otherwise raise a DuckDB "
            "error on a non-ingest path.",
            ", ".join(healed_tables),
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
        sub_agent_type=d.get("sub_agent_type"),
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
        tenant_id=d.get("tenant_id"),
        feature=d.get("feature"),
        environment=d.get("environment"),
        service_version=d.get("service_version"),
        commit_sha=d.get("commit_sha"),
        prompt_template_id=d.get("prompt_template_id"),
        prompt_template_version=d.get("prompt_template_version"),
        pricing_source=d.get("pricing_source"),
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
        source=d.get("source"),
        task_statement_hash=d.get("task_statement_hash"),
        dominant_model=d.get("dominant_model"),
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
         window_cost, window_calls, window_errors, window_tokens, last_seen}
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
            "window_tokens": 0,
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
               COUNT(*)                                      AS calls,
               COALESCE(SUM(input_tokens + output_tokens + cache_tokens + cache_write_tokens), 0)
                                                              AS tokens
        FROM spans
        WHERE start_time >= $1 AND agent_id IN ({ph})
        GROUP BY agent_id, b
        """,
        [window_start, *agent_ids],
    ).fetchall()
    for aid, b, cost, errors, calls, tokens in rows:
        r = result[aid]
        r["window_cost"] += float(cost or 0.0)
        r["window_calls"] += int(calls or 0)
        r["window_errors"] += int(errors or 0)
        r["window_tokens"] += int(tokens or 0)
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
    # DuckDB row cells arrive typed as ``object``; the columns this helper reads
    # are numeric, so the runtime value is always int-convertible.
    return int(cast(Any, val))


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


# Rows stamped with an epoch sentinel instead of an observed time. The ingest
# paths that could write one are closed (a record with no observed time is
# rejected at the boundary now), so this is a one-shot cleanup for corpora an
# older build already wrote — surfaced by `tj doctor` and removed by
# `tj doctor --repair` rather than living as a SQL snippet in a PR description.
_SENTINEL_TABLES: tuple[tuple[str, str], ...] = (
    ("spans", "start_time"),
    ("sessions", "started_at"),
)


def count_sentinel_timestamp_rows(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    """Per-table counts of rows dated before ``MIN_PLAUSIBLE_YEAR``.

    Only tables with a non-zero count appear, so an empty dict means clean. A
    missing table contributes nothing rather than failing the whole probe.
    """
    found: dict[str, int] = {}
    for table, column in _SENTINEL_TABLES:
        try:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {table} "  # noqa: S608 - table names are literals above
                f"WHERE {column} IS NOT NULL "
                f"AND EXTRACT(year FROM {column}) < $1",
                [MIN_PLAUSIBLE_YEAR],
            ).fetchone()
        except duckdb.Error:
            continue
        count = int(row[0]) if row else 0
        if count:
            found[table] = count
    return found


def purge_sentinel_timestamp_rows(
    conn: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    """Delete the sentinel-dated rows, returning what was removed per table.

    Deleted rather than corrected: there is nothing to correct TO. A row whose
    only recorded fact about time was false has no other evidence of when it
    happened, and leaving it would keep a row that every ``COUNT(*)`` counts and
    nothing can place on a calendar.
    """
    removed = count_sentinel_timestamp_rows(conn)
    for table, column in _SENTINEL_TABLES:
        if table not in removed:
            continue
        conn.execute(
            f"DELETE FROM {table} "  # noqa: S608 - table names are literals above
            f"WHERE {column} IS NOT NULL AND EXTRACT(year FROM {column}) < $1",
            [MIN_PLAUSIBLE_YEAR],
        )
    return removed


def check_spans_index_corruption(
    conn: duckdb.DuckDBPyConnection,
) -> list[tuple[str, str]]:
    """The ``spans`` secondary indexes that are missing or disagree with the table.

    Returns ``(index name, what is wrong)`` pairs; empty means all five are
    sound. ``check_spans_stats_corruption`` above asks one question about one
    column — is the row-group statistics fast-path lying about ``trace_id`` —
    which is narrower than "can this table be read and DELETED from
    predictably", and the gap between the two is where a secondary-index fault
    lives. DuckDB maintains every index inside the deleting transaction, so a
    damaged one can abort a ``DELETE`` part-way through and leave a statement
    reporting that it removed fewer rows than it matched. Retention runs exactly
    that ``DELETE``, which is what makes this load-bearing rather than cosmetic:
    a deletion that cannot be relied on to complete is a deletion whose extent
    nobody can state afterwards.

    Two independent faults, because they have different causes and the same
    remedy:

    * **absent** — the index is not in the catalogue at all. A rebuild that
      copies data without the DDL drops all five permanently (the ``CREATE
      TABLE … AS SELECT`` trap ``repair_spans_stats`` documents), and migrations
      are already recorded applied, so nothing puts them back.
    * **inconsistent** — the index answers a point lookup with fewer rows than
      the table holds. Probed the same way the stats check probes: take a value
      known to be present, ask for it once in a form an index can serve and once
      in a form it cannot.

      **The unindexable form must be ``CAST(col AS VARCHAR) || ''``, not a bare
      ``CAST``.** Four of the five columns here are already ``VARCHAR``, so
      casting them is a no-op the planner discards — the index then serves BOTH
      sides and the probe compares a damaged index against itself, reporting
      sound whatever the damage. Only ``start_time`` was ever really being
      tested. Concatenating an empty string is a real expression over the
      column for every type, so it always forces the scan. Demonstrated on a
      damaged index elsewhere in this schema: the equality form answered 7
      where the table held 94, and the bare-``CAST`` form also answered 7.

    An empty table, an unreadable column, or a column holding only NULLs
    contributes nothing: this reports what it can demonstrate, never suspicion.
    """
    try:
        # A pre-migration database has no spans table, so it has no indexes to
        # be missing. Reporting five absent ones there would flag every fresh
        # install as corrupt.
        conn.execute("SELECT 1 FROM spans LIMIT 0").fetchall()
        present = {
            str(row[0])
            for row in conn.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'spans'"
            ).fetchall()
        }
    except duckdb.Error:
        return []

    faults: list[tuple[str, str]] = []
    for index_name, column in SPANS_INDEXES:
        if index_name not in present:
            faults.append((index_name, "absent from the catalogue"))
            continue
        try:
            sample = conn.execute(
                f"SELECT {column} FROM spans WHERE {column} IS NOT NULL LIMIT 3"
            ).fetchall()
        except duckdb.Error:
            continue
        for (value,) in sample:
            try:
                indexed_row = conn.execute(
                    f"SELECT COUNT(*) FROM spans WHERE {column} = $1", [value]
                ).fetchone()
                scanned_row = conn.execute(
                    f"SELECT COUNT(*) FROM spans "
                    f"WHERE CAST({column} AS VARCHAR) || '' = CAST($1 AS VARCHAR)",
                    [value],
                ).fetchone()
            except duckdb.Error:
                break
            indexed = indexed_row[0] if indexed_row else 0
            scanned = scanned_row[0] if scanned_row else 0
            if indexed < scanned:
                faults.append((
                    index_name,
                    f"returns {indexed} of {scanned} matching row(s)",
                ))
                break
    return faults


def repair_spans_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop and recreate every ``spans`` secondary index from the canonical DDL.

    Cheaper than ``repair_spans_stats`` and sufficient for the index fault: the
    table's rows are the source of truth and are never touched, so a rebuilt
    index can only agree with them. Idempotent, and safe on a healthy database.
    """
    for index_name, _ in SPANS_INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {index_name}")
    for statement in SPANS_INDEX_SQL.split(";"):
        statement = statement.strip()
        if statement:
            conn.execute(statement)
    conn.execute("CHECKPOINT")


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


#: An index name / column name we are willing to interpolate into SQL. These
#: come from DuckDB's own catalogue rather than from a user, but a probe that
#: builds SQL by string-splicing validates its identifiers anyway.
_SQL_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _index_columns(expressions: str) -> list[str]:
    """Column names out of ``duckdb_indexes().expressions``.

    That column is a VARCHAR rendering of the indexed expression list, e.g.
    ``'[kind]'`` or ``'[agent_id, started_at]'`` — not a real list, so it is
    parsed rather than unnested.
    """
    inner = (expressions or "").strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    return [part.strip() for part in inner.split(",") if part.strip()]


def explicit_indexes(
    conn: duckdb.DuckDBPyConnection,
) -> list[tuple[str, str, list[str], str]]:
    """Every explicit index in the database: ``(name, table, columns, ddl)``.

    Read from ``duckdb_indexes()``, which lists only indexes created by an
    explicit ``CREATE INDEX``. A ``PRIMARY KEY``'s own ART is NOT in there, so
    a repair driven by this list structurally cannot touch a primary key —
    which is the property that makes the sweep safe to run unattended.
    ``is_primary``/``is_unique`` are filtered as well, belt and braces.
    """
    try:
        rows = conn.execute(
            "SELECT index_name, table_name, expressions, sql FROM duckdb_indexes() "
            "WHERE NOT is_primary AND NOT is_unique ORDER BY table_name, index_name"
        ).fetchall()
    except duckdb.Error:
        return []
    out = []
    for name, table, expressions, ddl in rows:
        if not _SQL_IDENTIFIER.match(str(name)) or not _SQL_IDENTIFIER.match(str(table)):
            continue
        out.append((str(name), str(table), _index_columns(expressions), str(ddl or "")))
    return out


#: Values to compare per index when the table is small enough to enumerate its
#: value space. Above `_PROBE_EXHAUSTIVE_MAX_ROWS` the probe falls back to a
#: few samples, because the scan side of each comparison is a full table scan.
_PROBE_VALUE_LIMIT = 200
_PROBE_EXHAUSTIVE_MAX_ROWS = 50_000
_PROBE_SAMPLE_VALUES = 3


def check_index_divergence(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Sweep EVERY explicit index for disagreement with its table.

    Returns ``(faults, unproven)``, each a list of ``(index, table, reason)``.

    **A clean `faults` list is NOT a proof of soundness, and `unproven` is how
    that is said out loud.** This probe compares counts for particular VALUES,
    so it can only find damage in the entries it looked at. Learned the hard
    way: on a genuinely damaged database a three-value sample found three of
    four damaged indexes, the fourth reported clean, and a repair driven by
    that verdict left the table still raising the fatal. So an index is only
    reported sound when every distinct value was compared; anything less lands
    in ``unproven`` with the reason, and callers that must be CORRECT rather
    than cheap repair everything instead of trusting this (see
    ``repair_explicit_indexes`` and ``recover_invalidated_database``).

    Coverage is exhaustive for a table small enough to enumerate and sampled
    above that, because the scan side of each comparison is a full table scan
    and the cost is per distinct value.

    **Why the scan side multiplies by an empty string.** The probe asks the
    same question in a form an index can serve and one it cannot. Picking the
    second form is the whole difficulty: ``CAST(col AS VARCHAR)`` is a NO-OP on
    a column already stored as ``VARCHAR``, so the planner discards it and the
    index ends up serving BOTH sides — the probe then compares a damaged index
    against itself and reports it sound whatever the damage. Concatenating an
    empty string is a real expression over the column for every type. Measured
    on a genuinely damaged index: the equality form answered 7 where the table
    held 94, the bare-``CAST`` form also answered 7, the concatenated form 94.
    """
    faults: list[tuple[str, str, str]] = []
    unproven: list[tuple[str, str, str]] = []
    row_counts: dict[str, int] = {}
    for index_name, table, columns, _ddl in explicit_indexes(conn):
        if len(columns) != 1 or not _SQL_IDENTIFIER.match(columns[0]):
            unproven.append((
                index_name, table,
                "indexed on an expression this single-column comparison "
                "cannot test",
            ))
            continue
        column = columns[0]
        try:
            if table not in row_counts:
                count_row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                row_counts[table] = int(count_row[0]) if count_row else 0
            small = row_counts[table] <= _PROBE_EXHAUSTIVE_MAX_ROWS
            limit = _PROBE_VALUE_LIMIT if small else _PROBE_SAMPLE_VALUES
            values = conn.execute(
                f"SELECT DISTINCT {column} FROM {table} "
                f"WHERE {column} IS NOT NULL LIMIT {limit + 1}"
            ).fetchall()
        except duckdb.Error as exc:
            unproven.append((index_name, table, f"could not be probed: {exc}"))
            continue
        if not values:
            continue  # empty table or all-NULL column: nothing to demonstrate
        complete = small and len(values) <= limit
        diverged = False
        for (value,) in values[:limit]:
            try:
                indexed_row = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {column} = $1", [value]
                ).fetchone()
                scanned_row = conn.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE CAST({column} AS VARCHAR) || '' = CAST($1 AS VARCHAR)",
                    [value],
                ).fetchone()
            except duckdb.Error as exc:
                unproven.append((index_name, table, f"could not be probed: {exc}"))
                diverged = True  # not a fault, but not proven either
                break
            indexed = indexed_row[0] if indexed_row else 0
            scanned = scanned_row[0] if scanned_row else 0
            if indexed < scanned:
                faults.append((
                    index_name, table,
                    f"returns {indexed} of {scanned} matching row(s)",
                ))
                diverged = True
                break
        if not diverged and not complete:
            unproven.append((
                index_name, table,
                f"only {min(len(values), limit)} of the table's distinct values "
                f"were compared, so a clean result here is not proof",
            ))
    return faults, unproven


def repair_explicit_indexes(
    conn: duckdb.DuckDBPyConnection,
    index_names: "Sequence[str] | None" = None,
) -> list[str]:
    """Drop and recreate explicit indexes from their own catalogue DDL.

    ``index_names`` limits the repair to those indexes; ``None`` repairs every
    explicit index. Returns the names actually rebuilt. Idempotent, and safe on
    a healthy database.

    **What it cannot do, structurally.** The DDL is read back from
    ``duckdb_indexes()``, which does not list a ``PRIMARY KEY``'s ART, so no
    primary key or unique constraint is reachable from here. No rows are read,
    written or moved and no table is rebuilt: the table's rows are the source
    of truth, so a re-issued index can only agree with them. That is why this
    is safe unattended where `repair_spans_stats`'s table rebuild would not be.

    An index whose catalogue entry carries no DDL is left alone rather than
    dropped — dropping without being able to re-create it would turn a damaged
    index into a missing one, and migrations are already recorded applied, so
    nothing else would put it back.
    """
    wanted = set(index_names) if index_names is not None else None
    rebuilt: list[str] = []
    for index_name, _table, _columns, ddl in explicit_indexes(conn):
        if wanted is not None and index_name not in wanted:
            continue
        if not ddl.strip():
            logger.warning(
                "index %s has no DDL in the catalogue; leaving it in place rather "
                "than dropping an index that could not be recreated", index_name,
            )
            continue
        conn.execute(f"DROP INDEX IF EXISTS {index_name}")
        for statement in ddl.split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(statement)
        rebuilt.append(index_name)
    if rebuilt:
        conn.execute("CHECKPOINT")
    return rebuilt


# ---------------------------------------------------------------------------
# Fatal errors and database invalidation
# ---------------------------------------------------------------------------
#
# A DuckDB `FatalException` is categorically different from every other error
# this module handles, and the difference is not visible at the call site that
# raises it. Verified against duckdb 1.5.5, holding one root connection, one
# sibling cursor and one connection opened afterwards:
#
#   * The exception invalidates the whole DATABASE INSTANCE, not the connection
#     that raised it. Every other cursor over that database starts raising
#     `FATAL Error: Failed: database has been invalidated because of a previous
#     fatal error. The database must be restarted prior to being used again.`
#   * `duckdb.connect(same_path)` AFTERWARDS hands back the SAME dead instance
#     — DuckDB caches instances per path within a process, so "just reconnect"
#     is not a recovery. This is why a fatal raised on a background scan's own
#     `DuckDBBackend` takes down the web server's unrelated connections too.
#   * Closing EVERY connection to that path evicts the instance from that
#     cache; the next `duckdb.connect` then opens a healthy database, in the
#     same process, with no restart. That is the only in-process recovery, and
#     it is why recovery has to be a property of the process rather than of one
#     backend object — hence the registry below.
#
# The consequence for error handling: any `except Exception` that treats a
# failure as skip-this-row-and-continue MUST re-check for a fatal first. After
# a fatal there are no more rows to skip, only queries that will all fail, and
# a handler that logs a per-record warning turns a hard stop into a process
# that keeps serving traffic on a database it can no longer read.

#: Text DuckDB uses for every post-fatal query on an invalidated instance.
DATABASE_INVALIDATED_MESSAGE = "database has been invalidated"


def is_fatal_db_error(exc: BaseException) -> bool:
    """True when ``exc`` means the database instance is gone, not this row.

    Matches on the exception TYPE and, as a backstop, on the invalidation text:
    the type is authoritative, but the message check keeps the classification
    correct if a fatal reaches us wrapped by an intermediate layer.
    """
    fatal_type = getattr(duckdb, "FatalException", None)
    if fatal_type is not None and isinstance(exc, fatal_type):
        return True
    return DATABASE_INVALIDATED_MESSAGE in str(exc)


# Every live `DuckDBBackend`, so recovery can close all of a path's connections
# — the necessary condition for DuckDB to evict the invalidated instance.
# Weak, so a backend that goes out of scope is not kept alive by being here.
_LIVE_BACKENDS: "weakref.WeakSet[DuckDBBackend]" = weakref.WeakSet()
_FATAL_LOCK = threading.RLock()
#: Set when a fatal is observed anywhere in this process; cleared by a
#: successful recovery. Process-wide because the invalidation is.
_FATAL_DB_ERROR: str | None = None


def note_fatal_db_error(exc: BaseException) -> None:
    """Record that a fatal happened, so surfaces stop claiming to be healthy."""
    global _FATAL_DB_ERROR
    with _FATAL_LOCK:
        if _FATAL_DB_ERROR is None:
            _FATAL_DB_ERROR = f"{type(exc).__name__}: {exc}".split("\n")[0]
    logger.error(
        "database instance invalidated by a fatal DuckDB error (%s: %s); every "
        "connection in this process is now dead until it is re-established",
        type(exc).__name__, str(exc).split("\n")[0],
    )


def fatal_db_error() -> str | None:
    """The recorded fatal, or None. Cheap; safe to call from a request path."""
    with _FATAL_LOCK:
        return _FATAL_DB_ERROR


def clear_fatal_db_error() -> None:
    global _FATAL_DB_ERROR
    with _FATAL_LOCK:
        _FATAL_DB_ERROR = None


def handle_if_fatal(exc: BaseException, *, what: str) -> bool:
    """Whether ``exc`` was fatal; if so, record it and re-establish connections.

    The hook for every broad `except Exception` that logs a failure and carries
    on. Those handlers are correct for the errors they were written for and
    catastrophic for this one, and the difference is invisible at the catch
    site — which is how a fatal ends up swallowed by a `pass` on a background
    thread while the request path quietly dies. Ask this first:

        except Exception as exc:
            if not handle_if_fatal(exc, what="the job"):
                logger.warning("the job failed", exc_info=True)

    Returns True when it handled a fatal (already logged, recovery attempted),
    so the caller's ordinary logging is skipped rather than duplicated.
    """
    if not is_fatal_db_error(exc):
        return False
    note_fatal_db_error(exc)
    if recover_invalidated_database():
        logger.warning("%s: database connections re-established", what)
    else:
        logger.error(
            "%s: the database could not be re-established; this process can no "
            "longer read it. /health reports unhealthy until `tj serve` is "
            "restarted and `tj doctor --repair` has run.", what,
        )
    return True


def recover_if_fatal_noted(*, what: str) -> bool:
    """Recover if a fatal was recorded anywhere in this process, however it was
    caught. Returns whether a recovery ran.

    **The swallow-proof backstop, and the reason it has to exist.**
    `handle_if_fatal` only fires when the exception REACHES the handler that
    calls it, and in this codebase a fatal from an analyzer's database write
    crosses several broad `except Exception` handlers on its way out — the
    per-analyzer one that records a failure and continues with the rest, and
    the store one that keeps a pass alive. Any of them can absorb it, and
    adding the classification to each is a game nobody wins: the next handler
    someone writes reopens the hole silently.

    Exception propagation is therefore the wrong channel. `note_fatal_db_error`
    is called at the point the fatal is RECOGNISED, before it is re-raised, so
    the process-wide record survives every handler that swallows the exception
    itself. Call this from the `finally` of any long-running job and the
    recovery happens whether or not the exception ever escaped.
    """
    if fatal_db_error() is None:
        return False
    logger.error(
        "%s: a fatal DuckDB error was recorded during this pass; the exception "
        "may have been absorbed by an intermediate handler. Recovering.", what,
    )
    if recover_invalidated_database():
        logger.warning("%s: database connections re-established", what)
    else:
        logger.error(
            "%s: the database could not be re-established; /health reports "
            "unhealthy until `tj serve` is restarted.", what,
        )
    return True


def recover_invalidated_database(*, repair: bool = True) -> bool:
    """Re-establish every connection in this process; returns whether it worked.

    Closes all registered backends' connections FIRST and only then reconnects
    them, because a single surviving connection pins the invalidated instance
    in DuckDB's per-path cache and every reconnect would hand back that same
    dead instance (see the note above). In-memory backends are skipped: their
    database IS their connection, so closing it discards the data, and there is
    nothing on disk to reopen.

    With ``repair``, rebuilds EVERY explicit index on the way back up.

    **All of them, not just the ones the probe flags, and that is deliberate.**
    The exception does not name the index that raised it, and
    ``check_index_divergence`` compares particular values, so a clean verdict
    from it is not a proof of soundness — measured on a real damaged database,
    a sampled sweep found three of four damaged indexes and a repair driven by
    that verdict left the table still raising the fatal on the next write.
    Recovering into the same fatal is the one outcome this path must not have,
    so it rebuilds the lot. That is affordable because the rebuild reads no
    rows and moves no data: measured at ~1.1s for all fourteen indexes of a
    3.9GB / 736k-row database, against a fault that otherwise 500s every route.
    """
    with _FATAL_LOCK:
        backends = [b for b in _LIVE_BACKENDS if b.recoverable]
        ok = True
        rebuilt: list[str] = []
        # Hold every backend's connection lock across teardown AND reopen, so
        # no thread can be handed a cursor from the window in between. Without
        # this, `conn` on another thread sees the bumped generation, calls
        # `.cursor()` on the already-closed root connection, and either raises
        # into a 500 or drops a write -- which would make the recovery path
        # itself do the thing this whole change exists to prevent. `conn`
        # blocks for the duration instead, which is the right trade: recovery
        # is sub-second and the alternative is serving a dead handle.
        with ExitStack() as locks:
            for backend in backends:
                locks.enter_context(backend._conn_lock)
            for backend in backends:
                backend._teardown_connections()
            for backend in backends:
                if not backend._reopen():
                    ok = False
            if ok and repair:
                # One rebuild per database, not per backend: every registered
                # backend on this path shares the instance we just reopened.
                # Still under the locks: a half-repaired index set is no safer
                # to hand out than a closed connection.
                for backend in backends:
                    try:
                        faults, _unproven = check_index_divergence(backend.conn)
                        if faults:
                            logger.error(
                                "index damage found while recovering: %s",
                                "; ".join(f"{n} on {t} {r}" for n, t, r in faults),
                            )
                        rebuilt = repair_explicit_indexes(backend.conn)
                    except duckdb.Error as exc:
                        logger.error("index repair failed after recovery: %s", exc)
                        ok = False
                    break
        if ok:
            clear_fatal_db_error()
            logger.warning(
                "database connections re-established after a fatal DuckDB error; "
                "rebuilt %d index(es) so the fault does not recur", len(rebuilt),
            )
        return ok


# ---------------------------------------------------------------------------
# DuckDBBackend
# ---------------------------------------------------------------------------

# The two totals policies `upsert_session` chooses between. See its docstring:
# REPLACE is for a caller whose record describes a session's whole life so far
# (the live path, which accumulates in Python); ACCUMULATE is for a caller whose
# record describes only what THIS write added (the per-file backfill).
_SESSION_TOTALS_REPLACE = """
                    total_cost_usd = EXCLUDED.total_cost_usd,
                    input_tokens = EXCLUDED.input_tokens,
                    output_tokens = EXCLUDED.output_tokens,
                    cache_tokens = EXCLUDED.cache_tokens,
                    cache_write_tokens = EXCLUDED.cache_write_tokens,
                    tool_call_count = EXCLUDED.tool_call_count,
                    error_count = EXCLUDED.error_count,
"""

_SESSION_TOTALS_ACCUMULATE = """
                    total_cost_usd = COALESCE(sessions.total_cost_usd, 0.0)
                                   + COALESCE(EXCLUDED.total_cost_usd, 0.0),
                    input_tokens = COALESCE(sessions.input_tokens, 0)
                                 + COALESCE(EXCLUDED.input_tokens, 0),
                    output_tokens = COALESCE(sessions.output_tokens, 0)
                                  + COALESCE(EXCLUDED.output_tokens, 0),
                    cache_tokens = COALESCE(sessions.cache_tokens, 0)
                                 + COALESCE(EXCLUDED.cache_tokens, 0),
                    cache_write_tokens = COALESCE(sessions.cache_write_tokens, 0)
                                       + COALESCE(EXCLUDED.cache_write_tokens, 0),
                    tool_call_count = COALESCE(sessions.tool_call_count, 0)
                                    + COALESCE(EXCLUDED.tool_call_count, 0),
                    error_count = COALESCE(sessions.error_count, 0)
                                + COALESCE(EXCLUDED.error_count, 0),
"""


def _connect_bounded(db_path: str, memory_limit: str, threads: int):
    """Open `db_path` with an explicit buffer-pool ceiling.

    Every connection to the telemetry database goes through here — `__init__`
    AND `_reopen`. That matters more than it looks: a bound applied only at
    startup silently disappears the first time the backend recovers from a
    fatal error, and the recovered daemon then runs unbounded with no symptom
    until the machine is swapping. See `StorageConfig.memory_limit` for why the
    default is small.

    `temp_directory` is what makes the ceiling safe rather than fatal — past the
    limit DuckDB spills there instead of raising — and it sits beside the
    database so the spill lands on the volume already allotted to this tool. A
    value DuckDB rejects must never make the database unopenable, so a bad
    override degrades to the default rather than propagating.
    """
    # Annotated because duckdb's `config=` parameter is typed as an invariant
    # dict over a union: an inferred dict[str, str] is rejected by mypy even
    # though every value here is a str.
    settings: dict[str, str | bool | int | float | list[str]] = {
        "memory_limit": memory_limit or StorageConfig.memory_limit,
        "threads": str(threads or StorageConfig.threads),
        "temp_directory": str(Path(db_path).parent / "duckdb_temp"),
    }
    try:
        return duckdb.connect(db_path, config=settings)
    except duckdb.Error:
        logger.warning(
            "storage.memory_limit=%r / storage.threads=%r rejected by DuckDB; "
            "falling back to %s / %s",
            memory_limit, threads,
            StorageConfig.memory_limit, StorageConfig.threads,
        )
        settings["memory_limit"] = StorageConfig.memory_limit
        settings["threads"] = str(StorageConfig.threads)
        return duckdb.connect(db_path, config=settings)


class DuckDBBackend:
    """Concrete DuckDB implementation of StorageBackend."""

    def __init__(self, config: StorageConfig) -> None:
        db_path = Path(config.path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path: str | None = str(db_path)
        self._memory_limit = config.memory_limit
        self._threads = config.threads
        self._conn = _connect_bounded(str(db_path), config.memory_limit, config.threads)
        run_migrations(self._conn)
        self._local = threading.local()
        # Every cursor `conn` has handed out. Recovery must close ALL of them —
        # a `threading.local` cannot be enumerated from another thread, and one
        # surviving connection is enough to keep an invalidated database
        # instance alive in DuckDB's per-path cache (see `is_fatal_db_error`).
        # `_generation` is how a thread notices its cursor belongs to a
        # torn-down connection and lazily takes a fresh one.
        self._cursors: list[duckdb.DuckDBPyConnection] = []
        self._generation = 0
        # Deliberately NOT `write_lock`. Connection lifecycle must not order
        # against the write path: a writer that hits a fatal records it under
        # `_FATAL_LOCK` while still on the write path, and recovery holds
        # `_FATAL_LOCK` while tearing connections down — sharing one lock
        # between the two would make that pair deadlock-able.
        self._conn_lock = threading.RLock()
        _LIVE_BACKENDS.add(self)
        # Serializes *writes* across threads. Reads use per-thread cursors and
        # stay lock-free (#124), but DuckDB uses optimistic concurrency control:
        # two transactions mutating the same table from different threads can
        # abort with a write-write conflict. In async-hooks mode the main thread
        # writes spans/cost while the TjHookWorker thread writes alerts, so every
        # mutating method takes this re-entrant lock. Held only for the duration
        # of a single write, so it never blocks the concurrent read path.
        self._write_lock = threading.RLock()

    @property
    def write_lock(self) -> "threading.RLock":
        """Re-entrant lock guarding cross-thread writes (see __init__)."""
        return self._write_lock

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
        if cur is None or getattr(self._local, "generation", None) != self._generation:
            with self._conn_lock:
                cur = self._conn.cursor()
                self._cursors.append(cur)
            self._local.cursor = cur
            self._local.generation = self._generation
        return cur

    # -- connection health and recovery --
    #
    # See the module-level "Fatal errors and database invalidation" note for
    # why recovery cannot be done by one backend alone, and why it works at all.

    @property
    def recoverable(self) -> bool:
        """Whether this backend can be torn down and reopened from disk.

        False for `InMemoryBackend`, whose database only exists inside its
        connection — closing it would discard the data rather than recover it.
        """
        return self._db_path is not None

    def check_health(self) -> bool:
        """Whether this backend can still answer a query.

        `SELECT 1` is enough: an invalidated instance fails it, which is what
        makes a health probe able to tell "the process is up" apart from "the
        process can still read its database". Never raises.
        """
        try:
            self.conn.execute("SELECT 1").fetchone()
        except Exception as exc:  # noqa: BLE001 - a probe reports, never raises
            if is_fatal_db_error(exc):
                note_fatal_db_error(exc)
            return False
        return True

    def _teardown_connections(self) -> None:
        """Close every connection this backend holds, ignoring close errors.

        Errors are ignored deliberately: closing an already-invalidated handle
        can itself raise, and a failure to close cleanly must not stop us from
        closing the REST — the eviction only happens once they are all gone.
        """
        with self._conn_lock:
            for cur in self._cursors:
                try:
                    cur.close()
                except Exception:  # noqa: BLE001
                    pass
            self._cursors.clear()
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            # Bump BEFORE reopening so any thread racing in on the old cursor
            # is forced to take a fresh one rather than reusing a closed handle.
            self._generation += 1

    def _reopen(self) -> bool:
        if self._db_path is None:
            return False
        try:
            with self._conn_lock:
                self._conn = _connect_bounded(
                    self._db_path, self._memory_limit, self._threads,
                )
                run_migrations(self._conn)
            return self.check_health()
        except Exception as exc:  # noqa: BLE001
            logger.error("could not re-establish the database connection: %s", exc)
            return False

    # -- writes --

    def insert_span(self, span: NormalizedSpan) -> None:
        # Named-column INSERT so future migrations adding columns don't break
        # positional-arg ordering (migration 4 added billing_account at the
        # end of the table, but we don't want to silently rely on that).
        from tokenjam.core.optimize import ingest_watermark

        with self._write_lock:
            self.conn.execute(
                "INSERT INTO spans ("
                "span_id, trace_id, parent_span_id, session_id, agent_id, "
                "name, kind, status_code, status_message, start_time, end_time, "
                "duration_ms, attributes, provider, model, tool_name, "
                "input_tokens, output_tokens, cache_tokens, cost_usd, "
                "request_type, conversation_id, events, billing_account, "
                "cache_write_tokens, request_params, request_tools, sub_agent_id, "
                "tenant_id, feature, environment, service_version, commit_sha, "
                "prompt_template_id, prompt_template_version, pricing_source, "
                "sub_agent_type"
                ") VALUES "
                "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,"
                "$29,$30,$31,$32,$33,$34,$35,$36,$37)",
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
                    span.tenant_id, span.feature, span.environment, span.service_version,
                    span.commit_sha, span.prompt_template_id, span.prompt_template_version,
                    span.pricing_source, span.sub_agent_type,
                ],
            )
        ingest_watermark.bump(1)

    def bulk_insert_spans(self, spans: Sequence[NormalizedSpan]) -> None:
        """Columnar bulk-append of many spans in a single vectorized statement.

        Replaces the per-row `insert_span`/`executemany` marshalling on the
        backfill hot path: the whole batch is written once as newline-delimited
        JSON and DuckDB's native `read_json` scans it in one `INSERT..SELECT`, so
        a multi-GB Claude Code history ingests in seconds instead of minutes.
        Dependency-free (DuckDB reads JSON natively — no pandas/pyarrow).

        Idempotent: a `WHERE NOT EXISTS` anti-join skips any `span_id` already
        present, so callers that pre-filter for counting still get correct DB
        contents even if a concurrent writer inserts the same id mid-flight
        (no PRIMARY KEY conflict is raised). Resulting rows — columns, JSON, and
        TIMESTAMPTZ instants — are identical to the per-row path.
        """
        if not spans:
            return
        from tokenjam.core.optimize import ingest_watermark

        # Write the NDJSON payload OUTSIDE the write lock (pure CPU/IO, no DB),
        # then hold the lock only for the single vectorized INSERT..SELECT.
        fd, path = tempfile.mkstemp(prefix="tj-spans-", suffix=".ndjson")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for span in spans:
                    fh.write(json.dumps(_span_to_json_obj(span)))
                    fh.write("\n")
            with self._write_lock:
                self.conn.execute(_BULK_SPAN_INSERT_SQL, [path])
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        # Upper bound, not the exact post-anti-join insert count: the
        # anti-join can skip already-present span_ids, and counting precisely
        # would mean a second query on the hot path for a signal that only
        # needs "did anything happen". Overcounting only makes the watermark
        # gate marginally more willing to fire — never less safe.
        ingest_watermark.bump(len(spans))

    def bulk_overlay_span_attrs(
        self, updates: Sequence[tuple[str, str | None, str | None, dict | None]],
    ) -> int:
        """Fill `sub_agent_id`/`sub_agent_type`/`attributes` on EXISTING
        spans, additively.

        `updates` is `(span_id, sub_agent_id, sub_agent_type, attributes)`
        tuples — the freshly re-parsed values for spans a caller already
        knows are present in the store (e.g. a backfill re-walking
        transcripts it has ingested before, or re-parsing now that
        `[capture]` is on). `attributes` is the span's FULL freshly-parsed
        attributes dict (or `None` to skip the content overlay for that
        span); see `_SUBAGENT_OVERLAY_MATCH_PREDICATE` for why the merge can
        never clobber a key the stored row already carries, scalar or JSON.
        Returns the number of spans that actually changed (not
        `len(updates)` — most calls offer values the row already has, which
        is a no-op).
        """
        if not updates:
            return 0
        fd, path = tempfile.mkstemp(prefix="tj-subagent-overlay-", suffix=".ndjson")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for span_id, sub_agent_id, sub_agent_type, attributes in updates:
                    fh.write(json.dumps({
                        "span_id": span_id,
                        "sub_agent_id": sub_agent_id,
                        "sub_agent_type": sub_agent_type,
                        "attributes": attributes,
                    }))
                    fh.write("\n")
            with self._write_lock:
                # COUNT first, then the plain UPDATE (no RETURNING — see the
                # module comment above `_BULK_SUBAGENT_OVERLAY_COUNT_SQL` for
                # why). Both share `_SUBAGENT_OVERLAY_MATCH_PREDICATE` and run
                # back-to-back under the write lock, so nothing else can
                # change the matched set between the two.
                changed = self.conn.execute(
                    _BULK_SUBAGENT_OVERLAY_COUNT_SQL, [path],
                ).fetchone()
                self.conn.execute(_BULK_SUBAGENT_OVERLAY_UPDATE_SQL, [path])
            return int(changed[0]) if changed else 0
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def insert_alert(self, alert: Alert) -> None:
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO alerts VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
                [
                    alert.alert_id, alert.agent_id, alert.session_id, alert.span_id,
                    alert.fired_at, alert.type.value, alert.severity.value, alert.title,
                    json.dumps(alert.detail), alert.acknowledged, alert.suppressed,
                ],
            )

    def insert_validation(self, result: SchemaValidationResult) -> None:
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO schema_validations VALUES ($1,$2,$3,$4,$5,$6)",
                [
                    result.validation_id, result.span_id, result.agent_id,
                    result.validated_at, result.passed, json.dumps(result.errors),
                ],
            )

    def insert_policy_decision(self, decision: PolicyDecisionRecord) -> None:
        # Append-only audit log (#221). Named columns so future migrations stay safe.
        with self._write_lock:
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
        with self._write_lock:
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

    def upsert_session(
        self, session: SessionRecord, *, accumulate_totals: bool = False,
    ) -> None:
        """Write a session row.

        By default the incoming totals REPLACE the stored ones, because the
        live path already accumulates in Python (`_build_or_update_session`
        reads the row, adds the span, writes the new total back) and a second
        accumulation in SQL would double every live figure.

        `accumulate_totals=True` ADDS them instead, for a caller whose record
        describes a DELTA rather than a session's whole life. The Claude Code
        backfill is that caller: a session is split across files sharing one
        session_id (main thread plus each `subagents/agent-*.jsonl`), so a
        replacing write per file leaves the row describing only the last file
        processed — `SUM(spans)` and `sessions.total_cost_usd` then disagree,
        which is exactly the drift `session_cost_drift` reports. The delta is
        computed over the spans that write actually INSERTED, so re-running a
        file whose spans are all already present adds zero and idempotency
        holds.

        The two SQL bodies differ only in their totals assignments; the
        interpolated fragment is a module constant, never user data.
        """
        totals = _SESSION_TOTALS_ACCUMULATE if accumulate_totals else _SESSION_TOTALS_REPLACE
        # plan_tier: promote unknown → known on conflict; never overwrite a
        # session that already has a known tier (backfill re-runs must not
        # clobber historical tiers when config plan changes).
        with self._write_lock:
            self.conn.execute(
                f"""
                INSERT INTO sessions (
                    session_id, agent_id, conversation_id, started_at, ended_at,
                    status, total_cost_usd, input_tokens, output_tokens, cache_tokens,
                    tool_call_count, error_count, plan_tier, service_namespace,
                    service_instance_id, cache_write_tokens, run_id, parent_session_id,
                    source, task_statement_hash, dominant_model
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21)
                ON CONFLICT (session_id) DO UPDATE SET
                    -- `started_at` was absent from this list entirely, which
                    -- made it WRITE-ONCE: whatever the first span to reach a
                    -- session stamped was permanent, and no genuinely earlier
                    -- span arriving later could correct it. Since only
                    -- `ended_at` ever advanced, a session opened by an
                    -- out-of-order or mis-stamped span stayed wrong forever —
                    -- which is why bad session timestamps accumulated in a
                    -- corpus instead of healing. A session starts when its
                    -- EARLIEST observed span does, so take the minimum; the
                    -- COALESCE keeps a NULL incoming value from erasing a
                    -- stored one, since MIN semantics here must not be
                    -- confused with "unknown wins".
                    started_at = LEAST(
                        sessions.started_at,
                        COALESCE(EXCLUDED.started_at, sessions.started_at)
                    ),
                    ended_at = COALESCE(EXCLUDED.ended_at, sessions.ended_at),
                    -- Refuse to downgrade a row the live path already marked
                    -- 'active' when the incoming write's own last-activity is
                    -- STALER than what's already stored -- e.g. a backfill/
                    -- catch-up pass re-parsing a transcript whose on-disk
                    -- snapshot lags spans the live OTLP path already recorded
                    -- moments earlier. Only blocks the specific case of
                    -- (stored='active', incoming!='active', incoming older);
                    -- a genuinely newer completion always wins, and an
                    -- explicit close never goes through this path at all
                    -- (close_session_by_id / close_sessions_by_instance are
                    -- direct UPDATEs, so they are never subject to this guard).
                    status = CASE
                        WHEN sessions.status = 'active'
                         AND EXCLUDED.status != 'active'
                         AND COALESCE(sessions.ended_at, sessions.started_at)
                             > COALESCE(EXCLUDED.ended_at, EXCLUDED.started_at)
                        THEN sessions.status
                        ELSE EXCLUDED.status
                    END,
                    {totals}
                    plan_tier = CASE
                        WHEN COALESCE(sessions.plan_tier, 'unknown') != 'unknown'
                        THEN sessions.plan_tier
                        ELSE EXCLUDED.plan_tier
                    END,
                    service_namespace = COALESCE(EXCLUDED.service_namespace, sessions.service_namespace),
                    service_instance_id = COALESCE(EXCLUDED.service_instance_id, sessions.service_instance_id),
                    run_id = COALESCE(EXCLUDED.run_id, sessions.run_id),
                    parent_session_id = COALESCE(EXCLUDED.parent_session_id, sessions.parent_session_id),
                    -- Provenance and task identity are properties of the
                    -- session as a WHOLE, fixed at its first observation —
                    -- fill once, like plan_tier above, never flip a value a
                    -- prior write already resolved.
                    source = COALESCE(sessions.source, EXCLUDED.source),
                    task_statement_hash = COALESCE(sessions.task_statement_hash, EXCLUDED.task_statement_hash),
                    dominant_model = COALESCE(sessions.dominant_model, EXCLUDED.dominant_model)
                """,
                [
                    session.session_id, session.agent_id, session.conversation_id,
                    session.started_at, session.ended_at, session.status,
                    session.total_cost_usd, session.input_tokens, session.output_tokens,
                    session.cache_tokens, session.tool_call_count, session.error_count,
                    session.plan_tier, session.service_namespace,
                    session.service_instance_id, session.cache_write_tokens,
                    session.run_id, session.parent_session_id,
                    session.source, session.task_statement_hash, session.dominant_model,
                ],
            )

    def recompute_session_totals_from_spans(self, session_ids: list[str]) -> None:
        """Reconcile session aggregates to canonical logical-call observations.

        Backfill upserts a session row once per on-disk file, but a Claude Code
        session is split across files that share one session_id (the main-thread
        transcript plus each subagents/agent-<id>.jsonl). Because upsert_session
        uses replace semantics, the per-file upserts would otherwise leave the
        the row holding only the last-processed file's totals. Cross-source
        restatements are counted once using the duplicate-observation winner
        rule; same-source repeats remain real calls. Scoped to the given ids so
        it never touches unrelated sessions. Idempotent.
        """
        if not session_ids:
            return
        with self._write_lock:
            self.conn.execute(
                f"""
                WITH redundant AS (
                    {_duplicate_observation_sql(
                        "obs.span_id",
                        scope_sql="session_id IN (SELECT unnest($1))",
                    )}
                ), canonical AS (
                    SELECT session_id, input_tokens, output_tokens,
                           cache_tokens, cache_write_tokens, cost_usd, tool_name
                    FROM spans
                    WHERE session_id IN (SELECT unnest($1))
                      AND span_id NOT IN (SELECT span_id FROM redundant)
                ), agg AS (
                    SELECT session_id,
                           COALESCE(SUM(input_tokens), 0)       AS input_tokens,
                           COALESCE(SUM(output_tokens), 0)      AS output_tokens,
                           COALESCE(SUM(cache_tokens), 0)       AS cache_tokens,
                           COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                           COALESCE(SUM(cost_usd), 0.0)         AS total_cost_usd,
                           COUNT(*) FILTER (WHERE tool_name IS NOT NULL) AS tool_call_count
                    FROM canonical
                    GROUP BY session_id
                )
                UPDATE sessions AS s SET
                    input_tokens       = agg.input_tokens,
                    output_tokens      = agg.output_tokens,
                    cache_tokens       = agg.cache_tokens,
                    cache_write_tokens = agg.cache_write_tokens,
                    total_cost_usd     = agg.total_cost_usd,
                    tool_call_count    = agg.tool_call_count
                FROM agg
                WHERE s.session_id = agg.session_id
                """,
                [list(session_ids)],
            )

    def reconcile_backfill_spans(
        self, keep_by_session: dict[str, set[str]], source: str
    ) -> int:
        """Purge stale-scheme `source`-tagged spans across the given sessions.

        Self-healing reconciliation for the stale-scheme duplicate bug (#294/#300
        cross-version): a DB written by ≤v0.5.1 keyed backfill span_ids on the
        record `uuid`; current code keys on the stable `message.id`. The two
        schemes are DISJOINT, so a re-backfill of an old DB ADDS a full duplicate
        set alongside the stale uuid-keyed rows, inflating token/cost totals ~2.6×.

        `keep_by_session` maps each session_id to the COMPLETE current-scheme
        span_id set for that session (LLM + tool spans, unioned across all of its
        on-disk files). Any `source`-tagged span in one of those sessions whose
        span_id is NOT in the session's keep set can only be a stale-scheme
        orphan, so we delete it. Returns the number of rows deleted.

        Scoped to (session_id, source): never touches live-ingested spans, spans
        from other sources, or sessions not in `keep_by_session`. A session with
        an empty keep set is skipped (defensive: never wipe a session on a parse
        that produced no spans). Idempotent — on a clean current-scheme DB the
        keep sets already cover every stored backfill span, so nothing matches.

        ART-index workaround: DuckDB (through ≥1.5.2) raises a FATAL "Failed to
        delete all rows from index" — invalidating the whole connection — when
        deleting indexed span rows on some persisted DB states. We drop the five
        secondary spans indexes, run the deletes, then recreate them (identical
        DDL to migration 2/3). Drop+recreate is cheap (~20ms on 50k rows). All of
        it runs under `write_lock`, so no concurrent cursor sees the table without
        its indexes. The recreate is in a `finally` so a mid-delete error can't
        leave the table permanently unindexed.
        """
        # Materialize the stale-span ids up front (a read) so the write section
        # is a tight drop → delete → recreate with no interleaved reads.
        to_delete: list[tuple[str, list[str]]] = []
        for session_id, keep in keep_by_session.items():
            if not keep:
                continue
            rows = self.conn.execute(
                """
                SELECT span_id FROM spans
                WHERE session_id = $1
                  AND json_extract_string(attributes, '$.source') = $2
                """,
                [session_id, source],
            ).fetchall()
            stale = [r[0] for r in rows if r[0] not in keep]
            if stale:
                to_delete.append((session_id, stale))
        if not to_delete:
            return 0

        deleted = 0
        with self._write_lock:
            self.conn.execute(
                "DROP INDEX IF EXISTS idx_spans_trace_id;\n"
                "DROP INDEX IF EXISTS idx_spans_agent_id;\n"
                "DROP INDEX IF EXISTS idx_spans_start_time;\n"
                "DROP INDEX IF EXISTS idx_spans_tool_name;\n"
                "DROP INDEX IF EXISTS idx_spans_conv_id;\n"
                "DROP INDEX IF EXISTS idx_spans_session_id"
            )
            try:
                chunk = 5000
                for session_id, stale in to_delete:
                    for start in range(0, len(stale), chunk):
                        batch = stale[start:start + chunk]
                        placeholders = ",".join(
                            f"${i + 2}" for i in range(len(batch))
                        )
                        self.conn.execute(
                            f"DELETE FROM spans WHERE session_id = $1 "
                            f"AND span_id IN ({placeholders})",
                            [session_id, *batch],
                        )
                        deleted += len(batch)
            finally:
                self.conn.execute(SPANS_INDEX_SQL)
        return deleted

    def upsert_agent(self, agent: AgentRecord) -> None:
        with self._write_lock:
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
        with self._write_lock:
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
            with self._write_lock:
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
            with self._write_lock:
                self.conn.execute(
                    "UPDATE sessions SET status = 'closed', "
                    "ended_at = COALESCE(ended_at, $2) "
                    "WHERE session_id = $1 AND status = 'active'",
                    [session_id, now],
                )
        return count

    def mark_sessions_completed(self, session_ids: list[str]) -> None:
        """Correct raw `status='active'` rows to 'completed' for the given ids.

        Used by the periodic zombie sweep (`transcript_sync.
        sweep_stale_active_sessions`) to write back a terminal status for
        sessions whose COMPUTED status (`SessionRecord.status_at` /
        `status_with_transcript_mtime`) already reads as stale, so raw-column
        consumers (MCP tools, `tj status`, the sessions route, relearn_apply)
        stop overstating "active" indefinitely.

        `AND status = 'active'` re-checks at write time (belt-and-braces
        against a race with a concurrent explicit close or live span landing
        between the sweep's read and this write) and never touches
        `ended_at`, tokens, or cost -- status only. Idempotent: re-running
        against ids already corrected (or since closed) is a no-op for them.
        """
        if not session_ids:
            return
        with self._write_lock:
            placeholders = ", ".join(f"${i + 1}" for i in range(len(session_ids)))
            self.conn.execute(
                f"UPDATE sessions SET status = 'completed' "
                f"WHERE session_id IN ({placeholders}) AND status = 'active'",
                session_ids,
            )

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
        # The persona scope, applied HERE so every consumer of this WHERE — the
        # row list, `count_traces`, and `get_trace_cost_stats`' outlier
        # quartiles — covers the same population by construction.
        add_persona_clause(clauses, filters.persona)
        where = " AND ".join(clauses) if clauses else "1=1"
        return where, params, idx

    # Trace-cost-ranking sort options. "recent" (default) preserves the
    # historical reverse-chronological order; "cost" ranks the highest-spend
    # trace first within the same filtered/paginated window — additive, not a
    # replacement (see TracesListView in ui/index.html for the toggle).
    _TRACE_SORT_EXPR = {
        "recent": "tc.start_time DESC",
        "cost": "tc.cost_usd DESC NULLS LAST, tc.start_time DESC",
    }

    # Statistical cost-outlier rule (Tukey's fence): a trace is flagged when its
    # cost sits above Q3 + 1.5 * IQR of the *priced* (cost_usd > 0) traces in the
    # same filtered window. Below this many priced traces the quartiles are too
    # noisy to mean anything, so nothing is flagged at all. This is the classic
    # box-plot outlier rule — conservative, well-known, and cheap to compute
    # alongside the existing per-trace aggregation (no second table scan).
    MIN_OUTLIER_SAMPLE = 8

    def get_traces(self, filters: TraceFilters) -> list[TraceRecord]:
        where, params, idx = self._trace_filter_where(filters)
        sort_expr = self._TRACE_SORT_EXPR.get(filters.sort, self._TRACE_SORT_EXPR["recent"])
        having = ""
        if filters.min_cost_usd is not None:
            having = f"WHERE tc.cost_usd >= ${idx}"
            params.append(filters.min_cost_usd)
            idx += 1
        # Use FIRST(name ORDER BY start_time) to pick the root span name —
        # the previous correlated-subquery variant returned NULL for most
        # rows in DuckDB, leaving the TYPE column blank in `tj traces` (U2).
        #
        # trace_costs aggregates spans -> traces over the SAME filtered window
        # as before (bounded by since/until/agent_id/etc, same complexity class
        # as the prior query — no new scan). cost_stats computes the window's
        # cost quartiles ONCE, over the unfiltered-by-threshold trace set, so a
        # `min_cost_usd` filter never skews the outlier fence. Both are derived
        # from one pass over trace_costs; the min-cost filter and pagination are
        # applied only in the outer SELECT.
        sql = (
            f"WITH trace_costs AS ("
            f"  SELECT trace_id, MAX(agent_id) AS agent_id, "
            f"  FIRST(name ORDER BY start_time) AS name, "
            f"  MIN(start_time) AS start_time, "
            f"  SUM(duration_ms) AS duration_ms, "
            f"  SUM(cost_usd) AS cost_usd, "
            f"  CASE WHEN SUM(CASE WHEN status_code='error' THEN 1 ELSE 0 END) > 0 THEN 'error' "
            f"       WHEN SUM(CASE WHEN status_code='ok' THEN 1 ELSE 0 END) > 0 THEN 'ok' "
            f"       ELSE 'unset' END AS status_code, "
            f"  COUNT(*) AS span_count, "
            f"  SUM(input_tokens) AS input_tokens, "
            f"  SUM(output_tokens) AS output_tokens "
            f"  FROM spans WHERE {where} "
            f"  GROUP BY trace_id"
            f"), cost_stats AS ("
            f"  SELECT "
            f"  PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY cost_usd) AS q1, "
            f"  PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY cost_usd) AS q3, "
            f"  COUNT(*) AS priced_count "
            f"  FROM trace_costs WHERE cost_usd > 0"
            f") "
            f"SELECT tc.trace_id, tc.agent_id, tc.name, tc.start_time, tc.duration_ms, "
            f"tc.cost_usd, tc.status_code, tc.span_count, tc.input_tokens, tc.output_tokens, "
            f"cs.q1, cs.q3, cs.priced_count "
            f"FROM trace_costs tc CROSS JOIN cost_stats cs "
            f"{having} "
            f"ORDER BY {sort_expr} "
            f"LIMIT ${idx} OFFSET ${idx + 1}"
        )
        params.extend([filters.limit, filters.offset])
        rows = self.conn.execute(sql, params).fetchall()
        result = []
        for r in rows:
            cost_usd = r[5]
            q1, q3, priced_count = r[10], r[11], int(r[12] or 0)
            is_outlier = _is_cost_outlier(cost_usd, q1, q3, priced_count, self.MIN_OUTLIER_SAMPLE)
            result.append(TraceRecord(
                trace_id=r[0], agent_id=r[1], name=r[2], start_time=r[3],
                duration_ms=r[4], cost_usd=cost_usd, status_code=r[6],
                span_count=r[7],
                input_tokens=int(r[8] or 0), output_tokens=int(r[9] or 0),
                is_outlier=is_outlier,
            ))
        return result

    def get_trace_cost_stats(self, filters: TraceFilters) -> TraceCostStats:
        """Window-level cost distribution behind `TraceRecord.is_outlier`.

        Mirrors the `cost_stats` CTE in `get_traces` (same `where`, no
        pagination/threshold) so the numbers the UI shows for "why is this
        flagged" always match the flags actually returned.
        """
        where, params, _ = self._trace_filter_where(filters)
        sql = (
            f"WITH trace_costs AS ("
            f"  SELECT trace_id, SUM(cost_usd) AS cost_usd "
            f"  FROM spans WHERE {where} GROUP BY trace_id"
            f") "
            f"SELECT "
            f"PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY cost_usd) AS q1, "
            f"PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY cost_usd) AS q3, "
            f"COUNT(*) AS priced_count "
            f"FROM trace_costs WHERE cost_usd > 0"
        )
        row = self.conn.execute(sql, params).fetchone()
        q1, q3, priced_count = (row[0], row[1], int(row[2] or 0)) if row else (None, None, 0)
        threshold = None
        if q1 is not None and q3 is not None and priced_count >= self.MIN_OUTLIER_SAMPLE:
            threshold = q3 + 1.5 * (q3 - q1)
        return TraceCostStats(
            method="iqr_1.5x",
            sample_size=priced_count,
            min_sample=self.MIN_OUTLIER_SAMPLE,
            q1_usd=q1,
            q3_usd=q3,
            threshold_usd=threshold,
        )

    def count_traces(self, filters: TraceFilters) -> int:
        where, params, idx = self._trace_filter_where(filters)
        if filters.min_cost_usd is not None:
            sql = (
                f"SELECT COUNT(*) FROM ("
                f"  SELECT trace_id FROM spans WHERE {where} "
                f"  GROUP BY trace_id HAVING SUM(cost_usd) >= ${idx}"
                f")"
            )
            params.append(filters.min_cost_usd)
        else:
            sql = f"SELECT COUNT(DISTINCT trace_id) FROM spans WHERE {where}"
        row = self.conn.execute(sql, params).fetchone()
        return int(row[0] or 0) if row else 0

    def get_session_ids_for_trace(self, trace_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT session_id FROM spans "
            "WHERE trace_id = $1 AND session_id IS NOT NULL "
            "ORDER BY session_id",
            [trace_id],
        ).fetchall()
        return [str(row[0]) for row in rows]

    def get_marker_session_ids_for_trace(self, trace_id: str) -> list[str]:
        """Return distinct session ids from explicit session marker spans.

        A trace can contain several sessions during fan-out. The old
        ``LIMIT 1`` lookup silently selected whichever row happened to come
        first, so a trace-only span could be charged to the wrong session.
        Marker identity is narrower than arbitrary span membership and lets
        ingest distinguish an unambiguous trace from a shared one.
        """
        from tokenjam.otel.semconv import GenAIAttributes

        rows = self.conn.execute(
            "SELECT DISTINCT session_id FROM spans "
            "WHERE trace_id = $1 AND name = $2 AND session_id IS NOT NULL "
            "AND (json_extract_string(attributes, '$.\"tokenjam.session_attribution\"') "
            "IS NULL OR json_extract_string(attributes, '$.\"tokenjam.session_attribution\"') "
            "IN ('explicit', 'conversation')) "
            "ORDER BY session_id",
            [trace_id, GenAIAttributes.SPAN_INVOKE_AGENT],
        ).fetchall()
        return [str(row[0]) for row in rows]

    def reconcile_trace_session_attribution(self, trace_id: str) -> None:
        """Reconcile provisional trace-derived spans after a marker arrives.

        A span that arrived before its marker may have been placed in a
        provisional session, while a span after the marker may have used a
        trace-derived session. If the trace has one marker, those derived
        spans can be moved to it. If the trace has multiple markers, derived
        spans are cleared rather than charged to every session. Explicit and
        parent-derived assignments based on an explicit identity are never
        rewritten here.
        """
        with self._write_lock:
            marker_ids = self.get_marker_session_ids_for_trace(trace_id)
            if not marker_ids:
                return

            derived_path = (
                "json_extract_string(attributes, '$.\"tokenjam.session_attribution\"') "
                "IN ('generated', 'trace', 'parent_generated')"
            )
            old_rows = self.conn.execute(
                "SELECT DISTINCT session_id FROM spans "
                "WHERE trace_id = $1 AND session_id IS NOT NULL AND "
                f"{derived_path}",
                [trace_id],
            ).fetchall()
            old_session_ids = {str(row[0]) for row in old_rows}
            if not old_session_ids:
                return

            self.conn.execute("BEGIN TRANSACTION")
            try:
                if len(marker_ids) == 1:
                    self.conn.execute(
                        "UPDATE spans SET session_id = $1 "
                        "WHERE trace_id = $2 AND "
                        f"{derived_path}",
                        [marker_ids[0], trace_id],
                    )
                    affected = old_session_ids | set(marker_ids)
                else:
                    self.conn.execute(
                        "UPDATE spans SET session_id = NULL "
                        "WHERE trace_id = $1 AND "
                        f"{derived_path}",
                        [trace_id],
                    )
                    affected = old_session_ids

                self.recompute_session_totals_from_spans(sorted(affected))
                for session_id in old_session_ids - set(marker_ids):
                    self.conn.execute(
                        "DELETE FROM sessions WHERE session_id = $1 "
                        "AND NOT EXISTS (SELECT 1 FROM spans WHERE session_id = $1)",
                        [session_id],
                    )
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def get_session_id_for_trace(self, trace_id: str) -> str | None:
        """Return the trace session only when the trace is unambiguous."""
        session_ids = self.get_session_ids_for_trace(trace_id)
        return session_ids[0] if len(session_ids) == 1 else None

    def get_trace_spans(self, trace_id: str) -> list[NormalizedSpan]:
        cur = self.conn.execute(
            "SELECT * FROM spans WHERE trace_id = $1 ORDER BY start_time", [trace_id]
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [_row_to_span(r, cols) for r in rows]

    def get_span(self, trace_id: str, span_id: str) -> NormalizedSpan | None:
        """Targeted single-span fetch (#653).

        A WHERE span_id=? lookup so the span-detail lazy-load reads ONE row
        instead of scanning + deserializing the whole trace's attributes on
        every expand. `trace_id` is part of the predicate so the route's
        404-on-unknown behavior stays scoped to the trace the user is viewing.
        """
        cur = self.conn.execute(
            "SELECT * FROM spans WHERE trace_id = $1 AND span_id = $2 LIMIT 1",
            [trace_id, span_id],
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return _row_to_span(row, cols)

    def get_cost_summary(self, filters: CostFilters) -> list[CostRow]:
        # SDK cost-attribution dimensions (tenant/feature/environment/prompt
        # version) — added alongside agent/model/day/tool. All four are plain
        # columns on `spans` (see migration 17).
        attribution_dims = ("tenant", "feature", "environment", "prompt_version")
        group_col_map = {
            "day": "CAST(start_time AT TIME ZONE 'UTC' AS DATE)",
            "agent": "agent_id",
            "model": "model",
            "tool": "tool_name",
            "tenant": "tenant_id",
            "feature": "feature",
            "environment": "environment",
            "prompt_version": "prompt_template_version",
        }
        group_expr = group_col_map.get(
            filters.group_by, "CAST(start_time AT TIME ZONE 'UTC' AS DATE)"
        )

        # gen_ai.tool.call spans are separate rows from the LLM completion
        # spans that carry model/cost/tokens (otel/otlp_parsing.py) — a span
        # has tool_name set XOR model set, never both. Filtering every
        # grouping on `model IS NOT NULL` silently dropped every tool span, so
        # `--group-by tool` collapsed to one bogus "TOOL: None" bucket built
        # from LLM spans (whose tool_name is NULL). Tool grouping needs the
        # presence predicate get_tool_calls() already uses below.
        presence_clause = (
            "tool_name IS NOT NULL" if filters.group_by == "tool" else "model IS NOT NULL"
        )
        clauses: list[str] = [presence_clause]
        # Attribution dims additionally require the dimension itself to be set
        # (rather than folding NULL/unset spend into a misleading "(none)"
        # bucket) — a caller wanting to see unattributed spend can compare this
        # grouping's total against the ungrouped window total. This is the
        # "degrade honestly" contract: an empty `rows` list here means the
        # dimension was never set at the call site, not that spend was zero.
        if filters.group_by in attribution_dims:
            clauses.append(f"{group_expr} IS NOT NULL")
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
        # Equality filters for the attribution dimensions themselves — independent
        # of group_by, so e.g. group_by="model" + tenant_id="acme" scopes a
        # per-model breakdown to one tenant.
        if filters.tenant_id:
            clauses.append(f"tenant_id = ${idx}")
            params.append(filters.tenant_id)
            idx += 1
        if filters.feature:
            clauses.append(f"feature = ${idx}")
            params.append(filters.feature)
            idx += 1
        if filters.environment:
            clauses.append(f"environment = ${idx}")
            params.append(filters.environment)
            idx += 1
        if filters.prompt_version:
            clauses.append(f"prompt_template_version = ${idx}")
            params.append(filters.prompt_version)
            idx += 1
        add_persona_clause(clauses, filters.persona)
        where = " AND ".join(clauses)

        # Cache-read + cache-write are summed alongside in/out so callers can
        # show the full token picture (cache-write is often the dominant cost
        # driver yet was invisible above the DB before this column existed).
        # call_count is the only genuinely honest metric for the "tool"
        # grouping: tool-call spans carry no cost/tokens of their own — cost
        # is attributed to the LLM completion span the tool call accompanied,
        # not the tool invocation itself.
        token_cols = (
            "COALESCE(SUM(input_tokens), 0), "
            "COALESCE(SUM(output_tokens), 0), "
            "COALESCE(SUM(cache_tokens), 0), "
            "COALESCE(SUM(cache_write_tokens), 0), "
            "COALESCE(SUM(cost_usd), 0.0), "
            "COUNT(*) "
        )
        if filters.group_by in ("agent", "model"):
            sql = (
                f"SELECT {group_expr} AS grp, agent_id, model, " + token_cols
                + f"FROM spans WHERE {where} "
                f"GROUP BY grp, agent_id, model "
                f"ORDER BY grp DESC"
            )
        elif filters.group_by == "tool":
            # Busiest tool first — the grp-alphabetical order used for day
            # buckets is meaningless once cost/tokens are uniformly zero.
            sql = (
                f"SELECT {group_expr} AS grp, NULL AS agent_id, NULL AS model, " + token_cols
                + f"FROM spans WHERE {where} "
                f"GROUP BY grp "
                f"ORDER BY COUNT(*) DESC"
            )
        elif filters.group_by in attribution_dims:
            # Biggest spender first — the whole point of this grouping is
            # concentration (which tenant/feature/environment/prompt version is
            # driving spend), so cost order is the useful default.
            sql = (
                f"SELECT {group_expr} AS grp, NULL AS agent_id, NULL AS model, " + token_cols
                + f"FROM spans WHERE {where} "
                f"GROUP BY grp "
                f"ORDER BY COALESCE(SUM(cost_usd), 0.0) DESC"
            )
        else:
            # day: group only by the primary expression to avoid cross-product
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
                cost_usd=r[7] or 0.0, call_count=r[8] or 0,
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
        add_persona_clause(clauses, filters.persona)
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
        terminal_placeholders = ", ".join("?" for _ in TERMINAL_STATUSES)
        cur = self.conn.execute(
            "SELECT * FROM sessions WHERE agent_id = ? AND status IN ("
            + terminal_placeholders
            + ") ORDER BY COALESCE(ended_at, started_at) DESC LIMIT ?",
            [agent_id, *TERMINAL_STATUSES, limit],
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [_row_to_session(r, cols) for r in rows]

    def get_completed_session_count(self, agent_id: str) -> int:
        terminal_placeholders = ", ".join("?" for _ in TERMINAL_STATUSES)
        result = self.conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE agent_id = ? AND status IN ("
            + terminal_placeholders
            + ")",
            [agent_id, *TERMINAL_STATUSES],
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

    def get_daily_cost_for_agents(self, agent_ids: list[str], date: date) -> float:
        """Summed daily cost across a SET of agent_ids for one UTC calendar
        day — generalizes `get_daily_cost` for a coding-tool GROUP cap
        (e.g. every `claude-code-<project>` variant), where the ceiling
        applies to the group's combined spend, not any one member alone.
        """
        if not agent_ids:
            return 0.0
        placeholders = ", ".join(f"${i + 2}" for i in range(len(agent_ids)))
        result = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM spans "
            f"WHERE agent_id IN ({placeholders}) "
            "AND CAST(start_time AT TIME ZONE 'UTC' AS DATE) = $1",
            [date, *agent_ids],
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

    def update_span_cost(
        self, span_id: str, cost_usd: float, pricing_source: str | None = None,
    ) -> None:
        """Persist a computed span cost, optionally stamping its provenance.

        `pricing_source` is `None` for callers that only know the dollar
        figure (existing tests, any future caller that hasn't adopted
        provenance yet) — in that case the column is left untouched rather
        than overwritten with NULL, so a span's recorded provenance from an
        earlier call is never silently erased by a later cost-only update.
        """
        with self._write_lock:
            if pricing_source is not None:
                self.conn.execute(
                    "UPDATE spans SET cost_usd = $1, pricing_source = $2 WHERE span_id = $3",
                    [cost_usd, pricing_source, span_id],
                )
            else:
                self.conn.execute(
                    "UPDATE spans SET cost_usd = $1 WHERE span_id = $2",
                    [cost_usd, span_id],
                )

    def increment_session_cost(self, session_id: str, delta_usd: float) -> None:
        with self._write_lock:
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
        persona: str | None = None,
    ) -> tuple[int, int, int, int, int, float]:
        clauses = ["start_time >= $1", "start_time < $2"]
        params: list = [since, until]
        if agent_id:
            clauses.append(f"agent_id = ${len(params) + 1}")
            params.append(agent_id)
        add_persona_clause(clauses, persona)
        where = " AND ".join(clauses)
        row = self.conn.execute(
            f"SELECT COUNT(DISTINCT session_id) AS sessions, "
            f"COALESCE(SUM(input_tokens), 0)        AS in_tok, "
            f"COALESCE(SUM(output_tokens), 0)       AS out_tok, "
            f"COALESCE(SUM(cache_tokens), 0)        AS cache_tok, "
            f"COALESCE(SUM(cache_write_tokens), 0)  AS cache_write_tok, "
            f"COALESCE(SUM(cost_usd), 0.0)          AS cost "
            f"FROM spans WHERE {where}",
            params,
        ).fetchone()
        if row is None:  # COALESCE aggregate always returns a row; guard for typing
            return (0, 0, 0, 0, 0, 0.0)
        return (
            int(row[0] or 0), int(row[1] or 0), int(row[2] or 0),
            int(row[3] or 0), int(row[4] or 0), float(row[5] or 0.0),
        )
    def get_cost_delta_by_group(
        self, group_col: str, current_since: datetime, current_until: datetime,
        prev_since: datetime, prev_until: datetime, top_n: int,
        persona: str | None = None,
    ) -> list[dict]:
        # group_col is an internal, fixed identifier (never user input); the
        # allow-list keeps it that way so the interpolation below stays safe.
        if group_col not in ("agent_id", "model"):
            raise ValueError(f"Unsupported group_col {group_col!r}")
        # Parameter-free, so it interpolates into the SQL below without
        # disturbing the positional `$n` numbering that block depends on.
        persona_clause = persona_agent_clause(persona)
        persona_sql = f" AND {persona_clause}" if persona_clause else ""
        sql = f"""
            SELECT {group_col} AS grp,
                   COALESCE(SUM(CASE WHEN start_time >= $1 AND start_time < $2
                                     THEN cost_usd ELSE 0 END), 0.0) AS cur_cost,
                   COALESCE(SUM(CASE WHEN start_time >= $3 AND start_time < $4
                                     THEN cost_usd ELSE 0 END), 0.0) AS prev_cost,
                   COALESCE(SUM(CASE WHEN start_time >= $1 AND start_time < $2
                                     THEN input_tokens + output_tokens + cache_tokens
                                          + cache_write_tokens ELSE 0 END), 0) AS cur_tokens,
                   COALESCE(SUM(CASE WHEN start_time >= $3 AND start_time < $4
                                     THEN input_tokens + output_tokens + cache_tokens
                                          + cache_write_tokens ELSE 0 END), 0) AS prev_tokens
            FROM spans
            WHERE (start_time >= $3 AND start_time < $2)
              AND {group_col} IS NOT NULL{persona_sql}
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
             "delta": float(r[1]) - float(r[2]),
             "current_tokens": int(r[3] or 0), "previous_tokens": int(r[4] or 0),
             "tokens_delta": int(r[3] or 0) - int(r[4] or 0)}
            for r in rows
        ]

    def delete_spans_before(
        self,
        cutoff: datetime,
        *,
        retention_days: int | None = None,
        analysis_span_days: int | None = None,
    ) -> tuple[int, int]:
        """Delete aged-out history AND write its ledger row, ATOMICALLY.

        Returns ``(spans deleted, sessions deleted)``.

        **The ledger row is written in the SAME TRANSACTION as the deletes, and
        that is the point rather than a detail.** What this mechanism has to
        guarantee is that a delete of the user's own history is observable after
        the fact; a ledger the process can skip by dying between two commits
        delivers that only on the happy path, which is exactly the path where
        nobody needs it. This job runs from an apscheduler cron inside an ad-hoc
        ``tj serve``, so being killed mid-run is an ordinary event — and a
        completed delete with no trace is the precise failure the ledger exists
        to make impossible. Either both land or neither does.

        Three statements, in order:

        1. Aged-out spans go.
        2. Sessions the delete ORPHANED go with them. Deleting only from
           ``spans`` used to leave parent ``sessions`` rows in place forever,
           and those are not inert: ``data_span`` unions ``sessions.started_at``
           into the day set it measures the available span from, so every orphan
           went on asserting that a day carried data after that day's data was
           destroyed — the deletion skewed the measure of its own aftermath. A
           session goes only once it has NO spans left, which is strictly
           narrower than "started before the cutoff": a long session straddling
           the boundary keeps every span the cutoff spared, so deleting it would
           discard live rows' parent. A pre-cutoff session that never had spans
           goes too — it is aged-out history like any other, and leaving it
           would let it keep asserting a day beyond the retention horizon.
        3. The ledger row, with ``oldest_kept`` read after the deletes (visible
           within the transaction) so it states what survived rather than what
           was intended to.

        **Both counts come from the DELETEs themselves, not from a preceding
        ``COUNT(*)``.** DuckDB returns the affected-row count for a ``DELETE``,
        and using it is what makes the ledger's figure the number of rows this
        transaction actually removed rather than an estimate of how many it
        expected to. A separate count is wrong even inside the transaction and
        badly wrong outside it: an ingest committing an aged-out span between
        the count and the delete would have that span destroyed while the ledger
        persisted the smaller, earlier number. Deriving the figure from the
        delete makes that race structurally impossible instead of merely narrow,
        which matters because the entire purpose of this ledger is that it can
        be trusted about what was destroyed.
        """
        with self._write_lock:
            self.conn.execute("BEGIN TRANSACTION")
            try:
                span_row = self.conn.execute(
                    "DELETE FROM spans WHERE start_time < $1", [cutoff]
                ).fetchone()
                spans_deleted = int(span_row[0]) if span_row else 0
                session_row = self.conn.execute(
                    "DELETE FROM sessions s WHERE s.started_at < $1 "
                    "AND NOT EXISTS ("
                    "SELECT 1 FROM spans p WHERE p.session_id = s.session_id)",
                    [cutoff],
                ).fetchone()
                sessions_deleted = int(session_row[0]) if session_row else 0
                oldest_row = self.conn.execute(
                    "SELECT MIN(start_time) FROM spans WHERE start_time IS NOT NULL"
                ).fetchone()
                self.conn.execute(
                    "INSERT INTO retention_events (event_id, ran_at, cutoff, "
                    "retention_days, analysis_span_days, spans_deleted, "
                    "sessions_deleted, oldest_kept) "
                    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                    [
                        str(uuid.uuid4()), utcnow(), cutoff, retention_days,
                        analysis_span_days, spans_deleted, sessions_deleted,
                        oldest_row[0] if oldest_row else None,
                    ],
                )
            except Exception:
                # A ledger row that cannot be written takes the delete down with
                # it. Keeping the delete and merely logging the failure — which
                # is what this used to do — produces exactly the state the
                # ledger exists to prevent: history gone, nothing saying so.
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

        return spans_deleted, sessions_deleted

    def close(self) -> None:
        # Every cursor explicitly, then the root connection. Closing the root
        # alone does tear the cursors down, but DuckDB only evicts an
        # invalidated instance from its per-path cache once no handle to it
        # survives, and relying on GC for that makes recovery non-deterministic.
        self._teardown_connections()


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
        # `_db_path = None` marks it unrecoverable: there is no file to reopen,
        # so tearing the connection down would destroy the data, not restore it.
        self._db_path = None
        self._conn = duckdb.connect(":memory:")
        run_migrations(self._conn)
        self._local = threading.local()
        self._cursors = []
        self._generation = 0
        self._conn_lock = threading.RLock()
        # Inherited write methods take this lock (async-hooks concurrency, #124).
        self._write_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def open_db(config: StorageConfig) -> DuckDBBackend:
    """Open the database and return a backend instance."""
    return DuckDBBackend(config)
