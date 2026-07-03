"""Regression tests for schema self-heal on a recorded-but-unlanded migration (#55).

`run_migrations` keys purely on the version INTEGER in `schema_migrations`. If a
version was recorded-applied under an older or renumbered definition, the current
SQL for that version never re-runs and its `ADD COLUMN` silently never lands. The
live DB then has the migration marked applied but the column absent, so every
ingest that writes that column hits a DuckDB Binder Error and is silently dropped
— surfacing to the user as a blank/stale Status page.

These tests simulate exactly that state (record version 7 as applied, then drop
the columns it adds) and assert that:
  * `missing_expected_columns` detects the gap,
  * `run_migrations` self-heals it on next open despite the version being recorded,
  * `ensure_expected_columns` is idempotent, and
  * a real span carrying `request_params` ingests successfully after the heal
    instead of being dropped on a Binder Error.
"""
from __future__ import annotations

import duckdb
import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import (
    DuckDBBackend,
    EXPECTED_ADDITIVE_COLUMNS,
    ensure_expected_columns,
    missing_expected_columns,
    run_migrations,
)
from tokenjam.core.ingest import IngestPipeline

from tests.factories import make_llm_span


@pytest.fixture
def conn(tmp_path):
    """Fresh on-disk DuckDB with the standard migrations applied."""
    c = duckdb.connect(str(tmp_path / "test.duckdb"))
    run_migrations(c)
    yield c
    c.close()


def _drop_all_indexes(conn) -> None:
    """DuckDB refuses to ALTER a table that has dependent indexes, so tests that
    drop columns to simulate schema drift must drop the indexes first. The
    indexes are not the subject of these tests."""
    for idx in (
        "idx_spans_trace_id", "idx_spans_agent_id", "idx_spans_start_time",
        "idx_spans_tool_name", "idx_spans_conv_id",
        "idx_sessions_agent_id", "idx_sessions_conv_id",
    ):
        conn.execute(f"DROP INDEX IF EXISTS {idx}")


def _simulate_unlanded_migration_7(conn) -> None:
    """Recreate the #55 failure state: migration 7 recorded applied, columns gone.

    Drops the two columns migration 7 adds while leaving version 7 in
    `schema_migrations`, exactly as a renumbered/recorded-under-older-definition
    version would leave the live DB.
    """
    _drop_all_indexes(conn)
    conn.execute("ALTER TABLE spans DROP COLUMN request_params")
    conn.execute("ALTER TABLE spans DROP COLUMN request_tools")
    # Migration 7 is already recorded applied by run_migrations; make the intent
    # explicit and independent of migration numbering.
    conn.execute("DELETE FROM schema_migrations WHERE version = 7")
    conn.execute("INSERT INTO schema_migrations VALUES (7, now())")


def _spans_columns(conn) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'spans'"
        ).fetchall()
    }


class TestMissingExpectedColumns:
    def test_healthy_db_reports_nothing_missing(self, conn):
        assert missing_expected_columns(conn) == []

    def test_detects_dropped_migration_7_columns(self, conn):
        _simulate_unlanded_migration_7(conn)
        missing = missing_expected_columns(conn)
        assert "spans.request_params" in missing
        assert "spans.request_tools" in missing

    def test_pre_migration_empty_db_reports_nothing(self, tmp_path):
        """A DB with no tables yet must not report the whole expected set."""
        c = duckdb.connect(str(tmp_path / "empty.duckdb"))
        try:
            assert missing_expected_columns(c) == []
        finally:
            c.close()


class TestEnsureExpectedColumns:
    def test_heals_dropped_columns(self, conn):
        _simulate_unlanded_migration_7(conn)
        added = ensure_expected_columns(conn)
        assert set(added) == {"spans.request_params", "spans.request_tools"}
        assert {"request_params", "request_tools"} <= _spans_columns(conn)

    def test_idempotent_on_healthy_db(self, conn):
        assert ensure_expected_columns(conn) == []

    def test_idempotent_when_called_twice(self, conn):
        _simulate_unlanded_migration_7(conn)
        ensure_expected_columns(conn)
        assert ensure_expected_columns(conn) == []

    def test_every_expected_column_is_additive(self, conn):
        """Guard: each spec column must ADD cleanly onto a table it belongs to.

        Drop them all, then heal — proving each column_def is a valid,
        idempotent `ADD COLUMN` (nullable or DEFAULTed, no NOT NULL surprises).
        """
        _drop_all_indexes(conn)
        for table, column, _ in EXPECTED_ADDITIVE_COLUMNS:
            conn.execute(f'ALTER TABLE {table} DROP COLUMN IF EXISTS "{column}"')
        added = ensure_expected_columns(conn)
        assert set(added) == {f"{t}.{c}" for t, c, _ in EXPECTED_ADDITIVE_COLUMNS}
        assert missing_expected_columns(conn) == []


class TestRunMigrationsSelfHeals:
    def test_recorded_but_unlanded_migration_reruns_on_open(self, conn):
        """The core #55 contract: version 7 recorded applied, columns dropped ->
        the NEXT run_migrations (i.e. next open) restores them without touching
        the recorded version set."""
        _simulate_unlanded_migration_7(conn)
        assert "request_params" not in _spans_columns(conn)

        # Re-running migrations must reconcile despite version 7 being recorded.
        run_migrations(conn)
        assert {"request_params", "request_tools"} <= _spans_columns(conn)


class TestIngestNoLongerDroppedOnMissingColumn:
    """End-to-end: a token-bearing span survives ingest after self-heal, where
    the recorded-but-unlanded state would previously drop it on a Binder Error."""

    def test_backend_self_heals_on_open_and_ingest_succeeds(self, tmp_path):
        db_path = tmp_path / "telemetry.duckdb"

        # 1. Open once to migrate, then corrupt into the #55 state and close.
        first = DuckDBBackend(StorageConfig(path=str(db_path)))
        _simulate_unlanded_migration_7(first.conn)
        assert "request_params" not in _spans_columns(first.conn)
        first.close()

        # 2. Reopen: DuckDBBackend.__init__ runs migrations -> self-heals.
        db = DuckDBBackend(StorageConfig(path=str(db_path)))
        try:
            assert {"request_params", "request_tools"} <= _spans_columns(db.conn)

            # 3. A span carrying request_params (the exact payload that tripped the
            # Binder Error) now ingests and is queryable — not silently dropped.
            span = make_llm_span(
                request_params={"temperature": 0.7, "max_tokens": 1024},
                request_tools={"tools": [{"name": "search"}]},
            )
            IngestPipeline(db=db, config=TjConfig(version="1")).process(span)

            row = db.conn.execute(
                "SELECT request_params FROM spans WHERE span_id = $1",
                [span.span_id],
            ).fetchone()
            assert row is not None
            assert row[0] is not None
        finally:
            db.close()

    def test_missing_column_would_break_ingest_without_heal(self, conn):
        """Sanity: prove the failure is real — a raw insert of the request_params
        column against the un-healed schema raises a Binder Error."""
        _simulate_unlanded_migration_7(conn)
        with pytest.raises(duckdb.Error):
            conn.execute(
                "INSERT INTO spans (span_id, trace_id, name, kind, status_code, "
                "start_time, request_params) VALUES "
                "($1, $2, $3, $4, $5, now(), $6)",
                ["s1", "t1", "n", "internal", "ok", "{}"],
            )
