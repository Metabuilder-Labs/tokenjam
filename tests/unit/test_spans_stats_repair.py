"""Unit tests for the spans column-statistics corruption check + repair.

The actual DuckDB v1.5.x bug (#56) is hard to reproduce synthetically — it
arises from a specific bulk-write pattern that corrupts per-row-group min/max
stats. These tests focus on the contract instead:

  * `check_spans_stats_corruption` returns False on a healthy table.
  * `check_spans_stats_corruption` returns False when the spans table is
    empty (nothing to check, so nothing to fix).
  * `repair_spans_stats` is idempotent and preserves all rows.
"""
from __future__ import annotations

import duckdb
import pytest

from tokenjam.core.db import (
    check_spans_stats_corruption,
    repair_spans_stats,
    run_migrations,
)


@pytest.fixture
def conn(tmp_path):
    """Fresh on-disk DuckDB with the standard migrations applied."""
    path = tmp_path / "test.duckdb"
    c = duckdb.connect(str(path))
    run_migrations(c)
    yield c
    c.close()


def _insert_minimal_span(conn, *, trace_id: str, span_id: str) -> None:
    """Insert just enough to satisfy NOT NULL constraints; everything else NULL.

    Named columns, deliberately: a positional ``INSERT INTO spans VALUES (...)``
    pins the placeholder count to the column count, so every additive migration
    broke this helper (and with it every test in this file) for a reason that
    has nothing to do with what any of them assert. Naming only the NOT NULL
    columns lets each new nullable column default to NULL on its own.
    """
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    conn.execute(
        "INSERT INTO spans ("
        "span_id, trace_id, session_id, agent_id, name, kind, status_code, "
        "start_time, end_time, duration_ms, attributes, input_tokens, "
        "output_tokens, cache_tokens, cost_usd, events"
        ") VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)",
        [
            span_id, trace_id, "session-1", "test-agent", "test-span",
            "internal", "ok", now, now, 0, "{}", 0, 0, 0, 0.0, "[]",
        ],
    )


class TestCheckSpansStatsCorruption:
    def test_empty_table_returns_false(self, conn):
        """No rows to check → no corruption → False (don't flag healthy emptiness)."""
        assert check_spans_stats_corruption(conn) is False

    def test_healthy_table_returns_false(self, conn):
        for i in range(5):
            _insert_minimal_span(conn, trace_id=f"trace{i:02d}", span_id=f"span{i:02d}")
        assert check_spans_stats_corruption(conn) is False

    def test_missing_table_returns_false(self, tmp_path):
        """If the spans table doesn't exist (pre-migration), don't blow up."""
        c = duckdb.connect(str(tmp_path / "fresh.duckdb"))
        try:
            assert check_spans_stats_corruption(c) is False
        finally:
            c.close()


class TestRepairSpansStats:
    def test_preserves_all_rows(self, conn):
        for i in range(20):
            _insert_minimal_span(conn, trace_id=f"trace{i:02d}", span_id=f"span{i:02d}")
        before = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        assert before == 20
        repair_spans_stats(conn)
        after = conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        assert after == 20

    def test_preserves_span_data_exactly(self, conn):
        _insert_minimal_span(conn, trace_id="abc123", span_id="span1")
        before = conn.execute("SELECT trace_id, span_id, agent_id FROM spans").fetchone()
        repair_spans_stats(conn)
        after = conn.execute("SELECT trace_id, span_id, agent_id FROM spans").fetchone()
        assert before == after

    def test_idempotent_on_empty_table(self, conn):
        """Running repair on a freshly-migrated empty table must not error."""
        repair_spans_stats(conn)
        # Table still exists and is queryable.
        assert conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 0

    def test_idempotent_when_called_twice(self, conn):
        for i in range(3):
            _insert_minimal_span(conn, trace_id=f"trace{i}", span_id=f"span{i}")
        repair_spans_stats(conn)
        repair_spans_stats(conn)
        assert conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 3


class TestRepairPreservesSchema:
    """Regression for #38: the repair must not strip the spans table's
    PRIMARY KEY, NOT NULL constraints, or secondary indexes."""

    @staticmethod
    def _constraints(conn) -> list[str]:
        return [
            r[0]
            for r in conn.execute(
                "SELECT constraint_type FROM duckdb_constraints() "
                "WHERE table_name = 'spans'"
            ).fetchall()
        ]

    @staticmethod
    def _index_names(conn) -> set[str]:
        return {
            r[0]
            for r in conn.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'spans'"
            ).fetchall()
        }

    def test_primary_key_survives_repair(self, conn):
        _insert_minimal_span(conn, trace_id="t1", span_id="s1")
        repair_spans_stats(conn)
        assert "PRIMARY KEY" in self._constraints(conn)

    def test_not_null_constraints_survive_repair(self, conn):
        _insert_minimal_span(conn, trace_id="t1", span_id="s1")
        repair_spans_stats(conn)
        assert "NOT NULL" in self._constraints(conn)

    def test_secondary_indexes_survive_repair(self, conn):
        _insert_minimal_span(conn, trace_id="t1", span_id="s1")
        repair_spans_stats(conn)
        expected = {
            "idx_spans_trace_id",
            "idx_spans_agent_id",
            "idx_spans_start_time",
            "idx_spans_tool_name",
            "idx_spans_conv_id",
        }
        assert expected <= self._index_names(conn)

    def test_duplicate_span_id_rejected_after_repair(self, conn):
        _insert_minimal_span(conn, trace_id="t1", span_id="dup")
        repair_spans_stats(conn)
        with pytest.raises(duckdb.ConstraintException):
            _insert_minimal_span(conn, trace_id="t2", span_id="dup")

    def test_schema_identical_to_freshly_migrated(self, conn, tmp_path):
        """After repair, the spans columns match a freshly-migrated table."""
        _insert_minimal_span(conn, trace_id="t1", span_id="s1")
        repair_spans_stats(conn)
        repaired = conn.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_name = 'spans' ORDER BY ordinal_position"
        ).fetchall()

        fresh = duckdb.connect(str(tmp_path / "fresh.duckdb"))
        try:
            run_migrations(fresh)
            expected = fresh.execute(
                "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                "WHERE table_name = 'spans' ORDER BY ordinal_position"
            ).fetchall()
        finally:
            fresh.close()
        assert repaired == expected
