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
    """Insert just enough to satisfy NOT NULL constraints; everything else NULL."""
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    conn.execute(
        "INSERT INTO spans VALUES "
        "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23)",
        [
            span_id, trace_id, None, "session-1",
            "test-agent", "test-span", "internal", "ok",
            None, now, now, 0,
            "{}", None, None, None,
            0, 0, 0, 0.0,
            None, None, "[]",
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
