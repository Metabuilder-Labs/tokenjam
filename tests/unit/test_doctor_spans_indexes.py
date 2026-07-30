"""`tj doctor` must FAIL on a damaged secondary index, not merely mention it.

The reason this is an error rather than a warning is a write path, not a read
one. Retention deletes user history with `DELETE FROM spans WHERE start_time <
?`, and DuckDB maintains every secondary index inside that transaction — so a
damaged one can abort the statement part-way and leave it reporting fewer rows
removed than it matched. The extent of a deletion of the user's own history then
becomes unknowable after the fact, which is worse than the deletion itself.

The column-statistics check next door is structurally blind to this: it asks one
question about `trace_id`'s row-group statistics and answers OK on a spans table
whose indexes have been destroyed outright. The test below pins exactly that
asymmetry, because "doctor said OK before and after" is what let the fault sit.
"""
from __future__ import annotations

import duckdb
import pytest
from click.testing import CliRunner

from tokenjam.cli.cmd_doctor import _check_spans_indexes, _check_spans_stats
from tokenjam.core.config import StorageConfig
from tokenjam.core.db import SPANS_INDEXES, DuckDBBackend
from tests.factories import make_llm_span


@pytest.fixture
def backend(tmp_path):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "telemetry.duckdb")))
    db.insert_span(make_llm_span(agent_id="a", session_id="s1"))
    yield db
    db.close()


def test_a_healthy_store_passes(backend):
    check = _check_spans_indexes(backend)
    assert check["level"] == "ok"
    assert "repair_action" not in check


def test_a_damaged_index_is_an_error_with_a_repair_path(backend):
    backend.conn.execute("DROP INDEX idx_spans_start_time")
    check = _check_spans_indexes(backend)
    assert check["level"] == "error"
    assert check["repair_action"] == "rebuild_spans_indexes"
    # Names the index, so the message says which query paths are unreliable.
    assert "idx_spans_start_time" in check["message"]
    # And names the consequence the user cares about, not just the symptom.
    assert "delete" in check["message"].lower()


def test_the_column_statistics_check_cannot_see_this_fault(backend):
    """Why the new check had to exist at all rather than extending the old one."""
    for index_name, _ in SPANS_INDEXES:
        backend.conn.execute(f"DROP INDEX {index_name}")
    assert _check_spans_stats(backend)["level"] == "ok"
    assert _check_spans_indexes(backend)["level"] == "error"


def test_doctor_exits_nonzero_when_an_index_is_damaged(tmp_path, monkeypatch):
    """End to end: the check has to reach the exit code, not just the report."""
    from tokenjam.cli.main import cli

    db_path = tmp_path / "telemetry.duckdb"
    db = DuckDBBackend(StorageConfig(path=str(db_path)))
    db.insert_span(make_llm_span(agent_id="a", session_id="s1"))
    db.conn.execute("DROP INDEX idx_spans_agent_id")

    monkeypatch.setattr("tokenjam.cli.main.open_db", lambda *a, **k: db)
    try:
        result = CliRunner().invoke(cli, ["doctor", "--json"])
    finally:
        db.close()

    assert result.exit_code == 2, result.output
    assert "idx_spans_agent_id" in result.output


def test_repair_puts_the_index_back_and_doctor_goes_green(tmp_path):
    from tokenjam.cli.cmd_doctor import _attempt_repairs

    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "telemetry.duckdb")))
    try:
        db.insert_span(make_llm_span(agent_id="a", session_id="s1"))
        db.conn.execute("DROP INDEX idx_spans_tool_name")

        check = _check_spans_indexes(db)
        assert check["level"] == "error"
        _attempt_repairs([check], db, output_json=True)

        assert _check_spans_indexes(db)["level"] == "ok"
        # The repair rebuilds indexes from the table; it must move no rows.
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 1
    finally:
        db.close()


def test_a_daemon_held_database_is_skipped_not_failed():
    """Through the read-only HTTP shim there is no connection to probe.

    An error there would be a claim about a database this process cannot see.
    """
    class NoConn:
        conn = None

    check = _check_spans_indexes(NoConn())
    assert check["level"] == "info"
    assert "repair_action" not in check


def test_a_pre_migration_database_reports_nothing(tmp_path):
    from tokenjam.core.db import check_spans_index_corruption

    conn = duckdb.connect(str(tmp_path / "fresh.duckdb"))
    try:
        assert check_spans_index_corruption(conn) == []
    finally:
        conn.close()
