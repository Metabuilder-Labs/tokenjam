"""`tj mcp`'s read-only fallback (`cmd_mcp.py`, when `tj serve` isn't reachable)
connects `duckdb.connect(path, read_only=True)` directly — bypassing
`DuckDBBackend.__init__`'s `run_migrations`/schema-self-heal entirely. A store
written by an older `tj` build, missing a column a newer read path depends on,
used to surface as a raw DuckDB BinderError the first time a tool touched it.
This pins the degrade path end to end against a REAL DuckDB file with a REAL
missing column — not a mocked exception."""
from __future__ import annotations

import duckdb
import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend, missing_expected_columns
from tokenjam.core.models import CostFilters
from tokenjam.mcp import server as mcp_server


@pytest.fixture
def stale_readonly_conn(tmp_path):
    """A real DuckDB file, fully migrated, then with `spans.feature`
    (migration 17 — an `EXPECTED_ADDITIVE_COLUMNS` entry, and a column
    `get_cost_summary(group_by="feature")` names directly) dropped to
    simulate a store an older `tj` build wrote before that migration existed.
    Reopened read-only, exactly like `cmd_mcp.py`'s fallback path.
    """
    db_path = tmp_path / "stale.duckdb"
    db = DuckDBBackend(StorageConfig(path=str(db_path)))
    # DuckDB refuses a plain `ALTER TABLE ... DROP COLUMN` here (the table's
    # PRIMARY KEY makes it report unspecified "entries that depend on it"),
    # so simulate the pre-migration-17 shape by rebuilding the table without
    # the column instead.
    db.conn.execute("CREATE TABLE spans_new AS SELECT * EXCLUDE (feature) FROM spans")
    db.conn.execute("DROP TABLE spans")
    db.conn.execute("ALTER TABLE spans_new RENAME TO spans")
    db.close()

    ro_conn = duckdb.connect(str(db_path), read_only=True)
    yield ro_conn
    ro_conn.close()


def test_the_dropped_column_is_flagged_by_the_read_only_safe_schema_check(stale_readonly_conn):
    """Sanity check the fixture actually represents the gap `cmd_mcp.py`
    detects up front — `missing_expected_columns` must be safe to run against
    a read-only connection (pure SELECT against information_schema)."""
    gap = missing_expected_columns(stale_readonly_conn)
    assert "spans.feature" in gap


def test_a_tool_against_the_missing_column_degrades_instead_of_raising(stale_readonly_conn):
    """Force the real failure: `get_cost_summary(group_by="feature")`
    references the dropped column directly, so this raises a genuine
    `duckdb.Error` inside `_ReadOnlyDB` — not a mock. With `init()` told
    about the gap, the @mcp.tool()-style error path must degrade to the
    clear message, never propagate the raw BinderError."""
    gap = missing_expected_columns(stale_readonly_conn)
    mcp_server.init(
        ro_conn=stale_readonly_conn,
        config=TjConfig(version="1"),
        schema_gap=gap,
    )
    try:
        ro_db = mcp_server._ro_db
        assert ro_db is not None
        try:
            ro_db.get_cost_summary(CostFilters(group_by="feature"))
            pytest.fail("expected the dropped column to raise a real duckdb error")
        except duckdb.Error as e:
            result = mcp_server._wrap_tool_error(e)
        assert "error" in result
        assert "run `tj serve` once" in result["error"] or "tj doctor --repair" in result["error"]
        assert "Binder" not in result["error"]  # the raw DuckDB message must not leak through
    finally:
        mcp_server.init(ro_conn=None, config=None, schema_gap=[])


def test_without_a_known_schema_gap_the_raw_error_still_surfaces(stale_readonly_conn):
    """The friendly message is only for the KNOWN-gap case — an unrelated
    duckdb.Error with no schema gap recorded must still pass its real message
    through (don't paper over a genuinely different fault)."""
    mcp_server.init(ro_conn=stale_readonly_conn, config=TjConfig(version="1"), schema_gap=[])
    try:
        try:
            mcp_server._ro_db.get_cost_summary(CostFilters(group_by="feature"))
            pytest.fail("expected the dropped column to raise")
        except duckdb.Error as e:
            result = mcp_server._wrap_tool_error(e)
        assert "error" in result
        assert "run `tj serve` once" not in result["error"]
    finally:
        mcp_server.init(ro_conn=None, config=None, schema_gap=[])
