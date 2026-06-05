"""Migration 8: repair ended_at on already-closed sessions.

A prior bug advanced ended_at to the close time. The backfill recomputes
ended_at from each closed session's actual spans (max end/start time) and only
LOWERS it, never touching sessions already consistent with their spans.
"""
from __future__ import annotations

from datetime import timedelta

from tokenjam.core.config import StorageConfig
from tokenjam.core.db import MIGRATIONS, DuckDBBackend
from tokenjam.utils.time_parse import utcnow

from tests.factories import make_session


def _migration_8_sql() -> str:
    # The ended_at-repair migration, selected by content so it survives any
    # renumbering (it moved from 8 -> 9 when the cache_write migration landed).
    return next(sql for _v, sql in MIGRATIONS if "SET ended_at = sub.max_ts" in sql)


def _insert_span(db, session_id: str, start, end) -> None:
    db.conn.execute(
        "INSERT INTO spans (span_id, trace_id, session_id, name, kind, "
        "status_code, start_time, end_time) "
        "VALUES (?, ?, ?, 'n', 'INTERNAL', 'OK', ?, ?)",
        [f"sp-{session_id}", f"tr-{session_id}", session_id, start, end],
    )


def test_backfill_lowers_bumped_ended_at(tmp_path):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    try:
        now = utcnow()
        last_span = now - timedelta(days=2)
        bumped = now  # the wrong "close time" ended_at
        db.upsert_session(make_session(
            agent_id="cc", session_id="closed1", status="closed",
            ended_at=bumped))
        _insert_span(db, "closed1", last_span - timedelta(seconds=5), last_span)

        db.conn.execute(_migration_8_sql())

        ended = db.get_session("closed1").ended_at
        assert ended == last_span          # lowered to real last activity
        assert ended < bumped
    finally:
        db.close()


def test_backfill_leaves_consistent_session_untouched(tmp_path):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    try:
        now = utcnow()
        last_span = now - timedelta(hours=1)
        db.upsert_session(make_session(
            agent_id="cc", session_id="ok1", status="closed",
            ended_at=last_span))
        _insert_span(db, "ok1", last_span - timedelta(seconds=5), last_span)

        db.conn.execute(_migration_8_sql())

        assert db.get_session("ok1").ended_at == last_span
    finally:
        db.close()


def test_backfill_ignores_active_sessions(tmp_path):
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    try:
        now = utcnow()
        bumped = now
        db.upsert_session(make_session(
            agent_id="cc", session_id="live1", status="active",
            ended_at=bumped))
        _insert_span(db, "live1", now - timedelta(days=2, seconds=5),
                     now - timedelta(days=2))

        db.conn.execute(_migration_8_sql())

        # status != 'closed' → untouched.
        assert db.get_session("live1").ended_at == bumped
    finally:
        db.close()
