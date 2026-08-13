"""`core/ingest_freshness.py` is the single source of truth `tj doctor` and
`tj status` both read the corpus-staleness threshold from -- pinned here so
neither surface can silently drift from the other on what "stale" means.
"""
from __future__ import annotations

from datetime import timedelta

from tokenjam.core.config import StorageConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.core.ingest_freshness import corpus_freshness, staleness_threshold_hours
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_session


def test_threshold_scales_with_the_configured_interval() -> None:
    assert staleness_threshold_hours(30) > staleness_threshold_hours(15)


def test_threshold_never_drops_below_the_floor_for_a_tiny_interval() -> None:
    from tokenjam.core.ingest_freshness import STALENESS_FLOOR_HOURS

    assert staleness_threshold_hours(1) == STALENESS_FLOOR_HOURS


def test_no_sessions_is_not_stale() -> None:
    db = DuckDBBackend(StorageConfig(path=":memory:"))
    try:
        result = corpus_freshness(db.conn, 30, now=utcnow())
    finally:
        db.close()
    assert result.newest_session_at is None
    assert result.is_stale is False


def test_a_recent_session_is_not_stale() -> None:
    db = DuckDBBackend(StorageConfig(path=":memory:"))
    try:
        db.upsert_session(make_session(started_at=utcnow() - timedelta(minutes=5)))
        result = corpus_freshness(db.conn, 30, now=utcnow())
    finally:
        db.close()
    assert result.is_stale is False


def test_a_session_far_past_the_cadence_is_stale() -> None:
    db = DuckDBBackend(StorageConfig(path=":memory:"))
    try:
        now = utcnow()
        db.upsert_session(make_session(started_at=now - timedelta(days=2)))
        result = corpus_freshness(db.conn, 30, now=now)
    finally:
        db.close()
    assert result.is_stale is True
    assert result.age_hours is not None and result.age_hours > 47
