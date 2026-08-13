"""`tj status`'s ingest-freshness note (`_ingest_freshness_note`, cmd_status.py).

Purely advisory (unlike `tj doctor`'s "Corpus freshness" check, which shares
the same threshold math via `core/ingest_freshness.py` but FAILS on it) —
`tj status` never exits non-zero on a stale corpus, since a quiet corpus is
also the everyday shape of a user who just hasn't run an agent lately.
Three-state gating (root anti-pattern 22): silent with no config/DB, silent
when known-and-fresh, a line only when known-and-stale.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.cli.cmd_status import _ingest_freshness_note
from tokenjam.core.config import IngestConfig, StorageConfig, TjConfig
from tokenjam.core.db import DuckDBBackend
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_session


@pytest.fixture()
def db():
    backend = DuckDBBackend(StorageConfig(path=":memory:"))
    yield backend
    backend.close()


def _config() -> TjConfig:
    config = TjConfig(version="1")
    config.ingest = IngestConfig(interval_minutes=30)
    return config


def test_silent_with_no_config() -> None:
    class _NoConn:
        pass

    assert _ingest_freshness_note(None, _NoConn()) is None


def test_silent_when_no_sessions_ever(db: DuckDBBackend) -> None:
    assert _ingest_freshness_note(_config(), db) is None


def test_silent_when_recently_ingested(db: DuckDBBackend) -> None:
    db.upsert_session(make_session(started_at=utcnow() - timedelta(minutes=5)))
    assert _ingest_freshness_note(_config(), db) is None


def test_fires_when_stale(db: DuckDBBackend) -> None:
    db.upsert_session(make_session(started_at=utcnow() - timedelta(days=2)))
    note = _ingest_freshness_note(_config(), db)
    assert note is not None
    assert "tj doctor" in note


def test_never_raises_on_a_broken_db(monkeypatch) -> None:
    class _Boom:
        @property
        def conn(self):
            raise RuntimeError("disk on fire")

    assert _ingest_freshness_note(_config(), _Boom()) is None
