"""Unit tests for small db.py query helpers."""
from __future__ import annotations

import pytest

from tokenjam.core.db import InMemoryBackend, session_active_seconds
from tests.factories import make_llm_span, make_session


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def test_active_seconds_sums_span_durations(db):
    sess = make_session(agent_id="a1", session_id="s1", status="completed")
    db.upsert_session(sess)
    for ms in (1000.0, 2500.0, 500.0):
        sp = make_llm_span(agent_id="a1", duration_ms=ms)
        sp.session_id = "s1"
        db.insert_span(sp)
    # 4000 ms total → 4.0 s of active (compute) time.
    assert session_active_seconds(db.conn, "s1") == pytest.approx(4.0)


def test_active_seconds_none_when_no_spans(db):
    sess = make_session(agent_id="a1", session_id="s-empty", status="completed")
    db.upsert_session(sess)
    assert session_active_seconds(db.conn, "s-empty") is None


def test_active_seconds_none_for_unknown_session(db):
    assert session_active_seconds(db.conn, "does-not-exist") is None
