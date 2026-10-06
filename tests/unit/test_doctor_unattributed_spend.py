"""Unit tests for doctor's unattributed spend reporting (#749)."""
from __future__ import annotations

from tokenjam.cli.cmd_doctor import _check_unattributed_spend
from tokenjam.core.db import InMemoryBackend
from tests.factories import make_llm_span, make_session


def test_unattributed_spend_check_ok_when_empty():
    db = InMemoryBackend()
    session = make_session(session_id="s1")
    db.upsert_session(session)
    db.insert_span(make_llm_span(session_id="s1", cost_usd=1.0))

    check = _check_unattributed_spend(db)
    assert check["name"] == "Unattributed spend"
    assert check["level"] == "ok"
    assert "All spend is attributed" in check["message"]


def test_unattributed_spend_check_discloses_spans_and_traces():
    db = InMemoryBackend()
    # 2 spans on 1 shared trace without session
    db.insert_span(make_llm_span(trace_id="tr-unatt-1", session_id=None, cost_usd=2.5))
    db.insert_span(make_llm_span(trace_id="tr-unatt-1", session_id=None, cost_usd=1.5))
    # 1 span on another shared trace without session
    db.insert_span(make_llm_span(trace_id="tr-unatt-2", session_id=None, cost_usd=3.0))

    check = _check_unattributed_spend(db)
    assert check["name"] == "Unattributed spend"
    assert check["level"] == "info"
    assert "$7.00" in check["message"]
    assert "3 span(s)" in check["message"]
    assert "2 trace(s)" in check["message"]
    assert "not assigned to a named session" in check["message"]
