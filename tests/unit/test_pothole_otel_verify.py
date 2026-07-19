"""Verify, OTel lane: recurrence measured off spans against a loop fix marker.

Covers before/after exposure counting, the honest verdicts (improved /
regressed / insufficient_data), the distilled-family "can't re-match this"
admission, and the write-back into the loop's own expectation ledger.

All spans go through tests/factories (Critical Rule 8). The backend is in-memory
and the loop tables are created on it: nothing reads the real ~/.tj or ~/.claude.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.loop import create_expectation, list_expectation_runs
from tokenjam.core.optimize.pothole_otel_verify import (
    measure_span_recurrence,
    verify_otel_expectation,
)
from tokenjam.core.optimize.pothole_verify import (
    VERDICT_IMPROVED,
    VERDICT_INSUFFICIENT_DATA,
    VERDICT_REGRESSED,
)
from tests.factories import make_tool_span

MARKER = datetime(2026, 5, 15, tzinfo=timezone.utc)
SIGNATURE = "http_call:connectionreseterror: peer closed"


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _seed(db, *, session_id, offset_minutes, message="ConnectionResetError: peer closed",
          agent_id="billing-svc", tool="http_call"):
    span = make_tool_span(
        agent_id=agent_id, tool_name=tool, status="error",
        session_id=session_id, start_time=MARKER + timedelta(minutes=offset_minutes),
    )
    span.status_message = message
    db.insert_span(span)


def _seed_window(db, *, prefix, count, sign, per_session_failures=1):
    """`count` sessions on one side of the marker (sign -1 before, +1 after)."""
    for s in range(count):
        for k in range(per_session_failures):
            _seed(db, session_id=f"{prefix}{s}",
                  offset_minutes=sign * (60 * (s + 1) + k))


# -- Exposure counting -------------------------------------------------------

def test_counts_occurrences_and_sessions_either_side_of_the_marker(db):
    _seed_window(db, prefix="pre", count=3, sign=-1)
    _seed_window(db, prefix="post", count=2, sign=+1)

    m = measure_span_recurrence(db.conn, signature=SIGNATURE, at=MARKER)

    assert m["measurable"] is True
    assert m["pre_sessions"] == 3
    assert m["pre_occurrences"] == 3
    assert m["post_sessions"] == 2
    assert m["post_occurrences"] == 2


def test_non_matching_signature_counts_exposure_but_no_occurrences(db):
    _seed_window(db, prefix="pre", count=3, sign=-1)

    m = measure_span_recurrence(
        db.conn, signature="http_call:something else entirely", at=MARKER,
    )

    assert m["pre_sessions"] == 3        # sessions are exposure, regardless of match
    assert m["pre_occurrences"] == 0


def test_agent_filter_scopes_the_measurement(db):
    _seed_window(db, prefix="pre", count=3, sign=-1)
    _seed(db, session_id="other", offset_minutes=-30, agent_id="search-svc")

    scoped = measure_span_recurrence(
        db.conn, signature=SIGNATURE, agent_id="search-svc", at=MARKER,
    )

    assert scoped["pre_sessions"] == 1


def test_distilled_family_is_admitted_unmeasurable(db):
    _seed_window(db, prefix="pre", count=3, sign=-1)

    m = measure_span_recurrence(
        db.conn, family_key="distilled:flaky_upstream", at=MARKER,
    )

    assert m["measurable"] is False
    assert "distilled" in (m["reason"] or "")


def test_never_raises_without_a_connection():
    m = measure_span_recurrence(None, signature=SIGNATURE, at=MARKER)
    assert m["measurable"] is False


# -- Verdicts against an expectation marker ----------------------------------

def _expectation(db, agent_id="billing-svc"):
    """An expectation whose created_at IS the fix marker (pinned to the fixture
    clock so the before/after split is deterministic)."""
    from tokenjam.core.loop import get_expectation

    exp = create_expectation(
        db, name="peer-closed retries", agent_id=agent_id,
        description="fix deployed", origin_session_id="pre0",
    )
    db.conn.execute(
        "UPDATE expectations SET created_at = $1 WHERE expectation_id = $2",
        [MARKER, exp.expectation_id],
    )
    return get_expectation(db, exp.expectation_id)


def test_recurrence_drop_reads_as_improved_and_records_a_pass(db):
    # Heavy before (2 per session over 5 sessions), silent after.
    _seed_window(db, prefix="pre", count=5, sign=-1, per_session_failures=2)
    _seed_window(db, prefix="post", count=6, sign=+1, per_session_failures=0)
    # Post sessions need exposure without occurrences: seed a benign span each.
    for s in range(6):
        span = make_tool_span(
            agent_id="billing-svc", tool_name="http_call", status="error",
            session_id=f"post{s}", start_time=MARKER + timedelta(hours=s + 1),
        )
        span.status_message = "TLS handshake failed"     # a DIFFERENT signature
        db.insert_span(span)

    exp = _expectation(db)
    result = verify_otel_expectation(db, exp, signature=SIGNATURE)

    assert result["verdict"] == VERDICT_IMPROVED
    assert result["realized_tokens_saved"] > 0
    runs = list_expectation_runs(db, exp.expectation_id)
    assert [r.outcome for r in runs] == ["pass"]


def test_unchanged_recurrence_reads_as_regressed_and_records_a_regress(db):
    _seed_window(db, prefix="pre", count=5, sign=-1)
    _seed_window(db, prefix="post", count=6, sign=+1)

    exp = _expectation(db)
    result = verify_otel_expectation(db, exp, signature=SIGNATURE)

    assert result["verdict"] == VERDICT_REGRESSED
    runs = list_expectation_runs(db, exp.expectation_id)
    assert [r.outcome for r in runs] == ["regress"]


def test_thin_post_window_is_insufficient_data_not_a_verdict(db):
    _seed_window(db, prefix="pre", count=5, sign=-1)
    _seed_window(db, prefix="post", count=2, sign=+1)   # under the min gate

    exp = _expectation(db)
    result = verify_otel_expectation(db, exp, signature=SIGNATURE)

    assert result["verdict"] == VERDICT_INSUFFICIENT_DATA
    # Nothing decisive was written to the ledger.
    assert list_expectation_runs(db, exp.expectation_id) == []


def test_record_false_measures_without_writing_to_the_ledger(db):
    _seed_window(db, prefix="pre", count=5, sign=-1, per_session_failures=2)
    _seed_window(db, prefix="post", count=6, sign=+1)

    exp = _expectation(db)
    result = verify_otel_expectation(db, exp, signature=SIGNATURE, record=False)

    assert result["verdict"]
    assert list_expectation_runs(db, exp.expectation_id) == []


def test_result_carries_the_correlational_basis_and_marker(db):
    _seed_window(db, prefix="pre", count=5, sign=-1)
    _seed_window(db, prefix="post", count=6, sign=+1)

    exp = _expectation(db)
    result = verify_otel_expectation(db, exp, signature=SIGNATURE)

    assert "correlation" in result["estimate_basis"]
    assert result["fix_marker_at"].startswith("2026-05-15")
