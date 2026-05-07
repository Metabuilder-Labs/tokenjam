"""Synthetic tests for drift detection: baseline building and drift evaluation."""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from tj.core.config import AgentConfig, DriftConfig, TjConfig
from tj.core.db import InMemoryBackend
from tj.core.drift import DriftDetector, build_baseline, evaluate_drift
from tj.core.models import (
    AgentRecord,
    AlertType,
    DriftBaseline,
    Severity,
)
from tj.utils.ids import new_uuid
from tj.utils.time_parse import utcnow
from tests.factories import make_session, make_session_with_spans, make_tool_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _setup_agent(db, agent_id="test-agent"):
    now = utcnow()
    db.upsert_agent(AgentRecord(agent_id=agent_id, first_seen=now, last_seen=now))


def _make_config(agent_id="test-agent", baseline_sessions=10, threshold=2.0,
                 tool_seq_diff=0.4, drift_enabled=True):
    return TjConfig(
        version="1",
        agents={
            agent_id: AgentConfig(
                drift=DriftConfig(
                    enabled=drift_enabled,
                    baseline_sessions=baseline_sessions,
                    token_threshold=threshold,
                    tool_sequence_diff=tool_seq_diff,
                ),
            ),
        },
    )


def _insert_completed_sessions(db, agent_id, count, input_tokens=1000,
                                output_tokens=200, tool_call_count=10):
    """Insert N completed sessions with consistent stats."""
    sessions = []
    for _ in range(count):
        s = make_session(
            agent_id=agent_id,
            status="completed",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_call_count=tool_call_count,
            duration_seconds=60.0,
        )
        db.upsert_session(s)
        sessions.append(s)
    return sessions


# -- Baseline building --

def test_baseline_built_after_n_sessions(db):
    _setup_agent(db)
    config = _make_config(baseline_sessions=5)
    alert_engine = MagicMock()
    detector = DriftDetector(db, alert_engine, config)

    # Insert 5 completed sessions
    sessions = _insert_completed_sessions(db, "test-agent", 5)

    # Trigger on_session_end — should build baseline
    detector.on_session_end("test-agent", sessions[-1])

    baseline = db.get_baseline("test-agent")
    assert baseline is not None
    assert baseline.sessions_sampled == 5
    assert baseline.avg_input_tokens == 1000.0


def test_drift_inactive_before_baseline_complete(db):
    _setup_agent(db)
    config = _make_config(baseline_sessions=10)
    alert_engine = MagicMock()
    detector = DriftDetector(db, alert_engine, config)

    # Only 3 sessions — not enough for baseline
    sessions = _insert_completed_sessions(db, "test-agent", 3)
    detector.on_session_end("test-agent", sessions[-1])

    assert db.get_baseline("test-agent") is None
    alert_engine.fire.assert_not_called()


# -- Drift evaluation --

def test_drift_fires_on_token_explosion(db):
    _setup_agent(db)
    config = _make_config(baseline_sessions=5, threshold=2.0)
    alert_engine = MagicMock()
    detector = DriftDetector(db, alert_engine, config)

    # Build baseline from sessions with some variance around 1000 tokens
    for tokens in [900, 1000, 1100, 950, 1050]:
        s = make_session(agent_id="test-agent", status="completed",
                         input_tokens=tokens, duration_seconds=60.0)
        db.upsert_session(s)

    # Trigger baseline build (we now have 5 completed sessions)
    trigger = make_session(agent_id="test-agent", status="completed",
                           input_tokens=1000, duration_seconds=60.0)
    db.upsert_session(trigger)
    detector.on_session_end("test-agent", trigger)
    assert db.get_baseline("test-agent") is not None

    # Now a session with 10x tokens (way beyond 2 sigma)
    anomalous = make_session(
        agent_id="test-agent",
        status="completed",
        input_tokens=10000,
        output_tokens=200,
        duration_seconds=60.0,
    )
    db.upsert_session(anomalous)
    detector.on_session_end("test-agent", anomalous)

    alert_engine.fire.assert_called_once()
    call_kwargs = alert_engine.fire.call_args
    assert call_kwargs[1]["alert_type"] == AlertType.DRIFT_DETECTED
    assert call_kwargs[1]["severity"] == Severity.WARNING


def test_drift_does_not_fire_within_threshold(db):
    _setup_agent(db)
    config = _make_config(baseline_sessions=5, threshold=2.0)
    alert_engine = MagicMock()
    detector = DriftDetector(db, alert_engine, config)

    # Build baseline with some variance in input_tokens; output_tokens and
    # duration are uniform so the test focuses purely on input_token drift.
    for tokens in [900, 1000, 1100, 950, 1050]:
        s = make_session(agent_id="test-agent", status="completed",
                         input_tokens=tokens, duration_seconds=60.0)
        db.upsert_session(s)
    trigger = make_session(agent_id="test-agent", status="completed",
                           input_tokens=1000, duration_seconds=60.0)
    db.upsert_session(trigger)
    detector.on_session_end("test-agent", trigger)

    # Session with slightly above average input_tokens — within threshold.
    # output_tokens and duration match the baseline exactly (both 0 / 60s)
    # so only the input dimension is evaluated for drift.
    normal = make_session(
        agent_id="test-agent",
        status="completed",
        input_tokens=1050,
        duration_seconds=60.0,
    )
    db.upsert_session(normal)
    detector.on_session_end("test-agent", normal)

    alert_engine.fire.assert_not_called()


def test_drift_tool_sequence_fires_on_different_tools(db):
    _setup_agent(db)

    # Build baseline with tool spans
    baseline_sessions = []
    for _ in range(5):
        sid = new_uuid()
        s = make_session(agent_id="test-agent", session_id=sid, status="completed")
        db.upsert_session(s)
        # Insert tool spans for this session
        for tool_name in ["search", "summarize", "answer"]:
            span = make_tool_span(agent_id="test-agent", tool_name=tool_name)
            # Override session_id
            span.session_id = sid
            db.insert_span(span)
        baseline_sessions.append(s)

    baseline = build_baseline("test-agent", baseline_sessions, db)
    db.upsert_baseline(baseline)

    # New session with completely different tools
    new_sid = new_uuid()
    new_session = make_session(agent_id="test-agent", session_id=new_sid, status="completed",
                               input_tokens=1000, output_tokens=200)
    db.upsert_session(new_session)
    for tool_name in ["delete_all", "format_disk", "shutdown"]:
        span = make_tool_span(agent_id="test-agent", tool_name=tool_name)
        span.session_id = new_sid
        db.insert_span(span)

    result = evaluate_drift(
        session=new_session,
        baseline=baseline,
        config_threshold=2.0,
        sequence_diff_threshold=0.4,
        db=db,
    )

    tool_violations = [v for v in result.violations if v.dimension == "tool_sequence"]
    assert len(tool_violations) == 1
    assert result.drifted


def test_drift_tool_sequence_passes_on_similar_tools(db):
    _setup_agent(db)

    baseline_sessions = []
    for _ in range(5):
        sid = new_uuid()
        s = make_session(agent_id="test-agent", session_id=sid, status="completed")
        db.upsert_session(s)
        for tool_name in ["search", "summarize", "answer"]:
            span = make_tool_span(agent_id="test-agent", tool_name=tool_name)
            span.session_id = sid
            db.insert_span(span)
        baseline_sessions.append(s)

    baseline = build_baseline("test-agent", baseline_sessions, db)
    db.upsert_baseline(baseline)

    # New session with the same tools
    new_sid = new_uuid()
    new_session = make_session(agent_id="test-agent", session_id=new_sid, status="completed",
                               input_tokens=1000, output_tokens=200)
    db.upsert_session(new_session)
    for tool_name in ["search", "summarize", "answer"]:
        span = make_tool_span(agent_id="test-agent", tool_name=tool_name)
        span.session_id = new_sid
        db.insert_span(span)

    result = evaluate_drift(
        session=new_session,
        baseline=baseline,
        config_threshold=2.0,
        sequence_diff_threshold=0.4,
        db=db,
    )

    tool_violations = [v for v in result.violations if v.dimension == "tool_sequence"]
    assert len(tool_violations) == 0


def test_rebuild_baseline_replaces_existing(db):
    _setup_agent(db)

    # Build initial baseline
    sessions = _insert_completed_sessions(db, "test-agent", 5, input_tokens=1000)
    baseline1 = build_baseline("test-agent", sessions, db)
    db.upsert_baseline(baseline1)

    # Build new baseline with different data
    sessions2 = _insert_completed_sessions(db, "test-agent", 5, input_tokens=5000)
    baseline2 = build_baseline("test-agent", sessions2, db)
    db.upsert_baseline(baseline2)

    result = db.get_baseline("test-agent")
    assert result is not None
    assert result.avg_input_tokens == 5000.0


def test_drift_disabled_for_agent(db):
    _setup_agent(db)
    config = _make_config(drift_enabled=False)
    alert_engine = MagicMock()
    detector = DriftDetector(db, alert_engine, config)

    sessions = _insert_completed_sessions(db, "test-agent", 10)
    detector.on_session_end("test-agent", sessions[-1])

    assert db.get_baseline("test-agent") is None
    alert_engine.fire.assert_not_called()


def test_drift_unknown_agent_is_noop(db):
    config = _make_config(agent_id="other-agent")
    alert_engine = MagicMock()
    detector = DriftDetector(db, alert_engine, config)

    session = make_session(agent_id="unknown-agent", status="completed")
    detector.on_session_end("unknown-agent", session)

    alert_engine.fire.assert_not_called()
