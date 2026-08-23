"""Restart-simulation tests for AlertEngine cooldown/dedup hydration (#592).

CooldownTracker and AlertEngine._failure_rate_fired are process-local and
start empty on every AlertEngine construction. These tests construct a
SECOND AlertEngine against the SAME InMemoryBackend to simulate a daemon
restart mid-cooldown / mid-session, and assert the still-true condition does
not insert or dispatch a duplicate alert.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

from tokenjam.core.alerts import AlertEngine, CooldownTracker
from tokenjam.core.config import (
    AgentConfig,
    AlertChannelConfig,
    AlertsConfig,
    SensitiveAction,
    TjConfig,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.models import Alert, AlertFilters, AlertType, Severity
from tokenjam.utils.ids import new_uuid
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_tool_span


def _make_alert(
    agent_id: str,
    alert_type: AlertType,
    fired_at,
    suppressed: bool = False,
) -> Alert:
    return Alert(
        alert_id=new_uuid(),
        fired_at=fired_at,
        type=alert_type,
        severity=Severity.WARNING,
        title="test alert",
        detail={},
        agent_id=agent_id,
        suppressed=suppressed,
    )


def _make_config(cooldown_seconds: int, agents: dict | None = None) -> TjConfig:
    return TjConfig(
        version="1",
        agents=agents or {},
        alerts=AlertsConfig(
            cooldown_seconds=cooldown_seconds,
            channels=[AlertChannelConfig(type="stdout")],
        ),
    )


def test_cooldown_survives_a_restart_within_the_window():
    """A restart mid-cooldown must not re-insert or re-dispatch the same
    (type, agent_id) alert: the second AlertEngine shares the DB with the
    first, so it should see the still-fresh cooldown row and suppress.
    """
    config = _make_config(
        cooldown_seconds=300,
        agents={"test-agent": AgentConfig(
            sensitive_actions=[SensitiveAction(name="send_email")],
        )},
    )
    db = InMemoryBackend()
    span = make_tool_span(agent_id="test-agent", tool_name="send_email")

    # Process 1: fires and dispatches the first alert.
    engine1 = AlertEngine(db, config)
    channel1 = MagicMock()
    engine1.dispatcher.channels = [channel1]
    engine1.evaluate(span)
    assert channel1.send.call_count == 1

    # Restart: a brand new AlertEngine, same DB, no in-memory carryover
    # except what hydration reconstructs from the alerts table.
    engine2 = AlertEngine(db, config)
    channel2 = MagicMock()
    engine2.dispatcher.channels = [channel2]
    engine2.evaluate(span)

    # Still suppressed post-restart: persisted (for the audit trail) but
    # never dispatched a second time.
    assert channel2.send.call_count == 0
    rows = db.get_alerts(AlertFilters(type=AlertType.SENSITIVE_ACTION, limit=10))
    assert len(rows) == 2
    assert [r.suppressed for r in sorted(rows, key=lambda r: r.fired_at)] == [False, True]


def test_cooldown_does_not_suppress_after_a_restart_past_the_window():
    """The hydration window is bounded by cooldown_seconds: an alert older
    than the cooldown must NOT suppress a fresh firing after restart.

    Inserts a row directly (bypassing evaluate()) with `fired_at` well
    outside a real, non-zero cooldown window, so a build that dropped the
    `since=` bound from `_hydrate_from_db`'s query, rather than one that
    never suppresses at all, would also be caught here.
    """
    config = _make_config(
        cooldown_seconds=60,
        agents={"test-agent": AgentConfig(
            sensitive_actions=[SensitiveAction(name="send_email")],
        )},
    )
    db = InMemoryBackend()
    old_alert = _make_alert(
        "test-agent", AlertType.SENSITIVE_ACTION, utcnow() - timedelta(hours=1),
    )
    db.insert_alert(old_alert)

    engine = AlertEngine(db, config)
    channel = MagicMock()
    engine.dispatcher.channels = [channel]
    span = make_tool_span(agent_id="test-agent", tool_name="send_email")
    engine.evaluate(span)

    # The persisted row is an hour outside the 60s window: hydration must
    # not have seeded `_last_fired` from it, so this fires and dispatches.
    assert channel.send.call_count == 1


def test_failure_rate_does_not_refire_for_the_same_session_after_a_restart():
    """A session that already crossed the failure-rate threshold before a
    restart must not fire a second FAILURE_RATE alert after it, even though
    the new process's `_failure_rate_fired` set starts out empty in code.
    """
    config = _make_config(cooldown_seconds=300)
    db = InMemoryBackend()
    session_id = "sess-failure-rate-restart"

    engine1 = AlertEngine(db, config)
    engine1.dispatcher.channels = []
    # 5 errors in a row crosses _FAILURE_RATE_THRESHOLD (0.20) at
    # _FAILURE_RATE_CHECK_INTERVAL (5), firing exactly once.
    for _ in range(5):
        span = make_llm_span(agent_id="test-agent", session_id=session_id, status="error")
        db.insert_span(span)
        engine1.evaluate(span)

    fired_before_restart = db.get_alerts(AlertFilters(type=AlertType.FAILURE_RATE, limit=10))
    assert len(fired_before_restart) == 1
    assert session_id in engine1._failure_rate_fired

    # Restart: a fresh AlertEngine over the same DB.
    engine2 = AlertEngine(db, config)
    assert session_id in engine2._failure_rate_fired

    # One more error in the same session must not re-fire.
    span6 = make_llm_span(agent_id="test-agent", session_id=session_id, status="error")
    db.insert_span(span6)
    engine2.evaluate(span6)

    fired_after_restart = db.get_alerts(AlertFilters(type=AlertType.FAILURE_RATE, limit=10))
    assert len(fired_after_restart) == 1


def test_a_different_session_still_fires_its_own_failure_rate_alert_after_a_restart():
    """Hydration must not over-suppress: a session that never fired before
    the restart is unaffected by another session's hydrated state.
    """
    config = _make_config(cooldown_seconds=300)
    db = InMemoryBackend()

    engine1 = AlertEngine(db, config)
    engine1.dispatcher.channels = []
    for _ in range(5):
        span = make_llm_span(agent_id="agent-a", session_id="sess-a", status="error")
        db.insert_span(span)
        engine1.evaluate(span)

    engine2 = AlertEngine(db, config)
    engine2.dispatcher.channels = []
    for _ in range(5):
        span = make_llm_span(agent_id="agent-b", session_id="sess-b", status="error")
        db.insert_span(span)
        engine2.evaluate(span)

    fired = db.get_alerts(AlertFilters(type=AlertType.FAILURE_RATE, limit=10))
    assert {a.session_id for a in fired} == {"sess-a", "sess-b"}


def test_hydrate_skips_suppressed_rows():
    """A suppressed row records that an alert was DEDUPED, not a fresh
    firing, so hydrating from it would suppress a genuinely new alert.
    """
    now = utcnow()
    tracker = CooldownTracker(cooldown_seconds=300)
    tracker.hydrate([
        _make_alert("test-agent", AlertType.SENSITIVE_ACTION, now, suppressed=True),
    ])
    assert tracker.is_suppressed("test-agent", AlertType.SENSITIVE_ACTION) is False


def test_hydrate_keeps_the_newest_firing_regardless_of_row_order():
    """hydrate() must not rely on `alerts` being newest-first: given rows
    for the same (agent_id, type) key in EITHER order, the surviving
    `_last_fired` timestamp is the newest one, not whichever came first.
    """
    now = utcnow()
    older = now - timedelta(seconds=200)

    newest_first = CooldownTracker(cooldown_seconds=300)
    newest_first.hydrate([
        _make_alert("test-agent", AlertType.SENSITIVE_ACTION, now),
        _make_alert("test-agent", AlertType.SENSITIVE_ACTION, older),
    ])

    oldest_first = CooldownTracker(cooldown_seconds=300)
    oldest_first.hydrate([
        _make_alert("test-agent", AlertType.SENSITIVE_ACTION, older),
        _make_alert("test-agent", AlertType.SENSITIVE_ACTION, now),
    ])

    assert (
        newest_first._last_fired[("test-agent", AlertType.SENSITIVE_ACTION.value)]
        == oldest_first._last_fired[("test-agent", AlertType.SENSITIVE_ACTION.value)]
        == now
    )


def test_hydration_failure_degrades_to_pre_592_behavior_instead_of_blocking_startup():
    """A storage error during `_hydrate_from_db()` must not prevent
    `AlertEngine.__init__` from completing: it degrades to empty in-memory
    state, exactly what construction produced before hydration existed.
    """
    config = _make_config(
        cooldown_seconds=300,
        agents={"test-agent": AgentConfig(
            sensitive_actions=[SensitiveAction(name="send_email")],
        )},
    )
    db = InMemoryBackend()

    with patch.object(db, "get_alerts", side_effect=RuntimeError("storage unavailable")):
        engine = AlertEngine(db, config)

    assert engine._failure_rate_fired == set()
    assert engine.cooldown._last_fired == {}

    # And the engine is fully usable afterward: a span fired post-construction
    # still evaluates and dispatches normally.
    channel = MagicMock()
    engine.dispatcher.channels = [channel]
    span = make_tool_span(agent_id="test-agent", tool_name="send_email")
    engine.evaluate(span)
    assert channel.send.call_count == 1
