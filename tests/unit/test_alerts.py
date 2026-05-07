"""Unit tests for CooldownTracker — pure logic, no I/O."""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

from tj.core.alerts import CooldownTracker
from tj.core.models import AlertType
from tj.utils.time_parse import utcnow


def test_cooldown_allows_first_alert():
    tracker = CooldownTracker(cooldown_seconds=60)
    assert not tracker.is_suppressed("agent-a", AlertType.SENSITIVE_ACTION)


def test_cooldown_suppresses_repeat_alert_within_window():
    tracker = CooldownTracker(cooldown_seconds=60)
    tracker.record("agent-a", AlertType.SENSITIVE_ACTION)
    assert tracker.is_suppressed("agent-a", AlertType.SENSITIVE_ACTION)


def test_cooldown_allows_alert_after_window_expires():
    tracker = CooldownTracker(cooldown_seconds=60)
    past = utcnow() - timedelta(seconds=120)
    tracker._last_fired[("agent-a", AlertType.SENSITIVE_ACTION.value)] = past
    assert not tracker.is_suppressed("agent-a", AlertType.SENSITIVE_ACTION)


def test_cooldown_tracks_per_agent_independently():
    tracker = CooldownTracker(cooldown_seconds=60)
    tracker.record("agent-a", AlertType.RETRY_LOOP)
    # Agent B should NOT be suppressed by agent A's alert
    assert not tracker.is_suppressed("agent-b", AlertType.RETRY_LOOP)
    # Agent A should be suppressed
    assert tracker.is_suppressed("agent-a", AlertType.RETRY_LOOP)


def test_cooldown_tracks_per_type_independently():
    tracker = CooldownTracker(cooldown_seconds=60)
    tracker.record("agent-a", AlertType.SENSITIVE_ACTION)
    # Same agent, different type should NOT be suppressed
    assert not tracker.is_suppressed("agent-a", AlertType.RETRY_LOOP)
    # Same agent, same type should be suppressed
    assert tracker.is_suppressed("agent-a", AlertType.SENSITIVE_ACTION)


def test_cooldown_handles_none_agent_id():
    tracker = CooldownTracker(cooldown_seconds=60)
    tracker.record(None, AlertType.FAILURE_RATE)
    assert tracker.is_suppressed(None, AlertType.FAILURE_RATE)
    assert not tracker.is_suppressed("agent-a", AlertType.FAILURE_RATE)
