"""Unit tests for CooldownTracker — pure logic, no I/O."""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest

from tokenjam.core.alerts import CooldownTracker, is_interactive_coding_agent
from tokenjam.core.models import AlertType
from tokenjam.utils.time_parse import utcnow


# --------------------------------------------------------------------------- #
# is_interactive_coding_agent — single source of truth for coding-vs-SDK
# classification, margin cases pinned so alerts.py and framing.py
# cannot silently drift apart again.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("agent_id,expected", [
    ("claude-code", True),               # bare id, no trailing slug — margin case
    ("claude-code-my-project", True),
    ("codex", True),                     # bare codex id — margin case
    ("codex-cli-session", True),
    ("sdk-agent-x", False),
    ("some-other-agent", False),
    (None, False),
    ("", False),
])
def test_is_interactive_coding_agent_margin_cases(agent_id, expected):
    assert is_interactive_coding_agent(agent_id) is expected


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
