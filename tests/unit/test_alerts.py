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


# --------------------------------------------------------------------------- #
# agent_display_name — the project, without the tool prefix. Display only.
# --------------------------------------------------------------------------- #
def test_agent_display_name_strips_the_tool_prefix():
    from tokenjam.core.alerts import agent_display_name

    assert agent_display_name("claude-code-tokenjam") == "tokenjam"
    assert agent_display_name("claude-code-splito") == "splito"
    assert agent_display_name("codex-app-server") == "app-server"


def test_agent_display_name_never_returns_an_empty_name():
    """The degenerate ids this corpus actually contains. A blank cell reads as
    missing data, so every one of these keeps something to show."""
    from tokenjam.core.alerts import agent_display_name

    # A bare tool id has no project to show; the honest answer is the tool.
    assert agent_display_name("claude-code") == "claude-code"
    assert agent_display_name("codex") == "codex"
    # A trailing separator would strip to nothing, so it keeps the whole id.
    assert agent_display_name("claude-code-") == "claude-code-"
    # Untouched: an SDK id is its own declared name, and there is nothing to
    # strip from it.
    assert agent_display_name("billing-service") == "billing-service"
    assert agent_display_name("sdk-workload-oversized-model") == "sdk-workload-oversized-model"
    # Absent stays absent rather than becoming a string.
    assert agent_display_name(None) is None
    assert agent_display_name("") == ""


def test_agent_display_name_reads_the_same_prefixes_as_the_classifier():
    """Derived from `_INTERACTIVE_AGENT_PREFIXES`, not from its own literals, so
    a new coding tool becomes strippable the moment it is classifiable."""
    from tokenjam.core import alerts

    for prefix in alerts._INTERACTIVE_AGENT_PREFIXES:
        assert alerts.agent_display_name(f"{prefix}-myproject") == "myproject"
        assert alerts.agent_display_name(prefix) == prefix
        assert alerts.is_interactive_coding_agent(f"{prefix}-myproject")


def test_agent_display_name_is_display_only_and_never_an_identity():
    """Two different ids may shorten to the same name (`claude-code-api` and
    `codex-api`). That is why this is a DISPLAY helper and callers keep
    `agent_id` beside it; the UI resolves such collisions per rendered list."""
    from tokenjam.core.alerts import agent_display_name

    assert agent_display_name("claude-code-api") == agent_display_name("codex-api") == "api"
