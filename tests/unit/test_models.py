from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from tj.core.models import (
    SESSION_STALE_THRESHOLD, SessionRecord, Severity, AlertType, SpanStatus, SpanKind,
)


class TestSessionRecord:
    def test_duration_seconds_with_both_times(self):
        started = datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)
        ended = datetime(2026, 3, 28, 12, 5, 30, tzinfo=timezone.utc)
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=started,
            ended_at=ended,
        )
        assert session.duration_seconds == 330.0

    def test_duration_seconds_none_without_end_time(self):
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
        )
        assert session.duration_seconds is None

    def test_default_status_is_active(self):
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
        )
        assert session.status == "active"

    def test_effective_status_completed_unchanged(self):
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            status="completed",
        )
        assert session.effective_status == "completed"

    def test_effective_status_active_recent_stays_active(self):
        now = datetime(2026, 3, 28, 12, 10, 0, tzinfo=timezone.utc)
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 3, 28, 12, 9, 0, tzinfo=timezone.utc),
            status="active",
        )
        with patch("tj.utils.time_parse.utcnow", return_value=now):
            assert session.effective_status == "active"

    def test_effective_status_active_stale_becomes_stale(self):
        now = datetime(2026, 3, 28, 12, 10, 0, tzinfo=timezone.utc)
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 3, 28, 12, 3, 0, tzinfo=timezone.utc),
            status="active",
        )
        # 7 minutes since last activity > 5 minute threshold
        with patch("tj.utils.time_parse.utcnow", return_value=now):
            assert session.effective_status == "stale"

    def test_effective_status_uses_started_at_when_no_ended_at(self):
        now = datetime(2026, 3, 28, 12, 10, 0, tzinfo=timezone.utc)
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            status="active",
        )
        # 10 minutes since started_at, no ended_at
        with patch("tj.utils.time_parse.utcnow", return_value=now):
            assert session.effective_status == "stale"


class TestEnums:
    def test_severity_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"

    def test_alert_type_values(self):
        assert AlertType.COST_BUDGET_DAILY.value == "cost_budget_daily"
        assert AlertType.RETRY_LOOP.value == "retry_loop"
        assert AlertType.DRIFT_DETECTED.value == "drift_detected"

    def test_span_status_values(self):
        assert SpanStatus.OK.value == "ok"
        assert SpanStatus.ERROR.value == "error"
        assert SpanStatus.UNSET.value == "unset"

    def test_span_kind_values(self):
        assert SpanKind.CLIENT.value == "client"
        assert SpanKind.INTERNAL.value == "internal"
        assert SpanKind.SERVER.value == "server"

    def test_severity_is_string(self):
        assert isinstance(Severity.CRITICAL, str)
        assert Severity.CRITICAL == "critical"

    def test_alert_type_is_string(self):
        assert isinstance(AlertType.RETRY_LOOP, str)
        assert AlertType.RETRY_LOOP == "retry_loop"
