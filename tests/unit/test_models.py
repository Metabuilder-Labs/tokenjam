from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from tokenjam.core.models import (
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
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            assert session.effective_status == "active"

    def test_effective_status_active_quiet_becomes_idle(self):
        now = datetime(2026, 3, 28, 12, 10, 0, tzinfo=timezone.utc)
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 3, 28, 12, 3, 0, tzinfo=timezone.utc),
            status="active",
        )
        # 7 min since last activity: past the 5-min active window, well within
        # the 4h idle window -> idle (not stale).
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            assert session.effective_status == "idle"

    def test_effective_status_active_old_becomes_stale(self):
        now = datetime(2026, 3, 28, 17, 10, 0, tzinfo=timezone.utc)
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 3, 28, 12, 3, 0, tzinfo=timezone.utc),
            status="active",
        )
        # >5h since last activity, beyond the 4h idle window -> stale.
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            assert session.effective_status == "stale"

    def test_effective_status_uses_started_at_when_no_ended_at(self):
        now = datetime(2026, 3, 28, 12, 10, 0, tzinfo=timezone.utc)
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            status="active",
        )
        # 10 min since started_at, no ended_at -> idle (within the 4h window).
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            assert session.effective_status == "idle"

    def test_effective_status_closed_unchanged(self):
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            status="closed",
        )
        assert session.effective_status == "closed"

    def test_status_at_honours_custom_idle_window(self):
        now = datetime(2026, 3, 28, 12, 30, 0, tzinfo=timezone.utc)
        session = SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 3, 28, 12, 3, 0, tzinfo=timezone.utc),
            status="active",
        )
        # 27 min gap. With a 10-min idle window it is stale; the default 4h
        # window keeps it idle.
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            assert session.status_at(timedelta(minutes=10)) == "stale"
            assert session.status_at(timedelta(hours=4)) == "idle"

    def _idle_session(self, now):
        """An 'active' session whose spans went quiet 30 min ago (-> stale at a
        10-min idle window, idle at the default)."""
        return SessionRecord(
            session_id="s1",
            agent_id="a1",
            started_at=now - timedelta(hours=1),
            ended_at=now - timedelta(minutes=30),
            status="active",
        )

    def test_transcript_mtime_rescues_idle_to_active(self):
        now = datetime(2026, 3, 28, 12, 30, 0, tzinfo=timezone.utc)
        session = self._idle_session(now)
        fresh = now - timedelta(minutes=1)  # transcript touched 1 min ago
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            # Spans are stale, but the transcript is fresh -> live.
            assert session.status_with_transcript_mtime(fresh) == "active"

    def test_transcript_mtime_stale_does_not_rescue(self):
        now = datetime(2026, 3, 28, 12, 30, 0, tzinfo=timezone.utc)
        session = self._idle_session(now)
        old = now - timedelta(minutes=20)  # transcript also went quiet
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            assert session.status_with_transcript_mtime(old) == "idle"

    def test_transcript_mtime_none_is_unaffected(self):
        # Non-CC sessions (no transcript) pass through the span-derived status.
        now = datetime(2026, 3, 28, 12, 30, 0, tzinfo=timezone.utc)
        session = self._idle_session(now)
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            assert session.status_with_transcript_mtime(None) == "idle"

    def test_transcript_mtime_never_revives_closed(self):
        # A fresh transcript must not resurrect an explicitly-closed session.
        now = datetime(2026, 3, 28, 12, 30, 0, tzinfo=timezone.utc)
        session = SessionRecord(
            session_id="s1", agent_id="a1",
            started_at=now - timedelta(hours=1), status="closed",
        )
        fresh = now - timedelta(minutes=1)
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            assert session.status_with_transcript_mtime(fresh) == "closed"


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
