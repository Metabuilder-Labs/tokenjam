"""Tests for the session-status regression fix set:

  FIX 1 — `session_record_from_parsed` derives status from the transcript's
          on-disk mtime instead of hardcoding "completed" for every backfilled
          Claude Code session.
  FIX 2 — `upsert_session`'s ON CONFLICT refuses to downgrade a row the live
          path already marked 'active' when the incoming write is staler.
  FIX 3 — a periodic sweep corrects raw `status='active'` zombie rows whose
          COMPUTED status already reads as stale, without ever touching a
          genuinely-recent row.

All three reuse the SAME staleness definitions already in the codebase
(SESSION_STALE_THRESHOLD / SESSION_IDLE_THRESHOLD /
SessionRecord.status_with_transcript_mtime) — no new threshold is invented.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tokenjam.core.backfill import ParsedSession, session_record_from_parsed
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.models import SESSION_STALE_THRESHOLD
from tokenjam.core.transcript_sync import sweep_stale_active_sessions

from tests.factories import make_session


def _parsed(transcript_mtime: datetime | None) -> ParsedSession:
    return ParsedSession(
        session_id="sess-status",
        agent_id="claude-code-proj",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        cwd="/proj",
        spans=[],
        total_input_tokens=111,
        total_output_tokens=22,
        total_cache_tokens=3,
        total_cost_usd=0.05,
        tool_call_count=1,
        transcript_mtime=transcript_mtime,
    )


# --- FIX 1: session_record_from_parsed derives status from transcript mtime ---

def test_recent_transcript_mtime_yields_active_status():
    now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    fresh_mtime = now - timedelta(minutes=1)
    with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
        rec = session_record_from_parsed(_parsed(fresh_mtime))
    assert rec.status == "active"


def test_quiet_transcript_mtime_yields_completed_status():
    now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    quiet_mtime = now - timedelta(hours=6)
    with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
        rec = session_record_from_parsed(_parsed(quiet_mtime))
    assert rec.status == "completed"


def test_transcript_mtime_exactly_at_threshold_boundary_is_still_active():
    now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    boundary_mtime = now - SESSION_STALE_THRESHOLD
    with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
        rec = session_record_from_parsed(_parsed(boundary_mtime))
    assert rec.status == "active"


def test_missing_transcript_mtime_defaults_to_completed():
    # No mtime available (couldn't stat, or a ParsedSession built directly, as
    # existing tests do) -- nothing to rescue it, so it stays terminal.
    rec = session_record_from_parsed(_parsed(None))
    assert rec.status == "completed"


def test_status_correction_never_touches_tokens_or_cost():
    now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    fresh_mtime = now - timedelta(minutes=1)
    with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
        rec = session_record_from_parsed(_parsed(fresh_mtime))
    assert rec.input_tokens == 111
    assert rec.output_tokens == 22
    assert rec.cache_tokens == 3
    assert rec.total_cost_usd == 0.05
    assert rec.tool_call_count == 1


# --- FIX 2: upsert_session guards against downgrading a fresher active row ---

def test_upsert_does_not_downgrade_a_more_recently_active_row():
    db = InMemoryBackend()
    try:
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Live OTLP path marks the session active with a recent last-activity.
        db.upsert_session(make_session(
            session_id="s-live", agent_id="a1", status="active",
            started_at=now - timedelta(minutes=10), ended_at=now,
            input_tokens=500, output_tokens=100,
        ))
        # A stale backfill/catch-up pass re-parses an OLDER snapshot of the
        # same session and tries to write it "completed".
        db.upsert_session(make_session(
            session_id="s-live", agent_id="a1", status="completed",
            started_at=now - timedelta(minutes=10),
            ended_at=now - timedelta(minutes=8),
            input_tokens=400, output_tokens=80,
        ))
        sess = db.get_session("s-live")
        assert sess.status == "active"
    finally:
        db.close()


def test_upsert_allows_a_genuinely_newer_completion():
    db = InMemoryBackend()
    try:
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        db.upsert_session(make_session(
            session_id="s-done", agent_id="a1", status="active",
            started_at=now - timedelta(minutes=10), ended_at=now - timedelta(minutes=5),
        ))
        # A later write carries GENUINELY newer last-activity and marks it
        # completed -- this is real information, not a stale replay, so it
        # must win.
        db.upsert_session(make_session(
            session_id="s-done", agent_id="a1", status="completed",
            started_at=now - timedelta(minutes=10), ended_at=now,
        ))
        sess = db.get_session("s-done")
        assert sess.status == "completed"
    finally:
        db.close()


def test_explicit_close_still_wins_over_active():
    # close_session_by_id / close_sessions_by_instance are direct UPDATEs, not
    # upsert_session -- so the anti-downgrade guard above never applies to them.
    db = InMemoryBackend()
    try:
        db.upsert_session(make_session(session_id="s-close", agent_id="a1", status="active"))
        closed_count = db.close_session_by_id("s-close")
        assert closed_count == 1
        assert db.get_session("s-close").status == "closed"
    finally:
        db.close()


def test_upsert_still_promotes_active_to_active_on_conflict():
    # Sanity: the guard only fires when incoming status != 'active'; an
    # ordinary re-activation (live path keeps sending spans) is untouched.
    db = InMemoryBackend()
    try:
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        db.upsert_session(make_session(
            session_id="s-reactive", agent_id="a1", status="completed",
            started_at=now - timedelta(minutes=10), ended_at=now - timedelta(minutes=9),
        ))
        db.upsert_session(make_session(
            session_id="s-reactive", agent_id="a1", status="active",
            started_at=now - timedelta(minutes=10), ended_at=now,
        ))
        assert db.get_session("s-reactive").status == "active"
    finally:
        db.close()


# --- FIX 3: periodic sweep corrects raw zombie 'active' rows -----------------

def test_sweep_corrects_a_stale_active_row(tmp_path):
    db = InMemoryBackend()
    try:
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        # A single dropped OTLP span: created active, no closing span ever
        # arrived, no transcript on disk at all (a non-CC / SDK session).
        db.upsert_session(make_session(
            session_id="z-1", agent_id="sdk-agent", status="active",
            started_at=now - timedelta(days=2), ended_at=now - timedelta(days=2),
            input_tokens=0, output_tokens=0, tool_call_count=0, error_count=1,
        ))
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            corrected = sweep_stale_active_sessions(db, root=tmp_path)
        assert corrected == 1
        assert db.get_session("z-1").status == "completed"
    finally:
        db.close()


def test_sweep_does_not_touch_a_genuinely_recent_active_row(tmp_path):
    db = InMemoryBackend()
    try:
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        db.upsert_session(make_session(
            session_id="live-1", agent_id="sdk-agent", status="active",
            started_at=now - timedelta(minutes=10), ended_at=now - timedelta(minutes=1),
        ))
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            corrected = sweep_stale_active_sessions(db, root=tmp_path)
        assert corrected == 0
        assert db.get_session("live-1").status == "active"
    finally:
        db.close()


def test_sweep_rescues_a_row_whose_transcript_is_still_being_written(tmp_path):
    # Span-recency alone reads this as stale, but its CC transcript on disk
    # was modified moments ago -- still a live terminal. The sweep must NOT
    # close it, mirroring the same mtime-rescue FIX 1 and the read-time
    # `_live_status` route both apply.
    db = InMemoryBackend()
    try:
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        db.upsert_session(make_session(
            session_id="cc-live", agent_id="claude-code-proj", status="active",
            started_at=now - timedelta(hours=6), ended_at=now - timedelta(hours=5),
        ))
        project_dir = tmp_path / "proj"
        project_dir.mkdir(parents=True)
        transcript = project_dir / "cc-live.jsonl"
        transcript.write_text("{}\n")
        recent = (now - timedelta(minutes=1)).timestamp()
        os.utime(transcript, (recent, recent))

        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            corrected = sweep_stale_active_sessions(db, root=tmp_path)
        assert corrected == 0
        assert db.get_session("cc-live").status == "active"
    finally:
        db.close()


def test_sweep_closes_cc_session_once_its_transcript_also_goes_quiet(tmp_path):
    db = InMemoryBackend()
    try:
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        db.upsert_session(make_session(
            session_id="cc-dead", agent_id="claude-code-proj", status="active",
            started_at=now - timedelta(hours=6), ended_at=now - timedelta(hours=5),
        ))
        project_dir = tmp_path / "proj"
        project_dir.mkdir(parents=True)
        transcript = project_dir / "cc-dead.jsonl"
        transcript.write_text("{}\n")
        quiet = (now - timedelta(hours=5)).timestamp()
        os.utime(transcript, (quiet, quiet))

        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            corrected = sweep_stale_active_sessions(db, root=tmp_path)
        assert corrected == 1
        assert db.get_session("cc-dead").status == "completed"
    finally:
        db.close()


def test_sweep_never_alters_tokens_or_cost(tmp_path):
    db = InMemoryBackend()
    try:
        now = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        db.upsert_session(make_session(
            session_id="z-tok", agent_id="sdk-agent", status="active",
            started_at=now - timedelta(days=2), ended_at=now - timedelta(days=2),
            input_tokens=250, output_tokens=60, total_cost_usd=1.23,
        ))
        with patch("tokenjam.utils.time_parse.utcnow", return_value=now):
            sweep_stale_active_sessions(db, root=tmp_path)
        sess = db.get_session("z-tok")
        assert sess.status == "completed"
        assert sess.input_tokens == 250
        assert sess.output_tokens == 60
        assert sess.total_cost_usd == 1.23
    finally:
        db.close()
