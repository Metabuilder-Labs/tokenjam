"""Unit tests for the shared Claude Code backfill progress UI (#443).

`tj backfill claude-code` and `tj onboard --claude-code` both drive
`ingest_claude_code`'s `progress=` hook through `backfill_progress` — this
covers the TTY-aware branch (Rich `Progress` vs. periodic plain prints) and
the running counters (session count, "of total", accumulated tokens).
"""
from __future__ import annotations

from datetime import datetime, timezone

from tokenjam.cli.backfill_progress import backfill_progress
from tokenjam.core.backfill import BackfillResult, ParsedSession


def _parsed(session_id: str, input_tokens: int = 100, output_tokens: int = 50,
           cache_tokens: int = 0) -> ParsedSession:
    now = datetime(2026, 6, 20, tzinfo=timezone.utc)
    return ParsedSession(
        session_id=session_id,
        agent_id="claude-code-proj",
        started_at=now,
        ended_at=now,
        cwd="/Users/me/proj",
        spans=[],
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_cache_tokens=cache_tokens,
        total_cost_usd=0.01,
        tool_call_count=0,
    )


def _result(sessions_seen: int) -> BackfillResult:
    r = BackfillResult()
    r.sessions_seen = sessions_seen
    return r


def test_quiet_yields_noop_callback(capsys):
    with backfill_progress(10, quiet=True) as cb:
        cb(_parsed("s1"), _result(1))
    # No output at all — matches `tj backfill claude-code --quiet`.
    assert capsys.readouterr().out == ""


def test_non_tty_prints_periodically_not_every_session(capsys):
    # capsys-captured stdout is never a tty, so this exercises the plain-print
    # degrade path without needing to fake a terminal.
    with backfill_progress(250, quiet=False) as cb:
        for i in range(150):
            cb(_parsed(f"s{i}"), _result(i + 1))
    out = capsys.readouterr().out
    # Cadence is every 100th session — one line at #100, not at every session.
    assert out.count("Backfilling") == 1
    assert "100/250" in out


def test_non_tty_line_includes_running_token_total(capsys):
    with backfill_progress(None, quiet=False) as cb:
        for i in range(100):
            cb(_parsed(f"s{i}", input_tokens=100, output_tokens=50), _result(i + 1))
    out = capsys.readouterr().out
    # 100 sessions * 150 tokens = 15,000 -> humanized as "15.0k" by format_tokens.
    assert "15.0k tokens read" in out
    # No total was given, so no "/total" suffix — just the running count.
    assert "100 sessions" in out
    assert "/" not in out.split("sessions")[0]


def test_no_total_shows_bare_running_count(capsys):
    with backfill_progress(None, quiet=False) as cb:
        for i in range(100):
            cb(_parsed(f"s{i}"), _result(i + 1))
    out = capsys.readouterr().out
    assert "100 sessions" in out
