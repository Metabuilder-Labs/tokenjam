"""tj statusline — the zero-token Claude Code status line.

Covers the token/re-read math (deduped by message id), the badge + compaction
nudge threshold behavior, transcript discovery (transcript_path and the
~/.claude glob fallback), and the load-bearing fail-safe contract: any bad
input yields a minimal line (or nothing) and exit 0, never a traceback.
"""
from __future__ import annotations

import json

from click.testing import CliRunner

from tests.factories import (
    make_claude_transcript_assistant_line,
    write_claude_transcript,
)
from tokenjam.cli.cmd_statusline import (
    REREAD_CRIT,
    REREAD_WARN,
    cmd_statusline,
    render_line,
    session_shares,
)


def _run(payload) -> str:
    """Invoke the command with *payload* piped on stdin; return stdout."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    result = CliRunner().invoke(cmd_statusline, input=raw)
    assert result.exit_code == 0, result.output
    return result.output


# --- token / re-read math ---------------------------------------------------


def test_session_shares_sums_all_usage_buckets(tmp_path):
    path = write_claude_transcript(tmp_path / "s.jsonl", [
        make_claude_transcript_assistant_line(
            message_id="m1", input_tokens=100, output_tokens=50,
            cache_read_input_tokens=800, cache_creation_input_tokens=50,
        ),
    ])
    total, reread_pct = session_shares(path)
    # 100 + 50 + 800 + 50 = 1000 total; 800 of it re-read.
    assert total == 1000
    assert reread_pct == 80.0


def test_session_shares_dedupes_by_message_id(tmp_path):
    # Claude Code streams several transcript lines per message, each carrying the
    # SAME cumulative usage. Counting them all would inflate every number.
    dup = make_claude_transcript_assistant_line(
        message_id="same", input_tokens=1000, cache_read_input_tokens=0,
        output_tokens=0, cache_creation_input_tokens=0,
    )
    path = write_claude_transcript(tmp_path / "s.jsonl", [dup, dup, dup])
    total, _ = session_shares(path)
    assert total == 1000  # counted once, not 3000


def test_session_shares_dedupes_last_wins(tmp_path):
    # Mid-stream snapshots of one message carry PARTIAL, growing usage under the
    # same message id; the final record is authoritative. Last-wins keeps that
    # finalized total (matching core/backfill) — first-wins would undercount.
    early = make_claude_transcript_assistant_line(
        message_id="m1", input_tokens=100, output_tokens=10,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )
    final = make_claude_transcript_assistant_line(
        message_id="m1", input_tokens=100, output_tokens=400,
        cache_read_input_tokens=0, cache_creation_input_tokens=0,
    )
    path = write_claude_transcript(tmp_path / "s.jsonl", [early, final])
    total, _ = session_shares(path)
    assert total == 500  # 100 + 400 (final), not 110 (first)


def test_session_shares_ignores_non_assistant_and_bad_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    good = make_claude_transcript_assistant_line(
        message_id="m1", input_tokens=500, output_tokens=0,
        cache_read_input_tokens=500, cache_creation_input_tokens=0,
    )
    path.write_text(
        json.dumps({"type": "user", "message": {"content": "hi"}}) + "\n"
        + "not json at all\n"
        + json.dumps(good) + "\n",
        encoding="utf-8",
    )
    total, reread_pct = session_shares(str(path))
    assert total == 1000
    assert reread_pct == 50.0


# --- badge + nudge thresholds ----------------------------------------------


def _line_for_reread(tmp_path, pct: float) -> str:
    """Build a session whose re-read share is exactly *pct* percent."""
    reread = int(round(pct * 10))          # of 1000 total
    work = 1000 - reread
    path = write_claude_transcript(tmp_path / "s.jsonl", [
        make_claude_transcript_assistant_line(
            message_id="m1", input_tokens=work, output_tokens=0,
            cache_read_input_tokens=reread, cache_creation_input_tokens=0,
        ),
    ])
    return render_line({"model": "Opus 4.8", "transcript_path": path})


def test_healthy_session_has_check_badge_and_no_nudge(tmp_path):
    line = _line_for_reread(tmp_path, 40.0)
    assert "✓" in line
    assert "re-read 40%" in line
    assert "/compact" not in line


def test_warn_threshold_adds_consider_compact(tmp_path):
    line = _line_for_reread(tmp_path, REREAD_WARN + 1)  # just past warn
    assert "⚠" in line
    assert "consider /compact" in line


def test_crit_threshold_adds_reclaim_quota_nudge(tmp_path):
    line = _line_for_reread(tmp_path, REREAD_CRIT + 1)  # just past crit
    assert "re-read" in line
    assert "/compact to reclaim quota" in line


def test_warn_boundary_is_inclusive(tmp_path):
    # Exactly at the warn threshold should already nudge.
    line = _line_for_reread(tmp_path, REREAD_WARN)
    assert "/compact" in line


def test_line_reports_model_and_tokens(tmp_path):
    path = write_claude_transcript(tmp_path / "s.jsonl", [
        make_claude_transcript_assistant_line(
            message_id="m1", input_tokens=1_000_000, output_tokens=0,
            cache_read_input_tokens=1_000_000, cache_creation_input_tokens=0,
        ),
    ])
    line = render_line({"model": {"display_name": "Opus 4.8"}, "transcript_path": path})
    assert "◆ Opus 4.8" in line
    assert "2.0M tok" in line


# --- transcript discovery ---------------------------------------------------


def test_glob_fallback_locates_session_by_id(tmp_path, monkeypatch):
    # No transcript_path — must find ~/.claude/projects/**/<session_id>.jsonl.
    fake_home = tmp_path / "home"
    proj = fake_home / ".claude" / "projects" / "some-proj"
    proj.mkdir(parents=True)
    write_claude_transcript(proj / "sess-123.jsonl", [
        make_claude_transcript_assistant_line(
            message_id="m1", input_tokens=1000, output_tokens=0,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    ])
    monkeypatch.setenv("HOME", str(fake_home))
    line = render_line({"session_id": "sess-123", "model": "m"})
    assert "1000" not in line  # tokens rendered in M, not raw
    assert "0.0M tok" in line


# --- fail-safe contract -----------------------------------------------------


def test_invalid_json_stdin_exits_zero_with_minimal_line():
    out = _run("{not valid json")
    assert out.strip() == "◆ ?"


def test_empty_stdin_exits_zero():
    out = _run("")
    assert out.strip() == "◆ ?"


def test_non_dict_json_exits_zero():
    assert _run("[1, 2, 3]").strip() == "◆ ?"


def test_missing_transcript_degrades_to_model_only():
    out = _run({"model": "Opus 4.8", "transcript_path": "/no/such/file.jsonl"})
    assert out.strip() == "◆ Opus 4.8"


def test_real_session_end_to_end_line(tmp_path):
    path = write_claude_transcript(tmp_path / "s.jsonl", [
        make_claude_transcript_assistant_line(
            message_id="m1", input_tokens=200, output_tokens=100,
            cache_read_input_tokens=9500, cache_creation_input_tokens=200,
        ),
    ])
    out = _run({"model": {"display_name": "Opus 4.8"}, "transcript_path": path})
    # 200+100+9500+200 = 10000 total, 9500 re-read = 95% -> crit badge + nudge.
    assert "◆ Opus 4.8" in out
    assert "re-read 95%" in out
    assert "/compact to reclaim quota" in out
