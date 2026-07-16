"""tj statusline — the zero-token Claude Code status line.

Covers the token/re-read math (deduped by message id), the badge + compaction
nudge threshold behavior, transcript discovery (transcript_path and the
~/.claude glob fallback), and the load-bearing fail-safe contract: any bad
input yields a minimal line (or nothing) and exit 0, never a traceback.
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from tests.factories import (
    make_claude_transcript_assistant_line,
    write_claude_transcript,
)
from tokenjam.cli.cmd_statusline import (
    REREAD_CRIT,
    REREAD_WARN,
    cmd_statusline,
    format_status_line,
    render_line,
    session_shares,
)


@pytest.fixture(autouse=True)
def _isolate_attribution_cache(tmp_path, monkeypatch):
    # Every test in this module resolves the attribution cache to an (empty,
    # unless a test writes to it) tmp path — never the real
    # ~/.local/share/tj/attribution_cache.json — so results are deterministic
    # regardless of what's actually been backfilled on the host machine.
    monkeypatch.setattr(
        "tokenjam.core.attribution_cache._cache_path",
        lambda: tmp_path / "attribution_cache.json",
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


def test_warn_threshold_adds_nudge(tmp_path):
    # Past WARN with no cached driver (driver unknown): the ⚠ badge plus the
    # memory-preserving default remedy — a fresh session first, /compact offered
    # as a secondary, never as the sole "just /compact" it used to be.
    line = _line_for_reread(tmp_path, REREAD_WARN + 1)  # just past warn
    assert "⚠" in line
    assert "resume-brief" in line
    assert "/compact" in line


def test_crit_threshold_adds_nudge(tmp_path):
    line = _line_for_reread(tmp_path, REREAD_CRIT + 1)  # just past crit
    assert "re-read" in line
    # Even at CRIT, with the driver unknown we don't blindly command /compact —
    # we lead with the memory-preserving option.
    assert "resume-brief" in line
    assert "/compact" in line


def test_warn_boundary_is_inclusive(tmp_path):
    # Exactly at the warn threshold should already nudge.
    line = _line_for_reread(tmp_path, REREAD_WARN)
    assert "/compact" in line


# --- top re-read driver (cached attribution) ---------------------------------


def test_top_driver_shown_past_warn_when_cached(tmp_path):
    from tokenjam.core.attribution_cache import write_attribution_cache

    cache_path = tmp_path / "attribution_cache.json"
    write_attribution_cache("CLAUDE.md", 14, 3, path=cache_path)
    line = _line_for_reread(tmp_path, REREAD_WARN + 1)
    assert "CLAUDE.md ×14" in line


def test_top_driver_absent_below_warn_even_when_cached(tmp_path):
    from tokenjam.core.attribution_cache import write_attribution_cache

    cache_path = tmp_path / "attribution_cache.json"
    write_attribution_cache("CLAUDE.md", 14, 3, path=cache_path)
    line = _line_for_reread(tmp_path, REREAD_WARN - 5)
    assert "CLAUDE.md" not in line


def test_top_driver_absent_when_no_cache_file(tmp_path):
    line = _line_for_reread(tmp_path, REREAD_CRIT + 1)
    assert "×" not in line


def test_format_status_line_appends_top_driver_when_passed():
    line = format_status_line("Opus 4.8", 1000, 90.0, "CLAUDE.md ×14")
    assert "re-read 90% (CLAUDE.md ×14)" in line


def test_format_status_line_unchanged_when_top_driver_omitted():
    # Existing 3-arg callers (quickstart's preview, existing tests) render
    # byte-for-byte the same with the new parameter defaulted to None.
    assert format_status_line("Opus 4.8", 1000, 90.0) == format_status_line(
        "Opus 4.8", 1000, 90.0, None
    )


# --- driver-conditional nudge (the /compact-is-wrong-for-static fix) ---------


def _cache(tmp_path, label: str, inclusion_type: str) -> None:
    """Seed the (isolated) attribution cache with a top driver of a given kind."""
    from tokenjam.core.attribution_cache import write_attribution_cache

    write_attribution_cache(
        label, 14, 3, inclusion_type, path=tmp_path / "attribution_cache.json"
    )


def _line_with_window(tmp_path, reread_pct: float, window_tokens: int,
                      model="Opus 4.8") -> str:
    """A single-turn transcript whose re-read % and last-turn window are set.

    The last turn's ``input + cache_read + cache_write`` IS the live window
    occupancy, so a big cache-read here drives both a high re-read % and a
    near-full window — exactly the case (c) escalation.
    """
    reread = int(round(window_tokens * reread_pct / 100.0))
    work = window_tokens - reread
    path = write_claude_transcript(tmp_path / "s.jsonl", [
        make_claude_transcript_assistant_line(
            message_id="m1", input_tokens=work, output_tokens=0,
            cache_read_input_tokens=reread, cache_creation_input_tokens=0,
        ),
    ])
    return render_line({"model": model, "transcript_path": path})


def test_static_driver_never_suggests_compact(tmp_path):
    # The core bug: a CLAUDE.md (file_read) driver past CRIT must NOT be told to
    # /compact — compaction can't touch statically re-injected content.
    _cache(tmp_path, "CLAUDE.md", "file_read")
    line = _line_for_reread(tmp_path, REREAD_CRIT + 1)
    assert "CLAUDE.md ×14" in line
    assert "/compact" not in line
    assert "tj context" in line


def test_search_driver_is_structural_not_compact(tmp_path):
    _cache(tmp_path, "grep foo", "search")
    line = _line_for_reread(tmp_path, REREAD_WARN + 1)
    assert "/compact" not in line
    assert "tj context" in line


def test_tool_output_driver_leads_with_fresh_session(tmp_path):
    # History-bloat driver: fresh session (memory-preserving) leads, /compact
    # remains as the blunt second option.
    _cache(tmp_path, "Bash → …", "tool_output")
    line = _line_for_reread(tmp_path, REREAD_CRIT + 1)
    assert "resume-brief" in line
    assert "/compact" in line


def test_prompt_driver_leads_with_fresh_session(tmp_path):
    _cache(tmp_path, "the same prompt…", "prompt")
    line = _line_for_reread(tmp_path, REREAD_WARN + 1)
    assert "resume-brief" in line
    assert "/compact" in line


def test_unknown_driver_falls_back_to_memory_preserving_default(tmp_path):
    # No cached driver at all (no backfill yet) — still never a bare /compact.
    line = _line_for_reread(tmp_path, REREAD_CRIT + 1)
    assert "resume-brief" in line
    assert "/compact" in line


def test_pre_upgrade_cache_without_type_is_driver_agnostic(tmp_path):
    # A cache written before this change carries no inclusion_type: label still
    # renders, and the nudge degrades to the driver-agnostic default.
    from tokenjam.core.attribution_cache import write_attribution_cache

    write_attribution_cache(
        "CLAUDE.md", 14, 3, path=tmp_path / "attribution_cache.json"
    )
    line = _line_for_reread(tmp_path, REREAD_CRIT + 1)
    assert "CLAUDE.md ×14" in line
    assert "resume-brief" in line


def test_near_limit_window_overrides_static_driver_to_compact(tmp_path):
    # Case (c): when the window is genuinely near full, a user-chosen /compact
    # beats a forced auto-compact — even for a static driver.
    _cache(tmp_path, "CLAUDE.md", "file_read")
    line = _line_with_window(tmp_path, REREAD_CRIT + 1, window_tokens=190_000)
    assert "/compact now" in line
    assert "window near full" in line


def test_window_below_limit_keeps_structural_remedy(tmp_path):
    # Same static driver but a modest window: no near-limit escalation.
    _cache(tmp_path, "CLAUDE.md", "file_read")
    line = _line_with_window(tmp_path, REREAD_CRIT + 1, window_tokens=20_000)
    assert "/compact" not in line


def test_one_million_context_not_flagged_near_limit_at_200k_scale(tmp_path):
    # A 1M-context session ("[1m]" in the model name) at 200K-scale occupancy is
    # nowhere near full — it must not be pushed to /compact.
    _cache(tmp_path, "CLAUDE.md", "file_read")
    line = _line_with_window(
        tmp_path, REREAD_CRIT + 1, window_tokens=190_000, model="Opus 4.8 [1m]"
    )
    assert "/compact now" not in line
    assert "tj context" in line


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


def _line_for_total(tmp_path, total: int) -> str:
    """Render a line whose session total is exactly *total* tokens."""
    path = write_claude_transcript(tmp_path / "s.jsonl", [
        make_claude_transcript_assistant_line(
            message_id="m1", input_tokens=total, output_tokens=0,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        ),
    ])
    return render_line({"model": "m", "transcript_path": path})


def test_sub_million_session_renders_k_not_zero_megabytes(tmp_path):
    # The #103 bug: a ~42k-token session rendered as a trust-eroding "0.0M tok".
    line = _line_for_total(tmp_path, 42_300)
    assert "42.3k tok" in line
    assert "0.0M" not in line


def test_million_boundary_switches_to_megabytes(tmp_path):
    # Just below a million stays in k; exactly a million flips to M.
    assert "999.0k tok" in _line_for_total(tmp_path, 999_000)
    assert "1.0M tok" in _line_for_total(tmp_path, 1_000_000)


# --- format_status_line (shared with `tj quickstart`'s live preview) --------


def test_format_status_line_model_only_when_no_figures():
    assert format_status_line("Opus 4.8", None, None) == "◆ Opus 4.8"


def test_format_status_line_matches_render_line_for_same_inputs(tmp_path):
    # The preview (cmd_quickstart) calls format_status_line directly with a
    # point-in-time (total, reread_pct); it must render BYTE-IDENTICAL text to
    # what render_line would produce for a transcript with those same final
    # figures, since that's the whole point of sharing the formatter.
    path = write_claude_transcript(tmp_path / "s.jsonl", [
        make_claude_transcript_assistant_line(
            message_id="m1", input_tokens=200, output_tokens=100,
            cache_read_input_tokens=9500, cache_creation_input_tokens=200,
        ),
    ])
    via_render_line = render_line({"model": "Opus 4.8", "transcript_path": path})
    total, reread_pct = session_shares(path)
    via_formatter = format_status_line("Opus 4.8", total, reread_pct)
    assert via_formatter == via_render_line


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
    # A sub-1M session human-sizes to `k`, not a misleading "0.0M".
    assert "1.0k tok" in line
    assert "0.0M" not in line


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
    # No cached driver and a small window (9.9k occupancy, far from the limit),
    # so the nudge is the memory-preserving default, not a bare /compact command.
    assert "◆ Opus 4.8" in out
    assert "re-read 95%" in out
    assert "🕳️" in out
    assert "resume-brief" in out
