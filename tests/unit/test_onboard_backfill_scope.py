"""`tj onboard --claude-code` backfill-scope UX (#443).

Founder-reported issue: onboard used to backfill the ENTIRE on-disk Claude
Code history with no cap and no progress output. On a large `~/.claude`
history that's many silent, 100%-CPU minutes right after "tj config written
to...", indistinguishable from a hang at the exact moment a new user's trust
is most fragile. This covers the fix: a backfill-scope prompt (interactive)
or default (non-interactive) before ingest starts, explicit `--backfill-days`
/ `--backfill-all` flags for scripting, and the "complete it later" pointer at
the real `tj backfill claude-code` command.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

import tokenjam.core.backfill as backfill_mod
from tokenjam.cli.cmd_onboard import cmd_onboard

# `sess-recent` must stay well inside the default 30-day backfill window no
# matter when the suite runs. A hardcoded absolute timestamp is a time bomb:
# once wall-clock time passes `<that date> + DEFAULT_BACKFILL_DAYS` (30), the
# session falls out of scope and every assertion below starts failing with
# no code change involved. Anchor it relative to "now" instead.
_RECENT_SESSION_TS = (
    datetime.now(tz=timezone.utc) - timedelta(days=5)
).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _make_session_file(root: Path, session_id: str, cwd: str, ts: str) -> Path:
    project_dir = root / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    record = {
        "type": "assistant",
        "uuid": f"u-{session_id}",
        "timestamp": ts,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 100, "output_tokens": 50,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            },
        },
    }
    path.write_text(json.dumps(record))
    return path


@pytest.fixture
def _isolated_claude_code_with_history(monkeypatch, tmp_path):
    """Same isolation as `_isolated_claude_code` in test_onboard_first_run.py,
    but points CLAUDE_CODE_PROJECTS_ROOT at a real directory with session
    files, so the backfill path actually runs (has_data == True) instead of
    being skipped."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    projects_root = tmp_path / ".claude" / "projects"
    monkeypatch.setattr(backfill_mod, "CLAUDE_CODE_PROJECTS_ROOT", projects_root)
    monkeypatch.setattr("tokenjam.cli.cmd_onboard.shutil.which", lambda _x: None)
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._stop_serve_for_db_write", lambda: False,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._finish_onboard_serve", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "tokenjam.cli.cmd_onboard._try_apply_declared_plans", lambda *a, **k: None,
    )
    _make_session_file(projects_root, "sess-recent", "/Users/me/proj",
                      _RECENT_SESSION_TS)
    _make_session_file(projects_root, "sess-old", "/Users/me/proj",
                      "2020-01-01T10:00:00.000Z")
    return projects_root


def _flat(output: str) -> str:
    """Collapse Rich's terminal-width word-wrapping so assertions can match a
    phrase regardless of where the console happened to break the line."""
    return " ".join(output.split())


def _run_claude_code(tmp_path, *extra_args, input_str="3\n0\n"):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        return runner.invoke(
            cmd_onboard,
            ["--claude-code", "--no-daemon", *extra_args],
            input=input_str, obj={},
        )


# --- Non-interactive default (no TTY, no explicit flag) ----------------------


def test_non_interactive_derives_backfill_from_default_span(
    _isolated_claude_code_with_history, tmp_path,
):
    # CliRunner's stdin is never a tty, and `_is_interactive()` isn't patched
    # here — so this exercises the real non-interactive path. #643: the backfill
    # window is now DERIVED from the analysis span (the single "how far back"
    # question). The non-interactive default span is 90d, so backfill takes 90.
    res = _run_claude_code(tmp_path)
    assert res.exit_code == 0, res.output
    flat = _flat(res.output)
    # No second "Backfill your Claude Code history" menu — asked once now (#643).
    assert "Backfill your Claude Code history:" not in flat
    assert "Backfilling the last 90 days" in flat
    assert "tj backfill claude-code" in flat
    # The old session (2020) is outside the 90-day span, only the recent one.
    # #675: the completion line is now the simplified "✓ ... : N sessions, over
    # M days." (the verbose "N new · N total" detail was dropped).
    assert "1 session" in _flat(res.output)


# --- Single "how far back" question (#643) ------------------------------------


def test_no_second_backfill_prompt_span_drives_window(
    _isolated_claude_code_with_history, tmp_path, monkeypatch,
):
    """#643: onboard asks "how far back" ONCE (the analysis-span prompt). There
    is no separate backfill-scope menu; the chosen span drives the backfill
    window. Choosing span "1" (30d) backfills the last 30 days."""
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._is_interactive", lambda: True)
    # Plan "3", budget "0", analysis span "1" (30d). No fourth answer needed.
    res = _run_claude_code(tmp_path, input_str="3\n0\n1\n")
    assert res.exit_code == 0, res.output
    flat = _flat(res.output)
    assert "Backfill your Claude Code history:" not in flat
    assert "How far back should tj analyze?" in flat
    assert "Backfilling the last 30 days" in flat
    assert "Run `tj backfill claude-code` afterwards for your full history" in flat


def test_all_span_backfills_everything_uncapped(
    _isolated_claude_code_with_history, tmp_path, monkeypatch,
):
    """#643: the "all available" analysis span (choice 3) backfills everything
    with no window — both the recent and the 2020 session land."""
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._is_interactive", lambda: True)
    # Plan "3", budget "0", analysis span "3" (all).
    res = _run_claude_code(tmp_path, input_str="3\n0\n3\n")
    assert res.exit_code == 0, res.output
    flat = _flat(res.output)
    assert "Backfill your Claude Code history:" not in flat
    # The "Backfilling all available Claude Code history" preamble line was
    # removed (founder review, demo trim) — the honest sessionId-deduped
    # count on the result line below is what the user sees instead.
    assert "Backfilling all available Claude Code history" not in flat
    # Both sessions (recent + old) should be backfilled with no window.
    # #675: simplified completion line — "N sessions, over M days."
    assert "2 sessions" in flat


# --- Explicit scripting flags -------------------------------------------------


def test_backfill_days_flag_skips_prompt(_isolated_claude_code_with_history, tmp_path):
    res = _run_claude_code(tmp_path, "--backfill-days", "7")
    assert res.exit_code == 0, res.output
    flat = _flat(res.output)
    assert "Backfill your Claude Code history:" not in flat
    assert "Backfilling the last 7 days" in flat


def test_backfill_all_flag_skips_prompt_and_note(
    _isolated_claude_code_with_history, tmp_path,
):
    res = _run_claude_code(tmp_path, "--backfill-all")
    assert res.exit_code == 0, res.output
    flat = _flat(res.output)
    assert "Backfill your Claude Code history:" not in flat
    assert "afterwards for your full history" not in flat
    assert "2 sessions" in flat  # #675: simplified completion line


def test_backfill_days_and_backfill_all_are_mutually_exclusive(
    _isolated_claude_code_with_history, tmp_path,
):
    res = _run_claude_code(tmp_path, "--backfill-days", "7", "--backfill-all")
    assert res.exit_code != 0
    assert "Use either --backfill-days or --backfill-all" in _flat(res.output)


def test_backfill_days_must_be_positive(_isolated_claude_code_with_history, tmp_path):
    res = _run_claude_code(tmp_path, "--backfill-days", "0")
    assert res.exit_code != 0
    assert "--backfill-days must be > 0" in _flat(res.output)


# --- Big-scope heads-up (removed) ---------------------------------------------
# The "~N sessions in scope — this may take a few minutes" heads-up was removed
# (founder review, demo trim). It counted transcript FILES, which over-reports
# the true sessionId-deduped session count (Critical Rule 34) — printing e.g.
# ~1,298 when only 27 sessions actually backfilled. The honest post-backfill
# "N total sessions" result line is the only session count the user sees now.


def test_no_sessions_in_scope_headsup_is_ever_printed(
    _isolated_claude_code_with_history, tmp_path,
):
    """The transcript-FILE-count heads-up line must never appear (Critical
    Rule 34: a file count over-reports the true session count)."""
    res = _run_claude_code(tmp_path, "--backfill-all")
    assert res.exit_code == 0, res.output
    assert "sessions in scope" not in _flat(res.output)


# --- since actually reaches ingest_claude_code --------------------------------


def test_default_window_actually_filters_old_session(
    _isolated_claude_code_with_history, tmp_path,
):
    res = _run_claude_code(tmp_path)
    assert res.exit_code == 0, res.output
    # Only the recent (5-days-ago) session is within a 30-day-from-now window;
    # the 2020 session must be excluded from the backfilled total.
    # #675: simplified completion line — "N sessions, over M days."
    assert "1 session" in _flat(res.output)


# --- Backfill completion line carries no dollar figure (#675) -----------------
# The old "N total sessions ... $N total spend" summary was simplified to a
# single "✓ Claude Code sessions backfilled: N sessions, over M days." line,
# which drops the spend detail entirely — so it can never leak a dollar figure
# on any plan tier, subscription or api alike.


def test_backfill_completion_line_hides_spend_for_subscription_plan(
    _isolated_claude_code_with_history, tmp_path,
):
    """A Pro/Max user just declared a flat-fee subscription — the completion
    line must not answer with "$N total spend" (core/framing.py suppresses
    dollar figures for subscription tiers on every surface)."""
    res = _run_claude_code(tmp_path, "--plan", "max_20x", input_str="")
    assert res.exit_code == 0, res.output
    assert "session" in _flat(res.output)  # backfill itself still reported
    assert "total spend" not in res.output


def test_backfill_completion_line_hides_spend_for_api_plan_too(
    _isolated_claude_code_with_history, tmp_path,
):
    """#675: the simplified completion line drops the dollar detail for EVERY
    plan tier — an api user's spend is now surfaced by `tj cost` / `tj optimize`,
    not on the onboarding payoff screen."""
    res = _run_claude_code(tmp_path, "--plan", "api", input_str="0\n")
    assert res.exit_code == 0, res.output
    assert "session" in _flat(res.output)  # backfill still reported
    assert "total spend" not in res.output
