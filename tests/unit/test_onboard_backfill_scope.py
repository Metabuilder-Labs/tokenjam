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
from pathlib import Path

import pytest
from click.testing import CliRunner

import tokenjam.core.backfill as backfill_mod
from tokenjam.cli.cmd_onboard import DEFAULT_BACKFILL_DAYS, cmd_onboard


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
                      "2026-06-25T10:00:00.000Z")
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
            ["--claude-code", "--no-daemon", "--project", "testproj", *extra_args],
            input=input_str, obj={},
        )


# --- Non-interactive default (no TTY, no explicit flag) ----------------------


def test_non_interactive_takes_default_window_and_notes_it(
    _isolated_claude_code_with_history, tmp_path,
):
    # CliRunner's stdin is never a tty, and `_is_interactive()` isn't patched
    # here — so this exercises the real non-interactive path.
    res = _run_claude_code(tmp_path)
    assert res.exit_code == 0, res.output
    flat = _flat(res.output)
    assert (
        f"Non-interactive: backfilling the last {DEFAULT_BACKFILL_DAYS} days" in flat
    )
    assert "by default" in flat
    assert "tj backfill claude-code" in flat
    # The old session (2020) is outside the 30-day default window, only the
    # recent one should have been counted as "in scope".
    assert "1 new" in res.output or "1 total session" in res.output


# --- Interactive prompt (menu) ------------------------------------------------


def test_interactive_prompt_shown_and_recent_choice_notes_complete_later(
    _isolated_claude_code_with_history, tmp_path, monkeypatch,
):
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._is_interactive", lambda: True)
    # Plan choice "3", budget "0", THEN backfill-scope choice "1" (recent).
    res = _run_claude_code(tmp_path, input_str="3\n0\n1\n")
    assert res.exit_code == 0, res.output
    flat = _flat(res.output)
    assert "Backfill your Claude Code history:" in flat
    assert f"Last {DEFAULT_BACKFILL_DAYS} days" in flat
    assert "Everything" in flat
    assert "Run `tj backfill claude-code` afterwards for your full history" in flat


def test_interactive_prompt_everything_choice_skips_complete_later_note(
    _isolated_claude_code_with_history, tmp_path, monkeypatch,
):
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._is_interactive", lambda: True)
    res = _run_claude_code(tmp_path, input_str="3\n0\n2\n")
    assert res.exit_code == 0, res.output
    flat = _flat(res.output)
    assert "afterwards for your full history" not in flat
    # Both sessions (recent + old) should be backfilled with no window.
    assert "2 total session" in flat


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
    assert "2 total session" in flat


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


# --- Big-scope heads-up --------------------------------------------------------


def test_headsup_printed_when_scope_exceeds_threshold(
    _isolated_claude_code_with_history, tmp_path, monkeypatch,
):
    monkeypatch.setattr("tokenjam.cli.cmd_onboard._BACKFILL_HEADSUP_THRESHOLD", 1)
    res = _run_claude_code(tmp_path, "--backfill-all")
    assert res.exit_code == 0, res.output
    flat = _flat(res.output)
    assert "sessions in scope" in flat
    assert "this may take a few minutes" in flat


def test_no_headsup_below_threshold(_isolated_claude_code_with_history, tmp_path):
    res = _run_claude_code(tmp_path, "--backfill-all")
    assert res.exit_code == 0, res.output
    assert "sessions in scope" not in _flat(res.output)


# --- since actually reaches ingest_claude_code --------------------------------


def test_default_window_actually_filters_old_session(
    _isolated_claude_code_with_history, tmp_path,
):
    res = _run_claude_code(tmp_path)
    assert res.exit_code == 0, res.output
    # Only the recent (2026-06-25) session is within a 30-day-from-now window;
    # the 2020 session must be excluded from the backfilled total.
    assert "1 total session" in res.output


# --- Backfill summary dollar line is plan-gated (framing discipline) ----------


def test_backfill_summary_hides_spend_for_subscription_plan(
    _isolated_claude_code_with_history, tmp_path,
):
    """A Pro/Max user just declared a flat-fee subscription — the backfill
    summary must not answer with "$N total spend" (core/framing.py suppresses
    dollar figures for subscription tiers on every other surface)."""
    res = _run_claude_code(tmp_path, "--plan", "max_20x", input_str="")
    assert res.exit_code == 0, res.output
    assert "total session" in res.output  # backfill itself still reported
    assert "total spend" not in res.output


def test_backfill_summary_keeps_spend_for_api_plan(
    _isolated_claude_code_with_history, tmp_path,
):
    """Per-token billing keeps the dollar line — it's their real marginal cost."""
    res = _run_claude_code(tmp_path, "--plan", "api", input_str="0\n")
    assert res.exit_code == 0, res.output
    assert "total spend" in res.output
