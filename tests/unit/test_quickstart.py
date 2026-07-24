"""Unit tests for the zero-install / zero-config first-run (`tj quickstart`, #6).

The contract under test: a user with NO prior setup runs one command and sees
quota composition + a session timeline straight from on-disk Claude Code JSONL —
with no daemon, no onboarding, and crucially **no on-disk DB** (the command uses
a transient in-memory backend and must never call `open_db`).
"""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.session_timeline import (
    compute_session_timeline,
    timeline_to_dict,
)


def _make_session_file(root: Path, session_id: str, cwd: str,
                       records: list[dict]) -> Path:
    project_dir = root / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def _assistant(uuid: str, session_id: str, cwd: str, ts: str, *,
               input_tokens: int = 500, output_tokens: int = 200,
               cache_read: int = 8000, cache_creation: int = 0) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": ts,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "model": "claude-sonnet-4-5-20250929",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            },
        },
    }


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    # Two sessions across two projects, recent timestamps.
    _make_session_file(root, "sess-a", "/Users/me/projA", [
        _assistant("a1", "sess-a", "/Users/me/projA", "2026-06-20T10:00:00.000Z"),
        _assistant("a2", "sess-a", "/Users/me/projA", "2026-06-20T10:05:00.000Z"),
    ])
    _make_session_file(root, "sess-b", "/Users/me/projB", [
        _assistant("b1", "sess-b", "/Users/me/projB", "2026-06-21T11:00:00.000Z",
                   cache_read=50000),
    ])
    return root


# ── Session-timeline core (pure logic over an in-memory DB) ──────────────────

def test_timeline_summarizes_backfilled_sessions(tmp_path):
    from tokenjam.core.backfill import ingest_claude_code

    root = _fixture_root(tmp_path)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root)

    timeline = compute_session_timeline(db.conn)

    assert timeline.has_data
    assert timeline.total_sessions == 2
    assert timeline.project_count == 2
    # Most-recent first.
    assert timeline.sessions[0].started_at >= timeline.sessions[-1].started_at
    # Project label is derived from the claude-code-<name> agent_id.
    projects = {s.project for s in timeline.sessions}
    assert "proja" in projects and "projb" in projects


def test_timeline_reread_share_reflects_cache_reads(tmp_path):
    from tokenjam.core.backfill import ingest_claude_code

    root = _fixture_root(tmp_path)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root)

    timeline = compute_session_timeline(db.conn)
    for s in timeline.sessions:
        # Every fixture turn has cache reads, so re-read share is > 0.
        assert s.reread_share > 0
        assert s.total_tokens >= s.cache_tokens


def test_timeline_to_dict_is_json_serialisable(tmp_path):
    from tokenjam.core.backfill import ingest_claude_code

    root = _fixture_root(tmp_path)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root)

    payload = timeline_to_dict(compute_session_timeline(db.conn))
    # Round-trips through json without error.
    round_tripped = json.loads(json.dumps(payload, default=str))
    assert round_tripped["total_sessions"] == 2
    assert len(round_tripped["sessions"]) == 2


def test_timeline_empty_db_has_no_data():
    db = InMemoryBackend()
    timeline = compute_session_timeline(db.conn)
    assert not timeline.has_data
    assert timeline.total_sessions == 0


# ── CLI: the zero-setup first run, with NO on-disk DB ────────────────────────

def _invoke_quickstart(args):
    """Run the zero-install report command directly.

    It has no public/typeable name on the `cli` group — `cli/main.py`'s
    no-subcommand branch invokes it via `ctx.invoke` only when the npm
    wrapper's `TJ_NPX_ZERO_INSTALL_REPORT` env var is set — so tests invoke
    the underlying `click.Command` object directly rather than through
    `cli`'s subcommand dispatch. The whole point of the command is that it
    never opens the on-disk DB or contacts the daemon — it manages its own
    transient in-memory backend.
    """
    from tokenjam.cli.cmd_quickstart import cmd_quickstart

    return CliRunner().invoke(cmd_quickstart, args)


def test_quickstart_renders_without_daemon_or_ondisk_db(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    # Leads with the reads-your-local-logs framing.
    assert "where your quota actually goes" in result.output
    assert "~/.claude/projects" in result.output
    # Both halves of the first-run value are present.
    assert "quota" in result.output.lower()
    assert "Session timeline" in result.output
    # The opt-in "go deeper" pointer prints a CTA — see the two footer tests
    # below for the ephemeral (`npx tokenjam onboard`) vs installed (`tj
    # onboard`) forms (#507). Here we just assert an onboard CTA is present.
    assert "onboard" in result.output
    # The outro sells the local dashboard (#120) exactly once — the most
    # product-looking asset shouldn't be invisible at the conversion moment,
    # but a second, redundant mention right under the CTA was dropped (#436
    # review) to keep the outro tight and consistent with the npx-form CTA.
    assert result.output.count("dashboard") == 1
    assert "Lens" in result.output


# ── "Go deeper" footer CTA is context-aware (#507) ─────────────────────────
#
# Bare `npx tokenjam` / `uvx --from tokenjam tj` runs quickstart from a
# throwaway uvx/pipx-run cache → the CTA is the zero-install `npx tokenjam
# onboard`. But when quickstart runs from an already-installed `tj` binary the
# user obviously has it installed → the CTA drops the `npx tokenjam` prefix and
# points straight at `tj onboard`. `_go_deeper_command()` picks based on
# `cmd_onboard._is_ephemeral_runner()`.


def test_go_deeper_footer_ephemeral_runner_shows_npx_cta():
    import unittest.mock as mock

    from tokenjam.cli.cmd_quickstart import _go_deeper_command

    with mock.patch("tokenjam.cli.cmd_onboard._is_ephemeral_runner", return_value=True):
        assert _go_deeper_command() == "npx tokenjam onboard"


def test_go_deeper_footer_installed_binary_shows_tj_cta():
    import unittest.mock as mock

    from tokenjam.cli.cmd_quickstart import _go_deeper_command

    with mock.patch("tokenjam.cli.cmd_onboard._is_ephemeral_runner", return_value=False):
        assert _go_deeper_command() == "tj onboard"


# ── Quota-weighted headline + no named-session reclaim list (#119) ──────────
#
# The headline used to report a RAW token share as "quota" (mixing the two
# framings) and named individual ended sessions as "actionable" — but a user
# never returns to a session closed days ago, so a per-session retrospective
# callout is unactionable noise. The headline must now read as quota-weighted,
# and the default output must never name a past session.

def test_quickstart_headline_reads_as_quota_not_raw_tokens(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "of your quota went to" in result.output
    # The old raw-token wording is gone.
    assert "of your tokens went to" not in result.output


def _heavy_reread_fixture_root(tmp_path: Path) -> Path:
    """A session with one huge-cache-read turn — clears the compact-candidate
    thresholds (>= 200k re-read tokens, >= 80% re-read share) so the aggregate
    reclaim line renders."""
    root = tmp_path / "projects"
    _make_session_file(root, "sess-heavy", "/Users/me/projHeavy", [
        _assistant("h1", "sess-heavy", "/Users/me/projHeavy",
                   "2026-06-20T10:00:00.000Z",
                   input_tokens=500, output_tokens=200, cache_read=300_000),
    ])
    return root


def test_quickstart_reclaim_section_is_aggregate_not_named_sessions(tmp_path):
    root = _heavy_reread_fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    # Rich wraps panel text to console width and interleaves the panel's own
    # box-drawing border characters, so compare against normalized output
    # (whitespace-collapsed, borders stripped) rather than a raw substring.
    stripped = result.output.translate({ord(c): " " for c in "│╭╮╰╯─"})
    normalized = " ".join(stripped.split())

    assert result.exit_code == 0, result.output
    # The old per-session list (and its heading) is gone entirely.
    assert "Biggest reclaim opportunities" not in result.output
    # The aggregate reclaim line inside the quota-composition panel never
    # names a session (#119). Scope this to the panel itself: the SEPARATE
    # statusline live preview section (#438) legitimately names one session
    # as a concrete "what you'd have seen live" example — a whole-output
    # check would false-positive against that unrelated, later section.
    panel_text = result.output.split("With the statusline installed")[0]
    assert "sess-heavy" not in panel_text
    # An aggregate line takes its place — no named session, live-signal framing.
    assert "ran context-heavy enough to warrant a mid-session" in normalized
    assert "/compact" in normalized
    assert "a closed session can't be reclaimed" in normalized


def test_quickstart_no_compact_candidates_omits_reclaim_section(tmp_path):
    """A history with no context-heavy sessions renders no reclaim section at
    all (same gating as before — this isn't about forcing the line to show)."""
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "ran context-heavy enough to warrant a mid-session" not in result.output
    assert "Biggest reclaim opportunities" not in result.output


def test_quickstart_json_emits_both_views(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--json"])

    assert result.exit_code == 0, result.output
    # The JSON line is the last line (Rich logging may precede it on stderr).
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert "quota_composition" in payload
    assert "session_timeline" in payload
    assert payload["session_timeline"]["total_sessions"] == 2
    assert payload["backfill"]["sessions_ingested"] == 2


def test_quickstart_no_logs_is_graceful(tmp_path):
    missing = tmp_path / "does-not-exist"
    result = _invoke_quickstart(["--root", str(missing)])
    assert result.exit_code == 0, result.output
    assert "No Claude Code logs" in result.output


# ── Pre-ingest progress: ingest was previously the ONE silent stretch in the
# whole command (~40s dead cursor on a large history, nothing printed until
# after it returned). An honest status line now lands before ingest starts,
# and the shared streaming counter (`backfill_progress`) advances per session
# through to render. `--json` must stay byte-for-byte clean on stdout.

def test_quickstart_prints_pre_ingest_status_before_render(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "Reading your last 90 days of Claude Code history" in result.output
    assert "(most-recent 300 sessions)" in result.output
    # It's the FIRST thing printed -- ahead of the quota-composition panel,
    # not tacked on after ingest already finished.
    assert result.output.index("Reading your last 90 days") < result.output.index(
        "Where your quota goes"
    )


def test_quickstart_pre_ingest_status_omits_cap_when_full(tmp_path):
    """`--full` lifts the session cap (#13) -- the status line must not claim
    a "most-recent N sessions" scope that no longer applies."""
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--full"])

    assert result.exit_code == 0, result.output
    assert "Reading your last 90 days of Claude Code history…" in result.output
    assert "most-recent" not in result.output.split("Where your quota goes")[0]


def test_quickstart_json_stdout_stays_pure(tmp_path):
    """`--json` must be pipeable straight into a JSON parser: stdout carries
    ONLY the JSON payload, never the pre-ingest status line or the streaming
    progress counter -- those route to stderr instead."""
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--json"])

    assert result.exit_code == 0, result.output
    # stdout parses as JSON on its own -- no leading/trailing progress noise.
    payload = json.loads(result.stdout.strip())
    assert "quota_composition" in payload
    assert payload["backfill"]["sessions_ingested"] == 2
    # The status line still printed -- just on stderr, never stdout.
    assert "Reading your last 90 days of Claude Code history" in result.stderr
    assert "Reading your last 90 days" not in result.stdout


def test_quickstart_advancing_counter_on_large_history(tmp_path):
    """On a large history the shared streaming counter keeps advancing
    through ingest (not just a single static pre-ingest line) -- non-TTY
    output (as under CliRunner) degrades to periodic plain prints every 100
    sessions, mirroring `tj onboard --claude-code`'s backfill counter."""
    root = _large_fixture_root(tmp_path, n_sessions=250)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "Backfilling 100/250 sessions" in result.output
    assert "Backfilling 200/250 sessions" in result.output


def _large_fixture_root(tmp_path: Path, n_sessions: int) -> Path:
    """A synthetic history with `n_sessions` sessions, two turns each, recent.

    Mtimes are staggered so the most-recent-first cap is deterministic: higher
    session index = newer file. This lets the cap tests assert *which* sessions
    survive without depending on filesystem write ordering.
    """
    import os

    root = tmp_path / "projects"
    base_ts = 1_900_000_000  # arbitrary recent epoch
    for i in range(n_sessions):
        sid = f"sess-{i:05d}"
        cwd = f"/Users/me/proj{i % 5}"
        path = _make_session_file(root, sid, cwd, [
            _assistant(f"{sid}-a", sid, cwd, "2026-06-20T10:00:00.000Z"),
            _assistant(f"{sid}-b", sid, cwd, "2026-06-20T10:05:00.000Z"),
        ])
        # Newer index => newer mtime, so the cap keeps the highest indices.
        os.utime(path, (base_ts + i, base_ts + i))
    return root


# ── First-run cap on a large history (#13) ───────────────────────────────────

def test_quickstart_caps_sessions_on_large_history(tmp_path):
    """The first-run path bounds its work: only `max_sessions` are ingested even
    when far more exist on disk, and the cap is flagged."""
    from tokenjam.core.backfill import ingest_claude_code

    root = _large_fixture_root(tmp_path, n_sessions=120)
    db = InMemoryBackend()
    result = ingest_claude_code(db, root=root, max_sessions=25)

    # Bounded work: exactly the cap was ingested, not the full 120.
    assert result.sessions_ingested == 25
    assert result.sessions_seen == 25
    assert result.limit_reached is True
    # The transient DB holds only the capped sessions' rows.
    (session_rows,) = db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
    assert session_rows == 25


def test_quickstart_cap_keeps_most_recent_sessions(tmp_path):
    """The cap retains the freshest sessions (by mtime), not arbitrary ones."""
    from tokenjam.core.backfill import ingest_claude_code

    root = _large_fixture_root(tmp_path, n_sessions=50)
    db = InMemoryBackend()
    ingest_claude_code(db, root=root, max_sessions=10)

    kept = {
        r[0] for r in db.conn.execute("SELECT session_id FROM sessions").fetchall()
    }
    # The 10 highest indices (newest mtimes) survive; older ones are dropped.
    assert kept == {f"sess-{i:05d}" for i in range(40, 50)}


def test_quickstart_no_cap_ingests_everything(tmp_path):
    """`max_sessions=None` (the full `tj backfill claude-code` path) is unbounded
    and never sets the limit flag — the cap is opt-in, not a regression."""
    from tokenjam.core.backfill import ingest_claude_code

    root = _large_fixture_root(tmp_path, n_sessions=40)
    db = InMemoryBackend()
    result = ingest_claude_code(db, root=root, max_sessions=None)

    assert result.sessions_ingested == 40
    assert result.limit_reached is False


def test_quickstart_below_cap_does_not_flag_limit(tmp_path):
    """A small history under the cap is not falsely reported as truncated."""
    from tokenjam.core.backfill import ingest_claude_code

    root = _large_fixture_root(tmp_path, n_sessions=5)
    db = InMemoryBackend()
    result = ingest_claude_code(db, root=root, max_sessions=300)

    assert result.sessions_ingested == 5
    assert result.limit_reached is False


def test_quickstart_cli_discloses_truncation(tmp_path, monkeypatch):
    """When the cap truncates, the CLI says so honestly and points at the full
    picture — no silent truncation that reads as 'this is everything'."""
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "DEFAULT_MAX_SESSIONS", 8)
    root = _large_fixture_root(tmp_path, n_sessions=30)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    # The disclosure names the cap and points at the full-history escape hatch.
    # Assert on stable, non-wrapping tokens only: Rich word-wraps the inline
    # "npx tokenjam onboard" CTA across a line break at narrow widths (it lands
    # as "npx tokenjam \nonboard"), so asserting that literal is flaky. The
    # footer-CTA form is covered by the dedicated footer tests above.
    assert "most-recent" in result.output
    assert "tj context" in result.output
    assert "full history" in result.output


def test_quickstart_cli_full_flag_lifts_cap(tmp_path, monkeypatch):
    """`--full` processes the whole history and emits no truncation note."""
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "DEFAULT_MAX_SESSIONS", 3)
    root = _large_fixture_root(tmp_path, n_sessions=12)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d",
                                 "--full", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["backfill"]["sessions_ingested"] == 12
    assert payload["backfill"]["limit_reached"] is False
    assert payload["backfill"]["max_sessions"] is None


def test_quickstart_json_reports_cap_metadata(tmp_path, monkeypatch):
    """JSON output exposes the cap state so machine consumers see the scoping."""
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "DEFAULT_MAX_SESSIONS", 6)
    root = _large_fixture_root(tmp_path, n_sessions=20)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["backfill"]["sessions_ingested"] == 6
    assert payload["backfill"]["limit_reached"] is True
    assert payload["backfill"]["max_sessions"] == 6


# ── Statusline live preview ("what you'd see live") ──────────────────────────
#
# `_display_model_name` reconstructs "Sonnet 4.5" from the raw transcript
# `claude-sonnet-4-5-20250929` id every fixture session below uses.

def _flat(output: str) -> str:
    """Collapse Rich's line-wrapping so long-sentence substring checks aren't
    sensitive to where the terminal happened to wrap a word."""
    return " ".join(output.split())


def _session_with_crossing(root: Path, session_id: str, cwd: str, base_date: str,
                            *, n_turns: int, crossing_turn: int) -> Path:
    """A synthetic session whose cumulative re-read %% stays low through
    `crossing_turn - 1`, then jumps hard enough that it crosses REREAD_WARN
    (70%%) starting exactly at `crossing_turn` (1-indexed) and stays crossed.
    """
    records = []
    for i in range(1, n_turns + 1):
        ts = f"{base_date}T10:{i:02d}:00.000Z"
        cache = 20 if i < crossing_turn else 100_000
        records.append(_assistant(
            f"{session_id}-{i}", session_id, cwd, ts,
            input_tokens=100, output_tokens=50, cache_read=cache,
        ))
    return _make_session_file(root, session_id, cwd, records)


def test_quickstart_preview_shows_most_recent_substantial_crossing_session(
    tmp_path, monkeypatch,
):
    from tokenjam.cli import cmd_quickstart as q
    from tokenjam.cli.cmd_statusline import format_status_line

    monkeypatch.setattr(q, "PREVIEW_MIN_TURNS", 3)
    root = tmp_path / "projects"
    # Older, more turns, but NOT more recent -> must lose to the recent one.
    _session_with_crossing(
        root, "sess-old", "/Users/me/projA", "2026-06-10",
        n_turns=10, crossing_turn=2,
    )
    # Recent AND substantial (5 >= PREVIEW_MIN_TURNS=3) -> wins on recency.
    recent_path = _session_with_crossing(
        root, "sess-recent", "/Users/me/projB", "2026-06-25",
        n_turns=5, crossing_turn=3,
    )

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "With the statusline installed" in flat
    assert "session sess-recent would have shown this at turn 3" in flat
    assert "tj onboard" in flat.split("With the statusline installed")[1]

    # Formatter reuse: the rendered line is byte-identical to what
    # format_status_line produces for the FIRST turn whose cumulative re-read
    # crosses the nudge threshold — never a hand-rolled duplicate of the live
    # statusline's text.
    from tokenjam.cli.cmd_statusline import REREAD_WARN
    from tokenjam.core.usage import iter_cumulative_usage
    with open(recent_path, encoding="utf-8") as fh:
        crossing = next(
            (turn_index, usage) for turn_index, _model, usage in iter_cumulative_usage(fh)
            if usage.total and 100.0 * usage.cache_read_tokens / usage.total >= REREAD_WARN
        )
    turn_index, usage = crossing
    assert turn_index == 3
    total = usage.total
    reread_pct = 100.0 * usage.cache_read_tokens / total
    expected_line = format_status_line("Sonnet 4.5", total, reread_pct)
    # Flatten Rich's line-wrapping (as the assertions above do) — the preview
    # reuses the same formatter, but the longer driver-conditional nudge can wrap
    # at the console width; the point is byte-identical *content*, not layout.
    assert _flat(expected_line) in _flat(result.output)


def test_quickstart_preview_stops_walking_after_first_substantial_candidate(
    tmp_path, monkeypatch,
):
    """Sessions are walked most-recent-first; once the most-recent SUBSTANTIAL
    crossing candidate is found, selection must stop rather than re-reading
    every remaining session's transcript — a large `~/.claude` history would
    otherwise blow past quickstart's fast-first-run budget."""
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "PREVIEW_MIN_TURNS", 3)
    root = tmp_path / "projects"
    # Most-recent session is ALREADY substantial + crossing -> must win
    # without inspecting any of the five older sessions below it.
    _session_with_crossing(
        root, "sess-newest", "/Users/me/projZ", "2026-06-28",
        n_turns=5, crossing_turn=2,
    )
    for i in range(5):
        _session_with_crossing(
            root, f"sess-old-{i}", f"/Users/me/proj{i}", f"2026-06-{10 + i:02d}",
            n_turns=5, crossing_turn=2,
        )

    calls: list[str] = []
    original_walk = q._walk_for_preview

    def _counting_walk(path):
        calls.append(path)
        return original_walk(path)

    monkeypatch.setattr(q, "_walk_for_preview", _counting_walk)

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    assert result.exit_code == 0, result.output
    assert "session sess-newest" in _flat(result.output)
    # Only the winning (most-recent, already-substantial) candidate's
    # transcript was walked -- the 5 older sessions were never opened.
    assert len(calls) == 1


def test_quickstart_preview_falls_back_to_largest_when_none_substantial(
    tmp_path, monkeypatch,
):
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "PREVIEW_MIN_TURNS", 50)  # neither session qualifies
    root = tmp_path / "projects"
    _session_with_crossing(
        root, "sess-recent", "/Users/me/projB", "2026-06-25",
        n_turns=5, crossing_turn=3,
    )
    _session_with_crossing(
        root, "sess-old", "/Users/me/projA", "2026-06-10",
        n_turns=8, crossing_turn=5,
    )

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    assert result.exit_code == 0, result.output
    # Neither is "substantial" -> falls back to the largest (by turns), not
    # simply the most recent.
    flat = _flat(result.output)
    assert "session sess-old would have shown this at turn 5" in flat


def test_quickstart_preview_omitted_when_no_session_crosses_threshold(tmp_path):
    root = tmp_path / "projects"
    # Healthy sessions: tiny cache reads relative to input/output, never near
    # the 70% nudge threshold.
    _make_session_file(root, "sess-a", "/Users/me/projA", [
        _assistant("a1", "sess-a", "/Users/me/projA", "2026-06-20T10:00:00.000Z",
                   input_tokens=1000, output_tokens=200, cache_read=10),
    ])

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    assert result.exit_code == 0, result.output
    assert "With the statusline installed" not in result.output


def test_quickstart_preview_omitted_when_no_sessions(tmp_path):
    missing = tmp_path / "does-not-exist"
    result = _invoke_quickstart(["--root", str(missing)])
    assert result.exit_code == 0, result.output
    assert "With the statusline installed" not in result.output


# ── Session Story teaser (reuses `tj session-story`'s own renderer) ─────────

def test_quickstart_session_story_teaser_appears_for_qualifying_session(
    tmp_path, monkeypatch,
):
    """The teaser renders for the SAME session the statusline preview already
    picked — no extra file globbing, just the shared renderer on that session's
    already-confirmed-readable transcript."""
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "PREVIEW_MIN_TURNS", 3)
    root = tmp_path / "projects"
    _session_with_crossing(
        root, "sess-recent", "/Users/me/projB", "2026-06-25",
        n_turns=5, crossing_turn=3,
    )

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    assert result.exit_code == 0, result.output
    assert "Session Story" in result.output
    assert "tj session-story" in result.output
    # The teaser follows the statusline preview in the output, not before it.
    assert result.output.index("With the statusline installed") < result.output.index(
        "Session Story"
    )


def test_quickstart_session_story_teaser_omitted_when_no_preview_candidate(tmp_path):
    """Silent degrade: no crossing session -> no statusline preview -> no
    Session Story teaser either (nothing to reuse the selection from)."""
    root = tmp_path / "projects"
    _make_session_file(root, "sess-a", "/Users/me/projA", [
        _assistant("a1", "sess-a", "/Users/me/projA", "2026-06-20T10:00:00.000Z",
                   input_tokens=1000, output_tokens=200, cache_read=10),
    ])

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    assert result.exit_code == 0, result.output
    assert "Session Story" not in result.output
