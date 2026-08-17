"""Unit tests for the zero-install / zero-config first-run (`tj quickstart`, #6).

The contract under test: a user with NO prior setup runs one command and sees a
MINIMAL screen straight from on-disk Claude Code JSONL — what tj read, one
avoidable-dollars sentence summed across the analyzers, and the `tj onboard`
pointer — with no daemon, no onboarding, and crucially **no on-disk DB** (the
command uses a transient in-memory backend and must never call `open_db`).

A large share of these tests exist to keep the screen minimal. The first run
used to carry a boxed quota-composition panel, a statusline preview, and a boxed
single-finding callout with evidence, a fix and two disclaimer paragraphs; all
of it is deliberately gone, and the "stays deleted" section below is what stops
it growing back one well-meaning panel at a time.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from click.testing import CliRunner

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.session_timeline import (
    compute_session_timeline,
    timeline_to_dict,
)

_NOW = datetime.now(timezone.utc)


def _date(month: int, day: int) -> str:
    """A `YYYY-MM-DD` fixture date anchored to test-execution time, not a
    hardcoded absolute literal.

    Every fixture below represents "recent" Claude Code history filtered
    through `--since 90d` (computed from real wall-clock `now()` inside the
    CLI). A fixed literal is a time bomb: once wall time passes
    `literal + 90d`, every assertion here starts failing with no code change
    involved -- the same class of bug fixed in
    `test_onboard_backfill_scope.py` and `test_transcript_sync.py`. The
    `(month, day)` pair only encodes the ORIGINAL relative spacing between
    fixture dates (e.g. `_date(6, 10)` is always ~20 days older than
    `_date(6, 30)`, `_date(6, 28)` ~2 days older); the actual calendar date
    floats with "now" so the gap to the `--since` cutoff never closes.
    """
    offset_days = (date(2026, 6, 30) - date(2026, month, day)).days
    return (_NOW - timedelta(days=offset_days)).strftime("%Y-%m-%d")


def _ts(month: int, day: int, time_of_day: str) -> str:
    """Full `...Z` timestamp for `_date(month, day)` at a given time-of-day
    (e.g. `"10:05:00.000"`)."""
    return f"{_date(month, day)}T{time_of_day}Z"


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
        _assistant("a1", "sess-a", "/Users/me/projA", _ts(6, 20, "10:00:00.000")),
        _assistant("a2", "sess-a", "/Users/me/projA", _ts(6, 20, "10:05:00.000")),
    ])
    _make_session_file(root, "sess-b", "/Users/me/projB", [
        _assistant("b1", "sess-b", "/Users/me/projB", _ts(6, 21, "11:00:00.000"),
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


def test_timeline_total_tokens_includes_cache_write_tokens():
    """`total_tokens` (both the per-session figure and the window aggregate)
    must sum all four token types, not just input+output+cache-read. Before
    the fix this silently understated spend by dropping cache_write_tokens
    (Cache token types in aggregates, root CLAUDE.md)."""
    from tests.factories import make_session

    db = InMemoryBackend()
    db.upsert_session(make_session(
        session_id="s1", input_tokens=2, output_tokens=465,
        cache_tokens=243597, cache_write_tokens=209000,
    ))

    timeline = compute_session_timeline(db.conn)

    expected = 2 + 465 + 243597 + 209000
    assert timeline.total_tokens == expected
    assert timeline.sessions[0].total_tokens == expected
    assert timeline.sessions[0].cache_write_tokens == 209000


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


def _assistant_with_unknown_model(uuid: str, session_id: str, cwd: str, ts: str) -> dict:
    record = _assistant(uuid, session_id, cwd, ts)
    record["message"]["model"] = "totally-unknown-model-xyz"
    return record


def test_quickstart_prints_unknown_model_pricing_warning_after_report(tmp_path):
    """Issue #585: pricing warnings encountered during ingest must land after
    the report body, not ahead of it."""
    root = tmp_path / "projects"
    _make_session_file(root, "sess-unknown", "/Users/me/projUnknown", [
        _assistant_with_unknown_model(
            "u1", "sess-unknown", "/Users/me/projUnknown",
            _ts(6, 20, "10:00:00.000"),
        ),
    ])

    import tokenjam.core.cost as cost_mod

    cost_mod._UNKNOWN_MODEL_WARNED.clear()
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "No pricing data for anthropic/totally-unknown-model-xyz" in result.output
    assert result.output.index("TokenJam reads your") < result.output.index(
        "No pricing data for anthropic/totally-unknown-model-xyz"
    )


def test_quickstart_json_keeps_pricing_warning_on_stderr(tmp_path):
    root = tmp_path / "projects"
    _make_session_file(root, "sess-unknown", "/Users/me/projUnknown", [
        _assistant_with_unknown_model(
            "u1", "sess-unknown", "/Users/me/projUnknown",
            _ts(6, 20, "10:00:00.000"),
        ),
    ])

    import tokenjam.core.cost as cost_mod

    cost_mod._UNKNOWN_MODEL_WARNED.clear()
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--json"])

    assert result.exit_code == 0, result.output
    assert "No pricing data for anthropic/totally-unknown-model-xyz" in result.stderr
    assert "No pricing data for anthropic/totally-unknown-model-xyz" not in result.stdout


def test_quickstart_renders_without_daemon_or_ondisk_db(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    # Leads with the reads-your-local-logs framing. Compared against the
    # flattened output: Rich wraps the lead to console width, so the phrase
    # can straddle a line break.
    flat = _flat(result.output)
    assert "TokenJam reads your ~/.claude/projects/*.jsonl session logs." in flat
    # The per-session timeline table and the window "Totals:" line are both gone.
    assert "Session timeline" not in result.output
    assert "Totals:" not in result.output
    # The onboard CTA — see the two footer tests below for the ephemeral (`npx
    # tokenjam onboard`) vs installed (`tj onboard`) forms (#507).
    assert "onboard" in result.output
    # The closing block describes what SETTING UP tokenjam gets you. It sells
    # the local dashboard exactly once, by name, and does not restate the
    # figure the screen has already made its point about.
    assert result.output.count("dashboard") == 1
    assert "Lens" in flat
    assert "review and apply fixes" in flat
    assert "Runs on your machine. No signup." in flat


# ── The onboard CTA is the npx form, unconditionally ───────────────────────
#
# This screen is reachable from exactly ONE place: `cli/main.py`'s no-subcommand
# branch, gated on `TJ_NPX_ZERO_INSTALL_REPORT`, which only the npm wrapper
# (`npm-wrapper/bin/tj.js`) sets, and only on a bare `npx tokenjam`. An installed
# user running bare `tj` gets the home screen and never lands here. So every
# reader of this screen arrived through `npx`, and a bare `tj ...` instruction
# would name a binary they may not have.


def test_the_screen_only_names_commands_a_bare_npx_user_can_run(tmp_path):
    """No instruction on this screen may require a `tj` on PATH.

    Asserted structurally rather than as a single string match: any `tj `
    occurrence that is not part of `npx tokenjam ...` is the regression, whether
    it arrives as a CTA, a cross-reference (`tj context`, `tj optimize`) or a
    typeable line.
    """
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "npx tokenjam onboard" in flat
    # Strip every legitimate `npx tokenjam ...` form, then no bare `tj ` may
    # survive anywhere on the screen.
    residual = flat.replace("npx tokenjam ", "")
    assert "tj " not in residual, f"bare `tj` instruction on the npx screen: {residual!r}"


def test_the_onboard_cta_does_not_branch_on_the_running_binary(tmp_path, monkeypatch):
    """The CTA used to drop the prefix whenever the process was NOT running from
    a throwaway uvx/pipx cache. That probe answers "was this launched from a
    persistent install", which is a different question from "does this user have
    `tj` on PATH" — the wrapper's third runner IS an already-installed `tj`, so
    a real npx user could be handed an instruction that is not how they invoked
    the tool. The CTA must be the same either way.
    """
    import tokenjam.cli.cmd_onboard as onboard

    monkeypatch.setattr(onboard, "_is_ephemeral_runner", lambda: False)
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "Run npx tokenjam onboard to set up TokenJam." in flat
    # Named once, inline. The standalone indented repeat below it was cut: the
    # screen said the same command twice.
    assert flat.count("npx tokenjam onboard") == 1


def _heavy_reread_fixture_root(tmp_path: Path) -> Path:
    """A session with one huge-cache-read turn.

    Under the old design this cleared the compact-candidate thresholds and
    produced the busiest possible first-run screen; it is kept because it is
    still the fixture most likely to make a re-introduced quota panel fire.
    """
    root = tmp_path / "projects"
    _make_session_file(root, "sess-heavy", "/Users/me/projHeavy", [
        _assistant("h1", "sess-heavy", "/Users/me/projHeavy",
                   _ts(6, 20, "10:00:00.000"),
                   input_tokens=500, output_tokens=200, cache_read=300_000),
    ])
    return root


# ── What stays deleted ─────────────────────────────────────────────────────
#
# The first run is a hook, not a report. These assertions are the guard rail on
# that decision: each string below was on the screen and was removed on purpose,
# so a regression re-adds a named string rather than a vague "it got busier".

_DELETED_FROM_THE_FIRST_RUN = (
    # The quota-composition panel and everything inside it.
    "Where your quota goes",
    "Quota composition",
    "of your quota went to",
    "re-reading context",
    "net-new work",
    "Re-read:",
    "New work:",
    "ran context-heavy enough to warrant a mid-session",
    "/compact",
    "a closed session can't be reclaimed",
    "implied API value",
    "Totals:",
    # The statusline live preview.
    "With the statusline installed",
    "would have shown this at turn",
    "for zero model tokens",
    # The single-finding callout, its framing and its disclaimer wall.
    "What that already cost you",
    "of that was avoidable",
    "Fix:",
    "Estimated, correlational figure",
    "not a causal savings claim",
    "Review the evidence before changing anything",
    "billed at a reduced rate",
    "cache writes",
    "not guaranteed savings",
    # The old outro preamble. ("No signup" is NOT on this list: the cut string
    # was the whole "Go deeper: live capture, Lens ... No signup:" preamble, and
    # a plain "Runs on your machine. No signup." sentence is now the approved
    # closing line.)
    "Go deeper",
    # Framing the founder cut outright.
    "quota",
    "wasted",
)


def test_first_run_carries_none_of_the_deleted_framing(tmp_path):
    """The busiest fixture available still renders none of the removed copy."""
    root = _heavy_reread_fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    for phrase in _DELETED_FROM_THE_FIRST_RUN:
        assert phrase not in flat, f"deleted first-run copy is back: {phrase!r}"


def test_first_run_draws_no_boxes(tmp_path):
    """No Rich `Panel`, no borders. Plain lines and blank-line separation only.

    Asserted on the box-drawing glyphs themselves rather than on `Panel` usage,
    because a table, a rule or a hand-rolled border would read identically to a
    user and is the same regression.
    """
    root = _heavy_reread_fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    for glyph in "│╭╮╰╯─┌┐└┘├┤┬┴┼━┃":
        assert glyph not in result.output, f"box-drawing glyph rendered: {glyph!r}"


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
    # The cap is disclosed WITHOUT a number: it is a file budget, and a file
    # count is not a session count (every Task dispatch writes its own
    # subagents/agent-*.jsonl sharing the parent's session_id, so the file
    # count runs roughly double). One session count reaches this screen and
    # the report is the one that makes it.
    assert "(capped for a fast first run)" in result.output
    # It's the FIRST thing printed -- ahead of the report itself, not tacked
    # on after ingest already finished.
    assert result.output.index("Reading your last 90 days") < result.output.index(
        "TokenJam reads your"
    )


def test_quickstart_pre_ingest_status_omits_cap_when_full(tmp_path):
    """`--full` lifts the session cap (#13) -- the status line must not claim
    a "most-recent N sessions" scope that no longer applies."""
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d", "--full"])

    assert result.exit_code == 0, result.output
    assert "Reading your last 90 days of Claude Code history…" in result.output
    # Scoped to the status line: the report's own "Showing your most-recent N
    # sessions." is about the window, not the cap, and stands either way.
    status_line = result.output.split("\n")[0]
    assert "capped" not in status_line


def test_exactly_one_session_count_reaches_the_screen(tmp_path):
    """The pre-ingest header and the report cannot disagree, because only one
    of them states a session count at all.

    The header used to announce a `.jsonl` FILE count labelled "sessions". A
    Claude Code session is more than one file: every Task dispatch writes its
    own `subagents/agent-*.jsonl` sharing the parent's `session_id`, so on a
    real corpus the header read roughly double the report's number and the two
    looked like two answers to one question. Filtering the pre-scan to
    main-thread files lands on the report's number today, but the pre-scan
    filters by FILE MTIME before parsing while the report filters by SPAN
    TIMESTAMP after it, so agreement is not structural. A missing number is
    fine; two numbers that disagree is the bug.
    """
    import re as _re_local

    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    header, _, report = flat.partition("TokenJam reads your")
    # Nothing before the report claims a session total.
    assert "sessions" not in header
    assert not _re_local.search(r"\d+\s*/\s*\d+", header), (
        f"a total slipped back into the pre-ingest counter: {header!r}")
    # And the report still states exactly one.
    assert len(set(_re_local.findall(r"(\d+) sessions[.,]", report))) <= 1


def test_the_progress_counter_never_calls_files_sessions(tmp_path):
    """It counts `.jsonl` files walked, including subagent transcripts, so it
    says "transcripts". Ingest coverage is unchanged: subagent transcripts are
    still read, and must be."""
    root = _large_fixture_root(tmp_path, n_sessions=120)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "Backfilling" in result.output
    for line in result.output.splitlines():
        if "Backfilling" in line:
            assert "sessions" not in line, f"file count labelled sessions: {line!r}"


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
    assert "Backfilling 100 transcripts" in result.output
    assert "Backfilling 200 transcripts" in result.output
    # No denominator: the only cheap pre-count is of FILES, and this screen
    # states exactly one session count (the report's).
    assert "/250" not in result.output


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
            _assistant(f"{sid}-a", sid, cwd, _ts(6, 20, "10:00:00.000")),
            _assistant(f"{sid}-b", sid, cwd, _ts(6, 20, "10:05:00.000")),
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
    flat = _flat(result.output)
    # The cap is disclosed by the pre-ingest status line without a number, and
    # the screen points at the full-history escape hatch.
    assert "(capped for a fast first run)" in flat
    assert "to set up TokenJam." in flat
    # EXACTLY ONE session count reaches the screen, and both sentences that
    # state a population quote it. A figure summed over one population beside a
    # count from another is the defect this pairing exists to prevent.
    import re as _re_local
    counts = set(_re_local.findall(r"(\d+) sessions[.,]", flat))
    assert len(counts) == 1, f"more than one session population on screen: {counts}"


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


# ── Output-shape helpers ────────────────────────────────────────────────────


def _flat(output: str) -> str:
    """Collapse Rich's line-wrapping so long-sentence substring checks aren't
    sensitive to where the terminal happened to wrap a word."""
    return " ".join(output.split())


def _session_with_crossing(root: Path, session_id: str, cwd: str, base_date: str,
                            *, n_turns: int, crossing_turn: int) -> Path:
    """A synthetic session whose cumulative re-read %% stays low through
    `crossing_turn - 1`, then jumps hard.

    This used to be the fixture that made the statusline live-preview section
    render. The preview is deleted; the fixture survives as the shape most
    likely to bring it back, and is asserted against below.
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


def test_first_run_never_names_a_past_session(tmp_path):
    """The statusline preview was the one place the screen named a session id.

    It is gone, and with it the only reason this screen ever named one. A user
    never returns to a session closed days ago, so a per-session retrospective
    is unactionable on a first run either way.
    """
    root = tmp_path / "projects"
    _session_with_crossing(
        root, "sess-recent", "/Users/me/projB", _date(6, 25),
        n_turns=5, crossing_turn=3,
    )

    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "sess-recent" not in result.output
    assert "With the statusline installed" not in result.output


# ── The avoidable total: ONE clubbed figure across the analyzers ───────────
#
# The screen shows exactly one finding sentence: the avoidable dollars SUMMED
# across every analyzer enabled for this window's persona. It names no analyzer,
# shows no evidence and hands out no fix; that detail lives in the Review inbox
# behind `tj onboard`.
#
# Honesty rules under test, all from the repo CLAUDE.md field contract:
#   * `past_overspend_usd` is the canonical figure, observed over the window,
#     never paced and never summed with `observed_cost_usd`;
#   * the sum goes through `cost_proposals.past_overspend_rollup` — THE one
#     aggregate every other surface reads — so this screen and the dashboard
#     cannot disagree, and a duplicate signature is never counted twice;
#   * the enabled analyzer set is DERIVED (`COST_ANALYZERS` minus the runtime
#     bound, with the persona gate inside `build_report`), never transcribed;
#   * `None` means "not measured" and renders NO sentence; `0.0` means
#     "measured, nothing found" and renders an explicit empty state. A `$0.00`
#     styled as a finding is never printed, and neither claim is ever made
#     while the computation has not run.


def _cheapest_model_assistant(uuid: str, session_id: str, cwd: str,
                              ts: str) -> dict:
    """An assistant turn on the cheapest model tokenjam prices, so no analyzer
    has a cheaper alternative to propose."""
    record = _assistant(uuid, session_id, cwd, ts,
                        input_tokens=200, output_tokens=80, cache_read=400)
    record["message"]["model"] = "claude-haiku-4-5"
    return record


def _proposal(*, analyzer="deadweight", signature=None,
              title="Unused MCP server: posthog",
              evidence="0 tool calls across 12 sessions.",
              advise_text="Remove the `posthog` MCP server.",
              usd=None, tokens=None, caveat="Estimated, correlational figure."):
    from tokenjam.core.optimize.cost_proposals import CostProposal

    return CostProposal(
        kind="cost",
        analyzer=analyzer,
        signature=signature or f"cost:{analyzer}",
        title=title,
        target_key={},
        evidence=evidence,
        baseline={},
        advise_text=advise_text,
        past_overspend_usd=usd,
        past_overspend_tokens=tokens,
        caveat=caveat,
    )


class _StubWindow:
    def __init__(self, total_cost_usd: float, sessions: int = 12) -> None:
        self.total_cost_usd = total_cost_usd
        self.sessions = sessions


class _StubReport:
    def __init__(self, total_cost_usd: float, sessions: int = 12) -> None:
        self.window = _StubWindow(total_cost_usd, sessions)


def _patch_optimize(monkeypatch, proposals, window_cost=1000.0, sessions=12):
    """Stub the two optimize entry points `_compute_avoidable_total` imports.

    Both imports happen inside the function body, so patching the modules they
    resolve from is enough — no import-order games. `past_overspend_rollup` is
    deliberately NOT stubbed: the sum under test is the real one every other
    surface reads, and stubbing it would test nothing.
    """
    import tokenjam.core.optimize as opt
    import tokenjam.core.optimize.cost_proposals as cp

    monkeypatch.setattr(opt, "build_report",
                        lambda **kwargs: _StubReport(window_cost, sessions))
    monkeypatch.setattr(cp, "cost_proposals_from_report",
                        lambda *args, **kwargs: list(proposals))


def _compute(monkeypatch, proposals, window_cost=1000.0, sessions=12,
             fallback_sessions=0):
    from tokenjam.cli.cmd_quickstart import _compute_avoidable_total

    _patch_optimize(monkeypatch, proposals, window_cost=window_cost,
                    sessions=sessions)
    return _compute_avoidable_total(
        object(), _NOW - timedelta(days=30), _NOW,
        fallback_sessions=fallback_sessions,
    )


def test_avoidable_total_sums_every_analyzer_not_just_the_largest(monkeypatch):
    """The founder-mandated change: one clubbed figure across the analyzers.

    The screen used to show the single largest finding (42.5 here). It now
    shows the sum, because a first-run reader is being told what their history
    cost them in total, not which one card was biggest.
    """
    total = _compute(monkeypatch, [
        _proposal(analyzer="subagent", title="Right-size a subagent", usd=4.0,
                  tokens=100),
        _proposal(analyzer="deadweight", usd=42.5, tokens=900),
        _proposal(analyzer="summarize", title="Review 3 oversized files",
                  usd=9.0, tokens=300),
    ])

    assert total is not None
    assert total.usd == 55.5


def test_avoidable_total_dedupes_by_signature(monkeypatch):
    """The sum routes through `past_overspend_rollup`, which dedupes by the
    proposal `signature`, so a stale or duplicated card is counted once."""
    total = _compute(monkeypatch, [
        _proposal(analyzer="deadweight", signature="cost:deadweight:posthog",
                  usd=30.0),
        _proposal(analyzer="deadweight", signature="cost:deadweight:posthog",
                  usd=30.0),
        _proposal(analyzer="subagent", signature="cost:subagent", usd=5.0),
    ])

    assert total is not None
    assert total.usd == 35.0


def test_avoidable_total_folds_in_relearn_through_the_shared_gatherer(monkeypatch):
    """Quickstart's rollup is not a hand-rolled sum: it is whatever
    `inbox_contribution.gather_rollup_population` — the SAME gatherer
    `tj relearn cost-proposals` and `GET /relearn/cost-proposals` use — would
    return for the same cost proposals and the same relearn finding.

    Relearn is deliberately skipped from `build_report`'s own analyzer set on
    this screen today (runtime cost — see `_OVERSPEND_SKIP_ANALYZERS`), so
    `report.findings["relearn"]` is normally absent. This test constructs the
    stub report AS IF relearn had run (a `.findings["relearn"]` payload
    exactly like a live `RelearnFinding.asdict()` would produce), which is
    the shape the wiring has to handle correctly whether or not today's skip
    stands. That is the structural claim this pins: `_compute_avoidable_total`
    reaches relearn's money through the one shared gatherer rather than a
    parallel derivation that could quietly drop it, exactly as
    `cmd_quickstart`'s prior comment falsely claimed it already did.
    """
    from tokenjam.core.optimize import inbox_contribution
    from tokenjam.cli.cmd_quickstart import _compute_avoidable_total
    import tokenjam.core.optimize as opt
    import tokenjam.core.optimize.cost_proposals as cp

    relearn_finding = {
        # The FINDING-level window vocabulary — read by `contribution_window_
        # label` to pick which label matches this screen's window. Distinct
        # from each cluster's own `past_overspend_windows` below (the actual
        # bucketed figures); both have to carry the "30d" key.
        "past_overspend_windows": {"30d": {"window_days": 30}},
        "clusters": [
            {
                "signature": "relearn:retry-loop",
                "title": "Recurring retry-loop failure",
                "past_overspend_usd": 999.0,
                "past_overspend_tokens": 50_000,
                "past_overspend_windows": {
                    "30d": {
                        "window_days": 30,
                        "past_overspend_usd": 12.0,
                        "past_overspend_tokens": 4_000,
                        "past_reread_usd": 2.0,
                        "past_reread_tokens": 500,
                    },
                },
            },
        ],
    }

    class _StubReportWithRelearn(_StubReport):
        def __init__(self, total_cost_usd, sessions, findings):
            super().__init__(total_cost_usd, sessions)
            self.findings = findings

    proposals = [_proposal(analyzer="deadweight", usd=42.5, tokens=900)]
    monkeypatch.setattr(
        opt, "build_report",
        lambda **kwargs: _StubReportWithRelearn(
            1000.0, 12, {"relearn": relearn_finding},
        ),
    )
    monkeypatch.setattr(
        cp, "cost_proposals_from_report", lambda *a, **kw: list(proposals),
    )

    since, until = _NOW - timedelta(days=30), _NOW
    total = _compute_avoidable_total(object(), since, until, fallback_sessions=0)

    window_days = max((until - since).total_seconds() / 86400.0, 1.0)
    expected = inbox_contribution.gather_rollup_population(
        proposals, relearn_finding, window_days=window_days,
    )

    assert total is not None
    # 42.5 (deadweight) + 10.0 (relearn's 30d bucket, net of its 2.0 re-read
    # share) — relearn's money reached the total through the same gatherer.
    assert total.usd == expected["past_overspend_usd"] == 52.5
    assert "relearn" in total.contributors


def test_avoidable_total_drops_a_figure_larger_than_the_window_cost(monkeypatch):
    """A figure bigger than what the whole window cost is self-refuting: a
    reader can disprove it from their own billing. The over-ceiling proposal is
    DROPPED from the sum, never rescaled into a paced number."""
    total = _compute(monkeypatch, [
        _proposal(analyzer="deadweight", usd=357.0),
        _proposal(analyzer="summarize", title="Review 3 oversized files", usd=88.0),
    ], window_cost=245.0)

    assert total is not None
    assert total.usd == 88.0


def test_avoidable_total_is_zero_not_none_when_nothing_was_found(monkeypatch):
    """Measured and empty is a REAL answer, distinct from not measured.

    `0.0` renders an explicit "no avoidable spend found" sentence; `None`
    renders no sentence at all. Collapsing the two would either fabricate an
    all-clear the run never established, or hide a genuine all-clear.
    """
    empty = _compute(monkeypatch, [])
    unpriced = _compute(monkeypatch, [_proposal(usd=None, tokens=None)])

    assert empty is not None and empty.usd == 0.0
    assert unpriced is not None and unpriced.usd == 0.0


def test_avoidable_total_is_none_when_the_analyzers_raise(monkeypatch):
    """A first run must degrade, not crash — and an un-run computation must
    report itself as unknown rather than as zero."""
    import tokenjam.core.optimize as opt

    def _boom(**kwargs):
        raise RuntimeError("analyzer exploded")

    monkeypatch.setattr(opt, "build_report", _boom)

    from tokenjam.cli.cmd_quickstart import _compute_avoidable_total

    assert _compute_avoidable_total(
        object(), _NOW - timedelta(days=30), _NOW, fallback_sessions=12,
    ) is None


def test_avoidable_total_takes_its_population_from_the_analyzed_window(monkeypatch):
    """The session count travels ON the figure, and comes from the REPORT's own
    window, not from however many session files the ingest happened to load.

    The two genuinely differ on a real corpus: the ingest picks the most-recent
    N files by mtime while the report filters by span timestamp, so 300 ingested
    files can be 143 analyzed sessions. Quoting the ingest count beside a figure
    summed over the analyzed one is a mixed-population claim.
    """
    total = _compute(monkeypatch, [_proposal(usd=12.0)],
                     sessions=143, fallback_sessions=300)

    assert total is not None
    assert total.sessions == 143


def test_avoidable_total_falls_back_to_the_ingest_count_only_without_one(monkeypatch):
    """A report shape carrying no session count degrades to the ingest count
    rather than printing a bare `0 sessions`. There is no figure to mispair it
    with in that case, because the report produced none."""
    total = _compute(monkeypatch, [], sessions=0, fallback_sessions=300)

    assert total is not None
    assert total.sessions == 300


# ── The rendered screen ────────────────────────────────────────────────────


def _stub_total(monkeypatch, usd, sessions=300, contributors=("resend",)):
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(
        q, "_compute_avoidable_total",
        lambda *a, **k: q.AvoidableTotal(usd=usd, sessions=sessions,
                                         contributors=contributors),
    )


def test_the_clubbed_line_renders_once_and_names_no_analyzer(tmp_path, monkeypatch):
    _stub_total(monkeypatch, 635.0, sessions=300)
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert flat.count("was avoidable.") == 1
    assert "$635 in your last 300 sessions was avoidable." in flat
    # No per-analyzer title, evidence or fix on this screen.
    for name in ("downsize", "subagent", "resend", "relearn", "deadweight",
                 "summarize"):
        assert name not in flat


def test_a_single_session_population_reads_as_a_sentence(tmp_path, monkeypatch):
    """"your last 1 session" reads as a bug even though it is arithmetically
    right, so the count is dropped at n == 1."""
    _stub_total(monkeypatch, 4.0, sessions=1)
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "$4.00 in your last session was avoidable." in flat
    assert "your last 1 session" not in flat


def test_a_measured_empty_window_says_so_without_a_zero_figure(tmp_path, monkeypatch):
    _stub_total(monkeypatch, 0.0, sessions=42)
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "No avoidable spend found in your last 42 sessions." in flat
    # Never a `$0` styled like a finding.
    assert "$0" not in result.output
    assert "was avoidable" not in flat


def test_an_unknown_figure_renders_no_claim_in_either_direction(tmp_path, monkeypatch):
    """The empty-state string must be UNABLE to render while the figure is
    unknown: "we have not computed it" and "we computed it and found nothing"
    are different claims, and zero is the worst possible placeholder for the
    first (it reads as reassurance).
    """
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "_compute_avoidable_total", lambda *a, **k: None)
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "No avoidable spend found" not in flat
    assert "was avoidable" not in flat
    assert "$" not in result.output
    # The rest of the screen is unaffected: the reader still gets the pointer.
    assert "to set up TokenJam." in flat


def test_quickstart_degrades_cleanly_when_nothing_is_recoverable(tmp_path):
    """The REAL analyzer path on a corpus with nothing to report: no fabricated
    number, no `$0.00`, and the rest of the screen still renders.

    The fixture already runs on the cheapest model in the pricing table, so
    `downsize` has nothing to route it down to, and an isolated HOME (see
    `tests/conftest.py`) leaves every config-reading analyzer with nothing to
    find. No stubbing: this is the path a first-time user with a clean, small
    history actually takes.
    """
    root = tmp_path / "projects"
    _make_session_file(root, "sess-cheap", "/Users/me/projCheap", [
        _cheapest_model_assistant("c1", "sess-cheap", "/Users/me/projCheap",
                                  _ts(6, 20, "10:00:00.000")),
    ])
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "$0.00" not in result.output
    flat = _flat(result.output)
    assert "TokenJam reads your ~/.claude/projects/*.jsonl session logs." in flat
    assert "to set up TokenJam." in flat


# ── The screen is not Claude-Code-only ─────────────────────────────────────
#
# Standing rule: entry copy never reads as Claude Code only. The daemon mounts
# an OTLP receiver an instrumented SDK or API agent can post to, so the Claude
# Code transcripts this screen reads are not the whole product. The source named
# on screen is verified against the code below, because a source that does not
# work is a worse defect than the framing it was added to fix.


def test_the_closing_block_names_a_non_claude_code_source(tmp_path):
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "OTel" in flat
    assert "SDK or API agents send it" in flat


def test_the_screen_does_not_name_codex(tmp_path):
    """A real Codex parser exists and passes its own suites, which is exactly
    why a later reader will be tempted to "fix" this omission. Do not.

    Shipping-readiness is an operator call, not a code-presence one, and the
    operator's is that tokenjam is Claude Code only for now. Advertising a
    half-supported source on the FIRST screen a stranger sees is the specific
    defect the source sentence was verified against in the first place.
    """
    from tokenjam.cli.cmd_quickstart import _OTHER_SOURCES

    assert "codex" not in _OTHER_SOURCES.lower()

    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "codex" not in _flat(result.output).lower()


def test_the_otel_source_named_on_screen_really_receives():
    """The daemon mounts an OTLP receiver unconditionally, so an external
    OTel-instrumented app can post to it with no extra opt-in."""
    from tokenjam.api.routes.otlp import router

    paths = {route.path for route in router.routes}
    assert "/v1/traces" in paths
    assert "/v1/logs" in paths


def test_the_screen_does_not_claim_metrics_or_mcp_ingest(tmp_path):
    """Two things the old copy got wrong, kept wrong-proof.

    `POST /v1/metrics` is a stub that returns 200 and discards the body, so the
    copy says "spans", never "metrics". And `mcp/server.py` exposes only
    read/query and apply tools with no ingest tool, so the MCP server is not a
    source: the copy this replaced claimed SDK traffic arrives "from OTel spans
    or the tokenjam MCP server", and the second half was never true.
    """
    from tokenjam.cli.cmd_quickstart import _OTHER_SOURCES

    assert "metrics" not in _OTHER_SOURCES.lower()
    assert "mcp" not in _OTHER_SOURCES.lower()
    # Nor an "any OTel app" claim: the receiver is OTLP/HTTP JSON only (no
    # protobuf decoder, no gRPC listener), so a stock OTel SDK on its default
    # http/protobuf exporter gets a 400. The copy describes an agent you point
    # at tokenjam on purpose, which is exactly what works.
    assert "any " not in _OTHER_SOURCES.lower()

    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])
    flat = _flat(result.output).lower()
    assert "mcp" not in flat
    assert "metrics" not in flat


def test_the_closing_block_stays_four_lines(tmp_path, monkeypatch):
    """It describes what setting up tokenjam gets you. It must not grow into a
    feature list, and it must not restate the figure."""
    _stub_total(monkeypatch, 936.0, sessions=153)
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    block = result.output.split("Run npx tokenjam onboard")[1]
    assert len([ln for ln in block.splitlines() if ln.strip()]) <= 5  # 4 + wrap
    # No restatement of the figure or its population.
    assert "$" not in block
    assert "avoidable" not in block
    assert "153" not in block


# ── The explanatory sentence adapts to what actually contributed ───────────
#
# A bare dollar figure's likeliest misread is that it is what the sessions COST,
# so one sentence under it says what "avoidable" means and gestures at the
# mechanism. The gesture has to describe THIS run: a corpus whose figure is
# dominated by an unused MCP server must not be explained by "oversized models".
# That is the same defect class as printing a session count from a different
# population than the dollars.


def test_the_shape_clause_names_only_what_contributed(tmp_path, monkeypatch):
    """A deadweight-dominated corpus is described by ITS cause, not the usual
    model/context/failure trio."""
    _stub_total(monkeypatch, 877.0, sessions=155, contributors=("deadweight",))
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "MCP servers connected but never used" in flat
    # None of the shapes belonging to analyzers that contributed nothing.
    for absent in ("oversized models", "context re-sent every turn",
                   "mistakes that repeated without a fix",
                   "always-loaded files longer than they need to be"):
        assert absent not in flat, f"shape named without a contribution: {absent!r}"


def test_no_shape_is_ever_named_for_a_zero_contributor(monkeypatch):
    """Exhaustive over the shape map: for every analyzer, a run it did NOT
    contribute to must not carry its phrase.

    Asserted against the map itself rather than a hand-picked pair, so an
    analyzer added later is covered without editing this test.
    """
    from tokenjam.cli.cmd_quickstart import _ANALYZER_SHAPES, _shape_clause

    for analyzer, phrase in _ANALYZER_SHAPES.items():
        others = tuple(a for a in _ANALYZER_SHAPES if _ANALYZER_SHAPES[a] != phrase)
        clause = _shape_clause(others[:1])
        assert phrase not in clause, (
            f"{analyzer!r}'s shape appears with no contribution from it"
        )


def test_every_cost_analyzer_has_a_shape_phrase():
    """A cost analyzer with no phrase does not silently vanish from the
    sentence, but it does make the sentence vaguer than it needs to be. Keeping
    the map complete against the live registry is the cheap half of that."""
    from tokenjam.cli.cmd_quickstart import _ANALYZER_SHAPES
    from tokenjam.core.optimize.cost_proposals import COST_ANALYZERS

    missing = [name for name in COST_ANALYZERS if name not in _ANALYZER_SHAPES]
    assert not missing, f"cost analyzers with no plain-English shape: {missing}"


def test_more_contributors_than_fit_does_not_imply_exhaustiveness(monkeypatch):
    """Five contributors, three shapes shown. The sentence must read as a
    sample, never as the complete set of causes."""
    from tokenjam.cli.cmd_quickstart import _shape_clause

    clause = _shape_clause(
        ("resend", "summarize", "deadweight", "relearn", "verbosity"))

    assert clause.startswith(
        "That is the part a change to your setup would have removed, including ")
    assert ": " not in clause          # the exhaustive form's colon is absent
    assert clause.count(",") >= 3      # three shapes, comma-joined


def test_an_unmapped_contributor_also_drops_the_exhaustive_form(monkeypatch):
    """An analyzer this module has no phrase for degrades the sentence to the
    vaguer but still TRUE form, rather than to a confident list that omits it."""
    from tokenjam.cli.cmd_quickstart import _shape_clause

    assert "including" in _shape_clause(("resend", "an-analyzer-added-later"))


def test_the_shape_clause_collapses_duplicate_phrases(monkeypatch):
    """`downsize` and `subagent` are different findings that read as one shape
    to a user. The sentence says it once."""
    from tokenjam.cli.cmd_quickstart import _shape_clause

    clause = _shape_clause(("downsize", "subagent"))

    assert clause.count("oversized models") == 1
    assert "including" not in clause   # nothing was omitted, so it IS exhaustive


def test_the_meaning_only_fallback_holds_for_any_mix(monkeypatch):
    """When nothing maps, the sentence still does the more important of its two
    jobs: correcting "this is what my sessions cost"."""
    from tokenjam.cli.cmd_quickstart import _shape_clause

    clause = _shape_clause(("something-unmapped",))

    assert clause == ("That is the part a change to your setup would have "
                      "removed, not what your sessions cost.")


def test_no_explanation_when_the_figure_is_zero(tmp_path, monkeypatch):
    _stub_total(monkeypatch, 0.0, sessions=42, contributors=())
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "would have removed" not in _flat(result.output)


def test_no_explanation_when_the_figure_is_unknown(tmp_path, monkeypatch):
    from tokenjam.cli import cmd_quickstart as q

    monkeypatch.setattr(q, "_compute_avoidable_total", lambda *a, **k: None)
    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "would have removed" not in _flat(result.output)


def test_contributors_come_from_the_rollup_breakdown_biggest_first(monkeypatch):
    """The explanation and the figure are read off the SAME rollup, so they can
    never describe different sets, and the order is by contribution."""
    total = _compute(monkeypatch, [
        _proposal(analyzer="subagent", signature="cost:subagent", usd=4.0),
        _proposal(analyzer="resend", signature="cost:resend", usd=90.0),
        _proposal(analyzer="deadweight", signature="cost:deadweight", usd=30.0),
        # Token-only: explains a DOLLAR figure it contributed nothing to.
        _proposal(analyzer="verbosity", signature="cost:verbosity",
                  usd=None, tokens=5_000),
    ])

    assert total is not None
    assert total.contributors == ("resend", "deadweight", "subagent")


# ── The analyzer pass is not silent, and leaves nothing behind ─────────────
#
# Measured on a real corpus, ~14s elapses between the final backfill line and
# the first line of the report while the analyzers run. That read as a hang.


def test_the_analyzing_status_leaves_no_residue_on_the_screen(tmp_path):
    """Nothing the status line said may survive into the rendered report."""
    from tokenjam.cli.cmd_quickstart import _ANALYZING_STATUS

    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert _ANALYZING_STATUS not in result.output
    assert "Finding avoidable spend" not in result.output
    # And no spinner/live escape codes reach captured output.
    assert "\x1b[" not in result.output


def test_the_analyzing_status_makes_no_claim_about_the_figure(tmp_path):
    """It says what is being looked for, never how much was found. Unknown
    stays unknown until it is known, which is the same rule the figure itself
    already follows."""
    from tokenjam.cli.cmd_quickstart import _ANALYZING_STATUS

    lowered = _ANALYZING_STATUS.lower()
    assert "$" not in _ANALYZING_STATUS
    assert not any(ch.isdigit() for ch in _ANALYZING_STATUS)
    for claim in ("no ", "nothing", "found ", "avoidable spend found",
                  "saved", "wasted"):
        assert claim not in lowered, f"status line claims a result: {claim!r}"
    assert "—" not in _ANALYZING_STATUS


def test_the_transient_status_erases_itself_before_the_next_output():
    """The erase contract, asserted on real control codes: on a terminal the
    line is written and then cleared, so whatever renders next starts clean."""
    import io

    from rich.console import Console

    from tokenjam.cli.backfill_progress import transient_status

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, highlight=False, width=80)
    with transient_status("Working on it…", console=console):
        pass
    console.print("NEXT")

    out = buf.getvalue()
    # Rich moves up and clears the line before the following output lands.
    assert "\x1b[2K" in out
    assert out.index("\x1b[2K") < out.index("NEXT")


def test_the_transient_status_prints_nothing_on_a_non_terminal():
    """Piped output, CI, redirected logs: Rich cannot erase there, so the only
    choices are permanent residue or silence, and residue is worse. The
    backfill counter is the sign of life on that path."""
    import io

    from rich.console import Console

    from tokenjam.cli.backfill_progress import transient_status

    buf = io.StringIO()
    with transient_status("Working on it…", console=Console(file=buf, width=80)):
        pass

    assert buf.getvalue() == ""


# ── Colour discipline: near-monochrome, one accent ─────────────────────────
#
# The design reference is the Claude CLI: grey-forward body text with a single
# accent that means "typeable". On this screen the accent marks the dollar
# figure and the `onboard` command; nothing else on it is coloured at all, and
# no state colour (red / green / yellow / cyan) appears, because there is no
# genuine state here for one to mean.

_FORBIDDEN_SGR = (
    "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[36m",     # red green yellow cyan
    "\x1b[91m", "\x1b[92m", "\x1b[93m", "\x1b[96m",     # their bright forms
)


def _render_with_ansi(monkeypatch, usd, sessions=300):
    """Render the screen through a truecolor console and return raw ANSI.

    `CliRunner` gives a non-tty console that strips every style, so the colour
    contract cannot be asserted from its output. This drives `_render` directly
    against a forced-truecolor console instead.
    """
    import io

    from rich.console import Console

    from tokenjam.cli import cmd_quickstart as q
    from tokenjam.utils.theme import TJ_THEME

    buf = io.StringIO()
    monkeypatch.setattr(q, "console", Console(
        file=buf, force_terminal=True, color_system="truecolor",
        highlight=False, theme=TJ_THEME, width=100,
    ))
    q._render(q.AvoidableTotal(usd=usd, sessions=sessions))
    return buf.getvalue()


def test_the_screen_uses_the_accent_and_no_state_colour(monkeypatch):
    from tokenjam.utils.theme import ACCENT

    ansi = _render_with_ansi(monkeypatch, 635.0)

    r, g, b = (int(ACCENT[i:i + 2], 16) for i in (1, 3, 5))
    accent_sgr = f"38;2;{r};{g};{b}"
    # The accent is present, and on both the things that earn it: the dollar
    # figure and the inline typeable command.
    assert ansi.count(accent_sgr) >= 2
    for code in _FORBIDDEN_SGR:
        assert code not in ansi, f"state colour on the first-run screen: {code!r}"


def test_the_empty_state_sentence_is_muted_not_accented(monkeypatch):
    """The accent means "typeable". An empty state is neither typeable nor a
    figure, so the only accent left on the screen is the command."""
    from tokenjam.utils.theme import ACCENT

    ansi = _render_with_ansi(monkeypatch, 0.0)

    r, g, b = (int(ACCENT[i:i + 2], 16) for i in (1, 3, 5))
    # Only the inline command carries it; the empty-state sentence does not.
    assert ansi.count(f"38;2;{r};{g};{b}") == 1
    assert "No avoidable spend found" in " ".join(ansi.split())


def test_quickstart_user_facing_copy_has_no_em_dashes(tmp_path):
    """Standing copy rule for tokenjam: periods, semicolons or colons, never an
    em dash. The screen now renders only its OWN copy (no analyzer `advise_text`
    or `caveat` is surfaced verbatim any more), so this covers all of it.
    """
    root = _heavy_reread_fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
    assert "—" not in result.output


def test_overspend_analyzer_set_drops_only_relearn():
    """Runtime bound, not a second persona filter: `relearn` reports on
    `cost_of_waste_*` and leaves `past_overspend_*` at None, so it can never
    contribute to this sum, while costing the large majority of the analyzer time
    on a real corpus. Everything else in `COST_ANALYZERS` is passed straight
    through, and the persona gate stays inside `build_report`."""
    from tokenjam.cli.cmd_quickstart import _overspend_analyzers
    from tokenjam.core.optimize.cost_proposals import COST_ANALYZERS

    selected = _overspend_analyzers(COST_ANALYZERS)

    assert "relearn" not in selected
    assert selected == [n for n in COST_ANALYZERS if n != "relearn"]


def test_overspend_analyzer_set_keeps_deadweight_when_not_capped():
    """`deadweight` scans every matching transcript on disk regardless of
    `--max-sessions`, but when the ingest was NOT truncated its population
    is identical to what got ingested — no reason to drop it."""
    from tokenjam.cli.cmd_quickstart import _overspend_analyzers
    from tokenjam.core.optimize.cost_proposals import COST_ANALYZERS

    selected = _overspend_analyzers(COST_ANALYZERS, population_capped=False)

    assert "deadweight" in selected


def test_overspend_analyzer_set_drops_deadweight_when_population_capped():
    """When quickstart's session ingest actually truncated at the cap,
    `deadweight`'s own disk scan reasons over strictly MORE sessions than the
    ones ingested and rendered — a population mismatch the magnitude ceiling
    alone can't catch (a smaller out-of-population figure still clears it).
    Excluding the analyzer, not rescaling its figure, is the only honest
    move here."""
    from tokenjam.cli.cmd_quickstart import _overspend_analyzers
    from tokenjam.core.optimize.cost_proposals import COST_ANALYZERS

    selected = _overspend_analyzers(COST_ANALYZERS, population_capped=True)

    assert "deadweight" not in selected
    assert "relearn" not in selected
    assert selected == [n for n in COST_ANALYZERS if n not in ("relearn", "deadweight")]


def test_avoidable_total_excludes_deadweight_from_the_report_when_population_capped(monkeypatch):
    """`_compute_avoidable_total(population_capped=True)` must not even ASK
    `build_report` to run `deadweight` — filtering its proposal out after
    the fact would still let evidence/cost from out-of-window sessions leak
    into the report object."""
    import tokenjam.core.optimize as opt
    from tokenjam.cli.cmd_quickstart import _compute_avoidable_total

    captured: dict = {}

    def _fake_build_report(**kwargs):
        captured["findings"] = kwargs.get("findings")
        return _StubReport(1000.0)

    monkeypatch.setattr(opt, "build_report", _fake_build_report)
    import tokenjam.core.optimize.cost_proposals as cp
    monkeypatch.setattr(cp, "cost_proposals_from_report", lambda *a, **k: [])

    _compute_avoidable_total(
        object(), _NOW - timedelta(days=30), _NOW, fallback_sessions=8,
        population_capped=True,
    )

    assert "deadweight" not in captured["findings"]

    _compute_avoidable_total(
        object(), _NOW - timedelta(days=30), _NOW, fallback_sessions=8,
        population_capped=False,
    )

    assert "deadweight" in captured["findings"]


def test_quickstart_reads_no_config_and_opens_no_ondisk_db(tmp_path, monkeypatch):
    """The zero-setup promise, now that the run also builds an optimize report:
    the analyzers get an in-memory `TjConfig()` default, so no config file is
    read or written and no on-disk DB is opened."""
    import tokenjam.cli.main as main
    import tokenjam.core.config as cfg

    def _boom(*args, **kwargs):
        raise AssertionError("quickstart must not touch config / the on-disk DB")

    monkeypatch.setattr(cfg, "load_config", _boom)
    monkeypatch.setattr(cfg, "find_config_file", _boom)
    monkeypatch.setattr(cfg, "write_config", _boom, raising=False)
    monkeypatch.setattr(main, "open_db", _boom, raising=False)

    root = _fixture_root(tmp_path)
    result = _invoke_quickstart(["--root", str(root), "--since", "90d"])

    assert result.exit_code == 0, result.output
