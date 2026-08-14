"""Unit tests for the backfill parser + ingest path."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from click.testing import CliRunner

from tokenjam.cli import cmd_backfill as cmd_backfill_module
from tokenjam.core.backfill import (
    BackfillResult,
    count_claude_code_sessions_in_scope,
    ingest_claude_code,
    iter_claude_code_sessions,
    parse_claude_code_session,
)
from tokenjam.core.config import CaptureConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.otel.semconv import GenAIAttributes, TjAttributes

from tests.factories import make_llm_span, make_tool_span


def _make_session_file(tmp_path: Path, session_id: str, cwd: str,
                        records: list[dict]) -> Path:
    project_dir = tmp_path / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def _assistant_record(uuid: str, model: str, input_tokens: int, output_tokens: int,
                       timestamp: str, session_id: str, cwd: str,
                       tool_uses: list[tuple[str, str]] | None = None,
                       cache_read: int = 0, cache_creation: int = 0,
                       message_id: str | None = None,
                       is_sidechain: bool = False,
                       agent_id: str | None = None) -> dict:
    content: list[dict] = [{"type": "text", "text": "ok"}]
    if tool_uses:
        for tu_id, tu_name in tool_uses:
            content.append({"type": "tool_use", "id": tu_id, "name": tu_name})
    message: dict = {
        "model": model,
        "content": content,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
    }
    # The Anthropic API response id — stable per real call, regenerated `uuid`
    # notwithstanding (#294). Optional so existing tests use the uuid fallback.
    if message_id is not None:
        message["id"] = message_id
    record = {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": timestamp,
        "sessionId": session_id,
        "cwd": cwd,
        "message": message,
    }
    # Claude Code marks subagent (Task-tool) turns with these top-level fields;
    # records in a session's subagents/agent-<id>.jsonl carry isSidechain=true
    # plus the subagent's own agentId.
    if is_sidechain:
        record["isSidechain"] = True
    if agent_id is not None:
        record["agentId"] = agent_id
    return record


def test_parse_extracts_assistant_turns_and_tool_uses(tmp_path):
    path = _make_session_file(
        tmp_path,
        session_id="sess-1",
        cwd="/Users/me/proj",
        records=[
            {"type": "user", "message": {"role": "user", "content": "hi"}},
            _assistant_record(
                "msg-1", "claude-opus-4-7", 1000, 200,
                "2026-04-01T10:00:00.000Z", "sess-1", "/Users/me/proj",
                tool_uses=[("tu-1", "Read"), ("tu-2", "Edit")],
            ),
            _assistant_record(
                "msg-2", "claude-opus-4-7", 500, 100,
                "2026-04-01T10:00:05.000Z", "sess-1", "/Users/me/proj",
            ),
        ],
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert parsed.session_id == "sess-1"
    assert parsed.agent_id == "claude-code-proj"
    # 2 LLM spans + 2 tool spans
    assert len(parsed.spans) == 4
    assert parsed.tool_call_count == 2
    assert parsed.total_input_tokens == 1500
    assert parsed.total_output_tokens == 300
    # Cost is recomputed from pricing — must be > 0 for Opus
    assert parsed.total_cost_usd > 0


def test_parse_returns_none_for_file_with_no_assistant_turns(tmp_path):
    path = _make_session_file(
        tmp_path,
        session_id="sess-empty",
        cwd="/Users/me/proj",
        records=[
            {"type": "user", "message": {"role": "user", "content": "hi"}},
        ],
    )
    assert parse_claude_code_session(path) is None


# -- tokenjam's own internal model calls must never be ingested as a session -

def test_parse_excludes_tokenjam_internal_invoke_cwd(tmp_path):
    """core.distill/core.rulewrite.presence shell out to the SAME `claude`
    CLI, from a private marker cwd under the system temp root
    (`core.distill.INVOKE_CWD_DIRNAME`). Their transcripts must never be
    ingested as a user session — they are tokenjam naming its own findings,
    not agent work anyone did."""
    from tokenjam.core.distill import INVOKE_CWD_DIRNAME

    marker_cwd = f"/private/var/folders/xx/yyyy/T/{INVOKE_CWD_DIRNAME}"
    path = _make_session_file(
        tmp_path,
        session_id="sess-internal",
        cwd=marker_cwd,
        records=[
            _assistant_record(
                "msg-1", "claude-haiku-4-5", 100, 20,
                "2026-04-01T10:00:00.000Z", "sess-internal", marker_cwd,
            ),
        ],
    )
    assert parse_claude_code_session(path) is None


def test_parse_keeps_a_real_session_genuinely_working_out_of_tmp(tmp_path):
    """A user's own project can legitimately live under /tmp — only
    tokenjam's OWN marker subdirectory is excluded, never the bare temp root
    or an unrelated subdirectory of it."""
    real_tmp_cwd = "/tmp/my-scratch-project"
    path = _make_session_file(
        tmp_path,
        session_id="sess-real-tmp",
        cwd=real_tmp_cwd,
        records=[
            _assistant_record(
                "msg-1", "claude-opus-4-7", 1000, 200,
                "2026-04-01T10:00:00.000Z", "sess-real-tmp", real_tmp_cwd,
            ),
        ],
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert parsed.session_id == "sess-real-tmp"


def test_iter_walks_root(tmp_path):
    _make_session_file(
        tmp_path,
        session_id="sess-a",
        cwd="/Users/me/proj-a",
        records=[_assistant_record(
            "msg-a", "claude-sonnet-4-6", 1000, 100,
            "2026-04-01T10:00:00.000Z", "sess-a", "/Users/me/proj-a",
        )],
    )
    _make_session_file(
        tmp_path,
        session_id="sess-b",
        cwd="/Users/me/proj-b",
        records=[_assistant_record(
            "msg-b", "claude-sonnet-4-6", 1000, 100,
            "2026-04-02T10:00:00.000Z", "sess-b", "/Users/me/proj-b",
        )],
    )
    sessions = list(iter_claude_code_sessions(root=tmp_path))
    assert {s.session_id for s in sessions} == {"sess-a", "sess-b"}


def test_ingest_is_idempotent(tmp_path):
    _make_session_file(
        tmp_path,
        session_id="sess-i",
        cwd="/Users/me/proj",
        records=[_assistant_record(
            "msg-i", "claude-haiku-4-5", 1000, 100,
            "2026-04-01T10:00:00.000Z", "sess-i", "/Users/me/proj",
            tool_uses=[("tu-i", "Read")],
        )],
    )
    db = InMemoryBackend()
    try:
        r1 = ingest_claude_code(db, root=tmp_path)
        assert r1.spans_ingested == 2  # 1 LLM + 1 tool
        # Re-run: no new spans
        r2 = ingest_claude_code(db, root=tmp_path)
        assert r2.spans_ingested == 0
        assert r2.spans_skipped_existing == 2
    finally:
        db.close()


def test_ingest_writes_session_record(tmp_path):
    _make_session_file(
        tmp_path,
        session_id="sess-w",
        cwd="/Users/me/proj",
        records=[_assistant_record(
            "msg-w", "claude-haiku-4-5", 800, 150,
            "2026-04-01T10:00:00.000Z", "sess-w", "/Users/me/proj",
        )],
    )
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        sess = db.get_session("sess-w")
        assert sess is not None
        assert sess.agent_id == "claude-code-proj"
        assert sess.input_tokens == 800
    finally:
        db.close()


# --- #176: backfill propagates the config plan tier to SessionRecord -------- #

def _plan_session_file(tmp_path, sid: str):
    _make_session_file(
        tmp_path, session_id=sid, cwd="/Users/me/proj",
        records=[_assistant_record(
            f"msg-{sid}", "claude-haiku-4-5", 500, 100,
            "2026-04-01T10:00:00.000Z", sid, "/Users/me/proj",
        )],
    )


def test_backfill_propagates_config_plan_tier(tmp_path):
    # Acceptance #1/#2: config declares max_5x -> sessions get plan_tier=max_5x,
    # not the "unknown" default (the live ingest path already does this).
    from tokenjam.core.config import ProviderBudget, TjConfig

    _plan_session_file(tmp_path, "sess-plan")
    cfg = TjConfig(version="1")
    cfg.budgets["anthropic"] = ProviderBudget(plan="max_5x")
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path, config=cfg)
        assert db.get_session("sess-plan").plan_tier == "max_5x"
    finally:
        db.close()


def test_backfill_plan_tier_unknown_without_config(tmp_path):
    # Acceptance #3: no config -> "unknown" fallback preserved (defensive).
    _plan_session_file(tmp_path, "sess-noconfig")
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)  # config=None
        assert db.get_session("sess-noconfig").plan_tier == "unknown"
    finally:
        db.close()


def test_backfill_plan_tier_unknown_when_config_has_no_plan(tmp_path):
    # Config present but no plan set under [budget.anthropic] -> still "unknown".
    from tokenjam.core.config import ProviderBudget, TjConfig

    _plan_session_file(tmp_path, "sess-noplan")
    cfg = TjConfig(version="1")
    cfg.budgets["anthropic"] = ProviderBudget()  # no plan
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path, config=cfg)
        assert db.get_session("sess-noplan").plan_tier == "unknown"
    finally:
        db.close()


# --- #243: backfilled spans group into one session-level trace ------------- #

def test_backfill_groups_session_into_one_trace(tmp_path):
    # A conversation with two assistant turns; the first issues two tool calls.
    # All four spans (2 LLM + 2 tool) should land in ONE trace, with the tool
    # spans as children of their assistant message (not per-message fragments).
    _make_session_file(
        tmp_path, session_id="sess-trace", cwd="/Users/me/proj",
        records=[
            _assistant_record(
                "msg-1", "claude-opus-4-7", 1000, 200,
                "2026-04-01T10:00:00.000Z", "sess-trace", "/Users/me/proj",
                tool_uses=[("tu-1", "Bash"), ("tu-2", "Read")],
            ),
            _assistant_record(
                "msg-2", "claude-opus-4-7", 500, 100,
                "2026-04-01T10:00:05.000Z", "sess-trace", "/Users/me/proj",
            ),
        ],
    )
    db = InMemoryBackend()
    try:
        from tokenjam.core.models import TraceFilters

        ingest_claude_code(db, root=tmp_path)

        # Exactly one trace for the whole session.
        trace_ids = [
            r[0] for r in db.conn.execute(
                "SELECT DISTINCT trace_id FROM spans"
            ).fetchall()
        ]
        assert len(trace_ids) == 1

        traces = db.get_traces(TraceFilters())
        assert len(traces) == 1
        assert traces[0].span_count == 4  # 2 LLM + 2 tool

        # The trace holds both LLM calls and both tool calls, and every tool
        # span is parented to an LLM span in the same trace.
        spans = db.get_trace_spans(trace_ids[0])
        llm = [s for s in spans if s.name == "gen_ai.llm.call"]
        tools = [s for s in spans if s.name == "gen_ai.tool.call"]
        assert len(llm) == 2
        assert len(tools) == 2
        llm_ids = {s.span_id for s in llm}
        assert all(t.parent_span_id in llm_ids for t in tools)
        assert {t.tool_name for t in tools} == {"Bash", "Read"}
    finally:
        db.close()


def test_backfill_separate_sessions_get_separate_traces(tmp_path):
    # Two distinct sessions must NOT collapse into one trace.
    for sid in ("sess-x", "sess-y"):
        _make_session_file(
            tmp_path, session_id=sid, cwd="/Users/me/proj",
            records=[_assistant_record(
                f"m-{sid}", "claude-haiku-4-5", 100, 50,
                "2026-04-01T10:00:00.000Z", sid, "/Users/me/proj",
            )],
        )
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        n_traces = db.conn.execute(
            "SELECT COUNT(DISTINCT trace_id) FROM spans"
        ).fetchone()[0]
        assert n_traces == 2
    finally:
        db.close()


# --- #245: backfill persists the cache read/write split -------------------- #

def test_backfill_persists_cache_read_write_split(tmp_path):
    # An assistant turn that both reads cached prefix and creates new cache.
    _make_session_file(
        tmp_path, session_id="sess-cache", cwd="/Users/me/proj",
        records=[_assistant_record(
            "msg-cache", "claude-haiku-4-5", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-cache", "/Users/me/proj",
            cache_read=4321, cache_creation=8765,
        )],
    )
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        row = db.conn.execute(
            "SELECT cache_tokens, cache_write_tokens FROM spans "
            "WHERE name = 'gen_ai.llm.call'"
        ).fetchone()
        # Read in cache_tokens, creation in cache_write_tokens — NOT collapsed
        # into one field (the #245 bug summed them and left write = 0).
        assert row == (4321, 8765)
    finally:
        db.close()


def test_backfill_session_cache_tokens_is_read_only(tmp_path):
    # SessionRecord.cache_tokens tracks cache-READ only (it has no write field),
    # matching the live ingest path.
    _make_session_file(
        tmp_path, session_id="sess-cache2", cwd="/Users/me/proj",
        records=[_assistant_record(
            "msg-cache2", "claude-haiku-4-5", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-cache2", "/Users/me/proj",
            cache_read=300, cache_creation=700,
        )],
    )
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        assert db.get_session("sess-cache2").cache_tokens == 300
    finally:
        db.close()


# --- #238: new / existing / total count reporting -------------------------- #

def test_backfill_counts_match_sessions_table(tmp_path):
    # Two distinct sessions -> two rows in the sessions table.
    _make_session_file(
        tmp_path, session_id="sess-1", cwd="/Users/me/proj-a",
        records=[_assistant_record(
            "m1", "claude-haiku-4-5", 100, 50,
            "2026-04-01T10:00:00.000Z", "sess-1", "/Users/me/proj-a",
        )],
    )
    _make_session_file(
        tmp_path, session_id="sess-2", cwd="/Users/me/proj-b",
        records=[_assistant_record(
            "m2", "claude-haiku-4-5", 100, 50,
            "2026-04-02T10:00:00.000Z", "sess-2", "/Users/me/proj-b",
        )],
    )
    db = InMemoryBackend()
    try:
        r1 = ingest_claude_code(db, root=tmp_path)
        table_count = db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        # First run: every session is new, total matches the table.
        assert r1.sessions_total == table_count == 2
        assert r1.sessions_new == 2
        assert r1.sessions_existing == 0

        # Idempotent re-run: nothing new, but total still reports the full state
        # (not new-only, which read as "barely worked" — #238).
        r2 = ingest_claude_code(db, root=tmp_path)
        assert r2.sessions_total == 2
        assert r2.sessions_new == 0
        assert r2.sessions_existing == 2
        assert db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
    finally:
        db.close()


def test_backfill_multiple_files_one_session_does_not_inflate_count(tmp_path):
    # Two conversation files sharing one sessionId collapse to ONE session row
    # (Claude Code writes continuations/sidechains). conversations_seen counts
    # files; sessions_total matches the table (#238).
    _make_session_file(
        tmp_path, session_id="file-a", cwd="/Users/me/proj",
        records=[_assistant_record(
            "m-a", "claude-haiku-4-5", 100, 50,
            "2026-04-01T10:00:00.000Z", "sess-shared", "/Users/me/proj",
        )],
    )
    _make_session_file(
        tmp_path, session_id="file-b", cwd="/Users/me/proj",
        records=[_assistant_record(
            "m-b", "claude-haiku-4-5", 100, 50,
            "2026-04-01T10:05:00.000Z", "sess-shared", "/Users/me/proj",
        )],
    )
    db = InMemoryBackend()
    try:
        r = ingest_claude_code(db, root=tmp_path)
        assert r.conversations_seen == 2          # two files parsed
        assert r.sessions_total == 1              # one distinct session
        table_count = db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert table_count == 1
    finally:
        db.close()


def test_backfill_total_cost_is_window_total_on_rerun(tmp_path):
    # Cost reflects the full in-window spend on every run, not just newly
    # inserted spans (which would show $0 on an idempotent re-run) — #238.
    _make_session_file(
        tmp_path, session_id="sess-cost", cwd="/Users/me/proj",
        records=[_assistant_record(
            "m-cost", "claude-opus-4-7", 5000, 1000,
            "2026-04-01T10:00:00.000Z", "sess-cost", "/Users/me/proj",
        )],
    )
    db = InMemoryBackend()
    try:
        r1 = ingest_claude_code(db, root=tmp_path)
        r2 = ingest_claude_code(db, root=tmp_path)
        assert r1.total_cost_usd > 0
        assert r2.total_cost_usd == r1.total_cost_usd  # not zeroed on re-run
    finally:
        db.close()


def test_iter_skips_files_before_since(tmp_path):
    p = _make_session_file(
        tmp_path,
        session_id="sess-old",
        cwd="/Users/me/proj",
        records=[_assistant_record(
            "msg-old", "claude-haiku-4-5", 100, 50,
            "2026-04-01T10:00:00.000Z", "sess-old", "/Users/me/proj",
        )],
    )
    # Force mtime far in the past
    import os
    old = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(p, (old, old))
    cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
    sessions = list(iter_claude_code_sessions(root=tmp_path, since=cutoff))
    assert sessions == []


def test_claude_code_backfill_accepts_since_window(tmp_path, monkeypatch):
    captured: dict[str, datetime | None] = {}
    fixed_now = datetime(2026, 6, 24, 12, 0, tzinfo=timezone.utc)

    def fake_ingest(db, *, root, since, progress, config, reingest=False):
        captured["since"] = since
        return BackfillResult()

    monkeypatch.setattr("tokenjam.utils.time_parse.utcnow", lambda: fixed_now)
    monkeypatch.setattr(cmd_backfill_module, "ingest_claude_code", fake_ingest)

    result = CliRunner().invoke(
        cmd_backfill_module.claude_code,
        ["--root", str(tmp_path), "--since", "30d", "--quiet"],
        obj={"db": object(), "config": None},
    )

    assert result.exit_code == 0, result.output
    assert captured["since"] == datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)


def test_claude_code_backfill_keeps_since_days_alias(tmp_path, monkeypatch):
    captured: dict[str, datetime | None] = {}
    fixed_now = datetime(2026, 6, 24, 12, 0, tzinfo=timezone.utc)

    def fake_ingest(db, *, root, since, progress, config, reingest=False):
        captured["since"] = since
        return BackfillResult()

    monkeypatch.setattr(cmd_backfill_module, "utcnow", lambda: fixed_now)
    monkeypatch.setattr(cmd_backfill_module, "ingest_claude_code", fake_ingest)

    result = CliRunner().invoke(
        cmd_backfill_module.claude_code,
        ["--root", str(tmp_path), "--since-days", "7", "--quiet"],
        obj={"db": object(), "config": None},
    )

    assert result.exit_code == 0, result.output
    assert captured["since"] == datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc)


def test_claude_code_backfill_rejects_two_since_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(cmd_backfill_module, "ingest_claude_code", lambda **kwargs: BackfillResult())

    result = CliRunner().invoke(
        cmd_backfill_module.claude_code,
        ["--root", str(tmp_path), "--since", "30d", "--since-days", "7", "--quiet"],
        obj={"db": object(), "config": None},
    )

    assert result.exit_code != 0
    assert "Use either --since or --since-days" in result.output


def test_claude_code_backfill_prints_unknown_model_warning_after_summary(tmp_path):
    import tokenjam.core.cost as cost_mod

    _make_session_file(
        tmp_path,
        session_id="sess-unknown",
        cwd="/Users/me/proj",
        records=[
            _assistant_record(
                "u1", "totally-unknown-model-xyz", 1000, 200,
                "2026-06-20T10:00:00.000Z", "sess-unknown", "/Users/me/proj",
            ),
        ],
    )
    cost_mod._UNKNOWN_MODEL_WARNED.clear()
    db = InMemoryBackend()

    result = CliRunner().invoke(
        cmd_backfill_module.claude_code,
        ["--root", str(tmp_path), "--quiet"],
        obj={"db": db, "config": None},
    )

    assert result.exit_code == 0, result.output
    assert "Backfilled" in result.output
    assert "No pricing data for anthropic/totally-unknown-model-xyz" in result.output
    assert result.output.index("Backfilled") < result.output.index(
        "No pricing data for anthropic/totally-unknown-model-xyz"
    )


# --- #443: cheap in-scope session count (progress bar total + heads-up) -----


def test_count_sessions_in_scope_no_root_returns_zero(tmp_path):
    assert count_claude_code_sessions_in_scope(root=tmp_path / "no-such-dir") == 0


def test_count_sessions_in_scope_counts_all_files_with_no_since(tmp_path):
    for i in range(5):
        _make_session_file(tmp_path, f"sess-{i}", "/Users/me/proj", [
            _assistant_record(f"u{i}", "claude-sonnet-4-5-20250929", 10, 5,
                              "2026-06-20T10:00:00.000Z", f"sess-{i}", "/Users/me/proj"),
        ])
    assert count_claude_code_sessions_in_scope(root=tmp_path) == 5


def test_count_sessions_in_scope_honors_since_mtime_filter(tmp_path):
    old = _make_session_file(tmp_path, "sess-old", "/Users/me/proj", [
        _assistant_record("u1", "claude-sonnet-4-5-20250929", 10, 5,
                          "2020-01-01T10:00:00.000Z", "sess-old", "/Users/me/proj"),
    ])
    _make_session_file(tmp_path, "sess-new", "/Users/me/proj", [
        _assistant_record("u2", "claude-sonnet-4-5-20250929", 10, 5,
                          "2026-06-20T10:00:00.000Z", "sess-new", "/Users/me/proj"),
    ])
    import os
    import time
    old_time = time.time() - 3600 * 24 * 400  # 400 days ago
    os.utime(old, (old_time, old_time))

    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert count_claude_code_sessions_in_scope(root=tmp_path, since=cutoff) == 1


def test_count_sessions_in_scope_caps_at_max_sessions(tmp_path):
    for i in range(10):
        _make_session_file(tmp_path, f"sess-{i}", "/Users/me/proj", [
            _assistant_record(f"u{i}", "claude-sonnet-4-5-20250929", 10, 5,
                              "2026-06-20T10:00:00.000Z", f"sess-{i}", "/Users/me/proj"),
        ])
    assert count_claude_code_sessions_in_scope(root=tmp_path, max_sessions=4) == 4
    assert count_claude_code_sessions_in_scope(root=tmp_path, max_sessions=100) == 10


def test_ingest_since_and_max_sessions_together_caps_and_keeps_most_recent(tmp_path):
    """The onboard "fast" path (#443) passes `since` AND `max_sessions`
    together — `since` alone doesn't reliably bound the work on a machine
    where most session files share recent mtimes, so `max_sessions` is the
    real guarantee. Confirms both apply: `limit_reached` fires, and the cap
    keeps the most-recent (by mtime) sessions, not an arbitrary subset."""
    import os

    for i in range(10):
        sid = f"sess-{i:02d}"
        path = _make_session_file(tmp_path, sid, _CWD, [
            _assistant_record(f"u{i}", "claude-sonnet-4-5-20250929", 10, 5,
                              "2026-06-20T10:00:00.000Z", sid, _CWD),
        ])
        # Newer index => newer mtime; all comfortably within `since`.
        os.utime(path, (1_900_000_000 + i, 1_900_000_000 + i))

    db = InMemoryBackend()
    since = datetime(2020, 1, 1, tzinfo=timezone.utc)  # wide enough to include all 10
    result = ingest_claude_code(db, root=tmp_path, since=since, max_sessions=4)

    assert result.limit_reached is True
    assert result.sessions_ingested == 4
    session_ids = {r[0] for r in db.conn.execute("SELECT session_id FROM sessions").fetchall()}
    # The 4 highest-indexed (newest-mtime) sessions were kept, not the oldest.
    assert session_ids == {"sess-06", "sess-07", "sess-08", "sess-09"}


# --- #294: dedup resumed/branched sessions (over-counted tokens) -------------- #

_CWD = "/Users/me/proj"


def test_resumed_session_dedups_same_call_by_message_id(tmp_path):
    """The same logical call replayed under a NEW record uuid (same message.id)
    on resume must collapse to ONE span with single-call totals (#294)."""
    path = _make_session_file(
        tmp_path, session_id="sess-resume", cwd=_CWD,
        records=[
            # Original turn.
            _assistant_record("uuid-A", "claude-opus-4-7", 3289, 692,
                              "2026-04-01T10:00:00.000Z", "sess-resume", _CWD,
                              cache_creation=42981, message_id="msg_stable_1"),
            # A user turn in between (ignored).
            {"type": "user", "message": {"role": "user", "content": "more"}},
            # Resume replays the SAME assistant turn — fresh uuid, SAME message.id.
            _assistant_record("uuid-B", "claude-opus-4-7", 3289, 692,
                              "2026-04-01T10:05:00.000Z", "sess-resume", _CWD,
                              cache_creation=42981, message_id="msg_stable_1"),
            # …and a third replay (the 3–4× repeat seen in real data).
            _assistant_record("uuid-C", "claude-opus-4-7", 3289, 692,
                              "2026-04-01T10:05:01.000Z", "sess-resume", _CWD,
                              cache_creation=42981, message_id="msg_stable_1"),
        ],
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    llm_spans = [s for s in parsed.spans if s.name == "gen_ai.llm.call"]
    assert len(llm_spans) == 1, "the same message.id must collapse to one span"
    # Totals reflect a SINGLE call, not 3×.
    assert parsed.total_input_tokens == 3289
    assert parsed.total_output_tokens == 692
    assert llm_spans[0].cache_write_tokens == 42981


def test_resume_last_wins_keeps_finalized_usage(tmp_path):
    """Early replay snapshots carry partial output_tokens; the LAST record has the
    complete generation. Dedup keeps the finalized usage (last-wins, #294)."""
    path = _make_session_file(
        tmp_path, session_id="sess-snap", cwd=_CWD,
        records=[
            # Partial snapshot: tiny output.
            _assistant_record("uuid-1", "claude-opus-4-7", 2, 1,
                              "2026-04-01T10:00:00.000Z", "sess-snap", _CWD,
                              cache_read=15764, cache_creation=4317,
                              message_id="msg_snap"),
            # Finalized: full output.
            _assistant_record("uuid-2", "claude-opus-4-7", 2, 575,
                              "2026-04-01T10:00:02.000Z", "sess-snap", _CWD,
                              cache_read=15764, cache_creation=4317,
                              message_id="msg_snap"),
        ],
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    llm_spans = [s for s in parsed.spans if s.name == "gen_ai.llm.call"]
    assert len(llm_spans) == 1
    # The complete output (575), not the partial snapshot (1) nor their sum (576).
    assert parsed.total_output_tokens == 575
    assert llm_spans[0].output_tokens == 575


def test_distinct_calls_with_identical_usage_not_deduped(tmp_path):
    """Two REAL calls can legitimately share identical token counts. Dedup keys on
    the stable message.id, never on a usage signature, so both survive (#294)."""
    path = _make_session_file(
        tmp_path, session_id="sess-twins", cwd=_CWD,
        records=[
            _assistant_record("uuid-x", "claude-opus-4-7", 2, 691,
                              "2026-04-01T10:00:00.000Z", "sess-twins", _CWD,
                              message_id="msg_call_A"),
            _assistant_record("uuid-y", "claude-opus-4-7", 2, 691,
                              "2026-04-01T10:00:03.000Z", "sess-twins", _CWD,
                              message_id="msg_call_B"),
        ],
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    llm_spans = [s for s in parsed.spans if s.name == "gen_ai.llm.call"]
    assert len(llm_spans) == 2, "distinct message.ids are distinct calls"
    assert parsed.total_output_tokens == 1382  # 691 + 691, not deduped


def test_tool_use_dedups_on_resume(tmp_path):
    """A tool_use replayed on resume (stable tool_use id) collapses to one span."""
    path = _make_session_file(
        tmp_path, session_id="sess-tool", cwd=_CWD,
        records=[
            _assistant_record("uuid-a", "claude-opus-4-7", 10, 5,
                              "2026-04-01T10:00:00.000Z", "sess-tool", _CWD,
                              tool_uses=[("toolu_stable", "Read")],
                              message_id="msg_tool"),
            _assistant_record("uuid-b", "claude-opus-4-7", 10, 5,
                              "2026-04-01T10:05:00.000Z", "sess-tool", _CWD,
                              tool_uses=[("toolu_stable", "Read")],
                              message_id="msg_tool"),
        ],
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    tool_spans = [s for s in parsed.spans if s.name == "gen_ai.tool.call"]
    assert len(tool_spans) == 1
    assert parsed.tool_call_count == 1


def test_falls_back_to_uuid_when_message_id_absent(tmp_path):
    """Without message.id (older logs), distinct uuids stay distinct calls."""
    path = _make_session_file(
        tmp_path, session_id="sess-noid", cwd=_CWD,
        records=[
            _assistant_record("uuid-p", "claude-opus-4-7", 100, 20,
                              "2026-04-01T10:00:00.000Z", "sess-noid", _CWD),
            _assistant_record("uuid-q", "claude-opus-4-7", 100, 20,
                              "2026-04-01T10:00:03.000Z", "sess-noid", _CWD),
        ],
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert len([s for s in parsed.spans if s.name == "gen_ai.llm.call"]) == 2


def test_ingest_resumed_session_writes_one_span_per_call(tmp_path):
    """End-to-end through ingest: a resumed session lands deduped in the DB with
    single-call session totals (#294)."""
    _make_session_file(
        tmp_path, session_id="sess-e2e", cwd=_CWD,
        records=[
            _assistant_record("u1", "claude-opus-4-7", 1000, 200,
                              "2026-04-01T10:00:00.000Z", "sess-e2e", _CWD,
                              message_id="msg_e2e_1"),
            _assistant_record("u2", "claude-opus-4-7", 1000, 200,
                              "2026-04-01T10:05:00.000Z", "sess-e2e", _CWD,
                              message_id="msg_e2e_1"),  # resume replay
            _assistant_record("u3", "claude-opus-4-7", 500, 80,
                              "2026-04-01T10:06:00.000Z", "sess-e2e", _CWD,
                              message_id="msg_e2e_2"),  # a second real call
        ],
    )
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        rows = db.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(output_tokens),0) FROM spans "
            "WHERE name = 'gen_ai.llm.call'"
        ).fetchone()
        assert rows[0] == 2, "two distinct calls, not three records"
        assert rows[1] == 280, "200 + 80, not 200 + 200 + 80"
        sess = db.get_session("sess-e2e")
        assert sess is not None
        assert sess.output_tokens == 280
    finally:
        db.close()


def test_parse_tags_subagent_spans_with_sub_agent_id(tmp_path):
    """Spans from a sidechain (Task-tool) turn carry the subagent's agentId;
    main-thread spans carry None. This is what lets a session's cost be broken
    down per subagent."""
    path = _make_session_file(
        tmp_path,
        session_id="sess-sa",
        cwd="/Users/me/proj",
        records=[
            _assistant_record(
                "m-main", "claude-opus-4-7", 1000, 200,
                "2026-04-01T10:00:00.000Z", "sess-sa", "/Users/me/proj",
            ),
            _assistant_record(
                "m-sub", "claude-haiku-4-5", 5000, 500,
                "2026-04-01T10:00:01.000Z", "sess-sa", "/Users/me/proj",
                tool_uses=[("tu-s", "Read")], is_sidechain=True, agent_id="ag-1",
            ),
        ],
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    # Main-thread LLM span -> no subagent id
    main_llm = [s for s in parsed.spans
                if s.name == "gen_ai.llm.call" and s.sub_agent_id is None]
    assert len(main_llm) == 1
    # Subagent LLM span + its tool span -> tagged with the subagent's agentId
    sub_spans = [s for s in parsed.spans if s.sub_agent_id == "ag-1"]
    assert len(sub_spans) == 2
    assert {s.name for s in sub_spans} == {"gen_ai.llm.call", "gen_ai.tool.call"}


def test_ingest_attributes_tokens_per_subagent(tmp_path):
    """End-to-end: a session whose subagent lives in subagents/agent-*.jsonl
    gets its tokens folded under the parent session_id AND remains attributable
    per subagent via sub_agent_id."""
    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path,
        session_id="sess-x",
        cwd=proj,
        records=[_assistant_record(
            "m-main", "claude-opus-4-7", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-x", proj,
        )],
    )
    # Subagent transcript: <project>/<sid>/subagents/agent-<id>.jsonl
    sub_dir = tmp_path / proj.replace("/", "-") / "sess-x" / "subagents"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "agent-ag1.jsonl").write_text(json.dumps(_assistant_record(
        "m-sub", "claude-haiku-4-5", 5000, 500,
        "2026-04-01T10:00:01.000Z", "sess-x", proj,
        is_sidechain=True, agent_id="ag1",
    )))

    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        rows = db.conn.execute(
            "SELECT sub_agent_id, SUM(input_tokens) FROM spans "
            "WHERE session_id = $1 AND name = $2 GROUP BY sub_agent_id",
            ["sess-x", "gen_ai.llm.call"],
        ).fetchall()
        per_subagent = {r[0]: r[1] for r in rows}
        assert per_subagent.get(None) == 1000     # main thread
        assert per_subagent.get("ag1") == 5000     # subagent, attributable
        # Span-derived session cost includes the subagent's spend (fold-in).
        assert db.get_session_cost("sess-x") > 0
    finally:
        db.close()


def _content_session_file(tmp_path: Path) -> Path:
    """A session with a human prompt, an assistant narration + a tool_use with
    real input args — exactly what the context-cost diagnostic (#4) needs."""
    return _make_session_file(
        tmp_path,
        session_id="sess-cap",
        cwd="/Users/me/proj",
        records=[
            {"type": "user", "message": {"role": "user",
                                         "content": "please read the config"}},
            {
                "type": "assistant",
                "uuid": "msg-cap",
                "timestamp": "2026-04-01T10:00:00.000Z",
                "sessionId": "sess-cap",
                "cwd": "/Users/me/proj",
                "message": {
                    "model": "claude-opus-4-7",
                    "content": [
                        {"type": "text", "text": "Reading the config file now."},
                        {"type": "tool_use", "id": "tu-cap", "name": "Read",
                         "input": {"file_path": "/etc/app/config.toml"}},
                    ],
                    "usage": {
                        "input_tokens": 1000, "output_tokens": 200,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            },
        ],
    )


def _llm_and_tool(parsed):
    llm = next(s for s in parsed.spans if s.name == "gen_ai.llm.call")
    tool = next(s for s in parsed.spans if s.name == "gen_ai.tool.call")
    return llm, tool


def test_capture_off_extracts_no_content_only_provenance(tmp_path):
    """Every toggle explicitly off extracts NO content (#3 default-off).

    Asserted as "no content key is present" rather than as an exact attribute
    dict: provenance — the ingest source, and the call id that lets a second
    observer of the same call be recognised instead of counted twice — is not
    content and is stamped regardless of the capture toggles.
    """
    path = _content_session_file(tmp_path)

    parsed = parse_claude_code_session(
        path, capture=CaptureConfig(prompts=False, tool_inputs=False),
    )
    assert parsed is not None
    llm, tool = _llm_and_tool(parsed)
    content_keys = {
        GenAIAttributes.PROMPT_CONTENT, GenAIAttributes.COMPLETION_CONTENT,
        GenAIAttributes.TOOL_INPUT, GenAIAttributes.TOOL_OUTPUT,
        TjAttributes.SYSTEM_PREFIX_CONTENT,
        # The compact prefix keys ride the same capture toggle: SAMPLE is
        # literal prompt text, and HASH/LENGTH are a fingerprint of a file the
        # user just said not to capture.
        TjAttributes.SYSTEM_PREFIX_HASH,
        TjAttributes.SYSTEM_PREFIX_SAMPLE,
        TjAttributes.SYSTEM_PREFIX_LENGTH,
    }
    for span in (llm, tool):
        assert span.attributes["source"] == "backfill.claude_code"
        assert content_keys.isdisjoint(span.attributes)
    assert llm.attributes[TjAttributes.CALL_ID] == "msg-cap"


def test_capture_default_extracts_prompt_and_tool_input(tmp_path):
    """`prompts` and `tool_inputs` both default on (E33 / this fix) so
    `capture=None` and a bare `CaptureConfig()` both extract prompt content
    (needed for `trim` / `cache-recommend` / `reuse`) and tool_input (needed
    for `script` / `verbosity`'s argument-shape clustering) out of the box,
    while completions/tool_outputs stay off."""
    path = _content_session_file(tmp_path)

    for capture in (None, CaptureConfig()):
        parsed = parse_claude_code_session(path, capture=capture)
        assert parsed is not None
        llm, tool = _llm_and_tool(parsed)
        assert llm.attributes[GenAIAttributes.PROMPT_CONTENT] == \
            "please read the config"
        assert GenAIAttributes.COMPLETION_CONTENT not in llm.attributes
        assert tool.attributes[GenAIAttributes.TOOL_INPUT] == \
            {"file_path": "/etc/app/config.toml"}


def test_capture_on_populates_prompt_completion_and_tool_input(tmp_path):
    """With every toggle on, a backfilled span carries the human prompt, the
    agent narration, and the raw tool_input — the data #4 needs for per-message
    / per-inclusion token attribution."""
    path = _content_session_file(tmp_path)
    parsed = parse_claude_code_session(
        path,
        capture=CaptureConfig(
            prompts=True, completions=True, tool_inputs=True, tool_outputs=True,
        ),
    )
    assert parsed is not None
    llm, tool = _llm_and_tool(parsed)
    assert llm.attributes[GenAIAttributes.PROMPT_CONTENT] == "please read the config"
    assert llm.attributes[GenAIAttributes.COMPLETION_CONTENT] == \
        "Reading the config file now."
    assert tool.attributes[GenAIAttributes.TOOL_INPUT] == \
        {"file_path": "/etc/app/config.toml"}


def test_capture_flags_are_independent(tmp_path):
    """Each toggle gates only its own field — flipping one never leaks another.

    `prompts` and `tool_inputs` are both explicitly set in the cases below:
    they default on (E33 / this fix), so leaving either unset would leak
    PROMPT_CONTENT / TOOL_INPUT into a test about the other flags gating
    independently of them.
    """
    path = _content_session_file(tmp_path)

    parsed = parse_claude_code_session(
        path, capture=CaptureConfig(prompts=False, tool_inputs=True),
    )
    llm, tool = _llm_and_tool(parsed)
    assert GenAIAttributes.TOOL_INPUT in tool.attributes
    assert GenAIAttributes.PROMPT_CONTENT not in llm.attributes
    assert GenAIAttributes.COMPLETION_CONTENT not in llm.attributes

    parsed = parse_claude_code_session(
        path, capture=CaptureConfig(prompts=False, completions=True, tool_inputs=False),
    )
    llm, tool = _llm_and_tool(parsed)
    assert GenAIAttributes.COMPLETION_CONTENT in llm.attributes
    assert GenAIAttributes.PROMPT_CONTENT not in llm.attributes
    assert GenAIAttributes.TOOL_INPUT not in tool.attributes


def test_capture_prompts_reads_project_claude_md_as_system_prefix(tmp_path):
    """#272: the human's per-turn message never repeats verbatim, so
    cache-recommend's prefix-hash needs a different, genuinely stable
    signal. The project's CLAUDE.md is read straight off disk (it's not in
    the transcript) and stamped as `TjAttributes.SYSTEM_PREFIX_HASH` --
    identical on every assistant span for the same project, unlike
    PROMPT_CONTENT.

    The prefix is stored as a fingerprint rather than as the text: it was the
    same value on every span, so keeping it whole cost (size x span count).
    What this asserts is unchanged -- that a stable per-project signal exists
    and that it is NOT the per-turn prompt."""
    from tokenjam.core.system_prefix import prefix_hash
    from tokenjam.otel.semconv import TjAttributes

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "CLAUDE.md").write_text("# Project rules\nAlways use tabs.")
    cwd = str(project_dir)

    path = _make_session_file(
        tmp_path,
        session_id="sess-claude-md",
        cwd=cwd,
        records=[
            {"type": "user", "message": {"role": "user", "content": "turn one"}},
            _assistant_record(
                "msg-a", "claude-opus-4-7", 1000, 200,
                "2026-04-01T10:00:00.000Z", "sess-claude-md", cwd,
            ),
            {"type": "user", "message": {"role": "user", "content": "turn two, different"}},
            _assistant_record(
                "msg-b", "claude-opus-4-7", 900, 150,
                "2026-04-01T10:05:00.000Z", "sess-claude-md", cwd,
            ),
        ],
    )

    parsed = parse_claude_code_session(path, capture=CaptureConfig(prompts=True))
    assert parsed is not None
    llm_spans = [s for s in parsed.spans if s.name == "gen_ai.llm.call"]
    assert len(llm_spans) == 2
    expected = prefix_hash("# Project rules\nAlways use tabs.")
    for span in llm_spans:
        assert span.attributes[TjAttributes.SYSTEM_PREFIX_HASH] == expected
        # The text itself is never stored -- that is the 4.06 GB defect.
        assert TjAttributes.SYSTEM_PREFIX_CONTENT not in span.attributes
    # The per-turn human prompt still differs call to call -- confirming the
    # two signals are genuinely distinct, not the same field renamed.
    assert llm_spans[0].attributes[GenAIAttributes.PROMPT_CONTENT] == "turn one"
    assert llm_spans[1].attributes[GenAIAttributes.PROMPT_CONTENT] == \
        "turn two, different"

    # prompts=False captures neither.
    parsed_off = parse_claude_code_session(path, capture=CaptureConfig(prompts=False))
    llm_off = next(s for s in parsed_off.spans if s.name == "gen_ai.llm.call")
    assert TjAttributes.SYSTEM_PREFIX_HASH not in llm_off.attributes
    assert TjAttributes.SYSTEM_PREFIX_CONTENT not in llm_off.attributes


def test_claude_md_lookup_retries_after_a_record_with_no_cwd(tmp_path):
    """The lazy load uses `None` = "not tried" / `""` = "tried, found nothing".
    A leading record with no `cwd` can't resolve anything, so it must NOT
    commit the `""` outcome -- doing so locked the sentinel permanently and
    every later record that DID carry a cwd silently lost its system prefix."""
    from tokenjam.core.system_prefix import prefix_hash
    from tokenjam.otel.semconv import TjAttributes

    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    (project_dir / "CLAUDE.md").write_text("# Project rules\nAlways use tabs.")
    cwd = str(project_dir)

    # The first assistant record carries no `cwd` at all; the second does.
    first = _assistant_record(
        "msg-a", "claude-opus-4-7", 1000, 200,
        "2026-04-01T10:00:00.000Z", "sess-late-cwd", cwd,
    )
    first.pop("cwd")
    records = [
        {"type": "user", "message": {"role": "user", "content": "turn one"}},
        first,
        {"type": "user", "message": {"role": "user", "content": "turn two"}},
        _assistant_record(
            "msg-b", "claude-opus-4-7", 900, 150,
            "2026-04-01T10:05:00.000Z", "sess-late-cwd", cwd,
        ),
    ]
    path = _make_session_file(
        tmp_path, session_id="sess-late-cwd", cwd=cwd, records=records,
    )

    parsed = parse_claude_code_session(path, capture=CaptureConfig(prompts=True))
    assert parsed is not None
    llm_spans = [s for s in parsed.spans if s.name == "gen_ai.llm.call"]
    assert len(llm_spans) == 2
    # The retry happened: the later, cwd-bearing record resolved the file.
    assert llm_spans[-1].attributes[TjAttributes.SYSTEM_PREFIX_HASH] == \
        prefix_hash("# Project rules\nAlways use tabs.")


def test_capture_prompts_on_without_claude_md_omits_system_prefix(tmp_path):
    """No CLAUDE.md at the project cwd -> no attribute, never a KeyError or
    an empty-string placeholder."""
    from tokenjam.otel.semconv import TjAttributes

    path = _content_session_file(tmp_path)
    parsed = parse_claude_code_session(path, capture=CaptureConfig(prompts=True))
    llm, _tool = _llm_and_tool(parsed)
    assert TjAttributes.SYSTEM_PREFIX_HASH not in llm.attributes
    assert TjAttributes.SYSTEM_PREFIX_CONTENT not in llm.attributes


def test_ingest_persists_captured_content_when_config_enables_it(tmp_path):
    """End-to-end through ingest: with config.capture enabled, the stored span's
    attributes column carries the content; a capture-all-off config stores
    nothing."""
    from tokenjam.core.config import TjConfig

    _content_session_file(tmp_path)

    # Capture-all-off config -> no content persisted.
    off_cfg = TjConfig(version="1")
    off_cfg.capture = CaptureConfig(prompts=False, tool_inputs=False)
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path, config=off_cfg)
        attrs = db.conn.execute(
            "SELECT attributes FROM spans WHERE name = $1",
            ["gen_ai.llm.call"],
        ).fetchone()[0]
        parsed_attrs = json.loads(attrs) if isinstance(attrs, str) else attrs
        assert GenAIAttributes.PROMPT_CONTENT not in parsed_attrs
        assert GenAIAttributes.COMPLETION_CONTENT not in parsed_attrs
    finally:
        db.close()

    # Capture-enabled config -> content persisted on the backfilled span.
    cfg = TjConfig(version="1")
    cfg.capture = CaptureConfig(prompts=True, completions=True, tool_inputs=True)
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path, config=cfg)
        llm_attrs = db.conn.execute(
            "SELECT attributes FROM spans WHERE name = $1",
            ["gen_ai.llm.call"],
        ).fetchone()[0]
        llm_attrs = json.loads(llm_attrs) if isinstance(llm_attrs, str) else llm_attrs
        assert llm_attrs[GenAIAttributes.COMPLETION_CONTENT] == \
            "Reading the config file now."
        assert llm_attrs[GenAIAttributes.PROMPT_CONTENT] == "please read the config"

        tool_attrs = db.conn.execute(
            "SELECT attributes FROM spans WHERE name = $1",
            ["gen_ai.tool.call"],
        ).fetchone()[0]
        tool_attrs = json.loads(tool_attrs) if isinstance(tool_attrs, str) else tool_attrs
        assert tool_attrs[GenAIAttributes.TOOL_INPUT] == \
            {"file_path": "/etc/app/config.toml"}
    finally:
        db.close()


def test_reingest_retags_existing_spans(tmp_path):
    """--reingest re-populates sub_agent_id on spans an older backfill ingested
    before the column existed. A PLAIN re-run does the same now (the default
    bulk path overlays newly-resolvable identity columns onto existing spans,
    not just newly-inserted ones — see `_dedup_new_spans`'s overlay_candidates
    / `bulk_overlay_span_attrs`), so both paths converge on the same
    result; --reingest is left with nothing to do afterwards."""
    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-rt", cwd=proj,
        records=[_assistant_record(
            "m-main", "claude-opus-4-7", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-rt", proj,
        )],
    )
    sub_dir = tmp_path / proj.replace("/", "-") / "sess-rt" / "subagents"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "agent-rt1.jsonl").write_text(json.dumps(_assistant_record(
        "m-rt1", "claude-haiku-4-5", 5000, 500,
        "2026-04-01T10:00:01.000Z", "sess-rt", proj,
        tool_uses=[("tu-rt", "Read")], is_sidechain=True, agent_id="rt1",
    )))

    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        # Simulate a pre-column backfill: blank the tags.
        db.conn.execute("UPDATE spans SET sub_agent_id = NULL")

        # A plain re-run inserts no new spans but overlays the now-resolvable
        # sub_agent_id back onto the existing rows.
        before = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        r_plain = ingest_claude_code(db, root=tmp_path)
        assert r_plain.spans_ingested == 0
        assert r_plain.spans_retagged == 2  # the subagent's LLM span + its tool span
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == before
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE sub_agent_id = 'rt1'"
        ).fetchone()[0] == 2

        # Nothing left for the PLAIN path to overlay — it only queues a span as
        # a candidate when a value would actually change (see
        # `_SUBAGENT_OVERLAY_MATCH_PREDICATE`'s WHERE clause).
        r_plain2 = ingest_claude_code(db, root=tmp_path)
        assert r_plain2.spans_retagged == 0
        # `--reingest` UPDATEs every existing span unconditionally (its
        # per-row `retagged` counts rows TOUCHED, not rows CHANGED — matches
        # its pre-existing attributes-overlay semantics); the data is still a
        # no-op via the same COALESCE.
        before_dump = db.conn.execute(
            "SELECT span_id, sub_agent_id, sub_agent_type FROM spans ORDER BY span_id"
        ).fetchall()
        r_re = ingest_claude_code(db, root=tmp_path, reingest=True)
        assert r_re.spans_ingested == 0
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == before
        assert db.conn.execute(
            "SELECT span_id, sub_agent_id, sub_agent_type FROM spans ORDER BY span_id"
        ).fetchall() == before_dump
    finally:
        db.close()


def test_reingest_backfills_captured_content_onto_existing_spans(tmp_path):
    """#10: enabling [capture] AFTER a session is already ingested, then
    re-running backfill, populates content / tool_input onto the EXISTING
    spans — no fresh DB required. Without this, #4's recurring-inclusion
    detection (which reads that content) only worked against a fresh DB.

    A PLAIN (non-reingest) re-run now does this too (the default bulk path
    overlays newly-available attributes onto existing spans, not just
    newly-inserted ones — see `_dedup_new_spans`'s `overlay_candidates` /
    `bulk_overlay_span_attrs`'s `json_merge_patch` half), so both paths
    converge on the same result; --reingest is left with nothing further to
    add afterwards."""
    from tokenjam.core.config import TjConfig

    _content_session_file(tmp_path)

    db = InMemoryBackend()
    try:
        # 1. First ingest with capture explicitly OFF — spans land with NO
        #    content, exactly the pre-#10 already-ingested state.
        off_cfg = TjConfig(version="1")
        off_cfg.capture = CaptureConfig(prompts=False, tool_inputs=False)
        ingest_claude_code(db, root=tmp_path, config=off_cfg)

        def _attrs(name: str) -> dict:
            raw = db.conn.execute(
                "SELECT attributes FROM spans WHERE name = $1", [name],
            ).fetchone()[0]
            return json.loads(raw) if isinstance(raw, str) else raw

        llm_before = _attrs("gen_ai.llm.call")
        tool_before = _attrs("gen_ai.tool.call")
        assert GenAIAttributes.PROMPT_CONTENT not in llm_before
        assert GenAIAttributes.COMPLETION_CONTENT not in llm_before
        assert GenAIAttributes.TOOL_INPUT not in tool_before

        before_rows = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]

        # 2. A PLAIN (non-reingest) re-run with capture ON overlays content
        #    onto the existing rows: no new spans, no new rows.
        cfg = TjConfig(version="1")
        cfg.capture = CaptureConfig(prompts=True, completions=True, tool_inputs=True)
        r_plain = ingest_claude_code(db, root=tmp_path, config=cfg)
        assert r_plain.spans_ingested == 0
        assert r_plain.spans_retagged > 0
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans"
        ).fetchone()[0] == before_rows

        llm_after = _attrs("gen_ai.llm.call")
        tool_after = _attrs("gen_ai.tool.call")
        assert llm_after[GenAIAttributes.PROMPT_CONTENT] == "please read the config"
        assert llm_after[GenAIAttributes.COMPLETION_CONTENT] == \
            "Reading the config file now."
        assert tool_after[GenAIAttributes.TOOL_INPUT] == \
            {"file_path": "/etc/app/config.toml"}
        # The pre-existing "source" key is preserved through the merge.
        assert llm_after["source"] == "backfill.claude_code"

        # 3. Nothing left to overlay -> idempotent, both on a plain re-run and
        #    on --reingest.
        r_plain2 = ingest_claude_code(db, root=tmp_path, config=cfg)
        assert r_plain2.spans_retagged == 0
        r_re = ingest_claude_code(db, root=tmp_path, config=cfg, reingest=True)
        assert r_re.spans_ingested == 0
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans"
        ).fetchone()[0] == before_rows
    finally:
        db.close()


def test_reingest_capture_off_does_not_wipe_existing_content(tmp_path):
    """#10 safety: a --reingest run with capture OFF must NOT delete content a
    prior capture-on backfill already stored — the merge overlays parsed keys,
    it never blanks the stored attributes."""
    from tokenjam.core.config import TjConfig

    _content_session_file(tmp_path)

    db = InMemoryBackend()
    try:
        # Seed with capture ON so the stored spans already carry content.
        cfg_on = TjConfig(version="1")
        cfg_on.capture = CaptureConfig(prompts=True, completions=True, tool_inputs=True)
        ingest_claude_code(db, root=tmp_path, config=cfg_on)

        # Reingest with a plain default config (prompts/tool_inputs on,
        # completions off): existing content must survive the merge either way.
        ingest_claude_code(db, root=tmp_path, config=TjConfig(version="1"), reingest=True)

        raw = db.conn.execute(
            "SELECT attributes FROM spans WHERE name = $1", ["gen_ai.llm.call"],
        ).fetchone()[0]
        attrs = json.loads(raw) if isinstance(raw, str) else raw
        assert attrs[GenAIAttributes.PROMPT_CONTENT] == "please read the config"
        assert attrs[GenAIAttributes.COMPLETION_CONTENT] == \
            "Reading the config file now."
    finally:
        db.close()


# --- method snapshot is captured at backfill so it survives a later prune ----- #

def test_backfill_captures_method_snapshot_for_ingested_session(tmp_path):
    """Backfill snapshots each newly-ingested session's reconstructed method into
    `session_story`, with source='backfill', so a historical session keeps its
    method even after Claude Code prunes the transcript."""
    from tokenjam.core.method_capture import load_session_method

    _make_session_file(
        tmp_path, session_id="sess-snap", cwd="/Users/me/proj",
        records=[
            {"type": "user", "message": {"role": "user", "content": "Do the thing."}},
            _assistant_record(
                "m-snap", "claude-opus-4-7", 1000, 200,
                "2026-04-01T10:00:00.000Z", "sess-snap", "/Users/me/proj",
                tool_uses=[("tu-snap", "Read")],
            ),
        ],
    )
    db = InMemoryBackend()
    try:
        r = ingest_claude_code(db, root=tmp_path)
        assert "sess-snap" in r.new_session_ids

        # A snapshot row exists and is tagged as a backfill capture.
        row = db.conn.execute(
            "SELECT source FROM session_story WHERE session_id = $1", ["sess-snap"],
        ).fetchone()
        assert row is not None
        assert row[0] == "backfill"

        # And load_session_method returns the reconstructed Story.
        snapshot = load_session_method(db, "sess-snap")
        assert snapshot is not None
        assert snapshot["story"] is not None
        assert snapshot["story"]["task"] == "Do the thing."
    finally:
        db.close()


def test_backfill_capture_is_noop_without_transcript_text(tmp_path):
    """A session whose transcript yields no reconstructable Story (SDK-style /
    no on-disk content for the parser) leaves no session_story row and does not
    raise — capture stays best-effort on the backfill path."""
    from tokenjam.core.method_capture import load_session_method

    # Ingest a normal session, then point capture at a DIFFERENT empty root so
    # the per-session re-read finds no transcript: no row, no raise.
    _make_session_file(
        tmp_path, session_id="sess-ghost", cwd="/Users/me/proj",
        records=[_assistant_record(
            "m-ghost", "claude-haiku-4-5", 100, 50,
            "2026-04-01T10:00:00.000Z", "sess-ghost", "/Users/me/proj",
        )],
    )
    empty_root = tmp_path / "empty"
    empty_root.mkdir()

    db = InMemoryBackend()
    try:
        # Parse from the real root, but the second arg here is what capture re-reads.
        # Simulate the no-transcript case by ingesting then capturing a session id
        # that has no file under the capture root.
        r = ingest_claude_code(db, root=tmp_path)
        assert "sess-ghost" in r.new_session_ids

        # The ingested session got a snapshot (transcript present)...
        assert load_session_method(db, "sess-ghost") is not None

        # ...but a session with no transcript under the root yields nothing and
        # never raises when capture re-reads it.
        from tokenjam.core.method_capture import capture_session_method

        assert capture_session_method(
            db, "no-such-session", projects_dir=empty_root, source="backfill"
        ) is False
        assert load_session_method(db, "no-such-session") is None
    finally:
        db.close()


def test_ingest_session_row_totals_include_subagents(tmp_path):
    """Regression: the sessions table row must reflect main + ALL subagent files,
    not just the last-processed one. Backfill upserts the row once per file with
    replace semantics, so without reconciliation the row held only one file's
    totals. Two subagents make the bug unambiguous (replace would leave 3000)."""
    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-tot", cwd=proj,
        records=[_assistant_record(
            "m-main", "claude-opus-4-7", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-tot", proj,
        )],
    )
    sub_dir = tmp_path / proj.replace("/", "-") / "sess-tot" / "subagents"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "agent-s1.jsonl").write_text(json.dumps(_assistant_record(
        "m-s1", "claude-haiku-4-5", 5000, 500,
        "2026-04-01T10:00:01.000Z", "sess-tot", proj, is_sidechain=True, agent_id="s1",
    )))
    (sub_dir / "agent-s2.jsonl").write_text(json.dumps(_assistant_record(
        "m-s2", "claude-haiku-4-5", 3000, 300,
        "2026-04-01T10:00:02.000Z", "sess-tot", proj, is_sidechain=True, agent_id="s2",
    )))

    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        sess = db.get_session("sess-tot")
        assert sess is not None
        assert sess.input_tokens == 1000 + 5000 + 3000   # main + both subagents
        assert sess.output_tokens == 200 + 500 + 300
        # The stored row total now matches the span-derived total (both include
        # every subagent), and a second ingest is idempotent (no double-count).
        assert abs((sess.total_cost_usd or 0) - db.get_session_cost("sess-tot")) < 1e-9
        ingest_claude_code(db, root=tmp_path)
        sess2 = db.get_session("sess-tot")
        assert sess2 is not None
        assert sess2.input_tokens == 9000
    finally:
        db.close()


# --- stale-scheme duplicate reconciliation (#294/#300 cross-version) --------- #

def _dup_session_records(session_id: str, cwd: str) -> list[dict]:
    """A transcript with two assistant turns (one carrying a tool_use), each with
    a stable Anthropic `message.id`. Current backfill keys span_ids on message.id;
    a pre-v0.5.2 DB keyed them on the record `uuid` (disjoint scheme)."""
    return [
        _assistant_record(
            "uuid-A", "claude-opus-4-7", 1000, 200,
            "2026-04-01T10:00:00.000Z", session_id, cwd,
            tool_uses=[("tu-A", "Read")],
            message_id="msg_A",
        ),
        _assistant_record(
            "uuid-B", "claude-opus-4-7", 500, 100,
            "2026-04-01T10:00:05.000Z", session_id, cwd,
            message_id="msg_B",
        ),
    ]


def test_backfill_purges_stale_scheme_duplicate_spans(tmp_path):
    """Simulate a pre-v0.5.2 DB (old uuid-keyed backfill spans) then re-backfill
    with current (message.id-keyed) code. The stale spans must be purged so the
    session totals equal a single clean run — not old+new inflated (#294/#300)."""
    from tokenjam.core.backfill import (
        _span_id_for_assistant,
        _span_id_for_tool,
    )

    session_id, cwd = "sess-dup", "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id=session_id, cwd=cwd,
        records=_dup_session_records(session_id, cwd),
    )

    # Baseline: a clean single run on a fresh DB gives the correct figures.
    clean = InMemoryBackend()
    try:
        ingest_claude_code(clean, root=tmp_path)
        clean_sess = clean.get_session(session_id)
        assert clean_sess is not None
        clean_span_count = clean.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE session_id = $1", [session_id]
        ).fetchone()[0]
        clean_input = clean_sess.input_tokens
        clean_output = clean_sess.output_tokens
        clean_cost = clean_sess.total_cost_usd
    finally:
        clean.close()
    # 2 LLM + 1 tool = 3 current-scheme spans.
    assert clean_span_count == 3
    assert clean_input == 1500
    assert clean_output == 300

    # Now build a DB that already holds OLD-scheme (uuid-keyed) backfill spans
    # for the same session — what a pre-v0.5.2 install left behind.
    db = InMemoryBackend()
    try:
        for uuid, in_tok, out_tok in (("uuid-A", 1000, 200), ("uuid-B", 500, 100)):
            stale_llm = make_llm_span(
                agent_id="claude-code-proj",
                model="claude-opus-4-7",
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=0.5,
                session_id=session_id,
                conversation_id=session_id,
                span_id=_span_id_for_assistant(session_id, uuid),  # OLD scheme
                extra_attributes={"source": "backfill.claude_code"},
            )
            db.insert_span(stale_llm)
        stale_tool = make_tool_span(
            agent_id="claude-code-proj",
            tool_name="Read",
            session_id=session_id,
            conversation_id=session_id,
        )
        # Override with old-scheme tool span_id + backfill source tag.
        import dataclasses
        stale_tool = dataclasses.replace(
            stale_tool,
            span_id=_span_id_for_tool(session_id, "tu-A"),  # OLD scheme
            attributes={"source": "backfill.claude_code"},
        )
        db.insert_span(stale_tool)

        # A live-ingested span in the SAME session must NEVER be touched.
        live_span = make_llm_span(
            agent_id="claude-code-proj",
            input_tokens=42,
            output_tokens=7,
            cost_usd=0.01,
            session_id=session_id,
            conversation_id=session_id,
            span_id="live-span-keepme",
            extra_attributes={"source": "live"},
        )
        db.insert_span(live_span)

        stale_before = db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE session_id = $1 "
            "AND json_extract_string(attributes, '$.source') = 'backfill.claude_code'",
            [session_id],
        ).fetchone()[0]
        assert stale_before == 3  # 2 old LLM + 1 old tool

        # Re-backfill with CURRENT code — should purge the 3 stale spans and
        # insert the 3 current-scheme spans.
        ingest_claude_code(db, root=tmp_path)

        backfill_spans = db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE session_id = $1 "
            "AND json_extract_string(attributes, '$.source') = 'backfill.claude_code'",
            [session_id],
        ).fetchone()[0]
        assert backfill_spans == 3, "stale + new must NOT coexist (was inflating)"

        # Live span survives.
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE span_id = 'live-span-keepme'"
        ).fetchone()[0] == 1

        # Session totals equal the clean single-run figures + the live span's
        # contribution (recompute_session_totals sums ALL spans in the session).
        sess = db.get_session(session_id)
        assert sess is not None
        assert sess.input_tokens == clean_input + 42
        assert sess.output_tokens == clean_output + 7
        assert abs(sess.total_cost_usd - (clean_cost + 0.01)) < 1e-6

        # Second run is a no-op: no stale rows remain to delete, spans skipped.
        r2 = ingest_claude_code(db, root=tmp_path)
        assert r2.spans_ingested == 0
        backfill_spans_2 = db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE session_id = $1 "
            "AND json_extract_string(attributes, '$.source') = 'backfill.claude_code'",
            [session_id],
        ).fetchone()[0]
        assert backfill_spans_2 == 3
    finally:
        db.close()


def test_reconcile_never_touches_other_sessions_or_sources(tmp_path):
    """reconcile_backfill_spans is scoped to (session_id, source): a backfill
    span in a DIFFERENT session and a non-backfill span are both preserved."""
    from tokenjam.core.backfill import _span_id_for_assistant

    session_id, cwd = "sess-scoped", "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id=session_id, cwd=cwd,
        records=_dup_session_records(session_id, cwd),
    )
    db = InMemoryBackend()
    try:
        # Stale backfill span in ANOTHER session.
        other = make_llm_span(
            session_id="other-session",
            span_id=_span_id_for_assistant("other-session", "uuid-X"),
            extra_attributes={"source": "backfill.claude_code"},
        )
        db.insert_span(other)
        ingest_claude_code(db, root=tmp_path)
        # Other session's backfill span untouched.
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE session_id = 'other-session'"
        ).fetchone()[0] == 1
    finally:
        db.close()


# --- columnar bulk-append batching: parity across the flush boundary --------- #

def _multi_session_tree(tmp_path, n_sessions: int = 6) -> None:
    """A handful of small sessions, each with an LLM turn + a tool_use — enough
    that a small flush target forces several batch flushes."""
    for i in range(n_sessions):
        sid = f"batch-sess-{i:02d}"
        _make_session_file(
            tmp_path, session_id=sid, cwd="/Users/me/proj",
            records=[_assistant_record(
                f"m-{i}", "claude-opus-4-7", 1000 + i, 200 + i,
                "2026-04-01T10:00:00.000Z", sid, "/Users/me/proj",
                tool_uses=[(f"tu-{i}", "Read")], cache_read=i, cache_creation=2 * i,
            )],
        )


def _dump_spans(db) -> list[tuple]:
    return db.conn.execute(
        "SELECT span_id, session_id, name, input_tokens, output_tokens, "
        "cache_tokens, cache_write_tokens, cost_usd, tool_name "
        "FROM spans ORDER BY span_id"
    ).fetchall()


def _dump_sessions(db) -> list[tuple]:
    return db.conn.execute(
        "SELECT session_id, input_tokens, output_tokens, cache_tokens, "
        "cache_write_tokens, total_cost_usd, tool_call_count "
        "FROM sessions ORDER BY session_id"
    ).fetchall()


def test_bulk_batch_flush_boundary_matches_single_flush(tmp_path, monkeypatch):
    """A tiny flush target (multiple batch flushes, sessions split across
    boundaries) must produce byte-for-byte the same spans + session rows as a
    single-flush ingest — the whole point of set-based batching is that where the
    flush boundary lands is invisible in the result."""
    import tokenjam.core.backfill as bf

    _multi_session_tree(tmp_path, n_sessions=6)

    # Reference: one big batch (target far above the total span count).
    ref = InMemoryBackend()
    # Forced multi-flush: target of 1 span flushes essentially per session.
    chunked = InMemoryBackend()
    try:
        monkeypatch.setattr(bf, "_BULK_FLUSH_SPAN_TARGET", 10_000)
        r_ref = ingest_claude_code(ref, root=tmp_path)

        monkeypatch.setattr(bf, "_BULK_FLUSH_SPAN_TARGET", 1)
        r_chunked = ingest_claude_code(chunked, root=tmp_path)

        assert _dump_spans(chunked) == _dump_spans(ref)
        assert _dump_sessions(chunked) == _dump_sessions(ref)
        # Reported counts agree too (6 sessions × 2 spans each).
        assert r_chunked.spans_ingested == r_ref.spans_ingested == 12
        assert r_chunked.sessions_ingested == r_ref.sessions_ingested == 6
        assert r_chunked.sessions_new == r_ref.sessions_new == 6
    finally:
        ref.close()
        chunked.close()


def test_bulk_progress_counts_increase_monotonically(tmp_path):
    """Regression: in the bulk path the progress callback must observe
    `spans_ingested` climbing per session — not sit at flat zero until a single
    end-of-run flush, which would make a live backfill display (onboard) show 0
    the whole run then jump. Uses the DEFAULT (large) flush target so every
    insert is deferred to ONE flush at the end — the exact scenario that exposed
    the bug — and asserts the callback still saw increasing counts."""
    _multi_session_tree(tmp_path, n_sessions=6)  # 6 sessions × 2 spans each

    observed: list[int] = []

    def _capture(parsed, result):
        observed.append(result.spans_ingested)

    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path, progress=_capture)
        # One callback per session; counts advance 2 per session (LLM + tool),
        # never flat-zero-then-jump.
        assert observed == [2, 4, 6, 8, 10, 12]
        assert observed[0] > 0
        assert all(b > a for a, b in zip(observed, observed[1:]))
        # And the spans really did land (deferred flush committed at the end).
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 12
    finally:
        db.close()


def test_bulk_batch_is_idempotent_across_flush_boundary(tmp_path, monkeypatch):
    """Re-running the backfill with a tiny flush target inserts nothing new and
    leaves the spans unchanged — idempotency holds across batch boundaries."""
    import tokenjam.core.backfill as bf

    _multi_session_tree(tmp_path, n_sessions=5)
    db = InMemoryBackend()
    try:
        monkeypatch.setattr(bf, "_BULK_FLUSH_SPAN_TARGET", 3)
        r1 = ingest_claude_code(db, root=tmp_path)
        before = _dump_spans(db)

        r2 = ingest_claude_code(db, root=tmp_path)
        assert r1.spans_ingested == 10          # 5 sessions × 2 spans
        assert r2.spans_ingested == 0
        assert r2.spans_skipped_existing == 10
        assert _dump_spans(db) == before        # untouched
    finally:
        db.close()


# --- Stable subagent identity (sub_agent_type) --------------------------------
#
# `sub_agent_id` is Claude Code's `agentId`: minted fresh per Task dispatch, so
# it never recurs across sessions and can form no per-subagent cohort. The
# stable identity is the dispatched agent TYPE, read off the
# `agent-<id>.meta.json` sidecar Claude Code writes next to every subagent
# transcript. See `backfill._subagent_type_for`.

def _subagent_transcript(tmp_path: Path, proj: str, session_id: str,
                         agent_id: str, records: list[dict],
                         meta: dict | None) -> Path:
    """Write ``<proj>/<sid>/subagents/agent-<id>.jsonl`` plus its meta sidecar.

    ``meta=None`` writes no sidecar (the pre-sidecar / unreadable case).
    """
    sub_dir = tmp_path / proj.replace("/", "-") / session_id / "subagents"
    sub_dir.mkdir(parents=True, exist_ok=True)
    path = sub_dir / f"agent-{agent_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    if meta is not None:
        (sub_dir / f"agent-{agent_id}.meta.json").write_text(json.dumps(meta))
    return path


def test_parse_reads_the_stable_subagent_type_from_the_meta_sidecar(tmp_path):
    """Every span of a subagent transcript carries the dispatched agent TYPE
    alongside its per-dispatch id — the id keeps its old meaning."""
    proj = "/Users/me/proj"
    path = _subagent_transcript(
        tmp_path, proj, "sess-st", "a1f2e3d4c5b6a7981",
        records=[_assistant_record(
            "m-st", "claude-opus-4-7", 5000, 500,
            "2026-04-01T10:00:01.000Z", "sess-st", proj,
            tool_uses=[("tu-st", "Read")],
            is_sidechain=True, agent_id="a1f2e3d4c5b6a7981",
        )],
        meta={"agentType": "code-reviewer", "description": "review the diff",
              "toolUseId": "toolu_01abc", "spawnDepth": 1},
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert len(parsed.spans) == 2                       # LLM span + tool span
    for span in parsed.spans:
        assert span.sub_agent_id == "a1f2e3d4c5b6a7981"  # per-dispatch, unchanged
        assert span.sub_agent_type == "code-reviewer"    # stable, new


def test_a_main_thread_transcript_never_carries_a_subagent_type(tmp_path):
    """The type is gated on the same isSidechain flag as the id, so a
    main-thread span can never pick one up."""
    path = _make_session_file(
        tmp_path, session_id="sess-main", cwd="/Users/me/proj",
        records=[_assistant_record(
            "m-main", "claude-opus-4-7", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-main", "/Users/me/proj",
        )],
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert all(s.sub_agent_type is None for s in parsed.spans)


def test_a_per_dispatch_instance_label_is_not_recorded_as_a_stable_type(tmp_path):
    """An ``in_process_teammate`` is spawned with a caller-chosen one-off
    ``name`` ("worker-428") that Claude Code writes into ``agentType``. That
    names no agent definition and recurs no more than the dispatch id does, so
    recording it as a stable type would silently mis-attribute. It stays NULL.
    """
    proj = "/Users/me/proj"
    path = _subagent_transcript(
        tmp_path, proj, "sess-tm", "aworker-428-63df1d3c53338de1",
        records=[_assistant_record(
            "m-tm", "claude-opus-4-7", 5000, 500,
            "2026-04-01T10:00:01.000Z", "sess-tm", proj,
            is_sidechain=True, agent_id="aworker-428-63df1d3c53338de1",
        )],
        meta={"agentType": "worker-428", "name": "worker-428",
              "taskKind": "in_process_teammate", "description": "fix a review finding"},
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert all(s.sub_agent_id == "aworker-428-63df1d3c53338de1" for s in parsed.spans)
    assert all(s.sub_agent_type is None for s in parsed.spans)


def test_a_workflow_nested_subagent_still_resolves_its_type(tmp_path):
    """A workflow dispatch nests one level deeper —
    ``subagents/workflows/<wf-id>/agent-<id>.jsonl`` — so ``subagents`` is not
    the immediate parent. Matching on the immediate parent alone silently
    dropped the type for every one of these (998 transcripts on a real corpus,
    the second-largest dispatch type there)."""
    proj = "/Users/me/proj"
    sub_dir = (tmp_path / proj.replace("/", "-") / "sess-wf"
               / "subagents" / "workflows" / "wf_5ef70bd5-87f")
    sub_dir.mkdir(parents=True)
    path = sub_dir / "agent-awf1.jsonl"
    path.write_text(json.dumps(_assistant_record(
        "m-wf", "claude-opus-4-7", 5000, 500,
        "2026-04-01T10:00:01.000Z", "sess-wf", proj,
        is_sidechain=True, agent_id="awf1",
    )))
    (sub_dir / "agent-awf1.meta.json").write_text(
        json.dumps({"agentType": "workflow-subagent", "description": "step 1"})
    )
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert all(s.sub_agent_type == "workflow-subagent" for s in parsed.spans)


def test_a_missing_or_garbled_sidecar_degrades_to_no_type(tmp_path):
    """No sidecar, and a sidecar that isn't readable JSON, both leave the type
    NULL rather than raising — the id half is unaffected either way."""
    proj = "/Users/me/proj"
    records = [_assistant_record(
        "m-ns", "claude-opus-4-7", 5000, 500,
        "2026-04-01T10:00:01.000Z", "sess-ns", proj,
        is_sidechain=True, agent_id="ans1",
    )]
    path = _subagent_transcript(tmp_path, proj, "sess-ns", "ans1", records, meta=None)
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert all(s.sub_agent_type is None for s in parsed.spans)
    assert all(s.sub_agent_id == "ans1" for s in parsed.spans)

    (path.parent / "agent-ans1.meta.json").write_text("{not json")
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert all(s.sub_agent_type is None for s in parsed.spans)


def test_the_stable_type_forms_a_cohort_the_dispatch_id_never_can(tmp_path):
    """The acceptance shape: one agent type dispatched once per session across
    five sessions is ONE identity with five sessions on `sub_agent_type`, and
    five identities with one session each on `sub_agent_id`."""
    proj = "/Users/me/proj"
    for i in range(5):
        sid = f"sess-co-{i}"
        _make_session_file(
            tmp_path, session_id=sid, cwd=proj,
            records=[_assistant_record(
                f"m-main-{i}", "claude-opus-4-7", 1000, 200,
                f"2026-04-0{i + 1}T10:00:00.000Z", sid, proj,
            )],
        )
        _subagent_transcript(
            tmp_path, proj, sid, f"a{i}f2e3d4c5b6a7981",
            records=[_assistant_record(
                f"m-sub-{i}", "claude-opus-4-7", 5000, 500,
                f"2026-04-0{i + 1}T10:00:01.000Z", sid, proj,
                is_sidechain=True, agent_id=f"a{i}f2e3d4c5b6a7981",
            )],
            meta={"agentType": "code-reviewer", "description": "review"},
        )

    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        by_id = db.conn.execute(
            "SELECT COUNT(DISTINCT sub_agent_id), MAX(n) FROM ("
            "  SELECT sub_agent_id, COUNT(DISTINCT session_id) AS n FROM spans"
            "  WHERE sub_agent_id IS NOT NULL GROUP BY sub_agent_id)"
        ).fetchone()
        assert by_id == (5, 1), "every dispatch id is confined to one session"
        by_type = db.conn.execute(
            "SELECT COUNT(DISTINCT sub_agent_type), MAX(n) FROM ("
            "  SELECT sub_agent_type, COUNT(DISTINCT session_id) AS n FROM spans"
            "  WHERE sub_agent_type IS NOT NULL GROUP BY sub_agent_type)"
        ).fetchone()
        assert by_type == (1, 5), "one identity, a five-session cohort"
    finally:
        db.close()


def test_reingest_retags_the_stable_type_without_disturbing_anything_else(tmp_path):
    """`--reingest` populates `sub_agent_type` on spans an older backfill wrote
    before the column existed, inserts nothing, and re-running it again changes
    nothing further."""
    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-rst", cwd=proj,
        records=[_assistant_record(
            "m-main", "claude-opus-4-7", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-rst", proj,
        )],
    )
    _subagent_transcript(
        tmp_path, proj, "sess-rst", "arst1",
        records=[_assistant_record(
            "m-rst", "claude-haiku-4-5", 5000, 500,
            "2026-04-01T10:00:01.000Z", "sess-rst", proj,
            tool_uses=[("tu-rst", "Read")], is_sidechain=True, agent_id="arst1",
        )],
        meta={"agentType": "code-reviewer", "description": "review"},
    )

    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        # Simulate history ingested before migration 19 landed.
        db.conn.execute("UPDATE spans SET sub_agent_type = NULL")
        before_rows = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        before_tokens = db.conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens), SUM(cost_usd) FROM spans"
        ).fetchone()

        r = ingest_claude_code(db, root=tmp_path, reingest=True)
        assert r.spans_ingested == 0
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == before_rows
        assert db.conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens), SUM(cost_usd) FROM spans"
        ).fetchone() == before_tokens
        # The subagent's LLM span AND its tool span both carry the type; the
        # main-thread span is untouched.
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE sub_agent_type = 'code-reviewer'"
        ).fetchone()[0] == 2
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE sub_agent_id IS NULL "
            "AND sub_agent_type IS NOT NULL"
        ).fetchone()[0] == 0

        # Idempotent: a second reingest is a no-op on the data. Snapshot the
        # identity columns explicitly — `_dump_spans` doesn't carry them.
        def _identity_dump() -> list[tuple]:
            return db.conn.execute(
                "SELECT span_id, sub_agent_id, sub_agent_type FROM spans "
                "ORDER BY span_id"
            ).fetchall()

        snapshot = (_dump_spans(db), _identity_dump())
        ingest_claude_code(db, root=tmp_path, reingest=True)
        assert (_dump_spans(db), _identity_dump()) == snapshot
    finally:
        db.close()


def test_a_fresh_ingest_through_the_full_pipeline_resolves_the_type_via_the_sidecar(
    tmp_path,
):
    """Pins the LINKAGE, not just the derivation: a session ingested through
    `ingest_claude_code` (the mechanism a continuous catch-up run and
    `tj backfill claude-code` both call — there is no separate ingest path for
    Claude Code subagent data; see `core/transcript_sync.py`'s module
    docstring) lands `sub_agent_type` on its subagent spans straight from a
    realistic on-disk `<project>/<session-uuid>/subagents/agent-<id>.jsonl` +
    `.meta.json` layout, on the very first ingest — no reingest/repair needed
    when the sidecar is present from the start."""
    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-link", cwd=proj,
        records=[_assistant_record(
            "m-main", "claude-opus-4-7", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-link", proj,
        )],
    )
    _subagent_transcript(
        tmp_path, proj, "sess-link", "alink1",
        records=[_assistant_record(
            "m-link", "claude-haiku-4-5", 5000, 500,
            "2026-04-01T10:00:01.000Z", "sess-link", proj,
            tool_uses=[("tu-link", "Read")], is_sidechain=True, agent_id="alink1",
        )],
        meta={"agentType": "code-reviewer", "description": "review the diff"},
    )

    db = InMemoryBackend()
    try:
        result = ingest_claude_code(db, root=tmp_path)
        assert result.spans_ingested == 3  # main LLM + subagent LLM + its tool span
        rows = db.conn.execute(
            "SELECT sub_agent_id, sub_agent_type FROM spans "
            "WHERE sub_agent_id IS NOT NULL"
        ).fetchall()
        assert rows and all(r == ("alink1", "code-reviewer") for r in rows)
    finally:
        db.close()


def test_a_plain_backfill_overlays_newly_resolvable_type_onto_existing_spans(
    tmp_path,
):
    """The Part-2 gap: a default (non-`--reingest`) backfill re-run must
    OVERLAY a newly-resolvable `sub_agent_type` onto rows that were already in
    the DB, not just skip them as already-present. Before this, the bulk path
    (`_dedup_new_spans`) only ever partitioned new-vs-existing and silently
    dropped the existing half — a span written before migration 19 (or before
    its sidecar existed) could never gain a type on its own."""
    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-ov", cwd=proj,
        records=[_assistant_record(
            "m-main", "claude-opus-4-7", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-ov", proj,
        )],
    )
    _subagent_transcript(
        tmp_path, proj, "sess-ov", "aov1",
        records=[_assistant_record(
            "m-ov", "claude-haiku-4-5", 5000, 500,
            "2026-04-01T10:00:01.000Z", "sess-ov", proj,
            tool_uses=[("tu-ov", "Read")], is_sidechain=True, agent_id="aov1",
        )],
        meta={"agentType": "code-reviewer", "description": "review the diff"},
    )

    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        # Simulate history ingested before migration 19 landed.
        db.conn.execute("UPDATE spans SET sub_agent_type = NULL")
        before_rows = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        before_tokens = db.conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens), SUM(cost_usd) FROM spans"
        ).fetchone()

        # A PLAIN re-run (no --reingest) must overlay the type back on.
        r = ingest_claude_code(db, root=tmp_path)
        assert r.spans_ingested == 0  # nothing new — every span already existed
        assert r.spans_retagged == 2  # the subagent's LLM span + its tool span
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == before_rows
        assert db.conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens), SUM(cost_usd) FROM spans"
        ).fetchone() == before_tokens  # additive-only: no other field moved
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE sub_agent_type = 'code-reviewer'"
        ).fetchone()[0] == 2

        # Idempotent: a second plain re-run finds nothing left to overlay.
        r2 = ingest_claude_code(db, root=tmp_path)
        assert r2.spans_ingested == 0
        assert r2.spans_retagged == 0
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == before_rows
    finally:
        db.close()


def test_the_overlay_never_clobbers_an_already_resolved_type_with_a_different_one(
    tmp_path,
):
    """Additive means additive: if a span already carries a (non-NULL)
    `sub_agent_type` from any source, a re-parse that would resolve to a
    DIFFERENT value must never overwrite it. Guards the `COALESCE` in
    `_SUBAGENT_OVERLAY_MATCH_PREDICATE` — an unconditional `UPDATE ... SET
    sub_agent_type = src.sub_agent_type` would silently flip an already-correct
    value the moment two runs disagree (e.g. a sidecar that changed between
    runs, or a value set by a future direct-write path)."""
    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-noclob", cwd=proj,
        records=[_assistant_record(
            "m-main", "claude-opus-4-7", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-noclob", proj,
        )],
    )
    _subagent_transcript(
        tmp_path, proj, "sess-noclob", "anc1",
        records=[_assistant_record(
            "m-nc", "claude-haiku-4-5", 5000, 500,
            "2026-04-01T10:00:01.000Z", "sess-noclob", proj,
            is_sidechain=True, agent_id="anc1",
        )],
        meta={"agentType": "code-reviewer", "description": "review the diff"},
    )

    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        # Pretend a different value was already resolved for this span.
        db.conn.execute(
            "UPDATE spans SET sub_agent_type = 'planner' WHERE sub_agent_id = 'anc1'"
        )
        ingest_claude_code(db, root=tmp_path)  # re-parses the SAME transcript
        assert db.conn.execute(
            "SELECT DISTINCT sub_agent_type FROM spans WHERE sub_agent_id = 'anc1'"
        ).fetchone() == ("planner",)
    finally:
        db.close()


def test_doctor_reports_and_repairs_unresolved_subagent_types(tmp_path, monkeypatch):
    """The user-facing route into the repair: `tj doctor` names the gap and
    `--repair` closes as much of it as the on-disk transcripts still support,
    same shape as the duplicate-call-ingest repair
    (test_doctor_reports_and_repairs_a_double_ingested_db in
    test_ingest_accounting.py)."""
    from tokenjam.cli.cmd_doctor import (
        _attempt_repairs,
        _check_unresolved_subagent_type,
    )

    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-doc", cwd=proj,
        records=[_assistant_record(
            "m-main", "claude-opus-4-7", 1000, 200,
            "2026-04-01T10:00:00.000Z", "sess-doc", proj,
        )],
    )
    _subagent_transcript(
        tmp_path, proj, "sess-doc", "adoc1",
        records=[_assistant_record(
            "m-doc", "claude-haiku-4-5", 5000, 500,
            "2026-04-01T10:00:01.000Z", "sess-doc", proj,
            is_sidechain=True, agent_id="adoc1",
        )],
        meta={"agentType": "code-reviewer", "description": "review the diff"},
    )

    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        # Simulate the pre-migration-19 corpus this check exists for.
        db.conn.execute("UPDATE spans SET sub_agent_type = NULL")

        check = _check_unresolved_subagent_type(db)
        assert check["level"] == "warning"
        assert check["repair_action"] == "resolve_subagent_types"
        assert "1 subagent span" in check["message"]

        # The repair walks the real CLAUDE_CODE_PROJECTS_ROOT by default — the
        # repair branch imports it fresh from core.backfill at call time, so
        # patch it there to point at our fixture tree instead.
        monkeypatch.setattr(
            "tokenjam.core.backfill.CLAUDE_CODE_PROJECTS_ROOT", tmp_path,
        )
        _attempt_repairs([check], db, output_json=True)

        assert _check_unresolved_subagent_type(db)["level"] == "ok"
        assert db.conn.execute(
            "SELECT sub_agent_type FROM spans WHERE sub_agent_id = 'adoc1'"
        ).fetchone()[0] == "code-reviewer"
    finally:
        db.close()
