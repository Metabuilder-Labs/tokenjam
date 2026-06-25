"""Unit tests for the backfill parser + ingest path."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tokenjam.core.backfill import (
    ingest_claude_code,
    iter_claude_code_sessions,
    parse_claude_code_session,
)
from tokenjam.core.db import InMemoryBackend


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
                       is_sidechain: bool = False,
                       agent_id: str | None = None,
                       cache_read: int = 0, cache_creation: int = 0,
                       message_id: str | None = None) -> dict:
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


# --- #3: capture-gated per-message content + tool_input on backfill --------- #

from tokenjam.core.config import CaptureConfig  # noqa: E402
from tokenjam.otel.semconv import GenAIAttributes  # noqa: E402


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


def test_capture_off_leaves_attributes_unchanged(tmp_path):
    """Default (no capture / all-False) extracts NO content — attributes stay
    exactly {"source": ...} on both the LLM and tool span (#3 default-off)."""
    path = _content_session_file(tmp_path)

    # Both the explicit None default and an all-False CaptureConfig.
    for capture in (None, CaptureConfig()):
        parsed = parse_claude_code_session(path, capture=capture)
        assert parsed is not None
        llm, tool = _llm_and_tool(parsed)
        assert llm.attributes == {"source": "backfill.claude_code"}
        assert tool.attributes == {"source": "backfill.claude_code"}


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
    """Each toggle gates only its own field — flipping one never leaks another."""
    path = _content_session_file(tmp_path)

    parsed = parse_claude_code_session(path, capture=CaptureConfig(tool_inputs=True))
    llm, tool = _llm_and_tool(parsed)
    assert GenAIAttributes.TOOL_INPUT in tool.attributes
    assert GenAIAttributes.PROMPT_CONTENT not in llm.attributes
    assert GenAIAttributes.COMPLETION_CONTENT not in llm.attributes

    parsed = parse_claude_code_session(path, capture=CaptureConfig(completions=True))
    llm, tool = _llm_and_tool(parsed)
    assert GenAIAttributes.COMPLETION_CONTENT in llm.attributes
    assert GenAIAttributes.PROMPT_CONTENT not in llm.attributes
    assert GenAIAttributes.TOOL_INPUT not in tool.attributes


def test_ingest_persists_captured_content_when_config_enables_it(tmp_path):
    """End-to-end through ingest: with config.capture enabled, the stored span's
    attributes column carries the content; default config stores nothing."""
    from tokenjam.core.config import TjConfig

    _content_session_file(tmp_path)

    # Default config -> capture all-False -> no content persisted.
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path, config=TjConfig(version="1"))
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
    before the column existed; a plain idempotent re-run leaves them NULL."""
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

        # Plain re-run is idempotent -> existing spans skipped -> still NULL.
        r_plain = ingest_claude_code(db, root=tmp_path)
        assert r_plain.spans_ingested == 0
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE sub_agent_id IS NOT NULL"
        ).fetchone()[0] == 0

        # --reingest re-tags in place: no new rows, no duplicates.
        before = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        r_re = ingest_claude_code(db, root=tmp_path, reingest=True)
        assert r_re.spans_ingested == 0
        assert r_re.spans_retagged > 0
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == before
        # The subagent's LLM span + its tool span are both re-tagged.
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE sub_agent_id = 'rt1'"
        ).fetchone()[0] == 2
    finally:
        db.close()


def test_reingest_backfills_captured_content_onto_existing_spans(tmp_path):
    """#10: enabling [capture] AFTER a session is already ingested, then
    re-running backfill with --reingest, populates content / tool_input onto the
    EXISTING spans — no fresh DB required. Without this, #4's recurring-inclusion
    detection (which reads that content) only worked against a fresh DB."""
    from tokenjam.core.config import TjConfig

    _content_session_file(tmp_path)

    db = InMemoryBackend()
    try:
        # 1. First ingest with capture OFF (default config) — spans land with
        #    NO content, exactly the pre-#10 already-ingested state.
        ingest_claude_code(db, root=tmp_path, config=TjConfig(version="1"))

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

        # 2. A plain (non-reingest) re-run with capture ON still does NOT touch
        #    existing rows — the conflict path skips them. This is the gap #10
        #    fixes: --reingest is required.
        cfg = TjConfig(version="1")
        cfg.capture = CaptureConfig(prompts=True, completions=True, tool_inputs=True)
        r_plain = ingest_claude_code(db, root=tmp_path, config=cfg)
        assert r_plain.spans_ingested == 0
        assert GenAIAttributes.PROMPT_CONTENT not in _attrs("gen_ai.llm.call")

        before_rows = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]

        # 3. --reingest WITH capture on backfills content in place: no new rows.
        r_re = ingest_claude_code(db, root=tmp_path, config=cfg, reingest=True)
        assert r_re.spans_ingested == 0
        assert r_re.spans_retagged > 0
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

        # Reingest with capture OFF (default config): content must survive.
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


# --- #15: bulk insert/update path --------------------------------------------

class _CountingConn:
    """Wraps a DuckDB cursor and counts execute / executemany calls so a test
    can assert the bulk path issues a BOUNDED number of statements for a large
    session (not ~2 per span). Bind-param *rows* passed to executemany don't
    each count as a statement — that's the whole point of the bulk path.
    """

    def __init__(self, inner):
        self._inner = inner
        self.execute_calls = 0
        self.executemany_calls = 0

    def execute(self, *args, **kwargs):
        self.execute_calls += 1
        return self._inner.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        self.executemany_calls += 1
        return self._inner.executemany(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _CountingBackend(InMemoryBackend):
    """InMemoryBackend whose `conn` property hands back a `_CountingConn` so a
    test can count execute/executemany calls. `conn` is a data-descriptor
    property on DuckDBBackend, so wrapping must be done at the property level —
    setting an instance attribute would never be seen by `getattr(db, "conn")`.
    """

    def __init__(self):
        super().__init__()
        self._counting = _CountingConn(super().conn)

    @property
    def conn(self):  # type: ignore[override]
        return self._counting


def _big_session_file(tmp_path: Path, n_assistant: int, session_id: str = "sess-big",
                      cwd: str = "/Users/me/big") -> Path:
    """A single session with `n_assistant` assistant turns, each carrying one
    tool_use — so 2*n spans land from one file."""
    records = []
    for i in range(n_assistant):
        records.append(_assistant_record(
            f"m-{i}", "claude-haiku-4-5", 1000 + i, 100 + i,
            f"2026-04-01T10:{i // 60:02d}:{i % 60:02d}.000Z", session_id, cwd,
            tool_uses=[(f"tu-{i}", "Read")],
        ))
    return _make_session_file(tmp_path, session_id=session_id, cwd=cwd, records=records)


def test_bulk_insert_issues_bounded_statements_not_per_span(tmp_path):
    """#15: a large session must NOT do a SELECT+INSERT round-trip per span.
    The bulk path partitions existence in ONE query and inserts via a single
    executemany, so the statement count is BOUNDED regardless of span count.
    """
    n = 200  # 200 assistant turns * (1 llm + 1 tool) = 400 spans
    _big_session_file(tmp_path, n_assistant=n)

    db = _CountingBackend()
    try:
        counting = db.conn
        r = ingest_claude_code(db, root=tmp_path)
        assert r.spans_ingested == 2 * n  # all spans inserted

        # The per-span path would issue ~2*N execute() calls (one SELECT + one
        # INSERT each). The bulk path issues a small bounded number: a handful of
        # partition SELECTs + a SINGLE bulk-insert executemany. Assert we're FAR
        # below per-span — well under N, and the insert went through executemany.
        assert counting.executemany_calls >= 1
        assert counting.execute_calls < n, (
            f"expected bounded statements, got {counting.execute_calls} "
            f"execute() calls for {2 * n} spans"
        )
    finally:
        db.close()


def test_bulk_path_new_spans_insert_correctly(tmp_path):
    """#15: the bulk insert lands every new span with correct content."""
    _big_session_file(tmp_path, n_assistant=50)
    db = InMemoryBackend()
    try:
        r = ingest_claude_code(db, root=tmp_path)
        assert r.spans_ingested == 100
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 100
        # Spot-check a known row round-tripped.
        row = db.conn.execute(
            "SELECT input_tokens, output_tokens, tool_name FROM spans "
            "WHERE name = 'gen_ai.tool.call' LIMIT 1"
        ).fetchone()
        assert row[2] == "Read"
    finally:
        db.close()


def test_bulk_path_existing_spans_untouched_without_reingest(tmp_path):
    """#15: existing spans are skipped (not re-inserted, not duplicated) on a
    plain re-run — the bulk partition preserves the no-reingest skip contract."""
    _big_session_file(tmp_path, n_assistant=30)
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        before = db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
        # Blank sub_agent_id to prove a non-reingest re-run leaves rows untouched.
        db.conn.execute("UPDATE spans SET sub_agent_id = 'sentinel'")

        r2 = ingest_claude_code(db, root=tmp_path)
        assert r2.spans_ingested == 0
        assert r2.spans_skipped_existing == before
        assert db.conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == before
        # Untouched: the sentinel survives (no UPDATE ran without --reingest).
        assert db.conn.execute(
            "SELECT COUNT(*) FROM spans WHERE sub_agent_id = 'sentinel'"
        ).fetchone()[0] == before
    finally:
        db.close()


def test_bulk_reingest_merges_attributes_and_updates_subagent(tmp_path):
    """#15: the bulk UPDATE path preserves the #10 reingest contract — existing
    spans get sub_agent_id updated AND attributes per-key merged (overlay, no
    wipe) — and does so in batched executemany, not one UPDATE per span."""
    from tokenjam.core.config import TjConfig

    _content_session_file(tmp_path)
    db = _CountingBackend()
    counting = db.conn
    try:
        # 1. First ingest capture OFF -> spans land with no content.
        ingest_claude_code(db, root=tmp_path, config=TjConfig(version="1"))
        # Simulate pre-column history: blank the tag, add a stored-only key that
        # the merge must PRESERVE (parsed wins per-key, never wipes).
        counting.execute("UPDATE spans SET sub_agent_id = NULL")
        counting.execute(
            "UPDATE spans SET attributes = $1 WHERE name = 'gen_ai.llm.call'",
            [json.dumps({"source": "backfill.claude_code", "keepme": "yes"})],
        )

        before_rows = counting.execute("SELECT COUNT(*) FROM spans").fetchone()[0]

        # 2. --reingest with capture ON; assert the UPDATEs are batched.
        cfg = TjConfig(version="1")
        cfg.capture = CaptureConfig(prompts=True, completions=True, tool_inputs=True)
        counting.executemany_calls = 0  # reset; count only the reingest's writes

        r = ingest_claude_code(db, root=tmp_path, config=cfg, reingest=True)
        assert r.spans_ingested == 0
        assert r.spans_retagged > 0
        # No new rows.
        assert counting.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == before_rows
        # The UPDATEs went through executemany (batched), not per-span execute().
        assert counting.executemany_calls >= 1

        def _attrs(name: str) -> dict:
            raw = counting.execute(
                "SELECT attributes FROM spans WHERE name = $1", [name]
            ).fetchone()[0]
            return json.loads(raw) if isinstance(raw, str) else raw

        llm = _attrs("gen_ai.llm.call")
        # Parsed content was overlaid...
        assert llm[GenAIAttributes.PROMPT_CONTENT] == "please read the config"
        assert llm[GenAIAttributes.COMPLETION_CONTENT] == "Reading the config file now."
        # ...the stored-only key survived (overlay, never wipe)...
        assert llm["keepme"] == "yes"
        assert llm["source"] == "backfill.claude_code"
        # ...and tool_input landed on the tool span.
        assert _attrs("gen_ai.tool.call")[GenAIAttributes.TOOL_INPUT] == \
            {"file_path": "/etc/app/config.toml"}
    finally:
        db.close()


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
