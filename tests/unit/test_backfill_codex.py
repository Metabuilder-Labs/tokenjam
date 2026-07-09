"""Unit tests for the Codex CLI rollout backfill adapter.

Mirrors tests/unit/test_backfill.py (the Claude Code path): build on-disk
rollout JSONL fixtures, parse/ingest them, and assert on the produced spans and
session records. Spans are the OUTPUT under test (parsed from disk), so — like
the Claude Code backfill tests — they're produced by the adapter, never
hand-constructed (Critical Rule 8; the NormalizedSpan factory is for test
*inputs*, and this adapter's input is JSONL on disk).
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest_adapters.codex import (
    ingest_codex,
    iter_codex_sessions,
    parse_codex_rollout,
)


# --- Rollout fixture builders (mirror the real codex-rs RolloutLine shape) ----

def _session_meta_line(session_id: str, cwd: str = "/Users/me/proj",
                       ts: str = "2026-07-01T10:00:00Z") -> dict:
    return {
        "timestamp": ts,
        "type": "session_meta",
        "payload": {
            "session_id": session_id,
            "id": session_id,
            "timestamp": ts,
            "cwd": cwd,
            "originator": "codex_cli_rs",
            "cli_version": "0.130.0",
            "model_provider": "openai",
        },
    }


def _turn_context_line(model: str, cwd: str = "/Users/me/proj",
                       ts: str = "2026-07-01T10:00:01Z") -> dict:
    return {
        "timestamp": ts,
        "type": "turn_context",
        "payload": {
            "cwd": cwd,
            "approval_policy": "on-request",
            "sandbox_policy": {"mode": "workspace-write"},
            "model": model,
        },
    }


def _token_count_line(input_tokens: int, cached_input_tokens: int,
                      output_tokens: int, reasoning_output_tokens: int = 0,
                      ts: str = "2026-07-01T10:00:05Z") -> dict:
    total = input_tokens + output_tokens + reasoning_output_tokens
    usage = {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "total_tokens": total,
    }
    return {
        "timestamp": ts,
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": usage,
                "last_token_usage": usage,
                "model_context_window": 272000,
            },
            "rate_limits": None,
        },
    }


def _function_call_line(name: str, call_id: str,
                        ts: str = "2026-07-01T10:00:04Z") -> dict:
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": name,
            "arguments": "{}",
            "call_id": call_id,
        },
    }


def _write_rollout(tmp_path: Path, session_id: str, lines: list[dict],
                   subdir: str = "2026/07/01") -> Path:
    d = tmp_path / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"rollout-2026-07-01T10-00-00-{session_id}.jsonl"
    p.write_text("\n".join(json.dumps(rec) for rec in lines) + "\n",
                 encoding="utf-8")
    return p


# --- Parser tests ------------------------------------------------------------

def test_parse_extracts_llm_and_tool_spans(tmp_path):
    path = _write_rollout(tmp_path, "sess-a", [
        _session_meta_line("sess-a"),
        _turn_context_line("gpt-5.5"),
        _function_call_line("shell", "call_1"),
        _token_count_line(input_tokens=1000, cached_input_tokens=200,
                          output_tokens=300, reasoning_output_tokens=50),
    ])
    parsed = parse_codex_rollout(path)
    assert parsed is not None
    assert parsed.session_id == "sess-a"
    assert parsed.cwd == "/Users/me/proj"
    assert parsed.agent_id == "codex_exec"

    llm = [s for s in parsed.spans if s.name == "gen_ai.llm.call"]
    tools = [s for s in parsed.spans if s.name == "gen_ai.tool.call"]
    assert len(llm) == 1
    assert len(tools) == 1

    span = llm[0]
    assert span.provider == "openai"
    assert span.billing_account == "openai"
    assert span.model == "gpt-5.5"
    # non-cached input = input - cached; reasoning folds into output
    assert span.input_tokens == 800
    assert span.output_tokens == 350
    assert span.cache_tokens == 200
    assert span.cache_write_tokens == 0
    assert span.cost_usd is not None and span.cost_usd > 0

    assert tools[0].tool_name == "shell"
    assert tools[0].parent_span_id is None  # flat call_id model, no nesting

    assert parsed.tool_call_count == 1
    assert parsed.total_input_tokens == 800
    assert parsed.total_output_tokens == 350


def test_parse_returns_none_when_no_token_turns(tmp_path):
    # A session that opened but never got a model response -> nothing to ingest.
    path = _write_rollout(tmp_path, "sess-empty", [
        _session_meta_line("sess-empty"),
        _turn_context_line("gpt-5.5"),
    ])
    assert parse_codex_rollout(path) is None


def test_parse_sums_per_turn_deltas(tmp_path):
    # Two token_count events (deltas) -> two LLM spans; session total is the sum.
    path = _write_rollout(tmp_path, "sess-multi", [
        _session_meta_line("sess-multi"),
        _turn_context_line("gpt-5.5"),
        _token_count_line(input_tokens=500, cached_input_tokens=0,
                          output_tokens=100),
        _token_count_line(input_tokens=700, cached_input_tokens=100,
                          output_tokens=200,
                          ts="2026-07-01T10:00:10Z"),
    ])
    parsed = parse_codex_rollout(path)
    assert parsed is not None
    llm = [s for s in parsed.spans if s.name == "gen_ai.llm.call"]
    assert len(llm) == 2
    # 500 + (700-100) non-cached input
    assert parsed.total_input_tokens == 500 + 600
    assert parsed.total_output_tokens == 100 + 200
    assert parsed.total_cache_tokens == 100


def test_model_tracks_latest_turn_context(tmp_path):
    path = _write_rollout(tmp_path, "sess-switch", [
        _session_meta_line("sess-switch"),
        _turn_context_line("gpt-5.4-mini"),
        _token_count_line(input_tokens=100, cached_input_tokens=0,
                          output_tokens=10),
        _turn_context_line("gpt-5.5", ts="2026-07-01T10:00:08Z"),
        _token_count_line(input_tokens=200, cached_input_tokens=0,
                          output_tokens=20, ts="2026-07-01T10:00:10Z"),
    ])
    parsed = parse_codex_rollout(path)
    assert parsed is not None
    llm = sorted((s for s in parsed.spans if s.name == "gen_ai.llm.call"),
                 key=lambda s: s.start_time)
    assert llm[0].model == "gpt-5.4-mini"
    assert llm[1].model == "gpt-5.5"


# --- iter tests --------------------------------------------------------------

def test_iter_walks_root(tmp_path):
    _write_rollout(tmp_path, "sess-1", [
        _session_meta_line("sess-1"),
        _turn_context_line("gpt-5.5"),
        _token_count_line(100, 0, 10),
    ])
    _write_rollout(tmp_path, "sess-2", [
        _session_meta_line("sess-2"),
        _turn_context_line("gpt-5.5"),
        _token_count_line(200, 0, 20),
    ], subdir="2026/07/02")
    sessions = list(iter_codex_sessions(root=tmp_path))
    assert {s.session_id for s in sessions} == {"sess-1", "sess-2"}


def test_iter_empty_when_root_missing(tmp_path):
    assert list(iter_codex_sessions(root=tmp_path / "nope")) == []


# --- ingest tests ------------------------------------------------------------

def test_ingest_writes_spans_and_session(tmp_path):
    _write_rollout(tmp_path, "sess-w", [
        _session_meta_line("sess-w"),
        _turn_context_line("gpt-5.5"),
        _function_call_line("shell", "call_x"),
        _token_count_line(1000, 200, 300),
    ])
    db = InMemoryBackend()
    try:
        result = ingest_codex(db, root=tmp_path)
        assert result["sessions_seen"] == 1
        assert result["sessions_written"] == 1
        assert result["spans_written"] == 2  # 1 llm + 1 tool
        sess = db.get_session("sess-w")
        assert sess is not None
        assert sess.agent_id == "codex_exec"
        assert sess.tool_call_count == 1
        assert sess.input_tokens == 800
    finally:
        db.close()


def test_ingest_is_idempotent(tmp_path):
    _write_rollout(tmp_path, "sess-idem", [
        _session_meta_line("sess-idem"),
        _turn_context_line("gpt-5.5"),
        _token_count_line(500, 0, 100),
    ])
    db = InMemoryBackend()
    try:
        r1 = ingest_codex(db, root=tmp_path)
        assert r1["spans_written"] == 1
        r2 = ingest_codex(db, root=tmp_path)
        # Second run inserts nothing new (deterministic span ids).
        assert r2["spans_written"] == 0
        assert r2["spans_skipped"] == 1
    finally:
        db.close()


def test_ingest_is_idempotent_on_duckdb_bulk_path(tmp_path):
    """The DuckDB path exercises the bulk `_existing_span_ids` idempotency check
    (#433) — one `WHERE span_id IN (...)` per session instead of a SELECT per
    span. A re-run must still skip every already-present span."""
    from tokenjam.core.config import StorageConfig
    from tokenjam.core.db import DuckDBBackend

    _write_rollout(tmp_path, "sess-idem-db", [
        _session_meta_line("sess-idem-db"),
        _turn_context_line("gpt-5.5"),
        _function_call_line("shell", "call_a"),
        _token_count_line(500, 0, 100),
        _token_count_line(300, 0, 50),
    ])
    db = DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))
    try:
        r1 = ingest_codex(db, root=tmp_path)
        assert r1["spans_written"] == 3  # 1 tool + 2 llm turns
        assert r1["spans_skipped"] == 0
        r2 = ingest_codex(db, root=tmp_path)
        assert r2["spans_written"] == 0
        assert r2["spans_skipped"] == 3
    finally:
        db.close()


def test_ingest_propagates_config_plan_tier(tmp_path):
    # Config declares an OpenAI plan -> sessions get that plan_tier, not
    # "unknown". Mirrors the Claude Code backfill #176 behavior.
    from tokenjam.core.config import ProviderBudget, TjConfig

    _write_rollout(tmp_path, "sess-plan", [
        _session_meta_line("sess-plan"),
        _turn_context_line("gpt-5.5"),
        _token_count_line(500, 0, 100),
    ])
    cfg = TjConfig(version="1")
    cfg.budgets["openai"] = ProviderBudget(plan="plus")
    db = InMemoryBackend()
    try:
        ingest_codex(db, root=tmp_path, config=cfg)
        assert db.get_session("sess-plan").plan_tier == "plus"
    finally:
        db.close()


def test_ingest_plan_tier_unknown_without_config(tmp_path):
    _write_rollout(tmp_path, "sess-noconfig", [
        _session_meta_line("sess-noconfig"),
        _turn_context_line("gpt-5.5"),
        _token_count_line(500, 0, 100),
    ])
    db = InMemoryBackend()
    try:
        ingest_codex(db, root=tmp_path)  # config=None
        assert db.get_session("sess-noconfig").plan_tier == "unknown"
    finally:
        db.close()


def test_ingest_since_filters_old_sessions(tmp_path):
    from tokenjam.utils.time_parse import parse_since

    _write_rollout(tmp_path, "sess-old", [
        _session_meta_line("sess-old", ts="2020-01-01T00:00:00Z"),
        _turn_context_line("gpt-5.5", ts="2020-01-01T00:00:01Z"),
        _token_count_line(500, 0, 100, ts="2020-01-01T00:00:02Z"),
    ])
    db = InMemoryBackend()
    try:
        result = ingest_codex(db, root=tmp_path, since=parse_since("7d"))
        # The old session's end_time is well before the 7d cutoff.
        assert result["sessions_seen"] == 0
        assert db.get_session("sess-old") is None
    finally:
        db.close()
