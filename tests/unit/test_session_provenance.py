"""Session provenance (`sessions.source`, migration 21) + task identity
(`task_statement_hash` / `dominant_model`).

`source` records what PRODUCED a session at ingest time — before this,
nothing did, and every reader re-derived "coding vs SDK" from `agent_id`
naming conventions via one of two independently-maintained, deliberately
divergent predicates (`core.agent_kind` vs
`core.alerts.is_interactive_coding_agent`; see `agent_kind`'s module
docstring). This does NOT merge them — both stay exactly as they were; these
tests confirm that explicitly, alongside the linkage the new column adds.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tokenjam.core.backfill import ingest_claude_code
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.models import NormalizedSpan

from tests.factories import make_llm_span


def _make_session_file(tmp_path: Path, session_id: str, cwd: str,
                        records: list[dict]) -> Path:
    project_dir = tmp_path / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def _assistant_record(uuid: str, model: str, input_tokens: int, output_tokens: int,
                       timestamp: str, session_id: str, cwd: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": timestamp,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "model": model,
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    }


def _user_record(session_id: str, cwd: str, text: str) -> dict:
    return {
        "type": "user", "sessionId": session_id, "cwd": cwd,
        "message": {"role": "user", "content": text},
    }


# ---------------------------------------------------------------------------
# Claude Code backfill: source is a literal fact, never a guess.
# ---------------------------------------------------------------------------

def test_claude_code_backfill_stamps_source(tmp_path):
    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-cc", cwd=proj,
        records=[
            _user_record("sess-cc", proj, "Fix the flaky test in ci.yml"),
            _assistant_record("m1", "claude-opus-4-7", 1000, 200,
                               "2026-04-01T10:00:00.000Z", "sess-cc", proj),
        ],
    )
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        row = db.conn.execute(
            "SELECT source FROM sessions WHERE session_id = 'sess-cc'"
        ).fetchone()
        assert row == ("claude-code",)
    finally:
        db.close()


def test_claude_code_backfill_stamps_task_hash_and_dominant_model(tmp_path):
    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-th", cwd=proj,
        records=[
            _user_record("sess-th", proj, "Fix the flaky test in ci.yml"),
            _assistant_record("m1", "claude-opus-4-7", 100, 50,
                               "2026-04-01T10:00:00.000Z", "sess-th", proj),
            _assistant_record("m2", "claude-opus-4-7", 5000, 2000,
                               "2026-04-01T10:00:01.000Z", "sess-th", proj),
            _assistant_record("m3", "claude-haiku-4-5", 10, 5,
                               "2026-04-01T10:00:02.000Z", "sess-th", proj),
        ],
    )
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        row = db.conn.execute(
            "SELECT task_statement_hash, dominant_model FROM sessions "
            "WHERE session_id = 'sess-th'"
        ).fetchone()
        task_hash, dominant_model = row
        assert task_hash is not None
        assert len(task_hash) == 32  # hex digest, truncated — see hash_task_statement
        # The bulk of the tokens (5050) ran on opus; haiku's 15 shouldn't win.
        assert dominant_model == "claude-opus-4-7"

        # Never the raw prompt.
        assert "flaky test" not in task_hash
        assert "ci.yml" not in task_hash
    finally:
        db.close()


def test_the_task_hash_is_reproducible_and_masks_variables(tmp_path):
    """Two sessions running the SAME templated task with a different id in
    the prompt must collide — the whole point of the hash."""
    from tokenjam.core.optimize.repeat_task import hash_task_statement

    a = hash_task_statement("Fix issue #4821 in the billing module")
    b = hash_task_statement("Fix issue #9013 in the billing module")
    assert a == b
    assert hash_task_statement(None) is None
    assert hash_task_statement("") is None


def test_a_subagent_only_file_never_supplies_the_task_hash(tmp_path):
    """A subagent transcript's "user" turns are the Task tool's dispatched
    instructions, not what the human actually typed — must not leak in as
    the session's task statement."""
    from tokenjam.core.backfill import parse_claude_code_session

    proj = "/Users/me/proj"
    sub_dir = tmp_path / proj.replace("/", "-") / "sess-sa" / "subagents"
    sub_dir.mkdir(parents=True)
    path = sub_dir / "agent-a1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"type": "user", "sessionId": "sess-sa", "cwd": proj, "isSidechain": True,
         "message": {"role": "user", "content": "dispatched sub-instruction"}},
        {**_assistant_record("m1", "claude-haiku-4-5", 100, 50,
                              "2026-04-01T10:00:00.000Z", "sess-sa", proj),
         "isSidechain": True, "agentId": "a1"},
    ]))
    parsed = parse_claude_code_session(path)
    assert parsed is not None
    assert parsed.first_user_prompt is None


# ---------------------------------------------------------------------------
# Codex backfill: also a literal fact.
# ---------------------------------------------------------------------------

def test_codex_backfill_stamps_source(tmp_path):
    from tokenjam.core.ingest_adapters.codex import (
        ParsedCodexSession,
        session_record_from_parsed,
    )

    parsed = ParsedCodexSession(
        session_id="sess-codex", agent_id="codex_exec",
        started_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 4, 1, 0, 5, tzinfo=timezone.utc),
        cwd="/Users/me/proj", spans=[],
        total_input_tokens=0, total_output_tokens=0, total_cache_tokens=0,
        total_cost_usd=0.0, tool_call_count=0,
    )
    record = session_record_from_parsed(parsed)
    assert record.source == "codex"


# ---------------------------------------------------------------------------
# Live ingest path: genuinely ambiguous, classified via `core.agent_kind`
# (the TIGHTER of the two existing predicates) rather than a new heuristic.
# ---------------------------------------------------------------------------

def _llm_span(agent_id: str, session_id: str) -> NormalizedSpan:
    return make_llm_span(
        agent_id=agent_id, session_id=session_id,
        model="claude-opus-4-7", input_tokens=10, output_tokens=5,
    )


@pytest.mark.parametrize("agent_id,expected_source", [
    ("claude-code", "claude-code"),
    ("claude-code-my-project", "claude-code"),
    ("codex_exec", "codex"),
    ("my-sdk-workflow", "sdk"),
    ("codex-cli-session", "sdk"),  # NOT codex under the tighter exact-match rule
])
def test_live_ingest_classifies_source_via_agent_kind(agent_id, expected_source):
    from tokenjam.core.config import TjConfig

    db = InMemoryBackend()
    pipeline = IngestPipeline(db=db, config=TjConfig(version="1"))
    try:
        span = _llm_span(agent_id, f"sess-{agent_id}")
        pipeline.process(span)
        row = db.conn.execute(
            "SELECT source FROM sessions WHERE session_id = $1", [span.session_id],
        ).fetchone()
        assert row == (expected_source,)
    finally:
        db.close()


def test_source_never_flips_once_resolved():
    """`source` is a fill-once field — a second span for the same session
    (even one this classifier would resolve differently) must not overwrite
    the value the first write already resolved. Mirrors plan_tier's existing
    fill-once discipline in the same upsert."""
    from tokenjam.core.config import TjConfig

    db = InMemoryBackend()
    pipeline = IngestPipeline(db=db, config=TjConfig(version="1"))
    try:
        span1 = _llm_span("claude-code", "sess-stable")
        pipeline.process(span1)
        span2 = _llm_span("claude-code", "sess-stable")
        pipeline.process(span2)
        row = db.conn.execute(
            "SELECT source FROM sessions WHERE session_id = 'sess-stable'"
        ).fetchone()
        assert row == ("claude-code",)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# The two existing predicates are UNCHANGED — this column does not merge
# them. `test_alerts.py::test_is_interactive_coding_agent_margin_cases` is
# the pinned regression; these are an explicit cross-check alongside it.
# ---------------------------------------------------------------------------

def test_the_new_column_does_not_change_either_existing_predicate():
    from tokenjam.core.agent_kind import classify_agent_kind
    from tokenjam.core.alerts import is_interactive_coding_agent

    # The documented disagreement case: a prefix match ("codex...") that is
    # NOT Codex's real, exact, hardcoded service name.
    agent_id = "codex-cli-session"
    assert is_interactive_coding_agent(agent_id) is True   # broader, prefix-based
    assert classify_agent_kind(agent_id).is_coding is False  # tighter, exact-match


def test_source_agrees_with_agent_kind_on_a_real_transcript(tmp_path):
    """The recorded column and `classify_agent_kind` (the predicate it's
    derived from) must actually agree — proven against a real parsed
    transcript, not asserted in isolation."""
    from tokenjam.core.agent_kind import classify_agent_kind

    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-agree", cwd=proj,
        records=[_assistant_record("m1", "claude-opus-4-7", 100, 50,
                                    "2026-04-01T10:00:00.000Z", "sess-agree", proj)],
    )
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        agent_id, source = db.conn.execute(
            "SELECT agent_id, source FROM sessions WHERE session_id = 'sess-agree'"
        ).fetchone()
        assert source == classify_agent_kind(agent_id).group
    finally:
        db.close()


# ---------------------------------------------------------------------------
# The existing backfill overlay (bulk path, non-`--reingest`) self-heals
# provenance/task-identity onto sessions ingested BEFORE this migration —
# no new repair path needed, reusing the same additive-overlay upsert
# discipline `source`/`task_statement_hash`/`dominant_model` were given
# above (COALESCE — fill once, never clobber).
# ---------------------------------------------------------------------------

def test_a_plain_backfill_rerun_fills_provenance_on_a_pre_migration_session(tmp_path):
    proj = "/Users/me/proj"
    _make_session_file(
        tmp_path, session_id="sess-heal", cwd=proj,
        records=[
            _user_record("sess-heal", proj, "Refactor the widget loader"),
            _assistant_record("m1", "claude-opus-4-7", 100, 50,
                               "2026-04-01T10:00:00.000Z", "sess-heal", proj),
        ],
    )
    db = InMemoryBackend()
    try:
        ingest_claude_code(db, root=tmp_path)
        # Simulate a session ingested before migration 21 landed.
        db.conn.execute(
            "UPDATE sessions SET source = NULL, task_statement_hash = NULL, "
            "dominant_model = NULL WHERE session_id = 'sess-heal'"
        )
        before = db.conn.execute(
            "SELECT input_tokens, output_tokens, total_cost_usd FROM sessions "
            "WHERE session_id = 'sess-heal'"
        ).fetchone()

        ingest_claude_code(db, root=tmp_path)  # plain re-run, no --reingest

        row = db.conn.execute(
            "SELECT source, task_statement_hash, dominant_model, "
            "input_tokens, output_tokens, total_cost_usd "
            "FROM sessions WHERE session_id = 'sess-heal'"
        ).fetchone()
        assert row[0] == "claude-code"
        assert row[1] is not None
        assert row[2] == "claude-opus-4-7"
        # Additive-only: the re-run added zero new spans, so totals must be
        # untouched (this is the SAME idempotency the subagent-type overlay
        # already guarantees for the spans it touches).
        assert row[3:] == before
    finally:
        db.close()
