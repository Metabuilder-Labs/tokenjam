"""A session's totals must never outlive the spans they were computed from.

The bulk backfill path defers span INSERTs into a cross-session buffer and
flushes it in batches. Nothing here has transaction control — every statement
auto-commits — so if the session-total delta is written before that flush, an
interruption in the gap leaves a row describing spans that never reached disk.
The row is then simply wrong until some later run completes: the end-of-run
reconciliation that would rewrite it from `SUM(spans)` never fired, because the
run that triggers it is the one that died. Every read in between — cost
totals, drift checks, the session view — sees spend that has no spans behind
it, and a re-run adds the same delta again before converging.

These tests pin the ordering that makes the failure benign instead: spans
first, delta second, so an interruption undercounts rather than overcounts and
the durable spans are what the next run reconciles against.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenjam.core import backfill as backfill_module
from tokenjam.core.backfill import ingest_claude_code
from tokenjam.core.config import StorageConfig
from tokenjam.core.db import DuckDBBackend


def _assistant_record(uuid: str, message_id: str, session_id: str,
                      cwd: str, input_tokens: int, output_tokens: int) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "sessionId": session_id,
        "cwd": cwd,
        "timestamp": "2026-04-01T10:00:00.000Z",
        "message": {
            "id": message_id,
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }


def _corpus(root: Path) -> None:
    cwd = "/Users/me/proj"
    project_dir = root / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "sess-int.jsonl").write_text(
        "\n".join(
            json.dumps(
                _assistant_record(f"u{i}", f"msg-{i}", "sess-int", cwd, 1000, 100)
            )
            for i in range(3)
        )
    )


def _totals(db, session_id: str) -> tuple[int, int]:
    """(session row input_tokens, SUM over that session's spans)."""
    session = db.get_session(session_id)
    row = db.conn.execute(
        "SELECT COALESCE(SUM(input_tokens), 0) FROM spans WHERE session_id = $1",
        [session_id],
    ).fetchone()
    return (session.input_tokens if session else 0), row[0]


def _duckdb(tmp_path: Path):
    return DuckDBBackend(StorageConfig(path=str(tmp_path / "t.duckdb")))


def test_interrupted_flush_leaves_no_totals_without_their_spans(
    tmp_path, monkeypatch,
):
    """Killed mid-flush, the row must not claim spans that never landed."""
    root = tmp_path / "projects"
    _corpus(root)
    db = _duckdb(tmp_path)

    def _die(*_args, **_kwargs):
        raise KeyboardInterrupt("process killed mid-flush")

    monkeypatch.setattr(backfill_module, "_flush_pending_spans", _die)
    try:
        with pytest.raises(KeyboardInterrupt):
            ingest_claude_code(db, root=root)

        session_tokens, span_tokens = _totals(db, "sess-int")
        assert span_tokens == 0
        # The delta rides with its spans, so nothing was committed for spans
        # that died in the buffer. A row may not exist at all — that is fine;
        # what must never happen is a row that OVERSTATES SUM(spans).
        assert session_tokens <= span_tokens
    finally:
        db.close()


def test_rerun_after_an_interrupted_flush_converges_on_sum_of_spans(
    tmp_path, monkeypatch,
):
    """A completed run always lands on `SUM(spans)`, whatever preceded it.

    The re-run re-adds the delta (its spans died in the buffer, so they are
    legitimately new), and the end-of-run recompute then rewrites the row from
    `SUM(spans)` — so the doubling does not survive a run that FINISHES. That
    recompute is the whole safety net, and it only ever fires on a completed
    run: it is exactly what the interrupted run above does not get. The
    ordering fix is what keeps the window between the two runs honest; this
    test pins the convergence the recompute is responsible for, so a later
    change to either mechanism cannot quietly drop it.
    """
    root = tmp_path / "projects"
    _corpus(root)
    db = _duckdb(tmp_path)

    def _die(*_args, **_kwargs):
        raise KeyboardInterrupt("process killed mid-flush")

    monkeypatch.setattr(backfill_module, "_flush_pending_spans", _die)
    try:
        with pytest.raises(KeyboardInterrupt):
            ingest_claude_code(db, root=root)

        monkeypatch.undo()
        result = ingest_claude_code(db, root=root)

        session_tokens, span_tokens = _totals(db, "sess-int")
        assert span_tokens == 3000, "all three calls should be ingested"
        assert session_tokens == span_tokens, (
            "session row must equal SUM(spans), not double it"
        )
        assert result.spans_ingested > 0
    finally:
        db.close()


def test_rerun_of_a_completed_backfill_is_still_a_no_op(tmp_path):
    """Idempotency is the property the fix must not trade away."""
    root = tmp_path / "projects"
    _corpus(root)
    db = _duckdb(tmp_path)
    try:
        first = ingest_claude_code(db, root=root)
        after_first, spans_first = _totals(db, "sess-int")

        second = ingest_claude_code(db, root=root)
        after_second, spans_second = _totals(db, "sess-int")

        assert first.spans_ingested > 0
        assert second.spans_ingested == 0
        assert (after_second, spans_second) == (after_first, spans_first)
        assert after_first == spans_first
    finally:
        db.close()
