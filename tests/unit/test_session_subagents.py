"""Unit tests for the session-detail per-subagent breakdown helper."""
from __future__ import annotations

from tokenjam.api.routes.sessions import _session_subagents
from tokenjam.core.db import InMemoryBackend
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span


def test_session_subagents_breakdown_with_flags():
    db = InMemoryBackend()
    try:
        now = utcnow()
        # Main thread — excluded from the subagent breakdown.
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", input_tokens=1000, output_tokens=1000,
            cost_usd=1.0, session_id="s1", sub_agent_id=None, start_time=now,
        ))
        # Over-provisioned subagent: huge context, tiny output.
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", input_tokens=80_000, output_tokens=100,
            cost_usd=0.60, session_id="s1", sub_agent_id="A", start_time=now,
        ))
        # Clean subagent below the noise floor.
        db.insert_span(make_llm_span(
            model="claude-haiku-4-5", input_tokens=2000, output_tokens=3000,
            cost_usd=0.02, session_id="s1", sub_agent_id="B", start_time=now,
        ))

        out = _session_subagents(db, "s1")
        assert out["total"] == 2            # A + B; main excluded
        assert out["flagged"] == 1
        assert abs(out["cost_usd"] - 0.62) < 1e-6
        # Ordered by cost desc; A (the expensive one) first.
        assert out["rows"][0]["sub_agent_id"] == "A"
        by_id = {r["sub_agent_id"]: r for r in out["rows"]}
        assert "over_provisioned" in by_id["A"]["flags"]
        assert by_id["B"]["flags"] == []
    finally:
        db.close()


def test_session_subagents_empty_when_no_subagents():
    db = InMemoryBackend()
    try:
        db.insert_span(make_llm_span(
            model="claude-opus-4-8", input_tokens=1000, output_tokens=200,
            cost_usd=1.0, session_id="s1", sub_agent_id=None, start_time=utcnow(),
        ))
        out = _session_subagents(db, "s1")
        assert out == {"rows": [], "total": 0, "cost_usd": 0.0, "tokens": 0, "flagged": 0}
    finally:
        db.close()
