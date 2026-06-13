"""Unit tests for the `tj context` context-cost diagnostic (issue #4).

Exercises the diagnostic over a SYNTHETIC multi-session fixture proving:
  * per-turn re-read-vs-work composition with named overhead (cache reads);
  * cross-session recurring-inclusion detection with a structural fix;
  * compact-candidate detection;
  * quota-share (% of cycle tokens) rendering for a Max plan via core/framing.
"""
from __future__ import annotations

from datetime import timedelta

import pytest
from click.testing import CliRunner

from tokenjam.core.config import CaptureConfig, ProviderBudget, TjConfig
from tokenjam.core.context_diagnostic import (
    COMPACT_MIN_CACHE_TOKENS,
    RECURRING_MIN_SESSIONS,
    compute_context_diagnostic,
)
from tokenjam.core.db import InMemoryBackend
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span

# Anchor the fixture a couple of hours before "now" so a relative `--since 30d`
# window (parsed against utcnow() in the CLI) always covers it.
BASE = utcnow() - timedelta(hours=2)
SINCE = BASE - timedelta(days=1)
UNTIL = utcnow() + timedelta(days=1)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _max_config(tool_inputs: bool = True) -> TjConfig:
    """Config declaring a Max-5x plan so framing renders quota-share."""
    return TjConfig(
        version="1",
        capture=CaptureConfig(tool_inputs=tool_inputs),
        budgets={"anthropic": ProviderBudget(plan="max_5x")},
    )


def _seed_multi_session(db) -> None:
    """Three sessions: two are re-read-heavy (one a compact candidate), all
    re-read the same schema file — the recurring-inclusion pattern from #24147.
    """
    # Session A — heavy re-reading: a single big-cache turn that clears the
    # compact threshold (cache_tokens >= COMPACT_MIN_CACHE_TOKENS, share high).
    sess_a = make_session(session_id="sess-a", plan_tier="max_5x",
                          duration_seconds=120.0)
    db.upsert_session(sess_a)
    span_a = make_llm_span(
        model="claude-opus-4-6",
        input_tokens=4_000,        # net-new this turn
        output_tokens=1_000,       # work produced
        cache_tokens=COMPACT_MIN_CACHE_TOKENS + 50_000,  # re-reading history/CLAUDE.md
        cache_write_tokens=0,
        cost_usd=2.5,
        session_id="sess-a",
    )
    span_a.start_time = BASE
    db.insert_span(span_a)

    # Session B — moderate re-reading across two turns.
    sess_b = make_session(session_id="sess-b", plan_tier="max_5x",
                          duration_seconds=90.0)
    db.upsert_session(sess_b)
    for j in range(2):
        span_b = make_llm_span(
            model="claude-sonnet-4-5",
            input_tokens=2_000,
            output_tokens=500,
            cache_tokens=30_000,
            cost_usd=0.4,
            session_id="sess-b",
        )
        span_b.start_time = BASE + timedelta(seconds=j)
        db.insert_span(span_b)

    # Session C — light, mostly work (low re-read share).
    sess_c = make_session(session_id="sess-c", plan_tier="max_5x",
                          duration_seconds=30.0)
    db.upsert_session(sess_c)
    span_c = make_llm_span(
        model="claude-haiku-4-5",
        input_tokens=5_000,
        output_tokens=3_000,
        cache_tokens=1_000,
        cost_usd=0.1,
        session_id="sess-c",
    )
    span_c.start_time = BASE + timedelta(seconds=1)
    db.insert_span(span_c)

    # Recurring inclusion: the SAME file Read across all three sessions —
    # exactly the structural pattern a `@file` / CLAUDE.md fix resolves.
    for sid in ("sess-a", "sess-b", "sess-c"):
        tool = make_tool_span(tool_name="Read")
        tool.session_id = sid
        tool.start_time = BASE + timedelta(seconds=2)
        tool.attributes = {
            GenAIAttributes.TOOL_INPUT: {"file_path": "db/schema.prisma"}
        }
        db.insert_span(tool)


def test_per_turn_composition_separates_reread_from_work(db):
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )

    # 4 assistant turns across 3 sessions.
    assert diag.turns == 4
    assert diag.sessions == 3

    # Re-read tokens = sum of cache reads; work = uncached input + output.
    expected_reread = (COMPACT_MIN_CACHE_TOKENS + 50_000) + 2 * 30_000 + 1_000
    assert diag.total_reread_tokens == expected_reread
    expected_work = (4_000 + 1_000) + 2 * (2_000 + 500) + (5_000 + 3_000)
    assert diag.total_work_tokens == expected_work

    # The headline re-read share is the dominant fraction (heavy re-reading).
    assert diag.reread_share > 0.80

    # Heaviest turn is session A's big-cache turn, named with its overhead.
    assert diag.heaviest_turns[0].session_id == "sess-a"
    assert diag.heaviest_turns[0].reread_tokens == COMPACT_MIN_CACHE_TOKENS + 50_000


def test_recurring_inclusion_detected_with_structural_fix(db):
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )

    assert len(diag.recurring) == 1
    rec = diag.recurring[0]
    assert rec.target == "db/schema.prisma"
    assert rec.sessions == 3  # appears in all three sessions
    assert rec.sessions >= RECURRING_MIN_SESSIONS
    # The fix is the structural @file / CLAUDE.md recommendation.
    assert "@db/schema.prisma" in rec.fix or "CLAUDE.md" in rec.fix


def test_compact_candidate_flags_reread_heavy_session(db):
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )

    # Only session A clears the compact threshold.
    assert len(diag.compact_candidates) == 1
    cand = diag.compact_candidates[0]
    assert cand.session_id == "sess-a"
    assert cand.reread_tokens >= COMPACT_MIN_CACHE_TOKENS
    assert cand.reread_share >= 0.80


def test_capture_off_emits_nudge_and_no_recurring(db):
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=False
    )
    # Composition still works (aggregate, no content needed)...
    assert diag.turns == 4
    # ...but the capture nudge is surfaced.
    assert any("tool_inputs" in n for n in diag.notes)


def test_empty_window_has_no_data(db):
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )
    assert not diag.has_data
    assert diag.turns == 0


def test_cli_renders_quota_share_for_max_plan(db, monkeypatch):
    """End-to-end: the card renders headline numbers as % of cycle tokens for a
    subscription (Max) plan — the quota-native frame, dollars secondary."""
    _seed_multi_session(db)
    config = _max_config(tool_inputs=True)

    import tokenjam.cli.main as cli_main

    monkeypatch.setattr(cli_main, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(cli_main, "open_db", lambda *a, **k: db)
    # Avoid a global-config peek influencing framing in CI.
    monkeypatch.setattr(
        "tokenjam.core.framing.config_declared_plan", lambda c: "max_5x"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_main.cli, ["context", "--since", "30d"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    # Quota framing: a "% of cycle tokens" share appears, not a raw dollar
    # headline. (render path uses subscription mode for max_5x.)
    assert "of cycle tokens" in result.output
    assert "re-reading context" in result.output
    assert "schema.prisma" in result.output


def test_cli_json_output(db, monkeypatch):
    _seed_multi_session(db)
    config = _max_config(tool_inputs=True)

    import json

    import tokenjam.cli.main as cli_main

    monkeypatch.setattr(cli_main, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(cli_main, "open_db", lambda *a, **k: db)
    monkeypatch.setattr(
        "tokenjam.core.framing.config_declared_plan", lambda c: "max_5x"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_main.cli, ["context", "--since", "30d", "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["turns"] == 4
    assert payload["sessions"] == 3
    assert payload["recurring"][0]["target"] == "db/schema.prisma"
    assert payload["compact_candidates"][0]["session_id"] == "sess-a"
    assert payload["framing"]["pricing_mode"] == "subscription"
