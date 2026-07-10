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
    INCLUSION_FILE_READ,
    INCLUSION_PROMPT,
    INCLUSION_SEARCH,
    INCLUSION_TOOL_OUTPUT,
    LARGE_OUTPUT_MIN_CHARS,
    RECURRING_MIN_OCCURRENCES,
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

    # Session B — moderate re-reading across two turns, each paying a cache-MISS
    # (cache-creation) premium — the named overhead source #11 breaks out.
    sess_b = make_session(session_id="sess-b", plan_tier="max_5x",
                          duration_seconds=90.0)
    db.upsert_session(sess_b)
    for j in range(2):
        span_b = make_llm_span(
            model="claude-sonnet-4-5",
            input_tokens=2_000,
            output_tokens=500,
            cache_tokens=30_000,
            cache_write_tokens=10_000,  # cache-MISS: written to cache (#11)
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

    # Recurring inclusion (file): the SAME file Read across all three sessions —
    # exactly the structural pattern a `@file` / CLAUDE.md fix resolves.
    for sid in ("sess-a", "sess-b", "sess-c"):
        tool = make_tool_span(tool_name="Read")
        tool.session_id = sid
        tool.start_time = BASE + timedelta(seconds=2)
        tool.attributes = {
            GenAIAttributes.TOOL_INPUT: {"file_path": "db/schema.prisma"}
        }
        db.insert_span(tool)

    # Recurring inclusion (search): the SAME Grep query re-run across all three
    # sessions — re-pastes its result every time → pin / capture once.
    for sid in ("sess-a", "sess-b", "sess-c"):
        grep = make_tool_span(tool_name="Grep")
        grep.session_id = sid
        grep.start_time = BASE + timedelta(seconds=3)
        grep.attributes = {
            GenAIAttributes.TOOL_INPUT: {"pattern": "TODO\\(perf\\)"}
        }
        db.insert_span(grep)

    # Recurring inclusion (prompt): the SAME user prompt re-sent across turns —
    # here re-pasted on three LLM turns (within and across sessions).
    repeated_prompt = "Always follow the repo conventions in CLAUDE.md exactly."
    for j, sid in enumerate(("sess-a", "sess-b", "sess-c")):
        pspan = make_llm_span(
            model="claude-sonnet-4-5",
            input_tokens=100,
            output_tokens=50,
            cache_tokens=0,
            cost_usd=0.01,
            session_id=sid,
            extra_attributes={GenAIAttributes.PROMPT_CONTENT: repeated_prompt},
        )
        pspan.start_time = BASE + timedelta(seconds=4 + j)
        db.insert_span(pspan)

    # Recurring inclusion (large tool output): the SAME big output re-pasted on
    # three tool turns (the live-ingest-only `gen_ai.tool.output` path).
    big_output = "X" * (LARGE_OUTPUT_MIN_CHARS + 500)
    for j, sid in enumerate(("sess-a", "sess-b", "sess-c")):
        ospan = make_tool_span(tool_name="Bash")
        ospan.session_id = sid
        ospan.start_time = BASE + timedelta(seconds=7 + j)
        ospan.attributes = {GenAIAttributes.TOOL_OUTPUT: big_output}
        db.insert_span(ospan)


def test_per_turn_composition_separates_reread_from_work(db):
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )

    # 4 work turns + 3 prompt-carrying turns (one per session) = 7 LLM turns.
    assert diag.turns == 7
    assert diag.sessions == 3

    # Re-read tokens = sum of cache reads (prompt turns carry zero cache).
    expected_reread = (COMPACT_MIN_CACHE_TOKENS + 50_000) + 2 * 30_000 + 1_000
    assert diag.total_reread_tokens == expected_reread
    # Work = uncached input + output, including the small prompt turns.
    expected_work = (
        (4_000 + 1_000) + 2 * (2_000 + 500) + (5_000 + 3_000)
        + 3 * (100 + 50)
    )
    assert diag.total_work_tokens == expected_work

    # The headline re-read share is the dominant fraction (heavy re-reading).
    assert diag.reread_share > 0.80

    # Heaviest turn is session A's big-cache turn, named with its overhead.
    assert diag.heaviest_turns[0].session_id == "sess-a"
    assert diag.heaviest_turns[0].reread_tokens == COMPACT_MIN_CACHE_TOKENS + 50_000


def test_quota_weighted_reread_share_discounts_cache_reads(db):
    """#119: the raw `reread_share` mixes "tokens" and "quota" framings — cache
    reads are billed at a fraction of a base token, so a raw share overstates
    re-reading's actual quota cost. `quota_weighted_reread_share` applies the
    documented discount (cache reads x0.1, output x5) and should read
    meaningfully lower than the raw share for this cache-read-heavy fixture.
    """
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )

    # Raw share is dominated by cache reads (see
    # test_per_turn_composition_separates_reread_from_work).
    assert diag.reread_share > 0.85

    # Manually replicate the weighting to pin the exact expected value rather
    # than just asserting a direction, catching any future formula drift.
    reread = diag.total_reread_tokens
    new_input = diag.total_new_input_tokens
    output = diag.total_output_tokens
    cache_write = diag.total_cache_write_tokens
    expected_total = reread * 0.1 + new_input + output * 5.0 + cache_write
    expected_share = (reread * 0.1) / expected_total

    assert diag.total_quota_weighted_tokens == pytest.approx(expected_total)
    assert diag.quota_weighted_reread_share == pytest.approx(expected_share)

    # The discount moves the headline substantially — this is the bug: a raw
    # share reads as ~90%+ while the quota-weighted share is well under half.
    assert diag.quota_weighted_reread_share < 0.5
    assert diag.quota_weighted_reread_share < diag.reread_share - 0.3

    # Work's quota-weighted share is NOT the raw work share either (output is
    # weighted 5x heavier than a base input token).
    assert diag.quota_weighted_work_share != pytest.approx(
        diag.total_work_tokens / diag.total_tokens
    )


def test_quota_weighted_fields_in_json_payload(db):
    """The quota-weighted headline fields round-trip into the `--json` payload."""
    from tokenjam.core.context_diagnostic import diagnostic_to_dict

    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )
    payload = diagnostic_to_dict(diag)

    assert payload["quota_weighted_reread_share"] == round(
        diag.quota_weighted_reread_share, 4
    )
    assert payload["quota_weighted_work_share"] == round(
        diag.quota_weighted_work_share, 4
    )
    assert payload["total_quota_weighted_tokens"] == round(
        diag.total_quota_weighted_tokens, 2
    )
    # Sanity: the quota-weighted share is meaningfully below the raw share.
    assert payload["quota_weighted_reread_share"] < payload["reread_share"]


def test_quota_weight_constants_match_anthropic_pricing_table():
    """#119: `CACHE_READ_QUOTA_WEIGHT` / `OUTPUT_QUOTA_WEIGHT` are asserted
    constants, not a live pricing lookup — so a future Anthropic pricing change
    in tokenjam/pricing/models.toml must fail this test loudly instead of
    silently drifting the quickstart headline away from what it claims to be
    (the whole credibility property #119 exists to fix)."""
    from tokenjam.core.context_diagnostic import (
        CACHE_READ_QUOTA_WEIGHT,
        OUTPUT_QUOTA_WEIGHT,
    )
    from tokenjam.core.pricing import load_pricing_table

    anthropic_rates = load_pricing_table().get("anthropic", {})
    assert anthropic_rates, "expected at least one anthropic pricing row"

    checked = 0
    for model, rates in anthropic_rates.items():
        if rates.input_per_mtok <= 0:
            continue  # skip the zero-priced internal test model (tj-ping-test)
        checked += 1
        cache_read_ratio = rates.cache_read_per_mtok / rates.input_per_mtok
        output_ratio = rates.output_per_mtok / rates.input_per_mtok
        assert cache_read_ratio == pytest.approx(CACHE_READ_QUOTA_WEIGHT), (
            f"{model}: cache_read/input ratio {cache_read_ratio} != "
            f"CACHE_READ_QUOTA_WEIGHT {CACHE_READ_QUOTA_WEIGHT} — pricing drifted, "
            f"update the constant (and re-verify the quickstart headline)."
        )
        assert output_ratio == pytest.approx(OUTPUT_QUOTA_WEIGHT), (
            f"{model}: output/input ratio {output_ratio} != "
            f"OUTPUT_QUOTA_WEIGHT {OUTPUT_QUOTA_WEIGHT} — pricing drifted, "
            f"update the constant (and re-verify the quickstart headline)."
        )
    assert checked > 0


def test_cache_miss_broken_out_as_named_overhead(db):
    """#11: cache-creation tokens are surfaced as their own named overhead
    category (prompt-cache MISS), distinct from re-read and net-new work."""
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )

    # Two session-B turns each pay a 10K cache-creation (miss) premium.
    expected_cache_miss = 2 * 10_000
    assert diag.total_cache_miss_tokens == expected_cache_miss
    assert diag.total_cache_miss_tokens == diag.total_cache_write_tokens
    # It is a real, non-zero share of the window's tokens.
    assert diag.cache_miss_share > 0.0
    # And it is NOT double-counted into either re-read or net-new work.
    assert diag.total_cache_miss_tokens not in (
        diag.total_reread_tokens, diag.total_work_tokens
    )


def test_mcp_injection_half_is_parked_with_a_precise_note(db):
    """#11: the MCP schema-injection half is parked (not fabricated) — the
    diagnostic carries a precise note on what data would be needed."""
    from tokenjam.core.context_diagnostic import MCP_INJECTION_PARK_NOTE

    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )
    assert MCP_INJECTION_PARK_NOTE in diag.notes
    assert "MCP" in MCP_INJECTION_PARK_NOTE
    # No invented attribution number — it names the missing data path instead.
    assert "per-request tool-schema token delta" in MCP_INJECTION_PARK_NOTE


def test_cache_miss_and_park_note_in_json_payload(db):
    """The named cache-miss overhead + parked-MCP note round-trip into JSON."""
    from tokenjam.core.context_diagnostic import (
        MCP_INJECTION_PARK_NOTE,
        diagnostic_to_dict,
    )

    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )
    payload = diagnostic_to_dict(diag)
    assert payload["total_cache_miss_tokens"] == 2 * 10_000
    assert payload["cache_miss_share"] > 0.0
    assert payload["mcp_injection_parked"] == MCP_INJECTION_PARK_NOTE
    # Per-turn rows also carry the cache-miss attribution.
    assert all("cache_miss_tokens" in t for t in payload["heaviest_turns"])


def test_recurring_file_read_detected_with_structural_fix(db):
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )

    by_type = {r.inclusion_type: r for r in diag.recurring}
    rec = by_type[INCLUSION_FILE_READ]
    assert rec.target == "db/schema.prisma"
    assert rec.sessions == 3  # appears in all three sessions
    assert rec.sessions >= RECURRING_MIN_SESSIONS
    # The fix is the structural @file / CLAUDE.md recommendation.
    assert "@db/schema.prisma" in rec.fix or "CLAUDE.md" in rec.fix


def test_recurring_search_detected_with_pin_fix(db):
    """A repeated Grep query is flagged with a pin-the-result structural fix."""
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )

    by_type = {r.inclusion_type: r for r in diag.recurring}
    rec = by_type[INCLUSION_SEARCH]
    assert rec.tool_name == "Grep"
    assert rec.target == "TODO\\(perf\\)"
    assert rec.sessions == 3
    assert rec.occurrences == 3
    assert "Pin" in rec.fix or "capture it once" in rec.fix


def test_recurring_prompt_detected_with_slash_command_fix(db):
    """An identical user prompt re-sent across turns is flagged with a
    save-as-slash-command / CLAUDE.md structural fix. Gated on `prompts`."""
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, prompts_captured=True
    )

    by_type = {r.inclusion_type: r for r in diag.recurring}
    rec = by_type[INCLUSION_PROMPT]
    assert rec.occurrences == 3
    assert rec.occurrences >= RECURRING_MIN_OCCURRENCES
    assert "CLAUDE.md" in rec.target  # excerpt of the repeated prompt text
    assert "slash-command" in rec.fix


def test_recurring_large_output_detected_with_reference_fix(db):
    """A large identical tool output re-pasted across turns is flagged with a
    reference-the-artifact structural fix. Gated on `tool_outputs`."""
    _seed_multi_session(db)
    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_outputs_captured=True
    )

    by_type = {r.inclusion_type: r for r in diag.recurring}
    rec = by_type[INCLUSION_TOOL_OUTPUT]
    assert rec.tool_name == "Bash"
    assert rec.occurrences == 3
    assert "reference the artifact" in rec.fix


def test_small_repeated_output_below_size_gate_not_flagged(db):
    """A repeated but SMALL tool output isn't worth flagging — it costs almost
    no quota to re-paste."""
    for sid in ("sess-a", "sess-b", "sess-c"):
        ospan = make_tool_span(tool_name="Bash")
        ospan.session_id = sid
        ospan.start_time = BASE
        ospan.attributes = {GenAIAttributes.TOOL_OUTPUT: "ok"}
        db.insert_span(ospan)

    diag = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_outputs_captured=True
    )
    assert not any(
        r.inclusion_type == INCLUSION_TOOL_OUTPUT for r in diag.recurring
    )


def test_each_capture_flag_gates_its_own_inclusion_kind(db):
    """tool_inputs → file+search only; prompts → prompt only; tool_outputs →
    output only. Flags are independent."""
    _seed_multi_session(db)

    inputs_only = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_inputs_captured=True
    )
    assert {r.inclusion_type for r in inputs_only.recurring} == {
        INCLUSION_FILE_READ, INCLUSION_SEARCH
    }

    prompts_only = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, prompts_captured=True
    )
    assert {r.inclusion_type for r in prompts_only.recurring} == {
        INCLUSION_PROMPT
    }

    outputs_only = compute_context_diagnostic(
        db.conn, SINCE, UNTIL, tool_outputs_captured=True
    )
    assert {r.inclusion_type for r in outputs_only.recurring} == {
        INCLUSION_TOOL_OUTPUT
    }

    all_on = compute_context_diagnostic(
        db.conn, SINCE, UNTIL,
        tool_inputs_captured=True, prompts_captured=True,
        tool_outputs_captured=True,
    )
    assert {r.inclusion_type for r in all_on.recurring} == {
        INCLUSION_FILE_READ, INCLUSION_SEARCH, INCLUSION_PROMPT,
        INCLUSION_TOOL_OUTPUT,
    }


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
        db.conn, SINCE, UNTIL,
        tool_inputs_captured=False, prompts_captured=False,
        tool_outputs_captured=False,
    )
    # Composition still works (aggregate, no content needed)...
    assert diag.turns == 7
    # ...no recurring inclusions are detected with every capture flag off...
    assert diag.recurring == []
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
    # #11: the cache-MISS named overhead line renders (session B pays a premium).
    assert "Cache-miss:" in result.output


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
    assert payload["turns"] == 7
    assert payload["sessions"] == 3
    targets = {r["target"] for r in payload["recurring"]}
    assert "db/schema.prisma" in targets
    # Each recurring row is tagged with its inclusion type.
    assert all("inclusion_type" in r for r in payload["recurring"])
    assert payload["compact_candidates"][0]["session_id"] == "sess-a"
    assert payload["framing"]["pricing_mode"] == "subscription"


# --- Subagent accounting (#60) ----------------------------------------------
#
# A Claude Code session that delegates to Task subagents under-counts the
# model-weighted quota when the subagent turns aren't in the data. These fixtures
# model both halves of the ticket's "Done when": (a) captured subagent turns are
# included in the weighted quota, and (b) a delegating session whose subagent
# turns are MISSING is explicitly flagged partial (a lower bound), so the number
# is never silently low.


def _seed_delegating_session(db, sid: str, *, capture_subagents: bool) -> None:
    """A session that delegates via a `Task` tool span in the parent.

    Always records the parent's own LLM turn plus the `Task` delegation span.
    When ``capture_subagents`` is True it also records the subagent's own LLM
    turns (as backfill would, folded in under a `sub_agent_id`); when False the
    subagent transcript is absent — the exact #60 A/B shape where tj saw the
    parent turns but not the delegated work.
    """
    db.upsert_session(make_session(session_id=sid, plan_tier="max_5x"))
    # Parent-thread LLM turn (main model, no sub_agent_id).
    parent = make_llm_span(
        model="claude-opus-4-6", input_tokens=2_000, output_tokens=400,
        cost_usd=0.3, session_id=sid, sub_agent_id=None, start_time=BASE,
    )
    db.insert_span(parent)
    # The delegation itself: a `Task` tool call recorded in the parent.
    db.insert_span(make_tool_span(
        tool_name="Task", session_id=sid, start_time=BASE + timedelta(seconds=1),
    ))
    if capture_subagents:
        # Two subagent turns backfill folded in from subagents/agent-*.jsonl.
        for i, said in enumerate(("sub-1", "sub-2")):
            db.insert_span(make_llm_span(
                model="claude-haiku-4-5", input_tokens=30_000, output_tokens=1_500,
                cost_usd=0.25, session_id=sid, sub_agent_id=said,
                start_time=BASE + timedelta(seconds=2 + i),
            ))


def test_captured_subagent_turns_are_in_weighted_quota_and_not_flagged(db):
    """A delegating session whose subagent transcripts were captured: its
    subagent turns are counted in the weighted quota and it is NOT flagged."""
    _seed_delegating_session(db, "sess-deleg", capture_subagents=True)
    diag = compute_context_diagnostic(db.conn, SINCE, UNTIL)

    # Parent turn + two subagent turns all counted in the window.
    assert diag.turns == 3
    assert diag.subagent_turns == 2
    # The subagent output tokens (2 x 1500) are in the weighted total — the
    # under-count the ticket describes is gone when the data is present.
    assert diag.total_output_tokens == 400 + 1_500 + 1_500
    # Delegation was seen and fully accounted → no partial flag, no note.
    assert diag.delegating_sessions == 1
    assert diag.unaccounted_subagent_sessions == []
    assert diag.subagent_accounting_partial is False
    assert not any("LOWER BOUND" in n for n in diag.notes)


def test_delegating_session_without_subagent_turns_is_flagged_partial(db):
    """A delegating session whose subagent transcript is MISSING is flagged: the
    weighted quota is a lower bound, surfaced instead of silently under-counted."""
    _seed_delegating_session(db, "sess-blind", capture_subagents=False)
    diag = compute_context_diagnostic(db.conn, SINCE, UNTIL)

    # Only the parent turn is visible; no subagent turns were captured.
    assert diag.turns == 1
    assert diag.subagent_turns == 0
    # The delegation is detected via the parent's Task span and flagged partial.
    assert diag.delegating_sessions == 1
    assert diag.unaccounted_subagent_sessions == ["sess-blind"]
    assert diag.subagent_accounting_partial is True
    assert any("LOWER BOUND" in n for n in diag.notes)


def test_subagent_accounting_fields_in_json_payload(db):
    """The subagent-accounting flag round-trips through diagnostic_to_dict."""
    from tokenjam.core.context_diagnostic import diagnostic_to_dict

    _seed_delegating_session(db, "sess-blind", capture_subagents=False)
    payload = diagnostic_to_dict(compute_context_diagnostic(db.conn, SINCE, UNTIL))

    assert payload["subagent_turns"] == 0
    assert payload["delegating_sessions"] == 1
    assert payload["unaccounted_subagent_sessions"] == ["sess-blind"]
    assert payload["subagent_accounting_partial"] is True


def test_attach_tool_activity_does_not_truncate_on_null_start_time():
    """A turn with a missing start_time must be treated as earliest (consistent
    with the sort key), NOT truncate attribution: a later, genuinely-preceding
    turn still receives the tool span. A `break` on the null turn would drop
    every real turn after it and mis-attribute the tool to the null turn."""
    from tokenjam.core.context_diagnostic import (
        TurnComposition,
        _attach_tool_activity,
    )
    from tokenjam.utils.time_parse import utcnow

    t0 = utcnow()

    def _turn_comp(start_time):
        return TurnComposition(
            session_id="s", sub_agent_id=None, model="claude-opus-4-8",
            reread_tokens=0, new_input_tokens=100, output_tokens=50,
            cache_write_tokens=0, cost_usd=0.0, start_time=start_time,
        )

    null_turn = _turn_comp(None)
    real_turn = _turn_comp(t0)

    class _FakeConn:
        def execute(self, _sql, _params):
            # One Read tool span AFTER the real turn (and after any turn).
            self._rows = [("s", "Read", t0 + timedelta(seconds=5))]
            return self

        def fetchall(self):
            return self._rows

    _attach_tool_activity(
        _FakeConn(), t0 - timedelta(days=1), t0 + timedelta(days=1), None,
        [null_turn, real_turn],
    )

    # The tool span attributes to the real preceding turn, not the null turn.
    assert real_turn.tool_fanout == 1
    assert null_turn.tool_fanout == 0
