"""Unit tests for the segment-level premium quota audit (`tj quota-audit`, issue #5).

The audit walks assistant turns per session in ``start_time`` order and reports
ONE honest figure (founder D1): the share of premium (Opus/Fable) quota that
went to Sonnet-shaped *work* — whole Sonnet-shaped sessions PLUS mechanical
stretches inside otherwise-hard sessions — on exact per-turn model attribution
(D2), as a labelled estimate with a wide bootstrap CI (D3).

These pin the load-bearing behaviour:
  * a mechanical stretch inside a hard session is flagged (the old whole-session
    audit structurally missed it);
  * a mixed-model session attributes tokens per actual span model, not MODE;
  * min-stretch + contiguity + delegation gate which stretches count;
  * the headline carries the confidence label + CI;
  * the honesty caveat / estimate basis stay Rule-14;
  * new fields round-trip; the CLI renders per-persona copy correctly.
"""
from __future__ import annotations

import re
from datetime import timedelta

import pytest
from click.testing import CliRunner

import tokenjam.core.optimize.analyzers.model_downgrade as md
from tokenjam.core.config import ProviderBudget, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize.analyzers.model_downgrade import audit_opus_quota
from tokenjam.core.optimize.types import (
    OPUS_QUOTA_AUDIT_CAVEAT,
    audit_from_dict,
    audit_to_dict,
)
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


def _max_config() -> TjConfig:
    """Config declaring a Max-5x plan so framing renders quota-share."""
    return TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="max_5x")})


def _api_config() -> TjConfig:
    """Config declaring an API plan so framing renders the dollar counterfactual."""
    return TjConfig(version="1", budgets={"anthropic": ProviderBudget(plan="api")})


def _new_session(db, session_id, *, plan_tier="max_5x") -> None:
    db.upsert_session(make_session(session_id=session_id, plan_tier=plan_tier,
                                   duration_seconds=600.0))


def _turn(db, session_id, seq, *, model="claude-opus-4-7", input_tokens=500,
          output_tokens=100, cache_tokens=0, cost_usd=1.0, tool_calls=0,
          delegate=False) -> None:
    """Append one assistant turn (an LLM span + optional tool/Task spans) to a
    session, at minute ``seq`` — so turns order deterministically by start_time
    and their tool spans attribute to the enclosing turn (nearest-preceding)."""
    span = make_llm_span(
        model=model, input_tokens=input_tokens, output_tokens=output_tokens,
        cache_tokens=cache_tokens, cost_usd=cost_usd, session_id=session_id,
    )
    span.start_time = BASE + timedelta(minutes=seq)
    db.insert_span(span)
    for j in range(tool_calls):
        tool = make_tool_span(tool_name="Read")
        tool.session_id = session_id
        tool.start_time = BASE + timedelta(minutes=seq, seconds=1 + j)
        db.insert_span(tool)
    if delegate:
        task = make_tool_span(tool_name="Task")
        task.session_id = session_id
        task.start_time = BASE + timedelta(minutes=seq, seconds=30)
        db.insert_span(task)


def _audit(db):
    return audit_opus_quota(db.conn, SINCE, UNTIL, agent_id=None, window_days=30.0)


# ── (a) the core fix: a mechanical stretch inside a hard session ─────────────

def test_mechanical_stretch_inside_hard_session_is_flagged(db):
    """A session that is 2 hard turns + a 2-turn cheap stretch: the stretch is
    flagged even though the WHOLE-session sums (17.9K input) blow past the
    session-level threshold the old audit used — exactly the case (a) miss."""
    _new_session(db, "mixed")
    _turn(db, "mixed", 0, input_tokens=8_000, output_tokens=1_000)   # hard
    _turn(db, "mixed", 1, input_tokens=500, output_tokens=100)        # cheap
    _turn(db, "mixed", 2, input_tokens=400, output_tokens=90)         # cheap
    _turn(db, "mixed", 3, input_tokens=9_000, output_tokens=1_200)    # hard

    audit = _audit(db)

    assert audit.opus_sessions == 1
    assert audit.candidate_sessions == 1
    assert audit.segment_count == 1
    # Quota-weighted numerator = the two cheap turns; denominator = all four.
    #   cheap: (500 + 100*5) + (400 + 90*5) = 1000 + 850 = 1850
    #   hard:  (8000 + 1000*5) + (9000 + 1200*5) = 13000 + 15000 = 28000
    assert audit.candidate_tokens == 1_850
    assert audit.opus_tokens == 29_850
    assert audit.percent_quota_misallocated == pytest.approx(6.2, abs=0.1)
    # The old whole-session heuristic (input < 5K) would never flag this session.
    assert 8_000 + 500 + 400 + 9_000 > md.SMALL_INPUT_TOKENS


def test_multi_model_session_example_labels_dominant_model(db):
    """A flagged session spanning two premium models must label its spot-check
    example (and routing hint) with the model carrying the MOST misallocated
    quota — not whichever premium turn appeared first. The first turn here is a
    small Fable turn, so the old first-turn-frozen label would mislabel it."""
    _new_session(db, "multi")
    _turn(db, "multi", 0, model="claude-fable-5",
          input_tokens=400, output_tokens=80)                        # ~800 quota
    _turn(db, "multi", 1, model="claude-opus-4-8",
          input_tokens=1_500, output_tokens=250)                     # ~2750 quota
    _turn(db, "multi", 2, model="claude-opus-4-8",
          input_tokens=1_500, output_tokens=250)                     # ~2750 quota

    audit = _audit(db)
    assert len(audit.examples) == 1
    ex = audit.examples[0]
    # Opus 4.8 carries ~5500 of ~6300 flagged quota → the dominant label.
    assert ex.model == "claude-opus-4-8"
    assert ex.alt_model == "claude-haiku-4-5"
    # Both premium models still appear in the aggregate suggestions.
    assert set(audit.suggestions) == {"claude-fable-5", "claude-opus-4-8"}


# ── (b) per-span attribution: mixed-model session ───────────────────────────

def test_mixed_model_session_attributes_tokens_per_actual_model(db):
    """An Opus turn + a Sonnet turn in ONE session. The Sonnet turn's tokens must
    land in NEITHER the premium denominator nor the misallocated numerator — the
    old MODE(model) sum would have collapsed both under the dominant model."""
    _new_session(db, "mm")
    _turn(db, "mm", 0, model="claude-opus-4-7",
          input_tokens=500, output_tokens=100, cache_tokens=50_000)   # premium
    _turn(db, "mm", 1, model="claude-sonnet-4-6",
          input_tokens=500, output_tokens=100, cache_tokens=50_000)   # not premium

    audit = _audit(db)

    # Only the Opus turn is premium: quota = 500 + 100*5 + 50000*0.1 = 6000.
    assert audit.opus_tokens == 6_000
    # Both turns are cheap → one whole-session segment; only the premium turn
    # inside it counts toward misallocation.
    assert audit.candidate_tokens == 6_000
    assert audit.percent_quota_misallocated == pytest.approx(100.0, abs=0.1)
    assert audit.opus_sessions == 1


# ── (c) the headline is a labelled estimate + CI ────────────────────────────

def test_headline_carries_confidence_label_and_ci(db):
    """Two flagged segments → the estimate is bracketed by a bootstrap CI over
    resampled segments, and marked an explicit 'estimate'."""
    _new_session(db, "c1")
    _turn(db, "c1", 0, input_tokens=500, output_tokens=100)
    _turn(db, "c1", 1, input_tokens=500, output_tokens=100)
    _new_session(db, "c2")
    _turn(db, "c2", 0, input_tokens=1_500, output_tokens=250)
    _turn(db, "c2", 1, input_tokens=1_500, output_tokens=250)

    audit = _audit(db)

    assert audit.segment_count == 2
    assert audit.segment_estimate_confidence == "estimate"
    assert audit.segment_ci_low is not None
    assert audit.segment_ci_high is not None
    assert audit.segment_ci_high >= audit.segment_ci_low


def test_single_segment_has_no_ci_but_stays_labelled(db):
    """A lone flagged segment has no spread to bootstrap — the CI bounds stay
    None (inherently wide) while the estimate label persists (design §9)."""
    _new_session(db, "solo")
    _turn(db, "solo", 0, input_tokens=500, output_tokens=100)
    _turn(db, "solo", 1, input_tokens=500, output_tokens=100)

    audit = _audit(db)

    assert audit.segment_count == 1
    assert audit.segment_ci_low is None
    assert audit.segment_ci_high is None
    assert audit.segment_estimate_confidence == "estimate"


# ── contiguity + min-stretch gate ───────────────────────────────────────────

def test_lone_cheap_turn_between_hard_turns_is_not_flagged(db):
    """A single cheap turn wedged between two hard turns is noise, not a
    reclaimable stretch (MIN_STRETCH_TURNS = 2), and does not span the session."""
    _new_session(db, "lone")
    _turn(db, "lone", 0, input_tokens=8_000, output_tokens=1_000)  # hard
    _turn(db, "lone", 1, input_tokens=500, output_tokens=100)       # lone cheap
    _turn(db, "lone", 2, input_tokens=9_000, output_tokens=1_200)  # hard

    audit = _audit(db)
    assert audit.opus_sessions == 1
    assert audit.candidate_sessions == 0
    assert audit.segment_count == 0
    assert audit.percent_quota_misallocated == 0.0


def test_min_stretch_of_one_flags_the_lone_turn(db, monkeypatch):
    """Flipping MIN_STRETCH_TURNS to 1 recovers 'every cheap turn counts'."""
    monkeypatch.setattr(md, "MIN_STRETCH_TURNS", 1)
    _new_session(db, "lone")
    _turn(db, "lone", 0, input_tokens=8_000, output_tokens=1_000)  # hard
    _turn(db, "lone", 1, input_tokens=500, output_tokens=100)       # lone cheap
    _turn(db, "lone", 2, input_tokens=9_000, output_tokens=1_200)  # hard

    audit = _audit(db)
    assert audit.candidate_sessions == 1
    assert audit.segment_count == 1


def test_tool_fanout_breaks_the_cheap_shape(db):
    """A turn with more than TURN_SMALL_TOOL_CALLS tool calls is not cheap-shaped,
    so it breaks a stretch even with small input/output — the per-turn fan-out is
    measured from the real interleaved tool spans."""
    _new_session(db, "fan")
    _turn(db, "fan", 0, input_tokens=500, output_tokens=100)                 # cheap
    _turn(db, "fan", 1, input_tokens=500, output_tokens=100, tool_calls=5)   # busy
    _turn(db, "fan", 2, input_tokens=500, output_tokens=100)                 # cheap

    audit = _audit(db)
    # Two length-1 segments, session has 3 turns → neither counts.
    assert audit.segment_count == 0
    assert audit.candidate_sessions == 0


# ── delegation gate ─────────────────────────────────────────────────────────

def test_task_delegation_breaks_the_stretch(db):
    """A Task handoff mid-stretch disqualifies the delegating turn (design §2.4
    Option B), splitting the run so neither half is a countable stretch — while
    an otherwise-identical non-delegating session IS flagged."""
    _new_session(db, "deleg")
    _turn(db, "deleg", 0, input_tokens=500, output_tokens=100)
    _turn(db, "deleg", 1, input_tokens=500, output_tokens=100, delegate=True)
    _turn(db, "deleg", 2, input_tokens=500, output_tokens=100)
    _new_session(db, "nodeleg")
    _turn(db, "nodeleg", 0, input_tokens=500, output_tokens=100)
    _turn(db, "nodeleg", 1, input_tokens=500, output_tokens=100)
    _turn(db, "nodeleg", 2, input_tokens=500, output_tokens=100)

    audit = _audit(db)

    flagged = {ex.session_id for ex in audit.examples}
    assert "nodeleg" in flagged
    assert "deleg" not in flagged
    assert audit.segment_count == 1


# ── backward-compat derivation (the nesting invariant, equality case) ───────

def test_fully_cheap_sessions_yield_hundred_percent(db):
    """Fully cheap single-model sessions: every premium token is inside a cheap
    segment, so the segment share equals the whole-session share (100%) — the
    ``segment% >= whole-session%`` invariant at equality."""
    for i in range(3):
        _new_session(db, f"f{i}")
        _turn(db, f"f{i}", 0, input_tokens=500, output_tokens=100)

    audit = _audit(db)
    assert audit.opus_sessions == 3
    assert audit.candidate_sessions == 3
    assert audit.percent_quota_misallocated == pytest.approx(100.0, abs=0.1)


# ── honesty discipline (Rule 14) ────────────────────────────────────────────

def test_estimate_basis_and_caveat_are_honest(db):
    _new_session(db, "f0")
    _turn(db, "f0", 0, input_tokens=500, output_tokens=100)
    audit = _audit(db)

    assert "no quality validation" in audit.estimate_basis
    assert audit.segment_estimate_confidence == "estimate"
    caveat = audit.caveat.lower()
    assert "spot-check" in caveat
    assert "wasted" not in caveat
    assert "safe to downgrade" in caveat  # only ever as "Never safe to downgrade"


# ── round-trip (a dropped new field must fail the parity guard) ─────────────

def test_new_fields_round_trip_with_ci(db):
    _new_session(db, "c1")
    _turn(db, "c1", 0, input_tokens=500, output_tokens=100)
    _turn(db, "c1", 1, input_tokens=1_500, output_tokens=250)
    _new_session(db, "c2")
    _turn(db, "c2", 0, input_tokens=800, output_tokens=200)
    _turn(db, "c2", 1, input_tokens=600, output_tokens=150)
    audit = _audit(db)

    round_tripped = audit_from_dict(audit_to_dict(audit))
    assert round_tripped.segment_count == audit.segment_count
    assert round_tripped.segment_estimate_confidence == audit.segment_estimate_confidence
    assert round_tripped.estimate_basis == audit.estimate_basis
    assert round_tripped.segment_ci_low == audit.segment_ci_low
    assert round_tripped.segment_ci_high == audit.segment_ci_high
    assert round_tripped.percent_quota_misallocated == audit.percent_quota_misallocated


def test_none_ci_round_trips_as_none(db):
    _new_session(db, "solo")
    _turn(db, "solo", 0, input_tokens=500, output_tokens=100)
    _turn(db, "solo", 1, input_tokens=500, output_tokens=100)
    audit = _audit(db)
    assert audit.segment_ci_low is None

    round_tripped = audit_from_dict(audit_to_dict(audit))
    assert round_tripped.segment_ci_low is None
    assert round_tripped.segment_ci_high is None


def test_context_tagged_and_bedrock_premium_stretches_are_audited(db):
    """A premium turn must be counted + flagged regardless of id SHAPE — a [1m]
    context tag or a Bedrock region/provider prefix must not make it vanish from
    the audit (it's still recognized by the subagent analyzer, so dropping it
    here would be a silent inconsistency)."""
    # 1M-context Fable, Sonnet-shaped → whole-session cheap → candidate.
    _new_session(db, "fable-1m")
    _turn(db, "fable-1m", 0, model="claude-fable-5[1m]",
          input_tokens=1_500, output_tokens=250, cache_tokens=98_250,
          cost_usd=5.0, tool_calls=2)
    # Bedrock-hosted Opus 4.8, Sonnet-shaped → candidate.
    _new_session(db, "bedrock-opus")
    _turn(db, "bedrock-opus", 0,
          model="us-anthropic-claude-opus-4-8-20260115-v1",
          input_tokens=1_000, output_tokens=200, cache_tokens=98_800,
          cost_usd=3.0, tool_calls=1)

    audit = _audit(db)

    # Both counted in the premium denominator (not silently dropped by shape).
    assert audit.opus_sessions == 2
    assert audit.candidate_sessions == 2
    assert {ex.session_id for ex in audit.examples} == {"fable-1m", "bedrock-opus"}
    # The id-normalizing downgrade lookup resolves an alt for each shaped id.
    assert audit.suggestions["claude-fable-5[1m]"] == "claude-sonnet-4-6"
    assert audit.suggestions["us-anthropic-claude-opus-4-8-20260115-v1"] == "claude-haiku-4-5"


# ── empty / honest states ───────────────────────────────────────────────────

def test_no_premium_sessions_is_clean_empty(db):
    _new_session(db, "sonnet-only")
    _turn(db, "sonnet-only", 0, model="claude-sonnet-4-6",
          input_tokens=500, output_tokens=100)
    audit = _audit(db)
    assert not audit.has_opus
    assert audit.opus_sessions == 0
    assert audit.percent_quota_misallocated == 0.0


def test_all_hard_premium_session_flags_nothing(db):
    _new_session(db, "hard")
    _turn(db, "hard", 0, input_tokens=8_000, output_tokens=1_000)
    _turn(db, "hard", 1, input_tokens=9_000, output_tokens=1_200)
    audit = _audit(db)
    assert audit.opus_sessions == 1
    assert audit.candidate_sessions == 0
    assert audit.segment_count == 0
    assert audit.percent_quota_misallocated == 0.0


def test_fable_and_opus_4_8_stretches_are_audited(db):
    """Fable (above Opus) and Opus 4.8 must both count in the premium denominator
    and be flagged when their turns form a Sonnet-shaped stretch — the point of
    routing tier membership through the shared predicate, not a substring gate."""
    _new_session(db, "fable")
    _turn(db, "fable", 0, model="claude-fable-5", input_tokens=500, output_tokens=100)
    _turn(db, "fable", 1, model="claude-fable-5", input_tokens=600, output_tokens=120)
    _new_session(db, "opus48")
    _turn(db, "opus48", 0, model="claude-opus-4-8", input_tokens=500, output_tokens=100)
    _turn(db, "opus48", 1, model="claude-opus-4-8", input_tokens=400, output_tokens=90)

    audit = _audit(db)
    assert audit.opus_sessions == 2
    assert audit.candidate_sessions == 2
    assert {ex.session_id for ex in audit.examples} == {"fable", "opus48"}
    assert audit.suggestions["claude-fable-5"]
    assert audit.suggestions["claude-opus-4-8"]


# ── CLI rendering (per-persona copy) ────────────────────────────────────────

def _seed_cli(db, *, plan_tier="max_5x") -> None:
    """A mixed session (hard + a flagged cheap stretch) plus a fully cheap
    session → 2 flagged segments so the render exercises the CI + persona copy."""
    _new_session(db, "cli-mixed", plan_tier=plan_tier)
    _turn(db, "cli-mixed", 0, input_tokens=8_000, output_tokens=1_000, cost_usd=4.0)
    _turn(db, "cli-mixed", 1, input_tokens=500, output_tokens=100, cost_usd=1.0)
    _turn(db, "cli-mixed", 2, input_tokens=400, output_tokens=90, cost_usd=1.0)
    _new_session(db, "cli-cheap", plan_tier=plan_tier)
    _turn(db, "cli-cheap", 0, input_tokens=500, output_tokens=100, cost_usd=1.0)
    _turn(db, "cli-cheap", 1, input_tokens=600, output_tokens=120, cost_usd=1.0)


def test_cli_renders_quota_audit_with_caveat(db, monkeypatch):
    """The card reports the % of premium quota that WENT to Sonnet-shaped WORK
    (misallocated, not 'reclaimable'), labels it an estimate, lists spot-check
    sessions, and surfaces the honesty caveat — in quota (not dollar) language
    for a subscription (Max) plan."""
    _seed_cli(db)
    config = _max_config()

    import tokenjam.cli.main as cli_main
    monkeypatch.setattr(cli_main, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(cli_main, "open_db", lambda *a, **k: db)
    monkeypatch.setattr("tokenjam.core.framing.config_declared_plan", lambda c: "max_5x")

    result = CliRunner().invoke(
        cli_main.cli, ["quota-audit", "--since", "30d"], catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    out = result.output
    flat = " ".join(re.sub(r"[│╭╮╰╯─]", " ", out).split())
    # Retrospective mirror headline (not dollars), new grain-accurate noun.
    assert "went to Sonnet-shaped work" in flat
    assert "reclaimable" not in flat.lower()
    assert "Opus/Fable" in out
    # The single number is presented as a labelled estimate.
    assert "estimate" in flat
    # Subscription users get the habit nudge, not dollars.
    assert "stays available for hard problems next window" in flat
    assert "/model sonnet" in flat
    # Spot-check sessions listed.
    assert "spot-check" in out.lower()
    assert "cli-mixed" in out
    # Honesty caveat present.
    assert "spot-check" in OPUS_QUOTA_AUDIT_CAVEAT.lower()
    assert "safe to downgrade" in out


def test_cli_api_persona_shows_dollar_counterfactual(db, monkeypatch):
    """API-billed users get the already-billed dollar counterfactual under the
    measured headline instead of the subscription habit nudge."""
    _seed_cli(db, plan_tier="api")
    config = _api_config()

    import tokenjam.cli.main as cli_main
    monkeypatch.setattr(cli_main, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(cli_main, "open_db", lambda *a, **k: db)
    monkeypatch.setattr("tokenjam.core.framing.config_declared_plan", lambda c: "api")

    result = CliRunner().invoke(
        cli_main.cli, ["quota-audit", "--since", "30d"], catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    flat = " ".join(re.sub(r"[│╭╮╰╯─]", " ", result.output).split())
    assert "went to Sonnet-shaped work" in flat
    assert "Same work at the suggested tiers" in flat
    assert "already billed" in flat
    assert "stays available for hard problems" not in flat


def test_api_nudge_without_pricing_data_stays_neutral():
    """API mode + no pricing data (every candidate model absent from the pricing
    table → actual_cost_usd == 0) must NOT fall through to the subscription
    "next window" quota language — API billing has no rolling window. It gets a
    neutral routing prompt instead of a dollar counterfactual or a habit nudge."""
    from tokenjam.cli.cmd_quota_audit import _headline_nudge
    from tokenjam.core.framing import Framing
    from tokenjam.core.optimize.types import OpusQuotaAudit

    audit = OpusQuotaAudit(
        opus_sessions=3, opus_tokens=300_000,
        candidate_sessions=2, candidate_tokens=200_000,
        percent_quota_misallocated=66.7, percent_sessions=66.7,
        actual_cost_usd=0.0, alternative_cost_usd=0.0,
    )
    framing = Framing(pricing_mode="api", plan_tier="api")
    nudge = _headline_nudge(audit, framing).plain

    # Neither the subscription quota language nor a fabricated dollar figure.
    assert "next window" not in nudge
    assert "stays available for hard problems" not in nudge
    assert "$" not in nudge
    assert "Review these sessions" in nudge


def test_api_nudge_with_pricing_shows_dollar_counterfactual():
    """API mode WITH pricing data gets the already-billed dollar counterfactual,
    never the subscription 'next window' language."""
    from tokenjam.cli.cmd_quota_audit import _headline_nudge
    from tokenjam.core.framing import Framing
    from tokenjam.core.optimize.types import OpusQuotaAudit

    audit = OpusQuotaAudit(
        opus_sessions=3, opus_tokens=300_000,
        candidate_sessions=2, candidate_tokens=200_000,
        percent_quota_misallocated=66.7, percent_sessions=66.7,
        actual_cost_usd=12.0, alternative_cost_usd=3.0,
    )
    framing = Framing(pricing_mode="api", plan_tier="api")
    nudge = _headline_nudge(audit, framing).plain

    assert "already billed" in nudge
    assert "next window" not in nudge


def test_cli_json_output_has_segment_fields(db, monkeypatch):
    _seed_cli(db)
    config = _max_config()
    import json

    import tokenjam.cli.main as cli_main
    monkeypatch.setattr(cli_main, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(cli_main, "open_db", lambda *a, **k: db)
    monkeypatch.setattr("tokenjam.core.framing.config_declared_plan", lambda c: "max_5x")

    result = CliRunner().invoke(
        cli_main.cli, ["quota-audit", "--since", "30d", "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["percent_quota_misallocated"] > 0
    # DEPRECATED alias still mirrors the headline (0.5.4 compat).
    assert payload["percent_quota_reclaimable"] == payload["percent_quota_misallocated"]
    assert payload["candidate_sessions"] >= 1
    assert payload["segment_count"] >= 1
    assert payload["segment_estimate_confidence"] == "estimate"
    assert "estimate_basis" in payload
    assert len(payload["examples"]) >= 1
    assert payload["framing"]["pricing_mode"] == "subscription"
    assert payload["caveat"] == OPUS_QUOTA_AUDIT_CAVEAT
