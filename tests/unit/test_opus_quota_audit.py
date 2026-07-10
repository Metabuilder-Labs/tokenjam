"""Unit tests for the retroactive Opus quota audit (`tj quota-audit`, issue #5).

Exercises the audit over a SYNTHETIC mix of Opus sessions proving:
  * "% of Opus quota reclaimable" is computed as candidate-Opus-tokens over
    total-Opus-tokens (only Sonnet-shaped Opus sessions count);
  * Opus-shaped Opus sessions (big input/output, many tool calls) are NOT
    flagged, and non-Opus sessions are excluded from the denominator;
  * the spot-check example list surfaces the candidate sessions;
  * the CLI renders the audit in quota terms with the honesty caveat present.
"""
from __future__ import annotations

from datetime import timedelta

import pytest
from click.testing import CliRunner

from tokenjam.core.config import ProviderBudget, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize.analyzers.model_downgrade import audit_opus_quota
from tokenjam.core.optimize.types import OPUS_QUOTA_AUDIT_CAVEAT
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
    return TjConfig(
        version="1",
        budgets={"anthropic": ProviderBudget(plan="max_5x")},
    )


def _add_session(db, session_id, *, model, input_tokens, output_tokens,
                 cache_tokens=0, cost_usd=1.0, tool_calls=0):
    """Seed one session: an LLM span plus N tool spans."""
    sess = make_session(session_id=session_id, plan_tier="max_5x",
                        duration_seconds=60.0)
    db.upsert_session(sess)
    span = make_llm_span(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_tokens=cache_tokens,
        cost_usd=cost_usd,
        session_id=session_id,
    )
    span.start_time = BASE
    db.insert_span(span)
    for _ in range(tool_calls):
        tool = make_tool_span(tool_name="Read")
        tool.session_id = session_id
        tool.start_time = BASE + timedelta(seconds=1)
        db.insert_span(tool)


def _seed_mix(db) -> None:
    """A mix of Opus + Sonnet sessions:

      * opus-thin-1 / opus-thin-2 — Opus, Sonnet-shaped (small in/out, ≤5 tools)
        → quota-reclaim candidates.
      * opus-fat — Opus, genuinely Opus-shaped (big in/out, many tools) → NOT a
        candidate, but still in the Opus quota denominator.
      * sonnet-1 — already on Sonnet → excluded from the audit entirely (the
        audit only inspects Opus sessions).
    """
    # 100k Opus tokens each, Sonnet-shaped → candidates.
    _add_session(db, "opus-thin-1", model="claude-opus-4-7",
                 input_tokens=2_000, output_tokens=300, cache_tokens=97_700,
                 cost_usd=3.0, tool_calls=2)
    _add_session(db, "opus-thin-2", model="claude-opus-4-6",
                 input_tokens=1_000, output_tokens=200, cache_tokens=98_800,
                 cost_usd=3.0, tool_calls=1)
    # 100k Opus tokens, Opus-shaped → in denominator, NOT a candidate.
    _add_session(db, "opus-fat", model="claude-opus-4-7",
                 input_tokens=40_000, output_tokens=20_000, cache_tokens=40_000,
                 cost_usd=8.0, tool_calls=30)
    # Sonnet session → excluded from the audit (not Opus).
    _add_session(db, "sonnet-1", model="claude-sonnet-4-6",
                 input_tokens=2_000, output_tokens=300, cache_tokens=50_000,
                 cost_usd=0.5, tool_calls=1)


def test_percent_quota_reclaimable_counts_only_sonnet_shaped_opus(db):
    _seed_mix(db)
    audit = audit_opus_quota(db.conn, SINCE, UNTIL, agent_id=None, window_days=30.0)

    # Three Opus sessions in the denominator (the Sonnet session is excluded).
    assert audit.opus_sessions == 3
    assert audit.opus_tokens == 300_000  # 3 × 100k Opus tokens
    # Two are Sonnet-shaped candidates.
    assert audit.candidate_sessions == 2
    assert audit.candidate_tokens == 200_000  # 2 × 100k
    # Headline: candidate Opus tokens / total Opus tokens.
    assert audit.percent_quota_reclaimable == pytest.approx(66.7, abs=0.1)
    assert audit.percent_sessions == pytest.approx(66.7, abs=0.1)


def test_opus_shaped_session_not_flagged(db):
    _seed_mix(db)
    audit = audit_opus_quota(db.conn, SINCE, UNTIL, agent_id=None, window_days=30.0)
    flagged_ids = {ex.session_id for ex in audit.examples}
    # The genuinely Opus-shaped session is never a spot-check candidate.
    assert "opus-fat" not in flagged_ids
    # The Sonnet session is excluded from the audit entirely.
    assert "sonnet-1" not in flagged_ids


def test_example_sessions_surfaced_for_spot_check(db):
    _seed_mix(db)
    audit = audit_opus_quota(db.conn, SINCE, UNTIL, agent_id=None, window_days=30.0)

    assert {ex.session_id for ex in audit.examples} == {"opus-thin-1", "opus-thin-2"}
    # Each example carries the cheaper-model routing suggestion + shape data.
    for ex in audit.examples:
        assert ex.alt_model  # a concrete cheaper model is named
        assert ex.tool_calls <= 5
    # Suggestions aggregate the observed model→alt mapping.
    assert "claude-opus-4-7" in audit.suggestions
    assert "claude-opus-4-6" in audit.suggestions


def test_fable_and_opus_4_8_sessions_are_audited(db):
    """Fable (the tier above Opus) and Opus 4.8 must both be counted in the
    premium denominator and flagged as reclaim candidates when Sonnet-shaped —
    the whole point of routing tier membership through the shared predicate."""
    # Fable, Sonnet-shaped → candidate; suggested alt is a cheaper same-family model.
    _add_session(db, "fable-thin", model="claude-fable-5",
                 input_tokens=1_500, output_tokens=250, cache_tokens=98_250,
                 cost_usd=5.0, tool_calls=2)
    # Opus 4.8, Sonnet-shaped → candidate (4.8 was missing from the map before).
    _add_session(db, "opus48-thin", model="claude-opus-4-8",
                 input_tokens=1_000, output_tokens=200, cache_tokens=98_800,
                 cost_usd=3.0, tool_calls=1)
    # Fable, genuinely large-shape → in denominator, NOT a candidate.
    _add_session(db, "fable-fat", model="claude-fable-5",
                 input_tokens=40_000, output_tokens=20_000, cache_tokens=40_000,
                 cost_usd=12.0, tool_calls=30)

    audit = audit_opus_quota(db.conn, SINCE, UNTIL, agent_id=None, window_days=30.0)

    # All three premium sessions counted (Fable is premium, above Opus).
    assert audit.opus_sessions == 3
    assert audit.opus_tokens == 300_000
    # The two Sonnet-shaped premium sessions are reclaim candidates.
    assert audit.candidate_sessions == 2
    assert {ex.session_id for ex in audit.examples} == {"fable-thin", "opus48-thin"}
    # Each flagged premium session names a concrete cheaper routing target.
    assert audit.suggestions["claude-fable-5"]
    assert audit.suggestions["claude-opus-4-8"]
    for ex in audit.examples:
        assert ex.alt_model


def test_no_opus_sessions_is_clean_empty_state(db):
    # Only a Sonnet session — nothing for the Opus audit to inspect.
    _add_session(db, "sonnet-only", model="claude-sonnet-4-6",
                 input_tokens=2_000, output_tokens=300, cache_tokens=50_000)
    audit = audit_opus_quota(db.conn, SINCE, UNTIL, agent_id=None, window_days=30.0)
    assert not audit.has_opus
    assert audit.opus_sessions == 0
    assert audit.percent_quota_reclaimable == 0.0


def test_cli_renders_quota_audit_with_caveat(db, monkeypatch):
    """End-to-end: the card reports % of Opus quota reclaimable, lists the
    spot-check sessions, and surfaces the honesty caveat — in quota (not dollar)
    language for a subscription (Max) plan."""
    _seed_mix(db)
    config = _max_config()

    import tokenjam.cli.main as cli_main

    monkeypatch.setattr(cli_main, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(cli_main, "open_db", lambda *a, **k: db)
    monkeypatch.setattr(
        "tokenjam.core.framing.config_declared_plan", lambda c: "max_5x"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_main.cli, ["quota-audit", "--since", "30d"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    out = result.output
    # Quota framing headline (not dollars) — names both premium tiers.
    assert "quota is reclaimable" in out
    assert "Opus/Fable" in out
    assert "67%" in out or "66" in out
    # Spot-check sessions listed.
    assert "spot-check" in out.lower()
    assert "opus-thin-1" in out
    # Honesty caveat present (the load-bearing honesty discipline).
    assert "spot-check" in OPUS_QUOTA_AUDIT_CAVEAT.lower()
    assert "Never \"safe to downgrade.\"" in out or "safe to downgrade" in out


def test_cli_json_output_has_quota_fields(db, monkeypatch):
    _seed_mix(db)
    config = _max_config()
    import json

    import tokenjam.cli.main as cli_main

    monkeypatch.setattr(cli_main, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(cli_main, "open_db", lambda *a, **k: db)
    monkeypatch.setattr(
        "tokenjam.core.framing.config_declared_plan", lambda c: "max_5x"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_main.cli, ["quota-audit", "--since", "30d", "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["percent_quota_reclaimable"] == pytest.approx(66.7, abs=0.1)
    assert payload["candidate_sessions"] == 2
    assert len(payload["examples"]) == 2
    assert payload["framing"]["pricing_mode"] == "subscription"
    assert payload["caveat"] == OPUS_QUOTA_AUDIT_CAVEAT
