"""Unit tests for the cache-recommend analyzer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import CaptureConfig, OptimizeConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers.cache_recommend import (
    MIN_PREFIX_OCCURRENCES,
    _prefix_hash,
    _stringify_prompt,
)
from tokenjam.otel.semconv import GenAIAttributes
from tests.factories import make_llm_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _flat(out: str) -> str:
    """Collapse Rich's terminal-width line wrapping to a single line so a
    long fixed string can be matched by substring regardless of where the
    console happened to wrap it."""
    return " ".join(out.split())


def _config(capture_prompts: bool) -> TjConfig:
    return TjConfig(
        version="1",
        capture=CaptureConfig(prompts=capture_prompts),
    )


def _seed_with_prompt(db, *, prompt: str, count: int, provider: str = "anthropic",
                     start=None, input_tokens: int = 2000, model: str = "claude-sonnet-4-6"):
    """Insert N spans sharing the same captured prompt."""
    start = start or datetime(2026, 5, 10, tzinfo=timezone.utc)
    # IngestPipeline normally strips content based on capture config — but
    # these tests bypass IngestPipeline and write directly to db. The
    # analyzer reads attributes.gen_ai.prompt.content, which we set here.
    for i in range(count):
        span = make_llm_span(
            agent_id="test-agent",
            provider=provider,
            billing_account=provider,
            model=model,
            input_tokens=input_tokens,
            cost_usd=0.005,
            start_time=start + timedelta(minutes=i),
            extra_attributes={GenAIAttributes.PROMPT_CONTENT: prompt},
        )
        db.insert_span(span)


# -- Pure-function tests --

def test_stringify_prompt_str():
    assert _stringify_prompt("hello") == "hello"


def test_stringify_prompt_message_list():
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    out = _stringify_prompt(msgs)
    assert "you are helpful" in out
    assert "hi" in out
    assert "system" in out and "user" in out


def test_stringify_prompt_anthropic_block_list():
    """Anthropic message content can be a list of block dicts."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": "the prompt"}]}]
    assert "the prompt" in _stringify_prompt(msgs)


def test_prefix_hash_deterministic():
    assert _prefix_hash("foo" * 1000) == _prefix_hash("foo" * 1000)
    assert _prefix_hash("foo") != _prefix_hash("bar")


# -- Integration via build_report --

def test_disabled_when_capture_prompts_off(db):
    """Without capture.prompts the analyzer returns a hint, not candidates."""
    _seed_with_prompt(db, prompt="x" * 2500, count=5)
    config = _config(capture_prompts=False)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])
    finding = report.findings["cache-recommend"]
    assert finding.enabled is False
    assert finding.candidates == []
    assert "capture" in finding.hint.lower()


def test_identifies_repeated_prefix(db):
    """Three+ Anthropic calls sharing a long prefix produce a candidate."""
    _seed_with_prompt(db, prompt="SYSTEM: " + "you are helpful. " * 200,
                      count=5, input_tokens=2500)
    config = _config(capture_prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])
    finding = report.findings["cache-recommend"]
    assert finding.enabled is True
    assert len(finding.candidates) == 1
    c = finding.candidates[0]
    assert c.occurrences == 5
    assert c.avg_input_tokens == pytest.approx(2500.0)
    assert "you are helpful" in c.sample_chars


def test_candidate_carries_a_ready_cache_control_snippet(db):
    """cache-recommend's whole job is placement advice, so a candidate must
    ship a pasteable cache_control snippet, not just prose stats (issue: the
    analyzer previously had no snippet field at all)."""
    _seed_with_prompt(db, prompt="SYSTEM: " + "you are helpful. " * 200,
                      count=5, input_tokens=2500, model="claude-sonnet-4-6")
    config = _config(capture_prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])
    c = report.findings["cache-recommend"].candidates[0]
    assert c.cache_control_snippet
    assert "cache_control" in c.cache_control_snippet
    assert "ephemeral" in c.cache_control_snippet
    assert "claude-sonnet-4-6" in c.cache_control_snippet
    assert "5 calls" in c.cache_control_snippet
    # A placeholder `text` value, not the real captured prefix pasted in
    # full: the snippet stays short (a short preview + boilerplate) even
    # though the actual captured prompt repeats "you are helpful." 200
    # times over.
    assert "<the stable prefix" in c.cache_control_snippet
    assert len(c.cache_control_snippet) < 500


def test_skips_non_anthropic_providers(db):
    """OpenAI/Gemini spans are counted in skipped_provider_count and not as candidates."""
    _seed_with_prompt(db, prompt="x" * 3000, count=5, provider="openai")
    config = _config(capture_prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])
    finding = report.findings["cache-recommend"]
    assert finding.enabled is True
    assert finding.candidates == []
    assert finding.skipped_provider_count == 5


def test_below_min_occurrences_not_flagged(db):
    """Two calls sharing a prefix is below MIN_PREFIX_OCCURRENCES — not flagged."""
    assert MIN_PREFIX_OCCURRENCES == 3  # sanity
    _seed_with_prompt(db, prompt="x" * 3000, count=2)
    config = _config(capture_prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])
    finding = report.findings["cache-recommend"]
    assert finding.candidates == []
    assert finding.min_prefix_occurrences == MIN_PREFIX_OCCURRENCES


def test_config_lowers_occurrence_bar_surfaces_previously_hidden_candidate(db):
    """The exact 2-call data from test_below_min_occurrences_not_flagged
    produces nothing at the default bar; lowering [optimize]
    min_prefix_occurrences to 2 surfaces it."""
    _seed_with_prompt(db, prompt="x" * 3000, count=2)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)

    default_report = build_report(
        db=db, config=_config(capture_prompts=True), since=since, until=until,
        findings=["cache-recommend"],
    )
    assert default_report.findings["cache-recommend"].candidates == []

    lowered_config = TjConfig(
        version="1", capture=CaptureConfig(prompts=True),
        optimize=OptimizeConfig(min_prefix_occurrences=2),
    )
    lowered_report = build_report(
        db=db, config=lowered_config, since=since, until=until,
        findings=["cache-recommend"],
    )
    lowered_finding = lowered_report.findings["cache-recommend"]
    assert len(lowered_finding.candidates) == 1
    assert lowered_finding.candidates[0].occurrences == 2
    assert lowered_finding.min_prefix_occurrences == 2


def test_short_prompts_skipped(db):
    """Prompts under 200 chars are skipped — no caching opportunity worth flagging."""
    _seed_with_prompt(db, prompt="too short", count=10)
    config = _config(capture_prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])
    assert report.findings["cache-recommend"].candidates == []


# -- N36: pricing the candidate --

def test_candidate_and_finding_carry_a_priced_recoverable_estimate(db):
    """A repeated prefix on a priced model gets a dollar figure, reusing
    `cache_efficacy`'s rate-lookup + rate-delta pricing pattern."""
    _seed_with_prompt(db, prompt="SYSTEM: " + "you are helpful. " * 200,
                      count=5, input_tokens=2500, model="claude-sonnet-4-6")
    config = _config(capture_prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])
    finding = report.findings["cache-recommend"]
    c = finding.candidates[0]
    assert c.model == "claude-sonnet-4-6"
    assert c.estimated_recoverable_usd is not None
    assert c.estimated_recoverable_usd > 0
    assert c.estimated_recoverable_tokens == c.estimated_cacheable_tokens * (c.occurrences - 1)
    assert finding.estimated_recoverable_usd == pytest.approx(c.estimated_recoverable_usd)
    assert finding.estimated_recoverable_tokens == c.estimated_recoverable_tokens
    assert finding.estimate_basis


def test_no_dollar_figure_for_unpriced_model(db):
    """No rate observed for the model -> None, never a $0.00 or a borrowed
    rate (CLAUDE.md anti-pattern #22)."""
    _seed_with_prompt(db, prompt="SYSTEM: " + "you are helpful. " * 200,
                      count=5, input_tokens=2500,
                      model="totally-unpriced-model-xyz")
    config = _config(capture_prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])
    finding = report.findings["cache-recommend"]
    c = finding.candidates[0]
    assert c.model == "totally-unpriced-model-xyz"
    assert c.estimated_recoverable_usd is None
    assert c.estimated_recoverable_tokens is None
    assert finding.estimated_recoverable_usd is None
    assert finding.estimated_recoverable_tokens is None


# -- CLI rendering respects pricing_mode --

def test_render_cache_recommend_shows_dollars_on_api(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_recommend

    _seed_with_prompt(db, prompt="SYSTEM: " + "you are helpful. " * 200,
                      count=5, input_tokens=2500, model="claude-sonnet-4-6")
    config = _config(capture_prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])

    _render_cache_recommend(report.findings["cache-recommend"], pricing_mode="api")
    out = capsys.readouterr().out
    assert "$" in out
    assert "estimated" in out


def test_render_cache_recommend_suppresses_dollars_off_api(db, capsys):
    """Subscription/local plans don't bill per token, so no dollar figure is
    shown; the token counts still print (CLAUDE.md anti-pattern #22)."""
    from tokenjam.cli.cmd_optimize import _render_cache_recommend

    _seed_with_prompt(db, prompt="SYSTEM: " + "you are helpful. " * 200,
                      count=5, input_tokens=2500, model="claude-sonnet-4-6")
    config = _config(capture_prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])

    for mode in ("subscription", "local"):
        _render_cache_recommend(report.findings["cache-recommend"], pricing_mode=mode)
    out = capsys.readouterr().out
    assert "$" not in out
    assert "cacheable/call" in out       # the token-level opportunity still shows
    assert "doesn't bill per token" in out


# -- CLI rendering: cache_control snippet, persona-gated --
#
# Mirrors the gate `cost_proposals._persona_gated_cache_fields` applies to
# the Review-inbox proposal built from this same finding (see
# test_cache_root_cause_proposals.py for that side): a `cache_control` edit
# is on the raw Anthropic API request, code a Claude Code session never
# constructs itself. "unknown" stays actionable here (the CLI's default when
# no persona is threaded through) -- the risky direction for cache advice is
# withholding a real fix, not over-offering one.

def _build_cache_recommend_report(db):
    _seed_with_prompt(db, prompt="SYSTEM: " + "you are helpful. " * 200,
                      count=5, input_tokens=2500, model="claude-sonnet-4-6")
    config = _config(capture_prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["cache-recommend"])
    return report.findings["cache-recommend"]


def test_render_cache_recommend_shows_snippet_by_default(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_recommend

    finding = _build_cache_recommend_report(db)
    c = finding.candidates[0]
    assert c.cache_control_snippet

    _render_cache_recommend(finding, pricing_mode="api")
    out = capsys.readouterr().out

    assert "cache_control:" in out
    assert '"cache_control"' in out
    assert '"type": "ephemeral"' in out


def test_render_cache_recommend_unknown_persona_still_shows_snippet(db, capsys):
    """`unknown` (the CLI default) stays on the actionable branch -- the
    opposite grouping from `_persona_gated_write_fields`'s writes, and
    exactly the rule `_persona_gated_cache_fields` documents."""
    from tokenjam.cli.cmd_optimize import _render_cache_recommend

    finding = _build_cache_recommend_report(db)

    _render_cache_recommend(finding, pricing_mode="api", persona="unknown")
    out = capsys.readouterr().out

    assert '"cache_control"' in out


def test_render_cache_recommend_claude_code_suppresses_snippet(db, capsys):
    """A Claude Code session doesn't construct the raw Anthropic request --
    the harness does -- so the snippet is swapped for the honest no-lever
    explanation, imported straight from cost_proposals so the CLI never
    drifts from the web copy."""
    from tokenjam.cli.cmd_optimize import _render_cache_recommend
    from tokenjam.core.optimize.cost_proposals import CACHE_NO_LEVER_TEXT

    finding = _build_cache_recommend_report(db)

    _render_cache_recommend(finding, pricing_mode="api", persona="claude-code")
    out = _flat(capsys.readouterr().out)

    assert '"cache_control"' not in out
    assert CACHE_NO_LEVER_TEXT in out


def test_render_cache_recommend_mixed_persona_still_shows_snippet(db, capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_recommend

    finding = _build_cache_recommend_report(db)

    _render_cache_recommend(finding, pricing_mode="api", persona="mixed")
    out = capsys.readouterr().out

    assert '"cache_control"' in out


def test_render_cache_recommend_snippet_uses_plain_console_print(db, capsys):
    """Matches `_render_cache_root_causes`'s existing snippet treatment:
    printed on its own line via `markup=False, highlight=False,
    soft_wrap=True` -- not interpolated into a Rich-markup f-string, which
    would risk brackets in the JSON snippet being swallowed as style tags."""
    from tokenjam.cli.cmd_optimize import _render_cache_recommend

    finding = _build_cache_recommend_report(db)
    c = finding.candidates[0]

    _render_cache_recommend(finding, pricing_mode="api")
    out = capsys.readouterr().out

    assert c.cache_control_snippet in out
