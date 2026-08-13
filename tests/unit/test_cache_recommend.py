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


def test_claude_code_persona_produces_candidates_from_system_prefix(db):
    """#272: live PR-539 showed enabled=True but candidates=0 for
    Claude-Code-sourced data. Root cause: PROMPT_CONTENT on those spans is
    the human's per-turn message, a different string every call that never
    repeats verbatim -- hashing it can never find a shared prefix. Each span
    here carries a DIFFERENT PROMPT_CONTENT (simulating distinct human
    turns) but the SAME SYSTEM_PREFIX_CONTENT (simulating the project's
    CLAUDE.md, resent unchanged on every call) -- the analyzer must key off
    the latter and still surface a candidate."""
    from tokenjam.otel.semconv import TjAttributes
    from tests.factories import make_llm_span

    start = datetime(2026, 5, 10, tzinfo=timezone.utc)
    claude_md = "# Project rules\n" + "Always use tabs. " * 100
    for i in range(5):
        span = make_llm_span(
            agent_id="claude-code-proj",
            provider="anthropic",
            billing_account="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=2000,
            cost_usd=0.005,
            start_time=start + timedelta(minutes=i),
            extra_attributes={
                GenAIAttributes.PROMPT_CONTENT: f"human turn number {i}, always different",
                TjAttributes.SYSTEM_PREFIX_CONTENT: claude_md,
            },
        )
        db.insert_span(span)

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
    assert "Project rules" in c.sample_chars


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


def test_min_prefix_occurrences_survives_report_dict_round_trip():
    """`CacheRecommendFinding(**c)` (runner._cache_recommend) must round-trip a
    non-default `min_prefix_occurrences` — omitting it on reconstruction would
    silently revert every re-loaded report back to the class default
    regardless of what the user configured and what got serialized."""
    from tokenjam.core.optimize.analyzers.cache_recommend import CacheRecommendFinding
    from tokenjam.core.optimize.runner import report_from_dict, report_to_dict
    from tokenjam.core.optimize.types import OptimizeReport, WindowSummary

    assert MIN_PREFIX_OCCURRENCES != 7  # sanity: a genuinely non-default value

    w = WindowSummary(since=datetime(2026, 5, 1, tzinfo=timezone.utc),
                      until=datetime(2026, 5, 30, tzinfo=timezone.utc), days=29,
                      sessions=20, spans=20, total_tokens=24000, total_cost_usd=1.0,
                      thin_data=False)
    report = OptimizeReport(window=w, findings={
        "cache-recommend": CacheRecommendFinding(enabled=True, min_prefix_occurrences=7),
    })

    rebuilt = report_from_dict(report_to_dict(report))
    assert rebuilt.findings["cache-recommend"].min_prefix_occurrences == 7

    # An older payload with the key entirely absent still round-trips: default.
    old_payload = report_to_dict(report)
    del old_payload["findings"]["cache-recommend"]["min_prefix_occurrences"]
    old_rebuilt = report_from_dict(old_payload)
    assert old_rebuilt.findings["cache-recommend"].min_prefix_occurrences == MIN_PREFIX_OCCURRENCES


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
    assert c.past_overspend_usd is not None
    assert c.past_overspend_usd > 0
    assert c.past_overspend_tokens == c.estimated_cacheable_tokens * (c.occurrences - 1)
    assert finding.past_overspend_usd == pytest.approx(c.past_overspend_usd)
    assert finding.past_overspend_tokens == c.past_overspend_tokens
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
    assert c.past_overspend_usd is None
    assert c.past_overspend_tokens is None
    assert finding.past_overspend_usd is None
    assert finding.past_overspend_tokens is None


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


# --- the system prefix is stored compactly, not as text -----------------------
#
# It used to be stored whole: 92,514 spans each holding a ~43 KB copy of one of
# 61 distinct files, 4.06 GB of database to carry 1.84 MB of distinct text. The
# analyzer never read it as text — it hashed a fixed head, kept 120 characters
# for display, and compared a length. These pin that the compact form carries
# all three answers, that legacy spans still work, and that nothing silently
# reintroduces the weight.


def test_backfill_stores_a_fingerprint_not_the_file(tmp_path):
    """The defect, stated as a property: a big CLAUDE.md must not make a big span."""
    from tokenjam.core.backfill import _read_project_claude_md, _system_prefix_attrs
    from tokenjam.otel.semconv import TjAttributes

    marker = "PROJECT-RULES-MARKER"
    (tmp_path / "CLAUDE.md").write_text(marker + ("\nrule line" * 6000), encoding="utf-8")

    text = _read_project_claude_md(str(tmp_path))
    attrs = _system_prefix_attrs(text)

    assert len(text) > 50_000, "fixture should be a genuinely large file"
    stored = sum(len(str(v)) for v in attrs.values())
    assert stored < 300, f"stored {stored} bytes of a {len(text)}-byte file"
    assert TjAttributes.SYSTEM_PREFIX_CONTENT not in attrs
    assert not any(marker in str(v) for k, v in attrs.items()
                   if k != TjAttributes.SYSTEM_PREFIX_SAMPLE), \
        "only the display sample may quote the file"


def test_stored_hash_matches_what_the_analyzer_computes_from_text():
    """Identity must survive the change of storage.

    A span stamped with the compact hash and a legacy span carrying the same
    text have to land in the SAME candidate. If they didn't, one project's
    calls would split across two candidates after the upgrade and both
    occurrence counts would be wrong with nothing failing.
    """
    from tokenjam.core.backfill import _system_prefix_attrs
    from tokenjam.core.optimize.analyzers.cache_recommend import _prefix_hash
    from tokenjam.otel.semconv import TjAttributes

    text = "# rules\n" + ("a repeated instruction line\n" * 500)
    stamped = _system_prefix_attrs(text)[TjAttributes.SYSTEM_PREFIX_HASH]

    assert stamped == _prefix_hash(text)


def test_hash_window_is_single_sourced():
    """The analyzer's window and the parser's must be one constant.

    They were two copies of the same three lines. Divergence has no symptom —
    it just quietly regroups spans — so the guard is that both route through
    `core.system_prefix`.
    """
    from tokenjam.core import system_prefix
    from tokenjam.core.optimize.analyzers.cache_recommend import PREFIX_HASH_BYTES

    assert PREFIX_HASH_BYTES == system_prefix.HASH_CHARS


def test_two_prefixes_differing_after_the_window_are_one_candidate():
    """States what the window MEANS, so a change to it is a deliberate one.

    Anything shared past HASH_CHARS is the same cacheable prefix by this
    product's definition; the tail is what a breakpoint would not cover.
    """
    from tokenjam.core import system_prefix

    head = "x" * system_prefix.HASH_CHARS
    assert system_prefix.prefix_hash(head + "tail A") == \
           system_prefix.prefix_hash(head + "tail B")
    assert system_prefix.prefix_hash("y" + head) != system_prefix.prefix_hash(head)


def test_short_prefix_keeps_its_real_length(tmp_path):
    """Below the cacheable floor the analyzer must still be able to skip it."""
    from tokenjam.core.backfill import _read_project_claude_md, _system_prefix_attrs
    from tokenjam.otel.semconv import TjAttributes

    (tmp_path / "CLAUDE.md").write_text("tiny", encoding="utf-8")
    attrs = _system_prefix_attrs(_read_project_claude_md(str(tmp_path)))

    assert attrs[TjAttributes.SYSTEM_PREFIX_LENGTH] == 4


def test_capture_off_strips_every_system_prefix_key():
    """Turning capture off must not leave a fingerprint of the file behind."""
    from tokenjam.core.ingest import strip_captured_content
    from tokenjam.otel.semconv import TjAttributes

    attrs = {
        TjAttributes.SYSTEM_PREFIX_CONTENT: "the whole file",
        TjAttributes.SYSTEM_PREFIX_HASH: "deadbeefdeadbeef",
        TjAttributes.SYSTEM_PREFIX_SAMPLE: "first 120 chars",
        TjAttributes.SYSTEM_PREFIX_LENGTH: 40000,
        "keep.me": "untouched",
    }
    stripped = strip_captured_content(attrs, CaptureConfig(prompts=False))

    assert stripped.get("keep.me") == "untouched"
    for key in (TjAttributes.SYSTEM_PREFIX_CONTENT, TjAttributes.SYSTEM_PREFIX_HASH,
                TjAttributes.SYSTEM_PREFIX_SAMPLE, TjAttributes.SYSTEM_PREFIX_LENGTH):
        assert key not in stripped, key


def _seed_prefix_spans(db, *, attrs: dict, count: int, start_offset: int = 0):
    """Insert N spans carrying an arbitrary system-prefix attribute shape."""
    start = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(count):
        db.insert_span(make_llm_span(
            agent_id="test-agent", provider="anthropic", billing_account="anthropic",
            model="claude-sonnet-4-6", input_tokens=2500, cost_usd=0.005,
            start_time=start + timedelta(minutes=start_offset + i),
            extra_attributes=dict(attrs),
        ))


def _cache_finding(db):
    return build_report(
        db=db, config=_config(capture_prompts=True),
        since=datetime(2026, 5, 1, tzinfo=timezone.utc),
        until=datetime(2026, 5, 30, tzinfo=timezone.utc),
        findings=["cache-recommend"],
    ).findings["cache-recommend"]


def test_compact_prefix_spans_produce_a_candidate(db):
    """End-to-end: the analyzer works off the stored fingerprint alone."""
    from tokenjam.core.backfill import _system_prefix_attrs

    text = "# global rules\n" + ("an instruction that repeats\n" * 400)
    _seed_prefix_spans(db, attrs=_system_prefix_attrs(text), count=5)

    finding = _cache_finding(db)

    assert finding.enabled
    assert len(finding.candidates) == 1
    assert finding.candidates[0].occurrences == 5
    assert finding.candidates[0].sample_chars  # display survives the change


def test_legacy_and_compact_spans_land_in_one_candidate(db):
    """The upgrade must not split a project's history in two.

    Every span backfilled before this change carries the text; every one after
    carries the fingerprint. They describe the SAME prefix, so they have to
    group together — otherwise the day the release ships, one project's calls
    appear as two candidates and both occurrence counts are understated, with
    nothing failing to say so.
    """
    from tokenjam.core.backfill import _system_prefix_attrs
    from tokenjam.otel.semconv import TjAttributes

    text = "# global rules\n" + ("an instruction that repeats\n" * 400)

    _seed_prefix_spans(db, attrs={TjAttributes.SYSTEM_PREFIX_CONTENT: text},
                       count=4, start_offset=0)
    _seed_prefix_spans(db, attrs=_system_prefix_attrs(text),
                       count=3, start_offset=100)

    finding = _cache_finding(db)

    assert len(finding.candidates) == 1, \
        f"history split into {len(finding.candidates)} candidates"
    assert finding.candidates[0].occurrences == 7


def test_a_prefix_under_the_floor_is_still_skipped(db):
    """The length gate has to work off the stored length, not the sample."""
    from tokenjam.core.backfill import _system_prefix_attrs

    _seed_prefix_spans(db, attrs=_system_prefix_attrs("too short to cache"), count=5)

    assert _cache_finding(db).candidates == []
