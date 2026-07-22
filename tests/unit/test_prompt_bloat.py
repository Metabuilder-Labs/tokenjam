"""
Unit tests for the trim (Trim) analyzer.

The LLMLingua-2 model is mocked across these tests so CI doesn't have to
download ~110MB and run an actual BERT classifier. The mock returns
hand-crafted (token, score) tuples that exercise each branch of the
region-extraction logic.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tokenjam.core.config import CaptureConfig, OptimizeConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers import prompt_bloat as trim_module
from tokenjam.core.optimize.analyzers.prompt_bloat import (
    MIN_PROVENANCE_CHARS,
    MIN_REGION_LENGTH,
    PromptBloatFinding,
    _agent_repo_basename,
    _normalize_for_match,
    _ProvenanceIndex,
    attribute_provenance,
    build_provenance_index,
    _regions_from_scores,
)
from tokenjam.otel.semconv import GenAIAttributes
from tests.factories import make_llm_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _config(prompts: bool) -> TjConfig:
    return TjConfig(version="1", capture=CaptureConfig(prompts=prompts))


# -- Pure-function tests --

def test_regions_from_scores_finds_low_significance_span():
    """A long run of low-score tokens becomes a region."""
    # "important" gets high score; the 30-char blob of "x" gets low scores.
    text = "important " + ("x" * 30) + " important"
    scores = [
        ("important", 0.9),
    ] + [("x", 0.1)] * 30 + [
        ("important", 0.9),
    ]
    regions = _regions_from_scores(text, scores)
    assert len(regions) == 1
    r = regions[0]
    assert r.char_length >= MIN_REGION_LENGTH
    assert r.avg_score < 0.40


def test_regions_skips_short_low_significance_spans():
    """Short low-score runs aren't worth flagging."""
    text = "important x y important"
    scores = [("important", 0.9), ("x", 0.1), ("y", 0.1), ("important", 0.9)]
    assert _regions_from_scores(text, scores) == []


def test_regions_empty_input():
    assert _regions_from_scores("", []) == []


def test_regions_all_low_significance():
    """An entirely low-significance prompt becomes one region."""
    text = "x" * 50
    scores = [("x", 0.1)] * 50
    regions = _regions_from_scores(text, scores)
    # The trailing-flush branch should produce a single region covering the rest.
    assert len(regions) == 1


def test_significance_threshold_override_flags_previously_significant_span():
    """A 0.5-scored span isn't bloat at the default 0.40 threshold (0.5 is not
    below 0.40); raising `significance_threshold` (what run() threads from
    `[optimize] trim_significance_threshold`) makes the same span count."""
    text = "important " + ("x" * 30) + " important"
    scores = [("important", 0.9)] + [("x", 0.5)] * 30 + [("important", 0.9)]

    assert _regions_from_scores(text, scores) == []
    regions = _regions_from_scores(text, scores, significance_threshold=0.6)
    assert len(regions) == 1
    assert regions[0].char_length >= MIN_REGION_LENGTH


# -- Provenance pure-function tests --

def test_normalize_for_match_collapses_only_whitespace():
    """Runs of spaces/tabs collapse and trailing space is stripped, but word
    content and line order are untouched -- this is a verbatim check, not a
    fuzzy one."""
    raw = "Some   rule.\t\nAnother line.   \n\nThird.\t"
    assert _normalize_for_match(raw) == "Some rule.\nAnother line.\n\nThird."


def test_agent_repo_basename_parses_claude_code_prefix():
    assert _agent_repo_basename("claude-code-tokenjam") == "tokenjam"
    assert _agent_repo_basename("claude-code-") is None       # empty basename
    assert _agent_repo_basename("test-agent") is None          # not a CC agent_id
    assert _agent_repo_basename("") is None


def test_attribute_provenance_matches_verbatim_global_file():
    """A prompt that verbatim-contains a global catalog file's content is
    attributed to it, no project_root or agent_id match needed."""
    rule_text = "Rule: " + ("always be honest. " * 20)  # well over MIN_PROVENANCE_CHARS
    index = _ProvenanceIndex(
        global_candidates=[("/home/user/.claude/CLAUDE.md", _normalize_for_match(rule_text))],
    )
    prompt = f"System prompt preamble.\n\n{rule_text}\n\nUser: hello"
    path, basis = attribute_provenance(prompt, "any-agent-id", index)
    assert path == "/home/user/.claude/CLAUDE.md"
    assert "verbatim match" in basis


def test_attribute_provenance_rejects_paraphrase():
    """Content that merely RESEMBLES a catalog file (reworded, reordered, or
    partially quoted) must not match -- only an exact (whitespace-normalized)
    copy does."""
    rule_text = "Rule: " + ("always be honest. " * 20)
    index = _ProvenanceIndex(
        global_candidates=[("/home/user/.claude/CLAUDE.md", _normalize_for_match(rule_text))],
    )
    paraphrase = "Rule: " + ("be honest at all times. " * 20)
    path, basis = attribute_provenance(paraphrase, "any-agent-id", index)
    assert path is None
    assert basis == ""


def test_attribute_provenance_gates_project_files_on_agent_repo_match():
    """Project-scoped candidates are only checked when the prompt's agent_id
    names the SAME repo as project_root -- a prompt from a different (or
    unknown) repo never gets a project-file match, even if the text is a
    literal copy of that file."""
    file_text = "Project convention: " + ("keep functions small. " * 20)
    index = _ProvenanceIndex(
        project_candidates=[("/repo/CLAUDE.md", _normalize_for_match(file_text))],
        project_root=Path("/somewhere/myrepo"),
    )
    prompt = f"{file_text}\n\nUser turn."

    # Wrong repo named in agent_id -> no match, despite literal containment.
    path, basis = attribute_provenance(prompt, "claude-code-otherrepo", index)
    assert path is None
    assert basis == ""

    # Non-Claude-Code agent_id (no cwd hint at all) -> no match either.
    path, basis = attribute_provenance(prompt, "sdk-agent-7", index)
    assert path is None

    # Matching repo basename -> matches.
    path, basis = attribute_provenance(prompt, "claude-code-myrepo", index)
    assert path == "/repo/CLAUDE.md"


def test_attribute_provenance_no_candidates_returns_none():
    index = _ProvenanceIndex()
    path, basis = attribute_provenance("anything at all", "claude-code-x", index)
    assert path is None
    assert basis == ""


def test_build_provenance_index_excludes_files_below_min_chars(tmp_path):
    """A catalog file whose normalized content is under MIN_PROVENANCE_CHARS
    is never a candidate, however verbatim a "match" might look."""
    tiny = tmp_path / "CLAUDE.md"
    tiny.write_text("short")
    assert len(tiny.read_text()) < MIN_PROVENANCE_CHARS

    index = build_provenance_index(tmp_path)
    assert index.project_candidates == []


def test_build_provenance_index_includes_project_files_above_min_chars(tmp_path):
    long_text = "Project rule.\n" * 50  # comfortably over MIN_PROVENANCE_CHARS
    (tmp_path / "CLAUDE.md").write_text(long_text)

    index = build_provenance_index(tmp_path)
    paths = [p for p, _ in index.project_candidates]
    assert str(tmp_path / "CLAUDE.md") in paths


# -- Integration tests via build_report --

def test_disabled_without_capture_prompts(db):
    """Without capture.prompts the analyzer returns a hint, not a model run."""
    config = _config(prompts=False)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert isinstance(finding, PromptBloatFinding)
    assert finding.enabled is False
    assert "capture" in finding.hint.lower()


def test_disabled_when_llmlingua_missing(db, monkeypatch):
    """Without the bloat extra installed, the analyzer surfaces the install hint."""
    # Force the deferred import to fail by inserting a sentinel that raises.
    monkeypatch.setitem(sys.modules, "llmlingua", None)  # blocks import
    config = _config(prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.enabled is False
    assert "tokenjam[bloat]" in finding.hint


def _install_fake_llmlingua(monkeypatch, token_scores):
    """
    Inject a fake `llmlingua.PromptCompressor` whose
    `get_distillation_token_scores` returns the given (token, score) list.
    """
    fake_module = MagicMock()

    class FakeCompressor:
        def __init__(self, *args, **kwargs):
            pass

        def get_distillation_token_scores(self, text):
            return token_scores

    fake_module.PromptCompressor = FakeCompressor
    monkeypatch.setitem(sys.modules, "llmlingua", fake_module)


def _seed_prompt(db, *, text: str, count: int = 1, start=None, agent_id: str = "test-agent"):
    """Seed N captured-prompt spans."""
    start = start or datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(count):
        span = make_llm_span(
            agent_id=agent_id,
            provider="anthropic",
            billing_account="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=int(len(text) / 4),
            cost_usd=0.001,
            start_time=start + timedelta(minutes=i),
            extra_attributes={GenAIAttributes.PROMPT_CONTENT: text},
        )
        db.insert_span(span)


def test_scores_prompts_and_finds_bloat(db, monkeypatch):
    """When the model is available and prompts captured, the analyzer surfaces bloat."""
    text = "important " + ("filler " * 40) + "important"
    scores = (
        [("important", 0.9)]
        + [("filler", 0.1)] * 40
        + [("important", 0.9)]
    )
    _install_fake_llmlingua(monkeypatch, scores)
    _seed_prompt(db, text=text, count=3)
    config = _config(prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.enabled is True
    assert finding.prompts_scored == 3
    # Each scored prompt produces one BloatPrompt entry, up to the 10 cap.
    assert len(finding.per_prompt) == 3
    p = finding.per_prompt[0]
    assert p.bloat_chars > 0
    assert p.estimated_token_reduction > 0
    assert len(p.regions) >= 1


def test_config_raises_significance_bar_surfaces_previously_hidden_bloat(db, monkeypatch):
    """A prompt whose filler scores 0.5 isn't bloat at the default 0.40
    threshold; raising [optimize] trim_significance_threshold to 0.6 flags it
    on the identical seeded data."""
    text = "important " + ("filler " * 40) + "important"
    scores = (
        [("important", 0.9)]
        + [("filler", 0.5)] * 40
        + [("important", 0.9)]
    )
    _install_fake_llmlingua(monkeypatch, scores)
    _seed_prompt(db, text=text, count=3)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)

    default_report = build_report(db=db, config=_config(prompts=True), since=since,
                                  until=until, findings=["trim"])
    default_finding = default_report.findings["trim"]
    assert default_finding.per_prompt[0].bloat_chars == 0
    assert default_finding.significance_threshold == 0.40

    raised_config = TjConfig(
        version="1", capture=CaptureConfig(prompts=True),
        optimize=OptimizeConfig(trim_significance_threshold=0.6),
    )
    raised_report = build_report(db=db, config=raised_config, since=since,
                                 until=until, findings=["trim"])
    raised_finding = raised_report.findings["trim"]
    assert raised_finding.per_prompt[0].bloat_chars > 0
    assert raised_finding.significance_threshold == 0.6


def test_skips_short_prompts(db, monkeypatch):
    """Prompts under 200 chars are skipped — no model run, no finding."""
    _install_fake_llmlingua(monkeypatch, [("x", 0.5)])
    _seed_prompt(db, text="too short", count=5)
    config = _config(prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.prompts_scored == 0
    assert finding.prompts_skipped == 5
    assert finding.per_prompt == []


# -- Provenance integration tests via build_report --
#
# `find_repo_root`/`load_catalog` are monkeypatched at the call site (rather
# than relying on the real repo checkout pytest happens to run inside) so
# these tests are hermetic: they exercise the exact candidates they assert on,
# never real CLAUDE.md/AGENTS.md files that happen to be nearby on disk.

def test_run_attributes_prompt_to_project_catalog_file(db, monkeypatch, tmp_path):
    """A prompt that verbatim-contains the CURRENT project's CLAUDE.md is
    attributed to it, when the prompt's agent_id names that same repo."""
    rule_text = "Team rule: " + ("write small functions. " * 20)
    (tmp_path / "CLAUDE.md").write_text(rule_text)
    monkeypatch.setattr(trim_module, "find_repo_root", lambda _cwd: tmp_path)

    text = f"System preamble.\n\n{rule_text}\n\nUser: help me."
    _install_fake_llmlingua(monkeypatch, [(text, 0.9)])
    agent_id = f"claude-code-{tmp_path.name.lower()}"
    _seed_prompt(db, text=text, count=1, agent_id=agent_id)

    config = _config(prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.prompts_with_provenance == 1
    assert finding.per_prompt[0].source_path == str(tmp_path / "CLAUDE.md")
    assert "verbatim match" in finding.per_prompt[0].source_basis


def test_run_attributes_prompt_to_global_catalog_file(db, monkeypatch, tmp_path):
    """A prompt containing a GLOBAL catalog file's content is attributed
    regardless of project_root/agent_id -- global files need no cwd hint."""
    from tokenjam.core.summarize.catalog import Catalog

    rule_text = "Global rule: " + ("be terse. " * 20)
    global_file = tmp_path / "CLAUDE.md"
    global_file.write_text(rule_text)
    fake_catalog = Catalog(
        project_files=frozenset(), project_globs=(),
        global_paths=(str(global_file),), forbidden_roots=(),
    )
    monkeypatch.setattr(trim_module, "load_catalog", lambda: fake_catalog)
    monkeypatch.setattr(trim_module, "find_repo_root", lambda _cwd: None)

    text = f"{rule_text}\n\nUser turn."
    _install_fake_llmlingua(monkeypatch, [(text, 0.9)])
    _seed_prompt(db, text=text, count=1, agent_id="some-sdk-agent")

    config = _config(prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.prompts_with_provenance == 1
    assert finding.per_prompt[0].source_path == str(global_file)


def test_run_leaves_unrelated_prompt_without_provenance(db, monkeypatch, tmp_path):
    """The conservative, expected outcome: a prompt that isn't a verbatim
    copy of any catalog file gets no attribution."""
    (tmp_path / "CLAUDE.md").write_text("Team rule: " + ("write small functions. " * 20))
    monkeypatch.setattr(trim_module, "find_repo_root", lambda _cwd: tmp_path)

    text = "Completely unrelated prompt text. " + ("filler words here. " * 20)
    _install_fake_llmlingua(monkeypatch, [(text, 0.9)])
    agent_id = f"claude-code-{tmp_path.name.lower()}"
    _seed_prompt(db, text=text, count=1, agent_id=agent_id)

    config = _config(prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.prompts_with_provenance == 0
    assert finding.per_prompt[0].source_path is None
    assert finding.per_prompt[0].source_basis == ""


def test_run_does_not_attribute_project_file_for_mismatched_repo(db, monkeypatch, tmp_path):
    """A prompt whose agent_id names a DIFFERENT repo than project_root never
    gets a project-file match, even when its text is a literal copy of that
    file -- there is no honest cwd-based evidence linking the two."""
    rule_text = "Team rule: " + ("write small functions. " * 20)
    (tmp_path / "CLAUDE.md").write_text(rule_text)
    monkeypatch.setattr(trim_module, "find_repo_root", lambda _cwd: tmp_path)

    text = f"{rule_text}\n\nUser turn."
    _install_fake_llmlingua(monkeypatch, [(text, 0.9)])
    _seed_prompt(db, text=text, count=1, agent_id="claude-code-some-other-repo")

    config = _config(prompts=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.prompts_with_provenance == 0
    assert finding.per_prompt[0].source_path is None
