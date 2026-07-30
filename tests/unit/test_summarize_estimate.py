"""Unit tests for the per-candidate savings estimate."""
from __future__ import annotations

import json

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.summarize import detect, estimate, load_semantics
from tokenjam.core.summarize.session import ATTEMPT_SUFFIX, results_dir


def test_structure_excluded_from_savings():
    # Two files; the second is *longer* but mostly code. Only prose is
    # summarizable, so the code-heavy file's estimated saving is smaller.
    all_prose = "word " * 400
    half_prose_lots_of_code = "word " * 200 + "\n```\n" + "code " * 1200 + "\n```\n"
    saved_prose = estimate.tokens_saved(detect.analyze(all_prose))
    saved_code_heavy = estimate.tokens_saved(detect.analyze(half_prose_lots_of_code))
    assert saved_code_heavy > 0
    assert saved_prose > saved_code_heavy


def test_ratio_bounds_and_monotonic():
    b = detect.analyze("word " * 1000)
    assert estimate.tokens_saved(b, ratio=1.0) == 0
    assert estimate.tokens_saved(b, ratio=1.5) == 0
    # lower target ratio keeps less prose → saves more
    assert estimate.tokens_saved(b, ratio=0.0) > estimate.tokens_saved(b, ratio=0.5)
    assert estimate.tokens_saved(b, ratio=0.5) > estimate.tokens_saved(b, ratio=0.9)


# --------------------------------------------------------------------------- #
# The ratio is an ASK, not a prediction. `observed_prose_ratio` is the only way
# it becomes a measurement — and it must refuse to answer without evidence.
# --------------------------------------------------------------------------- #

@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _stage(cfg, name, *, prose_before, before, after, staged=True):
    d = results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps({
        "path": f"/x/{name}.md", "staged": staged,
        "prose_words_before": prose_before,
        "words_before": before, "words_after": after,
    }), encoding="utf-8")


def test_no_staged_results_means_no_observed_ratio(cfg):
    """No evidence is not a zero and not a nudged constant — the caller must
    fall back to the target and say that it is one."""
    assert estimate.observed_prose_ratio(cfg) == (None, 0)
    assert estimate.observed_prose_ratio(None) == (None, 0)


def test_sample_below_the_credibility_gate_is_refused(cfg):
    """One unusual rewrite must not set the reduction for every file a user
    owns, so the sample size is reported but the ratio is withheld."""
    _stage(cfg, "a", prose_before=4_000, before=5_000, after=2_500)
    ratio, samples = estimate.observed_prose_ratio(cfg)
    assert ratio is None
    assert samples == 1


def test_observed_ratio_is_weighted_by_prose_volume(cfg):
    """Summing both terms before dividing stops a handful of tiny files
    outvoting a large one."""
    _stage(cfg, "a", prose_before=1_000, before=1_200, after=1_100)   # 100 removed
    _stage(cfg, "b", prose_before=1_000, before=1_200, after=1_100)
    _stage(cfg, "c", prose_before=8_000, before=9_000, after=5_000)   # 4000 removed
    ratio, samples = estimate.observed_prose_ratio(cfg)

    assert samples == 3
    assert ratio == pytest.approx(1 - (4_200 / 10_000))


def test_failed_and_legacy_results_are_not_evidence(cfg):
    """A rewrite that failed the structure gate was never a usable outcome, and a
    result staged before the denominator existed carries none — both are skipped
    rather than counted as zero-reduction files, which would drag the ratio."""
    _stage(cfg, "ok1", prose_before=2_000, before=2_400, after=1_400)
    _stage(cfg, "ok2", prose_before=2_000, before=2_400, after=1_400)
    _stage(cfg, "ok3", prose_before=2_000, before=2_400, after=1_400)
    _stage(cfg, "failed", prose_before=9_000, before=9_000, after=9_000, staged=False)
    _stage(cfg, "legacy", prose_before=0, before=9_000, after=9_000)

    ratio, samples = estimate.observed_prose_ratio(cfg)

    assert samples == 3
    assert ratio == pytest.approx(1 - (3_000 / 6_000))


def test_a_rewrite_that_grew_the_file_never_reads_as_negative_saving(cfg):
    for n in ("a", "b", "c"):
        _stage(cfg, n, prose_before=1_000, before=1_000, after=1_200)
    ratio, _ = estimate.observed_prose_ratio(cfg)
    assert ratio == 1.0                      # no reduction, never below zero


# --------------------------------------------------------------------------- #
# A rewrite that failed the structure gate is EVIDENCE, just not ratio evidence:
# it stays out of the measurement and is still counted, so a sample made mostly
# of failures cannot read as a clean measurement.
# --------------------------------------------------------------------------- #

def _attempt(cfg, name):
    d = results_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}{ATTEMPT_SUFFIX}").write_text(json.dumps({
        "path": f"/x/{name}.md", "staged": False, "structure_ok": False,
        "record_kind": "attempt", "reason": "dropped blocks ['3']",
    }), encoding="utf-8")


def test_gate_failures_are_counted_but_never_enter_the_ratio(cfg):
    _stage(cfg, "ok1", prose_before=2_000, before=2_400, after=1_400)
    _stage(cfg, "ok2", prose_before=2_000, before=2_400, after=1_400)
    _stage(cfg, "ok3", prose_before=2_000, before=2_400, after=1_400)
    _attempt(cfg, "bad1")
    _attempt(cfg, "bad2")

    ratio, samples = estimate.observed_prose_ratio(cfg)

    assert samples == 3                              # attempts are not samples
    assert ratio == pytest.approx(1 - (3_000 / 6_000))
    assert estimate.gate_failed_attempts(cfg) == 2
    assert estimate.gate_failed_attempts(None) == 0


# --------------------------------------------------------------------------- #
# The published line target: an ASK, applied only where the guidance applies,
# and structurally unable to inflate a figure.
# --------------------------------------------------------------------------- #

def test_line_target_applies_only_to_an_oversized_always_resident_file():
    big = "word " * 40 + "\n"
    text = big * 400                                  # ~400 lines, all prose

    budget = estimate.line_target_prose_words(
        text=text, load_class=load_semantics.ALWAYS, prose_words=16_000)
    assert budget is not None
    assert budget < 16_000

    # An on-demand file: the published guidance is about a file that loads every
    # session, so it is not restated as if it covered a skill body.
    assert estimate.line_target_prose_words(
        text=text, load_class=load_semantics.SKILL, prose_words=16_000) is None

    # Already under the target: nothing to aim at, and NOT a claim that
    # compressing it is worthless.
    small = "word " * 40 + "\n"
    assert estimate.line_target_prose_words(
        text=small * 10, load_class=load_semantics.ALWAYS, prose_words=400) is None


def test_inline_code_does_not_make_its_line_unremovable():
    """An inline span must survive, but it packs into a much shorter sentence,
    so its line is compressible prose. Counting it as a protected LINE withheld
    the line target from exactly the backtick-heavy CLAUDE.md files it is for."""
    line = "Run the `pnpm run dev` command and never skip `pnpm run doctor`.\n"
    text = line * 300

    lb = detect.line_breakdown(text)

    assert lb.total_lines == 300
    assert lb.protected_lines == 0
    assert estimate.line_target_prose_words(
        text=text, load_class=load_semantics.ALWAYS, prose_words=3_000) is not None


def test_a_fenced_block_does_pin_every_line_it_covers():
    """A multi-line span is restored verbatim, so it really does hold its lines."""
    text = "prose line\n" * 10 + "```\n" + "code = 1\n" * 50 + "```\n"

    lb = detect.line_breakdown(text)

    assert lb.protected_lines == 52          # the fences plus their body
    assert lb.prose_lines == 10
    assert lb.protected_lines + lb.prose_lines == lb.total_lines


def test_line_target_refuses_when_structure_alone_exceeds_it():
    """A target the fix structurally cannot reach is an error the user can
    disprove, not a best case — so it is withheld rather than asserted."""
    code = "```\n" + "code = 1\n" * 400 + "```\n"
    text = code + "word " * 300

    assert estimate.line_target_prose_words(
        text=text, load_class=load_semantics.ALWAYS, prose_words=300) is None


def test_line_target_never_changes_what_is_claimed():
    """`tokens_saved` prices off the ratio alone. Whatever the rewriter is asked
    for, the figure stays bounded by what a rewrite is measured to deliver."""
    text = ("word " * 40 + "\n") * 600         # 3x the target: a far harsher ask
    breakdown = detect.analyze(text)
    budget = estimate.line_target_prose_words(
        text=text, load_class=load_semantics.ALWAYS, prose_words=breakdown.prose_words)
    assert budget is not None                              # the aggressive ask exists

    implied_ratio = budget / breakdown.prose_words
    assert implied_ratio < 0.5                             # ...and is more aggressive
    # The claim is unmoved by it: only `ratio` is a lever on the estimate, and
    # the estimate's default is the measured PRIOR, never the ask. (Inverted
    # from an earlier version that pinned default == DEFAULT_TARGET_RATIO —
    # estimating at the ask is the defect, so the guard now defends against it.)
    assert estimate.tokens_saved(breakdown) == estimate.tokens_saved(
        breakdown, ratio=estimate.UNMEASURED_PRIOR_RATIO)
    assert estimate.tokens_saved(breakdown) < estimate.tokens_saved(
        breakdown, ratio=estimate.DEFAULT_TARGET_RATIO)


def test_the_estimate_default_is_the_prior_not_the_ask():
    """Estimating at the ratio the rewriter is merely ASKED for overstated the
    figure by roughly an order of magnitude. The two constants must stay
    distinct, and the conservative one must be what estimates use."""
    assert estimate.UNMEASURED_PRIOR_RATIO != estimate.DEFAULT_TARGET_RATIO
    assert estimate.UNMEASURED_PRIOR_RATIO > estimate.DEFAULT_TARGET_RATIO   # less flattering
    lo, hi = estimate.UNMEASURED_PRIOR_RANGE
    assert lo < estimate.UNMEASURED_PRIOR_RATIO <= hi     # the prior sits inside its sample
    assert estimate.UNMEASURED_PRIOR_SAMPLES >= estimate.MIN_OBSERVED_SAMPLES


# --------------------------------------------------------------------------- #
# Reflow is not compression. A raw character delta booked un-hard-wrapping a
# file as a saving; on real instruction files that artifact was the majority of
# the claimed reduction.
# --------------------------------------------------------------------------- #

def test_reflowing_a_hard_wrapped_file_is_never_a_saving():
    hard_wrapped = "\n".join("word " * 8 for _ in range(60))
    reflowed = " ".join(["word"] * 480)

    # A raw length delta calls this a saving. It is not one: same words.
    assert len(hard_wrapped) > len(reflowed)
    assert detect.content_chars(hard_wrapped) == detect.content_chars(reflowed)


def test_tokens_saved_is_measured_on_content_not_raw_characters():
    """Two files with identical content and different wrapping must estimate
    identically, or the wrapping itself is being sold as compressible."""
    wrapped = detect.analyze("\n".join("word " * 8 for _ in range(60)))
    flowed = detect.analyze(" ".join(["word"] * 480))

    assert wrapped.total_chars != flowed.total_chars       # they differ on disk
    assert estimate.tokens_saved(wrapped, ratio=0.5) == estimate.tokens_saved(
        flowed, ratio=0.5)
