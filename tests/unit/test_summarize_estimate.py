"""Unit tests for the per-candidate savings estimate."""
from __future__ import annotations

import json

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.summarize import detect, estimate
from tokenjam.core.summarize.session import results_dir


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
