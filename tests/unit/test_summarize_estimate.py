"""Unit tests for the per-candidate savings estimate."""
from __future__ import annotations

from tokenjam.core.summarize import detect, estimate


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
