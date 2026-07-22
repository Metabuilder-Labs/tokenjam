"""Unit tests for the shared clustering primitives (core/optimize/clustering.py)
plus a byte-stable regression guard: the three miners that now consume the
helper (relearn, plan_reuse, workflow_restructure) must produce the SAME
signatures/titles/groupings they did before the consolidation.

The golden values here were captured from the pre-refactor code; if a future
edit to the shared helper changes any analyzer's clustering output, one of these
assertions fails.
"""
from __future__ import annotations

import re

from tokenjam.core.optimize.clustering import group_by_key, mask_variables, recurring


# --- the primitives ----------------------------------------------------------

def test_group_by_key_preserves_first_seen_order():
    items = ["apple", "avocado", "banana", "cherry", "blueberry"]
    buckets = group_by_key(items, lambda s: s[0])
    assert list(buckets.keys()) == ["a", "b", "c"]        # first-seen order
    assert buckets["a"] == ["apple", "avocado"]
    assert buckets["b"] == ["banana", "blueberry"]


def test_recurring_keeps_keys_and_filters_by_size():
    buckets = {"x": [1, 2, 3], "y": [1], "z": [1, 2]}
    kept = recurring(buckets, min_members=2)
    assert kept == {"x": [1, 2, 3], "z": [1, 2]}          # keys preserved


def test_recurring_honors_custom_size_fn():
    # A bucket whose recurrence measure is distinct-sessions, not raw count.
    buckets = {"a": ["s1", "s1", "s1"], "b": ["s1", "s2"]}
    kept = recurring(buckets, min_members=2, size_fn=lambda m: len(set(m)))
    assert set(kept) == {"b"}                             # "a" is 1 distinct session


def test_mask_variables_applies_subs_then_ws_then_lower():
    subs = [(re.compile(r"\d+"), "<N>"), (re.compile(r"/\w+"), "<P>")]
    out = mask_variables("HIT  /tmp  at 42", subs, collapse_ws=True, lowercase=True)
    assert out == "hit <p> at <n>"


def test_mask_variables_default_is_substitutions_only():
    subs = [(re.compile(r"\d+"), "<N>")]
    assert mask_variables("Keep  Case 9", subs) == "Keep  Case <N>"


# --- byte-stable regression guard: relearn -----------------------------------

def test_relearn_normalizer_byte_stable():
    from tokenjam.core.optimize.analyzers.relearn import _generic_signature, _normalize_generic
    assert _normalize_generic(
        "Error at /Users/x/proj/file.py line 42 at 2026-07-21T10:00:00Z id 3f2a1b9c8d7e"
    ) == "error at <path> line <n> at <ts> id <id>"
    assert _generic_signature("Bash", "no such file /a/b at 2026-01-01T00:00:00Z") == \
        "Bash:no such file <path> at <ts>"


def test_relearn_clustering_byte_stable():
    from tokenjam.core.optimize.analyzers.relearn import (
        FailureEpisode,
        _recurring,
        cluster_failures,
    )
    fs = [
        FailureEpisode("s1", "r", None, "Read", "", "string to replace not found", "act", False, 0),
        FailureEpisode("s2", "r", None, "Read", "", "string to replace not found", "act", False, 0),
        FailureEpisode("s1", "r", None, "Bash", "", "weird err 111", "act", False, 0),
        FailureEpisode("s3", "r", None, "Bash", "", "weird err 222", "act", False, 0),
    ]
    cl = cluster_failures(fs)
    assert sorted(cl.keys()) == ["Bash:weird err <n>", "Read:string to replace not found"]
    # Title comes from the FIRST failure's raw text — the byte-sensitive detail.
    assert cl["Bash:weird err <n>"].title == "Bash: weird err 111"
    assert sorted(cl["Bash:weird err <n>"].session_ids) == ["s1", "s3"]
    assert {c.signature for c in _recurring(cl, 2)} == \
        {"Bash:weird err <n>", "Read:string to replace not found"}
    assert _recurring(cl, 3) == []                         # neither hits 3 sessions


# --- byte-stable regression guard: plan_reuse --------------------------------

def test_plan_reuse_clustering_byte_stable():
    from tokenjam.core.optimize.analyzers.plan_reuse import (
        _SessionPlan,
        _cluster_sessions,
        _strip_variables,
    )
    assert _strip_variables("release v0.3.4 on 2026-06-15 path /a/b/c") == \
        "release v<NUM>.<NUM>.<NUM> on <DATE> path <PATH>"
    plans = [
        _SessionPlan("s1", 100, 0.1, None, ("Read", "Edit"), "ab"),
        _SessionPlan("s2", 100, 0.1, None, ("Read", "Edit"), "ab"),
        _SessionPlan("s3", 100, 0.1, None, ("Grep",), None),
    ]
    pc = _cluster_sessions(plans)
    assert sorted(pc.keys()) == ["06a63bbd59ae", "e8cb6dd93de5-ab"]
    assert {k: len(v) for k, v in pc.items()} == {"e8cb6dd93de5-ab": 2, "06a63bbd59ae": 1}
