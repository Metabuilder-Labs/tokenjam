"""Unit tests for Reuse skeleton rendering + Markdown sidecar (issue #116)."""
from __future__ import annotations

import pytest

from tokenjam.core.config import TjConfig
from tokenjam.core.export.reuse_report import _render_markdown, prepare_renders
from tokenjam.core.export.reuse_skeleton import (
    MAX_SLOTS,
    is_weak_match,
    render_skeleton,
)
from tokenjam.core.optimize.types import ReuseCluster, ReuseFinding


def _cluster(cluster_id="abc123", sig=("read", "edit")) -> ReuseCluster:
    return ReuseCluster(
        cluster_id=cluster_id,
        tool_signature=sig,
        prompt_prefix_hash=None,
        repetitions=4,
        avg_planning_tokens=1200,
        avg_planning_cost_usd=0.20,
        cache_reuse_recoverable_usd=0.60,
        script_replacement_recoverable_usd=0.80,
        cache_reuse_recoverable_tokens=3600,
        script_replacement_recoverable_tokens=4800,
        example_session_ids=["s2", "s1", "s0"],
        skeleton_session_id="s2",
    )


# -- prepare_renders source guard (#154 review) --

def test_prepare_renders_requires_conn_or_planning_texts(tmp_path):
    # Passing neither source is a programming error — guarded loudly instead of
    # silently rendering skeleton-less clusters or crashing in the DB fetch.
    with pytest.raises(ValueError, match="conn or planning_texts"):
        prepare_renders(
            ReuseFinding(clusters=[_cluster()]),
            config=TjConfig(version="1"),
            out_dir=tmp_path,
            version="0.0.0",
            generated_at_iso="2026-06-19T00:00:00+00:00",
        )


# -- render_skeleton --

def test_identical_tokens_preserved_divergent_replaced():
    skeleton, slot_map = render_skeleton(
        "release v1 on monday",
        ["release v2 on monday", "release v3 on monday"],
    )
    assert skeleton == "release {{slot_1}} on monday"
    assert slot_map == {"slot_1": ["v1", "v2", "v3"]}


def test_slot_values_are_sorted_and_deduped():
    skeleton, slot_map = render_skeleton(
        "deploy to prod now",
        ["deploy to prod now", "deploy to staging now", "deploy to staging now"],
    )
    assert skeleton == "deploy to {{slot_1}} now"
    assert slot_map["slot_1"] == ["prod", "staging"]


def test_no_examples_keeps_everything_literal():
    skeleton, slot_map = render_skeleton("just a plan", [])
    assert skeleton == "just a plan"
    assert slot_map == {}


def test_shorter_example_makes_position_divergent():
    # The second example has no token at the final position → divergent slot.
    skeleton, slot_map = render_skeleton(
        "build then test",
        ["build then test", "build then"],
    )
    assert skeleton == "build then {{slot_1}}"
    assert slot_map == {"slot_1": ["test"]}


def test_slot_cap_and_weak_match():
    n = MAX_SLOTS + 5
    plan = " ".join(f"w{i}" for i in range(n))
    other = " ".join(f"x{i}" for i in range(n))
    skeleton, slot_map = render_skeleton(plan, [other])
    assert len(slot_map) == MAX_SLOTS
    assert "{{…}}" in skeleton          # overflow positions collapse
    assert is_weak_match(slot_map) is True


def test_strong_match_is_not_weak():
    _, slot_map = render_skeleton("a b c", ["a x c"])
    assert is_weak_match(slot_map) is False


# -- Markdown sidecar --

def test_markdown_frontmatter_has_required_keys():
    md = _render_markdown(
        _cluster(), "release {{slot_1}} now", {"slot_1": ["v1", "v2"]},
        version="9.9.9", generated_at_iso="2026-06-18T00:00:00+00:00",
    )
    for key in ("cluster_id:", "tool_signature:", "repetitions:",
                "generated_at:", "tokenjam_version:"):
        assert key in md
    assert "abc123" in md
    assert "9.9.9" in md
    assert "{{slot_1}}" in md
    # Honesty caveat present.
    assert "Review before reusing" in md


def test_markdown_is_reproducible_for_same_input():
    args = dict(
        version="1.0.0", generated_at_iso="2026-06-18T00:00:00+00:00",
    )
    a = _render_markdown(_cluster(), "x {{slot_1}}", {"slot_1": ["a"]}, **args)
    b = _render_markdown(_cluster(), "x {{slot_1}}", {"slot_1": ["a"]}, **args)
    assert a == b
    # cluster_id drives the sidecar filename → must appear verbatim.
    assert "cluster_id: abc123" in a
