"""Unit tests for deterministic phase segmentation (core.phases, Task E)."""
from __future__ import annotations

from tokenjam.core.phases import (
    MAX_PHASES,
    MIN_STEPS_TO_SEGMENT,
    segment_phases,
)


def _text(t):
    return {"text": t, "tools": []}


def _tool(name, status="ok"):
    return {"text": "", "tools": [{"name": name, "label": "", "status": status}]}


def test_short_ask_is_not_segmented():
    steps = [_text("Just did one thing."), _tool("Bash")]
    assert segment_phases(steps) == []


def test_single_phase_returns_empty():
    # Enough steps to consider, but only one narration -> not a journey.
    steps = [_text("Doing the whole thing in one breath.")] + [_tool("Bash")] * 6
    assert segment_phases(steps) == []


def test_segments_by_narration_and_tallies_tools():
    steps = [
        _text("Explored the adapter."), _tool("Bash"), _tool("Read"), _tool("Read"),
        _text("Wrote the fix."), _tool("Edit"), _tool("Edit"),
        _text("Ran validation."), _tool("Bash"),
    ]
    phases = segment_phases(steps)
    assert [p["title"] for p in phases] == [
        "Explored the adapter.", "Wrote the fix.", "Ran validation.",
    ]
    # First phase tallies its trailing tool-only steps.
    assert phases[0]["tools"] == [
        {"name": "Bash", "count": 1}, {"name": "Read", "count": 2}
    ]
    assert phases[0]["tool_count"] == 3


def test_first_sentence_is_used_as_title():
    steps = [
        _text("Let me research the auth model. It is unusual and needs care."),
        _tool("WebFetch"), _tool("WebFetch"), _tool("Read"),
        _text("Now I'll apply the change across both call sites."),
        _tool("Edit"), _tool("Edit"),
    ]
    titles = [p["title"] for p in segment_phases(steps)]
    assert titles[0] == "Let me research the auth model."


def test_error_count_per_phase():
    steps = [
        _text("Tried a thing."), _tool("Bash", status="error"), _tool("Bash"),
        _tool("Bash"), _tool("Bash"),
        _text("Tried another."), _tool("Read"),
    ]
    phases = segment_phases(steps)
    assert phases[0]["error_count"] == 1
    assert phases[1]["error_count"] == 0


def test_caps_with_omitted_marker_no_silent_drop():
    # More phases than the hard cap -> head+tail with an omitted marker.
    total = MAX_PHASES + 12
    steps = []
    for i in range(total):
        steps.append(_text(f"Phase {i}."))
        steps.append(_tool("Bash"))
    phases = segment_phases(steps)
    omitted = [p for p in phases if "omitted" in p]
    assert len(omitted) == 1
    assert len(phases) <= MAX_PHASES + 1  # head + tail + the marker
    # The omitted count accounts for every dropped phase (nothing silently lost).
    real = [p for p in phases if "omitted" not in p]
    assert len(real) + omitted[0]["omitted"] == total


def test_under_cap_sends_full_arc_no_omitted():
    # A long-but-reasonable ask sends every phase (no omitted marker) so "show
    # all" can reveal the whole journey.
    steps = []
    for i in range(30):
        steps.append(_text(f"Phase {i}."))
        steps.append(_tool("Bash"))
    phases = segment_phases(steps)
    assert all("omitted" not in p for p in phases)
    assert len(phases) == 30


def test_omitted_marker_skipped_in_input():
    # A cap marker from the upstream step list must not become a phase.
    steps = [_text("A.")] + [{"omitted": 3}] + [_tool("Bash")] * 6 + [_text("B."), _tool("Read")]
    phases = segment_phases(steps)
    assert [p["title"] for p in phases] == ["A.", "B."]
    assert phases[0]["tools"] == [{"name": "Bash", "count": 6}]


def test_min_steps_constant_guards_tiny_asks():
    steps = [_text("x"), _tool("Bash")] * (MIN_STEPS_TO_SEGMENT // 2 - 1)
    assert segment_phases(steps) == []
