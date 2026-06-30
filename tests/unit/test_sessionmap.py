"""Unit tests for the pure session-map transform (core/sessionmap.py).

``build_session_map`` folds an ask-segmented session story (transcript-derived
structure + labels) into a flat, time-ordered tool-event list plus contiguous
phase spans. These tests drive it with hand-built dicts — no I/O — mirroring the
``_step`` / ``_tool`` factories in test_workmap.py.
"""
from __future__ import annotations

from tokenjam.core.sessionmap import build_session_map


def _tool(name: str, label: str = "", status: str = "ok") -> dict:
    return {"name": name, "label": label, "status": status}


def _step(tools: list[dict], **kw) -> dict:
    step = {"n": 1, "ts": kw.pop("ts", None), "text": kw.pop("text", ""),
            "tools": tools, "is_error": False,
            "is_retry": kw.pop("is_retry", False),
            "model": kw.pop("model", "claude-opus-4-8")}
    step.update(kw)
    return step


def _ask(n: int, steps: list[dict], **kw) -> dict:
    return {"n": n, "prompt": kw.get("prompt", "p"), "ts": kw.get("ts"),
            "step_count": len(steps), "truncated": kw.get("truncated", False),
            "steps": steps, "outcome": kw.get("outcome", "")}


def test_one_event_per_tool_with_category_mapping():
    """Each tool becomes one event; every category is covered, error wins."""
    tools = [
        _tool("Read", "a.py"),          # read
        _tool("Edit", "b.py"),          # edit
        _tool("Write", "c.py"),         # edit
        _tool("Grep", "needle"),        # search
        _tool("Bash", "ls"),            # bash
        _tool("Task", "worker"),        # task
        _tool("WebFetch", "https://x"),  # web
        _tool("Frobnicate", "z"),       # other (unknown tool)
        _tool("Read", "d.py", status="error"),  # error (overrides read)
    ]
    m = build_session_map({"asks": [_ask(1, [_step(tools)])]})

    cats = [e["category"] for e in m["events"]]
    assert cats == [
        "read", "edit", "edit", "search", "bash", "task", "web", "other", "error",
    ]
    # Labels ride through verbatim from the tool.
    assert m["events"][0]["label"] == "a.py"


def test_ordinals_monotonic_and_session_global():
    """Ordinals are a single 0-based monotonic sequence across asks + steps."""
    asks = {"asks": [
        _ask(1, [_step([_tool("Read", "a")]), _step([_tool("Bash", "b")])]),
        _ask(2, [_step([_tool("Edit", "c"), _tool("Edit", "d")])]),
    ]}
    m = build_session_map(asks)
    assert [e["ordinal"] for e in m["events"]] == [0, 1, 2, 3]


def test_omitted_and_toolless_steps_skipped():
    """``{"omitted": N}`` markers and tool-less (narration-only) steps emit no
    events, and ordinals stay gap-free."""
    steps = [
        _step([_tool("Read", "a")]),
        {"omitted": 5},
        _step([], text="just narration, no tools"),
        _step([_tool("Bash", "b")]),
    ]
    m = build_session_map({"asks": [_ask(1, steps)]})
    assert len(m["events"]) == 2
    assert [e["category"] for e in m["events"]] == ["read", "bash"]
    assert [e["ordinal"] for e in m["events"]] == [0, 1]


def test_event_carries_step_ts_and_retry():
    """Every event from a step inherits that step's ts + is_retry flag."""
    s = _step([_tool("Read", "a"), _tool("Read", "a")],
              ts="2026-06-30T09:00:00.000Z", is_retry=True)
    m = build_session_map({"asks": [_ask(1, [s])]})
    assert all(e["ts"] == "2026-06-30T09:00:00.000Z" for e in m["events"])
    assert all(e["is_retry"] is True for e in m["events"])


def test_missing_step_ts_falls_back_to_null():
    """A step without a ts yields events with ts=None (ordinal is the axis)."""
    m = build_session_map({"asks": [_ask(1, [_step([_tool("Read", "a")])])]})
    assert m["events"][0]["ts"] is None
    assert m["events"][0]["ordinal"] == 0


def test_phase_labelling_and_spans():
    """A long narrated ask labels each event with its phase and yields one
    contiguous span per phase."""
    narrations = [
        ("Explored the adapter.", "Bash"),
        ("Read the SSH client.", "Read"),
        ("Researched the auth model.", "Read"),
        ("Wrote the fix.", "Edit"),
        ("Ran end-to-end validation.", "Bash"),
        ("Cleaned up and verified.", "Bash"),
    ]
    steps = [_step([_tool(tool, "x")], text=text) for text, tool in narrations]
    m = build_session_map({"asks": [_ask(1, steps)]})

    # Each event tagged with its step's phase (the agent's first sentence).
    assert m["events"][0]["phase"] == "Explored the adapter."
    assert m["events"][-1]["phase"] == "Cleaned up and verified."

    names = [p["name"] for p in m["phases"]]
    assert names[0] == "Explored the adapter."
    assert "Cleaned up and verified." in names
    # Six single-event phases -> ordinals 0..5, each its own span.
    assert m["phases"][0]["start_ordinal"] == 0
    assert m["phases"][0]["end_ordinal"] == 0
    assert m["phases"][-1]["start_ordinal"] == 5
    assert m["phases"][-1]["end_ordinal"] == 5


def test_phase_span_merges_contiguous_tool_only_steps():
    """A narrated step's phase absorbs the tool-only steps that follow it, so a
    phase span covers all their events."""
    steps = [
        _step([_tool("Bash", "a")], text="Explored the adapter."),
        _step([_tool("Bash", "b")]),                       # attaches to phase 0
        _step([_tool("Read", "c")], text="Read the client."),
        _step([_tool("Read", "d")]),                       # attaches to phase 1
        _step([_tool("Edit", "e")], text="Wrote the fix."),
        _step([_tool("Bash", "f")], text="Verified."),
    ]
    m = build_session_map({"asks": [_ask(1, steps)]})
    first = m["phases"][0]
    assert first["name"] == "Explored the adapter."
    assert first["start_ordinal"] == 0
    assert first["end_ordinal"] == 1        # both Bash events fold in


def test_phase_ts_range_carried_when_present():
    """Phase spans carry start/end ts drawn from their events when available."""
    steps = [
        _step([_tool("Bash", "a")], text="Explored.", ts="2026-06-30T09:00:00Z"),
        _step([_tool("Bash", "b")], ts="2026-06-30T09:00:05Z"),
        _step([_tool("Read", "c")], text="Read.", ts="2026-06-30T09:00:10Z"),
        _step([_tool("Edit", "d")], text="Wrote.", ts="2026-06-30T09:00:15Z"),
        _step([_tool("Bash", "e")], text="Ran.", ts="2026-06-30T09:00:20Z"),
        _step([_tool("Bash", "f")], text="Done.", ts="2026-06-30T09:00:25Z"),
    ]
    m = build_session_map({"asks": [_ask(1, steps)]})
    first = m["phases"][0]
    assert first["start_ts"] == "2026-06-30T09:00:00Z"
    assert first["end_ts"] == "2026-06-30T09:00:05Z"


def test_short_ask_has_no_phase_labels():
    """An ask too short to segment leaves every event unlabelled and yields no
    phase spans (one phase is not a journey)."""
    steps = [_step([_tool("Read", "a")]), _step([_tool("Bash", "b")])]
    m = build_session_map({"asks": [_ask(1, steps)]})
    assert all(e["phase"] is None for e in m["events"])
    assert m["phases"] == []


def test_phases_not_merged_across_asks():
    """Two segmented asks keep their phase spans separate even if a phase title
    repeats at the boundary."""
    narr = [
        ("Explored.", "Bash"), ("Read.", "Read"), ("Researched.", "Read"),
        ("Wrote.", "Edit"), ("Validated.", "Bash"), ("Wrapped up.", "Bash"),
    ]
    steps_a = [_step([_tool(t, "x")], text=txt) for txt, t in narr]
    steps_b = [_step([_tool(t, "y")], text=txt) for txt, t in narr]
    m = build_session_map({"asks": [_ask(1, steps_a), _ask(2, steps_b)]})
    # 6 phases per ask, kept distinct -> 12 spans total.
    assert len(m["phases"]) == 12
    # Ordinals continue across the ask boundary.
    assert m["phases"][6]["start_ordinal"] == 6


def test_empty_asks_yield_empty():
    assert build_session_map({"asks": []}) == {"events": [], "phases": []}
    assert build_session_map({}) == {"events": [], "phases": []}
