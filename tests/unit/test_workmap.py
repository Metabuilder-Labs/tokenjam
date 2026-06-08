"""Unit tests for the pure work-map transform (core/workmap.py).

The work map folds an ask-segmented session story (transcript-derived structure
+ labels) with the per-subagent cost breakdown (span-derived) into a list of ask
nodes, newest first. These tests drive it with hand-built dicts — no I/O.
"""
from __future__ import annotations

from tokenjam.core.workmap import build_work_map


def _tool(name: str, label: str = "", status: str = "ok") -> dict:
    return {"name": name, "label": label, "status": status}


def _step(tools: list[dict], **kw) -> dict:
    step = {"n": 1, "ts": None, "text": kw.pop("text", ""), "tools": tools,
            "is_error": False, "is_retry": kw.pop("is_retry", False),
            "model": kw.pop("model", "claude-opus-4-8")}
    step.update(kw)
    return step


def _ask(n: int, prompt: str, steps: list[dict], **kw) -> dict:
    return {"n": n, "prompt": prompt, "ts": kw.get("ts"),
            "step_count": len(steps), "truncated": kw.get("truncated", False),
            "steps": steps, "outcome": kw.get("outcome", "")}


def test_asks_listed_newest_first_with_rollup_and_tokens():
    asks = {"asks": [
        _ask(1, "First ask: read", [_step([_tool("Read", "a.py"),
                                           _tool("WebSearch", "q")])]),
        _ask(2, "Second ask: build", [_step([_tool("Edit", "b.py"),
                                             _tool("Bash", "pytest")])]),
    ]}
    m = build_work_map(asks, None, ask_tokens={1: 1000, 2: 2000})

    assert m["ask_count"] == 2
    # newest first
    assert m["asks"][0]["n"] == 2
    assert m["asks"][0]["prompt"] == "Second ask: build"
    assert m["asks"][0]["tokens"] == 2000
    assert m["asks"][1]["n"] == 1
    assert m["asks"][1]["tokens"] == 1000
    # per-ask rollup (the first ask, now at index 1)
    act = m["asks"][1]["activity"]
    assert act["file_count"] == 1
    assert act["source_count"] == 1


def test_subagent_attributed_to_its_ask_with_cost():
    asks = {"asks": [
        _ask(1, "orchestrate", [
            _step([_tool("Task", "worker")], subagent={
                "agent_id": "A", "name": "worker",
                "task": "build", "outcome": "done",
                "steps": [_step([_tool("Edit", "x.py")])],
            }),
        ]),
    ]}
    subagents = {"rows": [{
        "sub_agent_id": "A", "model": "claude-opus-4-8",
        "input_tokens": 80_000, "output_tokens": 100,
        "cache_tokens": 0, "cache_write_tokens": 0,
        "cost_usd": 0.60, "flags": ["over_provisioned"],
    }]}

    m = build_work_map(asks, subagents)
    ask = m["asks"][0]
    assert ask["subagent_count"] == 1
    assert ask["flagged"] == 1
    child = ask["subagents"][0]
    assert child["name"] == "worker"
    assert child["tokens"] == 80_100
    assert child["flags"] == ["over_provisioned"]
    assert m["subagent_count"] == 1
    assert m["flagged"] == 1


def test_capped_subagent_still_surfaces_cost():
    asks = {"asks": [_ask(1, "x", [
        _step([_tool("Task", "deep")],
              subagent={"agent_id": "D", "name": "deep", "depth_capped": True}),
    ])]}
    subagents = {"rows": [{
        "sub_agent_id": "D", "model": "claude-haiku-4-5",
        "input_tokens": 10, "output_tokens": 10,
        "cache_tokens": 0, "cache_write_tokens": 0,
        "cost_usd": 0.01, "flags": [],
    }]}

    m = build_work_map(asks, subagents)
    child = m["asks"][0]["subagents"][0]
    assert child["capped"] == "depth"
    assert child["cost_usd"] == 0.01
    assert m["capped"] == 1


def test_unmapped_subagents_reported_not_dropped():
    asks = {"asks": [_ask(1, "x", [_step([_tool("Read", "f.py")])])]}
    subagents = {"rows": [{
        "sub_agent_id": "Z", "model": "m",
        "input_tokens": 4000, "output_tokens": 1000,
        "cache_tokens": 0, "cache_write_tokens": 0,
        "cost_usd": 2.0, "flags": [],
    }]}

    m = build_work_map(asks, subagents)
    assert m["subagent_count"] == 0
    assert m["unmapped_count"] == 1
    assert m["unmapped_tokens"] == 5000


def test_empty_session_has_no_asks():
    m = build_work_map({"asks": []}, None, session_tokens=0)
    assert m["ask_count"] == 0
    assert m["asks"] == []
    assert m["unmapped_count"] == 0


def test_file_list_capped_but_count_exact():
    tools = [_tool("Read", f"f{i}.py") for i in range(10)]
    m = build_work_map({"asks": [_ask(1, "x", [_step(tools)])]}, None)
    act = m["asks"][0]["activity"]
    assert act["file_count"] == 10
    assert len(act["files"]) == 8       # MAX_FILES_PER_NODE


def test_session_totals_passed_through():
    m = build_work_map({"asks": []}, None, session_tokens=999, session_cost_usd=1.5)
    assert m["session_tokens"] == 999
    assert m["session_cost_usd"] == 1.5
