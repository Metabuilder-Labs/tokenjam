"""Unit tests for the pure work-map transform (core/workmap.py).

The work map folds a session *story* (transcript-derived structure + labels)
with the per-subagent cost breakdown (span-derived) into one annotated tree.
These tests drive it with hand-built story/breakdown dicts — no I/O, no spans.
"""
from __future__ import annotations

from tokenjam.core.workmap import build_work_map


def _tool(name: str, label: str = "", status: str = "ok") -> dict:
    return {"name": name, "label": label, "status": status}


def _step(tools: list[dict], model: str = "claude-opus-4-8", **kw) -> dict:
    base = {"n": 1, "ts": None, "text": "", "tools": tools,
            "is_error": False, "is_retry": False, "model": model}
    base.update(kw)
    return base


def test_root_activity_rollup_dedups_files_and_counts_sources():
    story = {
        "task": "Do the thing.", "outcome": "Done.", "truncated": False,
        "steps": [
            _step([_tool("Read", "src/a.py"), _tool("Read", "src/a.py"),
                   _tool("WebSearch", "duckdb upsert")]),
            _step([_tool("WebFetch", "https://example.com"),
                   _tool("Grep", "TODO")]),
        ],
    }
    m = build_work_map(story, None, root_cost_usd=1.5, root_tokens=90_000)

    root = m["root"]
    assert root["id"] == "main"
    assert root["is_root"] is True
    assert root["cost_usd"] == 1.5
    assert root["tokens"] == 90_000
    act = root["activity"]
    assert act["file_count"] == 1          # a.py counted once
    assert act["source_count"] == 2        # 1 search + 1 fetch, distinct
    assert act["search_count"] == 1        # grep
    assert act["steps"] == 2
    assert "2 sources" in root["summary"]
    assert "1 file" in root["summary"]
    assert m["node_count"] == 1
    assert m["subagent_count"] == 0


def test_subagent_node_joins_cost_tokens_and_flags():
    story = {
        "task": "Orchestrate.", "outcome": "All done.", "truncated": False,
        "steps": [
            _step(
                [_tool("Task", "build-it")],
                subagent={
                    "agent_id": "A", "name": "build-it",
                    "task": "Build it.", "outcome": "Built.", "truncated": False,
                    "steps": [_step([_tool("Edit", "src/b.py"),
                                     _tool("Bash", "pytest")])],
                },
            ),
        ],
    }
    subagents = {"rows": [{
        "sub_agent_id": "A", "model": "claude-opus-4-8",
        "input_tokens": 80_000, "output_tokens": 100,
        "cache_tokens": 0, "cache_write_tokens": 0,
        "cost_usd": 0.60, "flags": ["over_provisioned"],
    }]}

    m = build_work_map(story, subagents, root_cost_usd=1.0, root_tokens=80_100)

    assert m["subagent_count"] == 1
    assert m["max_depth"] == 1
    assert m["flagged"] == 1
    child = m["root"]["children"][0]
    assert child["id"] == "A"
    assert child["name"] == "build-it"
    assert child["cost_usd"] == 0.60
    assert child["tokens"] == 80_100
    assert child["flags"] == ["over_provisioned"]
    assert child["activity"]["file_count"] == 1
    assert child["activity"]["bash_count"] == 1


def test_capped_subagent_still_surfaces_cost():
    story = {
        "task": "x", "outcome": "y",
        "steps": [_step([_tool("Task", "deep")],
                        subagent={"agent_id": "D", "name": "deep",
                                  "depth_capped": True})],
    }
    subagents = {"rows": [{
        "sub_agent_id": "D", "model": "claude-haiku-4-5",
        "input_tokens": 10, "output_tokens": 10,
        "cache_tokens": 0, "cache_write_tokens": 0,
        "cost_usd": 0.01, "flags": [],
    }]}

    m = build_work_map(story, subagents)
    child = m["root"]["children"][0]
    assert child["capped"] == "depth"
    assert child["cost_usd"] == 0.01      # cost shown despite no expansion
    assert "not expanded" in child["summary"]
    assert child["children"] == []
    assert m["capped"] == 1


def test_unmapped_subagents_reported_not_dropped():
    # 'Z' has recorded cost but never appears in the story tree.
    story = {"task": "x", "outcome": "y", "steps": [_step([_tool("Read", "f.py")])]}
    subagents = {"rows": [{
        "sub_agent_id": "Z", "model": "m",
        "input_tokens": 4000, "output_tokens": 1000,
        "cache_tokens": 0, "cache_write_tokens": 0,
        "cost_usd": 2.0, "flags": [],
    }]}

    m = build_work_map(story, subagents)
    assert m["subagent_count"] == 0
    assert m["unmapped_count"] == 1
    assert m["unmapped_cost_usd"] == 2.0
    assert m["unmapped_tokens"] == 5000


def test_empty_story_is_root_only():
    m = build_work_map({"task": "", "outcome": "", "steps": []}, None)
    assert m["node_count"] == 1
    assert m["root"]["summary"] == "no tool activity"
    assert m["unmapped_count"] == 0


def test_file_list_capped_but_count_exact():
    tools = [_tool("Read", f"f{i}.py") for i in range(10)]
    m = build_work_map({"steps": [_step(tools)]}, None)
    act = m["root"]["activity"]
    assert act["file_count"] == 10
    assert len(act["files"]) == 8       # MAX_FILES_PER_NODE


def test_omitted_marker_skipped_and_truncation_propagates():
    story = {
        "truncated": True,
        "steps": [_step([_tool("Read", "a")]), {"omitted": 5},
                  _step([_tool("Bash", "x")])],
    }
    m = build_work_map(story, None)
    assert m["root"]["activity"]["steps"] == 2
    assert m["root"]["truncated"] is True
    assert m["truncated"] is True
