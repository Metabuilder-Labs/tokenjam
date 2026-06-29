"""Unit tests for the pure method-spine transform (core/method_spine.py).

The method spine folds a session Story into an ordered list of intent-tagged
moves — ``delegate`` / ``dead_end`` / ``verify`` / ``act`` — recursively for
subagents. HONESTY (Critical Rule 14): only those four structurally-determinable
kinds exist; richer intent is the opt-in distill layer, never this module. These
tests drive it with hand-built story dicts (no I/O), mirroring test_workmap's
``_tool``/``_step`` helper style.
"""
from __future__ import annotations

from tokenjam.core.method_spine import build_method_spine


def _tool(name: str, label: str = "", status: str = "ok") -> dict:
    return {"name": name, "label": label, "status": status}


def _step(tools: list[dict], **kw) -> dict:
    step = {"n": 1, "ts": None, "text": kw.pop("text", ""), "tools": tools,
            "is_error": kw.pop("is_error", False),
            "is_retry": kw.pop("is_retry", False),
            "model": kw.pop("model", "claude-opus-4-8")}
    step.update(kw)
    return step


def _story(steps: list[dict], **kw) -> dict:
    return {"task": kw.get("task", ""), "outcome": kw.get("outcome", ""),
            "step_count": len(steps), "truncated": kw.get("truncated", False),
            "steps": steps}


# --- kinds -------------------------------------------------------------------

def test_act_is_the_default_kind():
    spine = build_method_spine(_story([
        _step([_tool("Read", "src/app.py")], text="Reading the app."),
    ]))
    assert len(spine) == 1
    assert spine[0]["kind"] == "act"


def test_delegate_from_task_tool():
    spine = build_method_spine(_story([
        _step([_tool("Task", "build-it")], text="Spawning a worker."),
    ]))
    assert spine[0]["kind"] == "delegate"


def test_delegate_from_attached_subagent_even_without_task_tool():
    # A resolved subagent child forces delegate regardless of tool names.
    spine = build_method_spine(_story([
        _step([_tool("Read", "x.py")], text="Orchestrate.",
              subagent={"agent_id": "A", "name": "w", "steps": []}),
    ]))
    assert spine[0]["kind"] == "delegate"


def test_dead_end_from_retry():
    spine = build_method_spine(_story([
        _step([_tool("Edit", "a.py")], text="Try again.", is_retry=True),
    ]))
    assert spine[0]["kind"] == "dead_end"
    assert spine[0]["is_retry"] is True


def test_dead_end_from_revert_command():
    for cmd in ("git checkout -- a.py", "git restore a.py",
                "git revert HEAD", "git reset --hard origin/main"):
        spine = build_method_spine(_story([
            _step([_tool("Bash", cmd)], text="Undo it."),
        ]))
        assert spine[0]["kind"] == "dead_end", cmd


def test_safe_git_is_not_a_dead_end():
    spine = build_method_spine(_story([
        _step([_tool("Bash", "git status")], text="Check state."),
    ]))
    assert spine[0]["kind"] == "act"


def test_verify_from_test_runner():
    for cmd in ("pytest -q", "python -m unittest", "npm test", "npm run test",
                "yarn test", "go test ./...", "cargo test", "make test",
                "tox", "jest", "vitest run"):
        spine = build_method_spine(_story([
            _step([_tool("Bash", cmd)], text="Run the tests."),
        ]))
        assert spine[0]["kind"] == "verify", cmd


def test_verify_failed_on_error_status():
    spine = build_method_spine(_story([
        _step([_tool("Bash", "pytest -q", status="error")], text="Run tests."),
    ]))
    assert spine[0]["kind"] == "verify"
    assert spine[0]["failed"] is True


def test_verify_failed_on_step_is_error():
    spine = build_method_spine(_story([
        _step([_tool("Bash", "pytest -q")], text="Run tests.", is_error=True),
    ]))
    assert spine[0]["failed"] is True


def test_verify_passing_is_not_failed():
    spine = build_method_spine(_story([
        _step([_tool("Bash", "pytest -q")], text="Run tests."),
    ]))
    assert spine[0]["kind"] == "verify"
    assert spine[0]["failed"] is False


def test_failed_only_for_verify_kind():
    # An erroring non-verify step is NOT marked failed (failed is verify-scoped).
    spine = build_method_spine(_story([
        _step([_tool("Edit", "a.py", status="error")], text="Edit it."),
    ]))
    assert spine[0]["kind"] == "act"
    assert spine[0]["failed"] is False


def test_kind_precedence_delegate_over_dead_end():
    # A retry that also spawns a subagent is delegate (highest precedence).
    spine = build_method_spine(_story([
        _step([_tool("Task", "w")], text="Retry via worker.", is_retry=True),
    ]))
    assert spine[0]["kind"] == "delegate"


def test_kind_precedence_dead_end_over_verify():
    # A retried test run is a dead_end (retry beats verify).
    spine = build_method_spine(_story([
        _step([_tool("Bash", "pytest -q")], text="Re-run.", is_retry=True),
    ]))
    assert spine[0]["kind"] == "dead_end"


# --- source / label / quote --------------------------------------------------

def test_source_agent_words_when_narrated():
    spine = build_method_spine(_story([
        _step([_tool("Read", "a.py")], text="First line.\nSecond line."),
    ]))
    move = spine[0]
    assert move["source"] == "agent_words"
    assert move["label"] == "First line."        # first non-empty line
    assert move["quote"] == "First line.\nSecond line."


def test_label_trimmed_to_eighty_chars():
    long = "x" * 200
    spine = build_method_spine(_story([_step([_tool("Read", "a.py")], text=long)]))
    assert len(spine[0]["label"]) <= 81           # 80 + ellipsis
    assert spine[0]["label"].endswith("…")


def test_quote_is_first_paragraph():
    text = "Para one line a.\nPara one line b.\n\nPara two."
    spine = build_method_spine(_story([_step([_tool("Read", "a.py")], text=text)]))
    assert spine[0]["quote"] == "Para one line a.\nPara one line b."


def test_structural_label_single_edit():
    spine = build_method_spine(_story([_step([_tool("Edit", "core/workmap.py")])]))
    move = spine[0]
    assert move["source"] == "structural"
    assert move["label"] == "edit workmap.py"
    assert move["quote"] is None


def test_structural_label_bash():
    spine = build_method_spine(_story([_step([_tool("Bash", "pytest")])]))
    assert spine[0]["label"] == "bash: pytest"
    # (it's also a verify by command match)
    assert spine[0]["kind"] == "verify"


def test_structural_label_multiple_reads():
    spine = build_method_spine(_story([_step([
        _tool("Read", "a.py"), _tool("Read", "b.py"), _tool("Read", "c.py"),
    ])]))
    assert spine[0]["source"] == "structural"
    assert spine[0]["label"] == "read 3 files"


def test_evidence_is_compact_tool_summary():
    spine = build_method_spine(_story([
        _step([_tool("Read", "a.py"), _tool("Bash", "ls", status="error")],
              text="Look around."),
    ]))
    assert spine[0]["evidence"] == [
        {"name": "Read", "label": "a.py", "status": "ok"},
        {"name": "Bash", "label": "ls", "status": "error"},
    ]


# --- recursion + caps --------------------------------------------------------

def test_recursion_into_nested_subagent():
    child = {
        "agent_id": "A", "name": "worker",
        "task": "build", "outcome": "done",
        "steps": [
            _step([_tool("Edit", "x.py")], text="Editing x."),
            _step([_tool("Bash", "pytest")], text="Verifying."),
        ],
    }
    spine = build_method_spine(_story([
        _step([_tool("Task", "worker")], text="Delegate to worker.",
              subagent=child),
    ]))
    move = spine[0]
    assert move["kind"] == "delegate"
    assert len(move["children"]) == 2
    assert move["children"][0]["kind"] == "act"
    assert move["children"][1]["kind"] == "verify"
    assert "capped" not in move


def test_multiple_parallel_subagents_concatenate_children():
    sub_a = {"agent_id": "A", "name": "a", "steps": [_step([_tool("Read", "a")])]}
    sub_b = {"agent_id": "B", "name": "b", "steps": [_step([_tool("Edit", "b")])]}
    spine = build_method_spine(_story([
        _step([_tool("Task", "a"), _tool("Task", "b")], text="Fan out.",
              subagents=[sub_a, sub_b]),
    ]))
    assert len(spine[0]["children"]) == 2


def test_capped_child_emits_empty_children_and_marker():
    spine = build_method_spine(_story([
        _step([_tool("Task", "deep")], text="Go deeper.",
              subagent={"agent_id": "D", "name": "deep", "depth_capped": True}),
    ]))
    move = spine[0]
    assert move["kind"] == "delegate"
    assert move["children"] == []
    assert move["capped"] == "depth"


def test_cycle_child_marked_capped():
    spine = build_method_spine(_story([
        _step([_tool("Task", "loop")], text="Loop.",
              subagent={"agent_id": "L", "name": "loop", "cycle": True}),
    ]))
    assert spine[0]["capped"] == "cycle"


# --- structure ---------------------------------------------------------------

def test_omitted_markers_skipped():
    spine = build_method_spine(_story([
        _step([_tool("Read", "a.py")], text="A."),
        {"omitted": 42},
        _step([_tool("Edit", "b.py")], text="B."),
    ]))
    assert len(spine) == 2
    assert [m["label"] for m in spine] == ["A.", "B."]


def test_empty_story_yields_empty_spine():
    assert build_method_spine(_story([])) == []
    assert build_method_spine({}) == []
