"""Unit tests for the resume-brief synthesizer (core/resume_brief.py).

The synthesizer folds the Story / asks shapes ``core/transcript`` produces (built
fresh OR read from the persisted ``session_story`` snapshot) into a compact
brief. These fixtures are those plain dicts — not spans and not raw JSONL — so no
span factory applies; we hand-build the documented Story/ask shape directly.
"""
from __future__ import annotations

from tokenjam.core.resume_brief import (
    _is_substantive_prompt,
    build_resume_brief,
    extract_interruptions,
    extract_todos,
    extract_working_files,
    select_scope_ask,
    select_task_prompt,
)


# --- shape builders ----------------------------------------------------------

def _step(text="", tools=None, is_error=False, is_retry=False):
    return {
        "n": 1,
        "text": text,
        "tools": tools or [],
        "is_error": is_error,
        "is_retry": is_retry,
        "model": "claude-opus-4-8",
    }


def _edit(path):
    return {"name": "Edit", "label": path, "status": "ok"}


def _todo_tool(items):
    return {"name": "TodoWrite", "label": "todos", "status": "ok", "todos": items}


def _ask(prompt, steps=None, outcome=""):
    steps = steps or []
    return {
        "n": 1, "prompt": prompt, "ts": None,
        "step_count": len(steps), "truncated": False,
        "steps": steps, "outcome": outcome,
    }


def _asks(*asks):
    return {"asks": list(asks)}


def _story(task="", outcome="", steps=None):
    steps = steps or []
    return {"task": task, "outcome": outcome, "step_count": len(steps), "steps": steps}


# --- substantive-prompt detection (reliability fix #1) -----------------------

def test_slash_only_prompts_are_not_substantive():
    assert not _is_substantive_prompt("/clear")
    assert not _is_substantive_prompt("/model opus")
    assert not _is_substantive_prompt("/model claude-sonnet-4-5")
    assert not _is_substantive_prompt("   ")
    assert not _is_substantive_prompt(None)


def test_prose_prompts_are_substantive():
    assert _is_substantive_prompt("Fix the flaky retry in worker.py")
    # a slash-command opener followed by real prose keeps the prose
    assert _is_substantive_prompt("/govern run the full loop and report back with a summary")


def test_task_extraction_skips_slash_opener():
    asks = _asks(_ask("/clear"), _ask("Fix the flaky retry in worker.py", [_step("hi")]))
    assert select_task_prompt(asks, None) == "Fix the flaky retry in worker.py"

    brief = build_resume_brief(None, asks, session_id="abc12345")
    assert "Fix the flaky retry in worker.py" in brief
    assert "/clear" not in brief


def test_task_falls_back_to_story_task_when_no_asks():
    story = _story(task="Add a caching layer to the API", steps=[_step("working")])
    brief = build_resume_brief(story, None)
    assert "Add a caching layer to the API" in brief


def test_story_fallback_skips_slash_only_task():
    # story-task fallback path: bare slash command must not surface as TASK
    story = _story(task="/clear", steps=[_step("working")])
    assert select_task_prompt(None, story) == ""
    assert select_task_prompt({}, story) == ""


# --- ask-scoping (reliability fix #2) ----------------------------------------

def test_scope_is_last_substantive_ask():
    asks = _asks(_ask("first task"), _ask("second task"), _ask("/model opus"))
    scope = select_scope_ask(asks)
    assert scope is not None
    assert scope["prompt"] == "second task"


def test_brief_scopes_progress_and_files_to_last_ask():
    ask_a = _ask("Refactor the auth module", [_step("edit", [_edit("src/auth.py")])])
    ask_b = _ask("Now write parser tests", [_step("edit", [_edit("tests/test_parser.py")])])
    brief = build_resume_brief(None, _asks(ask_a, ask_b))

    # task = first substantive ask; working files scoped to the LAST ask only
    assert "Refactor the auth module" in brief
    assert "current phase: Now write parser tests" in brief
    assert "tests/test_parser.py" in brief
    assert "src/auth.py" not in brief


# --- TodoWrite extraction ----------------------------------------------------

def test_extract_todos_returns_in_progress_and_pending():
    steps = [
        _step("plan", [_todo_tool([
            {"content": "old", "status": "completed"},
        ])]),
        _step("replan", [_todo_tool([
            {"content": "wire the hook", "status": "in_progress"},
            {"content": "open the PR", "status": "pending"},
            {"content": "read the code", "status": "completed"},
        ])]),
    ]
    in_prog, pending = extract_todos(steps)
    assert in_prog == ["wire the hook"]
    assert pending == ["open the PR"]


def test_brief_open_section_surfaces_todos():
    steps = [_step("replan", [_todo_tool([
        {"content": "wire the hook", "status": "in_progress"},
        {"content": "open the PR", "status": "pending"},
        {"content": "read the code", "status": "completed"},
    ])])]
    brief = build_resume_brief(None, _asks(_ask("Ship resume-brief", steps)))
    assert "in-progress: wire the hook" in brief
    assert "pending: open the PR" in brief
    assert "read the code" not in brief  # completed items are not "open"


# --- working files -----------------------------------------------------------

def test_working_files_deduped_in_order():
    steps = [
        _step("a", [_edit("a.py"), _edit("b.py")]),
        _step("b", [_edit("a.py")]),  # dup
        _step("c", [{"name": "Read", "label": "c.py", "status": "ok"}]),  # read != dirty
    ]
    assert extract_working_files(steps) == ["a.py", "b.py"]


# --- interruptions (live-transcript only) ------------------------------------

def test_extract_interruptions_from_records():
    records = [
        {"message": {"content": [{"type": "text", "text": "working... API Error: Connection closed mid-response"}]}},
    ]
    hits = extract_interruptions(records)
    assert hits and "Connection closed" in hits[0]


def test_brief_marks_interruption_in_open_section():
    steps = [_step("doing work", [_edit("x.py")])]
    records = [{"message": {"content": [{"type": "text", "text": "API Error: Connection closed"}]}}]
    brief = build_resume_brief(_story(task="t", steps=steps), None, records=records)
    assert "INTERRUPTED" in brief


def test_extract_interruptions_ignores_bad_records():
    assert extract_interruptions(None) == []
    assert extract_interruptions([1, "x", {}]) == []


# --- tried / dead-ends -------------------------------------------------------

def test_tried_section_surfaces_tool_errors():
    # a non-verify tool that errored -> "[tool error]"
    steps = [_step("look for it", [{"name": "Grep", "label": "needle", "status": "error"}])]
    brief = build_resume_brief(None, _asks(_ask("Find the bug", steps)))
    assert "TRIED / DEAD-ENDS" in brief
    assert "tool error" in brief


def test_tried_section_surfaces_failed_checks():
    # an errored test-runner Bash is a failed *verify* -> "[check failed]"
    steps = [_step("run tests", [{"name": "Bash", "label": "pytest tests", "status": "error"}])]
    brief = build_resume_brief(None, _asks(_ask("Make tests pass", steps)))
    assert "check failed" in brief


# --- fail-soft ---------------------------------------------------------------

def test_empty_inputs_return_empty_brief():
    assert build_resume_brief(None, None) == ""
    assert build_resume_brief({"steps": []}, {"asks": []}) == ""


def test_malformed_inputs_never_raise():
    # garbage shapes must degrade, not explode — no exceptions, output type is str
    assert build_resume_brief({"steps": [{"bad": True}]}, {"asks": "not-a-list"}) != ""
    assert isinstance(build_resume_brief({"steps": [None, 3]}, None), str)
    assert extract_todos("not-a-list") == ([], [])
    assert extract_working_files(None) == []


def test_brief_has_all_five_sections():
    steps = [_step("did a thing", [_edit("x.py")])]
    brief = build_resume_brief(None, _asks(_ask("Do the thing", steps)))
    for header in (
        "TASK", "DONE / PROGRESS", "TRIED / DEAD-ENDS",
        "OPEN / WHERE IT LEFT OFF", "WORKING FILES",
    ):
        assert header in brief
