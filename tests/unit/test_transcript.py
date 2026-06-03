"""Unit tests for the Session Story transcript parser (core/transcript.py).

These fixtures are Claude Code on-disk JSONL records (NOT NormalizedSpans), so
the span factories don't apply — we hand-write minimal CC records that match the
shapes verified against real transcripts.
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenjam.core.transcript import (
    MAX_STEP_TEXT_CHARS,
    MAX_STORY_STEPS,
    MAX_TOOL_LABEL_CHARS,
    build_session_story,
)


# --- Fixture builders --------------------------------------------------------

def _user_prompt(text: str, is_meta: bool = False) -> dict:
    rec = {"type": "user", "message": {"role": "user", "content": text}}
    if is_meta:
        rec["isMeta"] = True
    return rec


def _tool_result(tool_use_id: str, is_error: bool = False) -> dict:
    block: dict = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": "(omitted)",
    }
    if is_error:
        block["is_error"] = True
    return {"type": "user", "message": {"role": "user", "content": [block]}}


def _assistant(
    text: str | None,
    tools: list[dict] | None = None,
    model: str = "claude-opus-4-8",
    ts: str = "2026-06-15T09:11:36.133Z",
) -> dict:
    content: list[dict] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    for t in tools or []:
        content.append(
            {
                "type": "tool_use",
                "id": t["id"],
                "name": t["name"],
                "input": t.get("input", {}),
            }
        )
    return {
        "type": "assistant",
        "timestamp": ts,
        "message": {"role": "assistant", "model": model, "content": content},
    }


def _write_transcript(projects_root: Path, session_id: str, records: list[dict]) -> Path:
    """Write records as JSONL under <projects_root>/<project>/<session_id>.jsonl."""
    project_dir = projects_root / "-Users-test-project"
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _make_fixture(projects_root: Path, session_id: str = "sess-1") -> Path:
    """A realistic mini-session: prompt, two tool turns (one error), a retry, final."""
    records = [
        _user_prompt("Fix the failing auth test and make CI green."),
        # meta + tool-result-only user records must be skipped when picking task
        _user_prompt("internal meta noise", is_meta=True),
        _assistant(
            "Let me read the auth module first.",
            tools=[{"id": "t1", "name": "Read", "input": {"file_path": "src/auth.py"}}],
        ),
        _tool_result("t1", is_error=False),
        _assistant(
            "Now I'll run the test suite.",
            tools=[
                {"id": "t2", "name": "Bash", "input": {"command": "pytest tests/auth"}}
            ],
        ),
        _tool_result("t2", is_error=True),  # the test run failed
        # retry: same tool name + label as the immediately-preceding turn
        _assistant(
            "Retrying the test after a tweak.",
            tools=[
                {"id": "t3", "name": "Bash", "input": {"command": "pytest tests/auth"}}
            ],
        ),
        _tool_result("t3", is_error=False),
        _assistant("All tests pass now. CI should be green."),
    ]
    return _write_transcript(projects_root, session_id, records)


# --- Tests -------------------------------------------------------------------

def test_task_extracted_from_first_real_prompt(tmp_path):
    _make_fixture(tmp_path)
    story = build_session_story("sess-1", projects_root=tmp_path)
    assert story is not None
    assert story["task"] == "Fix the failing auth test and make CI green."


def test_step_count_counts_real_assistant_turns(tmp_path):
    _make_fixture(tmp_path)
    story = build_session_story("sess-1", projects_root=tmp_path)
    assert story is not None
    # 3 narration+tool turns + 1 final narration-only turn = 4 steps
    assert story["step_count"] == 4
    assert len(story["steps"]) == 4


def test_tool_label_uses_most_useful_arg(tmp_path):
    _make_fixture(tmp_path)
    story = build_session_story("sess-1", projects_root=tmp_path)
    assert story is not None
    first_step = story["steps"][0]
    assert first_step["tools"][0]["name"] == "Read"
    assert first_step["tools"][0]["label"] == "src/auth.py"
    assert first_step["tools"][0]["status"] == "ok"


def test_error_step_flagged(tmp_path):
    _make_fixture(tmp_path)
    story = build_session_story("sess-1", projects_root=tmp_path)
    assert story is not None
    # step 2 = the Bash turn whose tool_result has is_error=true
    bash_step = story["steps"][1]
    assert bash_step["tools"][0]["name"] == "Bash"
    assert bash_step["tools"][0]["status"] == "error"
    assert bash_step["is_error"] is True


def test_retry_step_flagged(tmp_path):
    _make_fixture(tmp_path)
    story = build_session_story("sess-1", projects_root=tmp_path)
    assert story is not None
    retry_step = story["steps"][2]
    assert retry_step["is_retry"] is True
    # the non-repeating earlier steps are not retries
    assert story["steps"][0]["is_retry"] is False
    assert story["steps"][1]["is_retry"] is False


def test_outcome_is_last_narration(tmp_path):
    _make_fixture(tmp_path)
    story = build_session_story("sess-1", projects_root=tmp_path)
    assert story is not None
    assert story["outcome"] == "All tests pass now. CI should be green."


def test_not_found_returns_none(tmp_path):
    # No transcript written at all.
    assert build_session_story("does-not-exist", projects_root=tmp_path) is None


def test_empty_projects_root_returns_none(tmp_path):
    missing = tmp_path / "nope"
    assert build_session_story("sess-1", projects_root=missing) is None


def test_thinking_blocks_excluded_from_text(tmp_path):
    records = [
        _user_prompt("Do the thing"),
        {
            "type": "assistant",
            "timestamp": "2026-06-15T09:11:36.133Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [
                    {"type": "thinking", "thinking": "secret internal reasoning"},
                    {"type": "text", "text": "Visible narration only."},
                ],
            },
        },
    ]
    _write_transcript(tmp_path, "sess-think", records)
    story = build_session_story("sess-think", projects_root=tmp_path)
    assert story is not None
    assert story["steps"][0]["text"] == "Visible narration only."
    assert "secret internal reasoning" not in json.dumps(story)


def test_text_truncation_sets_flag(tmp_path):
    long_text = "x" * (MAX_STEP_TEXT_CHARS + 50)
    records = [
        _user_prompt("go"),
        _assistant(long_text),
    ]
    _write_transcript(tmp_path, "sess-long", records)
    story = build_session_story("sess-long", projects_root=tmp_path)
    assert story is not None
    step = story["steps"][0]
    assert step["text_truncated"] is True
    assert len(step["text"]) <= MAX_STEP_TEXT_CHARS + 1  # +1 for the ellipsis


def test_tool_label_never_dumps_full_input(tmp_path):
    huge_cmd = "echo " + ("a" * 500)
    records = [
        _user_prompt("go"),
        _assistant(
            "running",
            tools=[{"id": "t1", "name": "Bash", "input": {"command": huge_cmd}}],
        ),
    ]
    _write_transcript(tmp_path, "sess-huge", records)
    story = build_session_story("sess-huge", projects_root=tmp_path)
    assert story is not None
    label = story["steps"][0]["tools"][0]["label"]
    assert len(label) <= MAX_TOOL_LABEL_CHARS + 1


def test_malformed_lines_are_tolerated(tmp_path):
    project_dir = tmp_path / "-Users-test-project"
    project_dir.mkdir(parents=True)
    path = project_dir / "sess-bad.jsonl"
    good = json.dumps(_user_prompt("do it"))
    asst = json.dumps(_assistant("ok done"))
    path.write_text(f"{good}\nNOT JSON{{\n\n{asst}\n", encoding="utf-8")
    story = build_session_story("sess-bad", projects_root=tmp_path)
    assert story is not None
    assert story["task"] == "do it"
    assert story["outcome"] == "ok done"


def test_step_cap_inserts_omitted_marker(tmp_path):
    records: list[dict] = [_user_prompt("big run")]
    for i in range(MAX_STORY_STEPS + 25):
        records.append(_assistant(f"step number {i}"))
    _write_transcript(tmp_path, "sess-cap", records)
    story = build_session_story("sess-cap", projects_root=tmp_path)
    assert story is not None
    assert story["truncated"] is True
    assert story["step_count"] == MAX_STORY_STEPS + 25
    markers = [s for s in story["steps"] if "omitted" in s]
    assert len(markers) == 1
    assert markers[0]["omitted"] == 25
    # head + tail + 1 marker
    assert len(story["steps"]) == MAX_STORY_STEPS + 1
