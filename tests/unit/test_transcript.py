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
    MAX_SUBAGENT_DEPTH,
    MAX_TOOL_LABEL_CHARS,
    _first_user_prompt,
    build_session_asks,
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
    # The failed tool also carries the transcript's error message.
    assert bash_step["tools"][0]["error"] == "(omitted)"


def test_error_message_wrapper_is_stripped(tmp_path):
    records = [
        _user_prompt("do it"),
        _assistant("Running.", tools=[{"id": "e1", "name": "Bash",
                                       "input": {"command": "x"}}]),
        {"type": "user", "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "e1", "is_error": True,
            "content": "<tool_use_error>Blocked: foreground sleep</tool_use_error>",
        }]}},
        _assistant("Pivoting."),
    ]
    _write_transcript(tmp_path, "err-sess", records)
    story = build_session_story("err-sess", projects_root=tmp_path)
    tool = story["steps"][0]["tools"][0]
    assert tool["status"] == "error"
    assert tool["error"] == "Blocked: foreground sleep"  # wrapper tags removed


def test_ok_tool_has_no_error_field(tmp_path):
    _make_fixture(tmp_path)
    story = build_session_story("sess-1", projects_root=tmp_path)
    ok_step = story["steps"][0]  # first Bash, is_error=False
    assert ok_step["tools"][0]["status"] == "ok"
    assert "error" not in ok_step["tools"][0]


def test_retry_step_flagged(tmp_path):
    _make_fixture(tmp_path)
    story = build_session_story("sess-1", projects_root=tmp_path)
    assert story is not None
    retry_step = story["steps"][2]
    assert retry_step["is_retry"] is True
    # the non-repeating earlier steps are not retries
    assert story["steps"][0]["is_retry"] is False
    assert story["steps"][1]["is_retry"] is False


def test_repeat_after_success_is_not_a_retry(tmp_path):
    """A retry means RE-ATTEMPT AFTER FAILURE. Two consecutive SUCCESSFUL steps
    with the same tool signature (e.g. editing the same file twice in a row —
    the most normal agent behavior there is) must NOT be flagged: the old
    signature-repeat-only rule painted half a real session's Map with retry
    marks (#58)."""
    records = [
        _user_prompt("Refactor the config module."),
        _assistant(
            "Editing the config.",
            tools=[{"id": "t1", "name": "Edit", "input": {"file_path": "src/config.py"}}],
        ),
        _tool_result("t1", is_error=False),
        # same signature, but the previous step SUCCEEDED -> not a retry
        _assistant(
            "One more tweak to the same file.",
            tools=[{"id": "t2", "name": "Edit", "input": {"file_path": "src/config.py"}}],
        ),
        _tool_result("t2", is_error=False),
    ]
    _write_transcript(tmp_path, "sess-rep", records)
    story = build_session_story("sess-rep", projects_root=tmp_path)
    assert story is not None
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


# --- Nested subagent fixtures + tests ----------------------------------------

def _agent_tool_result(tool_use_id: str, agent_id: str, is_error: bool = False) -> dict:
    """A Task tool_result whose content carries the spawned child's agentId.

    Mirrors real Claude Code: the result text embeds the child agentId (16-17
    hex) which links to ``subagents/agent-<agentId>.jsonl``.
    """
    text = f"Agent (agentId: {agent_id}) finished. See results above."
    block: dict = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text,
    }
    if is_error:
        block["is_error"] = True
    return {"type": "user", "message": {"role": "user", "content": [block]}}


def _task_turn(text: str, tool_id: str, name: str = "do work") -> dict:
    """An assistant turn that spawns a subagent via the Task tool."""
    return _assistant(
        text,
        tools=[{"id": tool_id, "name": "Task", "input": {
            "description": name, "subagent_type": "general-purpose",
            "prompt": "do the work",
        }}],
    )


def _write_subagent(
    projects_root: Path,
    root_session_id: str,
    agent_id: str,
    records: list[dict],
    name: str | None = None,
) -> None:
    """Write a subagent transcript under <root>/subagents/agent-<id>.jsonl.

    All subagents (any depth) live FLAT in the root session's subagents dir.
    """
    subdir = projects_root / "-Users-test-project" / root_session_id / "subagents"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / f"agent-{agent_id}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )
    if name is not None:
        meta = {"agentType": "general-purpose", "name": name,
                "description": name, "toolUseId": "toolu_x"}
        (subdir / f"agent-{agent_id}.meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )


def test_subagent_attached_to_task_step(tmp_path):
    sid = "root-1"
    child = "abc123def456abc78"  # 17 hex
    parent = [
        _user_prompt("Orchestrate the build."),
        _task_turn("Spawning a worker.", "tt1", name="impl-thing"),
        _agent_tool_result("tt1", child),
        _assistant("Worker finished, all good."),
    ]
    _write_transcript(tmp_path, sid, parent)
    _write_subagent(tmp_path, sid, child, [
        _user_prompt("Implement the thing."),
        _assistant("Reading.", tools=[
            {"id": "c1", "name": "Read", "input": {"file_path": "src/x.py"}}]),
        _tool_result("c1"),
        _assistant("Done implementing."),
    ], name="impl-thing")

    story = build_session_story(sid, projects_root=tmp_path)
    assert story is not None
    task_step = story["steps"][0]
    assert task_step["tools"][0]["name"] == "Task"
    sub = task_step["subagent"]
    assert sub["agent_id"] == child
    assert sub["name"] == "impl-thing"
    assert sub["task"] == "Implement the thing."
    assert sub["outcome"] == "Done implementing."
    assert len(sub["steps"]) == 2
    assert sub["steps"][0]["tools"][0]["label"] == "src/x.py"
    # internal marker never leaks into the payload
    assert "_spawns" not in json.dumps(story)


def test_subagent_recurses_to_grandchild(tmp_path):
    sid = "root-2"
    child = "1111111111111111a"
    grand = "2222222222222222b"
    parent = [
        _user_prompt("Top level task."),
        _task_turn("Spawn child.", "p1"),
        _agent_tool_result("p1", child),
        _assistant("Child done."),
    ]
    _write_transcript(tmp_path, sid, parent)
    # child itself spawns a grandchild
    _write_subagent(tmp_path, sid, child, [
        _user_prompt("Child task."),
        _task_turn("Spawn grandchild.", "g1"),
        _agent_tool_result("g1", grand),
        _assistant("Grandchild done."),
    ], name="child-agent")
    _write_subagent(tmp_path, sid, grand, [
        _user_prompt("Grandchild task."),
        _assistant("Deep work complete."),
    ], name="grand-agent")

    story = build_session_story(sid, projects_root=tmp_path)
    assert story is not None
    child_sub = story["steps"][0]["subagent"]
    assert child_sub["name"] == "child-agent"
    # the child's own Task step carries ITS subagent (the grandchild)
    grand_sub = child_sub["steps"][0]["subagent"]
    assert grand_sub["agent_id"] == grand
    assert grand_sub["name"] == "grand-agent"
    assert grand_sub["task"] == "Grandchild task."
    assert grand_sub["outcome"] == "Deep work complete."


def test_subagent_agent_id_resolved_from_tool_result_regex(tmp_path):
    """No meta.json present -> agentId comes purely from the tool_result regex."""
    sid = "root-regex"
    child = "deadbeefdeadbeef0"
    parent = [
        _user_prompt("go"),
        _task_turn("spawn", "x1"),
        _agent_tool_result("x1", child),
    ]
    _write_transcript(tmp_path, sid, parent)
    # write child WITHOUT a meta.json -> name falls back to Task input
    _write_subagent(tmp_path, sid, child, [
        _user_prompt("child"),
        _assistant("done"),
    ], name=None)

    story = build_session_story(sid, projects_root=tmp_path)
    assert story is not None
    sub = story["steps"][0]["subagent"]
    assert sub["agent_id"] == child
    # name fell back to the Task input (subagent_type wins over description).
    assert sub["name"] == "general-purpose"


def test_subagents_disabled_returns_flat(tmp_path):
    sid = "root-flat"
    child = "ffffffffffffffff1"
    parent = [
        _user_prompt("go"),
        _task_turn("spawn", "f1"),
        _agent_tool_result("f1", child),
    ]
    _write_transcript(tmp_path, sid, parent)
    _write_subagent(tmp_path, sid, child, [
        _user_prompt("child"), _assistant("done")], name="kid")

    story = build_session_story(sid, projects_root=tmp_path, include_subagents=False)
    assert story is not None
    assert "subagent" not in story["steps"][0]
    assert "_spawns" not in json.dumps(story)


def test_subagent_depth_cap(tmp_path):
    """A chain deeper than MAX_SUBAGENT_DEPTH gets a depth_capped marker."""
    sid = "root-depth"
    # build a linear chain of MAX_SUBAGENT_DEPTH + 2 agents
    chain = [f"{i:016x}c" for i in range(MAX_SUBAGENT_DEPTH + 2)]
    parent = [
        _user_prompt("root"),
        _task_turn("spawn", "d0"),
        _agent_tool_result("d0", chain[0]),
    ]
    _write_transcript(tmp_path, sid, parent)
    for i, aid in enumerate(chain):
        if i + 1 < len(chain):
            recs = [
                _user_prompt(f"level {i}"),
                _task_turn("deeper", f"d{i + 1}"),
                _agent_tool_result(f"d{i + 1}", chain[i + 1]),
            ]
        else:
            recs = [_user_prompt(f"level {i}"), _assistant("bottom")]
        _write_subagent(tmp_path, sid, aid, recs, name=f"a{i}")

    story = build_session_story(sid, projects_root=tmp_path)
    assert story is not None
    # descend the chain; somewhere at/below the cap a depth_capped marker appears
    node = story["steps"][0]["subagent"]
    depth = 1
    seen_cap = False
    while node is not None:
        if node.get("depth_capped"):
            seen_cap = True
            break
        steps = node.get("steps", [])
        node = steps[0].get("subagent") if steps else None
        depth += 1
    assert seen_cap is True
    assert depth <= MAX_SUBAGENT_DEPTH + 1


def test_subagent_cycle_guard_does_not_hang(tmp_path):
    """Two agents referencing each other must terminate, marked cycle."""
    sid = "root-cycle"
    a = "aaaaaaaaaaaaaaaa1"
    b = "bbbbbbbbbbbbbbbb2"
    parent = [
        _user_prompt("root"),
        _task_turn("spawn A", "r1"),
        _agent_tool_result("r1", a),
    ]
    _write_transcript(tmp_path, sid, parent)
    # A spawns B, B spawns A -> cycle
    _write_subagent(tmp_path, sid, a, [
        _user_prompt("A"),
        _task_turn("spawn B", "a1"),
        _agent_tool_result("a1", b),
    ], name="agent-a")
    _write_subagent(tmp_path, sid, b, [
        _user_prompt("B"),
        _task_turn("spawn A", "b1"),
        _agent_tool_result("b1", a),
    ], name="agent-b")

    story = build_session_story(sid, projects_root=tmp_path)
    assert story is not None
    a_sub = story["steps"][0]["subagent"]
    assert a_sub["name"] == "agent-a"
    b_sub = a_sub["steps"][0]["subagent"]
    assert b_sub["name"] == "agent-b"
    # B's attempt to re-enter A is caught by the seen-set -> cycle marker.
    a_again = b_sub["steps"][0]["subagent"]
    assert a_again["cycle"] is True
    assert a_again.get("steps") in (None, [])


def test_subagent_privacy_no_full_io_at_any_depth(tmp_path):
    """Full tool inputs/outputs never appear at any nesting level."""
    sid = "root-priv"
    child = "0123456789abcdef0"
    secret_cmd = "rm -rf /super/secret/" + ("z" * 400)
    secret_out = "LEAKED_OUTPUT_" + ("q" * 400)
    parent = [
        _user_prompt("go"),
        _task_turn("spawn", "p1"),
        _agent_tool_result("p1", child),
    ]
    _write_transcript(tmp_path, sid, parent)
    _write_subagent(tmp_path, sid, child, [
        _user_prompt("child"),
        _assistant("running", tools=[
            {"id": "c1", "name": "Bash", "input": {"command": secret_cmd}}]),
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": secret_out}]}},
        _assistant("done"),
    ], name="kid")

    story = build_session_story(sid, projects_root=tmp_path)
    assert story is not None
    blob = json.dumps(story)
    assert secret_cmd not in blob  # full command never surfaced
    assert secret_out not in blob  # tool output never surfaced
    # the Bash label is present but capped
    sub = story["steps"][0]["subagent"]
    bash_label = sub["steps"][0]["tools"][0]["label"]
    assert len(bash_label) <= MAX_TOOL_LABEL_CHARS + 1


# --- First-prompt cleaning: strip the Claude Code harness wrapper ------------ #

def test_first_prompt_strips_system_reminder():
    records = [_user_prompt(
        "<system-reminder>\n# claudeMd\nbig CLAUDE.md dump here\n</system-reminder>\n"
        "Build me a parser for the config file."
    )]
    assert _first_user_prompt(records) == "Build me a parser for the config file."


def test_first_prompt_surfaces_slash_command_when_only_wrapper():
    records = [_user_prompt(
        "<command-name>/review</command-name>"
        "<command-message>review</command-message>"
        "<command-args>PR 152</command-args>"
    )]
    assert _first_user_prompt(records) == "/review PR 152"


def test_first_prompt_strips_local_command_caveat_and_stdout():
    records = [_user_prompt(
        "<local-command-caveat>Caveat: generated while running local commands.</local-command-caveat>"
        "<command-name>/compact</command-name>"
        "<command-message>compact</command-message>"
        "<command-args></command-args>"
        "<local-command-stdout>Compacted.</local-command-stdout>"
    )]
    assert _first_user_prompt(records) == "/compact"


def test_first_prompt_skips_wrapper_only_message():
    records = [
        _user_prompt("<system-reminder>session init</system-reminder>"),
        _user_prompt("<system-reminder>x</system-reminder>\nThe actual question."),
    ]
    assert _first_user_prompt(records) == "The actual question."


def test_first_prompt_leaves_clean_text_unchanged():
    # No wrapper -> returned verbatim (no regression for normal prompts).
    records = [_user_prompt("Just do the thing.")]
    assert _first_user_prompt(records) == "Just do the thing."


# --- Ask segmentation: a session is a sequence of exchanges, not one task --- #

def test_asks_segment_by_user_prompt(tmp_path):
    sid = "asks-1"
    records = [
        _user_prompt("First question: read the file."),
        _assistant("Reading.",
                   tools=[{"id": "a1", "name": "Read", "input": {"file_path": "a.py"}}]),
        _tool_result("a1"),
        _assistant("Done with first."),
        _user_prompt("Second question: run the tests."),
        _assistant("Running.",
                   tools=[{"id": "b1", "name": "Bash", "input": {"command": "pytest"}}]),
        _tool_result("b1"),
        _assistant("Tests pass."),
    ]
    _write_transcript(tmp_path, sid, records)
    payload = build_session_asks(sid, projects_root=tmp_path)
    assert payload is not None
    asks = payload["asks"]
    assert len(asks) == 2
    assert asks[0]["n"] == 1
    assert asks[0]["prompt"] == "First question: read the file."
    assert asks[0]["step_count"] == 2          # "Reading." + "Done with first."
    assert asks[1]["prompt"] == "Second question: run the tests."
    assert asks[1]["step_count"] == 2
    assert asks[1]["outcome"] == "Tests pass."


def test_asks_skip_meta_and_tool_result_turns(tmp_path):
    sid = "asks-2"
    records = [
        _user_prompt("meta noise", is_meta=True),
        _user_prompt("The only real ask."),
        _assistant("Working.",
                   tools=[{"id": "x", "name": "Read", "input": {"file_path": "f.py"}}]),
        _tool_result("x"),       # tool-result-only user record -> not an ask
        _assistant("Done."),
    ]
    _write_transcript(tmp_path, sid, records)
    payload = build_session_asks(sid, projects_root=tmp_path)
    assert [a["prompt"] for a in payload["asks"]] == ["The only real ask."]


def test_asks_strip_harness_wrapper_per_prompt(tmp_path):
    sid = "asks-3"
    records = [
        _user_prompt("First."),
        _assistant("ok"),
        _user_prompt("<system-reminder>ctx</system-reminder>\nSecond real ask."),
        _assistant("ok2"),
    ]
    _write_transcript(tmp_path, sid, records)
    payload = build_session_asks(sid, projects_root=tmp_path)
    assert [a["prompt"] for a in payload["asks"]] == ["First.", "Second real ask."]


def test_asks_attribute_subagent_to_its_ask(tmp_path):
    sid = "asks-sub"
    child = "abc123def456abc78"
    records = [
        _user_prompt("Ask one: just read."),
        _assistant("reading",
                   tools=[{"id": "r1", "name": "Read", "input": {"file_path": "a.py"}}]),
        _tool_result("r1"),
        _user_prompt("Ask two: orchestrate a worker."),
        _task_turn("Spawning.", "tt1", name="worker"),
        _agent_tool_result("tt1", child),
        _assistant("worker done"),
    ]
    _write_transcript(tmp_path, sid, records)
    _write_subagent(tmp_path, sid, child,
                    [_user_prompt("Build it."), _assistant("built")], name="worker")
    payload = build_session_asks(sid, projects_root=tmp_path)
    asks = payload["asks"]
    assert len(asks) == 2
    # ask 1 spawned nothing
    assert all("subagent" not in s and "subagents" not in s for s in asks[0]["steps"])
    # ask 2's Task step carries the subagent
    task_step = next(
        s for s in asks[1]["steps"]
        if any(t["name"] == "Task" for t in s.get("tools", []))
    )
    assert task_step["subagent"]["name"] == "worker"


def test_asks_skip_task_notification_turns(tmp_path):
    # Background-task completion notices are injected as user messages but are
    # not human asks: they must not start a new ask, and the work that follows
    # folds into the preceding ask's segment.
    sid = "asks-tn"
    records = [
        _user_prompt("Real ask: do the work."),
        _assistant("working"),
        _user_prompt("<task-notification>\n<task-id>abc</task-id>\n"
                     "<status>completed</status>\n</task-notification>"),
        _assistant("got it, continuing"),
        _user_prompt("Second real ask."),
        _assistant("done"),
    ]
    _write_transcript(tmp_path, sid, records)
    payload = build_session_asks(sid, projects_root=tmp_path)
    assert [a["prompt"] for a in payload["asks"]] == [
        "Real ask: do the work.", "Second real ask."
    ]
    # the post-notification turn folds into the first ask
    assert payload["asks"][0]["step_count"] == 2


def test_first_prompt_strips_task_notification():
    records = [
        _user_prompt("<task-notification>done</task-notification>"),
        _user_prompt("The real first ask."),
    ]
    assert _first_user_prompt(records) == "The real first ask."


def test_story_marks_ask_boundaries_on_main_thread(tmp_path):
    # The Timeline story tags the first step after each human ask with the
    # prompt so the UI can mark where each exchange begins.
    sid = "story-asks"
    records = [
        _user_prompt("First ask."),
        _assistant("doing first",
                   tools=[{"id": "a", "name": "Read", "input": {"file_path": "a.py"}}]),
        _tool_result("a"),
        _assistant("more first work"),
        _user_prompt("Second ask."),
        _assistant("doing second"),
    ]
    _write_transcript(tmp_path, sid, records)
    story = build_session_story(sid, projects_root=tmp_path)
    steps = story["steps"]
    assert steps[0]["ask"] == "First ask."
    assert "ask" not in steps[1]          # continuation of the first ask
    second = next(s for s in steps if s.get("ask") == "Second ask.")
    assert second["text"] == "doing second"


def test_subagent_story_has_no_ask_markers(tmp_path):
    sid = "sa-noask"
    child = "abc123def456abc78"
    records = [
        _user_prompt("Spawn a worker."),
        _task_turn("spawning", "tt1", name="worker"),
        _agent_tool_result("tt1", child),
        _assistant("done"),
    ]
    _write_transcript(tmp_path, sid, records)
    _write_subagent(tmp_path, sid, child,
                    [_user_prompt("Build."), _assistant("built")], name="worker")
    story = build_session_story(sid, projects_root=tmp_path)
    sub = story["steps"][0]["subagent"]
    assert all("ask" not in s for s in sub["steps"])


# --- TodoWrite payload preservation (#67) -----------------------------------

def _real_todos() -> list[dict]:
    """A real-shaped Claude Code TodoWrite payload (content + status per item)."""
    return [
        {"content": "Read the auth module", "status": "completed",
         "activeForm": "Reading the auth module"},
        {"content": "Fix the failing test", "status": "in_progress",
         "activeForm": "Fixing the failing test"},
        {"content": "Run the full suite", "status": "pending",
         "activeForm": "Running the full suite"},
    ]


def test_todowrite_payload_preserved_not_dropped(tmp_path):
    records = [
        _user_prompt("Fix the auth test."),
        _assistant(
            "Let me lay out the plan.",
            tools=[{"id": "td1", "name": "TodoWrite",
                    "input": {"todos": _real_todos()}}],
        ),
        _tool_result("td1"),
        _assistant("Done."),
    ]
    _write_transcript(tmp_path, "todo-1", records)
    story = build_session_story("todo-1", projects_root=tmp_path)
    assert story is not None
    tool = story["steps"][0]["tools"][0]
    assert tool["name"] == "TodoWrite"
    # The label is a non-empty rollup — NOT the empty string the old parser gave.
    assert tool["label"] == "3 todos: 1 done, 1 in progress, 1 pending"
    # The structured payload survives: content + per-item status.
    assert tool["todos"] == [
        {"content": "Read the auth module", "status": "completed"},
        {"content": "Fix the failing test", "status": "in_progress"},
        {"content": "Run the full suite", "status": "pending"},
    ]


def test_todowrite_activeform_fallback_and_unknown_status(tmp_path):
    records = [
        _user_prompt("Plan the work."),
        _assistant(
            "Planning.",
            tools=[{"id": "td1", "name": "TodoWrite", "input": {"todos": [
                {"activeForm": "Investigating the bug", "status": "in_progress"},
                {"content": "Ship it", "status": "bogus-status"},
            ]}}],
        ),
        _tool_result("td1"),
    ]
    _write_transcript(tmp_path, "todo-2", records)
    story = build_session_story("todo-2", projects_root=tmp_path)
    todos = story["steps"][0]["tools"][0]["todos"]
    # activeForm fills in for a missing content; an unknown status -> "pending".
    assert todos == [
        {"content": "Investigating the bug", "status": "in_progress"},
        {"content": "Ship it", "status": "pending"},
    ]


def test_todowrite_malformed_payload_has_no_todos(tmp_path):
    records = [
        _user_prompt("Go."),
        _assistant("Working.", tools=[
            {"id": "td1", "name": "TodoWrite", "input": {"todos": "not-a-list"}},
        ]),
        _tool_result("td1"),
    ]
    _write_transcript(tmp_path, "todo-3", records)
    story = build_session_story("todo-3", projects_root=tmp_path)
    tool = story["steps"][0]["tools"][0]
    assert tool["name"] == "TodoWrite"
    assert "todos" not in tool


def test_todowrite_step_preserves_todos_list_and_statuses(tmp_path):
    """Ticket #67: a TodoWrite step must carry its real ``todos`` payload
    (``{"todos": [{content, status}]}``), not read a nonexistent one-line key.

    This is the "where it left off / what's incomplete" signal the resume-brief
    OPEN section consumes, so the pending/in_progress items must survive parsing.
    """
    records = [
        _user_prompt("Ship the resume-brief feature."),
        _assistant(
            "Tracking the remaining work.",
            tools=[{
                "id": "td1",
                "name": "TodoWrite",
                "input": {"todos": [
                    {"content": "read the codebase", "status": "completed"},
                    {"content": "wire the SessionStart hook", "status": "in_progress"},
                    {"content": "open the PR", "status": "pending"},
                    # unknown status normalises to pending; activeForm fallback
                    {"activeForm": "writing tests", "status": "weird"},
                ]},
            }],
        ),
        _tool_result("td1", is_error=False),
        _assistant("Done for now."),
    ]
    _write_transcript(tmp_path, "sess-todo", records)
    story = build_session_story("sess-todo", projects_root=tmp_path)
    assert story is not None

    todo_step = story["steps"][0]
    tool = todo_step["tools"][0]
    assert tool["name"] == "TodoWrite"
    todos = tool["todos"]
    assert {"content": "wire the SessionStart hook", "status": "in_progress"} in todos
    assert {"content": "open the PR", "status": "pending"} in todos
    # activeForm fallback with an unknown status -> pending
    assert {"content": "writing tests", "status": "pending"} in todos
    # the rollup label summarises, never the missing single-arg key
    assert "in progress" in tool["label"] and "pending" in tool["label"]
