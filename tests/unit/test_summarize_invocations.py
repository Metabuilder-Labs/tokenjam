"""Observed invocation counts for on-demand agent files.

The on-demand half of `summarize`'s pricing multiplies by "how many times was
this actually invoked". That number is OBSERVED from Claude Code transcripts,
never assumed — and the difference between "invoked zero times" (a
measurement) and "there was nothing to measure" (no corpus) has to survive all
the way to the finding, because Critical Rule 28 corollary (a) degrades both
the token and the dollar field together only in the second case.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from tokenjam.core.summarize.invocations import count_invocations
from tokenjam.utils.time_parse import utcnow


def _window():
    return utcnow() - timedelta(days=30), utcnow() + timedelta(hours=1)


def _write(root, project, session_id, records):
    d = root / project
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{session_id}.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8",
    )
    return path


def _tool_use(name, tool_input):
    return {"message": {"content": [
        {"type": "tool_use", "id": "t1", "name": name, "input": tool_input},
    ]}}


def _slash(command):
    return {"message": {"content": [
        {"type": "text", "text": f"<command-name>{command}</command-name>"},
    ]}}


def test_counts_each_invocation_shape_by_name(tmp_path):
    _write(tmp_path, "-repo-a", "s1", [
        _tool_use("Skill", {"skill": "browse"}),
        _tool_use("Skill", {"skill": "browse"}),
        _slash("/govern"),
        _tool_use("Task", {"subagent_type": "reviewer"}),
        _tool_use("Agent", {"subagent_type": "reviewer"}),
        _tool_use("Bash", {"command": "ls"}),          # not an invocation
    ])
    since, until = _window()
    counts = count_invocations(since, until, projects_root=tmp_path)

    assert counts.observed is True
    assert counts.sessions_scanned == 1
    assert counts.get("browse") == 2
    assert counts.get("govern") == 1        # leading "/" stripped
    assert counts.get("reviewer") == 2      # Task and Agent both spawn it
    assert counts.get("never-invoked") == 0
    assert counts.total_invocations == 5


def test_plugin_namespaced_name_also_indexes_under_its_bare_slug(tmp_path):
    """A plugin skill is invoked as `plugin:slug` but its SKILL.md on disk is
    named by the slug alone, which is the key a file path resolves to."""
    _write(tmp_path, "-repo-a", "s1", [
        _tool_use("Skill", {"skill": "superpowers:brainstorming"}),
    ])
    since, until = _window()
    counts = count_invocations(since, until, projects_root=tmp_path)

    assert counts.get("superpowers:brainstorming") == 1
    assert counts.get("brainstorming") == 1
    # ...but it is ONE event, not two -- summing counts.values() would
    # double-count every namespaced invocation.
    assert counts.total_invocations == 1


def test_transcripts_outside_the_window_are_not_counted(tmp_path):
    import os
    import time

    path = _write(tmp_path, "-repo-a", "old", [_tool_use("Skill", {"skill": "ship"})])
    stale = time.time() - 400 * 86400
    os.utime(path, (stale, stale))

    since, until = _window()
    counts = count_invocations(since, until, projects_root=tmp_path)
    assert counts.sessions_scanned == 0
    assert counts.get("ship") == 0
    assert counts.observed is True      # the corpus existed; it just held nothing


def test_sidechain_transcripts_are_skipped(tmp_path):
    """A spawned child's own transcript sits under `<session>/subagents/`. Its
    records are the parent's work, already counted there -- counting the file
    as its own session would inflate the transcript count."""
    _write(tmp_path, "-repo-a", "s1", [_tool_use("Skill", {"skill": "ship"})])
    _write(tmp_path / "-repo-a", "subagents", "agent-x",
           [_tool_use("Skill", {"skill": "ship"})])

    since, until = _window()
    counts = count_invocations(since, until, projects_root=tmp_path)
    assert counts.sessions_scanned == 1
    assert counts.get("ship") == 1


def test_missing_corpus_is_not_a_measurement(tmp_path):
    """`observed=False` is the whole point: no corpus means the caller must
    report NO figure for an on-demand file, not one priced at zero."""
    since, until = _window()
    counts = count_invocations(since, until, projects_root=tmp_path / "nope")
    assert counts.observed is False
    assert counts.get("ship") == 0
    assert counts.total_invocations == 0


def test_malformed_records_never_raise(tmp_path):
    d = tmp_path / "-repo-a"
    d.mkdir(parents=True)
    (d / "s1.jsonl").write_text(
        "not json\n"
        + json.dumps({"message": {"content": "a bare string"}}) + "\n"
        + json.dumps({"message": {"content": [{"type": "tool_use", "name": "Skill",
                                               "input": "not a dict"}]}}) + "\n"
        + json.dumps({"message": None}) + "\n"
        + json.dumps({"message": {"content": [{"type": "tool_use", "name": "Skill",
                                               "input": {"skill": "ship"}}]}}) + "\n",
        encoding="utf-8",
    )
    since, until = _window()
    counts = count_invocations(since, until, projects_root=tmp_path)
    assert counts.get("ship") == 1


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_nameless_invocation_is_not_counted(tmp_path, blank):
    _write(tmp_path, "-repo-a", "s1", [_tool_use("Skill", {"skill": blank})])
    since, until = _window()
    counts = count_invocations(since, until, projects_root=tmp_path)
    assert counts.total_invocations == 0
    assert counts.counts == {}


def test_collects_each_session_recorded_working_directory(tmp_path):
    """The corpus walk is the expensive part, so the working directories come
    back from the pass that was already reading every record. They are what
    `core/summarize/repo_roots` turns into the scanned file population."""
    _write(tmp_path, "-repo-a", "s1", [
        {"cwd": "/code/alpha"},
        _tool_use("Skill", {"skill": "browse"}),
        {"cwd": "/code/alpha"},
    ])
    _write(tmp_path, "-repo-b", "s2", [
        _tool_use("Skill", {"skill": "ship"}),
        {"cwd": "/code/beta"},
    ])
    since, until = _window()

    counts = count_invocations(since, until, projects_root=tmp_path)

    assert set(counts.session_cwds) == {"/code/alpha", "/code/beta"}
    assert counts.total_invocations == 2       # cwd collection changes no count


def test_transcripts_without_a_working_directory_contribute_none(tmp_path):
    """Empty means "the transcripts carried no cwd", not "the sessions ran
    nowhere" — no path is invented for them."""
    _write(tmp_path, "-repo-a", "s1", [_tool_use("Skill", {"skill": "browse"})])
    since, until = _window()

    counts = count_invocations(since, until, projects_root=tmp_path)

    assert counts.session_cwds == ()
    assert counts.observed is True
