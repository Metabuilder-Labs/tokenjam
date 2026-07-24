"""Unit tests for the cross-session relearn aggregator (core/optimize/analyzers/relearn.py).

Mirrors ``test_transcript.py``'s fixture style — hand-written Claude Code
on-disk JSONL records, no I/O beyond a ``tmp_path`` projects root. The
``claude`` CLI distill pass is never invoked in these tests (either the
residual bucket is empty, or ``distill_enabled=False`` is passed explicitly)
so nothing here shells out.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenjam.core.optimize.analyzers.relearn import (
    MIN_RECURRING_SESSIONS,
    FailureEpisode,
    analyze_relearns,
    classify_known_family,
    cluster_failures,
    compute_relearn_finding,
    extract_failures_for_session,
    is_already_codified,
)


# --- Fixture builders (mirrors test_transcript.py) ----------------------------

def _user_prompt(text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _assistant(text: str | None, tools: list[dict] | None = None,
               ts: str = "2026-06-15T09:11:36.133Z") -> dict:
    content: list[dict] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    for t in tools or []:
        content.append({"type": "tool_use", "id": t["id"], "name": t["name"],
                         "input": t.get("input", {})})
    return {"type": "assistant", "timestamp": ts,
            "message": {"role": "assistant", "model": "claude-opus-4-8", "content": content}}


def _tool_error(tool_use_id: str, error_text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": tool_use_id, "is_error": True,
        "content": error_text,
    }]}}


def _tool_ok(tool_use_id: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": tool_use_id, "content": "ok",
    }]}}


def _write_transcript(projects_root: Path, project: str, session_id: str, records: list[dict]) -> Path:
    project_dir = projects_root / project
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _cwd_confusion_session(root: Path, project: str, session_id: str) -> None:
    """One session hitting the wrong-cwd Bash error."""
    records = [
        _user_prompt("run the build"),
        _assistant("Running the build.", tools=[
            {"id": "t1", "name": "Bash", "input": {"command": "cd orchestrator && make"}},
        ]),
        _tool_error("t1", "(eval):cd:1: no such file or directory: orchestrator"),
        _assistant("Let me check the path first.", tools=[
            {"id": "t2", "name": "Bash", "input": {"command": "pwd"}},
        ]),
        _tool_ok("t2"),
    ]
    _write_transcript(root, project, session_id, records)


def _edit_before_read_session(root: Path, project: str, session_id: str) -> None:
    records = [
        _user_prompt("fix the bug"),
        _assistant("Editing directly.", tools=[
            {"id": "e1", "name": "Edit", "input": {"file_path": "src/app.py"}},
        ]),
        _tool_error("e1", "File has not been read yet. Read it first before writing to it."),
        _assistant("Reading first.", tools=[
            {"id": "e2", "name": "Read", "input": {"file_path": "src/app.py"}},
        ]),
        _tool_ok("e2"),
    ]
    _write_transcript(root, project, session_id, records)


def _command_not_found_session(root: Path, project: str, session_id: str) -> None:
    records = [
        _user_prompt("run the script"),
        _assistant("Running it.", tools=[
            {"id": "c1", "name": "Bash", "input": {"command": "python script.py"}},
        ]),
        _tool_error("c1", "Exit code 127\nbash: python: command not found"),
        _assistant("Trying python3.", tools=[
            {"id": "c2", "name": "Bash", "input": {"command": "python3 script.py"}},
        ]),
        _tool_ok("c2"),
    ]
    _write_transcript(root, project, session_id, records)


def _clean_session(root: Path, project: str, session_id: str) -> None:
    """A session with no errors at all — must contribute nothing."""
    records = [
        _user_prompt("say hi"),
        _assistant("Hello!"),
    ]
    _write_transcript(root, project, session_id, records)


# --- Pure classifier tests -----------------------------------------------------

def test_classify_cwd_confusion():
    assert classify_known_family(
        "Bash", "(eval):cd:1: no such file or directory: orchestrator"
    ) == "cwd_confusion"


def test_classify_edit_before_read():
    assert classify_known_family(
        "Edit", "File has not been read yet. Read it first before writing to it."
    ) == "edit_before_read"


def test_classify_sleep_chain_needs_leading_sleep_label():
    # Error text matches, but the command doesn't lead with sleep -> no match.
    assert classify_known_family("Bash", "Blocked: foreground sleep", label="ls && sleep 5") is None
    # Leading sleep command -> matches.
    assert classify_known_family("Bash", "Blocked: foreground sleep", label="sleep 5 && ls") == "sleep_chain"


def test_classify_no_match_returns_none():
    assert classify_known_family("Bash", "some unrelated transient network blip") is None


def test_classify_empty_error_returns_none():
    assert classify_known_family("Bash", "") is None


def test_classify_webfetch_domain_blocked():
    # Real wording validated against the local corpus, not "not allowed"/"blocked".
    assert classify_known_family(
        "WebFetch", "Claude Code is unable to fetch from www.reddit.com"
    ) == "webfetch_domain_blocked"


def test_classify_read_offset_array_not_shadowed_by_deferred_tool_cold():
    # Real evidence (validated against the local corpus, 2026-07-14): this
    # exact wording matches BOTH deferred_tool_cold's generic
    # "inputvalidationerror" pattern (tools=None -> any tool) AND
    # read_offset_malformed's tools={"Read"}-scoped pattern. The more
    # specific family must win, or ~35% of deferred_tool_cold's Read-tool
    # evidence gets mislabeled with the wrong fix (family-ordering shadow bug).
    text = (
        "InputValidationError: Read failed due to the following issue:\n"
        "The parameter `offset` type is expected as `number` but provided as `array`"
    )
    assert classify_known_family("Read", text) == "read_offset_malformed"


def test_classify_deferred_tool_cold_still_matches_non_offset_errors():
    # A generic InputValidationError on a DIFFERENT tool/parameter must still
    # land in deferred_tool_cold — the reorder must not swallow its own family.
    text = (
        "InputValidationError: Monitor failed due to the following issues:\n"
        "The required parameter `description` is missing"
    )
    assert classify_known_family("Monitor", text) == "deferred_tool_cold"


def test_command_not_found_is_rung_one_with_real_guidance():
    # Downgraded from rung 5 (no safe automatic config/env writer exists) to
    # a rung-1 CLAUDE.md note with genuinely useful guidance — not a stub.
    from tokenjam.core.optimize.analyzers.relearn import _FAMILY_BY_KEY

    fam = _FAMILY_BY_KEY["command_not_found"]
    assert fam["rung"] == 1
    assert "python3" in fam["fix"]
    assert "mapfile" in fam["fix"] or "shopt" in fam["fix"]


# --- User-decline exclusion (not a relearn) ------------------------------------

def test_user_decline_is_not_a_relearn(tmp_path):
    from tokenjam.core.optimize.analyzers.relearn import is_user_decline

    assert is_user_decline("The user doesn't want to proceed with this tool use.") is True
    assert is_user_decline("Exit plan mode?") is True
    assert is_user_decline("cd: no such file or directory: orchestrator") is False
    assert is_user_decline("") is False


def test_user_decline_excluded_from_extraction(tmp_path):
    records = [
        _user_prompt("do something"),
        _assistant("Trying.", tools=[{"id": "d1", "name": "AskUserQuestion", "input": {}}]),
        _tool_error("d1", "The user doesn't want to proceed with this tool use. The tool call has been cancelled."),
        _assistant("Understood, skipping."),
    ]
    _write_transcript(tmp_path, "-Users-test-decline", "decline-1", records)
    failures = extract_failures_for_session("decline-1", "repo-a", projects_root=tmp_path)
    assert failures == []


# --- Extraction ----------------------------------------------------------------

def test_extract_failures_finds_the_errored_tool(tmp_path):
    _cwd_confusion_session(tmp_path, "-Users-test-a", "sess-1")
    failures = extract_failures_for_session("sess-1", "repo-a", projects_root=tmp_path)
    assert len(failures) == 1
    f = failures[0]
    assert isinstance(f, FailureEpisode)
    assert f.tool_name == "Bash"
    assert "no such file or directory" in f.error_text
    assert f.repo == "repo-a"
    assert f.depth == 0


def test_extract_failures_missing_session_returns_empty(tmp_path):
    assert extract_failures_for_session("nope", "repo-a", projects_root=tmp_path) == []


def test_extract_failures_clean_session_returns_empty(tmp_path):
    _clean_session(tmp_path, "-Users-test-a", "sess-clean")
    assert extract_failures_for_session("sess-clean", "repo-a", projects_root=tmp_path) == []


# --- Clustering ------------------------------------------------------------

def test_cluster_groups_same_family_across_sessions(tmp_path):
    failures = [
        FailureEpisode("s1", "repo-a", None, "Bash", "cd x", "cd: no such file or directory: x", "act", False, 0),
        FailureEpisode("s2", "repo-b", None, "Bash", "cd y", "cd: no such file or directory: y", "act", False, 0),
    ]
    clusters = cluster_failures(failures)
    assert len(clusters) == 1
    cluster = next(iter(clusters.values()))
    assert cluster.family_key == "cwd_confusion"
    assert cluster.session_ids == {"s1", "s2"}
    assert cluster.repos == {"repo-a", "repo-b"}


def test_cluster_separates_different_families(tmp_path):
    failures = [
        FailureEpisode("s1", "repo-a", None, "Bash", "cd x", "no such file or directory", "act", False, 0),
        FailureEpisode("s2", "repo-a", None, "Edit", "a.py", "File has not been read yet.", "act", False, 0),
    ]
    clusters = cluster_failures(failures)
    assert len(clusters) == 2


# --- Full pipeline (analyze_relearns) ------------------------------------------

def test_recurring_cluster_surfaces_as_a_proposal(tmp_path):
    for i in range(MIN_RECURRING_SESSIONS):
        _cwd_confusion_session(tmp_path, f"-Users-test-repo{i}", f"cwd-{i}")
    sessions = [(f"cwd-{i}", f"repo{i}") for i in range(MIN_RECURRING_SESSIONS)]

    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)

    assert finding.sessions_scanned == MIN_RECURRING_SESSIONS
    assert len(finding.clusters) == 1
    cluster = finding.clusters[0]
    assert cluster.family_key == "cwd_confusion"
    assert cluster.sessions == MIN_RECURRING_SESSIONS
    assert cluster.rung == 3
    # Spread across 3 distinct repos -> user-global scope (§7).
    assert cluster.scope == "user-global"
    assert cluster.estimated_recoverable_tokens > 0
    assert len(cluster.examples) <= 3
    assert finding.estimated_recoverable_tokens == cluster.estimated_recoverable_tokens


def test_command_not_found_proposal_is_rung_one_note(tmp_path):
    for i in range(MIN_RECURRING_SESSIONS):
        _command_not_found_session(tmp_path, f"-Users-test-cnf{i}", f"cnf-{i}")
    sessions = [(f"cnf-{i}", f"repo{i}") for i in range(MIN_RECURRING_SESSIONS)]

    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)

    assert len(finding.clusters) == 1
    cluster = finding.clusters[0]
    assert cluster.family_key == "command_not_found"
    assert cluster.rung == 1
    assert "python3" in cluster.proposed_fix


def test_below_threshold_cluster_is_dropped(tmp_path):
    for i in range(MIN_RECURRING_SESSIONS - 1):
        _edit_before_read_session(tmp_path, f"-Users-test-repo{i}", f"ebr-{i}")
    sessions = [(f"ebr-{i}", f"repo{i}") for i in range(MIN_RECURRING_SESSIONS - 1)]

    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)

    assert finding.clusters == []
    assert finding.failures_examined == MIN_RECURRING_SESSIONS - 1
    assert finding.min_sessions == MIN_RECURRING_SESSIONS


def test_config_lowers_recurrence_bar_surfaces_previously_hidden_cluster(tmp_path):
    """The exact data from test_below_threshold_cluster_is_dropped clusters
    nothing at the default bar; passing a lower min_sessions (what run()
    threads from [optimize] min_recurring_sessions) surfaces it."""
    for i in range(MIN_RECURRING_SESSIONS - 1):
        _edit_before_read_session(tmp_path, f"-Users-test-repo{i}", f"ebr-{i}")
    sessions = [(f"ebr-{i}", f"repo{i}") for i in range(MIN_RECURRING_SESSIONS - 1)]

    default_finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)
    assert default_finding.clusters == []

    lowered_finding = analyze_relearns(
        sessions, projects_root=tmp_path, distill_enabled=False,
        min_sessions=MIN_RECURRING_SESSIONS - 1,
    )
    assert len(lowered_finding.clusters) == 1
    assert lowered_finding.clusters[0].sessions == MIN_RECURRING_SESSIONS - 1
    assert lowered_finding.min_sessions == MIN_RECURRING_SESSIONS - 1


def test_run_reads_min_recurring_sessions_from_ctx_config(tmp_path, monkeypatch):
    """The registered run(ctx) entry point reads
    ctx.config.optimize.min_recurring_sessions (not just analyze_relearns's
    direct min_sessions param)."""
    from tokenjam.core.config import OptimizeConfig, TjConfig
    from tokenjam.core.optimize.analyzers.relearn import run as run_relearn
    from tokenjam.core.optimize.types import AnalyzerContext, OptimizeReport, WindowSummary

    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))
    for i in range(MIN_RECURRING_SESSIONS - 1):
        _edit_before_read_session(tmp_path, f"-Users-test-repo{i}", f"ebr-{i}")

    since = datetime.now(timezone.utc) - timedelta(days=1)
    summary = WindowSummary(
        since=since, until=datetime.now(timezone.utc), days=1.0, sessions=0,
        spans=0, total_tokens=0, total_cost_usd=0.0, thin_data=False,
    )

    def _ctx(config) -> AnalyzerContext:
        return AnalyzerContext(
            conn=None, config=config, since=since, until=datetime.now(timezone.utc),
            agent_id=None, window_days=1.0, summary=summary,
            report=OptimizeReport(window=summary),
        )

    default_ctx = _ctx(TjConfig(version="1"))
    run_relearn(default_ctx)
    assert default_ctx.report.findings["relearn"].clusters == []

    lowered_ctx = _ctx(TjConfig(
        version="1",
        optimize=OptimizeConfig(min_recurring_sessions=MIN_RECURRING_SESSIONS - 1),
    ))
    run_relearn(lowered_ctx)
    assert len(lowered_ctx.report.findings["relearn"].clusters) == 1


def test_extract_failures_transcript_cache_dir_opt_in_matches_uncached(tmp_path):
    """Passing `transcript_cache_dir` must not change what's extracted."""
    _cwd_confusion_session(tmp_path, "-Users-test-cache", "sess-cache")

    uncached = extract_failures_for_session("sess-cache", "repo-a", projects_root=tmp_path)
    cached = extract_failures_for_session(
        "sess-cache", "repo-a", projects_root=tmp_path,
        transcript_cache_dir=tmp_path / "cache",
    )
    assert [f.tool_name for f in cached] == [f.tool_name for f in uncached]
    assert [f.error_text for f in cached] == [f.error_text for f in uncached]


def test_extract_failures_warm_cache_skips_reparsing(tmp_path, monkeypatch):
    _cwd_confusion_session(tmp_path, "-Users-test-cache", "sess-cache")
    cache_dir = tmp_path / "cache"

    first = extract_failures_for_session(
        "sess-cache", "repo-a", projects_root=tmp_path, transcript_cache_dir=cache_dir,
    )
    assert first  # sanity: a real failure, not an empty no-op

    def _boom(path):
        raise AssertionError(f"transcript.read_records reparsed {path} on a warm cache run")

    monkeypatch.setattr("tokenjam.core.transcript._parse_records", _boom)

    second = extract_failures_for_session(
        "sess-cache", "repo-a", projects_root=tmp_path, transcript_cache_dir=cache_dir,
    )
    assert [f.tool_name for f in second] == [f.tool_name for f in first]


def test_compute_relearn_finding_cache_invalidates_on_transcript_edit(tmp_path):
    """A session rewritten between two cached scans (its failure resolved,
    say) must be re-parsed, not served a stale extraction."""
    for i in range(MIN_RECURRING_SESSIONS):
        _cwd_confusion_session(tmp_path, f"-Users-test-inv{i}", f"inv-{i}")
    cache_dir = tmp_path / "cache"

    first = compute_relearn_finding(
        projects_root=tmp_path, distill_enabled=False, transcript_cache_dir=cache_dir,
    )
    assert len(first.clusters) == 1  # the recurring cwd-confusion cluster

    # Rewrite one session so it's clean (no more failure) — size + mtime both
    # change, which must invalidate that session's cache entry and drop the
    # cluster below the recurrence bar.
    _clean_session(tmp_path, "-Users-test-inv0", "inv-0")

    second = compute_relearn_finding(
        projects_root=tmp_path, distill_enabled=False, transcript_cache_dir=cache_dir,
    )
    assert second.clusters == []


def test_run_wires_the_persistent_transcript_cache(tmp_path, monkeypatch):
    """The registered `run(ctx)` entry point (the path `tj optimize` and
    `/cost/components` actually exercise) resolves and uses a real cache dir
    from `ctx.config`, not just the standalone functions tested above."""
    from tokenjam.core.config import TjConfig
    from tokenjam.core.optimize.analyzers.relearn import run as run_relearn
    from tokenjam.core.optimize.types import AnalyzerContext, OptimizeReport, WindowSummary

    monkeypatch.setenv("TJ_CLAUDE_PROJECTS_ROOT", str(tmp_path))
    for i in range(MIN_RECURRING_SESSIONS):
        _cwd_confusion_session(tmp_path, f"-Users-test-run{i}", f"run-{i}")

    since = datetime.now(timezone.utc) - timedelta(days=1)
    summary = WindowSummary(
        since=since, until=datetime.now(timezone.utc), days=1.0, sessions=0,
        spans=0, total_tokens=0, total_cost_usd=0.0, thin_data=False,
    )
    config = TjConfig(version="1")

    def _ctx() -> AnalyzerContext:
        return AnalyzerContext(
            conn=None, config=config, since=since, until=datetime.now(timezone.utc),
            agent_id=None, window_days=1.0, summary=summary,
            report=OptimizeReport(window=summary),
        )

    first_ctx = _ctx()
    run_relearn(first_ctx)
    first = first_ctx.report.findings["relearn"]
    assert len(first.clusters) == 1

    def _boom(path):
        raise AssertionError(f"transcript.read_records reparsed {path} on a warm cache run")

    monkeypatch.setattr("tokenjam.core.transcript._parse_records", _boom)

    second_ctx = _ctx()
    run_relearn(second_ctx)
    second = second_ctx.report.findings["relearn"]
    assert len(second.clusters) == 1


def test_single_repo_cluster_scopes_to_project(tmp_path):
    for i in range(MIN_RECURRING_SESSIONS):
        _edit_before_read_session(tmp_path, "-Users-test-onerepo", f"ebr-one-{i}")
    sessions = [(f"ebr-one-{i}", "onerepo") for i in range(MIN_RECURRING_SESSIONS)]

    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)

    assert len(finding.clusters) == 1
    assert finding.clusters[0].scope == "project"
    assert finding.clusters[0].repos == ["onerepo"]


def test_clean_sessions_produce_no_proposals(tmp_path):
    for i in range(5):
        _clean_session(tmp_path, "-Users-test-clean", f"clean-{i}")
    sessions = [(f"clean-{i}", "cleanrepo") for i in range(5)]

    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)

    assert finding.clusters == []
    assert finding.failures_examined == 0
    assert finding.estimated_recoverable_tokens is None


# --- Novelty filter -------------------------------------------------------------

def test_already_codified_cluster_is_dropped(tmp_path):
    for i in range(MIN_RECURRING_SESSIONS):
        _cwd_confusion_session(tmp_path, f"-Users-test-doc{i}", f"cwd-doc-{i}")
    sessions = [(f"cwd-doc-{i}", f"repo{i}") for i in range(MIN_RECURRING_SESSIONS)]

    doc_text = "known gotcha: always confirm cwd — errors read 'no such file or directory'."

    finding = analyze_relearns(
        sessions, projects_root=tmp_path, distill_enabled=False, codified_doc_text=doc_text,
    )

    assert finding.clusters == []
    assert finding.dropped_codified == 1


def test_is_already_codified_requires_the_exact_phrase():
    from tokenjam.core.optimize.analyzers.relearn import _RawCluster

    cluster = _RawCluster(signature="cwd_confusion", family_key="cwd_confusion", title="x")
    # A generic, partial mention isn't enough — the exact phrase must appear.
    assert is_already_codified(cluster, "mentions cwd but nothing else relevant") is False
    assert is_already_codified(cluster, "watch for 'no such file or directory' errors") is True
    assert is_already_codified(cluster, "") is False


def test_is_already_codified_never_drops_a_residual_cluster():
    """No known family_key -> always treated as novel (see is_already_codified
    docstring: a wrongful drop here would silently hide new signal)."""
    from tokenjam.core.optimize.analyzers.relearn import _RawCluster

    cluster = _RawCluster(signature="Bash:some weird thing", family_key=None, title="Bash: some weird thing")
    assert is_already_codified(cluster, "some weird thing is a documented gotcha") is False


# --- Distill confidence gate ----------------------------------------------------
# Real confabulation examples pulled from the local corpus (2026-07-14): distill,
# fed only bare/near-empty evidence from a benign multi-command `&&` Bash chain
# (last command exits nonzero with no error text of its own — often a trailing
# grep/find "no match"), invented FIVE different confident-but-wrong "fixes" for
# the SAME phenomenon. The gate must suppress all five while leaving genuinely
# well-evidenced clusters (branch-already-exists, EISDIR) untouched.

_REAL_THIN_SAMPLES: dict[str, list[str]] = {
    # -> was confabulated as "bash_stderr_missing"
    "bash_stderr_missing": ["Exit code 1", "Exit code 1", "Exit code 2"],
    # -> was confabulated as "bash_env_setup"
    "bash_env_setup": ["Exit code 1\n0", "Exit code 1\n0", "Exit code 7\n000"],
    # -> was confabulated as "bash_error_reporting"
    "bash_error_reporting": ["Exit code 1\n---", "Exit code 2\n---", "Exit code 1\n---"],
    # -> was confabulated as "bash_output_buffer_limit" — real content, but it's
    # leftover `ls -la` stdout from an EARLIER, successful chain step, not an
    # error description of why the chain's last command failed.
    "bash_output_buffer_limit": [
        "Exit code 1\ntotal 240\ndrwxr-xr-x@ 21 anshs  staff    672 Jun 28 14:44 .\n"
        "drwxr-xr-x@ 16 anshs  staff    512 Jun 28 14:44 ..\n"
        "-rw-r--r--@  1 anshs  staff     75 Jun 28 14:44 .git",
        "Exit code 1\ntotal 280\ndrwxr-xr-x@ 25 anshs  staff    800 Jun 25 17:52 .\n"
        "drwxr-xr-x@ 33 anshs  staff   1056 Jun 25 17:53 ..",
    ],
    # -> was confabulated as "bash_output_truncation"
    "bash_output_truncation": [
        "Exit code 1\ntotal 288\ndrwxr-xr-x@ 24 anshs  staff    768 Jul  3 12:56 .\n"
        "drwxr-xr-x@ 16 anshs  staff    512 Jul  3 12:56 ..\n"
        "-rw-r--r--@  1 anshs  staff    553 Jul  3 12:56 .dockerignore",
        "Exit code 1\ntotal 400\ndrwxr-xr-x@ 32 anshs  staff   1024 Jul  3 00:37 .",
    ],
}

_REAL_LEGIT_SAMPLES: dict[str, list[str]] = {
    "branch_already_exists": [
        "Exit code 128\nfatal: a branch named 'ticket-28' already exists",
        "Exit code 128\nfatal: a branch named 'ticket-322' already exists",
        "Exit code 128\nfatal: a branch named 'ticket-22' already exists",
    ],
    "read_tool_dir_not_file": [
        "EISDIR: illegal operation on a directory, read "
        "'/Users/anshs/Folder/code/shiploop.wt/ticket-5/shiploop/templates'",
        "EISDIR: illegal operation on a directory, read "
        "'/Users/anshs/Folder/code/vibelab.wt/ticket-5'",
    ],
}


@pytest.mark.parametrize("family, samples", _REAL_THIN_SAMPLES.items())
def test_confabulated_bash_chain_evidence_is_too_thin(family, samples):
    from tokenjam.core.optimize.analyzers.relearn import (
        FailureEpisode,
        _RawCluster,
        _evidence_too_thin_for_distill,
    )

    cluster = _RawCluster(
        signature=f"Bash:{family}", family_key=None, title=family,
        failures=[
            FailureEpisode(f"s{i}", "repo", None, "Bash", "cmd1 && cmd2", s, "act", False, 0)
            for i, s in enumerate(samples)
        ],
    )
    assert _evidence_too_thin_for_distill(cluster) is True


@pytest.mark.parametrize("family, samples", _REAL_LEGIT_SAMPLES.items())
def test_legit_evidence_is_not_too_thin(family, samples):
    from tokenjam.core.optimize.analyzers.relearn import (
        FailureEpisode,
        _RawCluster,
        _evidence_too_thin_for_distill,
    )

    cluster = _RawCluster(
        signature=f"X:{family}", family_key=None, title=family,
        failures=[
            FailureEpisode(f"s{i}", "repo", None, "Bash", "cmd", s, "act", False, 0)
            for i, s in enumerate(samples)
        ],
    )
    assert _evidence_too_thin_for_distill(cluster) is False


def test_confidence_gate_suppresses_confabulations_even_with_a_warm_cache(tmp_path):
    """The gate runs BEFORE the cache is ever consulted — even a stale cache
    entry from before this fix (holding the exact real confabulated answer)
    must not resurrect a suppressed cluster."""
    import hashlib
    import json

    from tokenjam.core.optimize.analyzers.relearn import (
        MIN_DISTILL_CLUSTER_SESSIONS,
        FailureEpisode,
        _RawCluster,
        apply_distill_to_residual,
    )

    cache_dir = tmp_path / "distill_cache"
    clusters = []
    fake_answers = {
        "bash_stderr_missing": {
            "title": "Bash tool suppressing stderr output", "family_key": "bash_stderr_missing",
            "fix": "Verify the Bash tool is capturing and displaying stderr.",
        },
        "bash_env_setup": {
            "title": "Shell environment initialization incomplete", "family_key": "bash_env_setup",
            "fix": "Ensure PATH, working directory, and shell state are explicitly configured.",
        },
    }
    for family, samples in _REAL_THIN_SAMPLES.items():
        n = max(len(samples), MIN_DISTILL_CLUSTER_SESSIONS)
        failures = [
            FailureEpisode(f"{family}-s{i}", "repo", None, "Bash", "cmd1 && cmd2",
                            samples[i % len(samples)], "act", False, 0)
            for i in range(n)
        ]
        cluster = _RawCluster(signature=f"Bash:{family}", family_key=None, title=family, failures=failures)
        clusters.append(cluster)
        # Pre-seed the cache with the REAL confabulated answer where we have
        # one — proves the gate short-circuits before the cache read.
        answer = fake_answers.get(family)
        if answer:
            payload = cluster.signature + "|" + "|".join(sorted(f.error_text for f in cluster.failures[:10]))
            cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{cache_key}.json").write_text(json.dumps(answer), encoding="utf-8")

    result = apply_distill_to_residual(clusters, cache_dir=cache_dir, enabled=True)
    result_family_keys = {c.family_key for c in result}
    for family in _REAL_THIN_SAMPLES:
        assert f"distilled:{family}" not in result_family_keys
    # Nothing survives at all — every seeded cluster here was thin evidence.
    assert result == []


# --- CLI text-view rendering regression -------------------------------------
# `relearn` was registered in ANALYZER_REGISTRY/ANALYZER_ORDER but never wired
# into cmd_optimize's `_FINDING_RENDERERS` dispatch table, so `_rank_findings`
# silently dropped it and `tj optimize relearn` (text view) fell through to
# the generic "No candidates flagged in this window" empty state — even with
# real clusters sitting in `--json`.

def test_relearn_in_click_choices_and_renderer():
    """relearn appears in the positional Click choices (auto-derived from the
    registry) and has a human-readable renderer wired into the dispatch
    table — mirrors the same regression guard used for `verbosity`."""
    from tokenjam.cli.cmd_optimize import _FINDING_RENDERERS, cmd_optimize

    findings_param = next(
        p for p in cmd_optimize.params if getattr(p, "name", None) == "findings"
    )
    assert "relearn" in findings_param.type.choices
    assert "relearn" in _FINDING_RENDERERS


def test_render_relearn_shows_clusters_without_error(tmp_path, capsys):
    """The finding renders through the CLI dispatch path and surfaces the
    cluster signature + occurrences + rung — not a generic empty state."""
    from tokenjam.cli.cmd_optimize import _render_relearn

    for i in range(MIN_RECURRING_SESSIONS):
        _cwd_confusion_session(tmp_path, f"-Users-test-render{i}", f"render-{i}")
    sessions = [(f"render-{i}", f"repo{i}") for i in range(MIN_RECURRING_SESSIONS)]
    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)
    assert finding.clusters  # sanity: the analyzer actually found the cluster

    for mode in ("api", "subscription", "local", "unknown"):
        _render_relearn(finding, pricing_mode=mode, marker="①")
    out = capsys.readouterr().out
    assert "cwd_confusion" in out
    assert f"{finding.clusters[0].occurrences}" in out
    assert "rung 3" in out
    assert "No candidates flagged" not in out


def test_render_report_surfaces_relearn_clusters_instead_of_no_candidates(tmp_path, capsys):
    """End-to-end regression: a report whose only finding is a
    populated `relearn` cluster set must NOT fall through to the
    cost-optimizer's generic "No candidates flagged" empty state."""
    from tokenjam.cli.cmd_optimize import _render_report
    from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
    from tokenjam.utils.time_parse import utcnow

    for i in range(MIN_RECURRING_SESSIONS):
        _cwd_confusion_session(tmp_path, f"-Users-test-e2e{i}", f"e2e-{i}")
    sessions = [(f"e2e-{i}", f"repo{i}") for i in range(MIN_RECURRING_SESSIONS)]
    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)
    assert finding.clusters

    now = utcnow()
    report = OptimizeReport(
        window=WindowSummary(
            since=now, until=now, days=365, sessions=len(sessions), spans=0,
            total_tokens=100_000, total_cost_usd=0.0, thin_data=False,
        ),
        downgrade=None,
        findings={"relearn": finding},
    )
    _render_report(report, agent=None, requested=["relearn"], pricing_mode="local")
    out = capsys.readouterr().out
    assert "No candidates flagged" not in out
    assert "cwd_confusion" in out


def test_render_report_surfaces_clusters_even_in_a_huge_token_window(tmp_path, capsys):
    """Collapse variant: a relearn finding must surface its clusters in
    full even when the window's total tokens are enormous (a heavy
    `tj optimize relearn --since 365d` run). Relearns are recurring-failure
    clusters, not a token-reclamation finding, so a huge window denominator
    must NOT push them below DE_MINIMIS_SHARE and collapse them into the
    "Minor findings — ~0.0% of window tokens" pointer — that hides the headline
    self-improve signal exactly as the "No candidates flagged" empty state did.
    """
    from tokenjam.cli.cmd_optimize import _render_report
    from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
    from tokenjam.utils.time_parse import utcnow

    for i in range(MIN_RECURRING_SESSIONS):
        _cwd_confusion_session(tmp_path, f"-Users-test-huge{i}", f"huge-{i}")
    sessions = [(f"huge-{i}", f"repo{i}") for i in range(MIN_RECURRING_SESSIONS)]
    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)
    assert finding.clusters
    assert finding.estimated_recoverable_tokens  # a positive, rank-able estimate

    now = utcnow()
    report = OptimizeReport(
        window=WindowSummary(
            # 2B tokens — the cluster's occurrence×heuristic estimate is a ~0%
            # share of this window, the exact condition that collapsed it.
            since=now, until=now, days=365, sessions=len(sessions), spans=0,
            total_tokens=2_000_000_000, total_cost_usd=0.0, thin_data=False,
        ),
        downgrade=None,
        findings={"relearn": finding},
    )
    _render_report(report, agent=None, requested=["relearn"], pricing_mode="local")
    out = capsys.readouterr().out
    assert "cwd_confusion" in out                 # clusters surfaced in full
    assert f"{finding.clusters[0].occurrences}" in out
    assert "Minor findings" not in out            # NOT collapsed to the pointer
    assert "of window tokens" not in out          # no de-minimis token framing
    assert "No candidates flagged" not in out


# --- Review inbox monthly-basis fields (§1/§2) --------------------------------
# Relearn scans unbounded on-disk history, so there's no "the window IS a
# month" shortcut the way a fixed-window cost analyzer has. These fields
# extrapolate the corpus's OWN observed timespan to 30 days instead — see
# `_corpus_window_days`/`_monthly_scale` and the "Recoverable-savings
# contract" note in model_downgrade.py. The window-basis
# `estimated_recoverable_tokens` field (asserted elsewhere in this file) is
# UNCHANGED by any of this — Overview/Optimize keep reading that field.

def _cwd_confusion_session_at(root: Path, project: str, session_id: str, ts: str) -> None:
    """Same fixture as `_cwd_confusion_session`, with a controllable
    timestamp on the erroring turn so the corpus's observed window span is
    something other than "everything at the same instant"."""
    records = [
        _user_prompt("run the build"),
        _assistant("Running the build.", tools=[
            {"id": "t1", "name": "Bash", "input": {"command": "cd orchestrator && make"}},
        ], ts=ts),
        _tool_error("t1", "(eval):cd:1: no such file or directory: orchestrator"),
        _assistant("Let me check the path first.", tools=[
            {"id": "t2", "name": "Bash", "input": {"command": "pwd"}},
        ], ts=ts),
        _tool_ok("t2"),
    ]
    _write_transcript(root, project, session_id, records)


def test_monthly_fields_extrapolate_from_the_corpus_own_observed_window(tmp_path):
    # Three occurrences spread across exactly 10 observed days -> a 3x
    # (30/10) extrapolation to the monthly figure, not a raw multiply-by-30.
    days = [0, 5, 10]
    for i, day in enumerate(days):
        ts = (datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(days=day)).isoformat().replace("+00:00", "Z")
        _cwd_confusion_session_at(tmp_path, f"-Users-test-mo{i}", f"mo-{i}", ts)
    sessions = [(f"mo-{i}", f"repo{i}") for i in range(len(days))]

    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)

    assert finding.window_days == pytest.approx(10.0)
    cluster = finding.clusters[0]
    assert cluster.estimated_monthly_tokens == round(cluster.estimated_recoverable_tokens * 3.0)
    # The window-basis field is untouched — Overview/Optimize still read it.
    assert cluster.estimated_recoverable_tokens == cluster.occurrences * 1_500
    assert finding.estimated_monthly_tokens == cluster.estimated_monthly_tokens
    # No DB connection was given, so there's no blended rate to derive a
    # dollar figure from — tokens-only, exactly the mockup's fallback.
    assert cluster.estimated_monthly_usd is None
    assert cluster.monthly_rate_basis == ""


def test_single_timestamp_corpus_floors_the_window_to_one_day(tmp_path):
    # All three occurrences at the same instant (the default fixture
    # behavior every other test in this file relies on) -> a degenerate
    # zero-length span, clamped to a 1-day floor rather than an undefined or
    # infinite scale (see `_corpus_window_days`'s floor).
    for i in range(MIN_RECURRING_SESSIONS):
        _cwd_confusion_session(tmp_path, f"-Users-test-flat{i}", f"flat-{i}")
    sessions = [(f"flat-{i}", f"repo{i}") for i in range(MIN_RECURRING_SESSIONS)]

    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False)

    assert finding.window_days == 1.0
    cluster = finding.clusters[0]
    assert cluster.estimated_monthly_tokens == cluster.estimated_recoverable_tokens * 30


class _FakeSpanConn:
    """Minimal stand-in for a DuckDB connection: `_blended_dollar_rate` only
    ever calls `.execute(sql, params).fetchall()` on it, so this returns
    canned `(provider, model, cost, tokens)` rows regardless of the SQL text."""
    def __init__(self, rows):
        self._rows = rows

    def execute(self, _sql, _params):
        return self

    def fetchall(self):
        return self._rows


def test_blended_dollar_rate_names_the_models_it_derived_from():
    from tokenjam.core.optimize.analyzers.relearn import _blended_dollar_rate

    conn = _FakeSpanConn([("anthropic", "claude-sonnet-5", 3.0, 1_000_000)])
    rate, basis = _blended_dollar_rate(conn, {"s1", "s2"})

    assert rate == pytest.approx(3.0 / 1_000_000)
    assert "anthropic/claude-sonnet-5" in basis
    assert "blended" in basis


def test_blended_dollar_rate_never_invents_a_rate_with_no_conn_or_sessions():
    from tokenjam.core.optimize.analyzers.relearn import _blended_dollar_rate

    assert _blended_dollar_rate(None, {"s1"}) == (None, "")
    assert _blended_dollar_rate(_FakeSpanConn([]), set()) == (None, "")


def test_blended_dollar_rate_degrades_on_query_failure_never_raises():
    from tokenjam.core.optimize.analyzers.relearn import _blended_dollar_rate

    class _RaisingConn:
        def execute(self, _sql, _params):
            raise RuntimeError("boom")

    assert _blended_dollar_rate(_RaisingConn(), {"s1"}) == (None, "")


def test_monthly_usd_derived_when_conn_has_priced_spans(tmp_path):
    # End-to-end: analyze_relearns(conn=...) stamps a cluster's
    # estimated_monthly_usd from the blended rate observed across its own
    # sessions, not a hardcoded or invented one.
    for i in range(MIN_RECURRING_SESSIONS):
        _cwd_confusion_session(tmp_path, f"-Users-test-usd{i}", f"usd-{i}")
    sessions = [(f"usd-{i}", f"repo{i}") for i in range(MIN_RECURRING_SESSIONS)]
    conn = _FakeSpanConn([("anthropic", "claude-sonnet-5", 3.0, 1_000_000)])

    finding = analyze_relearns(sessions, projects_root=tmp_path, distill_enabled=False, conn=conn)

    cluster = finding.clusters[0]
    expected_rate = 3.0 / 1_000_000
    assert cluster.estimated_monthly_usd == round(cluster.estimated_monthly_tokens * expected_rate, 6)
    assert "claude-sonnet-5" in cluster.monthly_rate_basis
