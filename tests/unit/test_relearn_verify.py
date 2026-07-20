"""Unit tests for the self-improve loop's Verify stage (core.optimize.relearn_verify).

Two layers, mirroring test_relearn.py / test_relearn_apply.py's fixture style:

  1. ``compute_verdict`` is pure (no I/O) — every verdict branch is exercised
     with plain numbers, no fixtures needed.
  2. The I/O-touching helpers (``count_sessions_in_scope``,
     ``measure_recurrence_since``, ``rescan_all``) use hand-written on-disk
     JSONL transcripts under ``tmp_path`` with controlled mtimes (``os.utime``)
     to simulate "before apply" vs "after apply" sessions — nothing here
     ever touches a real ``~/.tj`` or ``~/.claude``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import relearn_apply as pa
from tokenjam.core.optimize import relearn_verify as pv
from tokenjam.core.optimize.analyzers.relearn import _generic_signature

# --- Fixture builders (mirrors test_relearn.py) --------------------------------

def _user_prompt(text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _assistant(text: str | None, tools: list[dict] | None = None) -> dict:
    content: list[dict] = []
    if text is not None:
        content.append({"type": "text", "text": text})
    for t in tools or []:
        content.append({"type": "tool_use", "id": t["id"], "name": t["name"],
                         "input": t.get("input", {})})
    return {"type": "assistant", "timestamp": "2026-06-15T09:11:36.133Z",
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


def _write_transcript(
    root: Path, project: str, session_id: str, records: list[dict],
    mtime: datetime | None = None,
) -> Path:
    project_dir = root / project
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(path, (ts, ts))
    return path


def _cwd_confusion_session(root: Path, project: str, session_id: str, mtime: datetime | None = None) -> None:
    records = [
        _user_prompt("run the build"),
        _assistant("Running the build.", tools=[
            {"id": "t1", "name": "Bash", "input": {"command": "cd orchestrator && make"}},
        ]),
        _tool_error("t1", "(eval):cd:1: no such file or directory: orchestrator"),
        _assistant("Checking the path.", tools=[{"id": "t2", "name": "Bash", "input": {"command": "pwd"}}]),
        _tool_ok("t2"),
    ]
    _write_transcript(root, project, session_id, records, mtime=mtime)


def _clean_session(root: Path, project: str, session_id: str, mtime: datetime | None = None) -> None:
    _write_transcript(root, project, session_id, [_user_prompt("say hi"), _assistant("Hello!")], mtime=mtime)


class _FakeConn:
    """Mimics the subset of a DuckDB connection ``_repo_map_from_db`` uses."""

    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self._rows = rows

    def execute(self, _sql: str) -> "_FakeConn":
        return self

    def fetchall(self) -> list[tuple[str, str]]:
        return self._rows


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


# --- compute_verdict: every branch (pure, no I/O) ------------------------------

def test_verdict_improved_meaningful_drop():
    r = pv.compute_verdict(
        rung=1, enforcement=None,
        baseline_occurrences=10, baseline_total_sessions=10, baseline_sessions=5,
        post_occurrences=2, post_sessions=20,
    )
    assert r["verdict"] == pv.VERDICT_IMPROVED
    assert r["baseline_rate"] == pytest.approx(1.0)
    assert r["post_rate"] == pytest.approx(0.1)
    assert r["realized_tokens_saved"] > 0
    assert r["escalate_candidate"] is False


def test_verdict_no_change_modest_drop():
    # baseline_rate = 1.0/session; post_rate = 0.8/session (ratio 0.8, > 0.7 -> not "improved").
    r = pv.compute_verdict(
        rung=1, enforcement=None,
        baseline_occurrences=10, baseline_total_sessions=10, baseline_sessions=5,
        post_occurrences=16, post_sessions=20,
    )
    assert r["verdict"] == pv.VERDICT_NO_CHANGE
    assert r["realized_tokens_saved"] == 0
    # Rung-1 NOTE + no_change is the "codification != prevention" escalation signal.
    assert r["escalate_candidate"] is True


def test_verdict_regressed_same_or_up():
    r = pv.compute_verdict(
        rung=1, enforcement=None,
        baseline_occurrences=10, baseline_total_sessions=10, baseline_sessions=5,
        post_occurrences=25, post_sessions=20,
    )
    assert r["verdict"] == pv.VERDICT_REGRESSED
    assert r["realized_tokens_saved"] == 0
    assert r["escalate_candidate"] is True


def test_verdict_escalate_candidate_only_for_rung_one():
    r = pv.compute_verdict(
        rung=3, enforcement={"enabled": True},
        baseline_occurrences=10, baseline_total_sessions=10, baseline_sessions=5,
        post_occurrences=25, post_sessions=20,
    )
    assert r["verdict"] == pv.VERDICT_REGRESSED
    assert r["escalate_candidate"] is False   # already at the strongest rung — nothing to escalate to


def test_verdict_insufficient_data_too_few_post_sessions():
    r = pv.compute_verdict(
        rung=1, enforcement=None,
        baseline_occurrences=10, baseline_total_sessions=10, baseline_sessions=5,
        post_occurrences=0, post_sessions=2,
    )
    assert r["verdict"] == pv.VERDICT_INSUFFICIENT_DATA
    assert r["realized_tokens_saved"] is None
    assert "session" in r["reason"]


def test_verdict_insufficient_data_no_usable_baseline():
    r = pv.compute_verdict(
        rung=1, enforcement=None,
        baseline_occurrences=None, baseline_total_sessions=None, baseline_sessions=None,
        post_occurrences=1, post_sessions=10,
    )
    assert r["verdict"] == pv.VERDICT_INSUFFICIENT_DATA


def test_verdict_enforcement_disabled_checked_before_data_volume():
    # Zero post-apply data would otherwise read as insufficient_data — enforcement
    # rungs check "is the hook even live" FIRST since a disabled hook can't have
    # changed anything yet.
    r = pv.compute_verdict(
        rung=3, enforcement={"enabled": False},
        baseline_occurrences=10, baseline_total_sessions=10, baseline_sessions=5,
        post_occurrences=0, post_sessions=0,
    )
    assert r["verdict"] == pv.VERDICT_ENFORCEMENT_DISABLED
    assert r["realized_tokens_saved"] is None


def test_verdict_enforcement_enabled_is_measured_normally():
    r = pv.compute_verdict(
        rung=3, enforcement={"enabled": True},
        baseline_occurrences=10, baseline_total_sessions=10, baseline_sessions=5,
        post_occurrences=1, post_sessions=20,
    )
    assert r["verdict"] == pv.VERDICT_IMPROVED


def test_verdict_not_measurable_reports_insufficient_data_with_reason():
    r = pv.compute_verdict(
        rung=1, enforcement=None,
        baseline_occurrences=10, baseline_total_sessions=10, baseline_sessions=5,
        post_occurrences=0, post_sessions=0,
        measurable=False, unmeasurable_reason="distilled family — can't re-match yet",
    )
    assert r["verdict"] == pv.VERDICT_INSUFFICIENT_DATA
    assert r["reason"] == "distilled family — can't re-match yet"


def test_verdict_normalization_falls_back_to_affected_sessions_when_total_missing():
    r = pv.compute_verdict(
        rung=1, enforcement=None,
        baseline_occurrences=10, baseline_total_sessions=None, baseline_sessions=5,
        post_occurrences=2, post_sessions=20,
    )
    assert r["baseline_rate"] == pytest.approx(2.0)   # 10 / 5 (affected-session fallback)


def test_verdict_normalization_prefers_total_sessions_over_affected():
    # Same occurrences, DIFFERENT denominators — total wins when present.
    r = pv.compute_verdict(
        rung=1, enforcement=None,
        baseline_occurrences=10, baseline_total_sessions=50, baseline_sessions=5,
        post_occurrences=1, post_sessions=20,
    )
    assert r["baseline_rate"] == pytest.approx(0.2)   # 10 / 50, not 10 / 5


def test_verdict_zero_baseline_rate_with_zero_post_rate_is_not_improved():
    r = pv.compute_verdict(
        rung=1, enforcement=None,
        baseline_occurrences=0, baseline_total_sessions=10, baseline_sessions=5,
        post_occurrences=0, post_sessions=10,
    )
    # baseline_occurrences falsy -> no usable baseline was ever really captured.
    assert r["verdict"] == pv.VERDICT_INSUFFICIENT_DATA


# --- count_sessions_in_scope (cheap exposure count) ----------------------------

def test_count_sessions_in_scope_before_and_after(tmp_path):
    cutoff = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _clean_session(tmp_path, "-Users-test-a", "old-1", mtime=cutoff - timedelta(days=2))
    _clean_session(tmp_path, "-Users-test-a", "new-1", mtime=cutoff + timedelta(days=2))
    _clean_session(tmp_path, "-Users-test-a", "new-2", mtime=cutoff + timedelta(days=3))

    assert pv.count_sessions_in_scope(tmp_path, None, None, before=cutoff) == 1
    assert pv.count_sessions_in_scope(tmp_path, None, None, after=cutoff) == 2


def test_count_sessions_in_scope_repo_filter(tmp_path):
    cutoff = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _clean_session(tmp_path, "-Users-test-a", "s-a", mtime=cutoff + timedelta(days=1))
    _clean_session(tmp_path, "-Users-test-b", "s-b", mtime=cutoff + timedelta(days=1))
    conn = _FakeConn([("s-a", "claude-code-repo-a"), ("s-b", "claude-code-repo-b")])

    assert pv.count_sessions_in_scope(tmp_path, conn, "repo-a", after=cutoff) == 1
    assert pv.count_sessions_in_scope(tmp_path, conn, "repo-b", after=cutoff) == 1
    assert pv.count_sessions_in_scope(tmp_path, conn, None, after=cutoff) == 2


def test_count_sessions_in_scope_missing_root_returns_zero(tmp_path):
    assert pv.count_sessions_in_scope(tmp_path / "does-not-exist", None, None) == 0


# --- measure_recurrence_since ----------------------------------------------

def test_measure_recurrence_since_known_family(tmp_path):
    applied_at = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _cwd_confusion_session(tmp_path, "-Users-test-a", "before-1", mtime=applied_at - timedelta(days=1))
    _cwd_confusion_session(tmp_path, "-Users-test-a", "after-1", mtime=applied_at + timedelta(hours=1))
    _cwd_confusion_session(tmp_path, "-Users-test-a", "after-2", mtime=applied_at + timedelta(hours=2))
    _clean_session(tmp_path, "-Users-test-a", "after-3", mtime=applied_at + timedelta(hours=3))

    rec = {
        "family_key": "cwd_confusion", "signature": "cwd_confusion",
        "scope": "user-global", "applied_at": applied_at.isoformat(),
    }
    m = pv.measure_recurrence_since(rec, conn=None, projects_root=tmp_path)
    assert m.measurable is True
    assert m.post_sessions == 3          # 3 sessions after apply (2 afflicted + 1 clean)
    assert m.post_occurrences == 2       # only the cwd_confusion ones counted


def test_measure_recurrence_since_generic_signature_match(tmp_path):
    applied_at = datetime(2026, 7, 1, tzinfo=timezone.utc)
    records = [
        _user_prompt("x"),
        _assistant("try", tools=[{"id": "t1", "name": "SomeTool", "input": {}}]),
        _tool_error("t1", "weird failure 123"),
    ]
    _write_transcript(tmp_path, "-Users-test-a", "after-g1", records, mtime=applied_at + timedelta(hours=1))
    sig = _generic_signature("SomeTool", "weird failure 123")

    rec = {"family_key": None, "signature": sig, "scope": "user-global", "applied_at": applied_at.isoformat()}
    m = pv.measure_recurrence_since(rec, conn=None, projects_root=tmp_path)
    assert m.measurable is True
    assert m.post_occurrences == 1
    assert m.post_sessions == 1


def test_measure_recurrence_since_distilled_family_is_unmeasurable(tmp_path):
    rec = {
        "family_key": "distilled:something", "signature": "distilled:something",
        "scope": "user-global", "applied_at": "2026-07-01T00:00:00+00:00",
    }
    m = pv.measure_recurrence_since(rec, conn=None, projects_root=tmp_path)
    assert m.measurable is False
    assert "distilled" in m.reason


def test_measure_recurrence_since_unknown_family_is_unmeasurable(tmp_path):
    rec = {
        "family_key": "some_unknown_key", "signature": "some_unknown_key",
        "scope": "user-global", "applied_at": "2026-07-01T00:00:00+00:00",
    }
    m = pv.measure_recurrence_since(rec, conn=None, projects_root=tmp_path)
    assert m.measurable is False


def test_measure_recurrence_since_missing_applied_at_is_unmeasurable(tmp_path):
    rec = {"family_key": "cwd_confusion", "signature": "cwd_confusion", "scope": "user-global", "applied_at": None}
    m = pv.measure_recurrence_since(rec, conn=None, projects_root=tmp_path)
    assert m.measurable is False
    assert "applied_at" in m.reason


def test_measure_recurrence_since_scopes_to_project_repo(tmp_path):
    applied_at = datetime(2026, 7, 1, tzinfo=timezone.utc)
    _cwd_confusion_session(tmp_path, "-Users-test-a", "after-a", mtime=applied_at + timedelta(hours=1))
    _cwd_confusion_session(tmp_path, "-Users-test-b", "after-b", mtime=applied_at + timedelta(hours=1))
    conn = _FakeConn([("after-a", "claude-code-repo-a"), ("after-b", "claude-code-repo-b")])

    rec = {
        "family_key": "cwd_confusion", "signature": "cwd_confusion", "scope": "project",
        "repo_root": "/some/path/repo-a", "applied_at": applied_at.isoformat(),
    }
    m = pv.measure_recurrence_since(rec, conn=conn, projects_root=tmp_path)
    assert m.post_sessions == 1
    assert m.post_occurrences == 1


# --- compound_ledger ---------------------------------------------------------

def test_compound_ledger_sums_improved_and_skips_reverted():
    records = [
        {"state": "applied", "verify": {"verdict": "improved", "realized_tokens_saved": 500}},
        {"state": "applied", "verify": {"verdict": "no_change", "realized_tokens_saved": 0}},
        {"state": "applied", "verify": {"verdict": "regressed", "realized_tokens_saved": 0}},
        {"state": "applied", "verify": {"verdict": "enforcement_disabled"}},
        {"state": "applied", "verify": {"verdict": "insufficient_data"}},
        {"state": "applied", "verify": {"verdict": None}},   # never checked -> excluded
        {"state": "reverted", "verify": {"verdict": "improved", "realized_tokens_saved": 999}},
    ]
    ledger = pv.compound_ledger(records)
    assert ledger["total_realized_tokens_saved"] == 500
    assert ledger["verified_count"] == 5
    assert ledger["improved_count"] == 1
    assert ledger["no_change_count"] == 1
    assert ledger["regressed_count"] == 1
    assert ledger["enforcement_disabled_count"] == 1
    assert ledger["insufficient_data_count"] == 1


def test_compound_ledger_empty_is_all_zero():
    ledger = pv.compound_ledger([])
    assert ledger["total_realized_tokens_saved"] == 0
    assert ledger["verified_count"] == 0


# --- rescan_all (end-to-end: real ledger + fixture transcripts) ----------------

def test_rescan_all_updates_the_ledger_end_to_end(cfg, tmp_path):
    applied_at = datetime(2026, 7, 1, tzinfo=timezone.utc)
    projects_root = tmp_path / "projects"
    target = tmp_path / "CLAUDE.md"
    target.write_text("# root\n", encoding="utf-8")

    cluster = {
        "signature": "cwd_confusion", "family_key": "cwd_confusion",
        "title": "cwd confusion", "proposed_fix": "watch your cwd",
        "rung": 1, "sessions": 5, "occurrences": 10,
        "repos": ["demo"], "examples": [],
    }
    result = pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="user-global", go=True)
    fix_id = result["record"]["id"]

    # Freeze applied_at + baseline_total_sessions for a deterministic comparison
    # (apply_relearn_fix stamps utcnow(), which we can't control from here).
    ledger_path = pa.applied_fixes_path(cfg)
    records = json.loads(ledger_path.read_text())
    records[0]["applied_at"] = applied_at.isoformat()
    records[0]["verify"]["baseline_total_sessions"] = 10
    ledger_path.write_text(json.dumps(records))

    for i in range(4):
        _clean_session(projects_root, "-Users-test-demo", f"after-clean-{i}", mtime=applied_at + timedelta(hours=i + 1))
    for i in range(2):
        _cwd_confusion_session(projects_root, "-Users-test-demo", f"after-cwd-{i}", mtime=applied_at + timedelta(hours=10 + i))

    summary = pv.rescan_all(cfg, None, projects_root=projects_root)
    assert summary == {"checked": 1, "updated": 1}

    updated = pa.get_applied(cfg, fix_id)
    v = updated["verify"]
    assert v["post_sessions_since_apply"] == 6
    assert v["recurrence_since_apply"] == 2
    # baseline_rate = 10/10 = 1.0; post_rate = 2/6 ~= 0.333; ratio ~= 0.333 <= 0.7 -> improved.
    assert v["verdict"] == pv.VERDICT_IMPROVED
    assert v["realized_tokens_saved"] > 0
    assert v["last_checked_at"] is not None


def test_rescan_all_skips_reverted_fixes(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# root\n", encoding="utf-8")
    cluster = {
        "signature": "cwd_confusion", "family_key": "cwd_confusion",
        "title": "cwd confusion", "proposed_fix": "watch your cwd",
        "rung": 1, "sessions": 5, "occurrences": 10,
        "repos": ["demo"], "examples": [],
    }
    result = pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="user-global", go=True)
    fix_id = result["record"]["id"]
    pa.revert_applied_fix(cfg, fix_id)

    summary = pv.rescan_all(cfg, None, projects_root=tmp_path / "projects")
    assert summary == {"checked": 0, "updated": 0}


def test_rescan_all_reports_enforcement_disabled_for_unwired_hook(cfg, tmp_path):
    hook_target = tmp_path / ".claude" / "hooks" / "cwd.py"
    cluster = {
        "signature": "cwd_confusion", "family_key": "cwd_confusion",
        "title": "cwd confusion", "proposed_fix": "hook it",
        "rung": 3, "sessions": 5, "occurrences": 10,
        "repos": ["demo"], "examples": [],
    }
    result = pa.apply_relearn_fix(cfg, cluster, target_path=str(hook_target), scope="user-global", go=True)
    fix_id = result["record"]["id"]
    assert result["record"]["enforcement"]["enabled"] is False   # disabled by default

    pv.rescan_all(cfg, None, projects_root=tmp_path / "projects")
    updated = pa.get_applied(cfg, fix_id)
    assert updated["verify"]["verdict"] == pv.VERDICT_ENFORCEMENT_DISABLED
