"""Unit tests for the self-improve loop's Apply stage (core.optimize.pothole_apply).

Everything is routed at a tmp storage dir (``cfg.storage.path``) AND every
write TARGET is under ``tmp_path`` too — nothing here ever touches a real
``~/.tj`` or ``~/.claude``. Mirrors ``test_summarize_apply.py``'s fixture style.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import pothole_apply as pa

# --- Fixtures ----------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _cluster(**overrides) -> dict:
    base = {
        "signature": "cwd_confusion",
        "family_key": "cwd_confusion",
        "title": "cwd / relative-path confusion",
        "proposed_fix": "Verify an absolute cwd before a relative Read.",
        "rung": 1,
        "sessions": 5,
        "occurrences": 9,
        "repos": ["demo"],
        "examples": [{"session_id": "abc123def456", "repo": "demo", "snippet": "no such file"}],
    }
    base.update(overrides)
    return base


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


# --- slugify / default_target_path -------------------------------------------

def test_slugify_never_empty():
    assert pa.slugify("") == "fix"
    assert pa.slugify("cwd / relative-path confusion!!") == "cwd-relative-path-confusion"


def test_default_target_path_user_global(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: tmp_path))
    note = pa.default_target_path(1, "user-global", "", "slug")
    skill = pa.default_target_path(2, "user-global", "", "slug")
    hook = pa.default_target_path(3, "user-global", "", "slug")
    assert note == str(tmp_path / ".claude" / "CLAUDE.md")
    assert skill == str(tmp_path / ".claude" / "skills" / "slug" / "SKILL.md")
    assert hook == str(tmp_path / ".claude" / "hooks" / "slug.py")


def test_default_target_path_project_needs_cwd():
    assert pa.default_target_path(1, "project", "", "slug") == ""


def test_default_target_path_project_finds_nearest_claude_md(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("# root\n", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    assert pa.default_target_path(1, "project", str(sub), "slug") == str(tmp_path / "CLAUDE.md")


# --- rung 1: note ------------------------------------------------------------

def test_apply_note_dry_run_writes_nothing(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    result = pa.apply_pothole_fix(cfg, _cluster(), target_path=str(target), scope="project", go=False)
    assert result["dry_run"] is True
    assert target.read_text() == "# Repo\n"
    assert not pa.list_applied(cfg)


def test_apply_note_go_writes_marked_section(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    result = pa.apply_pothole_fix(cfg, _cluster(), target_path=str(target), scope="project", go=True)
    assert result["dry_run"] is False
    rec = result["record"]
    assert rec["kind"] == "note" and rec["state"] == "applied"
    written = target.read_text()
    assert pa.NOTE_SECTION_HEADER in written
    assert "<!-- tokenjam:pothole:cwd_confusion -->" in written
    assert "# Repo\n" in written        # original content preserved


def test_apply_note_creates_missing_file(cfg, tmp_path):
    target = tmp_path / "sub" / "CLAUDE.md"
    result = pa.apply_pothole_fix(cfg, _cluster(), target_path=str(target), scope="project", go=True)
    assert target.is_file()
    rec = result["record"]
    reverted = pa.revert_applied_fix(cfg, rec["id"])
    assert reverted["state"] == "reverted"
    assert not target.exists()          # a freshly-created file is deleted on revert


def test_reapply_same_signature_is_idempotent_not_duplicated(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    pa.apply_pothole_fix(cfg, _cluster(), target_path=str(target), scope="project", go=True)
    pa.apply_pothole_fix(cfg, _cluster(proposed_fix="updated fix text"),
                          target_path=str(target), scope="project", go=True)
    written = target.read_text()
    assert written.count("<!-- tokenjam:pothole:cwd_confusion -->") == 1
    assert "updated fix text" in written


# --- rung 2: skill ------------------------------------------------------------

def test_apply_skill_writes_frontmatter(cfg, tmp_path):
    cluster = _cluster(signature="deferred_tool_cold", family_key="deferred_tool_cold",
                        title="deferred tool cold", rung=2)
    target = tmp_path / ".claude" / "skills" / "deferred-tool-cold" / "SKILL.md"
    result = pa.apply_pothole_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    assert result["record"]["kind"] == "skill"
    content = target.read_text()
    assert content.startswith("---\nname: deferred-tool-cold")
    assert "tokenjam:pothole:deferred_tool_cold" in content


def test_apply_skill_refuses_to_clobber_foreign_file(cfg, tmp_path):
    cluster = _cluster(signature="x", rung=2)
    target = tmp_path / ".claude" / "skills" / "x" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("hand-authored skill, not ours\n", encoding="utf-8")
    with pytest.raises(pa.PotholeApplyRefused, match="wasn't written by TokenJam"):
        pa.apply_pothole_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    assert target.read_text() == "hand-authored skill, not ours\n"   # untouched


# --- rungs 3-5: enforcement (hook), disabled by default -----------------------

def test_apply_hook_is_written_disabled(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain",
                        title="blocked sleep-chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    result = pa.apply_pothole_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    rec = result["record"]
    assert rec["enforcement"]["enabled"] is False
    assert target.is_file()
    patch_path = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.settings-patch.json"
    assert patch_path.is_file()
    # settings.json itself was never touched by apply — only staged.
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_hook_fails_open_on_garbage_stdin(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    pa.apply_pothole_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    proc = subprocess.run([sys.executable, str(target)], input="not { valid json",
                           text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0   # fail-open: never blocks on a bug/bad input


def test_hook_blocks_real_sleep_chain(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    pa.apply_pothole_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "sleep 5 && ls"}})
    proc = subprocess.run([sys.executable, str(target)], input=payload,
                           text=True, capture_output=True, timeout=10)
    assert proc.returncode == 2
    assert "sleep-chain" in proc.stderr


def test_hook_allows_unrelated_command(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    pa.apply_pothole_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    proc = subprocess.run([sys.executable, str(target)], input=payload,
                           text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0


def test_stub_matcher_for_unknown_family_never_blocks(cfg, tmp_path):
    cluster = _cluster(signature="mystery", family_key="mystery_family",
                        title="some novel pothole", rung=3)
    target = tmp_path / ".claude" / "hooks" / "mystery.py"
    pa.apply_pothole_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "sleep 5 && ls"}})
    proc = subprocess.run([sys.executable, str(target)], input=payload,
                           text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0   # no matcher wired for an unknown family — never blocks


def test_enable_enforcement_requires_confirm(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    result = pa.apply_pothole_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    fix_id = result["record"]["id"]
    with pytest.raises(pa.PotholeApplyRefused, match="explicit"):
        pa.enable_enforcement(cfg, fix_id, confirm=False)
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_enable_then_disable_enforcement(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    result = pa.apply_pothole_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    fix_id = result["record"]["id"]

    enabled = pa.enable_enforcement(cfg, fix_id, confirm=True)
    assert enabled["enforcement"]["enabled"] is True
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == str(target)

    disabled = pa.disable_enforcement(cfg, fix_id)
    assert disabled["enforcement"]["enabled"] is False
    settings_after = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert settings_after["hooks"]["PreToolUse"] == []
    assert target.is_file()   # disable only unwires it, the hook file itself stays


def test_revert_disables_enforcement_and_deletes_hook(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    result = pa.apply_pothole_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    fix_id = result["record"]["id"]
    pa.enable_enforcement(cfg, fix_id, confirm=True)

    reverted = pa.revert_applied_fix(cfg, fix_id)
    assert reverted["state"] == "reverted"
    assert not target.exists()
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    assert settings["hooks"]["PreToolUse"] == []


# --- git-backed reversibility -------------------------------------------------

def test_apply_and_revert_commit_in_a_git_repo(tmp_path):
    repo = _git_repo(tmp_path)
    cfg = TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "state" / "t.duckdb")))
    target = repo / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add claude.md"], cwd=repo, check=True)

    result = pa.apply_pothole_fix(cfg, _cluster(), target_path=str(target), scope="project", go=True)
    rec = result["record"]
    assert rec["git_commit"] is not None
    assert rec["repo_root"] == str(repo)

    reverted = pa.revert_applied_fix(cfg, rec["id"])
    assert reverted["revert_commit"] is not None
    assert target.read_text() == "# Repo\n"

    log = subprocess.run(["git", "log", "--oneline"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout
    assert "tokenjam: apply" in log and "tokenjam: revert" in log


# --- revert idempotency / error paths ----------------------------------------

def test_revert_is_idempotent(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    result = pa.apply_pothole_fix(cfg, _cluster(), target_path=str(target), scope="project", go=True)
    fix_id = result["record"]["id"]
    first = pa.revert_applied_fix(cfg, fix_id)
    second = pa.revert_applied_fix(cfg, fix_id)
    assert first["state"] == second["state"] == "reverted"


def test_revert_unknown_fix_id_refuses(cfg):
    with pytest.raises(pa.PotholeApplyRefused, match="no applied fix"):
        pa.revert_applied_fix(cfg, "does-not-exist")


def test_unknown_rung_refuses(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    with pytest.raises(pa.PotholeApplyRefused, match="unknown rung"):
        pa.apply_pothole_fix(cfg, _cluster(rung=99), target_path=str(target), scope="project", go=True)


def test_empty_target_path_refuses(cfg):
    with pytest.raises(pa.PotholeApplyRefused, match="no target path"):
        pa.apply_pothole_fix(cfg, _cluster(), target_path="", scope="project", go=True)


# --- active-session guard ------------------------------------------------------

class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql):
        return self

    def fetchall(self):
        return self._rows


def test_active_session_warning_blocks_unless_forced(cfg, tmp_path):
    from datetime import timedelta

    from tokenjam.utils.time_parse import utcnow

    repo = _git_repo(tmp_path)
    target = repo / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add"], cwd=repo, check=True)

    now = utcnow()
    conn = _FakeConn([(f"claude-code-{repo.name}", "active", now - timedelta(minutes=1), None)])

    with pytest.raises(pa.PotholeApplyRefused, match="active session"):
        pa.apply_pothole_fix(cfg, _cluster(), target_path=str(target), scope="project",
                              go=True, conn=conn, force=False)
    assert not pa.list_applied(cfg)   # refused before any write/record happened

    result = pa.apply_pothole_fix(cfg, _cluster(), target_path=str(target), scope="project",
                                   go=True, conn=conn, force=True)
    assert result["dry_run"] is False   # force bypasses the warning


def test_active_session_warning_none_when_conn_missing(tmp_path):
    assert pa.active_session_warning(None, str(tmp_path / "CLAUDE.md")) is None


def test_active_session_warning_never_raises_on_bad_conn(tmp_path):
    class _BrokenConn:
        def execute(self, sql):
            raise RuntimeError("db is on fire")

    assert pa.active_session_warning(_BrokenConn(), str(tmp_path / "CLAUDE.md")) is None
