"""Unit tests for the self-improve loop's Apply stage (core.optimize.relearn_apply).

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
from tokenjam.core.optimize import relearn_apply as pa

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
    result = pa.apply_relearn_fix(cfg, _cluster(), target_path=str(target), scope="project", go=False)
    assert result["dry_run"] is True
    assert target.read_text() == "# Repo\n"
    assert not pa.list_applied(cfg)


def test_apply_note_go_writes_marked_section(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    result = pa.apply_relearn_fix(cfg, _cluster(), target_path=str(target), scope="project", go=True)
    assert result["dry_run"] is False
    rec = result["record"]
    assert rec["kind"] == "note" and rec["state"] == "applied"
    written = target.read_text()
    assert pa.NOTE_SECTION_HEADER in written
    assert "<!-- tokenjam:relearn:cwd_confusion -->" in written
    assert "# Repo\n" in written        # original content preserved


def test_command_not_found_apply_writes_real_note_not_stub(cfg, tmp_path):
    # Downgraded to rung 1 (SPEC honesty fix): there is no safe automatic
    # config/env writer, so Apply must write a real CLAUDE.md note with
    # genuinely useful guidance instead of the inert stub hook rung 5 used to
    # produce via _render_stub_hook.
    cluster = _cluster(
        signature="command_not_found", family_key="command_not_found",
        title="command not found (bashisms under zsh, bare interpreter)",
        rung=1,
        proposed_fix=(
            "CLAUDE.md/skill note: this shell doesn't have that binary/builtin on "
            "PATH. Common causes here: using bare `python` instead of `python3`, "
            "or a bash-only builtin (`mapfile`, `shopt`, `[[ ... ]]` extensions) "
            "that doesn't exist under this shell (e.g. zsh, sh) or POSIX mode. "
            "Prefer the portable/explicit form."
        ),
    )
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    result = pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    assert result["record"]["kind"] == "note"
    written = target.read_text()
    assert "python3" in written
    assert "<!-- tokenjam:relearn:command_not_found -->" in written
    # Must never route through the enforcement/stub-hook path.
    assert "no automatic matcher" not in written
    assert "never blocks" not in written
    assert not (tmp_path / ".claude").exists()


def test_apply_note_creates_missing_file(cfg, tmp_path):
    target = tmp_path / "sub" / "CLAUDE.md"
    result = pa.apply_relearn_fix(cfg, _cluster(), target_path=str(target), scope="project", go=True)
    assert target.is_file()
    rec = result["record"]
    reverted = pa.revert_applied_fix(cfg, rec["id"])
    assert reverted["state"] == "reverted"
    assert not target.exists()          # a freshly-created file is deleted on revert


def test_reapply_same_signature_is_idempotent_not_duplicated(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    pa.apply_relearn_fix(cfg, _cluster(), target_path=str(target), scope="project", go=True)
    pa.apply_relearn_fix(cfg, _cluster(proposed_fix="updated fix text"),
                          target_path=str(target), scope="project", go=True)
    written = target.read_text()
    assert written.count("<!-- tokenjam:relearn:cwd_confusion -->") == 1
    assert "updated fix text" in written


# --- rung 2: skill ------------------------------------------------------------

def test_apply_skill_writes_frontmatter(cfg, tmp_path):
    cluster = _cluster(signature="deferred_tool_cold", family_key="deferred_tool_cold",
                        title="deferred tool cold", rung=2)
    target = tmp_path / ".claude" / "skills" / "deferred-tool-cold" / "SKILL.md"
    result = pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    assert result["record"]["kind"] == "skill"
    content = target.read_text()
    assert content.startswith("---\nname: deferred-tool-cold")
    assert "tokenjam:relearn:deferred_tool_cold" in content


def test_apply_skill_refuses_to_clobber_foreign_file(cfg, tmp_path):
    cluster = _cluster(signature="x", rung=2)
    target = tmp_path / ".claude" / "skills" / "x" / "SKILL.md"
    target.parent.mkdir(parents=True)
    target.write_text("hand-authored skill, not ours\n", encoding="utf-8")
    with pytest.raises(pa.RelearnApplyRefused, match="wasn't written by TokenJam"):
        pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    assert target.read_text() == "hand-authored skill, not ours\n"   # untouched


# --- rungs 3-5: enforcement (hook), disabled by default -----------------------

def test_apply_hook_is_written_disabled(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain",
                        title="blocked sleep-chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    result = pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
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
    pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    proc = subprocess.run([sys.executable, str(target)], input="not { valid json",
                           text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0   # fail-open: never blocks on a bug/bad input


def test_hook_blocks_real_sleep_chain(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "sleep 5 && ls"}})
    proc = subprocess.run([sys.executable, str(target)], input=payload,
                           text=True, capture_output=True, timeout=10)
    assert proc.returncode == 2
    assert "sleep-chain" in proc.stderr


def test_hook_allows_unrelated_command(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    proc = subprocess.run([sys.executable, str(target)], input=payload,
                           text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0


def test_stub_matcher_for_unknown_family_never_blocks(cfg, tmp_path):
    cluster = _cluster(signature="mystery", family_key="mystery_family",
                        title="some novel relearn", rung=3)
    target = tmp_path / ".claude" / "hooks" / "mystery.py"
    pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "sleep 5 && ls"}})
    proc = subprocess.run([sys.executable, str(target)], input=payload,
                           text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0   # no matcher wired for an unknown family — never blocks


# --- rung 3: PostToolUseFailure REACTIVE hooks (cwd_confusion, stale_read_race,
# edit_string_not_found) — never block (PostToolUseFailure fires only after a
# tool already failed), only inject `additionalContext`. Every test below
# checks BOTH that the real failure signature fires AND that a battery of
# normal/valid inputs does NOT fire — the false-positive battery is the part
# that actually matters (SPEC §10: a false positive on real usage is worse
# than a no-op). ------------------------------------------------------------

def _apply_reactive_hook(cfg, tmp_path, family_key: str, title: str):
    cluster = _cluster(signature=family_key, family_key=family_key, title=title, rung=3)
    target = tmp_path / ".claude" / "hooks" / f"{family_key}.py"
    pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    return target


def _run_hook(target, payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(target)], input=json.dumps(payload),
        text=True, capture_output=True, timeout=10,
    )


def test_reactive_hooks_are_written_disabled_with_posttoolusefailure_wiring(cfg, tmp_path):
    for family_key in ("cwd_confusion", "stale_read_race", "edit_string_not_found"):
        target = _apply_reactive_hook(cfg, tmp_path, family_key, family_key)
        assert target.is_file()
        patch_path = target.parent / f"{pa.slugify(family_key)}.settings-patch.json"
        patch = json.loads(patch_path.read_text())
        assert list(patch["hooks"].keys()) == ["PostToolUseFailure"]
        assert patch["hooks"]["PostToolUseFailure"][0]["hooks"][0]["command"] == str(target)
        assert not (tmp_path / ".claude" / "settings.json").exists()


def test_cwd_confusion_hook_fires_on_real_bash_failure(cfg, tmp_path):
    target = _apply_reactive_hook(cfg, tmp_path, "cwd_confusion", "cwd / relative-path confusion")
    proc = _run_hook(target, {
        "tool_name": "Bash", "cwd": str(tmp_path),
        "error": "(eval):cd:1: no such file or directory: orchestrator",
    })
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "PostToolUseFailure"
    assert str(tmp_path) in ctx        # actual cwd injected
    assert "cwd_confusion" in ctx


def test_cwd_confusion_hook_fires_on_real_read_failure(cfg, tmp_path):
    target = _apply_reactive_hook(cfg, tmp_path, "cwd_confusion", "cwd / relative-path confusion")
    proc = _run_hook(target, {
        "tool_name": "Read", "cwd": str(tmp_path),
        "error": "File does not exist. Note: your current working directory is /some/other/dir",
    })
    assert proc.returncode == 0
    assert proc.stdout.strip()   # fired


def test_cwd_confusion_hook_false_positive_battery(cfg, tmp_path):
    """None of these NORMAL/VALID PostToolUseFailure payloads should fire."""
    target = _apply_reactive_hook(cfg, tmp_path, "cwd_confusion", "cwd / relative-path confusion")
    battery = [
        # Real failure but on a tool this hook doesn't cover (e.g. Grep
        # matching the literal phrase in file content it searched).
        {"tool_name": "Grep", "cwd": str(tmp_path), "error": "no such file or directory"},
        # Bash failure for an unrelated reason.
        {"tool_name": "Bash", "cwd": str(tmp_path), "error": "npm ERR! code ENOENT"},
        {"tool_name": "Bash", "cwd": str(tmp_path), "error": "permission denied"},
        {"tool_name": "Bash", "cwd": str(tmp_path), "error": "command not found: foo"},
        # Read failure for an unrelated reason.
        {"tool_name": "Read", "cwd": str(tmp_path), "error": "offset must be a scalar, not an array"},
        # No error text at all (shouldn't happen for PostToolUseFailure, but
        # must never crash / never fire).
        {"tool_name": "Bash", "cwd": str(tmp_path), "error": ""},
        {"tool_name": "Bash", "cwd": str(tmp_path)},
    ]
    for payload in battery:
        proc = _run_hook(target, payload)
        assert proc.returncode == 0
        assert proc.stdout == "", f"unexpected fire on {payload!r}: {proc.stdout!r}"


def test_stale_read_race_hook_fires_on_real_failure(cfg, tmp_path):
    target = _apply_reactive_hook(cfg, tmp_path, "stale_read_race", "file modified since read")
    proc = _run_hook(target, {
        "tool_name": "Edit", "cwd": str(tmp_path),
        "error": "File has been modified since it was last read.",
    })
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert "re-Read" in out["hookSpecificOutput"]["additionalContext"] or \
           "Read it again" in out["hookSpecificOutput"]["additionalContext"]


def test_stale_read_race_hook_false_positive_battery(cfg, tmp_path):
    target = _apply_reactive_hook(cfg, tmp_path, "stale_read_race", "file modified since read")
    battery = [
        {"tool_name": "Bash", "cwd": str(tmp_path), "error": "modified since read"},  # wrong tool
        {"tool_name": "Edit", "cwd": str(tmp_path), "error": "string to replace not found"},
        {"tool_name": "Write", "cwd": str(tmp_path), "error": "permission denied"},
        {"tool_name": "MultiEdit", "cwd": str(tmp_path), "error": ""},
        {"tool_name": "Edit", "cwd": str(tmp_path)},
    ]
    for payload in battery:
        proc = _run_hook(target, payload)
        assert proc.returncode == 0
        assert proc.stdout == "", f"unexpected fire on {payload!r}: {proc.stdout!r}"


def test_edit_string_not_found_hook_fires_on_real_failure(cfg, tmp_path):
    target = _apply_reactive_hook(cfg, tmp_path, "edit_string_not_found", "Edit string-not-found")
    proc = _run_hook(target, {
        "tool_name": "Edit", "cwd": str(tmp_path),
        "error": "String to replace not found in file.",
    })
    assert proc.returncode == 0
    assert proc.stdout.strip()


def test_edit_string_not_found_hook_false_positive_battery(cfg, tmp_path):
    target = _apply_reactive_hook(cfg, tmp_path, "edit_string_not_found", "Edit string-not-found")
    battery = [
        {"tool_name": "Write", "cwd": str(tmp_path), "error": "string to replace not found"},  # wrong tool
        {"tool_name": "Edit", "cwd": str(tmp_path), "error": "modified since it was last read"},
        {"tool_name": "MultiEdit", "cwd": str(tmp_path), "error": "permission denied"},
        {"tool_name": "Edit", "cwd": str(tmp_path), "error": ""},
        {"tool_name": "Edit", "cwd": str(tmp_path)},
    ]
    for payload in battery:
        proc = _run_hook(target, payload)
        assert proc.returncode == 0
        assert proc.stdout == "", f"unexpected fire on {payload!r}: {proc.stdout!r}"


@pytest.mark.parametrize("family_key", ["cwd_confusion", "stale_read_race", "edit_string_not_found"])
def test_reactive_hook_fails_open_on_garbage_stdin(cfg, tmp_path, family_key):
    target = _apply_reactive_hook(cfg, tmp_path, family_key, family_key)
    proc = subprocess.run([sys.executable, str(target)], input="not { valid json",
                           text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0
    assert proc.stdout == ""


@pytest.mark.parametrize("family_key", ["cwd_confusion", "stale_read_race", "edit_string_not_found"])
def test_reactive_hook_fails_open_on_empty_stdin(cfg, tmp_path, family_key):
    target = _apply_reactive_hook(cfg, tmp_path, family_key, family_key)
    proc = subprocess.run([sys.executable, str(target)], input="",
                           text=True, capture_output=True, timeout=10)
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_cwd_confusion_hook_survives_unreadable_cwd(cfg, tmp_path):
    """A `cwd` that doesn't exist on disk must never crash the hook — it just
    skips the directory listing (fail-soft), still injecting the note."""
    target = _apply_reactive_hook(cfg, tmp_path, "cwd_confusion", "cwd / relative-path confusion")
    proc = _run_hook(target, {
        "tool_name": "Bash", "cwd": "/definitely/does/not/exist/anywhere",
        "error": "no such file or directory",
    })
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["hookSpecificOutput"]["additionalContext"]


# --- Defensive error-field extraction ----------------------------------------
# The PostToolUseFailure input schema does not pin the exact field carrying the
# error, so the generated hook reads it from whichever variant is present
# (`tool_error` -> `error` -> `tool_response.error` -> stringified
# `tool_response`). If the harness uses a different field than we assumed, the
# hook must STILL fire on a real failure rather than silently no-op.

_CWD_ERR = "(eval):cd:1: no such file or directory: orchestrator"


@pytest.mark.parametrize("payload_key_path,wrap", [
    ("tool_error", lambda err: {"tool_error": err}),
    ("error", lambda err: {"error": err}),
    ("tool_response.error", lambda err: {"tool_response": {"error": err}}),
    ("stringified tool_response", lambda err: {"tool_response": {"stderr": err}}),
])
def test_reactive_hook_fires_regardless_of_error_field_name(cfg, tmp_path, payload_key_path, wrap):
    target = _apply_reactive_hook(cfg, tmp_path, "cwd_confusion", "cwd / relative-path confusion")
    payload = {"tool_name": "Bash", "cwd": str(tmp_path), **wrap(_CWD_ERR)}
    proc = _run_hook(target, payload)
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["hookSpecificOutput"]["hookEventName"] == "PostToolUseFailure"
    assert "cwd_confusion" in out["hookSpecificOutput"]["additionalContext"], (
        f"hook failed to extract error from {payload_key_path!r}"
    )


def test_reactive_hook_no_fire_when_error_absent_across_all_fields(cfg, tmp_path):
    """No error under ANY field variant -> fail-open, inject nothing."""
    target = _apply_reactive_hook(cfg, tmp_path, "cwd_confusion", "cwd / relative-path confusion")
    battery = [
        {"tool_name": "Bash", "cwd": str(tmp_path)},                       # no error field at all
        {"tool_name": "Bash", "cwd": str(tmp_path), "tool_error": ""},     # empty tool_error
        {"tool_name": "Bash", "cwd": str(tmp_path), "error": "   "},       # whitespace-only error
        {"tool_name": "Bash", "cwd": str(tmp_path), "tool_response": {}},  # empty tool_response dict
        {"tool_name": "Bash", "cwd": str(tmp_path), "tool_response": None},
        # error present under a variant but UNRELATED to this family -> no fire
        {"tool_name": "Bash", "cwd": str(tmp_path), "tool_error": "permission denied"},
        {"tool_name": "Bash", "cwd": str(tmp_path), "tool_response": {"error": "npm ERR! ENOENT"}},
    ]
    for payload in battery:
        proc = _run_hook(target, payload)
        assert proc.returncode == 0
        assert proc.stdout == "", f"unexpected fire on {payload!r}: {proc.stdout!r}"


def test_reactive_hook_stale_read_fires_via_tool_response_error(cfg, tmp_path):
    """Cross-family sanity: the defensive extraction is family-agnostic — a
    stale-read hook also fires when the error arrives under tool_response.error."""
    target = _apply_reactive_hook(cfg, tmp_path, "stale_read_race", "file modified since read")
    proc = _run_hook(target, {
        "tool_name": "Edit", "cwd": str(tmp_path),
        "tool_response": {"error": "File has been modified since it was last read."},
    })
    assert proc.returncode == 0
    assert proc.stdout.strip()


def test_enable_enforcement_requires_confirm(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    result = pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
    fix_id = result["record"]["id"]
    with pytest.raises(pa.RelearnApplyRefused, match="explicit"):
        pa.enable_enforcement(cfg, fix_id, confirm=False)
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_enable_then_disable_enforcement(cfg, tmp_path):
    cluster = _cluster(signature="sleep_chain", family_key="sleep_chain", rung=3)
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    result = pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
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
    result = pa.apply_relearn_fix(cfg, cluster, target_path=str(target), scope="project", go=True)
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

    result = pa.apply_relearn_fix(cfg, _cluster(), target_path=str(target), scope="project", go=True)
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
    result = pa.apply_relearn_fix(cfg, _cluster(), target_path=str(target), scope="project", go=True)
    fix_id = result["record"]["id"]
    first = pa.revert_applied_fix(cfg, fix_id)
    second = pa.revert_applied_fix(cfg, fix_id)
    assert first["state"] == second["state"] == "reverted"


def test_revert_unknown_fix_id_refuses(cfg):
    with pytest.raises(pa.RelearnApplyRefused, match="no applied fix"):
        pa.revert_applied_fix(cfg, "does-not-exist")


def test_unknown_rung_refuses(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    with pytest.raises(pa.RelearnApplyRefused, match="unknown rung"):
        pa.apply_relearn_fix(cfg, _cluster(rung=99), target_path=str(target), scope="project", go=True)


def test_empty_target_path_refuses(cfg):
    with pytest.raises(pa.RelearnApplyRefused, match="no target path"):
        pa.apply_relearn_fix(cfg, _cluster(), target_path="", scope="project", go=True)


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

    with pytest.raises(pa.RelearnApplyRefused, match="active session"):
        pa.apply_relearn_fix(cfg, _cluster(), target_path=str(target), scope="project",
                              go=True, conn=conn, force=False)
    assert not pa.list_applied(cfg)   # refused before any write/record happened

    result = pa.apply_relearn_fix(cfg, _cluster(), target_path=str(target), scope="project",
                                   go=True, conn=conn, force=True)
    assert result["dry_run"] is False   # force bypasses the warning


def test_active_session_warning_none_when_conn_missing(tmp_path):
    assert pa.active_session_warning(None, str(tmp_path / "CLAUDE.md")) is None


def test_active_session_warning_never_raises_on_bad_conn(tmp_path):
    class _BrokenConn:
        def execute(self, sql):
            raise RuntimeError("db is on fire")

    assert pa.active_session_warning(_BrokenConn(), str(tmp_path / "CLAUDE.md")) is None


# --- must-fix #2: rung-1 note target allowlist ---------------------------------

def test_note_apply_refuses_non_markdown_target(cfg, tmp_path):
    """rung=1 (note) at an arbitrary non-.md target (e.g. a .py file with no
    prior tokenjam marker) must be refused outright — otherwise a client-
    supplied target_path could corrupt any file on disk."""
    target = tmp_path / "some_script.py"
    target.write_text("print('do not touch me')\n", encoding="utf-8")
    with pytest.raises(pa.RelearnApplyRefused, match="not an allowlisted note target"):
        pa.apply_relearn_fix(cfg, _cluster(rung=1), target_path=str(target), scope="project", go=True)
    assert target.read_text() == "print('do not touch me')\n"   # untouched


def test_note_apply_refuses_dotfile_target(cfg, tmp_path):
    target = tmp_path / ".zshrc"
    target.write_text("export PATH=/usr/bin\n", encoding="utf-8")
    with pytest.raises(pa.RelearnApplyRefused, match="not an allowlisted note target"):
        pa.apply_relearn_fix(cfg, _cluster(rung=1), target_path=str(target), scope="project", go=True)
    assert target.read_text() == "export PATH=/usr/bin\n"   # untouched


def test_note_apply_allows_md_target(cfg, tmp_path):
    """A *.md target (not just literally CLAUDE.md) is allowed."""
    target = tmp_path / "AGENTS.md"
    target.write_text("# Agents\n", encoding="utf-8")
    result = pa.apply_relearn_fix(cfg, _cluster(rung=1), target_path=str(target), scope="project", go=True)
    assert result["dry_run"] is False
    assert "<!-- tokenjam:relearn:cwd_confusion -->" in target.read_text()


def test_note_apply_allows_re_apply_to_already_marked_non_md_file(cfg, tmp_path):
    """A non-.md file that ALREADY carries a tokenjam:relearn marker (e.g. a
    prior legitimate apply) may still be re-applied — the allowlist doesn't
    lock out idempotent re-apply, only a first write to a foreign extension."""
    target = tmp_path / "notes.txt"
    target.write_text(
        "<!-- tokenjam:relearn:cwd_confusion -->\nold\n<!-- /tokenjam:relearn:cwd_confusion -->\n",
        encoding="utf-8",
    )
    result = pa.apply_relearn_fix(cfg, _cluster(rung=1), target_path=str(target), scope="project", go=True)
    assert result["dry_run"] is False


# --- must-fix #3: symlink guard --------------------------------------------

def test_apply_refuses_symlinked_target(cfg, tmp_path):
    real = tmp_path / "real_claude.md"
    real.write_text("# elsewhere\n", encoding="utf-8")
    link = tmp_path / "CLAUDE.md"
    link.symlink_to(real)
    with pytest.raises(pa.RelearnApplyRefused, match="symlink"):
        pa.apply_relearn_fix(cfg, _cluster(rung=1), target_path=str(link), scope="project", go=True)
    assert real.read_text() == "# elsewhere\n"   # the real file was never touched through the link


def test_apply_refuses_dangling_symlink_target(cfg, tmp_path):
    """A BROKEN symlink (nothing at the other end) must also be refused, not
    silently treated as 'file doesn't exist yet, plain-create it' — that
    plain-create branch is exactly what would write through the link."""
    link = tmp_path / "CLAUDE.md"
    link.symlink_to(tmp_path / "does-not-exist.md")
    with pytest.raises(pa.RelearnApplyRefused, match="symlink"):
        pa.apply_relearn_fix(cfg, _cluster(rung=1), target_path=str(link), scope="project", go=True)
    assert not (tmp_path / "does-not-exist.md").exists()


def test_revert_refuses_when_target_became_a_symlink(cfg, tmp_path):
    """Apply normally, then swap the target for a symlink before reverting —
    revert must refuse rather than restore/delete through the new link."""
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    result = pa.apply_relearn_fix(cfg, _cluster(rung=1), target_path=str(target), scope="project", go=True)
    fix_id = result["record"]["id"]

    elsewhere = tmp_path / "elsewhere.md"
    elsewhere.write_text("do not touch\n", encoding="utf-8")
    target.unlink()
    target.symlink_to(elsewhere)

    with pytest.raises(pa.RelearnApplyRefused, match="symlink"):
        pa.revert_applied_fix(cfg, fix_id)
    assert elsewhere.read_text() == "do not touch\n"   # untouched


# --- must-fix #4: :memory:/"" storage never falls back to the real ~/.tj ------

def test_memory_storage_path_never_resolves_to_real_home(monkeypatch, tmp_path):
    """A fake, obviously-not-real HOME lets this test prove the resolved base
    dir is NOT under it -- i.e. relearn_apply_root/applied_fixes_path for a
    ':memory:'-configured TjConfig must land in a TEMP dir, never ~/.tj."""
    fake_home = tmp_path / "definitely-not-the-real-home"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))

    cfg = TjConfig(version="1", storage=StorageConfig(path=":memory:"))
    root = pa.relearn_apply_root(cfg)
    ledger_path = pa.applied_fixes_path(cfg)

    assert not str(root).startswith(str(fake_home))
    assert not str(ledger_path).startswith(str(fake_home))
    assert (fake_home / ".tj").exists() is False   # never created either


def test_memory_storage_path_stable_across_calls_same_config(tmp_path):
    """The SAME config object must resolve to the SAME temp root every call —
    apply-time and revert-time paths must agree within one process/config."""
    cfg = TjConfig(version="1", storage=StorageConfig(path=":memory:"))
    first = pa.relearn_apply_root(cfg)
    second = pa.relearn_apply_root(cfg)
    assert first == second


def test_empty_storage_path_also_never_resolves_to_real_home(monkeypatch, tmp_path):
    fake_home = tmp_path / "definitely-not-the-real-home-2"
    fake_home.mkdir()
    monkeypatch.setattr(pa.Path, "home", classmethod(lambda cls: fake_home))

    cfg = TjConfig(version="1", storage=StorageConfig(path=""))
    root = pa.relearn_apply_root(cfg)
    assert not str(root).startswith(str(fake_home))


def test_memory_storage_apply_and_revert_round_trip_via_temp_root(tmp_path):
    """End-to-end: with ':memory:' storage, apply then revert must both
    resolve against the SAME (temp, non-home) ledger/backup root."""
    cfg = TjConfig(version="1", storage=StorageConfig(path=":memory:"))
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    result = pa.apply_relearn_fix(cfg, _cluster(rung=1), target_path=str(target), scope="project", go=True)
    fix_id = result["record"]["id"]
    assert pa.get_applied(cfg, fix_id) is not None
    reverted = pa.revert_applied_fix(cfg, fix_id)
    assert reverted["state"] == "reverted"
    assert target.read_text() == "# Repo\n"
