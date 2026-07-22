"""The two model-routing write paths (core.optimize.model_apply).

Real disk, real ``git init`` repos, real apply/revert round-trips: the same
standard as ``test_relearn_apply.py``, because both kinds are actual writes into
a user's files. Every target lives under ``tmp_path``.
"""
from __future__ import annotations

import subprocess

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import model_apply as ma
from tokenjam.core.optimize import relearn_apply as pa

AGENT_FILE = """---
name: explore
description: Read-only search agent.
model: claude-opus-4-8
---

Body text stays untouched.
"""

AGENT_FILE_NO_MODEL = """---
name: explore
description: Read-only search agent.
---

Body.
"""


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _git_repo(tmp_path, name="repo"):
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def _commit_all(repo, message="add"):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)


def _agent_cluster(**overrides) -> dict:
    base = {
        "signature": "cost:subagent:explore",
        "title": "Over-powered subagent explore",
        "rung": 1,
        "apply_kind": ma.APPLY_KIND_AGENT_MODEL,
        "agent_name": "explore",
        "current_model": "claude-opus-4-8",
        "proposed_model": "claude-haiku-4-5",
    }
    base.update(overrides)
    return base


def _swap_cluster(source_path: str, **overrides) -> dict:
    base = {
        "signature": "cost:downsize:svc-a",
        "title": "Model over-sizing in svc-a",
        "rung": 1,
        "apply_kind": ma.APPLY_KIND_MODEL_SWAP,
        "source_path": source_path,
        "current_model": "claude-opus-4-8",
        "proposed_model": "claude-haiku-4-5",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# B2: the agent file's `model:` frontmatter key
# --------------------------------------------------------------------------- #

def test_default_agent_file_path_project_and_user_global(tmp_path, monkeypatch):
    monkeypatch.setattr(ma.Path, "home", classmethod(lambda cls: tmp_path))
    project = ma.default_agent_file_path("project", str(tmp_path / "repo"), "explore")
    user = ma.default_agent_file_path("user-global", "", "explore")
    assert project == str(tmp_path / "repo" / ".claude" / "agents" / "explore.md")
    assert user == str(tmp_path / ".claude" / "agents" / "explore.md")
    # Project scope with no repo to anchor on must ask, never guess.
    assert ma.default_agent_file_path("project", "", "explore") == ""


def test_render_agent_model_replaces_only_the_model_key():
    # Arrange / Act
    content, reason = ma.render_agent_model(AGENT_FILE, "claude-haiku-4-5")
    # Assert
    assert reason == ""
    assert "model: claude-haiku-4-5" in content
    assert "claude-opus-4-8" not in content
    assert "name: explore" in content
    assert content.endswith("Body text stays untouched.\n")


def test_render_agent_model_inserts_key_when_absent():
    content, reason = ma.render_agent_model(AGENT_FILE_NO_MODEL, "claude-haiku-4-5")
    assert reason == ""
    head, _, body = content.partition("\n---\n")
    assert "model: claude-haiku-4-5" in head
    assert body.strip() == "Body."


def test_render_agent_model_refuses_without_frontmatter():
    content, reason = ma.render_agent_model("# Just prose\n", "claude-haiku-4-5")
    assert content is None
    assert "frontmatter" in reason


def test_render_agent_model_refuses_a_no_op():
    content, reason = ma.render_agent_model(AGENT_FILE, "claude-opus-4-8")
    assert content is None
    assert "already runs on" in reason


def test_agent_model_apply_revert_round_trip(cfg, tmp_path):
    # Arrange: a real repo with a committed agent definition.
    repo = _git_repo(tmp_path)
    target = repo / ".claude" / "agents" / "explore.md"
    target.parent.mkdir(parents=True)
    target.write_text(AGENT_FILE, encoding="utf-8")
    _commit_all(repo, "add agent")

    # Act: apply for real.
    result = pa.apply_relearn_fix(
        cfg, _agent_cluster(), target_path=str(target), scope="project", go=True,
    )

    # Assert: the key moved, the file is committed, the ledger names the kind.
    assert result["dry_run"] is False
    record = result["record"]
    assert record["kind"] == ma.APPLY_KIND_AGENT_MODEL
    assert "model: claude-haiku-4-5" in target.read_text()
    assert record["git_commit"]
    log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"], cwd=repo, capture_output=True, text=True,
    )
    assert ma.APPLY_KIND_AGENT_MODEL in log.stdout

    # Act: one-call revert.
    reverted = pa.revert_applied_fix(cfg, record["id"])

    # Assert: byte-for-byte back, and the revert is committed too.
    assert reverted["state"] == "reverted"
    assert target.read_text() == AGENT_FILE
    assert reverted["revert_commit"]


def test_agent_model_dry_run_writes_nothing(cfg, tmp_path):
    target = tmp_path / ".claude" / "agents" / "explore.md"
    target.parent.mkdir(parents=True)
    target.write_text(AGENT_FILE, encoding="utf-8")
    result = pa.apply_relearn_fix(
        cfg, _agent_cluster(), target_path=str(target), scope="project", go=False,
    )
    assert result["dry_run"] is True
    assert "claude-haiku-4-5" in result["diff"]
    assert target.read_text() == AGENT_FILE
    assert not pa.list_applied(cfg)


def test_agent_model_refuses_a_non_agent_target(cfg, tmp_path):
    target = tmp_path / "notes.md"
    target.write_text(AGENT_FILE, encoding="utf-8")
    with pytest.raises(pa.RelearnApplyRefused, match="agent definition file"):
        pa.apply_relearn_fix(
            cfg, _agent_cluster(), target_path=str(target), scope="project", go=True,
        )


def test_agent_model_refuses_when_the_file_does_not_exist(cfg, tmp_path):
    # The inline Task-tool case: no definition file, so the caller falls back to
    # the guidance block rather than creating an agent nobody defined.
    target = tmp_path / ".claude" / "agents" / "explore.md"
    with pytest.raises(pa.RelearnApplyRefused, match="no agent file"):
        pa.apply_relearn_fix(
            cfg, _agent_cluster(), target_path=str(target), scope="project", go=True,
        )


# --------------------------------------------------------------------------- #
# B1b: the gated model-id string swap, and its four refusals
# --------------------------------------------------------------------------- #

def test_model_swap_apply_revert_round_trip(cfg, tmp_path):
    # Arrange: exactly one file carries the model id, and it is committed.
    repo = _git_repo(tmp_path)
    source = repo / "agent.py"
    original = 'MODEL = "claude-opus-4-8"\nclient.run(model=MODEL)\n'
    source.write_text(original, encoding="utf-8")
    _commit_all(repo, "add agent code")

    check = ma.model_swap_precheck(str(repo), "claude-opus-4-8")
    assert check["ok"] is True
    assert check["target_path"] == str(source)

    # Act
    result = pa.apply_relearn_fix(
        cfg, _swap_cluster(str(repo)), target_path=check["target_path"],
        scope="project", go=True,
    )

    # Assert
    record = result["record"]
    assert record["kind"] == ma.APPLY_KIND_MODEL_SWAP
    assert source.read_text() == 'MODEL = "claude-haiku-4-5"\nclient.run(model=MODEL)\n'
    assert record["git_commit"]

    reverted = pa.revert_applied_fix(cfg, record["id"])
    assert reverted["state"] == "reverted"
    assert source.read_text() == original


def test_model_swap_refuses_unregistered_path():
    check = ma.model_swap_precheck("", "claude-opus-4-8")
    assert check["ok"] is False
    assert "no local source path is registered" in check["reason"]


def test_model_swap_refuses_zero_matches(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "agent.py").write_text("MODEL = os.environ['MODEL']\n", encoding="utf-8")
    _commit_all(repo)
    check = ma.model_swap_precheck(str(repo), "claude-opus-4-8")
    assert check["ok"] is False
    assert "does not appear in any source file" in check["reason"]


def test_model_swap_refuses_multiple_matches(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "agent.py").write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    (repo / "worker.py").write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    _commit_all(repo)
    check = ma.model_swap_precheck(str(repo), "claude-opus-4-8")
    assert check["ok"] is False
    assert "appears in 2 files" in check["reason"]


def test_model_swap_refuses_dirty_working_tree(tmp_path):
    repo = _git_repo(tmp_path)
    source = repo / "agent.py"
    source.write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    _commit_all(repo)
    source.write_text('M = "claude-opus-4-8"  # local edit\n', encoding="utf-8")
    check = ma.model_swap_precheck(str(repo), "claude-opus-4-8")
    assert check["ok"] is False
    assert "uncommitted changes" in check["reason"]


def test_model_swap_refuses_a_non_git_path(tmp_path):
    plain = tmp_path / "plain"
    (plain / "src").mkdir(parents=True)
    (plain / "src" / "agent.py").write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    check = ma.model_swap_precheck(str(plain), "claude-opus-4-8")
    assert check["ok"] is False
    assert "not a git repository" in check["reason"]


def test_model_swap_ignores_vendored_and_unlisted_files(tmp_path):
    # A second copy of the id inside node_modules or a .md file must not turn a
    # clean single match into a refusal.
    repo = _git_repo(tmp_path)
    (repo / "agent.py").write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "dep.js").write_text('"claude-opus-4-8"\n', encoding="utf-8")
    (repo / "README2.md").write_text("we use claude-opus-4-8\n", encoding="utf-8")
    _commit_all(repo)
    check = ma.model_swap_precheck(str(repo), "claude-opus-4-8")
    assert check["ok"] is True
    assert check["target_path"] == str(repo / "agent.py")


def test_env_files_are_swappable_despite_having_no_suffix(tmp_path):
    assert ma.is_swappable_file(tmp_path / ".env") is True
    assert ma.is_swappable_file(tmp_path / "agent.py") is True
    assert ma.is_swappable_file(tmp_path / "README.md") is False


def test_model_swap_revalidates_at_write_time(cfg, tmp_path):
    # The card was built when the swap was clean; by apply time a second file
    # carries the id. The write must refuse rather than edit a stale target.
    repo = _git_repo(tmp_path)
    source = repo / "agent.py"
    source.write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    _commit_all(repo)
    check = ma.model_swap_precheck(str(repo), "claude-opus-4-8")
    assert check["ok"] is True

    (repo / "worker.py").write_text('M = "claude-opus-4-8"\n', encoding="utf-8")
    _commit_all(repo, "second use")

    with pytest.raises(pa.RelearnApplyRefused, match="appears in 2 files"):
        pa.apply_relearn_fix(
            cfg, _swap_cluster(str(repo)), target_path=str(source),
            scope="project", go=True,
        )
    assert source.read_text() == 'M = "claude-opus-4-8"\n'


def test_unknown_apply_kind_is_refused(cfg, tmp_path):
    target = tmp_path / "agent.py"
    target.write_text("x\n", encoding="utf-8")
    with pytest.raises(pa.RelearnApplyRefused, match="unknown apply kind"):
        pa.apply_relearn_fix(
            cfg, _swap_cluster(str(tmp_path), apply_kind="teleport"),
            target_path=str(target), scope="project", go=True,
        )
