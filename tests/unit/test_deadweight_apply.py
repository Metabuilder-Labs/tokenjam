"""Unit tests for the ``mcp_remove`` apply kind (core.optimize.analyzers.
deadweight's write functions: ``render_mcp_remove``, ``mcp_remove_precheck``,
``build_mcp_remove_plan``), plus its wiring into ``relearn_apply``.

Mirrors ``test_model_apply.py``'s standard: real disk, real ``git init``
repos, real apply/revert round-trips, everything under ``tmp_path``. The
hazard this kind exists to avoid is a ``json.loads`` -> mutate ->
``json.dumps`` round trip reformatting the whole file, so several tests here
assert the diff touches ONLY the removed server's bytes.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import relearn_apply as pa
from tokenjam.core.optimize.analyzers import deadweight as dw

# --- Fixtures ----------------------------------------------------------------

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


def _mcp_cluster(source_path: str, server_name: str = "apollo", **overrides) -> dict:
    base = {
        "signature": f"cost:deadweight:{server_name}",
        "title": f"Unused MCP server: {server_name}",
        "rung": 0,
        "apply_kind": dw.APPLY_KIND_MCP_REMOVE,
        "agent_name": server_name,
        "source_path": source_path,
    }
    base.update(overrides)
    return base


# Two servers, pretty-printed the way Claude Code actually writes ``.mcp.json``
# — 2-space indent, trailing newline. Used across the multi-entry tests.
_TWO_SERVERS = """{
  "mcpServers": {
    "apollo": {
      "command": "npx",
      "args": ["-y", "@apollo/mcp"]
    },
    "exa": {
      "command": "npx",
      "args": ["-y", "@exa/mcp"]
    }
  }
}
"""

_THREE_SERVERS = """{
  "mcpServers": {
    "apollo": {
      "command": "npx",
      "args": ["-y", "@apollo/mcp"]
    },
    "exa": {
      "command": "npx",
      "args": ["-y", "@exa/mcp"]
    },
    "gmail": {
      "command": "npx",
      "args": ["-y", "@gmail/mcp"]
    }
  }
}
"""

_ONE_SERVER = """{
  "mcpServers": {
    "apollo": {
      "command": "npx",
      "args": ["-y", "@apollo/mcp"]
    }
  }
}
"""


# --------------------------------------------------------------------------- #
# render_mcp_remove: the targeted text splice, and its diff-preservation
# property (only the removed server's bytes change).
# --------------------------------------------------------------------------- #

def test_render_mcp_remove_middle_entry_touches_only_that_block():
    content, reason = dw.render_mcp_remove(_THREE_SERVERS, "exa")
    assert reason == ""
    assert '"exa"' not in content
    assert '"apollo"' in content
    assert '"gmail"' in content
    # Every other server's block survives byte-for-byte, in original order.
    assert content == (
        '{\n  "mcpServers": {\n    "apollo": {\n      "command": "npx",\n'
        '      "args": ["-y", "@apollo/mcp"]\n    },\n    "gmail": {\n'
        '      "command": "npx",\n      "args": ["-y", "@gmail/mcp"]\n'
        '    }\n  }\n}\n'
    )


def test_render_mcp_remove_result_is_valid_json():
    content, reason = dw.render_mcp_remove(_THREE_SERVERS, "exa")
    assert reason == ""
    doc = json.loads(content)
    assert set(doc["mcpServers"]) == {"apollo", "gmail"}


def test_render_mcp_remove_first_entry():
    content, reason = dw.render_mcp_remove(_TWO_SERVERS, "apollo")
    assert reason == ""
    doc = json.loads(content)
    assert set(doc["mcpServers"]) == {"exa"}
    # The connective tissue right after the opening brace is reused untouched.
    assert content == (
        '{\n  "mcpServers": {\n    "exa": {\n      "command": "npx",\n'
        '      "args": ["-y", "@exa/mcp"]\n    }\n  }\n}\n'
    )


def test_render_mcp_remove_last_entry_leaves_no_trailing_comma():
    content, reason = dw.render_mcp_remove(_TWO_SERVERS, "exa")
    assert reason == ""
    assert ",\n  }" not in content
    assert ",\n    }" not in content
    doc = json.loads(content)  # a trailing comma would fail to parse at all
    assert set(doc["mcpServers"]) == {"apollo"}
    assert content == (
        '{\n  "mcpServers": {\n    "apollo": {\n      "command": "npx",\n'
        '      "args": ["-y", "@apollo/mcp"]\n    }\n  }\n}\n'
    )


def test_render_mcp_remove_only_entry_leaves_a_valid_empty_object():
    content, reason = dw.render_mcp_remove(_ONE_SERVER, "apollo")
    assert reason == ""
    doc = json.loads(content)
    assert doc["mcpServers"] == {}
    assert content == '{\n  "mcpServers": {\n  }\n}\n'


def test_render_mcp_remove_diff_is_scoped_to_the_removed_block():
    import difflib

    before = _THREE_SERVERS
    after, reason = dw.render_mcp_remove(before, "exa")
    assert reason == ""
    diff = "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), n=0,
    ))
    # Only "exa"-related lines appear in the diff; "apollo" and "gmail" never do.
    assert "exa" in diff
    assert "apollo" not in diff
    assert "gmail" not in diff


# --------------------------------------------------------------------------- #
# render_mcp_remove: refusals
# --------------------------------------------------------------------------- #

def test_render_mcp_remove_refuses_when_server_absent():
    content, reason = dw.render_mcp_remove(_TWO_SERVERS, "nope")
    assert content is None
    assert "not in that file" in reason


def test_render_mcp_remove_refuses_no_pre_image():
    content, reason = dw.render_mcp_remove(None, "apollo")
    assert content is None
    assert "no config file" in reason


def test_render_mcp_remove_refuses_empty_server_name():
    content, reason = dw.render_mcp_remove(_TWO_SERVERS, "")
    assert content is None
    assert "no server name" in reason


def test_render_mcp_remove_refuses_malformed_json():
    content, reason = dw.render_mcp_remove("{not json", "apollo")
    assert content is None
    assert "not valid JSON" in reason


def test_render_mcp_remove_refuses_no_mcp_servers_block():
    content, reason = dw.render_mcp_remove('{"other": {}}\n', "apollo")
    assert content is None
    assert "not in that file" in reason


# --------------------------------------------------------------------------- #
# mcp_remove_precheck: the apply-time re-verification
# --------------------------------------------------------------------------- #

def test_mcp_remove_precheck_ok(tmp_path):
    path = tmp_path / ".mcp.json"
    path.write_text(_TWO_SERVERS, encoding="utf-8")
    check = dw.mcp_remove_precheck(str(path), "apollo")
    assert check["ok"] is True
    assert check["target_path"] == str(path)


def test_mcp_remove_precheck_refuses_missing_args():
    assert dw.mcp_remove_precheck("", "apollo")["ok"] is False
    assert dw.mcp_remove_precheck("/x/y.json", "")["ok"] is False


def test_mcp_remove_precheck_refuses_missing_file(tmp_path):
    check = dw.mcp_remove_precheck(str(tmp_path / "nope.json"), "apollo")
    assert check["ok"] is False
    assert "no longer exists" in check["reason"]


def test_mcp_remove_precheck_refuses_malformed_json(tmp_path):
    path = tmp_path / ".mcp.json"
    path.write_text("{not json", encoding="utf-8")
    check = dw.mcp_remove_precheck(str(path), "apollo")
    assert check["ok"] is False
    assert "not valid JSON" in check["reason"]


def test_mcp_remove_precheck_refuses_already_removed_by_hand(tmp_path):
    # The server was configured when the card was built, but a human already
    # ran `claude mcp remove` themselves before the fix was approved.
    path = tmp_path / ".mcp.json"
    path.write_text(_TWO_SERVERS, encoding="utf-8")
    check = dw.mcp_remove_precheck(str(path), "nonexistent")
    assert check["ok"] is False
    assert "no longer" in check["reason"]


# --------------------------------------------------------------------------- #
# The real apply / revert round trip, through relearn_apply.apply_relearn_fix
# --------------------------------------------------------------------------- #

def test_mcp_remove_apply_revert_round_trip(cfg, tmp_path):
    # Arrange: a real repo with a committed .mcp.json.
    repo = _git_repo(tmp_path)
    target = repo / ".mcp.json"
    target.write_text(_TWO_SERVERS, encoding="utf-8")
    _commit_all(repo, "add mcp config")

    # Act: apply for real.
    result = pa.apply_relearn_fix(
        cfg, _mcp_cluster(str(target), "apollo"),
        target_path=str(target), scope="project", go=True,
    )

    # Assert: the entry is gone, everything else survives, it's committed.
    assert result["dry_run"] is False
    record = result["record"]
    assert record["kind"] == dw.APPLY_KIND_MCP_REMOVE
    after = target.read_text()
    assert json.loads(after)["mcpServers"] == {
        "exa": {"command": "npx", "args": ["-y", "@exa/mcp"]},
    }
    assert record["git_commit"]
    log = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"], cwd=repo, capture_output=True, text=True,
    )
    assert dw.APPLY_KIND_MCP_REMOVE in log.stdout

    # Act: one-call revert.
    reverted = pa.revert_applied_fix(cfg, record["id"])

    # Assert: byte-for-byte back, and the revert is committed too.
    assert reverted["state"] == "reverted"
    assert target.read_text() == _TWO_SERVERS
    assert reverted["revert_commit"]


def test_mcp_remove_dry_run_writes_nothing(cfg, tmp_path):
    target = tmp_path / ".mcp.json"
    target.write_text(_TWO_SERVERS, encoding="utf-8")
    result = pa.apply_relearn_fix(
        cfg, _mcp_cluster(str(target), "apollo"),
        target_path=str(target), scope="project", go=False,
    )
    assert result["dry_run"] is True
    assert "apollo" in result["diff"]
    assert target.read_text() == _TWO_SERVERS
    assert not pa.list_applied(cfg)


def test_mcp_remove_revalidates_at_write_time(cfg, tmp_path):
    # The card was built while "apollo" was still configured; by apply time a
    # human already removed it by hand. The write must refuse, not blow away
    # the rest of the file.
    target = tmp_path / ".mcp.json"
    target.write_text(_TWO_SERVERS, encoding="utf-8")
    check = dw.mcp_remove_precheck(str(target), "apollo")
    assert check["ok"] is True

    # Simulate the hand-removal: "apollo" is gone, "exa" is untouched.
    already_removed = '{\n  "mcpServers": {\n    "exa": {\n      "command": "npx",\n      "args": ["-y", "@exa/mcp"]\n    }\n  }\n}\n'
    assert json.loads(already_removed)["mcpServers"] == {"exa": {"command": "npx", "args": ["-y", "@exa/mcp"]}}
    target.write_text(already_removed, encoding="utf-8")

    with pytest.raises(pa.RelearnApplyRefused, match="no longer"):
        pa.apply_relearn_fix(
            cfg, _mcp_cluster(str(target), "apollo"),
            target_path=str(target), scope="project", go=True,
        )


def test_mcp_remove_refuses_stale_target_path(cfg, tmp_path):
    # cluster.source_path points somewhere else than the confirmed target_path
    # -- must never silently write the confirmed target using the cluster's
    # own (possibly stale) source instead.
    real = tmp_path / "real.json"
    real.write_text(_TWO_SERVERS, encoding="utf-8")
    stale_target = tmp_path / "stale.json"
    stale_target.write_text(_TWO_SERVERS, encoding="utf-8")

    with pytest.raises(pa.RelearnApplyRefused, match="not"):
        pa.apply_relearn_fix(
            cfg, _mcp_cluster(str(real), "apollo"),
            target_path=str(stale_target), scope="project", go=True,
        )


def test_mcp_remove_is_a_known_apply_kind(cfg, tmp_path):
    # Sibling to test_model_apply.py's test_unknown_apply_kind_is_refused --
    # mcp_remove must validate alongside the model-routing kinds, not be
    # rejected as unknown.
    target = tmp_path / ".mcp.json"
    target.write_text(_TWO_SERVERS, encoding="utf-8")
    result = pa.apply_relearn_fix(
        cfg, _mcp_cluster(str(target), "apollo"),
        target_path=str(target), scope="project", go=False,
    )
    assert result["dry_run"] is True
