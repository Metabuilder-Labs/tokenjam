"""``tj relearn`` (list / apply / enable / revert / verify) and the stored
proposal record the write paths now require.

Everything is routed at a tmp storage dir and every write target is under
``tmp_path``, so nothing here can touch a real ``~/.tj`` or ``~/.claude``
(same standard as ``test_relearn_apply.py``). The apply/revert round-trip runs
against a real ``git init`` repo with real disk writes, because a fix that
cannot be reverted on a real repo is not reverted.
"""
from __future__ import annotations

import json
import subprocess

import click
import pytest
from click.testing import CliRunner

from tokenjam.cli.relearn_write_verbs import register_write_verbs
from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import relearn_apply as pa
from tokenjam.core.optimize import relearn_proposals, relearn_store
from tokenjam.core.optimize.analyzers.relearn import (
    RelearnCluster,
    RelearnExample,
    RelearnFinding,
)

# --- Fixtures ----------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _cluster(**overrides) -> RelearnCluster:
    base = dict(
        signature="cwd_confusion", family_key="cwd_confusion",
        title="cwd / relative-path confusion", sessions=12, occurrences=324,
        repos=["demo"], rung=1, scope="project",
        proposed_fix="Verify an absolute cwd before a relative Read.",
        examples=[RelearnExample(session_id="s1", repo="demo", ts=None, snippet="no such file")],
        estimated_recoverable_tokens=486_000,
    )
    base.update(overrides)
    return RelearnCluster(**base)


def _store(cfg, *clusters) -> list[str]:
    """Persist a detector finding the way a real recompute would, and return
    the stored proposal IDs."""
    relearn_store.write_cache(RelearnFinding(clusters=list(clusters)), config=cfg)
    return [p["proposal_id"] for p in relearn_proposals.list_proposals(cfg)]


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "CLAUDE.md").write_text("# Repo\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def _group() -> click.Group:
    """The verbs attached to a ``tj relearn`` group, exactly as the real group
    picks them up. Built here rather than imported so this file tests the
    registration function too."""
    group = click.Group("relearn")
    return register_write_verbs(group)


def _run(cfg, args, *, output_json=False):
    return CliRunner().invoke(
        _group(), args,
        obj={"config": cfg, "db": None, "output_json": output_json},
    )


# --- F2: the stored proposal record -------------------------------------------

def test_proposal_ids_are_stamped_at_detection_time(cfg):
    _store(cfg, _cluster())
    # The ID is on the persisted record itself, not synthesised at read time.
    raw = json.loads(relearn_store.default_cache_path(cfg).read_text(encoding="utf-8"))
    stored_cluster = raw["finding"]["clusters"][0]
    assert stored_cluster["proposal_id"].startswith(relearn_proposals.PROPOSAL_ID_PREFIX)


def test_proposal_id_is_stable_across_recomputes(cfg):
    first = _store(cfg, _cluster())
    second = _store(cfg, _cluster(occurrences=999))
    assert first == second


def test_distinct_signatures_get_distinct_ids(cfg):
    ids = _store(cfg, _cluster(), _cluster(signature="sleep_chain", family_key="sleep_chain"))
    assert len(set(ids)) == 2


def test_get_proposal_returns_none_for_an_unknown_id(cfg):
    _store(cfg, _cluster())
    assert relearn_proposals.get_proposal("rp_deadbeefdead", config=cfg) is None


def test_cluster_for_apply_drops_display_only_fields(cfg):
    _store(cfg, _cluster())
    stored = relearn_proposals.list_proposals(cfg)[0]
    cluster = relearn_proposals.cluster_for_apply(stored)
    assert cluster["signature"] == "cwd_confusion"
    assert "estimated_recoverable_tokens" not in cluster
    assert "proposal_id" not in cluster


# --- F1: list ------------------------------------------------------------------

def test_list_shows_proposal_ids(cfg):
    [pid] = _store(cfg, _cluster())
    result = _run(cfg, ["list"])
    assert result.exit_code == 0, result.output
    assert pid in result.output


def test_list_empty_state_is_not_an_error(cfg):
    result = _run(cfg, ["list"])
    assert result.exit_code == 0
    assert "No proposals stored yet" in result.output


def test_list_json_carries_the_full_proposal(cfg):
    [pid] = _store(cfg, _cluster())
    result = _run(cfg, ["list"], output_json=True)
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["proposals"][0]["proposal_id"] == pid


# --- F1: apply is a dry run by default ----------------------------------------

def test_apply_defaults_to_a_dry_run(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    [pid] = _store(cfg, _cluster())

    result = _run(cfg, ["apply", pid, "--target", str(target)])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert target.read_text() == "# Repo\n"
    assert not pa.list_applied(cfg)


def test_apply_rejects_an_id_the_detector_never_produced(cfg, tmp_path):
    _store(cfg, _cluster())
    result = _run(cfg, ["apply", "rp_000000000000", "--target", str(tmp_path / "CLAUDE.md")])
    assert result.exit_code != 0
    assert "no stored proposal" in result.output


def test_apply_refuses_a_matcherless_family_at_an_enforcement_rung(cfg, tmp_path):
    [pid] = _store(cfg, _cluster(signature="mystery", family_key="mystery_family", rung=3))
    result = _run(cfg, ["apply", pid, "--go",
                        "--target", str(tmp_path / ".claude" / "hooks" / "mystery.py")])
    assert result.exit_code != 0
    assert "no matcher exists" in result.output
    assert "rung 1" in result.output


# --- F1: the end-to-end round trip against a real proposal + real repo ---------

def test_apply_go_then_revert_round_trips_on_a_real_repo(cfg, tmp_path):
    repo = _git_repo(tmp_path)
    target = repo / "CLAUDE.md"
    [pid] = _store(cfg, _cluster())

    applied = _run(cfg, ["apply", pid, "--go", "--target", str(target)], output_json=True)
    assert applied.exit_code == 0, applied.output
    record = json.loads(applied.output)["record"]
    assert pa.NOTE_SECTION_HEADER in target.read_text()
    assert record["git_commit"]                       # committed into the real repo

    reverted = _run(cfg, ["revert", record["id"]], output_json=True)
    assert reverted.exit_code == 0, reverted.output
    assert target.read_text() == "# Repo\n"           # pre-image restored byte for byte
    assert json.loads(reverted.output)["state"] == "reverted"


def test_apply_uses_the_proposals_suggested_target_when_none_is_given(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    [pid] = _store(cfg, _cluster(suggested_target=str(target)))

    result = _run(cfg, ["apply", pid, "--go"], output_json=True)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["record"]["target_path"] == str(target)


def test_apply_without_any_target_says_so(cfg):
    [pid] = _store(cfg, _cluster(suggested_target=""))
    result = _run(cfg, ["apply", pid, "--go"])
    assert result.exit_code != 0
    assert "--target" in result.output


# --- F1: enable stays human-gated ---------------------------------------------

def _apply_enforcement_fix(cfg, tmp_path) -> str:
    [pid] = _store(cfg, _cluster(signature="sleep_chain", family_key="sleep_chain",
                                 title="blocked sleep-chain", rung=3))
    target = tmp_path / ".claude" / "hooks" / "blocked-sleep-chain.py"
    result = _run(cfg, ["apply", pid, "--go", "--target", str(target)], output_json=True)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)["record"]["id"]


def test_enable_without_yes_refuses_and_writes_nothing(cfg, tmp_path):
    fix_id = _apply_enforcement_fix(cfg, tmp_path)
    result = _run(cfg, ["enable", fix_id])
    assert result.exit_code != 0
    assert not (tmp_path / ".claude" / "settings.json").exists()
    assert pa.get_applied(cfg, fix_id)["enforcement"]["enabled"] is False


def test_enable_with_yes_wires_the_hook(cfg, tmp_path):
    fix_id = _apply_enforcement_fix(cfg, tmp_path)
    result = _run(cfg, ["enable", fix_id, "--yes"])
    assert result.exit_code == 0, result.output
    assert pa.get_applied(cfg, fix_id)["enforcement"]["enabled"] is True


def test_revert_of_an_unknown_fix_id_is_a_clean_error(cfg):
    result = _run(cfg, ["revert", "notafixid"])
    assert result.exit_code != 0
    assert "notafixid" in result.output


# --- G2: on-demand verify ------------------------------------------------------

def test_verify_recomputes_the_receipt_for_an_applied_fix(cfg, tmp_path, monkeypatch):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    [pid] = _store(cfg, _cluster())
    applied = _run(cfg, ["apply", pid, "--go", "--target", str(target)], output_json=True)
    fix_id = json.loads(applied.output)["record"]["id"]
    assert pa.get_applied(cfg, fix_id)["verify"]["last_checked_at"] is None

    result = _run(cfg, ["verify"], output_json=True)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["fixes"] == {"checked": 1, "updated": 1}
    assert pa.get_applied(cfg, fix_id)["verify"]["last_checked_at"] is not None


def test_verify_with_no_applied_fixes_is_a_clean_no_op(cfg):
    result = _run(cfg, ["verify"])
    assert result.exit_code == 0, result.output
    assert "0/0 applied fixes" in result.output


# --- House voice ---------------------------------------------------------------

def test_every_verb_is_registered_on_the_group():
    assert sorted(_group().commands) == ["apply", "enable", "list", "revert", "verify"]


@pytest.mark.parametrize("args", [
    ["--help"], ["list", "--help"], ["apply", "--help"],
    ["enable", "--help"], ["revert", "--help"], ["verify", "--help"],
])
def test_cli_help_text_avoids_em_dashes_and_the_word_quota(args):
    out = CliRunner().invoke(_group(), args, obj={}).output
    assert "—" not in out
    assert "quota" not in out.lower()


def test_human_output_avoids_em_dashes_and_the_word_quota(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    [pid] = _store(cfg, _cluster())
    out = "".join(
        _run(cfg, args).output
        for args in (["list"], ["apply", pid, "--target", str(target)],
                     ["apply", pid, "--go", "--target", str(target)], ["verify"])
    )
    assert "—" not in out
    assert "quota" not in out.lower()
