"""``tj relearn cost-proposals`` / ``cost-apply`` / ``cost-mark-applied`` /
``cost-revert`` -- the CLI's first view of the cost-analyzer proposal store
(core.optimize.cost_proposals), which before this file had a web renderer
but no terminal one at all.

Isolated: every write is routed under ``tmp_path`` via ``cfg.storage.path``
(same standard as ``test_relearn_cli.py``); ``mark-applied`` / ``cost-apply
--go`` use a real ``InMemoryBackend`` (an ``Expectation`` marker needs a real
``expectations`` table) rather than touching a real ``~/.tj``.
"""
from __future__ import annotations

import json
import subprocess

import click
import pytest
from click.testing import CliRunner

from tokenjam.cli.cost_proposal_verbs import register_cost_proposal_verbs
from tokenjam.core.config import ProviderBudget, StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import cost_apply, relearn_store
from tokenjam.core.optimize.cost_proposals import CostProposal

# --- Fixtures ----------------------------------------------------------------

@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _advise_only(**overrides) -> CostProposal:
    """A deadweight-shaped advise-only proposal: a real copy-pasteable
    snippet, no workspace tokenjam can write into."""
    base = dict(
        kind="cost", analyzer="deadweight", signature="cost:deadweight:foo",
        title="Unused MCP server: foo",
        target_key={"server": "foo", "scope": "user", "source": "user config"},
        evidence="`foo` MCP server made 0 tool calls across 12 sessions.",
        baseline={"sessions_present": 12, "invocations": 0},
        advise_text="Remove it; it costs a standing schema-injection tax.",
        suggestion="claude mcp remove foo --scope user",
        estimated_recoverable_usd=12.5,
        estimated_recoverable_tokens=50_000,
        estimate_basis="measured over the last 90d.",
        apply_capable=False,
    )
    base.update(overrides)
    return CostProposal(**base)


def _apply_capable(*, target_path: str, **overrides) -> CostProposal:
    """A subagent-shaped proposal with a real workspace surface to write into."""
    base = dict(
        kind="cost", analyzer="subagent", signature="cost:subagent:bar",
        title="Right-size subagent bar",
        target_key={"agent_name": "bar"},
        evidence="`bar` subagent averages 200 tokens output across 40 calls.",
        baseline={"apply_sessions": 40, "apply_repos": ["demo"]},
        advise_text="Route `bar` to a smaller model.",
        estimated_recoverable_usd=8.0,
        estimated_recoverable_tokens=20_000,
        estimate_basis="measured over the last 30d.",
        apply_capable=True,
        rung=1,
        scope="project",
        proposed_fix="`bar` rarely needs deep reasoning -- size it to a smaller model.",
        target_path=target_path,
    )
    base.update(overrides)
    return CostProposal(**base)


def _store(cfg, *proposals: CostProposal) -> list[str]:
    """Persist cost proposals the way a real recompute would, and return the
    stored proposal IDs."""
    from tokenjam.core.optimize import relearn_proposals

    relearn_store.write_cost_proposals(list(proposals), config=cfg)
    return [p["proposal_id"] for p in relearn_proposals.list_cost_proposals(cfg)]


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
    group = click.Group("relearn")
    return register_cost_proposal_verbs(group)


def _run(cfg, args, *, db=None, output_json=False):
    return CliRunner().invoke(
        _group(), args,
        obj={"config": cfg, "db": db, "output_json": output_json},
    )


# --- cost-proposals: empty states ----------------------------------------------

def test_never_computed_says_so_not_a_bare_empty_state(cfg):
    result = _run(cfg, ["cost-proposals"])
    assert result.exit_code == 0, result.output
    assert "never been computed" in result.output
    assert "tj optimize" in result.output


def test_computed_but_nothing_flagged_says_why(cfg):
    relearn_store.write_cost_proposals([], config=cfg)
    result = _run(cfg, ["cost-proposals"])
    assert result.exit_code == 0, result.output
    assert "No cost-saving fixes found" in result.output


# --- cost-proposals: rendering --------------------------------------------------

def test_list_renders_the_snippet_on_its_own_line(cfg):
    _store(cfg, _advise_only())
    result = _run(cfg, ["cost-proposals"])
    assert result.exit_code == 0, result.output
    assert "claude mcp remove foo --scope user" in result.output


def test_list_marks_advise_only_and_states_no_apply_path(cfg):
    _store(cfg, _advise_only())
    result = _run(cfg, ["cost-proposals"])
    assert "advise-only" in result.output
    assert "no workspace" in result.output


def test_list_marks_apply_capable_and_names_the_real_command(cfg, tmp_path):
    [pid] = _store(cfg, _apply_capable(target_path=str(tmp_path / "CLAUDE.md")))
    result = _run(cfg, ["cost-proposals"])
    assert "workspace fix" in result.output
    assert f"tj relearn cost-apply {pid}" in result.output


def test_list_json_carries_the_full_proposal_and_framing(cfg):
    [pid] = _store(cfg, _advise_only())
    result = _run(cfg, ["cost-proposals"], output_json=True)
    payload = json.loads(result.output)
    assert payload["status"] == "ready"
    assert payload["proposals"][0]["proposal_id"] == pid
    assert "framing" in payload


def test_list_shows_estimated_recoverable_rollup(cfg):
    _store(cfg, _advise_only())
    result = _run(cfg, ["cost-proposals"])
    assert "estimated recoverable" in result.output


def test_list_omits_the_advise_only_footer_when_every_proposal_is_apply_capable(cfg, tmp_path):
    _store(cfg, _apply_capable(target_path=str(tmp_path / "CLAUDE.md")))
    result = _run(cfg, ["cost-proposals"])
    assert "Advise-only proposals have no apply path" not in result.output


# --- pricing-mode honesty -------------------------------------------------------

def test_subscription_plan_suppresses_the_raw_dollar_figure(tmp_path):
    cfg = TjConfig(
        version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")),
        budgets={"anthropic": ProviderBudget(plan="max_5x")},
    )
    _store(cfg, _advise_only())
    result = _run(cfg, ["cost-proposals"])
    assert "$12.50" not in result.output
    assert "12.5" not in result.output.replace("12,500", "")  # no bare dollar amount leaks


# --- cost-apply: the workspace-write verb ---------------------------------------

def test_cost_apply_rejects_an_id_the_detector_never_produced(cfg):
    result = _run(cfg, ["cost-apply", "rp_000000000000"])
    assert result.exit_code != 0
    assert "no stored cost proposal" in result.output


def test_cost_apply_refuses_an_advise_only_proposal(cfg):
    [pid] = _store(cfg, _advise_only())
    result = _run(cfg, ["cost-apply", pid])
    assert result.exit_code != 0
    assert "advise-only" in result.output
    assert "cost-mark-applied" in result.output


def test_cost_apply_defaults_to_a_dry_run(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    [pid] = _store(cfg, _apply_capable(target_path=str(target)))

    result = _run(cfg, ["cost-apply", pid])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert target.read_text() == "# Repo\n"
    assert not cost_apply.list_applied(cfg)


def test_cost_apply_go_writes_and_marks_the_cost_ledger(cfg, tmp_path, db):
    repo = _git_repo(tmp_path)
    target = repo / "CLAUDE.md"
    [pid] = _store(cfg, _apply_capable(target_path=str(target)))

    result = _run(cfg, ["cost-apply", pid, "--go"], db=db, output_json=True)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["applied"]["record"]["target_path"] == str(target)
    assert payload["cost_record"] is not None
    applied = cost_apply.list_applied(cfg)
    assert len(applied) == 1
    assert applied[0]["signature"] == "cost:subagent:bar"


# --- cost-mark-applied / cost-revert --------------------------------------------

def test_mark_applied_requires_a_direct_db_connection(cfg):
    [pid] = _store(cfg, _advise_only())
    result = _run(cfg, ["cost-mark-applied", pid])
    assert result.exit_code != 0
    assert "database connection" in result.output


def test_mark_applied_records_a_ledger_entry(cfg, db):
    [pid] = _store(cfg, _advise_only())
    result = _run(cfg, ["cost-mark-applied", pid], db=db, output_json=True)
    assert result.exit_code == 0, result.output
    rec = json.loads(result.output)
    assert rec["signature"] == "cost:deadweight:foo"
    assert rec["state"] == "applied"


def test_mark_applied_is_idempotent_per_signature(cfg, db):
    [pid] = _store(cfg, _advise_only())
    first = json.loads(_run(cfg, ["cost-mark-applied", pid], db=db, output_json=True).output)
    second = json.loads(_run(cfg, ["cost-mark-applied", pid], db=db, output_json=True).output)
    assert first["id"] == second["id"]
    assert len(cost_apply.list_applied(cfg)) == 1


def test_cost_revert_flips_the_ledger_state(cfg, db):
    [pid] = _store(cfg, _advise_only())
    rec = json.loads(_run(cfg, ["cost-mark-applied", pid], db=db, output_json=True).output)

    result = _run(cfg, ["cost-revert", rec["id"]], output_json=True)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["state"] == "reverted"


def test_cost_revert_of_an_unknown_record_is_a_clean_error(cfg):
    result = _run(cfg, ["cost-revert", "not-a-record"])
    assert result.exit_code != 0
    assert "not-a-record" in result.output


# --- registration + house voice -------------------------------------------------

def test_every_verb_is_registered():
    assert sorted(_group().commands) == [
        "cost-apply", "cost-mark-applied", "cost-proposals", "cost-revert",
    ]


@pytest.mark.parametrize("args", [
    ["cost-proposals", "--help"], ["cost-apply", "--help"],
    ["cost-mark-applied", "--help"], ["cost-revert", "--help"],
])
def test_help_text_avoids_em_dashes_and_the_word_quota(args):
    out = CliRunner().invoke(_group(), args, obj={}).output
    assert "—" not in out
    assert "quota" not in out.lower()


def test_human_output_avoids_em_dashes_and_the_word_quota(cfg, tmp_path):
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Repo\n", encoding="utf-8")
    _store(cfg, _advise_only(), _apply_capable(target_path=str(target)))
    out = "".join(
        _run(cfg, args).output
        for args in (["cost-proposals"], ["cost-apply", "rp_nope"])
    )
    assert "—" not in out
    assert "quota" not in out.lower()
