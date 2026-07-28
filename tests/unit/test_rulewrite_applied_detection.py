"""Do not offer a fix the user has already dealt with — and do not touch the
figure while doing it.

Two halves, and the second is the one that goes wrong. Suppressing the OFFER is
straightforward bookkeeping. What must NOT happen is the suppression reaching
back into `past_overspend_usd`/`_tokens`: the waste genuinely happened inside
the analyzed window, and the user having since fixed it does not un-spend the
money. Critical Rule 32 names this exact conflation — an action-availability
gate establishing that WE have no remedy, never that the spend was avoidable
or free — and it is easy to get wrong here because "it is fixed now" feels
like it ought to zero the number.

The ledger is the one already in the tree (`cost_applied.json` /
`applied_fixes.json`), read through `cost_apply.applied_signatures` /
`relearn_apply.applied_signatures`, and matched with
`cost_apply.signature_is_applied`. No second ledger, no second matcher.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tokenjam.core.rulewrite.kinds import DELIVERY_CLAUDE_MD_RULE

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import cost_apply, relearn_apply
from tokenjam.core.rulewrite import RuleWriteRefused, list_rule_writes, stage_rule
from tokenjam.core.rulewrite.plan import ALREADY_APPLIED_REASON


@pytest.fixture()
def config(tmp_path: Path) -> TjConfig:
    return TjConfig(
        version="1", storage=StorageConfig(path=str(tmp_path / "tj" / "tj.duckdb")),
    )


def _seed_proposals(config: TjConfig, *proposals: dict) -> None:
    """Write the cost-proposal cache the rule surface reads."""
    from tokenjam.core.optimize import relearn_store

    relearn_store.write_cost_proposals(
        list(proposals), config=config, window_days=30,
    )


def _proposal(**kw) -> dict:
    base = {
        "kind": "cost",
        "analyzer": "subagent",
        "signature": "cost:subagent",
        "title": "Right-size Task-dispatched subagents",
        "delivery": DELIVERY_CLAUDE_MD_RULE,
        "proposed_fix": "Default every subagent to the cheapest model that fits.",
        "write_offered": True,
        "past_overspend_usd": 412.5,
        "past_overspend_tokens": 9_100_000,
        "placement_scope": "project",
        "placement_paths": ["/repo/alpha/CLAUDE.md"],
    }
    base.update(kw)
    return base


def _mark_cost_applied(config: TjConfig, signature: str, state: str = "applied") -> None:
    path = cost_apply.cost_applied_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    existing.append({"id": signature, "signature": signature, "state": state})
    path.write_text(json.dumps(existing), encoding="utf-8")


# --- the offer is suppressed ------------------------------------------------#

def test_an_applied_rule_is_not_on_offer(config):
    _seed_proposals(config, _proposal())
    assert list_rule_writes(config)[0].offered is True

    _mark_cost_applied(config, "cost:subagent")
    rule = list_rule_writes(config)[0]
    assert rule.already_applied is True
    assert rule.offered is False
    assert rule.blocked_reason == ALREADY_APPLIED_REASON


def test_an_applied_rule_still_appears_rather_than_vanishing(config):
    """A user who applies something and then sees nothing cannot tell "done"
    from "broken". The row survives, carrying its state."""
    _seed_proposals(config, _proposal())
    _mark_cost_applied(config, "cost:subagent")
    rules = list_rule_writes(config)
    assert [r.signature for r in rules] == ["cost:subagent"]
    assert rules[0].already_applied is True


def test_a_reverted_mark_reopens_the_offer(config):
    """A revert is the user saying the fix is no longer in place."""
    _seed_proposals(config, _proposal())
    _mark_cost_applied(config, "cost:subagent", state="reverted")
    rule = list_rule_writes(config)[0]
    assert rule.already_applied is False
    assert rule.offered is True


def test_staging_an_applied_rule_is_refused_with_the_real_reason(config, tmp_path):
    """And it names the actual reason. "The budget did not select it" would be
    both wrong and actively confusing for a rule the user already applied."""
    target = tmp_path / "alpha" / "CLAUDE.md"
    target.parent.mkdir(parents=True)
    target.write_text("# alpha\n", encoding="utf-8")
    _seed_proposals(config, _proposal(placement_paths=[str(target)]))
    _mark_cost_applied(config, "cost:subagent")

    rule = list_rule_writes(config)[0]
    with pytest.raises(RuleWriteRefused) as exc:
        stage_rule(config, rule)
    assert "already applied" in str(exc.value)
    assert "budget" not in str(exc.value)
    assert target.read_text(encoding="utf-8") == "# alpha\n"


# --- THE figure is untouched (Critical Rule 32) -----------------------------#

def test_the_past_figure_is_byte_identical_with_and_without_the_ledger_entry(
    config,
):
    """The load-bearing assertion of this whole layer.

    Everything about the rule EXCEPT its offer state must be identical. The
    money was spent inside the window; the user fixing it afterwards does not
    un-spend it, and a gate on whether we still have an action available may
    never edit what a behaviour already cost.
    """
    _seed_proposals(config, _proposal())
    before = asdict(list_rule_writes(config)[0])

    _mark_cost_applied(config, "cost:subagent")
    after = asdict(list_rule_writes(config)[0])

    assert after["past_overspend_usd"] == before["past_overspend_usd"] == 412.5
    assert after["past_overspend_tokens"] == before["past_overspend_tokens"]
    # Byte-identical, not merely equal-valued: `repr` catches a float that got
    # rounded, re-derived, or coerced through a zero on the way.
    assert repr(after["past_overspend_usd"]) == repr(before["past_overspend_usd"])
    assert repr(after["past_overspend_tokens"]) == repr(before["past_overspend_tokens"])

    # And nothing else moved either — the diff is EXACTLY the offer fields.
    changed = {k for k in before if before[k] != after[k]}
    assert changed == {"already_applied", "offered", "blocked_reason"}


def test_an_applied_rule_never_reports_a_zero_figure(config):
    """The specific failure mode Rule 32 describes: a gate turning a real,
    incurred cost into `$0`. `None` means "not measured" and 0.0 would mean
    "this was free"; neither is what "the user fixed it" means."""
    _seed_proposals(config, _proposal())
    _mark_cost_applied(config, "cost:subagent")
    rule = list_rule_writes(config)[0]
    assert rule.past_overspend_usd == 412.5
    assert rule.past_overspend_usd not in (0.0, None)
    assert rule.past_overspend_tokens == 9_100_000


# --- matching goes through the shared helper --------------------------------#

def test_a_legacy_agent_only_mark_still_covers_the_model_qualified_signature(
    config,
):
    """`signature_is_applied` already resolves this, and reusing it is the
    point: a second matcher here would silently reopen every card that helper
    settles, re-nagging a user for a fix they already recorded."""
    _seed_proposals(config, _proposal(
        analyzer="downsize",
        signature="cost:downsize:claude-code:anthropic:opus:sonnet",
    ))
    _mark_cost_applied(config, "cost:downsize:claude-code")
    rule = list_rule_writes(config)[0]
    assert rule.already_applied is True
    assert rule.offered is False
    # The legacy resolution is the helper's, not a copy of its rules.
    assert cost_apply.signature_is_applied(
        "cost:downsize:claude-code:anthropic:opus:sonnet",
        {"cost:downsize:claude-code"},
    )


def test_the_relearn_lane_reads_its_own_ledger(config):
    """The two lanes have separate ledgers (`applied_fixes.json` vs
    `cost_applied.json`), so a cost mark must not settle a relearn rule."""
    assert relearn_apply.applied_signatures(config) == set()
    _mark_cost_applied(config, "relearn:some-family")
    # Recorded in the COST ledger, so the relearn lane must not see it.
    assert relearn_apply.applied_signatures(config) == set()
    assert "relearn:some-family" in cost_apply.applied_signatures(config)


def test_an_unreadable_ledger_leaves_every_rule_on_offer(config):
    """The safe direction. A corrupt ledger that read as "everything applied"
    would hide fixes the user never made; reading it as "nothing applied" can
    only waste attention."""
    _seed_proposals(config, _proposal())
    path = cost_apply.cost_applied_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not valid json", encoding="utf-8")
    assert cost_apply.applied_signatures(config) == set()
    assert list_rule_writes(config)[0].offered is True


# --- the derivation is stated once ------------------------------------------#

def test_the_applied_set_excludes_reverted_records_everywhere():
    """The filter used to be written out inline at each call site. Three copies
    is three chances for one to keep counting a reverted record and re-hide a
    card the user re-opened."""
    from pathlib import Path as _Path

    for module in (cost_apply, relearn_apply):
        source = _Path(module.__file__).read_text(encoding="utf-8")
        assert "def applied_signatures" in source, module.__name__
    for rel in (
        "tokenjam/api/routes/relearn.py",
        "tokenjam/cli/cost_proposal_verbs.py",
        "tokenjam/core/rulewrite/plan.py",
    ):
        source = _Path(rel).read_text(encoding="utf-8")
        assert 'if rec.get("state") != "reverted"' not in source, rel
