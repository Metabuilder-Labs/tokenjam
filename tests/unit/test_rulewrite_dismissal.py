"""Dismissal, moved out of the browser — durable, reversible, and figure-safe.

It used to live in `localStorage`, which means it was not really recorded: a
cleared profile, a second browser or a different machine and every dismissed
card came back. Making it durable raises the stakes of getting two things
right — it must not touch the observation, and it must be undoable.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from tokenjam.core.rulewrite.kinds import DELIVERY_CLAUDE_MD_RULE

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import cost_apply, dismissals
from tokenjam.core.rulewrite import RuleWriteRefused, list_rule_writes, stage_rule
from tokenjam.core.rulewrite.plan import DISMISSED_REASON


@pytest.fixture()
def config(tmp_path: Path) -> TjConfig:
    return TjConfig(
        version="1", storage=StorageConfig(path=str(tmp_path / "tj" / "tj.duckdb")),
    )


def _seed(config: TjConfig, **kw) -> None:
    from tokenjam.core.optimize import relearn_store

    proposal = {
        "kind": "cost", "analyzer": "subagent", "signature": "cost:subagent",
        "title": "Right-size subagents", "delivery": DELIVERY_CLAUDE_MD_RULE,
        "proposed_fix": "Default every subagent to the cheapest model that fits.",
        "past_overspend_usd": 412.5, "past_overspend_tokens": 9_100_000,
        "placement_scope": "project", "placement_paths": ["/repo/a/CLAUDE.md"],
    }
    proposal.update(kw)
    relearn_store.write_cost_proposals([proposal], config=config, window_days=30)


# --- it is durable and server-side -------------------------------------------#

def test_a_dismissal_is_written_beside_the_db_not_the_browser(config, tmp_path):
    """The founder's ask was that this stop living in the browser. It lands in
    the same server-side home the applied/reverted records use — a lock-free
    sink beside the DB, because `tj rules` must answer while the daemon holds
    the DuckDB write lock and DuckDB has no concurrent read-only escape."""
    dismissals.dismiss(config, "cost:subagent")
    path = dismissals.dismissals_path(config)
    assert path.is_file()
    assert path.parent == Path(config.storage.path).parent
    records = json.loads(path.read_text(encoding="utf-8"))
    assert records[0]["signature"] == "cost:subagent"
    assert records[0]["state"] == "dismissed"


def test_a_dismissal_survives_a_fresh_read(config):
    """The whole point: it is recorded, not held in a page's memory."""
    _seed(config)
    assert list_rule_writes(config)[0].offered is True
    dismissals.dismiss(config, "cost:subagent")
    rule = list_rule_writes(config)[0]
    assert rule.dismissed is True
    assert rule.offered is False
    assert rule.blocked_reason == DISMISSED_REASON


def test_dismissing_twice_does_not_duplicate_the_record(config):
    dismissals.dismiss(config, "cost:subagent")
    dismissals.dismiss(config, "cost:subagent")
    assert len(dismissals.list_dismissals(config)) == 1


# --- it is REVERSIBLE ---------------------------------------------------------#

def test_a_dismissal_can_be_undone(config):
    """The half that makes durability safe to offer at all. Without it the user
    trades a card that came back on every browser for one that never comes back
    anywhere."""
    _seed(config)
    dismissals.dismiss(config, "cost:subagent")
    assert list_rule_writes(config)[0].dismissed is True

    restored = dismissals.undismiss(config, "cost:subagent")
    assert restored is not None and restored["state"] == "restored"
    rule = list_rule_writes(config)[0]
    assert rule.dismissed is False
    assert rule.offered is True


def test_undismissing_something_never_dismissed_is_a_no_op(config):
    assert dismissals.undismiss(config, "cost:nope") is None


def test_a_dismissed_row_stays_listed_so_there_is_something_to_restore(config):
    """A durable dismissal that vanished the row would leave the user no way
    back — the same reason `applied` stays listed."""
    _seed(config)
    dismissals.dismiss(config, "cost:subagent")
    rules = list_rule_writes(config)
    assert [r.signature for r in rules] == ["cost:subagent"]
    assert rules[0].dismissed is True


# --- THE figure is untouched (Critical Rule 32) ------------------------------#

def test_the_past_figure_is_byte_identical_with_and_without_a_dismissal(config):
    """Dismissal is a statement about our RECOMMENDATION, not about the user's
    bill. The behaviour happened and cost what it cost; "not this one" cannot
    un-spend it."""
    _seed(config)
    before = asdict(list_rule_writes(config)[0])

    dismissals.dismiss(config, "cost:subagent")
    after = asdict(list_rule_writes(config)[0])

    assert after["past_overspend_usd"] == before["past_overspend_usd"] == 412.5
    assert repr(after["past_overspend_usd"]) == repr(before["past_overspend_usd"])
    assert repr(after["past_overspend_tokens"]) == repr(before["past_overspend_tokens"])
    # The diff is EXACTLY the offer fields — nothing else moved.
    assert {k for k in before if before[k] != after[k]} == {
        "dismissed", "offered", "blocked_reason",
    }


def test_a_dismissed_rule_never_reports_a_zero_figure(config):
    _seed(config)
    dismissals.dismiss(config, "cost:subagent")
    rule = list_rule_writes(config)[0]
    assert rule.past_overspend_usd == 412.5
    assert rule.past_overspend_usd not in (0.0, None)


# --- one matcher, one filter --------------------------------------------------#

def test_dismissal_reuses_the_shared_signature_matcher(config):
    """A dismissal has to honour the same legacy-signature equivalence an apply
    does, or a card dismissed under one signature reappears under its
    refinement — the defect a second matcher would reintroduce."""
    _seed(
        config, analyzer="downsize",
        signature="cost:downsize:claude-code:anthropic:opus:sonnet",
    )
    dismissals.dismiss(config, "cost:downsize:claude-code")
    assert list_rule_writes(config)[0].dismissed is True
    assert cost_apply.signature_is_applied(
        "cost:downsize:claude-code:anthropic:opus:sonnet",
        {"cost:downsize:claude-code"},
    )


def test_the_reverted_record_filter_is_not_copied_a_fourth_time():
    """Three ledgers, one exclusion rule, stated once each. An inline copy is
    a chance to forget the exclusion and keep hiding a card the user restored."""
    for module in (cost_apply, dismissals):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "def applied_signatures" in source or "def dismissed_signatures" in source
    for rel in (
        "tokenjam/api/routes/relearn.py",
        "tokenjam/cli/cost_proposal_verbs.py",
        "tokenjam/core/rulewrite/plan.py",
    ):
        source = Path(rel).read_text(encoding="utf-8")
        assert 'if rec.get("state") != "reverted"' not in source, rel


def test_an_unreadable_ledger_leaves_every_rule_on_offer(config):
    _seed(config)
    path = dismissals.dismissals_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert dismissals.dismissed_signatures(config) == set()
    assert list_rule_writes(config)[0].offered is True


# --- staging honours it -------------------------------------------------------#

def test_a_dismissed_rule_cannot_be_staged(config, tmp_path):
    target = tmp_path / "a" / "CLAUDE.md"
    target.parent.mkdir(parents=True)
    target.write_text("# a\n", encoding="utf-8")
    _seed(config, placement_paths=[str(target)])
    dismissals.dismiss(config, "cost:subagent")
    rule = list_rule_writes(config)[0]
    with pytest.raises(RuleWriteRefused):
        stage_rule(config, rule)
    assert target.read_text(encoding="utf-8") == "# a\n"
