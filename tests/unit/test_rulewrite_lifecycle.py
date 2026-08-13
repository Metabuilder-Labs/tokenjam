"""The shared rule-write lifecycle: stage -> check -> apply -> undo.

Four analyzers write rules and each used to do it independently. What is pinned
here is the safety model they now share, and the property that made a single
shared surface necessary in the first place: **one rule, N destinations, and a
partial outcome is a first-class result.** Three project files written and a
fourth skipped because it changed under us is the honest answer; a caller that
only counted successes would report the rule as fully applied.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.rulewrite import (
    RuleDestination,
    RuleWrite,
    RuleWriteRefused,
    apply_staged,
    check_staged,
    stage_rule,
    undo,
)
from tokenjam.core.rulewrite import store


@pytest.fixture()
def config(tmp_path: Path) -> TjConfig:
    return TjConfig(
        version="1", storage=StorageConfig(path=str(tmp_path / "tj" / "tj.duckdb")),
    )


def _rule(*paths: Path, **kw) -> RuleWrite:
    base = dict(
        signature="cost:subagent",
        analyzer="subagent",
        title="Right-size Task-dispatched subagents",
        artifact_text=(
            "Default every subagent to the cheapest same-family model that "
            "fits its shape."
        ),
        destinations=tuple(
            RuleDestination(path=str(p), scope="project", sessions=4) for p in paths
        ),
    )
    base.update(kw)
    return RuleWrite(**base)   # type: ignore[arg-type]


def _project(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True)
    target = root / "CLAUDE.md"
    target.write_text(f"# {name}\n\nExisting guidance.\n", encoding="utf-8")
    return target


# --- stage ------------------------------------------------------------------#

def test_staging_produces_one_diff_per_destination_and_writes_nothing(config, tmp_path):
    alpha, beta = _project(tmp_path, "alpha"), _project(tmp_path, "beta")
    before = {p: p.read_text(encoding="utf-8") for p in (alpha, beta)}

    staged = stage_rule(config, _rule(alpha, beta))

    assert [e.path for e in staged] == [str(alpha), str(beta)]
    for entry in staged:
        assert entry.diff.startswith("---")
        assert "Default every subagent" in entry.rendered
        assert entry.standing_tokens_per_session > 0
    # Staging is a read: the targets are untouched until an explicit apply.
    for path, text in before.items():
        assert path.read_text(encoding="utf-8") == text


def test_a_rule_the_budget_did_not_offer_cannot_be_staged(config, tmp_path):
    """The write budget's verdict is the product's own answer to "is this worth
    a permanent block". A staging path that ignored it would be a way to spend
    the budget without consulting it — while the rule's text stays copyable, so
    a deferral is still not a deletion."""
    alpha = _project(tmp_path, "alpha")
    rule = _rule(alpha, offered=False, blocked_reason="budget already allocated")
    with pytest.raises(RuleWriteRefused) as exc:
        stage_rule(config, rule)
    assert "budget already allocated" in str(exc.value)


def test_a_rule_with_no_resolved_destination_is_refused_with_the_reason(config):
    with pytest.raises(RuleWriteRefused) as exc:
        stage_rule(config, _rule())
    assert "no resolved destination" in str(exc.value)


def test_a_symlinked_destination_is_refused_outright(config, tmp_path):
    real = _project(tmp_path, "alpha")
    link = tmp_path / "link.md"
    link.symlink_to(real)
    with pytest.raises(RuleWriteRefused) as exc:
        stage_rule(config, _rule(link))
    assert "symlink" in str(exc.value)


# --- apply / undo round trip ------------------------------------------------#

def test_apply_is_a_dry_run_until_go_and_then_round_trips_through_undo(
    config, tmp_path,
):
    alpha, beta = _project(tmp_path, "alpha"), _project(tmp_path, "beta")
    original = {p: p.read_text(encoding="utf-8") for p in (alpha, beta)}
    stage_rule(config, _rule(alpha, beta))

    dry = apply_staged(config, "cost:subagent")
    assert dry["dry_run"] is True
    assert len(dry["applied"]) == 2
    assert alpha.read_text(encoding="utf-8") == original[alpha]

    wrote = apply_staged(config, "cost:subagent", go=True)
    assert wrote["dry_run"] is False
    assert wrote["skipped"] == []
    for path in (alpha, beta):
        text = path.read_text(encoding="utf-8")
        assert "Default every subagent" in text
        assert text != original[path]
    # Applying clears the staging entries: what was staged has landed.
    assert check_staged(config) == []

    reverted = undo(config, "cost:subagent", go=True)
    assert {r["path"] for r in reverted["restored"]} == {str(alpha), str(beta)}
    for path in (alpha, beta):
        assert path.read_text(encoding="utf-8") == original[path]


def test_undo_of_a_created_file_removes_it_rather_than_leaving_an_empty_one(
    config, tmp_path,
):
    """A rule written into a project with no CLAUDE.md creates the file.
    Restoring "the empty original" would leave a file the agent still loads,
    which reads as a successful revert while the rule's home persists."""
    root = tmp_path / "fresh"
    root.mkdir()
    target = root / "CLAUDE.md"
    stage_rule(config, _rule(target))
    apply_staged(config, "cost:subagent", go=True)
    assert target.is_file()

    undo(config, "cost:subagent", go=True)
    assert not target.exists()


def test_a_destination_that_drifted_is_skipped_with_its_reason_not_merged_into(
    config, tmp_path,
):
    """The load-bearing partial-outcome case. The diff a human approved
    described content that has since changed, so that file is left alone — and
    the result SAYS so, because "2 of 3 written" is the honest answer."""
    alpha, beta = _project(tmp_path, "alpha"), _project(tmp_path, "beta")
    stage_rule(config, _rule(alpha, beta))
    beta.write_text("# beta\n\nSomeone edited this.\n", encoding="utf-8")

    result = apply_staged(config, "cost:subagent", go=True)

    assert [row["path"] for row in result["applied"]] == [str(alpha)]
    assert [row["path"] for row in result["skipped"]] == [str(beta)]
    assert "changed since staging" in result["skipped"][0]["reason"]
    assert "Default every subagent" in alpha.read_text(encoding="utf-8")
    assert "Someone edited this" in beta.read_text(encoding="utf-8")


def test_check_reports_the_same_verdict_apply_will_enforce(config, tmp_path):
    alpha, beta = _project(tmp_path, "alpha"), _project(tmp_path, "beta")
    stage_rule(config, _rule(alpha, beta))
    beta.write_text("# beta\n\nEdited.\n", encoding="utf-8")

    rows = {row["path"]: row for row in check_staged(config)}
    assert rows[str(alpha)]["applyable"] is True
    assert rows[str(beta)]["applyable"] is False
    # Surfacing the reason up front is the difference between an Apply control
    # that explains itself and one that fails when clicked.
    assert rows[str(beta)]["reason"] == "changed since staging — re-stage it"


def test_undo_refuses_when_the_file_changed_after_apply(config, tmp_path):
    alpha = _project(tmp_path, "alpha")
    stage_rule(config, _rule(alpha))
    apply_staged(config, "cost:subagent", go=True)
    alpha.write_text(
        alpha.read_text(encoding="utf-8") + "\nA later human edit.\n",
        encoding="utf-8",
    )

    result = undo(config, "cost:subagent", go=True)

    assert result["restored"] == []
    assert "newer edits would be lost" in result["skipped"][0]["reason"]
    assert "A later human edit." in alpha.read_text(encoding="utf-8")


def test_re_applying_the_same_rule_replaces_its_block_rather_than_duplicating_it(
    config, tmp_path,
):
    """Idempotency comes from ``relearn_apply``'s marker format, which this
    lifecycle reuses rather than re-implementing — a second renderer here would
    emit blocks the existing Revert path cannot find."""
    alpha = _project(tmp_path, "alpha")
    stage_rule(config, _rule(alpha))
    apply_staged(config, "cost:subagent", go=True)
    first = alpha.read_text(encoding="utf-8")

    stage_rule(config, _rule(alpha))
    apply_staged(config, "cost:subagent", go=True)
    second = alpha.read_text(encoding="utf-8")

    assert second.count("<!-- tokenjam:relearn:cost:subagent -->") == 1
    assert second == first


def test_the_undo_surface_explains_why_a_row_cannot_be_undone(config, tmp_path):
    alpha = _project(tmp_path, "alpha")
    stage_rule(config, _rule(alpha))
    apply_staged(config, "cost:subagent", go=True)
    assert [row["undoable"] for row in store.list_backups(config)] == [True]

    alpha.write_text("hand-edited\n", encoding="utf-8")
    row = store.list_backups(config)[0]
    assert row["undoable"] is False
    assert row["reason"] == "changed since apply — undo would lose newer edits"


# --- delivery as a seam -----------------------------------------------------#
#
# Appending markdown to a CLAUDE.md is ONE way to get a rule in front of an
# agent, not the only one, and for several of these analyzers not the best one:
# a hook can deliver the same guidance at the moment of the decision rather
# than at the top of a long context. These pin that the lifecycle is not welded
# to markdown, so adding a mechanism is a new registration rather than a
# rewrite of this surface.

def test_a_rule_carries_and_persists_its_delivery_mechanism(config, tmp_path):
    """The staged entry records the mechanism that rendered it, so apply
    re-renders through the SAME one that produced the diff a reviewer approved
    rather than through whatever the default happens to be by then."""
    from tokenjam.core.rulewrite.delivery import DELIVERY_CLAUDE_MD_RULE

    alpha = _project(tmp_path, "alpha")
    staged = stage_rule(config, _rule(alpha))
    assert staged[0].delivery == DELIVERY_CLAUDE_MD_RULE
    # And it survives the store round trip, not just the in-memory object.
    assert store.read_staged(
        config, "cost:subagent", str(alpha),
    ).delivery == DELIVERY_CLAUDE_MD_RULE
    assert check_staged(config)[0]["delivery"] == DELIVERY_CLAUDE_MD_RULE


def test_an_unknown_delivery_is_refused_rather_than_rendered_as_markdown(
    config, tmp_path,
):
    """A staged entry naming a mechanism this build cannot render must not be
    quietly written as a markdown block into whatever file it named. A wrong
    write to a real file is the one failure this package exists to prevent, so
    the unknown case refuses instead of falling back."""
    alpha = _project(tmp_path, "alpha")
    before = alpha.read_text(encoding="utf-8")
    with pytest.raises(RuleWriteRefused) as exc:
        stage_rule(config, _rule(alpha, delivery="prompt_injection_hook"))
    assert "unknown delivery mechanism" in str(exc.value)
    assert alpha.read_text(encoding="utf-8") == before


def test_the_lifecycle_prices_through_the_mechanism(config, tmp_path):
    """The load-bearing half of the seam.

    "A hook is executed, never sent as prompt text" is true of an EXECUTING
    hook and false of a context-INJECTING one: a `UserPromptSubmit`
    re-injection is prompt text on a different schedule, and once injected it
    accretes into history and is re-sent every turn. So the question "does this
    cost tokens" is asked of the MECHANISM, which is the only thing that knows.
    """
    from dataclasses import replace as dc_replace

    from tokenjam.core.rulewrite import delivery as dv

    alpha = _project(tmp_path, "alpha")
    markdown = dv.DELIVERY_KINDS[dv.DELIVERY_CLAUDE_MD_RULE]
    assert markdown.carries_prompt_text is True

    # A hook mechanism that DOES put text in front of the model still pays —
    # and it prices ITSELF rather than inheriting another kind's answer. That is
    # the whole reason the pricer is per-mechanism: two mechanisms that look
    # alike from outside can have opposite cost behaviour.
    injecting = dc_replace(
        markdown, name="test_injecting_hook",
        standing_tokens=lambda rule, rendered, existing: 250,
    )
    free = dc_replace(markdown, name="test_executing_hook", carries_prompt_text=False)
    dv.DELIVERY_KINDS[injecting.name] = injecting
    dv.DELIVERY_KINDS[free.name] = free
    try:
        priced = stage_rule(config, _rule(alpha, delivery=injecting.name))[0]
        store.clear(config)
        unpriced = stage_rule(config, _rule(alpha, delivery=free.name))[0]
    finally:
        del dv.DELIVERY_KINDS[injecting.name]
        del dv.DELIVERY_KINDS[free.name]

    # Same shape, same text, opposite standing cost — decided by the mechanism.
    assert priced.standing_tokens_per_session == 250
    assert unpriced.standing_tokens_per_session == 0


def test_adding_a_mechanism_does_not_require_touching_the_lifecycle():
    """The seam's actual contract, asserted structurally: nothing in the
    staging / apply / store / plan machinery names the markdown mechanism."""
    from pathlib import Path as _Path

    import tokenjam.core.rulewrite.apply as apply_mod
    import tokenjam.core.rulewrite.plan as plan_mod
    import tokenjam.core.rulewrite.store as store_mod

    for module in (apply_mod, store_mod, plan_mod):
        source = _Path(module.__file__).read_text(encoding="utf-8")
        assert "render_note_content" not in source, module.__name__
        assert "render_skill_content" not in source, module.__name__
