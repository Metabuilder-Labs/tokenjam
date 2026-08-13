"""A rule already in the user's own files is DONE, not too expensive to write.

THE INVERSION THIS FIXES. ``rulewrite/plan`` resolved "already dealt with" from
tokenjam's own apply ledgers and nothing else, so a rule the user (or their own
harness) wrote by hand was invisible to it. The analyzer measured the recurrence,
found the guidance already present, and reported the rule as *blocked because your
instruction files already carry more standing context than the budget allows to
grow* — i.e. as too expensive, when the reason those files are large is that they
already contain the fix. The user read a refusal where the honest answer is a
checkmark.

Two mechanisms are pinned here:

* ``rulewrite/presence`` — asks the user's local ``claude`` (stubbed throughout;
  no test may spend a real subscription), caches on file content, and resolves
  every failure to "not present" so a rule stays on offer rather than silently
  vanishing.
* ``rulewrite/plan`` — the economics gate moved BELOW the rule surface, so a rule
  the write budget declined is absent rather than explained, while anything
  applied or already present survives that filter.
"""
from __future__ import annotations

import json

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.rulewrite import presence
from tokenjam.core.rulewrite.plan import _is_listable, _mark_present
from tokenjam.core.rulewrite.types import RuleDestination, RuleWrite


@pytest.fixture
def cfg(tmp_path) -> TjConfig:
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _rule(**kw) -> RuleWrite:
    base = dict(
        signature="relearn:cwd_confusion",
        analyzer="relearn",
        title="cwd / relative-path confusion",
        artifact_text="Always resolve paths from the repo root, never the cwd.",
        past_overspend_usd=113.66,
        past_overspend_tokens=1_000,
    )
    base.update(kw)
    return RuleWrite(**base)


# --------------------------------------------------------------------------- #
# The model call: batching, caching, and the safe direction
# --------------------------------------------------------------------------- #
def test_one_call_per_file_not_per_rule(cfg, tmp_path, monkeypatch):
    """Cost control is structural, not advisory. Three rules bound for one file
    is ONE question, or a corpus with dozens of clusters becomes dozens of
    subprocess launches against the user's subscription."""
    target = tmp_path / "CLAUDE.md"
    target.write_text("# rules\nResolve paths from the repo root.\n", encoding="utf-8")
    dest = (RuleDestination(path=str(target), scope="project", sessions=3),)
    rules = [
        _rule(signature=f"relearn:r{i}", artifact_text=f"rule {i}", destinations=dest)
        for i in range(3)
    ]

    calls = []

    def _fake(prompt, *, model, timeout):
        calls.append(prompt)
        return "1: PRESENT — Resolve paths from the repo root\n2: ABSENT\n3: ABSENT"

    import tokenjam.core.distill as distill

    monkeypatch.setattr(distill, "invoke_claude", _fake)

    out = presence.detect_presence(cfg, rules)
    assert len(calls) == 1, "one call per destination file, not one per rule"
    # All three candidates appear in the single prompt.
    for i in range(3):
        assert f"rule {i}" in calls[0]
    assert set(out) == {"relearn:r0"}


def test_a_cached_verdict_is_not_re_asked(cfg, tmp_path, monkeypatch):
    target = tmp_path / "CLAUDE.md"
    target.write_text("Resolve paths from the repo root.\n", encoding="utf-8")
    rule = _rule(destinations=(RuleDestination(path=str(target), scope="project"),))

    calls = []
    import tokenjam.core.distill as distill

    def _fake(prompt, *, model, timeout):
        calls.append(prompt)
        return "1: PRESENT — repo root"

    monkeypatch.setattr(distill, "invoke_claude", _fake)
    presence.detect_presence(cfg, [rule])
    presence.detect_presence(cfg, [rule])
    assert len(calls) == 1, "an unchanged file must not be re-asked"


def test_editing_the_file_invalidates_the_verdict(cfg, tmp_path, monkeypatch):
    """The answer can flip when the file changes — that is the whole point of a
    presence check, so the cache key has to include the file's content."""
    target = tmp_path / "CLAUDE.md"
    target.write_text("Resolve paths from the repo root.\n", encoding="utf-8")
    rule = _rule(destinations=(RuleDestination(path=str(target), scope="project"),))

    import tokenjam.core.distill as distill

    calls = []

    def _fake(prompt, *, model, timeout):
        calls.append(prompt)
        return "1: PRESENT — repo root"

    monkeypatch.setattr(distill, "invoke_claude", _fake)
    presence.detect_presence(cfg, [rule])
    target.write_text("something else entirely\n", encoding="utf-8")
    presence.detect_presence(cfg, [rule])
    assert len(calls) == 2


def test_changing_the_rule_text_invalidates_the_verdict(cfg, tmp_path, monkeypatch):
    target = tmp_path / "CLAUDE.md"
    target.write_text("Resolve paths from the repo root.\n", encoding="utf-8")
    dest = (RuleDestination(path=str(target), scope="project"),)

    import tokenjam.core.distill as distill

    calls = []
    monkeypatch.setattr(
        distill, "invoke_claude",
        lambda p, *, model, timeout: (calls.append(p), "1: ABSENT")[1],
    )
    presence.detect_presence(cfg, [_rule(destinations=dest, artifact_text="v1")])
    presence.detect_presence(cfg, [_rule(destinations=dest, artifact_text="v2")])
    assert len(calls) == 2, "a different rule is a different question"


@pytest.mark.parametrize("answer", [None, "", "I think maybe?", "PRESENT", "7: PRESENT"])
def test_every_unusable_answer_resolves_to_not_present(cfg, tmp_path, monkeypatch, answer):
    """THE SAFE DIRECTION. No CLI, a timeout, prose instead of the asked shape, an
    out-of-range index — all of it leaves the rule on offer. That can waste the
    user's attention on a rule they already have; the opposite hides a fix they
    never made, and a missing rule is invisible in a way a duplicate is not."""
    target = tmp_path / "CLAUDE.md"
    target.write_text("Resolve paths from the repo root.\n", encoding="utf-8")
    rule = _rule(destinations=(RuleDestination(path=str(target), scope="project"),))

    import tokenjam.core.distill as distill

    monkeypatch.setattr(distill, "invoke_claude", lambda p, *, model, timeout: answer)
    assert presence.detect_presence(cfg, [rule]) == {}
    assert presence.load_presence(cfg) == {}


def test_an_unreadable_file_is_not_recorded_as_absent(cfg, tmp_path, monkeypatch):
    """It is re-asked once the file is readable, rather than pinned to a verdict
    reached without ever seeing the file."""
    rule = _rule(destinations=(
        RuleDestination(path=str(tmp_path / "nope" / "CLAUDE.md"), scope="project"),
    ))
    import tokenjam.core.distill as distill

    called = []
    monkeypatch.setattr(
        distill, "invoke_claude",
        lambda p, *, model, timeout: (called.append(p), "1: PRESENT")[1],
    )
    assert presence.detect_presence(cfg, [rule]) == {}
    assert not called, "nothing to ask about when the file cannot be read"


def test_a_corrupt_store_reads_as_nothing_known(cfg):
    p = presence.presence_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert presence.load_presence(cfg) == {}


def test_load_presence_never_returns_an_absent_verdict(cfg):
    """"We asked and it is not there" must not be mistakable for a positive."""
    p = presence.presence_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "a": {"present": True, "key": "k", "source_path": "/x"},
        "b": {"present": False, "key": "k", "source_path": "/x"},
    }), encoding="utf-8")
    assert set(presence.load_presence(cfg)) == {"a"}


def test_the_prompt_pushes_toward_absent_when_unsure(cfg, tmp_path, monkeypatch):
    """The asymmetry is stated IN the prompt, not just relied on downstream: a
    false "present" hides a fix, so an unsure model must answer absent."""
    target = tmp_path / "CLAUDE.md"
    target.write_text("x\n", encoding="utf-8")
    rule = _rule(destinations=(RuleDestination(path=str(target), scope="project"),))
    import tokenjam.core.distill as distill

    seen = {}
    monkeypatch.setattr(
        distill, "invoke_claude",
        lambda p, *, model, timeout: (seen.setdefault("p", p), "1: ABSENT")[1],
    )
    presence.detect_presence(cfg, [rule])
    assert "unsure" in seen["p"].lower()
    assert "ABSENT" in seen["p"]


# --------------------------------------------------------------------------- #
# plan: presence withdraws the OFFER and keeps the FIGURE
# --------------------------------------------------------------------------- #
def test_presence_withdraws_the_offer_and_keeps_the_figure(cfg):
    """Critical Rule 32. The recurrence happened and cost what it cost; the
    guidance having been there the whole time does not un-spend it. If anything a
    rule that was present AND still recurring says the wording is not working,
    which is a reason to keep the figure visible rather than erase it."""
    p = presence.presence_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"relearn:cwd_confusion": {
        "present": True, "key": "k", "source_path": "/home/u/CLAUDE.md",
        "evidence": "Resolve paths from the repo root",
    }}), encoding="utf-8")

    out = _mark_present([_rule()], cfg)
    assert out[0].already_present is True
    assert out[0].offered is False
    assert out[0].presence_path == "/home/u/CLAUDE.md"
    assert out[0].presence_evidence == "Resolve paths from the repo root"
    assert out[0].past_overspend_usd == 113.66
    assert out[0].past_overspend_tokens == 1_000


def test_the_ledger_wins_over_the_models_verdict(cfg):
    """An apply ledger is exact; presence is an inference. Re-labelling a rule
    tokenjam itself applied from a model's answer would replace a fact with a
    guess."""
    p = presence.presence_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"relearn:cwd_confusion": {
        "present": True, "key": "k", "source_path": "/x",
    }}), encoding="utf-8")
    out = _mark_present([_rule(already_applied=True)], cfg)
    assert out[0].already_applied is True
    assert out[0].already_present is False


def test_a_dismissal_is_not_overwritten_by_presence(cfg):
    p = presence.presence_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"relearn:cwd_confusion": {
        "present": True, "key": "k", "source_path": "/x",
    }}), encoding="utf-8")
    out = _mark_present([_rule(dismissed=True)], cfg)
    assert out[0].dismissed is True
    assert out[0].already_present is False


# --------------------------------------------------------------------------- #
# plan: the economics gate moved below the surface
# --------------------------------------------------------------------------- #
def test_a_rule_the_budget_declined_does_not_reach_the_surface():
    """The founder call: sub-floor and net-negative proposals should not reach the
    rules surface at all, rather than reach it and explain themselves. With them
    gone there is no refusal prose left to render."""
    assert _is_listable(_rule(offered=False, blocked_reason="below the $5 floor")) is False


def test_an_actionable_rule_is_listable():
    assert _is_listable(_rule(offered=True)) is True


@pytest.mark.parametrize("state", ["already_applied", "already_present", "dismissed"])
def test_state_survives_the_gate_even_when_the_budget_declined(state):
    """THE ORDERING THAT MATTERS. A rule already in place is very often one the
    budget also declined — because the files holding it are what saturate the
    budget. If the gate ran before state resolution, exactly the population this
    work exists to surface would be filtered out of its own tab.

    Dismissed is here for a different reason: drop it and there is nothing left to
    un-dismiss.
    """
    assert _is_listable(_rule(offered=False, **{state: True})) is True


# --------------------------------------------------------------------------- #
# The rendered surface: two tabs, and no refusal prose left
# --------------------------------------------------------------------------- #
_UI = None


def _ui() -> str:
    global _UI
    if _UI is None:
        from pathlib import Path

        _UI = (Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html").read_text(
            encoding="utf-8",
        )
    return _UI


def _rules_page() -> str:
    src = _ui()
    start = src.index("function RulesView(")
    return src[start:src.index("\nfunction ", start + 10)]


def test_the_rules_page_has_pending_and_in_place_tabs():
    """Two tabs, one per population: something to do, or something already done.

    The LABELS moved (Open/Applied -> Pending/In place) when this page was
    brought onto the lens design; the STATE KEYS did not, which is why the
    `setTab` assertions are the ones that read unchanged. Both spellings are
    pinned — the new one present, the old one absent — so a revert to the old
    wording is a failure rather than a silent drift back.
    """
    page = _rules_page()
    assert "Pending " in page
    assert "In place " in page
    assert "Open " not in page
    assert "Applied (" not in page
    assert "setTab('open')" in page and "setTab('applied')" in page


def test_the_not_on_offer_line_is_gone():
    """The page used to render "Not on offer: <reason>" per row. The gate moved
    below the surface, so there is nothing left to refuse — and a page of
    refusals was the thing the founder asked to purge.

    Pinned on the RENDER, not on the prose: the comment explaining the removal
    necessarily quotes what was removed, and a bare substring search would fail on
    the explanation of its own fix. The interpolation is the thing that could
    actually reach a user.
    """
    page = _rules_page()
    assert "Not on offer: ${r.blocked_reason}" not in page
    assert "color:var(--warn)\">Not on offer" not in page


def test_neither_tab_count_can_render_before_its_fetch_resolves():
    """A "(0)" painted mid-flight reads as a settled "you have nothing" — the
    worst thing this page can say wrongly (root anti-pattern 22). Both counts sit
    behind `rules === null`."""
    page = _rules_page()
    for label in ("Pending ", "In place "):
        idx = page.index(label)
        window = page[idx:idx + 400]
        assert "rules === null" in window, f"{label} count is not gated on its fetch"
        assert "shimmer" in window


def test_the_applied_tab_shows_which_route_put_a_rule_there():
    """Two ways to be done, and they are not the same claim: tokenjam wrote it
    (a ledger knows, exactly) versus the user already had it (a model read their
    file). The second carries its evidence, because a reader who did not make
    that inference needs the file's own words to check it."""
    page = _rules_page()
    assert "Applied by tokenjam" in page
    assert "Already in your instruction files" in page
    assert "presence_evidence" in page


def test_the_surface_is_labelled_relearn_everywhere_the_user_reads_it():
    """One name for one thing.

    The screen used to be called "Relearn" in one place and "Rules" in another,
    which made the Dashboard tile unfindable from the nav. The label is now
    "Relearn" at every site a user reads; the ROUTE (`#/optimize/rules`), the
    registry KEY (`relearn`) and the CLI group (`tj rules`) are deliberately
    unchanged. Each site pins the new spelling present AND the old one absent,
    so a drift back is a failure rather than a silent revert.
    """
    src = _ui()
    for site, present, absent in (
        ("sidebar nav-child", "&#8627;</span> Relearn</a>", "&#8627;</span> Rules</a>"),
        ("dashboard action tile", "relearn:    { title: 'Relearn'", "relearn:    { title: 'Rules'"),
        ("optimize card", '<div class="opt-card-title">Relearn</div>',
         '<div class="opt-card-title">Rules</div>'),
        ("optimize card link", 'href="#/optimize/rules">Open Relearn',
         'href="#/optimize/rules">Open Rules'),
        ("persona-empty header title", "rules: 'Relearn'", "rules: 'Rules'"),
        ("page title", '<div class="page-title" style="margin-bottom:0">Relearn</div>',
         '<div class="page-title" style="margin-bottom:0">Rules</div>'),
        ("faq entry", "title: 'Relearn',", "title: 'Rules',"),
    ):
        assert present in src, f"{site} must read Relearn"
        assert absent not in src, f"{site} still reads Rules"
    # The label moved; the command did not. `tj rules` is the group that drives
    # this surface (list / stage / apply / undo / dismiss, one subcommand per
    # button); `tj relearn` only emits an eval-case artifact and reaches none of
    # it, so renaming the chip to match the page would name a command that
    # cannot do what the page does.
    assert '<span class="opt-cli-chip">tj rules</span>' in src
    # Route, key and nav wiring are untouched by the rename.
    assert 'href="#/optimize/rules"' in src
    assert 'data-view="optimize" data-param="rules"' in src


def test_each_tab_owns_the_stat_tiles_that_describe_it():
    """Tiles caption the list directly beneath them, or they lie.

    A single strip in the page head, pinned to the pending population, sat above
    a tab reading "In place (2)" — two real rules with real figures — and
    captioned them 0 / 0 / 0, which reads as a broken page. Each tab now renders
    its own strip inside its own branch, in its own vocabulary, and a tab that is
    known-empty renders no strip at all: "0 rules waiting to be written" is a
    true measurement that still reads as a broken tile, and the empty-state
    sentence is the only honest home for that claim (root anti-pattern 22a).
    """
    page = _rules_page()
    # Anchored on the RENDER branch (`? html`), not on the bare condition —
    # the same expression also drives each tab button's active class.
    open_at = page.index("${tab === 'open' ? html")
    applied_at = page.index("${tab === 'applied' ? html")
    head, open_branch, applied_branch = (
        page[:open_at], page[open_at:applied_at], page[applied_at:],
    )
    # No strip in the head any more — that was the mis-captioning.
    assert "opt-stats" not in head
    # Each tab's vocabulary is confined to its own branch.
    assert "'rules waiting to be written'" in open_branch
    assert "'rules waiting to be written'" not in applied_branch
    assert "'rules in place'" in applied_branch
    assert "'rules in place'" not in open_branch
    # A known-empty tab shows no numbers; not-yet-known still shows a skeleton.
    assert "rules === null || openRules.length > 0" in open_branch
    assert "rules === null || inPlaceRules.length > 0" in applied_branch
    # The staged tile keeps its OWN fetch's gate rather than borrowing /rules'.
    assert "staged === null ? null : staged.length" in open_branch
    # The two in-place route tiles are a split of the population tile, derived as
    # a complement so they cannot fail to add up (root anti-pattern 22b).
    assert "inPlaceRules.length - inPlaceWritten" in page
