"""``PERSONA_HIDDEN_VIEWS`` declares; CSS enforces. Neither implies the other.

The regression this exists to make impossible: Spend was hidden for both
personas by a pair of ``a.nav-link[data-lens="observe"]`` rules. Removing the
dead Improve/Observe lens deleted those rules and left both ``cost`` entries
sitting in ``PERSONA_HIDDEN_VIEWS``, so the declaration read as intact while the
enforcement was gone. The nav link rendered, ``personaHidesRoute`` still gated
the route, and a reader who clicked Spend got a blank page.

A per-rule assertion ("the cost rule exists") would not have caught it, because
the failure was a rule DISAPPEARING, not a wrong rule. So this pins the mapping
as a whole, in both directions:

  every (persona, view) entry  ->  a matching CSS rule
  every matching CSS rule      ->  an entry that asks for it

The reverse direction matters as much as the forward one: a rule that outlives
its list entry hides a view nothing declares hidden, which is just as invisible
and reads to the next person as an intentional gate.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _UI.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def declared(html: str) -> dict[str, list[str]]:
    """``PERSONA_HIDDEN_VIEWS``, parsed out of the served page."""
    start = html.index("const PERSONA_HIDDEN_VIEWS = {")
    body = html[start:html.index("};", start)]
    out: dict[str, list[str]] = {}
    for persona, views in re.findall(r"'([\w-]+)':\s*\[([^\]]*)\]", body):
        out[persona] = re.findall(r"'([\w-]+)'", views)
    assert out, "could not parse PERSONA_HIDDEN_VIEWS; update this extractor"
    return out


@pytest.fixture(scope="module")
def enforced(html: str) -> set[tuple[str, str]]:
    """Every persona-keyed nav-link hide rule actually present in the stylesheet."""
    pattern = re.compile(
        r'\.sidebar\[data-persona="([\w-]+)"\]\s+a\.nav-link\[data-view="([\w-]+)"\]'
        r'\s*\{[^}]*display:\s*none[^}]*\}'
    )
    return {(p, v) for p, v in pattern.findall(html)}


def test_the_lists_still_parse_as_expected(declared):
    # Guards the extractor itself: an empty parse would make every assertion
    # below vacuously true, which is the failure mode this module is about.
    assert set(declared) >= {"claude-code", "sdk"}
    assert all(declared[p] for p in declared), "a persona with an empty list is suspicious"


def test_every_declared_hidden_view_has_a_css_rule_enforcing_it(declared, enforced):
    missing = {
        (persona, view)
        for persona, views in declared.items()
        for view in views
        if (persona, view) not in enforced
    }
    assert not missing, (
        "PERSONA_HIDDEN_VIEWS declares these hidden but no CSS rule hides the "
        f"nav link, so the link renders and the gated route answers nothing: {sorted(missing)}"
    )


def test_no_css_rule_hides_a_view_nothing_declares_hidden(declared, enforced):
    orphans = {
        (persona, view)
        for persona, view in enforced
        if view not in declared.get(persona, [])
    }
    assert not orphans, (
        "these CSS rules hide a nav link that PERSONA_HIDDEN_VIEWS does not ask "
        f"to be hidden, so the route is reachable but unlinkable: {sorted(orphans)}"
    )


def test_spend_is_hidden_for_both_personas(enforced):
    """The specific casualty, pinned by name as well as by the mapping."""
    assert ("claude-code", "cost") in enforced
    assert ("sdk", "cost") in enforced


def test_the_hidden_everywhere_set_is_derived_not_transcribed(html: str, declared):
    """The third category, and it must follow the lists rather than copy them.

    A view hidden under EVERY persona key can never be made reachable by the
    persona read landing, so it is the one thing the nav may hide while the
    persona is still unknown. A hardcoded copy of that set would keep hiding a
    row after an entry was removed from one of the lists, which is the same
    declaration/enforcement drift this module exists to catch, one level up.
    """
    block = html[html.index("const PERSONA_HIDDEN_EVERYWHERE = ("):]
    block = block[: block.index("})();")]
    assert "PERSONA_HIDDEN_VIEWS" in block, "the set must be derived from the lists"
    # No view name may appear as a literal in the derivation.
    for views in declared.values():
        for view in views:
            assert f"'{view}'" not in block, (
                f"'{view}' is transcribed into the derivation; derive it instead"
            )


def test_only_the_hidden_everywhere_views_are_hidden_before_the_persona_lands(
    html: str, declared,
):
    """The rule that fires while `personaKnown` is false, checked against the lists.

    This is the anti-over-hiding property, kept: a view hidden under only SOME
    persona is still a question the read decides, so it must NOT be hidden here.
    Only the intersection may be.
    """
    everywhere = set.intersection(*(set(v) for v in declared.values()))
    union = set().union(*(set(v) for v in declared.values()))
    assert everywhere, "no view is hidden everywhere; this test's premise moved"
    assert union - everywhere, (
        "every hidden view is hidden everywhere, so the persona-dependent case "
        "is untested; this test's premise moved"
    )
    # The unknown-window hide is keyed on the derived flag, not on a view name.
    assert '.sidebar a.nav-link[data-persona-unreachable="1"]' in html
    assert "el.dataset.personaUnreachable = PERSONA_HIDDEN_EVERYWHERE.has(v)" in html
    # A persona-DEPENDENT view must never reach that flag: it gets the unsettled
    # treatment instead, so an unresolved read removes nothing.
    assert "el.classList.toggle('nav-link-pending', pending)" in html
    assert "const pending = !personaKnown && el.dataset.personaUnreachable !== '1'" in html
    assert "&& personaMayHide(v);" in html


def test_a_view_hidden_under_every_persona_records_why(html: str, declared):
    """Already the rule; asserted here because `cost` is the only such view."""
    everywhere = set.intersection(*(set(v) for v in declared.values()))
    block = html[html.index("const PERSONA_HIDDEN_DELIBERATE = {"):]
    block = block[: block.index("\n};")]
    for view in everywhere:
        assert (view + ":") in block, (
            f"'{view}' is hidden under every persona, so PERSONA_HIDDEN_DELIBERATE "
            "must record why, or the next reader reads it as a copy-paste mistake"
        )


def test_the_route_gate_and_the_css_gate_cover_the_same_views(html: str, declared):
    """`personaHides` reads the same list the rules mirror, so they cannot drift.

    The link and the route have to fire together: a hidden link over a live
    route is a page nobody can find, and a visible link over a gated route is
    the bug that just shipped.
    """
    fn = html[html.index("function personaHides(persona, view, known) {"):]
    fn = fn[: fn.index("\n}")]
    assert "PERSONA_HIDDEN_VIEWS[persona]" in fn, (
        "personaHides must read the same list the CSS mirrors, not a second copy"
    )
    # And the not-yet-known state stays "hide nothing" on both sides: the CSS
    # keys on [data-persona="..."], which syncNavState leaves empty until the
    # persona settles.
    assert "if (!known) return false;" in fn
    assert 'const personaAttr = personaKnown ? persona : \'\';' in html
