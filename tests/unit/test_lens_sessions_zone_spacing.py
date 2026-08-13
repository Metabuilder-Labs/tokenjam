"""A separator only exists when there are two things to separate.

The SDK zone's ``section-gap`` was written to sit above an "SDK services"
zone-head that has since been removed as a duplicate heading. The spacer kept
its place, so viewing as SDK -- where the coding zone renders nothing at all --
it opened a 28px band of dead space between the page header and the tab row.

Deleting it outright would be wrong in the other direction: with both zones on
screen it is the only thing between the archived table and the tab strip. So it
is conditional on the zone above having actually rendered, and this pins that
condition rather than the presence or absence of the element.
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
def status_view(html: str) -> str:
    start = html.index("function StatusView(")
    nxt = re.search(r"\n(?:function|const|class) ", html[start + 10:])
    assert nxt, "no top-level declaration follows StatusView; update this extractor"
    return html[start:start + 10 + nxt.start()]


def test_the_sdk_separator_is_conditional_on_the_zone_above_it(status_view: str):
    # Anchored on the LAST `showSdkZone` gate before the tab bar: an earlier one
    # opens the coding zone's own head, and slicing from that would sweep in the
    # archive's separator, which is a different (and correct) spacer.
    tab_bar = status_view.index('<div class="tab-bar" role="tablist">')
    sdk = status_view[status_view.rindex("${showSdkZone ? html`", 0, tab_bar):tab_bar]
    assert '<div class="section-gap"></div>' not in sdk.replace(
        '${codingZoneShown ? html`<div class="section-gap"></div>` : null}', ""
    ), "the spacer above the tab bar must not render unconditionally"
    assert '${codingZoneShown ? html`<div class="section-gap"></div>` : null}' in sdk


def test_showing_the_zone_and_the_zone_having_content_are_different_questions(status_view: str):
    """`showCodingZone` is persona permission; `codingZoneShown` is what renders.

    Conflating the two is what left the separator standing on its own: the
    persona permitted a zone that had nothing to put on the page.
    """
    assert (
        "const codingZoneShown = showCodingZone && "
        "!!(codingAgents.length || codingArchived.length);"
    ) in status_view
    # And every consumer reads the derived answer, so the three cannot disagree.
    assert "showCodingZone && (codingAgents.length || codingArchived.length)" not in status_view, (
        "the inlined form must not survive alongside the hoisted one"
    )
    assert status_view.count("codingZoneShown") >= 4


def test_the_coding_zone_head_is_still_gated_on_both_zones_being_present(status_view: str):
    """The "Coding sessions" heading exists to tell two zones apart.

    Under the SDK persona the coding zone does not render at all, so the heading
    cannot appear there; under a coding persona there is no second zone to
    distinguish it from, so it is not needed. Both facts follow from it being
    nested inside `codingZoneShown` and gated on `showSdkZone`.
    """
    zone = status_view[status_view.index("${codingZoneShown ? html`"):]
    head = zone[: zone.index("Coding sessions")]
    assert "${showSdkZone ? html`" in head, (
        "the zone label must only render when the SDK zone is also on screen"
    )
