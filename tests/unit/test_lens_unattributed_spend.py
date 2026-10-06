"""Lens must disclose spend that is outside the named-session cards."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.fixture(scope="module")
def status_view() -> str:
    html = _UI.read_text(encoding="utf-8")
    start = html.index("function StatusView(")
    nxt = re.search(r"\n(?:function|const|class) ", html[start + 10:])
    assert nxt, "no top-level declaration follows StatusView; update this extractor"
    return html[start:start + 10 + nxt.start()]


def test_status_view_explains_unattributed_spend(status_view: str):
    """A nonzero bucket needs an amount, reason, and concrete next action."""
    assert "const unattributedSpend = data.unattributed_spend;" in status_view
    assert "${unattributedSpend ? html`" in status_view
    assert "Unattributed spend" in status_view
    assert "Not assigned to a named session" in status_view
    assert "grand total can exceed the sum of named sessions" in status_view
    assert "fmtFramedDollar(unattributedSpend.cost_usd, data.framing)" in status_view
    assert "tj cost --group-by session" in status_view
    assert '<span class="range">tj cost --group-by session</span>' in status_view
    assert '<a class="range" href="#/cost?group_by=session">' not in status_view


def test_status_view_does_not_hide_the_disclosure_in_the_empty_state(status_view: str):
    """Sessionless telemetry has no agent card, but still needs to remain visible."""
    assert "!sdkServices.length && !unattributedSpend" in status_view
