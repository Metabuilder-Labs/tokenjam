"""Static-grep regression guards for Lens UI bug fixes.

The Python `test` job can't run a JS runner, so UI fixes are guarded here by
asserting the buggy pattern is *gone* and the fix's helpers/markers are present
(the same approach documented in CLAUDE.md → Web UI → "Testing the UI").

Currently guards:

- #653 — the trace-detail pane hung forever on large traces because
  ``GET /api/v1/traces/{id}`` shipped every span with its full ``attributes``
  dict (~1 GB for a 45k-span trace). The fix makes the waterfall payload
  attribute-free + capped, fetches a single span's attributes lazily on expand,
  and hardens ``TraceDetailView`` with a fetch timeout + error state so the
  skeleton always clears.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _UI.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# #653 — large-trace detail must not hang; payload is capped + lazy-attrs
# --------------------------------------------------------------------------- #
def test_trace_detail_has_load_error_state(html: str) -> None:
    """The skeleton must be able to clear into an error state, never spin forever."""
    assert "loadState" in html
    # An explicit error branch with a retry affordance.
    assert "loadState === 'error'" in html
    assert "Retry" in html


def test_trace_detail_has_fetch_timeout(html: str) -> None:
    """The trace-detail fetch is raced against a timeout so it can't hang."""
    assert "Promise.race" in html
    assert "TIMEOUT_MS" in html


def test_trace_detail_handles_truncation(html: str) -> None:
    """A capped large trace must disclose 'showing N of M spans' (no silent drop)."""
    assert "truncated" in html
    assert "Showing " in html and "of " in html and "spans" in html


def test_trace_detail_fetches_attributes_lazily(html: str) -> None:
    """Captured content is fetched per-span on expand, not shipped for all spans."""
    # The lazy per-span endpoint is called from the detail view.
    assert "/spans/" in html
    assert "selAttrs" in html
    # The old bug: rendering sel.attributes straight from the waterfall payload.
    assert "JSON.stringify(sel.attributes" not in html


def test_trace_detail_caps_rendered_rows(html: str) -> None:
    """Thousands of DOM rows freeze the tab; the render is capped + disclosed."""
    assert "RENDER_ROW_CAP" in html
