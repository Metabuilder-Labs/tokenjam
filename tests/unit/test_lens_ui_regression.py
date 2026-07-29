"""Static-grep regression guards for the Lens web-UI polish batch (#654–#657).

There is no JS test runner in the Python CI ``test`` job, so behaviour that is
pure state logic is guarded by extracting the relevant pure function and running
it under node (see ``test_lens_select_all_behaviour.py`` /
``test_lens_dashboard_states.py``). The fixes in this module are copy/markup and
default-value changes with no branching logic of their own, so a source
assertion is the right guard: it fails if a rewrite silently reverts the
behaviour these issues fixed.

Each assertion is anchored on the specific string the fix introduced (or the
buggy string it removed), not on incidental wording, so harmless copy tweaks
around it do not break the test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _UI.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# #654 — Dashboard is a persistent top-level item + the default landing route.
# --------------------------------------------------------------------------- #
def test_dashboard_nav_link_is_persistent_across_lenses(html: str) -> None:
    # The Dashboard link must carry data-lens="all" so the improve/observe hide
    # rules never remove it, and must NOT be scoped to the improve lens anymore.
    assert (
        '<a href="#/dashboard" class="nav-link" data-view="dashboard" data-lens="all">'
        in html
    ), "Dashboard nav link must be data-lens=\"all\" (persistent in both lenses)"
    assert (
        '<a href="#/dashboard" class="nav-link" data-view="dashboard" data-lens="improve">'
        not in html
    ), "Dashboard must no longer be improve-only"


def test_dashboard_link_sits_above_the_lens_switch(html: str) -> None:
    dash = html.index('data-view="dashboard" data-lens="all"')
    switch = html.index('<div class="lens-switch"')
    assert dash < switch, "Dashboard nav link must be ABOVE the Improve/Observe toggle"


def test_persistent_lens_items_have_a_style_rule(html: str) -> None:
    assert '.sidebar a.nav-link[data-lens="all"]' in html


def test_empty_hash_default_route_is_dashboard(html: str) -> None:
    # getRoute()'s default (both the path fallback and the parts[0] fallback)
    # must resolve to dashboard, not review — with no render-time hash redirect.
    assert ": raw) || 'dashboard';" in html
    assert "parts[0] || 'dashboard'" in html
    assert "|| 'review';" not in html, "empty-hash default must be dashboard, not review"


def test_default_route_normalization_agrees_with_getroute(html: str) -> None:
    # Greptile P1-1: the URL-normalization replaceState on an empty hash must
    # write #/dashboard, matching getRoute()'s default. Writing the old #/review
    # here made the page (Dashboard) and address bar disagree, so a refresh or
    # shared bare link opened Review instead.
    assert "history.replaceState(null, '', '#/dashboard');" in html
    assert (
        "history.replaceState(null, '', '#/review');" not in html
    ), "empty-hash normalization must write #/dashboard, not #/review"


def test_dashboard_is_lens_neutral_and_preserves_active_lens(html: str) -> None:
    # Greptile P1-3: Dashboard is data-lens="all", so opening it from Observe
    # must keep the user in Observe. It must therefore be ABSENT from VIEW_LENS
    # (a mapped lens would force a switch), and the route-sync effect must fall
    # back to the sidebar's CURRENT lens for an unmapped view, not to 'improve'.
    assert (
        "dashboard: 'improve'" not in html
    ), "Dashboard must not be classified as improve in VIEW_LENS (it is lens-neutral)"
    assert (
        "const lens = VIEW_LENS[view] || (sidebar && sidebar.dataset.lens) || 'improve';"
        in html
    ), "unmapped (lens-neutral) views must preserve the active lens, not reset to improve"


# --------------------------------------------------------------------------- #
# #655 — one shared 30d window default across window-driven screens.
# --------------------------------------------------------------------------- #
def test_shared_default_since_constant_is_30d(html: str) -> None:
    assert "const DEFAULT_SINCE = '30d';" in html


def test_window_driven_screens_use_the_shared_default(html: str) -> None:
    # Traces (was 24h) and Cost (was 7d) now read the shared constant, and no
    # window-driven DEFAULTS object hardcodes a divergent since anymore.
    assert "const DEFAULTS = { since: DEFAULT_SINCE, result:" in html  # Traces
    assert "const DEFAULTS = { since: DEFAULT_SINCE, group_by: 'total'" in html  # Cost
    assert "const DEFAULTS = { since: DEFAULT_SINCE, agent_id: '', compare:" in html  # Optimize
    assert "const DEFAULTS = { since: DEFAULT_SINCE, metric: 'spend'" in html  # Analytics
    assert "const DEFAULTS = { since: '24h', result:" not in html
    assert "const DEFAULTS = { since: '7d', group_by:" not in html


def test_drill_through_hrefs_omit_since_at_the_shared_default(html: str) -> None:
    # Greptile P1-2: drill-through href builders must decide "omit since" against
    # the shared DEFAULT_SINCE, not a hardcoded literal. With the default moved
    # to 30d, tracesHrefForWindow's old `!== '24h'` dropped a real 24h window and
    # silently reset Traces to 30d. Both the Traces and Optimize href builders
    # now compare to DEFAULT_SINCE.
    assert (
        "if (since && since !== DEFAULT_SINCE) sp.set('since', since);" in html
    ), "drill-through builders must omit since only at DEFAULT_SINCE"
    assert (
        "if (since && since !== '24h') sp.set('since', since);" not in html
    ), "tracesHrefForWindow must not hardcode the retired 24h default"
    assert (
        "if (since && since !== '30d') sp.set('since', since);" not in html
    ), "href builders must read DEFAULT_SINCE, not a hardcoded 30d literal"


# --------------------------------------------------------------------------- #
# #656 — Review inbox appliable-count clarity + honest Apply wording.
# --------------------------------------------------------------------------- #
def test_inbox_reports_auto_appliable_count(html: str) -> None:
    assert "const autoApplyCount = bulkRelearn.length;" in html
    assert "const showBulkSelect = autoApplyCount > 1;" in html
    assert "can be applied automatically — the rest are copy-and-apply." in html


def test_bulk_select_only_shows_when_more_than_one_appliable(html: str) -> None:
    # Both the select-all label and the bulk action bar gate on showBulkSelect,
    # not on the old "> 0" which showed a bulk control for a single row.
    assert "${showBulkSelect ? html`" in html
    assert "${bulkRelearn.length > 0 ? html`" not in html


def test_apply_wording_says_next_run_not_enforcement(html: str) -> None:
    # The Approve note and the per-kind hints must make clear the write is
    # effective on the next run and is NOT live enforcement (honesty Rule 14).
    assert "effective on the next run — not live enforcement" in html
    assert "takes effect on the next run, not live" in html
    # The old ambiguous single-word "Apply change" button must be gone.
    assert "button: 'Apply change'" not in html


# --------------------------------------------------------------------------- #
# #657 — Drift empty-state leads with a scannable headline.
# --------------------------------------------------------------------------- #
def test_drift_empty_state_has_scannable_headline(html: str) -> None:
    assert 'class="drift-empty-lead"' in html
    assert ".drift-empty-lead {" in html
    assert (
        "Drift needs live SDK agents with 10+ completed sessions" in html
    ), "Drift empty-state must lead with the one-line headline"
