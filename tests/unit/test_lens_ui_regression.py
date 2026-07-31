"""Static-grep regression guards for Lens web-UI fixes.

There is no JS test runner in the Python CI ``test`` job, so UI behaviour is
guarded here by asserting the buggy pattern is gone and the fix's markers are
present (the approach documented in CLAUDE.md → Web UI → "Testing the UI"), or by
extracting a pure function and running it under node (see
``test_lens_select_all_behaviour.py`` / ``test_lens_dashboard_states.py``).

Each assertion is anchored on the specific string a fix introduced or removed,
not on incidental wording, so harmless copy tweaks around it don't break it.

Guards the polish batch (#654–#657) and the trace-detail fix (#653 plus its #659
follow-ups: opt-in light payload, lazy per-span attributes, capped/pinned rows).
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
    # The lens-neutral fallback expression is still present for any future
    # persona that keeps both lenses; both currently-known personas (claude-code
    # and sdk) force 'improve' instead (guarded by
    # test_persona_forces_improve_lens below).
    assert (
        "(VIEW_LENS[view] || (sidebar && sidebar.dataset.lens) || 'improve')" in html
    ), "the lens-neutral fallback expression must remain"


def test_persona_forces_improve_lens(html: str) -> None:
    # Neither the claude-code nor the sdk persona has Observe pages or a
    # lens-switch, so both must be FORCED to 'improve'. Deriving the lens from a
    # lens-neutral view (Dashboard) would preserve a stale 'observe' left by a
    # prior session, collapsing the sidebar to Dashboard-only until a reload —
    # the persona-toggle collapse bug.
    assert "const lens = (persona === 'claude-code' || persona === 'sdk')" in html
    assert "? 'improve'" in html


def test_trace_detail_is_exempt_from_persona_redirect(html: str) -> None:
    # A trace DETAIL (traces/<id>) reached by drilling into a session's Traces
    # tab must NOT be redirected to the Dashboard the way the hidden top-level
    # Traces LIST is. The persona-redirect guard exempts a traces view that
    # carries a trace-id param.
    assert "!(route.view === 'traces' && route.param)" in html


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
    assert "it takes effect on the next run, not as live enforcement" in html
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


# --------------------------------------------------------------------------- #
# #659 P1-1 — the Lens waterfall must request the OPT-IN light payload so the
# default (full-attributes) response is left intact for exports / the API shim.
# --------------------------------------------------------------------------- #
def test_waterfall_fetch_uses_light_payload_param(html: str) -> None:
    """The waterfall fetch passes ?attributes=false; the default full payload is
    reserved for complete-span consumers (ApiBackend.get_trace_spans)."""
    assert "'/traces/' + traceId + '?attributes=false'" in html


# --------------------------------------------------------------------------- #
# #659 P1-3 — costliest "jump" badges must never target a row hidden by the
# render cap. Beyond-cap costliest spans are pinned into the rendered set, and
# the badge gates on the rendered-row id set so no badge is a dead link.
# --------------------------------------------------------------------------- #
def test_jump_badges_only_target_rendered_rows(html: str) -> None:
    """A jump badge must only render when its target row is actually rendered."""
    # The rendered-row id set exists and the badge gates on it.
    assert "renderedRowIds" in html
    assert "!renderedRowIds.has(sid)" in html
    # Beyond-cap costliest spans are pinned into the rendered set.
    assert "pinnedRows" in html


# --------------------------------------------------------------------------- #
# SDK-persona UX batch — the persona-empty banner renders in the CONTENT region
# below each page's header (never above it), the "switch back" copy is a real
# control that calls the persona setter, and the SDK sidebar is a clean flat
# list with no Improve/Observe lens toggle.
# --------------------------------------------------------------------------- #
def test_persona_empty_gate_renders_banner_below_the_header(html: str) -> None:
    """One shared gate wraps every primary view: it renders the page header
    THEN the banner in place of the body, so the banner never sits above (and
    mangles) the header. The gate must exist and be applied in App()'s view
    loop, and the old unconditional 'banner above the Dashboard header' render
    must be gone."""
    # The shared gate + its header component exist.
    assert "function PersonaEmptyGate(" in html
    assert "function PersonaEmptyHeader(" in html
    # App() wraps each mounted primary view in the gate, feeding it the header.
    assert "<${PersonaEmptyGate} persona=${persona} header=${html`<${PersonaEmptyHeader}" in html
    # The gate renders the header BEFORE the banner (content region below header).
    gate = html[html.index("function PersonaEmptyGate("):]
    gate = gate[: gate.index("function PersonaEmptyHeader(")]
    hdr = gate.index("${header")
    banner = gate.index("PersonaNoDataNotice")
    assert hdr < banner, "gate must render the header above the banner"
    # The old placement — banner rendered as the FIRST child of DashboardView's
    # own return, above its .ov-head header — must be gone.
    assert "<${PersonaNoDataNotice} persona=${persona} />\n    <div class=\"ov-head\">" not in html


def test_switch_back_is_a_clickable_control_calling_the_persona_setter(html: str) -> None:
    """'Switch back to <other>' is a button that calls the SAME persona setter
    the <select> uses (ctx.onChange), toggling to the other persona — not a
    page reload."""
    notice = html[html.index("function PersonaNoDataNotice("):]
    notice = notice[: notice.index("function PersonaEmptyGate(")]
    assert 'class="link-inline"' in notice
    assert "ctx.onChange && ctx.onChange(otherKey)" in notice
    assert "Switch back to ${other}" in notice
    # No reload / navigation to accomplish the switch.
    assert "location.reload" not in notice


def test_sdk_sidebar_is_identical_to_claude_code(html: str) -> None:
    """Switching persona changes ONLY the page data / empty-state, never the
    sidebar items: the SDK sidebar is byte-for-byte the same as claude-code —
    flat Improve nav, no lens toggle, observe suite hidden, Review inbox +
    Sessions shown."""
    # No Improve/Observe lens toggle for SDK (same as CC).
    assert '.sidebar[data-persona="sdk"] .lens-switch { display: none; }' in html
    # The observe suite is hidden for SDK — via the observe-lens hide rule AND
    # the same PERSONA_HIDDEN_VIEWS list CC uses.
    assert '.sidebar[data-persona="sdk"] a.nav-link[data-lens="observe"] { display: none !important; }' in html
    assert "'sdk': ['traces', 'cost', 'alerts', 'drift', 'budget']," in html
    # The predecessor's SDK-specific forcing of observe links visible, and its
    # hiding of Review inbox + Sessions, must be gone — SDK now mirrors CC.
    assert 'a.nav-link[data-lens="observe"] { display: flex !important; }' not in html
    assert '.sidebar[data-persona="sdk"] a.nav-link[data-view="review"]' not in html
    assert '.sidebar[data-persona="sdk"] a.nav-link[data-view="sessions"]' not in html
    # The lens is forced to 'improve' for both personas so the flat nav renders
    # regardless of stale lens state.
    assert "(persona === 'claude-code' || persona === 'sdk')" in html


# --------------------------------------------------------------------------- #
# Optimize IA redesign (PR #665) — summary landing + per-analyzer detail
# sub-pages, wired to the submenu and the Dashboard tiles.
# --------------------------------------------------------------------------- #
def test_optimize_submenu_has_no_new_badges(html: str) -> None:
    # The "NEW" badge is removed from the Summarize and Rules nav-children.
    for child in ("summarize", "rules"):
        # Grab the anchor line for the child and assert no nav-badge on it.
        marker = f'data-param="{child}" data-optimize-static="1"'
        assert marker in html, f"{child} nav-child must be tagged data-optimize-static"
    # No nav-badge "new" anywhere in the Optimize children.
    assert '<span class="nav-badge">new</span>' not in html, (
        "the NEW badge must be removed from the Optimize nav-children"
    )


def test_optimize_analyzer_children_are_injected_dynamically(html: str) -> None:
    # The submenu is populated from the findings, not a hardcoded per-analyzer
    # list of static anchors. The injector reads optimizeDetailAnalyzers off the
    # stored /optimize artifact and inserts one nav-child per finding before the
    # static Summarize/Rules children.
    assert "function optimizeDetailAnalyzers(opt)" in html
    assert 'a.nav-child[data-optimize-dyn]' in html, "dynamic children must be tagged for reconciliation"
    assert "setOptimizeChildren(optimizeDetailAnalyzers(d))" in html
    assert "a.dataset.optimizeDyn = '1'" in html
    # Rules stays a submenu item (cross-cutting rule-write surface).
    assert 'href="#/optimize/rules"' in html


def test_optimize_detail_route_renders_a_single_analyzer(html: str) -> None:
    # OptimizeView takes navParam (the path segment after #/optimize/) and, when
    # present, renders only that analyzer's OptimizeFinding detail card plus a
    # back link — not the whole stacked page.
    assert "function OptimizeView({ params, navParam })" in html
    assert "const detailName = navParam || null;" in html
    assert "if (detailName) {" in html
    assert "← Back to Optimize" in html
    # App threads the active path param into the view.
    assert "const navParam = isActive ? route.param : null;" in html
    assert "navParam=${navParam}" in html


def test_optimize_summary_no_longer_stacks_every_finding(html: str) -> None:
    # The old long page mapped OptimizeFinding over every analyzer on the summary.
    # That stacked render is gone; the summary is now ONE unified "Recommended
    # actions" list (each row a hint + figure + Review button), which replaced the
    # duplicated waste-list + separate Rules/Summarize/Findings sections.
    assert "${order.map(n => html`<${OptimizeFinding}" not in html, (
        "summary landing must not stack every analyzer's full detail section"
    )
    assert 'id="opt-actions"' in html
    assert 'class="opt-act-list"' in html
    # Each actionable row carries a Review button that opens its detail page.
    assert "class=\"opt-review-btn\"" in html
    # The old three-surface duplication is gone.
    assert 'id="opt-findings"' not in html
    assert 'id="opt-rules"' not in html


def test_dashboard_empty_tiles_are_not_clickable(html: str) -> None:
    # Only a tile with a real finding is an <a> to its detail page; empty-state /
    # at-ceiling tiles render as a non-clickable <div> (.static).
    assert "const hasPage = t.state === 'actionable' && DETAIL_ANALYZER_NAMES.has(t.name);" in html
    assert "const href = '#/optimize/' + t.name;" in html
    assert "? html`<a class=${cls} href=${href}>${inner}" in html
    # data tiles carry a persistent "→" cue inside the link; empty tiles do not
    assert '<span class="rec-go" aria-hidden="true">→</span></a>' in html
    assert "html`<div class=${cls}>${inner}</div>`" in html
    # The .static class strips the clickable affordance (no hover border, default
    # cursor) so an empty tile cannot read as a dead link.
    assert ".rec-tile.static { cursor: default; }" in html
    assert ".rec-tile.static:hover { border-color: var(--border); }" in html
