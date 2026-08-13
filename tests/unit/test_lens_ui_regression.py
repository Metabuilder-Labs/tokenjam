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

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _UI.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# #654 — Dashboard is a persistent top-level item + the default landing route.
# --------------------------------------------------------------------------- #
def test_dashboard_nav_link_is_the_persistent_front_door(html: str) -> None:
    """Dashboard is the persistent top-level entry.

    THIS TEST USED TO ENFORCE THE REMOVED LENS. It pinned the Dashboard link as
    carrying data-lens="all" so the improve/observe hide rules could not remove
    it. The Improve/Observe lens was dead for every persona and has been
    removed, so the assertion is INVERTED rather than deleted: the link stays
    pinned as present, and the lens attribute is pinned ABSENT so the removed
    concept cannot creep back in through this markup.
    """
    assert (
        '<a href="#/dashboard" class="nav-link" data-view="dashboard">' in html
    ), "Dashboard nav link must be present as the persistent front door"
    assert "data-lens" not in html, "the removed lens concept must not return"


def test_dashboard_link_sits_at_the_top_of_the_nav(html: str) -> None:
    # It used to be pinned above the Improve/Observe toggle. With the lens gone,
    # the durable property is that Dashboard still leads the nav.
    dash = html.index('<a href="#/dashboard" class="nav-link" data-view="dashboard">')
    optimize = html.index('<a href="#/optimize" class="nav-link" data-view="optimize">')
    assert dash < optimize, "Dashboard nav link must lead the sidebar"
    assert '<div class="lens-switch"' not in html, "the lens switch must stay removed"


def test_persistent_front_door_has_a_style_rule(html: str) -> None:
    # The spacing rule that seated Dashboard above the toggle is re-anchored to
    # the view, not to the removed lens attribute.
    assert '.sidebar a.nav-link[data-view="dashboard"]' in html
    assert '.sidebar a.nav-link[data-lens="all"]' not in html


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


def test_the_lens_concept_is_gone(html: str) -> None:
    """The Improve/Observe lens was dead UI and is removed.

    It was hidden by CSS for BOTH personas, so no user could ever reach it, but
    the hide was keyed on `.sidebar[data-persona="..."]` — an attribute
    syncNavState leaves EMPTY until the persona resolves. That made the switch
    render during the loading state and only then. The whole mechanism is gone;
    this pins every part of it absent so it cannot return piecemeal.
    """
    for dead in (
        '<div class="lens-switch"',
        "lens-btn",
        "data-lens",
        "VIEW_LENS",
        "dataset.lens",
        "LENS_HOME",
        ">Improve<",
        ">Observe<",
    ):
        assert dead not in html, f"removed lens machinery is back: {dead}"


def test_no_persona_computes_a_lens(html: str) -> None:
    # Both personas used to be FORCED to the improve lens so the flat nav
    # rendered regardless of stale lens state. With the lens removed the nav is
    # unconditionally flat, so the forcing expression must be gone too.
    assert "const lens = (persona === 'claude-code' || persona === 'sdk')" not in html
    assert "? 'improve'" not in html


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


def test_sdk_sidebar_differs_from_claude_code_only_where_sdk_has_a_lever(
    html: str,
) -> None:
    """SDK and claude-code share one flat Improve nav; SDK additionally keeps
    the surfaces that only make sense for a deployed service.

    THIS TEST USED TO ENFORCE THE DEFECT. It asserted the two personas' hidden
    lists were byte-for-byte identical — which pinned `traces`, `cost`,
    `alerts`, `drift` and `budget` as hidden under EVERY persona key, i.e. as
    views no user of any kind could reach. The five screens were built, routed
    and populated the whole time; only the gate was wrong, and a green suite was
    defending it. The assertion is INVERTED rather than deleted: the shared
    parts stay pinned as present, and the identical-lists state is now pinned as
    ABSENT so it cannot come back.
    """
    # Still shared: neither persona has a lens toggle, because the lens is gone
    # entirely (see test_the_lens_concept_is_gone).
    assert '<div class="lens-switch"' not in html
    # THE BAD STATE, pinned absent: the two persona keys must not carry the same
    # list, and the observe suite must not be hidden wholesale for SDK.
    assert "'sdk': ['traces', 'cost', 'alerts', 'drift', 'budget']," not in html, (
        "SDK must not hide the same five views claude-code does — that left no "
        "persona able to reach any of them"
    )
    # The correct state, pinned present. Spend is the only view hidden for BOTH,
    # and it carries a recorded reason (see the deliberate-hide test below).
    assert "'claude-code': ['traces', 'cost']," in html
    assert "'sdk': ['cost']," in html
    # Traces is SDK-only, hidden for claude-code by its own per-view rule rather
    # than by a blanket lens hide (per-session traces stay reachable for every
    # persona from the session detail's Traces tab).
    assert (
        '.sidebar[data-persona="claude-code"] a.nav-link[data-view="traces"] '
        '{ display: none !important; }' in html
    )
    # Alerts / Drift are no longer top-level nav entries at all — they moved
    # into the Sessions screen's SDK-services zone.
    assert '<a href="#/alerts" class="nav-link" data-view="alerts"' not in html
    assert '<a href="#/drift" class="nav-link" data-view="drift"' not in html
    # The predecessor's SDK-specific forcing of observe links visible, and its
    # hiding of Review inbox + Sessions, must still be gone.
    assert '.sidebar[data-persona="sdk"] a.nav-link[data-view="review"]' not in html
    assert '.sidebar[data-persona="sdk"] a.nav-link[data-view="sessions"]' not in html


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
    assert "optimizeDetailAnalyzers(d)" in html
    assert "a.dataset.optimizeDyn = '1'" in html
    # The submenu is PERSONA-AWARE (#671): the effect re-derives when the
    # selected persona changes, and a no-data persona yields an empty submenu.
    assert "}, [persona, personaHasNoData]);" in html, (
        "the submenu effect must depend on the selected persona so it re-derives on toggle"
    )
    assert "personaHasNoData" in html
    # The static Summarize/Rules children are gated (not injected
    # unconditionally) via a data-optimize-hidden flag syncNavState honors.
    assert "optimizeHidden" in html
    # Rules stays a submenu item (cross-cutting rule-write surface).
    assert 'href="#/optimize/rules"' in html


def test_optimize_detail_route_renders_a_single_analyzer(html: str) -> None:
    # OptimizeView takes navParam (the path segment after #/optimize/) and, when
    # present, renders only that analyzer's OptimizeFinding detail card plus a
    # back link — not the whole stacked page.
    # `persona` joins the signature: the view reads the report AS the selected
    # persona rather than as whichever one the corpus happens to be dominated by.
    assert "function OptimizeView({ params, navParam, persona })" in html
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


# --------------------------------------------------------------------------- #
# Lens table horizontal-overflow fix — wide .opt-table findings tables (long
# absolute paths, provider/model strings) were pushing the whole page into
# horizontal scroll instead of scrolling inside their own container.
# --------------------------------------------------------------------------- #
def test_body_never_scrolls_horizontally(html: str) -> None:
    # Belt-and-suspenders: no descendant, however wide, may push the PAGE
    # itself into horizontal scroll. Wide content must scroll inside its own
    # .table-wrap instead.
    assert "overflow-x: hidden;" in html
    body = html[html.index("\nbody {"):]
    body = body[: body.index("}")]
    assert "overflow-x: hidden;" in body, "body rule must set overflow-x: hidden"


def test_every_opt_table_is_wrapped_for_horizontal_scroll(html: str) -> None:
    # Every OptimizeFinding detail table (.opt-table) must sit inside a
    # .table-wrap (overflow-x: auto) container so a long unbreakable string
    # (a repo-relative path, a provider/model id) scrolls inside the table
    # instead of forcing the whole card, and the page, wider than the
    # viewport. A bare, unwrapped `<table class="opt-table"` is the bug.
    import re

    for m in re.finditer(r'<table class="opt-table"', html):
        preceding = html[max(0, m.start() - 40): m.start()]
        assert '<div class="table-wrap">' in preceding, (
            f"unwrapped .opt-table at offset {m.start()}: {preceding!r}"
        )


def test_recurring_inclusions_label_is_truncated_with_full_path_in_title(html: str) -> None:
    # The "What's re-included" column in the resend finding's recurring-
    # inclusions table renders absolute paths. Rendering them untruncated
    # forced the table (and the page) wider than the viewport, with the
    # label unreadable on both ends. shortPath() truncates to the last two
    # path segments for display; the full path still rides in title= for a
    # hover tooltip.
    assert (
        '<td class="mono" title=${r.label}>${shortPath(r.label)}</td>' in html
    ), "recurring-inclusions label must be shortPath()-truncated with the full value in title="


def test_cursor_listbox_scrolls_horizontally_and_truncates_paths(html: str) -> None:
    # The Summarize/Rules file pickers (.cur-listbox > .cur-table) render a
    # File column of absolute paths. The listbox must scroll horizontally on
    # its own (overflow-x, alongside its existing overflow-y) instead of
    # relying on the page to scroll, and the path itself must be
    # shortPath()-truncated with the full path in a hover title.
    assert (
        ".cur-listbox { max-height:380px; overflow-y:auto; overflow-x:auto;" in html
    ), "cur-listbox must scroll horizontally, not just vertically"
    assert '<td class="mono" title=${c.path}>${shortPath(c.path)}</td>' in html
    assert "title=${r.path + ' — review the diff'}" in html


# --------------------------------------------------------------------------- #
# Total opportunity tile — a 7th tile, first in the Dashboard's "Opportunities
# to optimize token efficiency" row, summing the six per-analyzer figures.
#
# `totalOpportunityFigure()` is the one pure function that decides the sum and
# its population; a static string match on the source would still pass if that
# arithmetic or exclusion logic were wrong (the exact critique
# test_lens_select_all_behaviour.py levels at grep-only tests), so it is
# extracted straight out of the served index.html and run under node instead,
# the same trick that module and test_lens_dashboard_states.py use.
# --------------------------------------------------------------------------- #
_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available for JS evaluation")


def _round_to_cents_source() -> str:
    src = _UI.read_text(encoding="utf-8")
    start = src.index("function roundToCents(n)")
    end = src.index("\n}\n", start) + 2
    return src[start:end]


def _total_figure_source() -> str:
    src = _UI.read_text(encoding="utf-8")
    start = src.index("function totalOpportunityFigure")
    end = src.index("// The TOTAL tile itself", start)
    # totalOpportunityFigure calls roundToCents (the same helper fmtDashUsd
    # uses) to round each contributor before summing -- pull it in too so
    # this extraction doesn't drift from the real dependency.
    return _round_to_cents_source() + "\n" + src[start:end]


def _total_figure(tiles: list[dict]):
    script = (
        _total_figure_source()
        + "\nconsole.log(JSON.stringify(totalOpportunityFigure(" + json.dumps(tiles) + ")));"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip())


@_node
def test_total_equals_the_plain_sum_of_actionable_contributors():
    tiles = [
        {"name": "subagent", "state": "actionable", "usd": 1528.89, "tokens": 100},
        {"name": "resend", "state": "actionable", "usd": 374.33, "tokens": 200},
        {"name": "downsize", "state": "actionable", "usd": 295.23, "tokens": 300},
    ]
    fig = _total_figure(tiles)
    assert fig["state"] == "populated"
    assert round(fig["totalUsd"], 2) == round(1528.89 + 374.33 + 295.23, 2)
    assert fig["totalTokens"] == 600
    assert fig["contributorCount"] == 3


@_node
def test_total_reconciles_with_the_sum_of_the_displayed_per_tile_figures():
    # The total must equal what you get from adding the SIX RENDERED figures
    # by hand, not a sum of raw values rounded once at the end -- those two
    # differ whenever contributors' raw cents round independently. Fixture
    # chosen so the two strategies disagree by a whole cent, unaffected by
    # any binary floating-point boundary ambiguity (verified directly: raw
    # sum 3.012 rounds once to 3.01, but each 1.004 individually rounds down
    # to 1.00, so the sum of the three DISPLAYED figures is 3.00).
    tiles = [
        {"name": "subagent", "state": "actionable", "usd": 1.004, "tokens": 1},
        {"name": "resend", "state": "actionable", "usd": 1.004, "tokens": 1},
        {"name": "downsize", "state": "actionable", "usd": 1.004, "tokens": 1},
    ]
    fig = _total_figure(tiles)
    # The sum of the values AS DISPLAYED (each tile renders $1.00 via
    # fmtDashUsd's 2dp rounding) is $3.00 -- not the sum-then-round-once
    # figure of $3.01 a naive raw sum would produce.
    assert fig["totalUsd"] == 3.00
    assert fig["totalUsd"] != round(1.004 + 1.004 + 1.004, 2)  # != 3.01


@_node
def test_an_absent_no_findings_analyzer_is_excluded_not_zeroed():
    # Deadweight with no candidates renders 'No candidates', never a $0 tile
    # (root CLAUDE.md anti-pattern 22). Its state carries no usd at all here,
    # mirroring classifyFinding()'s real 'no_findings' shape, and the sum must
    # come out identical to the same row with that tile removed entirely --
    # proof it is excluded structurally (by state), not by a falsy usd check.
    with_deadweight = [
        {"name": "subagent", "state": "actionable", "usd": 100.0, "tokens": 10},
        {"name": "deadweight", "state": "no_findings"},
    ]
    without_deadweight = [
        {"name": "subagent", "state": "actionable", "usd": 100.0, "tokens": 10},
    ]
    with_fig = _total_figure(with_deadweight)
    without_fig = _total_figure(without_deadweight)
    assert with_fig["totalUsd"] == without_fig["totalUsd"] == 100.0
    assert with_fig["contributorCount"] == without_fig["contributorCount"] == 1
    # The excluded tile is still counted as a KNOWN (resolved) tile, just not
    # a contributor -- it answered "nothing here", which is not the same as
    # "not yet known".
    assert with_fig["knownCount"] == 2


@_node
def test_an_at_ceiling_tile_contributes_nothing_to_the_sum():
    # cache's positive "already at the ceiling" state carries a metric string,
    # never a usd figure; it must be excluded the same way no_findings is.
    tiles = [
        {"name": "subagent", "state": "actionable", "usd": 50.0, "tokens": 5},
        {"name": "cache", "state": "at_ceiling", "metric": "98% cache efficacy"},
    ]
    fig = _total_figure(tiles)
    assert fig["totalUsd"] == 50.0
    assert fig["contributorCount"] == 1


@_node
def test_all_analyzers_unresolved_is_the_unknown_state_never_zero():
    # Nothing has resolved yet -- the tile must render a skeleton, not a $0.00
    # total (the worst possible placeholder: it reads as "no waste").
    tiles = [
        {"name": "subagent", "state": "not_ready", "hint": "Not run on Overview."},
        {"name": "resend", "state": "not_ready", "hint": "Not run on Overview."},
    ]
    fig = _total_figure(tiles)
    assert fig["state"] == "unknown"
    assert fig["totalUsd"] == 0
    assert fig["knownCount"] == 0


@_node
def test_every_analyzer_resolved_empty_is_the_empty_state():
    # Every tile answered, none had a recoverable figure: this is the ONE
    # legitimate home for empty-state copy, distinct from 'unknown'.
    tiles = [
        {"name": "subagent", "state": "no_findings"},
        {"name": "deadweight", "state": "no_findings"},
    ]
    fig = _total_figure(tiles)
    assert fig["state"] == "empty"
    assert fig["totalUsd"] == 0
    assert fig["contributorCount"] == 0
    assert fig["knownCount"] == 2


@_node
def test_a_partially_resolved_row_discloses_its_coverage_not_a_full_claim():
    # Some analyzers answered, some have not: the total must not claim to
    # cover every tile in the row. unresolvedCount is how the renderer knows
    # to disclose the partial population instead of publishing a total that
    # reads as complete.
    tiles = [
        {"name": "subagent", "state": "actionable", "usd": 10.0, "tokens": 1},
        {"name": "resend", "state": "not_ready", "hint": "Not run on Overview."},
    ]
    fig = _total_figure(tiles)
    assert fig["state"] == "populated"
    assert fig["totalUsd"] == 10.0
    assert fig["unresolvedCount"] == 1
    assert fig["totalCount"] == 2
    assert fig["knownCount"] == 1


def test_total_tile_is_first_in_the_row_and_visually_distinct(html: str) -> None:
    # Rendered before tiles.map(), so it is the first child of .tile-grid; a
    # dedicated CSS class carries the weight/border distinction (never the
    # accent colour, which means "typeable/clickable" and this tile links
    # nowhere). The per-tile share bar (lens redesign) wraps the map in an
    # IIFE to compute the rank/tone ramp once, so anchor on render ORDER
    # rather than a single unbroken substring.
    #
    # `fig` (not `tiles`) is what the tile now takes: it is the SAME netted,
    # server-computed rollup figure the Dashboard hero above this row reads,
    # rather than a client-side sum of the six sibling tiles' own figures —
    # see `rollupFigure()` / the comment above `totalOpportunityFigure()`.
    assert "<${TotalOpportunityTile} fig=${rollupFig} framing=${framing} />${" in html
    idx_total = html.index("<${TotalOpportunityTile} fig=${rollupFig} framing=${framing} />${")
    idx_map = html.index("return tiles.map(t => {", idx_total)
    assert idx_total < idx_map, "the total tile must render before tiles.map() in the tile-grid"
    assert ".rec-tile.total-tile" in html
    assert ".rec-tile.total-tile .rec-amount { color: var(--text); }" in html


def test_total_tile_cannot_render_a_dollar_figure_while_unresolved(html: str) -> None:
    # The 'unknown' branch returns before any amount/hint computation touches
    # fmtFramedSavings -- a skeleton, never a number, while nothing has
    # resolved. Anchor on the guard clause and its skeleton markup.
    fn = html[html.index("function TotalOpportunityTile("):]
    fn = fn[: fn.index("\n}\n")]
    assert "if (fig.state === 'unknown') {" in fn
    idx_guard = fn.index("if (fig.state === 'unknown')")
    idx_amount = fn.index("const amount")
    assert idx_guard < idx_amount, "the unresolved guard must return before computing an amount"
    assert 'class="rec-tile total-tile rec-skel" aria-hidden="true"' in fn


def test_total_tile_comment_marks_the_sum_as_deliberate_and_scoped(html: str) -> None:
    # The founder decision (naive sum now, netted rollup once `script` runs
    # for a persona that reaches this row) must be recorded at the summing
    # site, with no internal ticket id per root anti-pattern 11.
    fn_start = html.index("function totalOpportunityFigure")
    comment = html[html.index("// The TOTAL opportunity tile"): fn_start]
    assert "PLAIN SUM" in comment
    assert "netted cross-analyzer" in comment
    assert "persona-disabled" in comment
    import re
    assert not re.search(r"#\d+", comment), "no internal ticket id in a source comment"


def test_total_matches_the_displayed_per_tile_sum(html: str) -> None:
    # roundToCents backs fmtDashUsd's own rounding (a tile's displayed
    # figure) AND totalOpportunityFigure's summing step, from the same
    # helper -- never two independent rounding expressions that could drift
    # apart. Anchors both call sites plus the shared helper's definition.
    assert "function roundToCents(n) {" in html
    assert "return '$' + roundToCents(n).toLocaleString(" in html  # fmtDashUsd
    assert (
        "const totalUsd = roundToCents(contributors.reduce((sum, t) => sum + roundToCents(t.usd), 0));"
        in html
    )


# --------------------------------------------------------------------------- #
# `rollupFigure()` — what the Total opportunity tile AND the Dashboard hero
# now render instead of `totalOpportunityFigure(tiles)`'s plain sum. This is
# the JS-level pin for the netting fix: the figure must come straight off the
# wire's `past_overspend_usd` (GET /relearn/cost-proposals, already netted
# server-side via `_net_cross_analyzer_session_overlap`), never re-derived by
# adding per-analyzer figures client-side — which is exactly what would
# double-count once `reuse` and `script` (both cluster on the identical
# repeated-tool-sequence shape) are enabled together for one persona (`sdk`).
# --------------------------------------------------------------------------- #
def _rollup_figure_source() -> str:
    src = _UI.read_text(encoding="utf-8")
    start = src.index("function rollupFigure(read)")
    end = src.index("\n}\n", start) + 2
    return src[start:end]


def _rollup_figure(read: dict):
    script = (
        _rollup_figure_source()
        + "\nconsole.log(JSON.stringify(rollupFigure(" + json.dumps(read) + ")));"
    )
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip())


@_node
def test_rollup_figure_is_unknown_before_any_complete_read_lands():
    assert _rollup_figure({"phase": "loading", "data": None})["state"] == "unknown"


@_node
def test_rollup_figure_is_unknown_for_a_store_that_never_computed():
    # A 'never_run' payload answers the HTTP request but never set
    # `computed_at` -- it must not read as a measured empty answer (root
    # CLAUDE.md anti-pattern 22: an un-run scan is not an all-clear).
    data = {"status": "never_run", "computed_at": None, "past_overspend": {}}
    fig = _rollup_figure({"phase": "ready", "data": data})
    assert fig["state"] == "unknown"
    assert fig["totalUsd"] == 0


@_node
def test_rollup_figure_reads_the_wire_total_verbatim_never_a_client_side_sum():
    # The whole point of the fix: `totalUsd` is `past_overspend_usd` as
    # published, never re-summed from `by_analyzer`. Chosen so a naive
    # client-side sum of the two contributors (50.0 + 73.45 = 123.45) would
    # happen to match here on its own -- the assertion that matters is the
    # LAST one, proving this function never performs that addition at all.
    data = {
        "status": "ready",
        "computed_at": "2026-05-30T00:00:00Z",
        "past_overspend": {
            "past_overspend_usd": 123.45,
            "past_overspend_tokens": 999,
            "proposal_count": 3,
            "token_proposal_count": 3,
            "deduplicated_proposal_count": 4,
            "by_analyzer": [
                {"analyzer": "reuse", "usd": 50.0, "tokens": 400, "count": 1},
                {"analyzer": "script", "usd": 73.45, "tokens": 599, "count": 2},
            ],
        },
    }
    fig = _rollup_figure({"phase": "ready", "data": data})
    assert fig["state"] == "populated"
    assert fig["totalUsd"] == 123.45
    assert fig["totalTokens"] == 999
    assert fig["dedupedCount"] == 4
    assert fig["byAnalyzer"] == data["past_overspend"]["by_analyzer"]


@_node
def test_rollup_figure_double_counted_overlap_would_be_visible_if_ever_reintroduced():
    # Guards against a regression back to the old naive-sum shape: if
    # `rollupFigure` ever started re-summing `by_analyzer` instead of trusting
    # the server's own `past_overspend_usd`, this fixture (reuse and script
    # both claiming the SAME 20 sessions before netting, netted total well
    # below their raw sum) would silently start reporting the bigger, wrong
    # number. Mirrors the overlap shape
    # test_dashboard_hero_netted_rollup.py proves on the real wire.
    data = {
        "status": "ready",
        "computed_at": "2026-05-30T00:00:00Z",
        "past_overspend": {
            "past_overspend_usd": 60.0,   # the server's netted figure
            "past_overspend_tokens": 500,
            "proposal_count": 2,
            "token_proposal_count": 2,
            "deduplicated_proposal_count": 2,
            "by_analyzer": [
                # Raw, pre-netting figures a naive client-side sum would add
                # to 100.0 -- strictly more than the netted total above.
                {"analyzer": "reuse", "usd": 50.0, "tokens": 300, "count": 1},
                {"analyzer": "script", "usd": 50.0, "tokens": 300, "count": 1},
            ],
        },
    }
    fig = _rollup_figure({"phase": "ready", "data": data})
    naive_sum = sum(a["usd"] for a in data["past_overspend"]["by_analyzer"])
    assert fig["totalUsd"] == 60.0
    assert fig["totalUsd"] < naive_sum


@_node
def test_rollup_figure_is_empty_only_once_a_completed_pass_found_nothing():
    data = {
        "status": "ready",
        "computed_at": "2026-05-30T00:00:00Z",
        "past_overspend": {
            "past_overspend_usd": 0, "past_overspend_tokens": 0,
            "proposal_count": 0, "token_proposal_count": 0,
            "deduplicated_proposal_count": 0, "by_analyzer": [],
        },
    }
    fig = _rollup_figure({"phase": "ready", "data": data})
    assert fig["state"] == "empty"
    assert fig["totalUsd"] == 0


def test_opportunities_row_fits_seven_tiles_without_widening_the_shared_compact_grid(html: str) -> None:
    # The Total tile makes this a 7-tile row (was 6), which orphaned the 7th
    # (Deadweight) onto its own row at the shared .tile-grid.compact minmax.
    # .opp-grid narrows just this row's minmax so all seven fit across at
    # the normal content width; it must NOT touch the shared .compact rule
    # (the health-glance row and this row's own loading skeleton also use
    # it, and don't need narrowing), and both the answered-tiles grid and
    # its loading skeleton must carry the class so neither reflows against
    # the other when the real data lands.
    assert ".tile-grid.compact.opp-grid { grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); }" in html
    assert '.tile-grid.compact { grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));' in html
    assert 'class="tile-grid compact opp-grid"' in html
    # Both call sites carry it: the scanning skeleton and the answered tiles.
    assert html.count('class="tile-grid compact opp-grid"') == 2
    # The health-glance row is a separate grid and must NOT be narrowed.
    health = html[html.index('<div class="section-band">Health at a glance</div>'):]
    health = health[: health.index("<!-- The HERO")]
    assert 'class="tile-grid compact"' in health
    assert "opp-grid" not in health


def test_skeleton_tile_count_matches_the_seven_real_tiles(html: str) -> None:
    # REC_SKELETON_TILES stood in for the row before the Total tile existed
    # (6 placeholders for 6 real tiles). Left at 6 it would render one fewer
    # skeleton box than the 7 real tiles that land, a visible reflow.
    assert "const REC_SKELETON_TILES = [0, 1, 2, 3, 4, 5, 6];" in html
    assert "const REC_SKELETON_TILES = [0, 1, 2, 3, 4, 5];" not in html


# --------------------------------------------------------------------------- #
# Persona VIEW gate — which pages each persona is offered.
#
# The defect this section guards against is structural rather than cosmetic: a
# view listed under EVERY persona key is reachable by nobody, and it reads
# exactly like a view that was deliberately shelved. Five built, routed and
# populated screens sat in that state, and a green suite was defending it.
# --------------------------------------------------------------------------- #
def _persona_hidden_views(html: str) -> dict[str, list[str]]:
    """Parse ``PERSONA_HIDDEN_VIEWS`` out of the UI source.

    Parsed rather than re-declared: a test carrying its own copy of the map
    passes while the map says something else, which is the whole failure mode
    here. The literal is plain arrays of single-quoted names by construction
    (there is a separate guard forbidding a Set literal), so a small regex is
    enough and a shape change fails loudly instead of silently matching nothing.
    """
    body = html[html.index("const PERSONA_HIDDEN_VIEWS = {"):]
    body = body[: body.index("};")]
    out: dict[str, list[str]] = {}
    for line in body.splitlines():
        m = re.match(r"\s*'([a-z-]+)'\s*:\s*\[(.*)\]\s*,", line)
        if m:
            out[m.group(1)] = re.findall(r"'([a-z-]+)'", m.group(2))
    assert out, "PERSONA_HIDDEN_VIEWS must parse — its literal shape changed"
    return out


def _persona_hidden_deliberate(html: str) -> set[str]:
    body = html[html.index("const PERSONA_HIDDEN_DELIBERATE = {"):]
    body = body[: body.index("};")]
    return set(re.findall(r"^\s{2}([a-z-]+):", body, re.M))


def test_no_view_is_hidden_for_every_persona_without_a_recorded_reason(
    html: str,
) -> None:
    """THE pin for the hidden-views defect.

    A view hidden under every persona key cannot be opened by anyone. That is a
    legitimate product decision (Spend is one), but it is indistinguishable from
    the bug where a list was copied onto a second persona by mistake — so a
    deliberate one must record WHY in PERSONA_HIDDEN_DELIBERATE. Anything hidden
    everywhere with no entry there fails here.
    """
    hidden = _persona_hidden_views(html)
    assert set(hidden) == {"claude-code", "sdk"}, (
        "a new persona key changes what 'hidden for everyone' means — update "
        "this guard deliberately"
    )
    hidden_everywhere = set.intersection(*(set(v) for v in hidden.values()))
    recorded = _persona_hidden_deliberate(html)
    assert hidden_everywhere <= recorded, (
        f"{sorted(hidden_everywhere - recorded)} is hidden for every persona "
        f"with no reason recorded in PERSONA_HIDDEN_DELIBERATE — either it is "
        f"deliberate and must say so, or the gate is wrong"
    )
    # And the deliberate list may not grow beyond what is actually hidden
    # everywhere: a stale entry would license a future blanket hide.
    assert recorded <= hidden_everywhere, (
        f"{sorted(recorded - hidden_everywhere)} records a deliberate blanket "
        f"hide for a view that is not hidden everywhere"
    )


def test_spend_is_the_deliberately_hidden_one_and_traces_is_sdk_only(
    html: str,
) -> None:
    """The decided target state, named explicitly rather than only set-wise."""
    hidden = _persona_hidden_views(html)
    # Spend: hidden for both, on purpose.
    assert "cost" in hidden["claude-code"] and "cost" in hidden["sdk"]
    assert "cost" in _persona_hidden_deliberate(html)
    # Traces: the standalone cross-session browser is SDK-only. Per-session
    # traces reach every persona through the session detail's Traces tab.
    assert "traces" in hidden["claude-code"]
    assert "traces" not in hidden["sdk"]
    # Alerts / budget are not top-level views any more, so they are in neither
    # list — they are tabs of the Sessions screen instead.
    for view in ("alerts", "budget"):
        assert view not in hidden["claude-code"], view
        assert view not in hidden["sdk"], view
        assert view in html[html.index("const SESSIONS_SDK_TAB_VIEWS"):][:200], view
    # Drift is in neither list either, but for the OPPOSITE reason: it is not
    # gated per persona, it is un-surfaced for everyone. This loop used to
    # include it and would have read as "correctly relocated" while the tab was
    # gone, so the two cases are asserted separately.
    assert "drift" not in hidden["claude-code"]
    assert "drift" not in hidden["sdk"]
    tab_views_line = html[html.index("const SESSIONS_SDK_TAB_VIEWS"):]
    assert "drift" not in tab_views_line[: tab_views_line.index("\n")]


def test_alerts_drift_budget_are_sessions_tabs_not_top_level_views(
    html: str,
) -> None:
    """Alerts and Budget are tabs of the Sessions screen. Drift is UN-SURFACED.

    THIS TEST USED TO PIN THE STATE THAT WAS REMOVED. It asserted a Drift tab
    was rendered in the SDK-services zone and that the Sessions nav row lit up
    on `#/drift` -- i.e. it was an active defence of the surface the founder
    decided to withdraw, and it would have failed the moment anyone withdrew it.
    The assertions are INVERTED rather than deleted: alerts and budget stay
    pinned as tabs, and Drift is pinned ABSENT from the strip so the removal
    cannot silently come back.

    UN-SURFACED IS NOT DELETED. `DriftView`, the `/drift` route, the detector
    and its baselines all remain and are still covered by their own tests (see
    `test_drift_empty_state_has_scannable_headline`, which passes on the KEPT
    component). What is pinned here is that nothing renders it and nothing links
    to it.
    """
    # primaryKeyFor sends the surviving tabs to the Sessions screen.
    assert "if (SESSIONS_SDK_TAB_VIEWS.has(v)) return 'sessions';" in html
    # They are not keep-alive primary views (that would mount panes the router
    # can never activate). DriftView must not return to that list either.
    for entry in ("['alerts',    AlertsView]", "['drift',     DriftView]",
                  "['budget',    BudgetView]"):
        assert entry not in html, f"{entry} must leave PRIMARY_VIEWS"
    # StatusView renders the surviving two, inside the SDK-only zone.
    assert "${sdkTab === 'alerts' ? html`<${AlertsView}" in html
    assert "${sdkTab === 'budget' ? html`<${BudgetView}" in html
    # THE REMOVED STATE, pinned absent: no Drift tab is rendered anywhere.
    assert "${sdkTab === 'drift' ? html`<${DriftView}" not in html, (
        "Drift is un-surfaced: the SDK zone must not render a Drift tab"
    )
    # The Sessions nav row stays lit on the surviving tabs, and only those.
    assert 'data-view="sessions" data-view-alt="alerts,budget"' in html
    assert 'data-view-alt="alerts,drift,budget"' not in html
    assert "(el.dataset.viewAlt || '').split(',')" in html


def test_the_sessions_sub_tabs_receive_the_selected_persona(html: str) -> None:
    """A sub-tab is a view too, and its RENDER SITE is where the scope is lost.

    The Alerts tab shipped rendering as `<${AlertsView} params=${EMPTY_PARAMS} />`
    with no persona at all, so under the SDK persona it listed claude-code and
    codex alerts: the endpoint filters correctly, the client simply never asked
    it to. A signature-only assertion would not have caught this, because the
    parent dropped the prop rather than the child ignoring it. So pin BOTH ends.
    """
    # The render site hands it down.
    assert (
        "<${AlertsView} params=${EMPTY_PARAMS} persona=${persona} />" in html
    ), "the Alerts tab must receive the selected persona from its parent"
    # The component reads it and forwards it to its own fetch.
    alerts_at = html.index("function AlertsView(")
    sig_end = html.index(")", alerts_at)
    assert "persona" in html[alerts_at:sig_end], "AlertsView must take a persona"
    body = html[alerts_at:alerts_at + 4000]
    assert "persona: persona || undefined" in body, (
        "AlertsView must send the persona to /alerts"
    )
    # ...and reload when it changes, or a switch leaves the old list rendered.
    assert "[severity, unread, agentId, since, persona]" in body, (
        "persona must be a load dependency of the alerts read"
    )
    # Budget's own alert reads cover the same agents its caps do.
    assert "type: 'cost_budget_daily', since: '24h', persona: 'sdk'" in html
    assert "type: 'cost_budget_session', since: '24h', persona: 'sdk'" in html


def test_drift_is_unreachable_from_every_surface(html: str) -> None:
    """The founder's decision, as a guard: Drift has no way in.

    Every removal below is a separate door someone could reopen without
    noticing the others, so they are asserted as one set rather than one at a
    time: the tab registry, the route resolution, the page-title map, the
    Dashboard health tile, and the zone heading's copy.
    """
    # The tab registry: not offered, and not a recognised sub-tab view.
    assert "['drift', 'Drift']," not in html
    assert "const SESSIONS_SDK_TAB_VIEWS = new Set(['alerts', 'budget']);" in html
    # The hash still resolves, but to the Sessions screen, NOT to a Drift
    # surface. A 404 would be a worse answer for an existing bookmark.
    assert "const UNSURFACED_VIEWS = new Set(['drift']);" in html
    assert "if (UNSURFACED_VIEWS.has(v)) return 'sessions';" in html
    # No page title, so no surface can caption itself "Drift".
    assert "drift: 'Drift'" not in html
    # No health tile and no link to the route. The tile published a figure and
    # a caption ("within baseline") straight onto the Dashboard.
    assert 'label="Agents drifting"' not in html
    assert 'href="#/drift"' not in html
    # The zone heading is gone entirely -- it duplicated the page's own
    # "Sessions" H1 -- so neither wording can render. Pinned absent rather
    # than deleted so a restored heading cannot quietly bring drift back.
    assert "alerts, drift and caps across agents" not in html
    assert "alerts and caps across agents" not in html
    assert '<span class="zone-title">SDK services</span>' not in html


def test_the_drift_mechanism_is_kept_intact_behind_the_removed_surface(
    html: str,
) -> None:
    """Un-surfacing, not deleting -- so re-surfacing is a wiring change.

    Pinned as PRESENT so a later "this is dead code, nothing references it"
    sweep has to argue with a test rather than a comment. The component, its
    fetch, the signal helper and the Dashboard read that still feeds
    `data_span` all stay.
    """
    assert "function DriftView({ params }) {" in html
    assert "api('/drift', { agent_id: agentId || undefined })" in html
    assert "function driftSignalCount(drift) {" in html
    # The Dashboard read survives: it is this page's source for `data_span`,
    # and it is asserted independently in test_lens_dashboard_states.py.
    assert "const driftRead = useTriageRead(" in html


def test_persona_gate_hides_nothing_until_the_persona_is_known(html: str) -> None:
    """A not-yet-known persona must apply no PERSONA-DEPENDENT hiding rules.

    The old fallback resolved an unknown persona to a concrete one, so an
    unresolved read silently applied a real persona's hiding rules to a reader
    who might be the other persona. Both halves of the gate — the JS predicate
    and the CSS attribute it pairs with — now key on `known`.

    NARROWED, NOT WEAKENED. This still pins the anti-over-hiding property, which
    is the whole point of it: a view the read genuinely decides may not be hidden
    on a guess. What it no longer implies is that NOTHING may be hidden while
    unknown. A view hidden under every persona key is not a question the read can
    settle, and showing it advertised a route that renders nothing when clicked;
    that one case is hidden immediately, keyed on a set derived from the lists.
    See test_lens_persona_hidden_mirror.py for the split and for the assertion
    that the persona-dependent case is still exempt.
    """
    assert "function personaHides(persona, view, known) {" in html
    assert "if (!known) return false;" in html
    # The CSS half: syncNavState leaves data-persona EMPTY until settled, so no
    # [data-persona="..."] rule can match.
    assert "const personaAttr = personaKnown ? persona : '';" in html
    assert "sidebar.dataset.persona = personaAttr;" in html
    # The old unconditional write must be gone.
    assert "sidebar.dataset.persona = persona;" not in html
    # An explicit user choice counts as settled even before the fetch lands.
    assert "const personaKnown = personaOverride != null || personaInfo.known;" in html


# --------------------------------------------------------------------------- #
# Persona ANALYZER gate — the selected persona reaches the server.
# --------------------------------------------------------------------------- #
def test_optimize_reads_are_scoped_to_the_selected_persona(html: str) -> None:
    """Every /optimize read names the persona the reader picked.

    The picker used to be pure client-side state that never left the browser, so
    the Optimize submenu, the analyzer cards, the persona-gated chip and the
    Dashboard tiles all keyed off the STORED report's own dominant persona.
    """
    # OptimizeView, the Dashboard band, and App()'s submenu effect.
    assert "api('/optimize', { since, agent_id: agentId || undefined, persona })" in html
    assert "api('/optimize', { since, fast: 'true', persona })" in html
    assert "api('/optimize', { fast: 'true', persona })" in html
    # Persona is a real refetch dependency in both readers.
    assert "}, [since, agentId, compare, persona]);" in html
    assert "}, [since, persona, armOptWait]);" in html


def test_the_blank_the_submenu_on_mismatch_workaround_is_gone(html: str) -> None:
    """The workaround could only BLANK the submenu when the stored report's
    persona differed from the selection — it had no way to compute the right
    entries, so a machine with data for both personas showed nothing at all for
    the non-dominant one. Threading the persona to the server replaces it."""
    assert "if (d.persona && d.persona !== persona) {" not in html
    assert "(thread the selected persona to /optimize) is deferred" not in html
    assert "is deferred." not in html


def test_a_persona_switch_does_not_keep_the_previous_personas_figures(
    html: str,
) -> None:
    """Stale-but-shown is the right call for a refresh and the wrong call for a
    persona switch: the numbers on screen answer a different question, so the
    surface must go back to not-yet-known rather than relabel them."""
    assert "const personaChanged = optPersonaRef.current !== persona;" in html
    assert "data: personaChanged ? null : s.data" in html


def test_an_unanswered_analyzer_is_a_third_state_not_an_empty_result(
    html: str,
) -> None:
    """`persona_unanswered_analyzers` — a lever this persona HAS, that the
    stored pass never ran. It must render as unresolved, never as "No
    candidates", which would be a clean bill of health nothing measured."""
    assert "opt.persona_unanswered_analyzers" in html
    assert "st.opt.persona_unanswered_analyzers" in html
    assert "const PERSONA_UNANSWERED_HINT =" in html
    # It resolves to the not-ready tile state, which is already excluded from
    # every published total (see totalOpportunityFigure).
    assert "{ name: k, state: 'not_ready', hint: PERSONA_UNANSWERED_HINT }" in html
    # And it gets its own detail-page branch, checked before the generic
    # "ran, found nothing" one.
    assert "if (personaUnanswered.has(detailName)) {" in html


def test_sessions_page_filters_both_zones_on_the_selected_persona(
    html: str,
) -> None:
    """The coding-session list was not persona-filtered at all: switching to
    "SDK workflows" left a screenful of Claude Code cards under an SDK heading.
    Both zones now gate, and neither gates on a not-yet-known persona."""
    # The SDK zone's rule moved into `showSdkZoneFor` so the render and the
    # history fetch that feeds it read ONE rule; pin the rule, not where it is
    # spelled inline.
    assert "const showSdkZone = showSdkZoneFor(persona, personaKnown);" in html
    assert "return !personaKnown || persona !== 'claude-code';" in html
    assert "const showCodingZone = !personaKnown || persona !== 'sdk';" in html
    assert "function StatusView({ params, persona, personaKnown, routeView })" in html


# --------------------------------------------------------------------------- #
# SDK session history — the SDK persona's historical surface.
#
# The live-services panel is bounded by SDK_DISCOVERY_WINDOW (7 days,
# api/routes/status.py). That is deliberate and stays. What was missing is what
# happens on the other side of it: the SDK persona's Sessions page rendered that
# panel and nothing else, so an agent quiet for longer than the window left the
# product, and the page said "No SDK services live right now" — a true statement
# about the live window, sitting alone on the persona's ONLY session surface,
# where it reads as "there is no record of your SDK work".
# --------------------------------------------------------------------------- #
def test_sdk_history_is_fed_by_the_persona_scoped_sessions_endpoint(
    html: str,
) -> None:
    """A different question needs a different source. History must NOT come from
    /status's live discovery — that is the very thing bounded by the window."""
    assert "function SdkSessionHistory(" in html
    assert "api('/sessions', { persona: 'sdk', limit: SDK_HISTORY_LIMIT })" in html
    assert "const SDK_HISTORY_LIMIT = 50;" in html
    # Rendered beneath the live panel, on the services tab, with the panel now
    # ALWAYS rendering rather than being swapped out for a bare sentence.
    assert "<${SdkServicesPanel} services=${sdkServices} framing=${data.framing} history=${sdkHistory} />" in html
    assert "<${SdkSessionHistory} history=${sdkHistory} framing=${data.framing} />" in html
    # The bare fallback that bypassed the panel entirely — and so WAS the whole
    # page in the reported case — must be gone.
    assert (
        ': html`<div class="svc-empty">No SDK services live right now</div>'
        not in html
    ), "the bare live-empty fallback must not bypass the panel"


def test_the_live_empty_claim_cannot_render_while_its_fetch_is_unresolved(
    html: str,
) -> None:
    """THE pin for this class of bug: three states, gated on the history read's
    OWN resolved flag, with the not-yet-known branch rendering a skeleton and
    making no claim of any kind.

    Order matters and is asserted: the unresolved branch must come FIRST, so a
    fall-through can never reach an absence claim.
    """
    fn = html[html.index("function sdkLiveEmptyState(history) {"):]
    fn = fn[: fn.index("\n}")]
    # Not-yet-known is the first branch, and it renders a shimmer, not a claim.
    assert fn.index("if (!history || !history.known)") < fn.index("shimmer")
    assert fn.index("shimmer") < fn.index("No SDK")
    # Known-and-empty is the ONLY place an absence claim lives, and it makes the
    # claim the reader needs (nothing recorded at all), not the narrower
    # live-window one.
    assert "if (!history.sessions.length)" in fn
    assert "No SDK or API traffic has been recorded on this machine yet" in fn
    # Known-with-history keeps the live-window fact but never leaves it alone.
    assert "No SDK services live right now. Earlier sessions are in the history below." in fn
    # A failed read must not degrade into "empty" — `known` stays false.
    assert ".catch(() => {});" in html


def test_sdk_history_skeleton_publishes_no_count(html: str) -> None:
    """The unresolved history list renders a table skeleton and no figure. A
    zero here would read as "no SDK sessions", which is the reassurance the
    unanswered read cannot support."""
    fn = html[html.index("function SdkSessionHistory({ history, framing }) {"):]
    fn = fn[: fn.index("\nfunction _flatlineSvg")]
    head = fn[: fn.index("if (!history.sessions.length) return null;")]
    assert "TableRowsSkeleton" in head
    assert "session${rows.length" not in head, "no count may render before the read answers"
    # The cap is disclosed rather than published as a total.
    assert "history.capped" in fn
    assert "capped: sessions.length >= SDK_HISTORY_LIMIT," in html


def test_one_rule_decides_whether_the_sdk_zone_renders(html: str) -> None:
    """The render and the fetch that feeds it must not disagree about whether
    the zone exists — stated once, read twice."""
    assert "function showSdkZoneFor(persona, personaKnown) {" in html
    assert "return !personaKnown || persona !== 'claude-code';" in html
    assert "const showSdkZone = showSdkZoneFor(persona, personaKnown);" in html
    assert "if (!showSdkZoneFor(persona, personaKnown)) return undefined;" in html


def test_no_view_references_a_binding_its_refactor_deleted(html: str) -> None:
    """Relocating alerts out of this page deleted `const showAlerts = ...` and
    left a reference to it inside `agentCard`.

    Why it survived every existing guard: the reference sits in a row that only
    evaluates when there IS an active coding card to draw, so a corpus with no
    active session never reached it. And Preact does not surface the
    ReferenceError — it swallows a re-render exception into its rerender queue,
    so the entire page silently stops updating with an EMPTY console. The
    symptom is a permanent skeleton, which reads as a slow fetch.

    Pinned as a name-level check because a static grep cannot do scope analysis:
    any identifier this file USES from that refactor must also be DECLARED in
    it. Add to `retired` whenever a binding is removed.
    """
    # Comments are stripped first so the check can stay blunt (the identifier
    # must not appear AT ALL) while the source is still free to explain the bug
    # by name — the explanation is worth more than the convenience of a looser
    # pattern that would miss the next reference shape.
    code = re.sub(r"/\*.*?\*/", "", html, flags=re.S)
    retired = ["showAlerts"]
    for name in retired:
        assert name not in code, (
            f"`{name}` was deleted when alerts moved into the Sessions SDK zone; "
            f"a surviving reference throws at render and silently freezes the "
            f"whole Preact tree"
        )
    # The alerts row now gates on a binding that IS declared on this page.
    assert "${a.active_alerts > 0 && showSdkZone ?" in html
    assert "const showSdkZone = showSdkZoneFor(persona, personaKnown);" in html


# --------------------------------------------------------------------------- #
# The persona picker has ONE position.
# --------------------------------------------------------------------------- #
def test_the_persona_picker_has_exactly_one_placement_mechanism(html: str) -> None:
    """It used to have two: an in-header slot for four named views, and a
    floated fallback bar for every other view. The fallback was `float: right`,
    so the same control sat beside the title on some pages and at the far right
    on others, and moved as the reader navigated.

    The fallback is DELETED rather than restyled. A control positioned in two
    places will drift in two places.
    """
    assert (
        "const PERSONA_PICKER_VIEWS = "
        "new Set(['dashboard', 'optimize', 'sessions', 'traces']);"
    ) in html
    # The float, its component, and the app-level render site are all gone.
    assert "function PersonaBar(" not in html
    assert "VIEWS_WITH_PERSONA_SLOT" not in html
    assert ".persona-bar { float: right" not in html
    assert "${ownsPersonaSlot ? null : html`<${PersonaBar} />`}" not in html


def test_the_picker_is_absent_from_views_the_persona_does_not_change(
    html: str,
) -> None:
    """A control that changes nothing is worse than a misplaced one: it invites
    the reader to believe the page responded to it. The converse is worse
    still, and is what this test used to enforce: a page whose CONTENT the
    persona changes, with no control on it to say so.

    The rule is one rule in both directions — does the persona change what the
    page SHOWS? — so this pins both halves. FAQ, Relearn and Summarize read no
    persona and render identically for either, so they take no `persona` prop
    and show no picker; that is asserted structurally, so the day one of them
    becomes persona-aware this test says the picker has to come with it.

    Traces was in that list and is not any more. `GET /traces` now takes a
    persona and scopes the rows, the total count and the outlier quartiles to
    it, so the page genuinely shows something different for each — and while it
    took no persona, this test was ENFORCING that it could not. Pinning it
    positively is what keeps that from silently coming back.
    """
    picker_decl = html[html.index("const PERSONA_PICKER_VIEWS"):][:200]
    for view in ("dashboard", "optimize", "sessions", "traces"):
        assert f"'{view}'" in picker_decl
    traces_at = html.index("function TracesListView(")
    assert "persona" in html[traces_at:][:80], (
        "TracesListView must take a persona: its list, its total count and its "
        "outlier rule are all scoped to one, and a page that filters by persona "
        "without a picker gives the reader no way to see or change what it did"
    )
    assert "<${PersonaPicker} bare=${true} />" in html[traces_at:traces_at + 8000], (
        "Traces is persona-scoped, so it renders the picker in its own header"
    )
    for fn in ("function AnalyzerGuideView(",
               "function RulesView(", "function SummarizeView("):
        sig = html[html.index(fn):][: len(fn) + 60]
        assert "persona" not in sig, f"{fn} now takes a persona; give it a picker"


# --------------------------------------------------------------------------- #
# Spend: one bar, total wide, overspend shaded inside it.
# --------------------------------------------------------------------------- #
def test_the_overspend_portion_cannot_render_while_its_fetch_is_unresolved(
    html: str,
) -> None:
    """THE pin. The bar's width is a live query; the shaded region comes from
    the stored analyzer report and can be cold long after. Two sources, two
    states, and only one of them may license a claim about waste.

    An unshaded bar reads as "no waste found" — the most reassuring thing this
    surface could say and the one it has least evidence for. So the unresolved
    state renders a skeleton bar with no region and no figure, never a
    zero-width portion and never a zero amount.
    """
    bar = html[html.index("      ${(() => {\n        const c = st.comp;"):]
    bar = bar[: bar.index("      ${/* ONE unified action list.")]
    # Gated on the overlay's OWN availability, not on a page-wide flag.
    assert "c.recoverable_available && c[overField] != null" in bar
    # The shaded region requires the resolved state AND a real figure. The gate
    # TIGHTENED from `recKnown` to `basisKnown`: a shaded width is a SHARE, and
    # a share needs a denominator over the same window and agents as the
    # ceiling (`recoverable_basis_*`). `recKnown` alone let a known ceiling be
    # drawn against whatever total the page happened to hold, which is the
    # mismatched-population defect. `basisKnown` implies `recKnown`, so this is
    # strictly stronger than what it replaced.
    assert "const basisKnown = recKnown && basis != null && basis > 0;" in bar
    assert "${basisKnown && over > 0 ? html`<span class=\"opt-spendbar-over\"" in bar
    # THE LOOSER GATE, pinned absent.
    assert "${recKnown && over > 0 ? html`<span class=\"opt-spendbar-over\"" not in bar
    # And it must be sized to the real proportion. Shipped once without this:
    # the CSS min-width (a legibility floor for a few-percent figure) was then
    # the ONLY width, so a 19% ceiling drew as 0.6% of the bar and the picture
    # contradicted the legend beside it.
    assert "style=${'width:' + overPct + '%'}" in bar
    # Only the WIDTH clamps. The ceiling sums overlapping estimates, so it can
    # legitimately exceed the window's measured spend; clamping the FIGURE
    # would print a value the data never produced under a measured label, and
    # would hide the one thing that case tells you about the overlap.
    assert "const over = recKnown ? Math.max(0, c[overField]) : 0;" in bar
    # The proportion is taken against the CEILING'S OWN basis, never the page's
    # total. `(over / total)` was the division that paired a 30-day estimate
    # with a 7-day total and shaded 73% of a week's spend.
    assert "Math.min(100, (over / basis) * 100)" in bar
    assert "(over / total)" not in bar
    assert "const overExceedsSpend = basisKnown && over > basis;" in bar
    assert "exceeds measured spend" in bar
    # A ceiling whose share cannot be computed still shows its FIGURE; only the
    # proportion is withheld. Suppressing a known number would be the opposite
    # failure to the one above.
    assert "share of spend unknown" in bar
    # Unresolved renders the skeleton class and the unknown glyph, not a zero.
    assert "'opt-spendbar' + (basisKnown ? '' : ' is-skeleton')" in bar
    assert "Overspend ceiling <b>${UNKNOWN_FIGURE}</b>" in bar
    # Comment prose names the forbidden literals to explain them, so the
    # check runs against code only.
    bar_code = re.sub(r"/\*.*?\*/", "", bar, flags=re.S)
    assert "$0.00" not in bar_code and ">0%<" not in bar_code


def test_the_bar_carries_its_own_overlap_disclosure(html: str) -> None:
    """`recoverable_additive` is false: the estimates overlap, so the total is a
    CEILING across analyzers, not a slice of spend anyone could bank. Drawn as a
    filled portion of a total it looks like a measured sub-amount, so the
    server's own note travels WITH the bar rather than living where the eye
    skips, and the legend says "ceiling" rather than naming a recoverable
    amount."""
    bar = html[html.index("      ${(() => {\n        const c = st.comp;"):]
    bar = bar[: bar.index("      ${/* ONE unified action list.")]
    assert "c.recoverable_overlap_note" in bar
    assert "not a slice of the total you could bank" in bar
    assert "Overspend ceiling" in bar
    # The one standalone-honest figure stays reachable.
    assert "largest_recoverable_analyzer" in bar
    # Both figures read the SAME basis; a local-pricing reader must not get a
    # dollar total formatted as tokens.
    assert "const totalField = useTokens ? 'total_tokens' : 'total_cost_usd';" in bar
    assert "const overField = useTokens ? 'total_recoverable_tokens' : 'total_recoverable_usd';" in bar
    # The retired segmented composition chart and its helpers are gone.
    assert "function buildComponentWaste(" not in html
    assert "function dominantSplit(" not in html
    assert "spendSegs" not in html


# --------------------------------------------------------------------------- #
# Session names show the project, not the raw prefixed agent id.
# --------------------------------------------------------------------------- #
def test_rendered_session_names_strip_the_tool_prefix_but_keep_the_id(
    html: str,
) -> None:
    """Display only. `agent_id` stays the identity for links, filters and dedup
    keys, and every rendered name carries it as a `title`."""
    assert "function agentName(row)" in html
    assert "row.agent_display_name || row.agent_id" in html
    # The stripping RULE is server-side (beside the prefix list that defines
    # what a coding agent is), so the UI cannot grow a second copy of it.
    # Comments are stripped first: the helpers explain themselves with a
    # worked example that names a prefix, and the explanation is worth more
    # than a looser pattern.
    code = re.sub(r"/\*.*?\*/", "", html, flags=re.S)
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("//"))
    helpers = code[code.index("function agentName(row)"):]
    helpers = helpers[: helpers.index("const SDK_HISTORY_LIMIT")]
    assert "claude-code" not in helpers, "the prefix rule must stay server-side"
    # Every list that renders a name uses the helper, and keeps the id on hover.
    assert '<td class="a-agent" title=${s.agent_id || \'\'}>${names.get(s.agent_id) || agentName(s)}</td>' in html
    assert '<td class="a-agent" title=${s.agent_id || \'\'}>${archNames.get(s.agent_id) || agentName(s)}</td>' in html
    assert "const displayName = a.label || cardNames.get(a.agent_id) || agentName(a);" in html
    assert "const title = s.label || s.agent_display_name || s.agent_id;" in html
    # Collisions are a LIST property, so they are resolved per list.
    assert "function disambiguateAgentNames(rows)" in html
    assert "const cardNames = disambiguateAgentNames(codingAgents);" in html
    assert "const archNames = disambiguateAgentNames(codingArchived);" in html


# --------------------------------------------------------------------------- #
# The active-compute tile falls back to tokens rather than a dash.
# --------------------------------------------------------------------------- #
def test_the_compute_tile_never_renders_a_zero_duration(html: str) -> None:
    """A sum of span durations that lands on zero means the spans recorded no
    duration, not that the session did no work. `0s` states the second."""
    assert "const hasActiveCompute = s.active_seconds != null && s.active_seconds > 0;" in html
    tile = html[html.index("      <div class=\"ses-tile\">\n        ${hasActiveCompute ? html`"):]
    tile = tile[: tile.index("</div>\n      <div class=\"ses-tile\">")]
    # The compute branch is the ONLY one carrying the compute label, and the
    # fallback's label names the figure it actually shows.
    assert "active (compute time)" in tile
    assert "tokens in / out" in tile
    compute_at = tile.index("active (compute time)")
    tokens_at = tile.index("tokens in / out")
    assert tile.index("fmtDurLong(s.active_seconds * 1000)") < compute_at
    assert tile.index("fmtTokens(s.input_tokens)") < tokens_at
    # The old unconditional dash is gone.
    assert "${s.active_seconds != null ? fmtDurLong(s.active_seconds * 1000) : '-'}" not in html


def test_the_picker_does_not_name_a_persona_it_has_not_read(html: str) -> None:
    """Not-yet-known is a third state for the CONTROL too, not just the nav.

    The nav now renders unsettled rows while the persona read is in flight. The
    picker beside it went on asserting the fallback ("Claude Code") for that
    whole window and swapped when the read landed, which on an SDK-dominant
    corpus names the wrong persona and then corrects itself. Two surfaces
    answering the same unresolved question differently is worse than either
    alone, so the picker takes the same treatment.
    """
    picker = html[html.index("function PersonaPicker("):]
    picker = picker[: picker.index("\nfunction ")]
    # It reads the settled flag at all.
    assert "known" in picker, "PersonaPicker must consult the persona-known flag"
    # It does not display a persona it has not read.
    assert "value=${known ? persona : ''}" in picker, (
        "the picker must not select a persona while the read is unresolved"
    )
    assert 'value=${persona} onChange' not in picker, (
        "the unconditional fallback value must not come back"
    )
    # It says so, rather than looking settled.
    assert "persona-select-pending" in picker
    assert "aria-busy=${known ? null : 'true'}" in picker
    # ...and it stays usable: an explicit pick is how a reader settles it early.
    assert "onChange=${(e) => onChange(e.currentTarget.value)}" in picker, (
        "the picker must stay enabled while unsettled"
    )
    assert "disabled" not in picker.split("<select")[1].split(">")[0], (
        "the SELECT itself must never be disabled; only the placeholder option is"
    )
    # The unsettled styling exists and matches the nav's.
    assert ".persona-select.persona-select-pending { opacity: 0.42; }" in html
    assert ".sidebar a.nav-link.nav-link-pending { opacity: 0.42; }" in html

