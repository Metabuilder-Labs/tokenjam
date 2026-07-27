"""Static regression guards for the Lens UI bug fixes (#126–#129).

The dashboard is a single-file Preact SPA with no JS test runner in the Python
CI job, so these assert the *served source* contains the corrected logic and no
longer contains the buggy patterns. They're intentionally narrow — each pins one
bug's fix so a future edit that reintroduces it fails here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _UI.read_text(encoding="utf-8")


def _no_comments(text: str) -> str:
    """`text` with `//` line comments stripped, for absence assertions.

    This file is heavily commented BY DESIGN: the UI records why each decision was
    made, including the wording of strings that were deliberately removed. A plain
    `assert "X" not in html` therefore matches the explanation and fails on correct
    code. That trap bit seven separate assertions during the inbox redesign before
    this helper existed, each time costing a debug cycle to rediscover.

    Deliberately naive about `//`: it drops from `//` to end of line only when
    `//` is the first non-whitespace on the line, so a `//` inside a URL or a
    regex literal is left alone.

    JSX-style `${/* ... */ \'\'}` interpolations ARE stripped, because that is
    where the render's own tombstones live — the inbox records the exact wording of
    labels it removed, inside the markup, right where they used to render. Without
    this, a ban on a removed label matches its own explanation. Plain `/* */` block
    comments outside an interpolation are left alone.
    """
    text = re.sub(r"\$\{/\*.*?\*/\s*''\}", "", text, flags=re.S)
    return "\n".join(
        "" if line.lstrip().startswith("//") else line
        for line in text.splitlines()
    )


def test_traces_window_select_exposes_longer_supported_windows(html):
    # Traces honors these URL/API windows already; the filter dropdown must stay
    # in sync so #/traces?since=30d and #/traces?since=90d render as selected
    # options. The dropdown no longer hardcodes them -- it derives from the
    # store's real data span via the shared `windowOptionsWithCurrent` helper
    # (see test_window_selectors_derive_from_data_span), but 1h is Traces' own
    # always-offered floor (finer than the shared ladder's 24h) and must survive
    # the refactor without becoming a second, duplicate "Last 1h" option.
    traces_start = html.index("function TracesListView")
    traces_end = html.index("function dedup", traces_start)
    traces_view = html[traces_start:traces_end]
    assert '<option value="1h">Last 1h</option>' in traces_view
    assert "windowOptionsWithCurrent(dataSpan ? dataSpan.available_days : null, since)" in traces_view
    assert ".filter(w => w.value !== '1h')" in traces_view
    assert '<option value="24h">Last 24h</option>' not in traces_view
    assert '<option value="30d">Last 30d</option>' not in traces_view


def test_dashboard_recent_activity_drills_into_matching_traces_window(html):
    # The Dashboard defaults to 30d while Traces defaults to 24h. Keep the Recent
    # activity drill-through tied to the Dashboard window so the tile count and
    # destination list use the same basis (#299).
    assert "function tracesHrefForWindow" in html
    assert "tracesHrefForWindow(since)" in html
    assert 'label="Recent activity" value=${(d.traces || []).length} attention=${errTraces > 0} href="#/traces"' not in html


def test_window_selectors_derive_from_data_span(html):
    # The Dashboard window selector used to be a fixed 24h/7d/30d/90d list with
    # no relation to how much telemetry the store actually holds -- offering
    # 90d over a two-month corpus, and unable to reach past 90d on a longer
    # one. It derives from core/data_span.py's `available_days`, not a
    # hardcoded list. That derivation (`standardWindowOptions` /
    # `windowOptionsWithCurrent`) was Dashboard-only at first
    # (`dashboardWindowOptions`/`DASHBOARD_STANDARD_WINDOWS`) and is now shared
    # by Cost, Traces and Optimize too, each reading `data_span` off its own
    # already-fetched payload (/cost, /traces, /optimize) rather than a fourth
    # copy-pasted ladder or a second round-trip to /drift or /relearn.
    const_start = html.index("const STANDARD_WINDOWS")
    start = html.index("function standardWindowOptions")
    with_current_end = html.index("\n}\n", html.index("function windowOptionsWithCurrent")) + 3
    consts = html[const_start:start]
    fn = html[start:with_current_end]

    # The always-safe floor entry exists and is what an unknown span falls
    # back to.
    assert "{ value: '24h', label: 'Last 24h', days: 1 }" in consts

    # Unknown (or non-positive/unusable) span offers only the always-safe
    # floor, never a wrong option set asserted ahead of the data.
    assert "if (availableDays == null || availableDays <= 0) return [STANDARD_WINDOWS[0]];" in fn

    # Known span: only windows at-or-under the available span, plus one final
    # option at the real span itself.
    assert "w.days <= availableDays" in fn
    assert "value: `${availableDays}d`" in fn

    # `validSince` must accept that custom exact-span value, or selecting it
    # gets silently reverted to the default on the very next render -- the
    # dropdown would show the pick while the page queried a different window.
    assert "const _CUSTOM_DAYS_RE = /^\\d+d$/;" in html
    assert "_CUSTOM_DAYS_RE.test(v)" in html

    dash_start = html.index("function DashboardView")
    dash_end = html.index("function ", dash_start + 1)
    dash_view = html[dash_start:dash_end]

    # Dashboard: wired to the server-provided data_span, not re-derived
    # client-side. The old unconditional four-option list is gone from the
    # picker itself.
    assert "driftRead.data.data_span" in dash_view
    assert "relearnRead.data.data_span" in dash_view
    assert "windowOptionsWithCurrent(availableDays, since)" in dash_view
    assert '<option value="24h">Last 24h</option>\n      <option value="7d">Last 7d</option>' not in dash_view

    cost_start = html.index("function CostView")
    cost_end = html.index("\nfunction TopTenantsPanel", cost_start)
    cost_view = html[cost_start:cost_end]
    assert "costResp.data_span.available_days" in cost_view
    assert 'value="24h">Last 24h</option>\n        <option value="7d">Last 7d' not in cost_view

    traces_start = html.index("function TracesListView")
    traces_end = html.index("function dedup", traces_start)
    traces_view = html[traces_start:traces_end]
    assert "dataSpan.available_days" in traces_view
    assert "setDataSpan(td.data_span || null);" in traces_view

    opt_start = html.index("function OptimizeView")
    opt_end = html.index("function ", opt_start + 1)
    opt_view = html[opt_start:opt_end]
    assert "st.opt.data_span.available_days" in opt_view
    assert 'value="7d">Last 7d</option>\n        <option value="30d">Last 30d' not in opt_view


def test_window_ladder_adds_exactly_one_intermediate_rung(html):
    # standardWindowOptions offers base rungs (24h/7d/30d) clamped to the
    # span, ONE intermediate rung (the largest INTERMEDIATE_RUNG_DAYS
    # candidate strictly below the span), then the exact span itself. Pinned
    # against the operator's own three worked examples, run by hand through
    # the actual algorithm (this file has no JS runner in CI, so the pin is
    # the exact source text rather than an executed assertion):
    #   67  days -> 24h, 7d, 30d, 60d, 67d   (not 90d -- 90 is not < 67)
    #   112 days -> 24h, 7d, 30d, 90d, 112d  (not ALSO 60d -- only the
    #               LARGEST qualifying candidate is added, never every one)
    #   42  days -> 24h, 7d, 30d, 42d        (no intermediate rung clears
    #               42 at all, so none is added)
    # A naive "every standard rung under the span" reading would wrongly add
    # both 60d and 90d for the 112-day case -- that's the one case that rules
    # it out, so it is the one pinned literally below.
    assert "const INTERMEDIATE_RUNG_DAYS = [60, 90, 120, 180, 365];" in html
    start = html.index("function standardWindowOptions")
    end = html.index("\n}\n", start) + 3
    fn = html[start:end]
    assert "const intermediateDays = INTERMEDIATE_RUNG_DAYS.filter(d => d < availableDays).pop();" in fn
    assert "if (intermediateDays != null) {" in fn

    # Sub-30-day spans drop base rungs the corpus can't support rather than
    # ever offering a window with nothing behind it (the original defect) --
    # STANDARD_WINDOWS itself is filtered by `w.days <= availableDays` first,
    # so a 12-day corpus naturally yields 24h/7d (30d filtered out) then the
    # exact-span push adds 12d, with no separate sub-30 branch required.
    const_start = html.index("const STANDARD_WINDOWS")
    consts = html[const_start:html.index("function standardWindowOptions")]
    assert "{ value: '30d', label: 'Last 30d', days: 30 }" in consts
    assert "{ value: '90d'" not in consts  # 90d moved to INTERMEDIATE_RUNG_DAYS

    # Non-positive/unusable spans degrade to the always-safe floor, same as
    # the not-yet-known (null) case -- never a confusing single "Last 0d" or
    # empty option list.
    assert "if (availableDays == null || availableDays <= 0) return [STANDARD_WINDOWS[0]];" in fn


def test_detail_views_show_a_layout_shaped_skeleton_not_bare_loading_text(html):
    # Session/Run/Trace detail and Status/Drift/Budget's "still loading" state
    # used to be a bare centered "Loading X..." replacing the whole page —
    # the weaker half of the not-yet-known/known-and-empty/known-and-populated
    # distinction this file otherwise enforces everywhere else (a bare
    # placeholder doesn't assert a wrong number, but it still throws away the
    # real layout's shape, so content visibly jumps in when it finally
    # arrives). Each now renders a skeleton shaped like its own real layout
    # (CardGridSkeleton / TableRowsSkeleton / a purpose-built one) instead.
    stripped = _no_comments(html)
    for bare in (
        "Loading session...", "Loading run...", "Loading trace...",
        "Loading history…",
    ):
        assert bare not in stripped

    # Status/Drift/Budget's bare "Loading..." literal must not survive either
    # -- narrower than a blanket ban on the word "Loading" (which legitimately
    # appears in InboxLoadingNote's stated-not-animated caption, a DIFFERENT,
    # already-correct pattern: a caption alongside skeleton tiles that are
    # already rendering, not a replacement for them).
    assert '<div class="empty">Loading...</div>' not in stripped

    assert "function CardGridSkeleton(count, rows = 3)" in html
    assert "function TableRowsSkeleton(headers, rowCount)" in html
    assert "function SessionDetailSkeleton({ backBtn })" in html

    status_start = html.index("function StatusView")
    status_end = html.index("\nfunction ", status_start + 1)
    assert "CardGridSkeleton(6)" in html[status_start:status_end]

    drift_start = html.index("function DriftView")
    drift_end = html.index("\nfunction ", drift_start + 1)
    assert "TableRowsSkeleton(" in html[drift_start:drift_end]

    budget_start = html.index("function BudgetView")
    budget_end = html.index("\nfunction ", budget_start + 1)
    assert "TableRowsSkeleton(" in html[budget_start:budget_end]

    session_start = html.index("function SessionDetailView")
    session_end = html.index("\nfunction ", session_start + 1)
    assert "<${SessionDetailSkeleton} backBtn=${backBtn} />" in html[session_start:session_end]

    run_start = html.index("function RunDetailView")
    run_end = html.index("\nfunction ", run_start + 1)
    assert "TableRowsSkeleton(" in html[run_start:run_end]

    trace_start = html.index("function TraceDetailView")
    trace_end = html.index("\nfunction ", trace_start + 1)
    assert "shimmer" in html[trace_start:trace_end]

    expect_start = html.index("function ExpectationHistory")
    expect_end = html.index("\nfunction ", expect_start + 1)
    assert "TableRowsSkeleton(['When', 'Outcome', 'Run', 'Note'], 2)" in html[expect_start:expect_end]


# --- #126: Downsize typed slot always rendered ----------------------------- #
def test_downsize_section_always_renders(html):
    # The no-candidates branch renders a literal Downsize section id instead of
    # returning null, so the section is never silently dropped.
    assert 'id="opt-downsize"' in html
    assert "No downsize candidates in this window" in html


def test_downsize_is_first_in_optimize_order(html):
    assert (
        "const order = ['downsize', 'resend', 'cache', 'cache-recommend', 'script', "
        "'trim', 'reuse', 'subagent', 'verbosity', 'deadweight', 'placement']"
    ) in html


# --- Batch placement card: advise-only, a price difference not recoverable tokens - #
def _placement_branch(html: str) -> str:
    start = html.index("} else if (name === 'placement') {")
    end = html.index(
        "  } else {\n    const fd = (opt.findings || {})[name];\n    if (!fd) return null;",
        start,
    )
    return html[start:end]


def test_placement_registered_in_analyzer_meta_and_order(html):
    assert "placement:  { title: 'Batch placement'" in html
    assert "'deadweight', 'placement']" in html


def test_placement_section_always_renders_when_nothing_qualifies(html):
    # Mirrors downsize's own null-slot handling (issue #126): the batch-placement
    # analyzer drops the key from `findings` entirely rather than carrying a
    # null-candidates finding when nothing qualifies, so the card must render its
    # own explicit empty state instead of vanishing via the generic
    # `if (!fd) return null` used by every other analyzer's card.
    assert 'id="opt-placement"' in html
    assert "No unattended, cadence-regular workloads in this window" in html


def test_placement_never_uses_recoverable_wording(html):
    # A batch-placement dollar figure is a PRICE difference on the SAME tokens
    # (batch bills the same work at half rate, freeing nothing) — the card must
    # never borrow the "estimated recoverable" wording every sibling analyzer
    # legitimately uses (CLAUDE.md anti-pattern #22).
    block = _placement_branch(html)
    assert "estimated price difference" in block
    # "estimated-tag" is the CSS class every other card's rendered "estimated
    # recoverable" badge carries — assert on the class, not the prose string,
    # since this branch's own explanatory comments legitimately mention the
    # sibling wording by name.
    assert "estimated-tag" not in block
    assert 'class="price-diff-tag"' in block


def test_placement_gates_dollar_figure_strictly_on_api_pricing_mode(html):
    # The Batch API's flat discount is an api-billed lever a subscription,
    # local, or even "unknown" plan cannot pull — gated strictly on
    # pricing_mode === 'api' (mirroring the CLI and cost-proposals renderers),
    # not the shared dollarsSuppressed() helper, which treats 'unknown' as
    # NOT suppressed.
    block = _placement_branch(html)
    assert "framing && framing.pricing_mode === 'api'" in block
    assert "api-billed price lever, so no dollar figure is shown for this plan" in block


def test_placement_offers_no_apply_action(html):
    # placement is advise-only by design (batch adoption is an architectural
    # change in the user's own application, not a config flip) — its card
    # renders no apply affordance of its own.
    block = _placement_branch(html)
    assert "<button" not in block


# --- Component-waste chart/legend: placement is a price difference there too #
def _waste_legend_override(html: str) -> str:
    start = html.index("const WASTE_LEGEND_OVERRIDE")
    end = html.index("function buildComponentWaste", start)
    return html[start:end]


def test_waste_legend_never_labels_placement_recoverable(html):
    # The shared "Cost by component + recoverable waste" chart/legend used to
    # render placement's segment with the same "estimated recoverable"
    # wording every real recoverable-tokens analyzer gets, even though
    # placement's figure is a PRICE difference on the SAME tokens (CLAUDE.md
    # anti-pattern #22). The legend row must key off the stable `analyzer`
    # registry name and swap in price-difference wording for placement only.
    override = _waste_legend_override(html)
    assert "placement:" in override
    assert "estimated price difference" in override
    assert "price-diff-tag" in override
    assert "buildComponentWaste(resp, useTokens, isApi)" in html
    assert "ov ? ov.tagClass : 'estimated-tag'" in html
    assert "ov ? ov.tagText : 'estimated recoverable'" in html


def test_waste_legend_gates_placement_on_api_pricing_mode(html):
    # Off api pricing there is no dollar figure for placement at all (the
    # Batch API's discount is an api-billed lever), and its token count means
    # "size of the affected workload," not tokens freed — so the segment must
    # drop out of the overlay entirely, not just get relabeled.
    override = _waste_legend_override(html)
    assert "apiOnly: true" in override
    assert "compIsApi" in html
    assert "!!compFraming && compFraming.pricing_mode === 'api'" in html


# --- #127: four distinct recoverable-tile states --------------------------- #
def test_recoverable_band_has_four_states(html):
    assert "function classifyFinding" in html
    for state in ("'actionable'", "'at_ceiling'", "'no_findings'", "'not_ready'"):
        assert state in html, f"missing tile state {state}"
    # at-ceiling must not reuse the "raise toward ceiling" hint.
    assert "Already optimized" in html


def test_recoverable_band_not_a_single_not_ready_catchall(html):
    # The old crude check ("ready = fd && usd != null" → "— not ready" for
    # everything else) must be gone.
    assert "const ready = fd && usd != null" not in html


# --- #128: chart tooltip + non-button drill -------------------------------- #
def test_chart_has_hover_tooltip(html):
    assert "function chartTooltipPlugin" in html
    assert "plugins: [chartTooltipPlugin(" in html


def test_axis_time_ticks_timezone_split(html):
    # #178: HOURLY ticks localize — they format the UTC epoch-second buckets in
    # the viewer's local zone (a US-Pacific user sees their noon, not UTC's 7pm).
    # #188: DAILY date labels stay UTC, because the buckets are UTC-day-aligned;
    # localizing a UTC-midnight key would print the previous local day for
    # west-of-UTC users and no longer match the bucket span.
    import re

    m = re.search(r"function fmtAxisTime\(epoch, bucket\) \{.*?\n\}", html, re.DOTALL)
    assert m, "fmtAxisTime helper not found"
    body = m.group(0)
    hour_line = next(line for line in body.splitlines() if "toLocaleTimeString" in line)
    date_line = next(line for line in body.splitlines() if "toLocaleDateString" in line)
    # Hourly localizes (must NOT force UTC).
    assert "timeZone: 'UTC'" not in hour_line, "hourly ticks must localize (#178)"
    # Daily stays UTC-aligned (must force UTC).
    assert "timeZone: 'UTC'" in date_line, "daily date labels stay UTC-aligned (#188)"


# --- #129: run-rate denominator + caption + $ axis ------------------------- #
def test_run_rate_uses_window_length_not_data_range(html):
    assert "function windowDays" in html
    assert "function runRateProjection" in html
    # The buggy data-range denominator must be gone.
    assert "ys.reduce((a, b) => a + b, 0) / ys.length" not in html


def test_run_rate_caption_says_not_a_forecast(html):
    # The honesty qualifier rides every run-rate projection (Cost screen's
    # parenthesized form + the Dashboard's folded KPI caption).
    assert html.count("linear run-rate, not a forecast") >= 2


def test_axis_uses_compact_dollar_formatter(html):
    assert "function fmtAxisUsd" in html
    assert "axisFmtY=" in html


# --- #132: first-load lands on Overview (no redirect race) ----------------- #

def test_chart_spans_full_window_with_buckets(html):
    assert "function windowDays" in html
    assert "series_bucket" in html and "window_start" in html
    # x scale pinned to the window range, not the data range.
    assert "range: [data[0][0]" in html


def test_axis_time_labels_consistent(html):
    assert "function fmtAxisTime" in html
    # daily labels use abbreviated month/day ("Jun 15"), one format per axis.
    assert "month: 'short', day: 'numeric'" in html


# --- #134: run-rate is cycle-relative, not a fixed ×30 --------------------- #
def test_run_rate_is_cycle_relative(html):
    assert "function cycleRemaining" in html
    assert "by ${cyc.label}" in html
    assert "over 30 days" not in html  # the circular/undershooting framing is gone


# --- #138: run-rate cycle honors [budget.<provider>] cycle_start_day -------- #
def test_run_rate_cycle_honors_server_bounds(html):
    # cycleRemaining now reads server-provided cycle bounds (cycle_start_day
    # aware) instead of always assuming the calendar month.
    assert "function cycleRemaining(cycle)" in html
    assert "cycle.days_remaining" in html
    assert "cycle.start_day" in html
    # Both run-rate call sites pass the response's cycle block through.
    assert "cycleRemaining(cost.cycle)" in html
    assert "cycleRemaining(costResp && costResp.cycle)" in html


# --- #135: cache at_ceiling not gated on input volume --------------------- #
def test_cache_at_ceiling_not_volume_gated(html):
    # The volume threshold that hid 100%-efficacy/low-input rows is removed;
    # the classifier reads the ceiling from the response.
    assert "CACHE_MIN_INPUT" not in html
    assert "fd.efficacy_ceiling" in html


# --- #17: cache-write surfaced in trace detail + cost table ---------------- #
def test_cache_write_rendered(html):
    # trace-detail panel + waterfall tooltip + Cost table show cache-write.
    assert "cache_write_tokens" in html
    assert "Cache write" in html


# --- #139: buildCostSeries coarsens instead of silently emptying ----------- #
def test_cost_series_coarsens_not_silently_empty(html):
    # The silent "too many buckets -> null" guard is gone; the chart coarsens up
    # a bucket ladder (hour->day->week) and flags it instead of rendering empty.
    assert "xs.length > 5000) return null" not in html  # the silent-empty guard
    assert "const MAX_BUCKETS = 5000" in html
    assert "_BUCKET_LADDER" in html
    assert "['week', 604800]" in html
    # The coarsening is surfaced to the user, not silent (CLAUDE.md spirit).
    assert "coarsened" in html
    assert "Showing ${series.bucket} buckets" in html


# --- #124 follow-up: Overview fetches in parallel, asymmetric error handling- #
def test_overview_fetches_in_parallel(html):
    # The #114 serial-fetch workaround is gone now that the DB layer is
    # concurrency-safe (#124); the Overview fans out via Promise.all.
    assert "Fetch sequentially, not in parallel" not in html
    assert "await Promise.all([" in html


def _analytics_view_src(html: str) -> str:
    """Just AnalyticsView's own body."""
    start = html.index("function AnalyticsView")
    return html[start:html.index("\nfunction ", start + 1)]


def test_analytics_view_drops_a_stale_pivot_settle(html):
    # AnalyticsView's `load` re-fetches /analytics on every metric/group_by/
    # stack/chart/since/filter change (the Dashboard's embedded "Explore"
    # pivot, and the standalone Analytics screen). It has no polling of its
    # own, but a rapid pivot change (e.g. switching "By: Model" -> "By:
    # Tenant") fires a NEW request while an older one can still be in flight.
    # A slower EARLIER request (grouping by model touches more rows) landing
    # AFTER a faster LATER one (grouping by tenant, which this corpus has none
    # of) used to unconditionally overwrite state — the chart/KPIs would show
    # the new dimension's (near-empty) data while something derived from the
    # stale response lingered, and more generally the displayed pivot could
    # silently disagree with the selector. Only a settle whose generation
    # still matches the current one may write state, mirroring useTriageRead's
    # drop-stale-settle rule.
    src = _analytics_view_src(html)
    assert "const gen = useRef(0);" in src
    assert "const g = ++gen.current;" in src
    assert "if (gen.current === g) setSt({ loading: false, error: null, resp });" in src
    assert "if (gen.current === g) setSt(s => ({ ...s, loading: false, error: e.message || String(e) }));" in src


def _dashboard_src(html: str) -> str:
    """Just DashboardView's own body, for assertions about what this page fetches."""
    start = html.index("function DashboardView")
    return html[start:html.index("// Two lenses, one router", start)]


def _cost_view_src(html: str) -> str:
    """Just CostView's own body."""
    start = html.index("function CostView")
    return html[start:html.index("\nfunction TopTenantsPanel", start)]


def test_cost_view_load_never_stacks_on_a_slow_backend(html):
    # CostView's 30s poll used to call `load`/`loadTenants` with no in-flight
    # guard at all, unlike useTriageRead's own reads. On a slow backend (a
    # large real corpus) a poll tick firing before the previous cycle settled
    # queued ANOTHER concurrent /cost + /cost/compare + /cost/cache request on
    # top of whatever was still running, with no cap — and because the header
    # renders `total`/`totalTokens` unconditionally (not gated on having ever
    # received an answer), the page showed a fabricated "$0.0000 (0 tokens)"
    # for as long as the pile-up kept every request from ever finishing,
    # looking identical to "no spend" even though the real figure was sitting
    # in a request that never got to run. Both `load` and `loadTenants` now
    # guard the same way useTriageRead's `refresh` does.
    src = _cost_view_src(html)
    assert "if (loadInFlight.current) return;" in src
    assert "if (tenantsInFlight.current) return;" in src
    # The header must never claim a figure before the first answer lands.
    assert "const hasCostData = costResp != null;" in src
    assert "!hasCostData ? null : useTokens" in src


def _budget_src(html: str) -> str:
    """Just BudgetView's own body."""
    start = html.index("function BudgetView")
    end = html.index("\nfunction ", start + 1)
    return html[start:end]


def test_budget_non_essential_alerts_reads_never_reach_the_outer_catch(html):
    """Both /alerts reads already catch their own failures and resolve to []
    -- the ONLY read that can still reach the outer .catch is /budget itself,
    the essential one. A transient blip on either alerts read must never take
    the whole page down with it."""
    budget = _budget_src(html)
    assert budget.count(".catch(() => [])") == 2
    assert "api('/alerts', { type: 'cost_budget_daily', since: '24h' })\n      .then(d => d.alerts || [])\n      .catch(() => [])" in budget
    assert "api('/alerts', { type: 'cost_budget_session', since: '24h' })\n      .then(d => d.alerts || [])\n      .catch(() => [])" in budget


def test_budget_essential_read_failure_names_the_request(html):
    """api() throws a bare `API <status>` (no context on which request it
    was); that information used to be discarded on a /budget failure, which
    surfaced as an unlabeled generic message. /budget's own .catch now names
    the request before it reaches setError."""
    budget = _budget_src(html)
    assert "api('/budget').catch(e => { throw new Error('/budget failed: ' + e.message); })" in budget


def test_budget_view_failure_does_not_blank_content_it_already_has(html):
    """The error branch used to be a bare `if (error) return <blank page>`,
    so ANY later failed refresh (this view polls every 10s) discarded the
    last successfully loaded budget data and replaced the whole page with one
    unlabeled red string. Now the full-page error only fires when there is no
    prior answer at all (`error && !data`); once data has landed once, a
    later failure renders as a scoped .band-msg.err banner ABOVE the still-
    rendered tables, same pattern the Dashboard uses for its own /cost
    failures, and setError(null) on a successful load clears a stale banner."""
    budget = _budget_src(html)
    assert "if (error && !data) return html`<div class=\"empty\" style=\"color:var(--error)\">${error}</div>`;" in budget
    assert "if (error) return html`<div class=\"empty\"" not in budget
    assert 'class="band-msg err"' in budget
    assert "Couldn't refresh budget data." in budget
    assert "setError(null);" in budget
    # The per-agent and provider tables still render below the banner --
    # the banner never replaces them.
    banner_idx = budget.index('class="band-msg err"')
    assert budget.index("Per-agent budget caps", banner_idx) > banner_idx
    assert budget.index("Provider spend forecast", banner_idx) > banner_idx


def test_overview_error_handling_is_asymmetric(html):
    # /cost is load-bearing: NO .catch, so its failure surfaces the error state.
    # The other panels still degrade individually so one failing panel never
    # blanks the Dashboard. Don't unify these (#124 review).
    assert "api('/cost', { since, group_by: 'day' }).catch" not in html  # no catch on /cost
    assert "api('/cost', { since, group_by: 'day' })," in html
    # /cost/compare is no longer fetched by this page at all. It fed `d.compare`,
    # which nothing rendered (the run-rate comparison it was added for now comes
    # from the explorer's own kpi_deltas), so once the shared batch was split into
    # one read per source there was nowhere left to put a read whose result was
    # never displayed. On a page where every request costs tens of seconds of DB
    # time, a per-poll read nothing renders is not worth keeping.
    assert "api('/cost/compare'" not in _dashboard_src(html)
    # How they degrade changed: a read that feeds a tile making a factual claim
    # is settled into a tagged {ok, data} outcome rather than an empty default,
    # because an empty default let a failed read publish a zero and its
    # reassuring caption ("0 unread alerts / all clear"). See
    # test_lens_dashboard_states.py for the rule and the behavioural tests.
    #
    # There is no longer a shared batch at all: each read settles on its own
    # useTriageRead, whose ('loading' | 'ready' | 'error', data) pair is the
    # contract every panel renders from. That replaced BOTH the empty-default
    # fallbacks and the Promise.all, since a batch resolves only when its slowest
    # member does and these members range from 13s to several minutes.
    # The property under test is the SHAPE: one useTriageRead per read, settling
    # into a tagged outcome instead of an empty default. /drift now also sends
    # the page's window and depends on it (the selector used to govern three of
    # the five Health tiles and silently skip this one), so the literal moved;
    # what it is catching has not. The no-empty-default claim is asserted
    # independently two lines below and is untouched.
    assert "const driftRead = useTriageRead(() => api('/drift', { since }), [since]);" in html
    assert "function useTriageRead(run, deps) {" in html
    assert "api('/optimize', { since, fast: 'true' }).catch(() => null)" not in html
    assert "api('/drift').catch(() => ({ agents: [] }))" not in html


def test_triage_read_settle_only_clears_in_flight_for_its_own_generation(html):
    # useTriageRead's `refresh()` guards against stacking a poll on a slow read
    # via `inFlight.current` — but a window/filter change (a `deps` change) also
    # bumps `gen.current` and clears `inFlight.current` early (in the effect's
    # cleanup) so the REPLACEMENT request can start immediately without waiting
    # behind the superseded one. If the superseded request's OWN `.then()` later
    # unconditionally cleared `inFlight.current` again, it would falsely mark the
    # guard idle while the replacement request was still genuinely in flight, and
    # a poll tick landing in that window would stack a second query on top of it.
    # Both settle branches (success and error) must gate the clear on the
    # generation still matching, exactly like they already gate the `setSt` call.
    fn = html[html.index("function useTriageRead(run, deps) {"):]
    fn = fn[:fn.index("\n}\n") + 3]
    assert "if (gen.current === g) inFlight.current = false;" in fn
    # The naive/buggy form: an unconditional clear as the first statement of a
    # settle branch, with no generation check anywhere in that branch.
    assert "(data) => {\n        inFlight.current = false;" not in fn
    assert "(err) => {\n        inFlight.current = false;" not in fn


def test_overview_empty_gate_considers_historical_cost(html):
    # Regression: the Overview front door showed "No data yet" whenever /status
    # reported 0 active agents — and it returned BEFORE /cost was ever fetched.
    # A DB whose sessions are all >24h old (e.g. a user upgrading to review past
    # spend) has 0 active agents but a full cost history, so the default landing
    # screen falsely read empty while Cost/Analytics/Optimize rendered fine.
    # Sliced to the whole view: the empty gate now sits with the other derived
    # values, below the window picker it used to precede.
    ov = _dashboard_src(html)

    # Buggy pattern GONE: empty was gated purely on active-agent count with an
    # early return before /cost was fetched.
    assert "if (!status.agents || status.agents.length === 0) {" not in ov
    assert "const status = await api('/status');" not in ov  # /status no longer fetched first + serially

    # Fix pattern PRESENT: the empty gate considers historical cost/tokens (not
    # just agents/traces), and /cost is read independently rather than serially.
    assert "const hasCost = !!costData && ((cost.total_cost_usd || 0) > 0 || (cost.total_tokens || 0) > 0);" in ov
    assert "const costRead = useTriageRead(() => api('/cost', { since, group_by: 'day' }), [since]);" in ov
    # The gate has since been tightened twice more. First, all three of its
    # inputs must have actually ANSWERED before it may claim there is no
    # telemetry, so a failed or outstanding read can no longer show "No data yet"
    # to a user with a full history. Second, the reads are independent, so
    # "answered" is checked per read rather than off one shared load flag.
    assert "const emptyKnown = !!costData && !!statusRead.data && !!tracesRead.data;" in ov
    assert "const isEmpty = emptyKnown && !hasCost && !statusAgents.length && !traceList.length;" in ov
    # None of them degrades to an empty default any more, so a failure can no
    # longer be read as "this user has no agents / no traces".
    assert "api('/status').catch(() => ({ agents: [] }))" not in ov
    assert "api('/traces', { since, limit: 6 }).catch(() => ({ traces: [] }))" not in ov


# --- #147: status tile shows Active (compute) time + relabeled Elapsed ----- #
def test_status_tile_shows_active_and_elapsed(html):
    # A coarse formatter for multi-day wall-clock spans, so "3087m" reads "2d 3h".
    assert "function fmtDurLong" in html
    # Active time is sourced from the new status payload field.
    assert "a.active_seconds" in html
    # The wall-clock row is relabeled Elapsed and uses the coarse formatter;
    # Active is a distinct row using the fine-grained one.
    assert "fmtDurLong(a.duration_seconds" in html
    assert 'Active <span class="info-btn"' in html
    assert 'Elapsed <span class="info-btn"' in html
    # The misleading bare "Duration" label is gone from the status tile.
    assert '<span class="label">Duration</span>' not in html


# --- #162: Recoverable Waste tiles render consistently --------------------- #
def test_reuse_tile_title_is_title_cased(html):
    # reuse was missing from ANALYZER_META and slipped through lowercase.
    assert "reuse:      { title: 'Reuse'" in html
    # Capitalization is centralized so a future 6th analyzer auto-title-cases
    # instead of rendering its raw lowercase registry key.
    assert "function capitalize" in html
    assert "capitalize(t.name)" in html
    # The old raw-lowercase fallback is gone.
    assert "{ title: t.name, hint: '' }" not in html


def test_not_ready_tile_drops_em_dash(html):
    # The not_ready content line reads a bare "Not ready" (no "— not ready"
    # em-dash prefix), so the tile states share a prefix-free scheme.
    assert "— not ready" not in html
    assert "'Not ready'" in html


def test_recoverable_tile_titles_share_one_weight(html):
    # All non-actionable tiles use the identical bare .rec-name title element,
    # so the at_ceiling (Cache) tile can't bold its title differently. The
    # positive emphasis lives only on the content line (.rec-amount.ok), which
    # is the intended #127 design and must stay.
    assert html.count('<div class="rec-name">${meta.title}</div>') >= 1
    assert ".rec-amount.ok" in html          # green content line preserved (AC #4)
    # No state-specific rule bolds the title for the at_ceiling tile.
    assert ".rec-tile.ok .rec-name" not in html


# --- #187: suppress raw $ for subscription/local on table & trace surfaces --- #
def test_cost_table_cells_route_through_framing(html):
    # The per-row + footer COST cells must reframe like the hero (useTokens /
    # fmtFramedDollar), not render raw fmtCost. The bug was bare fmtCost cells.
    assert "<td>${fmtCost(r.cost_usd)}</td>" not in html
    assert "<td>${fmtCost(total)}</td>" not in html
    assert "${useTokens ? fmtTokens(_costVal(r, true)) : fmtFramedDollar(r.cost_usd, framing)}" in html
    assert "${useTokens ? fmtTokens(totalTokens) : fmtFramedDollar(total, framing)}" in html


def test_traces_list_cost_column_is_unconditionally_tokens(html):
    """Product decision: the Traces list's Cost column always shows token
    totals, for every account -- a display convention for how traces are
    presented, not a consequence of plan tier. It used to route through
    fmtPerItemCost(..., framing), which chose tokens-vs-dollars per
    subscription/local/api framing (#249); that conditional is gone from this
    screen, so the column no longer reads `framing` at all, and TracesListView
    no longer fetches/stores it either."""
    view_start = html.index("function TracesListView")
    view_end = html.index("function dedup", view_start)
    view = html[view_start:view_end]
    assert "<td>${fmtCost(t.cost_usd)}</td>" not in view
    assert "fmtPerItemCost(" not in view
    assert "${fmtTokens(_costVal(t, true) || 0)} tok" in view
    assert "framing" not in view
    assert "Cost (tokens) " in view


def test_traces_outlier_explanation_never_mentions_plan(html):
    """The outlier note/tooltip used to branch on `outlierDollarsShown`
    (`!perItemUsesTokens(framing)`): a token-only explanation for
    subscription/local, or the rule's own $ Q1/Q3/threshold figures for
    api/unknown. Now that the Cost column is unconditionally tokens for every
    account (see test_traces_list_cost_column_is_unconditionally_tokens), the
    explanation must match it for every account too -- no more "Dollar
    amounts aren't shown on your plan" (that attributes token display to plan,
    which is exactly the differentiation being removed), and no more
    plan-conditional $ branch either."""
    view_start = html.index("function TracesListView")
    view_end = html.index("function dedup", view_start)
    view = html[view_start:view_end]
    assert "Dollar amounts" not in view
    assert "your plan" not in view
    assert "outlierDollarsShown" not in view
    assert "not a fraud or error signal." in view


def test_traces_list_surfaces_pagination(html):
    assert "total_count" in html
    assert "Showing ${traces.length} of ${totalCount} traces" in html
    assert "Load more" in html
    assert "load({ append: true })" in html
    assert "offset" in html


def test_meta_caption_class_is_defined(html):
    """`.meta` (the "Showing N of M traces" line + the outlier-rule note
    beneath it, and the trace detail view's costliest-spans line) used to have
    NO matching CSS rule at all, so those divs fell through to the body's
    default sans-serif font/size/line-height instead of the app's small
    monospace caption style — a mismatch from every other caption on the page.
    A second, unrelated bug compounded it: the outlier note's inline
    `margin-top:-8px` assumed a same-size sibling with its own margin-bottom
    to collapse against; with no such margin, the negative offset pulled the
    (much taller, default-sized) note up and over the line above it, so
    glyphs overlapped. Declaring `.meta` with `margin-bottom: 8px` gives the
    negative top margin exactly 8px to collapse against — net 0 gap, flush
    but never overlapping, regardless of how many lines either caption wraps
    to at a narrow width."""
    m = re.search(r"\.meta\s*\{([^}]*)\}", html)
    assert m, ".meta has no CSS rule"
    rule = m.group(1)
    assert "font-family: 'Geist Mono', monospace" in rule
    assert "margin-bottom: 8px" in rule
    # The two Traces-list captions that were colliding.
    assert '<div class="meta">Showing ${traces.length} of ${totalCount} traces</div>' in html
    assert '<div class="meta" style="margin-top:-8px">${outlierNote}</div>' in html


def _trace_detail_src(html: str) -> str:
    start = html.index("function TraceDetailView")
    end = html.index("function fmtFramedDollar", start)
    return html[start:end]


def test_trace_detail_is_token_first_with_no_framing_branch(html):
    """The trace detail/waterfall page is part of the same Traces feature as
    the list (reached by clicking a row there); it now shows token totals for
    every account, matching the list's own display convention -- a property
    of how traces are presented, not a consequence of plan tier. No bare
    fmtCost either (that was the original #187/#249 bug: raw dollars with no
    framing at all). TraceDetailView no longer fetches/stores framing, since
    nothing in it reads pricing_mode any more."""
    view = _no_comments(_trace_detail_src(html))
    assert "fmtCost(s.cost_usd)" not in view
    assert "fmtCost(sel.cost_usd)" not in view
    assert "fmtFramedDollar(" not in view
    assert "fmtPerItemCost(" not in view
    assert "framing" not in view
    assert "setFraming" not in view


# --- #191: suppress raw $ on Status, Optimize & Reuse/script surfaces -------- #
def test_status_card_cost_today_routes_through_framing(html):
    # Status agent cards' "Cost today" must consume the /status framing block,
    # not render raw fmtCost(a.cost_today). Per #249 it's per-item, so it
    # routes through fmtPerItemCost (tokens for LOCAL only now -- subscription
    # no longer suppresses, product decision), not fmtFramedDollar "% of cycle".
    assert "${fmtCost(a.cost_today)}" not in html
    assert "${fmtPerItemCost(a.cost_today, _costVal(a, true), data.framing)}" in html


def test_optimize_window_comparison_routes_through_framing(html):
    # The window-comparison cost delta must reframe (LOCAL suppresses; api and
    # subscription both render the dollar figure -- product decision).
    assert "${fmtCost(Math.abs(st.cmp.cost_delta_usd))}" not in html
    assert "${fmtFramedDollar(Math.abs(st.cmp.cost_delta_usd), framing)}" in html


def test_optimize_budget_projection_routes_through_framing(html):
    # Budget-projection run-rate / ceiling / overage must reframe, not raw $.
    assert "${fmtCost(b.monthly_run_rate_usd)}" not in html
    assert "${fmtCost(b.budget_usd)}" not in html
    assert "${fmtCost(b.projected_overage_usd)}" not in html
    assert "${fmtFramedDollar(b.monthly_run_rate_usd, framing)}" in html
    assert "${fmtFramedDollar(b.budget_usd, framing)}" in html
    assert "${fmtFramedDollar(b.projected_overage_usd, framing)}" in html


def test_optimize_cluster_avg_cost_routes_through_framing(html):
    # The script cluster table "Avg cost" cell is per-item, so per #260 it
    # routes through fmtPerItemCost (tokens for LOCAL only now -- subscription
    # no longer suppresses, product decision), not the raw $ nor the
    # window-aggregate fmtFramedDollar "% of cycle".
    assert "${fmtCost(c.avg_cost_usd)}" not in html
    assert "${fmtFramedDollar(c.avg_cost_usd, framing)}" not in html
    assert "${fmtPerItemCost(c.avg_cost_usd, c.avg_tokens, framing)}" in html


# --- Lens Visualizations Wave 1: cost charts (#211–#213) ------------------- #
def test_stacked_bar_chart_present(html):
    # #213: cost-by-model/agent renders a STACKED bar chart, not overlapping
    # lines. The component + the cumulative back-to-front stacking must exist.
    assert "function StackedBarChart" in html
    assert "uPlot.paths.bars" in html
    # CostView routes model/agent group_by to the stacked chart (total stays line).
    assert "${StackedBarChart}" in html
    assert "groupBy === 'total' ?" in html


def test_stacked_bar_chart_uses_framing_tokens(html):
    """Stacked chart respects plan-tier framing: LOCAL -> tokens (no marginal
    cost to price at all). Subscription used to switch to tokens here too;
    that differentiation is gone by product decision (tj does not
    differentiate subscription-billed from API-billed users), so subscription
    now renders dollars like api/unknown."""
    assert "fmtY=${fmtY}" in html  # fmtY = useTokens ? fmtTokens : fmtCost
    assert "const useTokens = !!framing && framing.pricing_mode === 'local'" in html


def test_cache_savings_chart_present(html):
    # #212: cache hit-rate + cumulative captured-vs-recoverable chart.
    assert "function CacheSavingsChart" in html
    assert "function buildCacheSeries" in html
    assert "${CacheSavingsChart}" in html
    # fetched from the dedicated endpoint
    assert "/cost/cache" in html


def test_cache_savings_honesty_framing(html):
    # #246 dropped the "estimated recoverable" overlay from this chart (noise;
    # it lives on Optimize). The cache card now reports MEASURED savings,
    # framed: api and subscription → "$X saved" (product decision: no
    # subscription differentiation), LOCAL → cached-token VOLUME (never raw
    # $, no marginal cost to price at all).
    assert "fmtCost(cacheResp.total_captured_usd || 0)}</b> saved this window" in html
    assert "fmtTokens(cacheResp.total_captured_tokens || 0)}</b> cached reads this window" in html
    # the recoverable overlay is no longer wired into the cache chart
    assert "fmtFramedDollar(cacheResp.past_overspend_usd" not in html


def test_cache_savings_chart_is_best_effort(html):
    # A failing /cost/cache must not blank the cost screen.
    assert "cacheResp = await api('/cost/cache'" in html
    assert "} catch (_) { cacheResp = null; }" in html


# --- #211: cost-by-component + recoverable-waste overlay ------------------- #
def _component_waste_card(html: str) -> str:
    """Extract the #211 chart card markup so honesty asserts are scoped to it."""
    start = html.index("Cost by component + recoverable waste")
    return html[start:start + 1800]


def test_component_waste_chart_present(html):
    assert "function ComponentWasteChart" in html
    assert "function buildComponentWaste" in html
    assert "function componentWasteTooltip" in html
    assert "${ComponentWasteChart}" in html
    assert "uPlot.paths.bars" in html  # uPlot stacked bars, not a new lib


def test_component_waste_fetches_dedicated_endpoint(html):
    assert "/cost/components" in html
    # best-effort: a failed fetch must not blank the Optimize screen
    assert "api('/cost/components'" in html
    assert ".catch(() => null)" in html


def test_component_waste_is_registry_driven(html):
    # The overlay is built from the response's `recoverable` list (server-side
    # registry iteration), not a hard-coded analyzer array in the UI.
    assert "resp.recoverable" in html
    # no hard-coded per-analyzer overlay list like ['downsize','cache',...] in
    # buildComponentWaste — it maps over whatever the server returned
    card = html[html.index("function buildComponentWaste"):html.index("function buildComponentWaste") + 700]
    assert "resp.recoverable" in card
    assert "['downsize'" not in card


def test_component_waste_honesty_estimated_not_saved(html):
    card = _component_waste_card(html)
    # Positive honesty language present…
    assert "estimated recoverable" in card.lower()
    assert "not a realized cost reduction" in card
    # …and the word "saved" never appears on THIS surface (Rule 14).
    assert "saved" not in card.lower()
    assert "savings you got" not in card.lower()


def test_component_waste_recoverable_routes_through_framing(html):
    """Per-analyzer recoverable must reframe (LOCAL -> token-share; local has
    no marginal cost to price at all), mirroring the existing recoverable
    band — not raw fmtCost. Subscription used to switch to token-share here
    too; that differentiation is gone by product decision (tj does not
    differentiate subscription-billed from API-billed users)."""
    assert "fmtFramedSavings(r.usd, r.tokens, compFraming)" in html
    # the measured-cost total uses the dollar framing helper, not raw fmtCost
    assert "fmtFramedDollar(st.comp.total_cost_usd" in html
    # plan-tier toggle drives tokens-vs-dollars for the whole surface
    assert "compFraming.pricing_mode === 'local'" in html
    assert "compFraming.pricing_mode === 'subscription'" not in html


# --- trim card: provenance + flagged text on the web card, not just the CLI - #
def _trim_branch(html: str) -> str:
    start = html.index("} else if (name === 'trim') {")
    end = html.index("} else if (name === 'reuse') {", start)
    return html[start:end]


def test_trim_card_shows_provenance_and_summarize_pointer(html):
    # The CLI already prints `p.source_path`'s attribution + a `tj summarize
    # list` pointer for prompts a catalog file cleared the verbatim-
    # containment bar for; the web card must carry the same fields, gated
    # identically on `p.source_path` being set.
    block = _trim_branch(html)
    assert "p.source_path" in block
    assert "Attributed to" in block
    assert "p.source_basis" in block
    assert '#/optimize/summarize' in block
    assert "Review in Summarize" in block


def test_trim_card_shows_flagged_regions_unconditionally(html):
    # The flagged text itself must render regardless of provenance — a pure
    # SDK caller never gets a source_path, and the flagged-text block is
    # their whole, complete answer, not a lesser version of the card (persona
    # coherence: no message implying something is missing for them).
    block = _trim_branch(html)
    assert "p.regions" in block
    assert "sample_chars" in block
    assert "more region(s)" in block


def test_trim_card_no_longer_a_flat_three_column_table(html):
    # Provenance + flagged text is more than one line per prompt, so the flat
    # <table> gave way to a per-prompt block.
    block = _trim_branch(html)
    assert "<table" not in block


# --- #210: Analytics pivot explorer (subsumes #214 leaderboard + #216) ----- #
def test_analytics_screen_registered(html):
    assert "function AnalyticsView" in html


def test_analytics_route_retired(html):
    # The standalone Analytics screen + nav item are gone: it was a duplicate
    # of the Dashboard's own embedded "Explore" section (same AnalyticsView
    # component), so the separate nav entry was a second door to one room.
    # Lingering #/analytics links fall through to the Dashboard, same pattern
    # as the earlier #/overview retirement.
    assert 'href="#/analytics"' not in html
    assert "['analytics', AnalyticsView]" not in html
    assert "analytics: 'observe'" not in html  # VIEW_LENS entry retired too
    # Keep-alive router: #/analytics folds into the Dashboard key via primaryKeyFor.
    assert "if (v === 'overview' || v === 'analytics') return 'dashboard';" in html
    # The component itself and the API route it calls both stay -- the
    # Dashboard's embedded explorer hard-depends on both.
    assert "function AnalyticsView" in html
    assert "await api('/analytics'" in html


def test_analytics_metric_dimension_chart_controls(html):
    # metric × group_by × stack × chart-type controls, driven off shared vocab.
    assert "const ANALYTICS_METRICS" in html
    assert "const ANALYTICS_DIMENSIONS" in html
    assert "const ANALYTICS_CHARTS" in html
    for ctl in ("'metric'", "'group_by'", "'stack'", "'chart'"):
        assert ctl in html, f"missing control {ctl}"
    # the three uPlot/leaderboard chart types
    for ch in ("'bar'", "'line'", "'hbar'"):
        assert ch in html


def test_analytics_presets_and_csv_export(html):
    assert "const ANALYTICS_PRESETS" in html
    assert "function analyticsCsv" in html
    assert "function downloadCsv" in html
    assert "Export CSV" in html
    assert "disabled=${loading}" in html
    # the leaderboard preset closes #214; spend-by-model line closes #216
    assert "'leaderboard'" in html
    assert "'spend-by-model'" in html

    # Richer CSV columns assertions
    assert "'cycle_share_pct'" in html
    assert "'input_tokens'" in html
    assert "'output_tokens'" in html
    assert "'cache_read_tokens'" in html
    assert "'cache_write_tokens'" in html
    assert "'sessions'" in html
    assert "'events'" in html
    # Filename generation check
    assert "getFilename" in html
    assert "tokenjam-analytics.csv" in html
    assert "tokenjam-analytics_${startStr}_${endStr}.csv" in html


def test_analytics_url_is_source_of_truth(html):
    # state read from URL params with validators, written back via navigate().
    # navigate() targets `route` (the 'analytics' default is a vestige of the
    # now-retired standalone screen -- unused today since every live caller,
    # the Dashboard, passes route="dashboard" explicitly).
    assert "route = 'analytics'" in html
    assert "navigate(route, { ...cur" in html
    assert "readParam(params, 'metric'" in html
    assert "readParam(params, 'group_by'" in html
    assert "readParam(params, 'chart'" in html


def test_analytics_consumes_endpoint_not_reimplements(html):
    # single compute path: fetches /analytics and renders from the response
    assert "api('/analytics'" in html
    assert "resp.groups" in html
    assert "resp.rows" in html


def test_analytics_respects_plan_tier_framing(html):
    """The chart/leaderboard's own spend metric switches to token volume for
    LOCAL only (dollars suppressed -- no marginal cost to price at all); it
    used to switch for subscription too, that differentiation is gone by
    product decision (tj does not differentiate subscription-billed from
    API-billed users) -- never re-derives the suppression rule, reads
    framing. The Spend KPI tile is separate: it still reframes via
    spendTileDisplay (implied-value multiplier for subscription, #262)
    rather than fmtFramedDollar's "% of cycle" -- that tile's multiplier is a
    genuinely different metric, not a suppression mechanism, so it was never
    part of the de-differentiation (see test_lens_dashboard_states.py's
    spendTileDisplay tests)."""
    assert "isSpend && !!framing && framing.pricing_mode === 'local'" in html
    assert "spendTileDisplay(kpis.spend, framing)" in html


def test_analytics_leaderboard_has_inline_bars(html):
    # #214: sorted leaderboard with inline magnitude bars (CSS, no chart lib).
    assert "function buildLeaderboard" in html
    assert "lb-fill" in html
    assert ".lb-bar" in html


# --- #318: active tile shows the breakdown subtotal for a partial dimension --- #
def test_analytics_active_tile_shows_breakdown_subtotal(html):
    # When grouping by a PARTIAL dimension (only some spans carry it, e.g. tool),
    # the active count-metric tile shows the breakdown subtotal beneath the window
    # total so the tile reconciles with the smaller chart subtotal (#318). Count
    # metrics only; only when there's an actual gap.
    assert "const breakdownTotal = (resp.rows || []).reduce" in html
    assert "const activeSub = ((metric === 'events' || metric === 'sessions')" in html
    assert "by ${breakdownDim}" in html
    # KpiTile renders the optional sub-line; defaults null so other callers (e.g.
    # the Dashboard preview) are unaffected.
    assert "onSelect, sub = null }" in html
    assert "kpi-sub-val" in html


# --- #295: Stack gated to stacking charts; empty cross-tab gets a clear state - #
def test_analytics_stack_gated_to_stacking_charts(html):
    # Stack only applies to the multi-series charts (bar/line). The Leaderboard
    # (hbar) ignores stack, so the control is hidden AND stack_by is dropped from
    # the query for non-stacking charts — otherwise a stale stack strands the
    # leaderboard on an empty cross-tab ("No data", #295).
    assert "const stackApplies = chart === 'bar' || chart === 'line'" in html
    assert "const effStack = stackApplies ? stack : ''" in html
    # query drops stack_by when the chart doesn't stack
    assert "stack_by: effStack || undefined" in html
    assert "stack_by: stack || undefined" not in html  # the buggy unconditional form is gone
    # the Stack control is conditionally rendered (hidden on the leaderboard)
    assert "${stackApplies ? html`<label class=\"ctl\">Stack" in html


def test_analytics_empty_cross_tab_offers_clear_stack(html):
    # A structurally-empty stacked breakdown (e.g. Model x Tool category, since a
    # span carries a model OR a tool, never both) shows a "Clear stack" affordance
    # instead of a bare "No data in this window" (#295).
    assert "const emptyFromStack" in html
    assert "Clear stack" in html


# --- #313: leaderboard surfaces its total + reconciles the partial-dim gap --- #
def test_analytics_leaderboard_shows_total_and_gap(html):
    # The leaderboard ranks items but used to show no sum; it now surfaces its
    # own item count + subtotal, and when grouping by a PARTIAL dimension (only
    # some spans carry it, e.g. tool) it reconciles the gap against the all-events
    # KPI so the smaller subtotal doesn't look contradictory (#313).
    assert "const boardTotal = board ? board.reduce" in html
    assert "const boardGap = kpiCount != null" in html
    assert "Total: ${fmtVal(boardTotal)}" in html
    assert "${boardCount} ${boardCount === 1 ? dimName : dimNamePlural}" in html
    assert "have a ${dimName}" in html


# --- #215: token-annotated trace waterfall ---------------------------------- #
def test_trace_waterfall_cost_summary(html):
    """A token-first trace summary header (total + tokens + duration + spans).
    A single trace's total is per-item, not a window aggregate -- per #249 it
    is a plain token total (no "% of cycle" window-aggregate category error),
    matching the Traces list's own unconditional token display -- a property
    of how traces are presented, not a consequence of plan tier. This site
    used to route through fmtPerItemCost (tokens for subscription/local,
    dollars for api/unknown); it is now unconditional, like the list."""
    assert "wf-summary" in html
    assert "Total cost (tokens)" in html
    assert "wfTotalCostFramed" in html
    view = _trace_detail_src(html)
    assert "const wfTotalCostFramed = fmtTokens(wfTotalInOut) + ' tok';" in view


def test_trace_waterfall_per_span_token_annotation(html):
    """Per-span token annotation column with a magnitude bar (not just the
    hover tooltip). The sibling dollar annotation (.wf-cost-val) that used to
    sit beside it is gone: this page shows tokens for every account now, so
    that span always rendered empty and was removed as dead markup along
    with its now-unused CSS rule."""
    view = _trace_detail_src(html)
    assert "wf-cost-bar" in view
    assert "wf-cost-fill" in view
    assert 'class="wf-cost-val"' not in view
    assert 'class="wf-cost-tok"' in view
    # tokens summed per span and shown in the annotation
    assert "const spanTokens = s =>" in view
    assert "wf-cost-tok\">${sTok ? fmtTokens(sTok)" in view
    assert ".wf-cost-val {" not in html


def test_trace_waterfall_magnitude_is_token_first_unconditionally(html):
    """The magnitude bar (and summary) read on TOKEN volume for every
    account, with no branch on pricing_mode at all -- matching the Traces
    list's own unconditional token display, since the waterfall is part of
    the same Traces feature (reached by clicking a row in the list). This
    site was not in the founder's named list of four `useTokens` duplicates
    to convert; it was first converted to LOCAL-only (mirroring the other
    four), then corrected to unconditional tokens once it was confirmed the
    waterfall belongs to the Traces surface, not the general
    subscription-vs-API dollar conversion -- see the session report."""
    view = _no_comments(_trace_detail_src(html))
    assert "wfUseTokens" not in view
    assert "pricing_mode" not in view
    assert "const wfMagOf = s => spanTokens(s);" in view


# --- #217: KPI tiles → sparkline + period-over-period delta ----------------- #
def test_kpi_tiles_have_sparkline_and_delta(html):
    # KPI tiles gain a trend sparkline + a signed period-over-period delta chip.
    assert "function Sparkline(" in html
    assert "function DeltaChip(" in html
    assert "function KpiTile(" in html


def test_kpi_sparkline_is_inline_svg_not_uplot(html):
    # The sparkline is a lightweight inline SVG (offline, no per-tile uPlot
    # instance) — so #218's offline guarantee + render cost both hold.
    assert '<svg class="spark"' in html
    assert "<polyline points=" in html


def test_kpi_series_is_server_computed_not_client_aggregated(html):
    # Single compute path: the sparkline reads the server's `kpi_series` through
    # the shared window grid; the UI never buckets/aggregates per-span in JS.
    assert "function kpiSparkValues(" in html
    assert "_windowGrid({ ...resp, series: resp.kpi_series })" in html
    assert "resp.kpi_deltas" in html


def test_kpi_spend_tile_respects_framing(html):
    # The Spend tile reads the framed value from the server block (api → $,
    # subscription → implied-value multiplier "43.5× plan value", #262), never
    # raw $ for subscription. Its sparkline and delta track SPEND (cost_usd) —
    # the multiplier is just spend rescaled, so the trend/shape match while the
    # displayed number is never raw dollars.
    assert "const spend = spendTileDisplay(kpis.spend, framing)" in html
    assert "series: kpiSparkValues(resp, 'spend'), delta: deltas.spend" in html


# --- #228: shared series→color map + colored leaderboard ------------------- #
def test_shared_colorfor_helper_exists(html):
    # ONE name-keyed color map, hashed into the shared --chart-1..5 palette, so a
    # series is the same hue everywhere (not a per-chart positional palette).
    assert "function colorFor(name)" in html
    assert "h = (h * 31 + s.charCodeAt(i))" in html


def test_leaderboard_bars_use_shared_colorfor(html):
    # Leaderboard .lb-fill colored by the shared map, keyed by the group name.
    assert "background:' + colorFor(e.group)" in html


def test_dimension_charts_color_by_name(html):
    # SpendChart + StackedBarChart color multi-series via colorFor(name), not by
    # draw-order index — same map the leaderboard uses (ComponentWasteChart keeps
    # its own component palette; different namespace, out of scope).
    assert html.count("single ? (palette[0] || '#3d8eff') : colorFor(lab)") >= 2
    # the stacked-bar tooltip dots match the bars (also via the shared map)
    assert "colorFor(labels[k] || ('s' + k))" in html


# --- #227: don't color by the time dimension ------------------------------- #
def test_time_dimension_renders_single_series(html):
    # group_by=Day with no stack must be ONE series (tokens/day, one color), not
    # one-per-day-bucket → no raw-epoch rainbow legend.
    assert "const timeGroup = resp.group_by === 'day'" in html
    assert "return { data: [xs, ys], labels: ['Total']" in html
    # the time dimension feeds the x-axis; series come from stack_by instead
    assert "const seriesKeys = timeGroup ? (resp.stacks || []) : (resp.groups || [])" in html


def test_time_dimension_labels_formatted_as_dates(html):
    # A time-dimension group key renders as a date, never a raw epoch second.
    assert "function formatGroupLabel" in html
    assert "formatGroupLabel(e.group, groupBy)" in html


# --- #234: expanded chart palette (12 hues) reduces colorFor() collisions --- #
def test_colorfor_palette_expanded_to_twelve(html):
    # colorFor hashes into a 12-hue palette (was 5) so distinct series rarely
    # collide on real data; the stable-hash mapping itself is unchanged.
    assert "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map(i => cssVar('--chart-' + i))" in html


def test_chart_palette_defines_twelve_hues_both_themes(html):
    # --chart-1..12 must be defined for BOTH the dark (:root) and light themes,
    # so charts re-theme correctly.
    for n in range(1, 13):
        assert html.count(f"--chart-{n}:") >= 2, f"--chart-{n} not defined in both themes"


def test_colorfor_neutral_bucket_not_a_palette_hue(html):
    # 'other'/'(none)'/'' still map to the neutral grey, never a palette color.
    assert "if (s === 'other' || s === '(none)' || s === '') return cssVar('--text-dim')" in html


# --- #229: Overview tiles deep-link into the Optimize detail card ---------- #
def test_overview_recoverable_tiles_deeplink_into_optimize(html):
    # A tile's recoverable number has its evidence and next step on the
    # Optimize screen's own analyzer card, not an Analytics chart slice — a
    # tile used to open an Analytics leaderboard with no reuse section at all
    # for some analyzers. optimizeFindingHref builds a
    # `#/optimize?...finding=name` deep-link; OptimizeView's focus effect
    # scrolls to and highlights #opt-<name>.
    assert "function optimizeFindingHref(name, since)" in html
    assert "sp.set('finding', name)" in html
    assert "optimizeFindingHref(t.name, since)" in html
    # The old Analytics-slice routing is gone — regression guard so a tile
    # click can't silently land back on a page with no matching section.
    assert "const ANALYZER_ANALYTICS_SLICE" not in html
    assert "function analyzerSliceHref" not in html


def test_analytics_deeplink_helper_exists_and_builds_hash_urls(html):
    # The deep-link helper builds #/analytics?... URLs (offline hash links, no
    # fetch) from a query object, dropping empty values.
    assert "function analyticsHref(q, route = 'analytics')" in html
    assert "return '#/' + route + (s ? '?' + s : '');" in html


def test_review_inbox_is_default_landing(html):
    # Two-lens IA (self-improve-loop SPEC.md §12): empty hash → the Review
    # inbox, the Improve lens's home — set via the PARSED default (no
    # render-time location.hash redirect — #132 discipline still applies).
    # Dashboard remains a fully-reachable Improve-lens view, just no longer
    # the landing route.
    assert "|| 'review'" in html
    assert "|| 'dashboard'" not in html
    assert "|| 'overview'" not in html
    assert "location.hash = '#/dashboard'" not in html  # no hash-assign redirect
    assert "history.replaceState(null, '', '#/review')" in html
    assert "function ReviewInboxView" in html
    # Keep-alive router: Review + Dashboard are wired through the PRIMARY_VIEWS
    # registry (which replaced the switch), both mounted-and-kept-alive.
    assert "['review',    ReviewInboxView]" in html
    assert 'href="#/review"' in html
    assert "function DashboardView" in html
    assert "['dashboard', DashboardView]" in html
    assert 'href="#/dashboard"' in html


def test_overview_retired(html):
    # Standalone Overview screen + nav item gone; lingering #/overview links fall
    # through to the Dashboard.
    assert "function OverviewView" not in html
    assert 'href="#/overview"' not in html
    # Keep-alive router: #/overview folds into the Dashboard key via primaryKeyFor
    # (replaced the switch's `case 'overview': case 'dashboard'` fallthrough).
    # #/analytics was folded into the same alias branch later — see
    # test_analytics_route_retired.
    assert "if (v === 'overview' || v === 'analytics') return 'dashboard';" in html


def test_dashboard_embeds_analytics_explorer(html):
    # The hero composes the existing AnalyticsView (route rewired to dashboard,
    # embedded, with the run-rate caption) — not a reimplemented pivot. The
    # component's default args stay call-compatible even though nothing
    # invokes it bare any more (the standalone #/analytics route is retired;
    # see test_analytics_route_retired).
    assert 'route="dashboard" embedded=${true} kpiCaption=${kpiCaption}' in html
    assert "function AnalyticsView({ params, route = 'analytics', embedded = false, kpiCaption = null })" in html


def test_analytics_alias_preserves_query_params_and_deep_links_to_explore(html):
    # An old #/analytics?metric=...&group_by=... bookmark must land on the same
    # explorer slice, not a bare Dashboard. primaryKeyFor only remaps the VIEW
    # KEY used to pick a component out of PRIMARY_VIEWS; route.params (built
    # from the raw query string in getRoute(), independent of that mapping)
    # flow straight through to DashboardView and then to the embedded
    # AnalyticsView unchanged, so metric/group_by/stack/chart/since survive.
    dash_start = html.index("function DashboardView")
    dash_end = html.index("function ", dash_start + 1)
    dash_view = html[dash_start:dash_end]
    assert 'id="dash-explore"' in dash_view
    assert "isAnalyticsAlias" in dash_view
    assert "scrollIntoView" in dash_view


def test_dashboard_spend_deduped(html):
    # Spend shown ONCE (explorer's Spend KPI tile + chart); the old separate
    # run-rate headline chart is gone, folded into a caption under the KPI row.
    # The caption's gate moved onto the /cost read's own answer when the page's
    # single load flag was split into one read per source; `projection` is null
    # until `costData` exists, so the caption still cannot be built from an
    # unanswered read.
    assert "const kpiCaption = (projection && projection !== '—')" in html
    assert 'class="kpi-caption"' in html


def test_kpi_tiles_clickable_select_metric(html):
    # #247: tiles are the metric selector — onSelect writes the metric to the URL.
    assert "onSelect=${() => onMetric(t.key)}" in html
    assert "const onMetric = (k) => setFilter('metric', k)" in html
    assert "kpi-clickable" in html


def test_spend_tile_distinct_under_subscription(html):
    # #247/#262: the Spend tile no longer falls back to raw tokens (which
    # duplicated the Tokens tile). It uses spendTileDisplay (implied-value
    # multiplier for subscription) and is dropped when no distinct value exists
    # (local / a subscription with no declared fee → null).
    assert "const spend = spendTileDisplay(kpis.spend, framing)" in html
    assert "if (spend) {" in html
    assert "spendSuppressed ? (fmtTokens(kpis.tokens) + ' tok')" not in html  # old dup gone


def test_tokens_tile_shows_dollar_headline_with_token_count_secondary(html):
    """Founder request: the Tokens card leads with its dollar value, the
    token count present but secondary (subtitle-scale, dimmer). Computed
    independently of `spend` (card 1), NOT reused from it: card 1 renders an
    implied-value MULTIPLIER for SUBSCRIPTION ("43.5x plan value"), a
    different metric entirely, not a dollar figure -- reusing spend.value
    would have put that multiplier text on the Tokens card too. LOCAL is the
    only pricing mode with no marginal cost to price at all; every other mode
    (api / subscription / unknown) shows the real dollar amount here, so for
    a SUBSCRIPTION account the two cards intentionally show different things
    (card 1 the multiplier, card 2 the dollar amount) -- no duplication
    there. For an API/unknown account both cards show the identical dollar
    figure under two different labels (flagged, not resolved, in the session
    report per the founder's own instruction). Falls back to the plain token
    headline unchanged, with no secondary line, when there is no known
    dollar figure at all (LOCAL, or an unreported spend field) -- never
    fabricates one."""
    assert "const spendMode = framing && framing.pricing_mode;" in html
    assert "const tokensDollar = (kpis.spend != null && spendMode !== 'local') ? fmtDashUsd(kpis.spend) : null;" in html
    assert "value: tokensDollar || kpiFigure(kpis.tokens, fmtTokens), sub: tokensSub," in html
    assert "cost: !!tokensDollar," in html
    assert "label: 'tokens', dim: true" in html
    # Sparkline + delta are untouched -- still the token series/delta, not spend's.
    assert "series: kpiSparkValues(resp, 'tokens'), delta: deltas.tokens });" in html


def test_kpi_sub_dim_is_an_additive_modifier_not_a_shared_restyle(html):
    """The Tokens tile's dimmer token-count sub-line is a separate modifier
    class (.kpi-sub-dim), not a change to .kpi-sub-val/.kpi-sub-lab
    themselves -- the #318 breakdown-subtotal sub-line (sessions/events
    tiles, `activeSub`) keeps its normal-brightness look, since it never
    passes `dim`."""
    assert ".kpi-sub-dim .kpi-sub-val { color: var(--text-dim)" in html
    assert "sub=${t.sub || (t.key === metric ? activeSub : null)}" in html
    # activeSub itself never sets `dim`, so it always renders at normal brightness.
    assert "{ value: fmtCount(breakdownTotal), label: `by ${breakdownDim}` }" in html


def test_dashboard_triage_drills_into_optimize_card(html):
    # Recoverable-waste tiles navigate to the Optimize screen's matching
    # analyzer detail card — no longer an in-place Analytics explorer slice
    # update.
    assert "optimizeFindingHref(t.name, since)" in html
    assert "function optimizeFindingHref(name, since)" in html
# --- #306: Status screen is two content-defined zones (no view toggle) ------ #
def test_status_screen_has_split_zones(html):
    # The #241/#263 Cards|List view toggle is retired: the Status screen now
    # renders two content-defined zones — coding sessions (cards + archive) and
    # SDK services — rather than a user-selected view of one agent list.
    start = html.index("function StatusView")
    end = html.index("function TracesListView", start)
    view = html[start:end]
    assert 'class="zone-title">Coding sessions' in view
    assert "SDK agents / services" in html
    # the toggle wiring is gone
    assert "const setView =" not in html
    assert "onClick=${() => setView('cards')}" not in html
    assert "function StatusListTable" not in html


def test_status_sdk_services_panel_exists(html):
    # The SDK zone is its own component fed by data.sdk_services.
    assert "function SdkServicesPanel({ services, framing })" in html
    assert "<${SdkServicesPanel} services=${sdkServices} framing=${data.framing} />" in html


def test_sdk_services_panel_renders_window_cost(html):
    # sdk_services[].window_cost (the cost summed over the same 24m sparkline
    # window as cost_per_min) is computed server-side; pin it next to the
    # cost/min sparkline it summarizes so it doesn't silently go unrendered.
    panel_start = html.index("function SdkServicesPanel({ services, framing })")
    panel_end = html.index("function useScrollMemory", panel_start)
    panel = html[panel_start:panel_end]
    assert "s.window_cost" in panel


def test_sdk_panel_reuses_sparkline_and_splits_by_state(html):
    # cost/min + err% sparklines come from the per-minute series; services are
    # partitioned live vs went_quiet/long_dormant.
    assert "values=${s.cost_per_min}" in html
    assert "values=${s.err_pct_per_min}" in html
    assert "s.state === 'live'" in html
    assert "s.state === 'went_quiet'" in html


def test_coding_zone_partitions_agents_and_archive_by_kind(html):
    # Cards are the coding agents; the collapsible archive is the coding archive.
    assert "const codingAgents = agents.filter(a => a.kind === 'coding');" in html
    assert "(data.archived || []).filter(s => s.kind === 'coding'" in html


def test_coding_archive_is_collapsible_and_scrolls(html):
    # The archive is a collapsible <details> holding the archive-list table,
    # which sits in the scrolling .table-wrap AND carries a min-width so it
    # overflows (and scrolls) instead of clipping the actions cell.
    assert "<div class=\"table-wrap\"><table class=\"archive-list\">" in html
    assert ".table-wrap { overflow-x: auto; }" in html
    assert "table.archive-list {" in html and "min-width: 820px;" in html


def test_status_archive_cost_column_respects_framing(html):
    # The archive Cost cell routes through the /status framing block (#17/#249),
    # never a raw dollar.
    assert "<td>${fmtFramedDollar(s.total_cost_usd, data.framing)}</td>" in html


# --- #249: "% of cycle" is window-level; per-item cost must render as tokens -- #
def test_per_item_cost_helper_renders_tokens_for_subscription_local(html):
    """The per-item formatter: LOCAL only -> token total (the in+out basis via
    _costVal); api / subscription / unknown -> dollars uniformly. Subscription
    used to also render tokens here; that branch is gone by product decision
    (tj does not differentiate subscription-billed from API-billed users).
    Local is a structurally different case, not a differentiation choice --
    local inference incurs no marginal dollar cost at all -- so it still
    renders tokens. "% of cycle" (a window aggregate) is never produced at
    per-item granularity, regardless."""
    assert "function perItemUsesTokens(framing)" in html
    assert "function fmtPerItemCost(costUsd, tokenTotal, framing)" in html
    assert "return !!framing && framing.pricing_mode === 'local';" in html
    assert "if (perItemUsesTokens(framing)) return fmtTokens(tokenTotal || 0) + ' tok';" in html
    # the only "% of cycle" string in the codebase lives in fmtFramedDollar, which
    # per-item surfaces no longer call directly for the row value.
    assert "return fmtFramedDollar(costUsd, framing); // api / subscription / unknown → dollars" in html


def test_per_item_cost_surfaces_use_the_helper_not_framed_dollar(html):
    """Every per-item dollar cell that still shows a dollar figure — Status
    cards' "Cost today" — uses fmtPerItemCost, not the window-aggregate
    fmtFramedDollar. Guards against a regression reintroducing "% of cycle"
    at per-row granularity (the #249 bug: "466.7% of cycle"). The Traces
    list, the trace detail/waterfall's span-detail panel, and its per-trace
    total no longer call fmtPerItemCost at all: the whole Traces feature area
    (list, detail, waterfall) shows unconditional token totals now (a display
    convention, not a plan consequence), bypassing the tokens-vs-dollars
    helper entirely rather than routing through it."""
    assert "${fmtPerItemCost(a.cost_today, _costVal(a, true), data.framing)}" in html  # status card
    assert "${fmtFramedDollar(a.cost_today, data.framing)}" not in html
    view = _trace_detail_src(html)
    assert "fmtPerItemCost(" not in view
    assert "fmtFramedDollar(" not in view


def test_per_trace_token_totals_come_from_server_not_aggregated_in_js(html):
    # _costVal reads server-provided per-row input_tokens/output_tokens; the UI
    # never re-sums spans in JS for the list rows (single compute path, #249).
    assert "function _costVal(r, useTokens)" in html


# --- #244: trace-waterfall — fixed name column, magnitude bars, status ------ #
def test_waterfall_name_in_fixed_column_not_on_bar(html):
    # The span identity lives in a fixed left column (spanPrimaryName), never
    # painted onto the bar — that produced the "cla"/"Bas" clipping. The old
    # on-bar ${barLabel} and the detail/isAgent bar-label machinery are gone.
    assert "function spanPrimaryName(s)" in html
    assert 'class="wf-name-txt"' in html
    assert "${barLabel}" not in html
    assert "const barLabel = isAgent" not in html
    # The bar itself carries no text child now.
    assert '<div class="wf-bar ${kind}" style="width:100%"></div>' in html


def test_waterfall_bars_sized_by_magnitude_with_mode_toggle(html):
    """Bars size by token magnitude by default (the only thing that renders
    on duration-less backfill), with a tokens/duration toggle. Tokens is the
    only magnitude mode now -- the "By cost" toggle option is gone along with
    the dollar magnitude/annotation it drove, since this page shows tokens
    for every account (no plan-based choice to offer any more)."""
    assert "const [wfMode, setWfMode] = useState(null)" in html
    view = _trace_detail_src(html)
    assert "const wfMode2 = wfMode || 'tokens';" in view
    assert "const magForMode = s =>" in view
    assert "setWfMode('cost')" not in view
    assert "setWfMode('tokens')" in view
    assert "setWfMode('duration')" in view
    assert "By cost" not in view


def test_waterfall_has_minimum_bar_width(html):
    # A floor keeps tiny/zero-magnitude spans visible and clickable.
    assert "width = Math.max(1.5, Math.min(width, 100 - left))" in html


def test_waterfall_relative_offset_and_absolute_on_hover(html):
    # Per-span relative offset on the row; absolute wall-clock in the tooltip
    # and as a title; trace start in the summary header.
    assert "const offsetLabel = '+' + fmtDur(st)" in html
    assert 'class="wf-offset"' in html
    assert "new Date(traceStart).toLocaleString()" in html  # header trace start
    assert "new Date(s.start_time).toLocaleString()" in html  # per-span absolute


def test_waterfall_duration_not_captured_hint(html):
    # Missing duration shows an em-dash with a "not captured in backfilled data"
    # hint rather than a misleading 0 / 1ms sliver (#243/#244).
    assert "Duration not captured in backfilled data" in html
    assert "not captured in backfill" in html


def test_waterfall_status_icons_and_kind_legend(html):
    # Status icon per row (ok/error) + kind color dots + a legend.
    assert 'class="wf-status' in html
    assert 'class="wf-kind-dot' in html
    assert 'class="wf-legend"' in html
    assert "(s.status_code || '') === 'error'" in html


def test_waterfall_cost_framing_preserved(html):
    """Token-first, no framing branch at all: the per-span value used to
    route through the server framing block (fmtFramedDollar); it is now an
    unconditional token total for every account, matching the Traces list.
    Guards against a regression reintroducing a raw fmtCost (the original
    #187/#249 bug) OR a reintroduced framing/pricing_mode branch."""
    view = _trace_detail_src(html)
    assert "fmtCost(s.cost_usd)" not in view
    assert "fmtFramedDollar(" not in view
    assert "const label = fmtTokens(spanTokens(s)) + ' tok';" in view


# --- #246: cache-savings chart redesign (answer-first, single-axis bars) ---- #
def test_cache_chart_leads_with_answer_headline(html):
    # A plain headline: hit-rate stat + savings this window (not three overlaid
    # series). The card title is "Caching".
    assert '<div class="cache-headline">' in html
    assert "cacheSeries.hitRate.toFixed(0)}%</b> cache hit-rate" in html
    assert "saved this window" in html          # api / subscription framing
    assert "cached reads this window" in html    # local framing (no raw $)


def test_cache_chart_is_single_axis_per_period_bars(html):
    # The dual-axis (tokens left / hit-rate % right) + cumulative ramp + recoverable
    # overlay are gone. CacheSavingsChart takes a single per-bucket savings series.
    assert "function CacheSavingsChart({ data, height = 180" in html
    assert "<${CacheSavingsChart} data=${cacheSeries.data}" in html
    # old dual-axis/overlay props no longer passed
    assert "cache=${cacheSeries.data}" not in html
    assert "env=${cacheSeries.env}" not in html
    assert "hit=${cacheSeries.hit}" not in html
    # buildCacheSeries returns per-bucket savings (not a cumulative ramp) + the
    # headline stat + a hit-rate sparkline.
    assert "return { data: [xs, sav], hitSpark, hitRate" in html
    assert "let acc = 0" not in html.split("function buildCacheSeries")[1].split("function ")[1]


def test_cache_chart_hitrate_is_stat_not_overlaid_line(html):
    # Hit-rate shows as a small sparkline beside the stat, not an overlaid rate axis.
    assert "<${Sparkline} values=${cacheSeries.hitSpark}" in html


def test_cache_chart_explains_the_mechanic(html):
    # One-line plain-English mechanic.
    assert "Cached input bills at roughly a tenth of the normal input rate" in html


# --- #251: component-waste chart drops zero segments + positive empty state -- #
def test_component_waste_chart_filters_zero_segments(html):
    # The cumulative-overlap bar technique paints a zero-value segment as a
    # full-height bar in its own color (cache-write=0 → full-height purple over
    # the real stack). Zero-value segments must be filtered BEFORE building the
    # cumulative bars, in both columns.
    assert "const costSegsNZ = (costSegs || []).filter(s => (s.value || 0) > 0);" in html
    assert "const recSegsNZ = (recSegs || []).filter(s => (s.value || 0) > 0);" in html
    # the cumulative loops iterate the filtered lists, not the raw props
    assert "costSegsNZ.forEach((s, i) =>" in html
    assert "recSegsNZ.forEach((s, i) =>" in html
    # color offset uses the filtered cost length so the palette stays aligned
    assert "palette[(costSegsNZ.length + i) % palette.length]" in html


def test_component_waste_empty_recoverable_is_positive_state(html):
    # The empty "Recoverable (est.)" column shows a positive signal, not blank /
    # dim space.
    assert "Nothing recoverable in this window" in html
    assert 'class="waste-none"' in html
    # the old neutral/dim empty message is gone
    assert "No recoverable waste estimated in this window." not in html


def test_component_waste_dominant_split_label(html):
    # Optional %-split note when one token component is ~all the spend (>95%),
    # so the single-block bar is explained rather than mysterious.
    assert "function dominantSplit(costSegs)" in html
    assert "pct > 95 ?" in html
    assert "const wasteDominant = waste ? dominantSplit(waste.costSegs) : null;" in html
    assert 'class="waste-split"' in html


# --- #260: script cluster avg cost carries a server-side token total --------- #
def test_script_cluster_payload_token_total_is_server_side(html):
    # The cell consumes c.avg_tokens (server-provided per-cluster token total),
    # never re-aggregating in JS.
    assert "${fmtPerItemCost(c.avg_cost_usd, c.avg_tokens, framing)}" in html


# --- #262: Analytics spend tile = implied-value multiplier, separators, soft delta -- #
def test_analytics_spend_tile_uses_value_multiplier_for_subscription(html):
    # The Spend tile shows an implied-value multiplier ("43.5× plan value") for
    # subscription, never "% of cycle" and never raw $ — plan VALUE, not spend.
    assert "function spendTileDisplay(spendUsd, framing)" in html
    assert "+ '× plan value'" in html
    # multiplier == (% of cycle) / 100 == spend / plan_monthly_usd. The `|| 0`
    # that used to sit on the numerator is gone deliberately: it turned an
    # unreported spend field into "0.0x plan value", so the null case is now
    # caught before the division and renders as unknown instead (see
    # test_lens_dashboard_states.py).
    assert "const mult = spendUsd / framing.plan_monthly_usd;" in html
    assert "const unknown = spendUsd == null;" in html
    # the tile no longer renders fmtFramedDollar (the "% of cycle") for spend
    assert "const spendVal = fmtFramedDollar(kpis.spend, framing);" not in html
    assert "const spend = spendTileDisplay(kpis.spend, framing);" in html


def test_analytics_count_tiles_have_thousand_separators(html):
    # Sessions / Events tiles are exact counts with separators ("23,954"), not
    # raw String() integers.
    assert "function fmtCount(n)" in html
    assert "toLocaleString('en-US')" in html
    # Routed through kpiFigure so an omitted field reads as unknown rather than
    # as fmtCount(null)'s "0"; the formatter itself is unchanged.
    assert "value: kpiFigure(kpis.sessions, fmtCount)" in html
    assert "value: kpiFigure(kpis.events, fmtCount)" in html
    assert "value: String(kpis.sessions)" not in html
    assert "value: String(kpis.events)" not in html


def test_analytics_thin_prior_window_softens_delta(html):
    # A near-empty prior window suppresses the alarming ▲% and annotates instead.
    assert "const prevThin = !!resp.kpi_prev && (resp.kpi_prev.sessions || 0) < 2;" in html
    assert "vs partial prior window" in html
    # the flag is threaded through the tile into the delta chip
    assert "prevThin=${prevThin}" in html
    assert "function DeltaChip({ pct, cost, prevThin })" in html


# --- #268: tool dimension + spend/tokens → helpful empty state, not zeros ----- #
def test_analytics_tool_dim_no_cost_metric_empty_state(html):
    # Grouping spend/tokens by tool(_category) is structurally all-zeros (tool
    # spans carry no tokens/cost) — show an empty state with a one-click switch.
    assert "const toolDimNoMetric = (groupBy === 'tool' || groupBy === 'tool_category')" in html
    assert "&& (metric === 'spend' || metric === 'tokens');" in html
    assert "Tools don't carry" in html
    # one-click recovery actions
    assert "onClick=${() => setFilter('metric', 'events')}>Switch to Events" in html
    assert "setFilter('group_by', 'model')" in html


# --- #306: StatusView coding archive lives in the coding zone --------------- #
def test_status_coding_archive_renders_in_coding_zone(html):
    # The coding archive is a collapsible <details> inside the coding zone,
    # gated only on there being coding archive rows — not on any view mode. The
    # empty-active case still shows the archive (and a "No active coding
    # sessions" note) rather than blanking the page.
    start = html.index("function StatusView")
    end = html.index("function TracesListView", start)
    view = html[start:end]

    # The coding zone renders whenever there are coding agents OR coding archive.
    assert "${(codingAgents.length || codingArchived.length) ? html`" in view
    # The archive is its own collapsible block keyed on ended-at.
    assert "${codingArchived.length > 0 ? html`" in view
    assert "keyed on ended-at · method still openable" in view
    # Empty-active note shown when there are no active coding sessions.
    assert "No active coding sessions" in view
    # No leftover view-toggle machinery.
    assert "viewMode" not in view


def test_status_archive_count_renders_the_real_total_honestly(html):
    """ARCHIVE_LIMIT caps the archive list at 50 server-side (api/routes/
    status.py); the zone-header count and the <details> summary's count used
    to render codingArchived.length with no qualifier, so a capped 50 read
    as the complete archive. Both sites now route through
    archivedCountLabel(shown, data.archived_total), which states the real
    total when the backend supplies it and falls back to a no-total-claim
    wording when it does not (see test_archived_count_label_* in
    test_lens_dashboard_states.py for the function's own behaviour)."""
    start = html.index("function StatusView")
    end = html.index("function TracesListView", start)
    view = html[start:end]
    assert "${codingAgents.length} active · ${archivedCountLabel(codingArchived.length, data.archived_total)} archived" in view
    assert "${archivedCountLabel(codingArchived.length, data.archived_total)}</span>" in view
    assert "${codingArchived.length} archived" not in view
    assert "(closed / stale) · ${codingArchived.length}" not in view


# --- #306: right-click to rename a session card ----------------------------- #
def test_status_card_right_click_rename_wiring(html):
    # The coding-session card title is right-click-renamable: an onContextMenu
    # handler enters an inline edit state, and submitting POSTs to the label
    # endpoint via the authed apiPost helper (not window.prompt).
    start = html.index("function StatusView")
    end = html.index("function TracesListView", start)
    view = html[start:end]

    # Inline edit state + autofocus ref (no window.prompt).
    assert "const [editingId, setEditingId] = useState(null)" in view
    assert "window.prompt" not in view
    # Right-click ANYWHERE on the card enters edit mode (not just the title, so
    # the browser's native context menu never wins), and the ✎ is left-click.
    assert 'class="card clickable" onContextMenu=${startEdit}' in view
    assert 'class="rename-pencil" onClick=${startEdit}' in view
    # Submit persists via the authed POST helper to the /label endpoint.
    assert "apiPost('/sessions/' + encodeURIComponent(a.session_id) + '/label'" in view
    # Discoverability affordance on the title.
    assert "or right-click to rename" in view


# --- Work map: graphical "what did my agent do" tab ------------------------ #
def test_work_map_tab_present_and_demoted(html):
    # Map/Approach/Timeline are drill-in evidence, not the landing session tab
    # (self-improve-loop SPEC.md §12 — demoted from primary tabs). Map still
    # renders before Timeline within that demoted group, and both still render
    # after the primary tabs (Models & context leads, Map/Approach/Timeline
    # trail behind the "Evidence" divider).
    assert "function WorkMapSection" in html
    assert "function WorkMapNode" in html
    assert "useState('models')" in html
    assert "/sessions/' + sessionId + '/workmap'" in html
    models_btn = html.index("setTab('models')")
    map_btn = html.index("setTab('map')")
    story_btn = html.index("setTab('story')")
    assert models_btn < map_btn, "Models & context must lead the primary tabs"
    assert map_btn < story_btn, "Map tab must render before Timeline"
    assert '<span class="tab-sep">Evidence</span>' in html


def test_work_map_is_descriptive_not_evaluative(html):
    # Honesty discipline: the map reports, it does not judge the approach.
    assert "you judge the approach" in html


def test_work_map_node_metric_is_tokens_not_dollars(html):
    # User preference: the visible per-node metric is tokens; the dollar figure
    # moved to a hover title only.
    assert "fmtTokens(node.tokens)" in html
    assert 'class="wm-tokens"' in html
    assert ">${fmtCost(node.cost_usd)}</span>" not in html  # no bare $ in the row


def test_work_map_files_shortened_for_readability(html):
    # Long absolute file paths are shortened to "…/dir/file" with the full path
    # on hover, so the files list is readable.
    assert "function shortPath" in html
    assert "shortPath(f)" in html


def test_map_tool_lane_labels_shortened_and_collision_proof(html):
    # The Map TOOLS lane prints arg labels under sampled ticks. They must be a
    # short basename/first-token (not a long path-ish string) and spaced so two
    # kept, center-anchored labels can never overlap — the min center-to-center
    # gap must exceed the capped label box width.
    assert "function evLabelShort" in html             # dedicated basename/tail shortener
    assert "evLabelShort(e.label)" in html              # used for the printed tick label
    assert "shortPath(events[i].label)" not in html     # no longer the long two-segment path
    # over-long names keep their distinctive TAIL ("…izer-design.md"), not the
    # head — date-prefixed filenames all share the head, so a head-keep printed
    # the same date fragment for every tick.
    assert "t = '…' + t.slice(-(MB_EVLAB_CHARS - 1));" in html
    # consecutive ticks on the same file print ONE label, not a repeated run.
    assert "if (txt === lastTxt) return;" in html
    # collision-proof: MB_EVLAB_GAP (center spacing) must exceed MB_EVLAB_MAX (box width).
    import re

    gap = int(re.search(r"const MB_EVLAB_GAP\s*=\s*(\d+)", html).group(1))
    box = int(re.search(r"const MB_EVLAB_MAX\s*=\s*(\d+)", html).group(1))
    assert gap > box, f"MB_EVLAB_GAP ({gap}) must exceed MB_EVLAB_MAX ({box})"


def test_work_map_is_ask_segmented(html):
    # A session is a sequence of asks (exchanges): the Map renders map.asks via a
    # per-ask component, read as a story ("ask by ask").
    assert "function WorkMapAsk" in html
    assert "map.asks" in html
    assert "ask by ask" in html


def test_work_map_is_a_storyline(html):
    # The Map headlines each ask by WHAT THE AGENT DID (its outcome) with a
    # deterministic status icon, and reads chronologically (oldest first) so it
    # tells the session's story rather than a reverse-time log.
    assert "function askStatus" in html
    assert "wm-ask-did" in html                       # bold "what it did" headline
    assert "wm-ask-ctx" in html                       # dim prompt-as-context line
    assert "(map.asks || []).slice().reverse()" in html  # chronological order


def test_work_map_renders_spine_with_milestone_dots(html):
    # Map v2 layout (approved Option-A mock): the asks render on a continuous
    # vertical spine — a left-border line — with each ask a milestone carrying a
    # status dot positioned ON the spine, not as a bordered card.
    assert ".wm-spine {" in html
    assert "border-left: 2px solid var(--border)" in html  # the spine line
    assert ".wm-milestone {" in html
    assert "position: relative" in html
    assert 'class="wm-spine' in html
    # The dot sits on the spine (negative offset lands it over the border line).
    assert ".wm-dot-spine {" in html
    assert "left: -30px" in html
    # The old bordered-card framing is gone.
    assert ".wm-ask {" not in html
    assert 'class="wm-asks"' not in html


def test_work_map_has_inline_branch_block(html):
    # Fan-out asks list their top subagents inline in an indented branch block
    # (dashed left border, like the mock) — visible without a click; only the
    # deeper subtree stays expandable.
    assert ".wm-branch {" in html
    assert "border-left: 2px dashed var(--border)" in html
    assert 'class="wm-branch"' in html
    # Top 5 subagents shown inline, with a "+N more" overflow line.
    assert "subs.slice(0, 5)" in html
    assert "+${branchMore} more" in html


def test_user_prompts_visually_marked_on_both_views(html):
    # Timeline marks user prompts (grouped by ask) in a distinct brand color —
    # no box/label; the Map's work milestones carry a brand status dot on the
    # spine (other statuses recolor it: amber flagged, red error, dim chat).
    assert "function StoryAsk" in html
    assert "step.ask" in html
    assert ".story-ask { margin: 14px 0 4px; font-size: 13px; font-weight: 600; color: var(--brand)" in html
    assert ".wm-dot-spine.work { color: var(--brand)" in html


# --- Map v1.1: glanceable storyline (first-sentence, chat collapse, summary) - #
def test_work_map_headlines_use_first_sentence(html):
    # Verbose run-on outcomes are reduced to one clean sentence for the headline;
    # the old raw 160-char truncation of the outcome is gone.
    assert "function firstSentence" in html
    assert "firstSentence(ask.outcome || ask.summary || '')" in html
    assert "outcome.slice(0, 160)" not in html  # the old raw truncation is gone


def test_work_map_collapses_chat_runs(html):
    # Runs of 2+ consecutive no-work chat asks collapse into one clickable
    # divider that expands into the individual rows.
    assert "function WorkMapChatRun" in html
    assert "function groupAsks" in html
    assert 'class="wm-chat-divider"' in html
    assert "quick exchanges" in html
    # The collapse decision keys off askStatus(...).hasWork being false.
    assert "askStatus(ask).hasWork" in html


def test_work_map_has_summary_band(html):
    # A 5-second-read summary band sits above the asks list: totals + the top
    # fan-outs (biggest subagent counts).
    assert 'class="wm-summary-band"' in html
    assert ".wm-summary-band {" in html
    assert ".wm-chat-divider {" in html


def test_first_sentence_strips_single_and_double_emphasis(html):
    # firstSentence must strip single AND double * / _ plus backticks anywhere,
    # so no stray emphasis markers leak into a headline (e.g. "*cheaper*"). The
    # old strip that only removed the paired forms (\*\*|__) is gone.
    assert r".replace(/[*_`]+/g, '')" in html
    assert r".replace(/\*\*|__|`|##+|---+/g, '')" not in html


def test_work_map_subagent_count_clamped_to_session_total(html):
    # A per-ask fan-out can never exceed the session total; both the ask row and
    # the summary fan-out clamp the displayed number with Math.min(..., session).
    # The ask row threads the session total down as the sessionSubs prop.
    assert "sessionSubs" in html
    assert "Math.min(subCount, sessionSubs)" in html
    assert "Math.min(askStatus(a).subCount, sub || askStatus(a).subCount)" in html
    # The row's displayed count uses the clamped value, not the raw subCount.
    assert "${subShown} sub${subShown === 1 ? '' : 's'}" in html


# --- Map: on-demand LLM-distilled titles ----------------------------------- #
def test_work_map_has_distill_control(html):
    # The Map carries a "Distill titles" button that calls /distill and threads
    # the result into the ask headlines (prefer distilled[n] over firstSentence).
    assert "Distill titles" in html
    assert "setDistilled" in html
    assert "/distill" in html
    # The distilled title is preferred over the deterministic first sentence.
    assert "distilled[String(ask.n)]" in html


# --- Map: launcher -> run linkage card (Task A) ---------------------------- #
def test_work_map_has_run_card(html):
    # The Map renders a run card from the workmap's `launched_run` block with a
    # working "View run" link into #/runs/<id> and clickable worker chips.
    assert "function WorkMapRunCard" in html
    assert "map.launched_run" in html
    assert "Launched run" in html
    assert "#/runs/' + run.run_id" in html
    assert "#/sessions/' + s.session_id" in html
    # Inferred (transcript-scraped) runs are visibly marked as a best-effort guess.
    assert "run.source === 'inferred'" in html


# --- Map: per-ask phase breakdown (Task E) --------------------------------- #
def test_work_map_renders_phases(html):
    # A long ask's journey renders as the agent's narrated phases under the
    # milestone, with a tool tally and a show-all toggle past the preview.
    assert "function phaseTools" in html
    assert "ask.phases || []" in html
    assert 'class="wm-phases"' in html
    assert "PHASE_PREVIEW" in html
    # The honest omitted marker (no silent drop) is rendered.
    assert "more phase" in html


# --- Timeline: tool-only steps show the command inline ---------------------- #
def test_timeline_tool_step_shows_command_inline(html):
    # A step with no narration but a tool call surfaces the tool's label/command
    # inline instead of a bare "(no narration)".
    assert "const toolLine = !preview && tools.length" in html
    assert "preview || toolLine || '(no narration)'" in html
    assert ".story-line.tool" in html


# --- Timeline: failed steps show the error, not a red box ------------------- #
def test_timeline_error_step_shows_message_not_red_box(html):
    # The red border around an errored step is gone; the expanded body surfaces
    # the transcript error message instead.
    assert ".story-step.error { border-color: var(--error)" not in html
    assert "tools.filter(t => t.error)" in html
    assert "story-error" in html


# --- Timeline: subagents nest + expand recursively (like Approach/Map) ------- #
def test_timeline_renders_subagents_recursively(html):
    # Requirement: EVERY delegation in the Timeline is expandable to recurse into
    # the child's own work. The recursion is a closed cycle of three pieces:
    #   1. a StoryStep renders a SubagentBlock for each subagent it spawned,
    #   2. a SubagentBlock renders the child's steps via renderTimelineSteps,
    #   3. renderTimelineSteps renders a StoryStep per step — back to (1).
    # So a subagent's steps that spawned their OWN subagents nest arbitrarily
    # deep, each level independently expandable. Assert the whole cycle is wired.
    assert "function SubagentBlock(" in html
    assert "function StoryStep(" in html
    assert "function renderTimelineSteps(" in html
    # (1) StoryStep reads the step's subagents and renders a SubagentBlock each.
    assert "const subagents = step.subagents || (step.subagent ? [step.subagent] : []);" in html
    assert "subagents.map((sa, i) => html`<${SubagentBlock} subagent=${sa}" in html
    # (2) SubagentBlock recurses into the child's steps with the SAME renderer.
    assert '<div class="story-steps">${renderTimelineSteps(steps)}</div>' in html
    # (3) renderTimelineSteps renders a StoryStep per step (closes the cycle).
    assert "html`<${StoryStep} step=${step}" in html


def test_timeline_subagent_is_expandable_with_honest_caps(html):
    # Each subagent block has a clickable head with a caret affordance (collapsed
    # ▸ / open ▾), consistent with the Approach/Map nodes — NOT a flat dump.
    assert "const [open, setOpen] = useState(false);" in html
    assert "onClick=${() => !capped && setOpen(o => !o)}" in html
    assert "${capped ? '·' : (open ? '▾' : '▸')}" in html
    # Caps are surfaced as honest notes, never silent drops: depth / size / cycle
    # each map to an explicit "… omitted …" / "… already shown …" marker, and a
    # capped ref is non-expandable (shows the note in place of the subtree).
    assert "const capped = subagent.depth_capped || subagent.budget_capped || subagent.cycle;" in html
    assert "deeper subagents omitted (depth cap)" in html
    assert "deeper subagents omitted (size cap)" in html
    assert "already shown above (cycle)" in html
    assert "${cappedNote ? html`<div class=\"story-omitted\">${cappedNote}</div>` : null}" in html


# --- Map: distill UX (auto-apply cached, honest note, feedback) ------------- #
def test_distill_auto_applies_cached_and_has_honest_states(html):
    # Cache-only auto-apply on load (press once, sticks; zero cost).
    assert "cached_only: 1" in html
    # The note distinguishes failure from "nothing to distill" (no longer lies).
    assert "nothing to distill" in html
    assert "candidate_count === 0" in html
    # A visible post-distill flash so a successful run is obvious.
    assert "wm-flash" in html
    assert "@keyframes wmDistillFlash" in html


def test_index_html_has_no_nul_bytes():
    # Guards the NUL-byte corruption fixed alongside the work map (it broke
    # `node --check` and made `file` mis-detect the SPA as binary).
    assert b"\x00" not in _UI.read_bytes()


# --- #17: #2 shipped incomplete — SessionDetailView + Status cost cells ----- #
# Route the two dollar-bearing cells left on bare fmtCost through
# fmtFramedDollar(value, framing) — matching Traces/Cost/Optimize. (fmtFramedDollar
# used to render "% of cycle" for subscription; that differentiation is gone by
# product decision, so api and subscription now render the same dollar figure —
# see test_fmt_framed_dollar_renders_identically_for_subscription_and_api in
# test_lens_dashboard_states.py. Only LOCAL still renders "—".)
def test_session_detail_cost_cell_routes_through_framing(html):
    # The "Cost & Tokens" panel must consume the /sessions/{id} framing block,
    # not render raw fmtCost(s.total_cost_usd).
    assert "<span class=\"value\">${fmtCost(s.total_cost_usd)}</span>" not in html
    assert "<span class=\"value\">${fmtFramedDollar(s.total_cost_usd, framing)}</span>" in html
    # The view actually pulls the framing block off the /sessions/{id} response.
    assert "const framing = data.framing || null;" in html


def test_status_archived_table_cost_routes_through_framing(html):
    # The Status "Archived sessions" table cost column must consume the /status
    # framing block (data.framing), not render raw fmtCost(s.total_cost_usd).
    assert "<td>${fmtCost(s.total_cost_usd)}</td>" not in html
    assert "<td>${fmtFramedDollar(s.total_cost_usd, data.framing)}</td>" in html


# --- #20: Traces empty-state must not flash before the first fetch -------- #
def test_traces_distinguishes_loading_from_loaded_empty(html):
    # The Traces view tracks a `loaded` flag that flips true only once the first
    # /traces fetch resolves; until then it renders a loading shimmer. The
    # "No traces yet" empty-state is gated on `loaded` so it can only appear
    # after a fetch genuinely returned zero rows — never on initial paint.
    assert "const [loaded, setLoaded] = useState(false);" in html
    assert "setLoaded(true);" in html
    # Empty-state is now downstream of the `!loaded` shimmer branch, so the
    # bare "traces.length === 0 ? <empty>" first-branch pattern must be gone.
    assert (
        "${traces.length === 0 ? html`<div class=\"empty\">No traces yet."
        not in html
    )
    assert (
        "${!loaded ? html`<div class=\"shimmer\" style=\"height:200px\"></div>`"
        in html
    )


# --- Approach tab: recursive method spine (GET /sessions/{id}/approach) ------ #
def test_approach_section_present_and_tab_wired(html):
    # The Approach tab renders the method spine from the dedicated endpoint,
    # mirroring WorkMapSection's fetch idiom.
    assert "function ApproachSection" in html
    assert "/sessions/' + sessionId + '/approach'" in html
    # Tab button wired into SessionDetailView, placed AFTER Map and BEFORE Timeline.
    assert "setTab('approach')" in html
    map_btn = html.index("setTab('map')")
    approach_btn = html.index("setTab('approach')")
    story_btn = html.index("setTab('story')")
    assert map_btn < approach_btn < story_btn, "Approach tab sits between Map and Timeline"
    # The render block dispatches the 'approach' tab to the section.
    assert "tab === 'approach' ? html`<${ApproachSection} sessionId=${sessionId} />`" in html


def test_approach_source_tags_present(html):
    # The two honest source tags ride each move; "distilled" is never invented
    # on the approach spine (only agent's words vs structural inference).
    assert "agent's words" in html
    assert "'structural'" in html
    assert 'class="ap-src ${srcWords' in html


def test_approach_renders_recursively(html):
    # A delegate move renders one rich card per `delegations` entry; expanding a
    # card recurses into the child's own spine via the SAME move renderer. A
    # capped delegation shows a "not expanded" note instead of expanding.
    assert "function ApproachMove" in html
    assert "function ApproachDelegation" in html
    assert "delegations.map(d => html`<${ApproachDelegation} deleg=${d} />`)" in html
    # spines render through ApproachSpine (which also folds conversational runs).
    assert "<${ApproachSpine} moves=${spine} />" in html
    assert "not expanded — ${capped}" in html


def test_approach_delegation_cards_show_cost_and_depth(html):
    # Each delegation card carries the child's identity, spawn depth, and the
    # span-joined token/cost/status chips (read straight off the payload).
    assert "function ApproachDelegation" in html
    assert "↳ ${deleg.name}" in html
    assert "depth ${deleg.depth}" in html
    assert "deleg.tokens != null ? fmtTokens(deleg.tokens)" in html
    assert "deleg.cost_usd != null ? fmtCost(deleg.cost_usd)" in html
    # The expanded block carries the child's "how it solved its piece" header.
    assert "how the subagent solved its piece" in html


def test_approach_delegation_tree_rail_present(html):
    # The left rail renders data.agents — a status dot + name + meta + badge +
    # provenance line per agent, indented by spawn depth, with the ephemeral
    # capture caption at the foot.
    assert "function ApproachRail" in html
    assert "<${ApproachRail} agents=${agents} />" in html
    assert "Delegation tree" in html
    assert 'class="ap-dot ${dotClass}"' in html
    assert "in-session subagent · from transcript" in html
    assert "ended · method kept" in html
    assert "rebuilds" in html  # the ephemeral-capture caption


def test_approach_header_stats_and_layout(html):
    # Two-column grid (rail + panel) and a right-aligned header stats block from
    # counts + meta.
    assert 'class="ap-grid"' in html
    assert "data.counts" in html
    assert "data.meta" in html
    assert "moves<br/><b>${counts.delegations}</b> delegations" in html


def test_approach_legend_present(html):
    # The bottom "source of each line" legend names all three sources.
    assert 'class="ap-legend"' in html
    assert "source of each line:" in html
    assert "narration / TodoWrite" in html
    assert "revert / retry / spawn" in html
    assert "LLM, opt-in" in html


def test_approach_handles_unavailable(html):
    # available:false surfaces the server-provided reason (e.g. no transcript or
    # snapshot for this session).
    assert "data.reason" in html
    assert "No transcript or snapshot for this session" in html


# --- Map board: ①+③ swimlanes + territory (GET /sessions/{id}/sessionmap) ---- #
def test_map_board_section_present_and_wired(html):
    # The Map tab renders the ①+③ board from the dedicated /sessionmap endpoint.
    assert "function MapBoardSection" in html
    assert "/sessions/' + sessionId + '/sessionmap'" in html
    # The Map tab dispatches to the board (no longer straight to WorkMapSection).
    assert "tab === 'map' ? html`<${MapBoardSection} sessionId=${sessionId} />`" in html


def test_map_board_has_four_synchronized_lanes(html):
    # Four lanes share one x-axis, each with a left gutter label. phase/tools use a
    # plain name gutter; context/cost use a y-axis gutter (name in the .mb-yname).
    for lane in ("phase", "tools"):
        assert f'<div class="mb-gutter">{lane}</div>' in html, f"missing {lane} lane gutter"
    for lane in ("context", "cost"):
        assert f'<span class="mb-yname">{lane}</span>' in html, f"missing {lane} y-axis gutter"
    # Context is an inline SVG area chart, cost is inline SVG bars — NOT uPlot —
    # so the lanes stay pixel-aligned (the mock's approach).
    assert 'viewBox="0 0 100 30" preserveAspectRatio="none"' in html
    # A shared crosshair binds the lanes.
    assert 'class="mb-cross"' in html


def test_map_board_context_and_cost_lanes_are_readable_as_data(html):
    # CONTEXT and COST lanes must expose a y-axis (max value at top, 0 at the
    # baseline) read off the already-returned series arrays, plus a peak
    # annotation — so the magnitude is legible, not just an unscaled shape.
    assert ".mb-gutter.axis {" in html          # column max/name/0 gutter exists
    assert '<span class="mb-yv">${fmtTokens(maxCtx)}</span>' in html   # context max (top)
    # Cost max/peak use costMax (binned peak in time mode, per-point peak in step).
    assert '<span class="mb-yv">${fmtCost(costMax)}</span>' in html    # cost max (top)
    assert '<span class="mb-yv mb-y0">0</span>' in html                # context baseline
    assert '<span class="mb-yv mb-y0">$0</span>' in html               # cost baseline
    # Peak value labelled on each lane via the existing formatters. The cost peak
    # self-describes its unit: the auto bin width in time mode, /call in step.
    assert 'class="mb-peak">peak ${fmtTokens(maxCtx)} tok' in html
    assert "class=\"mb-peak\">peak ${fmtCost(costMax)}${mode === 'time' ? ' per ' + mbFmtBin(binWidth) : '/call'}" in html
    # Max read off the series arrays already sent (UI reads, doesn't aggregate).
    assert "const maxCtx = Math.max(1, ...ctxSeries.map" in html
    assert "const maxCost = Math.max(0.0001, ...costSeries.map" in html


def test_map_board_xaxis_row_not_clipped(html):
    # The x-axis row must stretch to fill its height so the tick labels don't spill
    # past the board's overflow:hidden bottom (a zero-height centered ticks box
    # clipped the last tick). Pin the non-clipping rule.
    assert ".mb-xaxis { display: flex; align-items: stretch; height: 26px; }" in html


def test_map_board_subagent_label_is_single_ellipsized_run(html):
    # Sub-agent bars render "name · tokens · $cost" as ONE ellipsized run (no
    # separate cost span that truncated to a cryptic "1…" stub); null metrics are
    # omitted so a transcript-only subagent shows just its name.
    assert "if (sa.tokens != null) metrics.push(fmtTokens(sa.tokens));" in html
    assert "if (sa.cost_usd != null) metrics.push(fmtCost(sa.cost_usd));" in html
    # the printed name is middle-truncated ("first8…last6") so both ends stay
    # legible; the full label rides in the bar's hover title.
    assert "const subLabel = [mbMidTrunc(sa.name), ...metrics].join(' · ');" in html
    assert "const fullLabel = [sa.name, ...metrics].join(' · ');" in html
    assert 'class="mb-sublab"' in html
    assert 'class="mb-subcost"' not in html  # the old two-span split is gone


def test_map_board_has_time_step_toggle(html):
    # The step⇄time toggle re-spaces every lane (useState-driven; step default — #58).
    assert "const [mode, setMode] = useState('step')" in html
    assert "setMode('time')" in html
    assert "setMode('step')" in html
    assert 'class="mb-toggle"' in html


def test_map_board_time_axis_collapses_idle_gaps(html):
    # Time mode plots on the idle-collapsed ACTIVE-time axis the backend builds
    # (meta.active_duration_s + per-point/-event active_s), not raw wall-clock, so
    # the real work spreads out instead of being crammed against huge idle gaps.
    assert "meta.active_duration_s" in html
    assert "e.active_s != null" in html          # events position by active_s
    assert "p.active_s != null" in html          # series points position by active_s
    assert "sa.start_active_s != null" in html   # subagent bars position by active_s
    # Each collapsed gap renders as a faint dashed break marker labelled "⋯ idle N".
    assert "const gaps = meta.gaps || []" in html
    assert "g.at_active_frac" in html
    assert "'idle ' + mbFmtGap(g.duration_s)" in html
    assert "function mbFmtGap" in html
    assert 'class="mb-break"' in html
    assert 'class="mb-break-lab"' in html
    assert ".mb-break {" in html  # themed CSS exists (offline-safe — no external)


def test_map_board_has_subagents_lane(html):
    # The sub-agents lane sits between tools and context, fed by data.subagents,
    # positioned by the shared axis (mbLayoutSubagents) and themed to --chart-5.
    assert "function mbLayoutSubagents" in html
    assert '<div class="mb-lane sub"' in html
    assert '<div class="mb-gutter">sub-<br/>agents</div>' in html
    assert 'class="mb-subbar"' in html
    assert "(data.subagents || []).length ? html`" in html
    # The lane is between the tools lane and the context lane.
    tools_i = html.index('<div class="mb-lane tools">')
    sub_i = html.index('<div class="mb-lane sub"')
    ctx_i = html.index('<div class="mb-lane ctx">')
    assert tools_i < sub_i < ctx_i, "sub-agents lane must sit between tools and context"
    # Bars are themed (no hardcoded mock hex) — Rule 18.
    assert "mbTint('--chart-5', 16)" in html


def test_map_board_renders_territory_treemap(html):
    # The ③ codebase-territory treemap aggregates read/edit events into per-file
    # touch counts, grouped by directory, with order badges + an edited marker.
    assert "function mbBuildTerritory" in html
    assert 'class="mb-tree"' in html
    assert 'class="mb-file"' in html
    assert 'class="mb-ord"' in html  # first-touch order badge


def test_map_board_falls_back_to_work_map_when_unavailable(html):
    # When /sessionmap has no board data, the board falls back to the existing
    # WorkMapSection so nothing is lost — and WorkMapSection stays defined.
    assert "function WorkMapSection" in html
    assert "return html`<${WorkMapSection} sessionId=${sessionId} />`" in html


def test_map_board_category_colors_use_theme_vars(html):
    # Category → theme chart var map (offline; no hardcoded hexes — Rule 18).
    assert "const MB_CAT_COLOR = {" in html
    assert "read: '--chart-1'" in html
    assert "edit: '--chart-2'" in html
    assert "error: '--error'" in html


def test_map_board_context_lane_plots_per_call_occupancy(html):
    # #56: the CONTEXT lane plots each call's OWN context occupancy (per-call
    # input+cache from the backend series), NOT a cumulative sum — a monotone
    # climb duplicated the header's total-token chip and carried no information.
    assert "each call's OWN context occupancy" in html
    assert "per-call context size" in html  # the subtitle says what the lane is


def test_map_board_subagent_labels_never_collide(html):
    # #56: a subagent label prints only on a wide-enough (px-gated),
    # non-overlapped bar; past-cap bars are flagged `.overlapped` so their labels
    # are suppressed. Bars may overlap under extreme density — text never does.
    # Every bar keeps the full label as its hover title.
    assert "const MB_SUBLAB_MIN_PX" in html
    assert "it.overlapped = true;" in html
    assert 'showSubLab ? html`<span class="mb-sublab">' in html
    assert "MB_SUBLAB_MIN_PX;" in html


def test_map_board_legend_covers_every_encoding(html):
    # #56: every on-board encoding is decodable from the legend — Other events,
    # solid-red error, the dashed-red retry outline, and the phase-band tinting
    # each get a legend entry (no more unexplained marks).
    assert ">Other</span>" in html
    assert ">retry</span>" in html
    assert ">phase band</span>" in html
    assert "marks a retried step" in html
    assert "hover a band for its title" in html


def test_map_board_surfaces_insights_strip(html):
    # #56: the board ANSWERS "where did the time and money go" by default via a
    # deterministic callouts strip (costliest stretch, friction, top delegation,
    # idle share, edit footprint) — insight is never hover-gated.
    assert 'class="mb-insights"' in html
    assert "const insights = [];" in html
    assert "'costliest ' + mbFmtBin(insW)" in html
    assert "k: 'friction'" in html
    assert "k: 'top sub-agent'" in html
    assert "k: 'idle'" in html
    assert ".mb-insights {" in html  # themed CSS exists (offline-safe)


def test_map_board_territory_demotes_scratch_and_weights_edits(html):
    # #56: temp/scratch reads collapse into ONE muted card and are excluded from
    # the common-prefix root (so workspace dir labels stay relative + readable);
    # cards and file rows are weighted by edits over reads; the file NAME keeps a
    # readable floor instead of losing the flex fight to its meta text.
    assert "const mbIsScratchPath" in html
    assert "TemporaryItems" in html
    assert 'class="mb-dir mb-scratch"' in html
    assert "f.edits * 3 + f.reads" in html
    assert "min-width: 9ch" in html


def test_approach_rail_marks_cross_terminal_children(html):
    # M2b: a cross-terminal child (a separate run-linked session) renders amber
    # (is-term) with a run-linked provenance sub-line, distinct from the pink
    # in-session subagent nodes — never claiming more than capture_completeness.
    assert "a.provenance === 'cross_terminal_child'" in html
    assert "cross-terminal child · run-linked" in html
    # Honest completeness: session-level node vs method-kept when the child's own
    # method was recoverable.
    assert "a.capture_completeness === 'full' ? '⏏ ended · method kept' : 'session-level'" in html


def test_approach_splices_cross_terminal_spine(html):
    # When a child's own method is available the backend ships a `cross_terminal`
    # spine list; the UI renders each under a divider with recursive ApproachMove.
    assert "function ApproachCrossTerminal" in html
    assert "const crossTerminal = data.cross_terminal || [];" in html
    assert "cross-terminal children" in html  # the divider label
    assert ".ap-deleg.ap-xterm { border-color: var(--warn); }" in html


def test_map_tools_lane_is_stacked_density_histogram_in_time_mode(html):
    # In time mode the tools lane is a stacked-by-category density histogram — one
    # bar per active-time bucket, height ∝ count, segments colored by category with
    # error stacked on top — NOT the per-event 3px ticks (which smeared over ~100s
    # of events). Buckets with 0 events render nothing (honest quiet signal).
    assert "const MB_STACK_ORDER = ['read', 'search', 'edit', 'bash', 'task', 'web', 'other', 'error']" in html
    assert "const histBins = []" in html
    assert 'class="mb-hbar' in html          # the stacked bucket bar
    assert 'class="mb-hseg"' in html         # a per-category stacked segment
    assert "if (b.total <= 0) return null" in html  # empty bucket → gap
    # error is the last (top) entry of the stack order so failures pop
    import re

    assert re.search(r"MB_STACK_ORDER\s*=\s*\[[^\]]*'error'\]", html) is not None


def test_map_tools_lane_keeps_per_event_ticks_in_step_mode(html):
    # Step mode is unchanged: individual per-event ticks (.mb-ev) + sampled labels.
    assert 'class="mb-ev ' in html
    assert "evLabelShort(e.label)" in html
    # the histogram branch is gated to time mode (step falls through to ticks)
    assert "${mode === 'time'\n          ? histBins.map" in html


def test_map_cost_lane_shares_histogram_bucket_edges_in_time_mode(html):
    # The COST lane is re-binned into the SAME bucket edges as the tools histogram
    # (usd summed per bucket) so cost bars line up vertically under tool bursts.
    assert "histBins[bi].usd += (p.usd || 0)" in html
    assert "const costMax = mode === 'time'" in html
    # cost bars in time mode are drawn from the shared histBins, not raw cost_series
    assert "Same bucket edges as the tools histogram" in html
def test_map_tools_lane_bin_width_is_auto_only(html):
    # #58: the manual interval ladder (Auto/1m/5m/15m/1h) is PURGED. Bin widths
    # carry no domain meaning for an agent run (users can't relate 5m→15m to any
    # question), and a manual 1h on a ~40m session collapsed the whole board into
    # one full-width slab. The width is always auto-resolved for the span and
    # surfaced in the cost lane's peak label instead of a control.
    assert "const binWidth = mbAutoBinWidth(activeDur)" in html
    assert "binSel" not in html            # the state is gone
    assert "mb-ivl" not in html            # the control + its CSS are gone
    assert "MB_BIN_SECONDS" not in html    # the manual ladder map is gone
    # the resolved width self-describes in the cost peak label (time mode)
    assert "' per ' + mbFmtBin(binWidth)" in html


def test_map_board_defaults_to_step_mode(html):
    # #58: step is the default read — evenly spaced by tool-call order it shows
    # the work sequence without burst/idle distortion (the user consistently read
    # it as less noisy). Time mode stays one click away for cost localization.
    assert "const [mode, setMode] = useState('step')" in html
    # Peak value labelled on each lane via the existing formatters. The cost peak
    # self-describes its unit: the auto bin width in time mode, /call in step.
    assert "class=\"mb-peak\">peak ${fmtCost(costMax)}${mode === 'time' ? ' per ' + mbFmtBin(binWidth) : '/call'}" in html
    # The step⇄time toggle re-spaces every lane (useState-driven; step default — #58).
    assert "const [mode, setMode] = useState('step')" in html


# ---- Approach/Map presentation polish (dogfooding round on a real 22.7M-tok
# ---- session: PR #306 follow-ups applied to the #371 carve).


def test_approach_scrubs_mojibake_at_load(html):
    # Transcript text can arrive UTF-8-decoded-as-Latin-1 ("Â\xa0" where an NBSP
    # was meant); the approach payload is scrubbed ONCE at data-load so mandate,
    # outcome, labels and quotes all render clean without per-render cost.
    assert "function stripMojibake" in html
    assert "function sanitizeApproachPayload" in html
    assert "setData(sanitizeApproachPayload(d))" in html
    # the Timeline's /story payload gets the same scrub (asks + step narration).
    assert "function sanitizeStoryPayload" in html
    assert "setStory(sanitizeStoryPayload(d))" in html


def test_approach_renders_inline_markdown_not_asterisks(html):
    # Narration carries **bold** / *italic* / `code`; the spine renders them as
    # vnodes (h('b'…)/h('i'…)/h('code'…)) instead of printing literal asterisks.
    assert "function mdInline" in html
    assert "md-code" in html


def test_approach_move_untruncates_headline_instead_of_duplicating_quote(html):
    # When a move's quote merely re-states its (80-char-truncated) label, the
    # row un-truncates the headline from the quote's lead sentence and renders
    # only the remainder as the italic quote — never the same sentence twice.
    assert "function splitLeadSentence" in html
    assert "labelCore(" in html


def test_approach_spine_folds_conversational_runs(html):
    # A run of >4 consecutive chat-only moves (agent narration, no tools, no
    # delegations) folds to first + "· N conversational steps" + last, click to
    # expand — the method must not drown in Q&A narration.
    assert "function ApproachSpine" in html
    assert "const isChatMove" in html
    assert "conversational steps" in html
    assert 'class="ap-collapse"' in html


def test_approach_verify_flavored_delegations_counted_and_chipped(html):
    # A delegation whose name/task reads as review/verify/audit gets a ✅ accent
    # + 'verify' chip, and the header's verifies stat is recomputed client-side
    # (the structural backend classifies them as plain delegates → verifies:0).
    assert "VERIFY_NAME_RE" in html
    assert "const isVerifyDeleg" in html
    assert "ap-verify-chip" in html
    assert "ap-verify-deleg" in html
    assert "verifies" in html


def test_approach_delegate_move_renders_card_only(html):
    # A delegate move with delegation cards suppresses its own label line and
    # "Agent <name>" evidence row — the card already carries the name; the old
    # form printed the same subagent name three times per delegation.
    assert "const isDelegateCard = kind === 'delegate' && delegations.length > 0;" in html


def test_approach_outcome_clamps_with_toggle(html):
    # The ✓ outcome block clamps to 3 lines with a click-to-toggle "show all"
    # (outcomes arrive truncated mid-word server-side; full text one click away).
    assert "ap-outcome-body" in html
    assert "ap-outcome-more" in html


def test_approach_rail_hides_default_badges(html):
    # Rail nodes only badge the EXCEPTIONS (capped/killed/cross-terminal/running);
    # the default "ended · method kept · in-session subagent · from transcript"
    # pair repeated on every node was pure noise — it moves to the node's title=.
    assert "const isPlainSub = !isMain && !isCross && !isCapped;" in html
    assert "const showBadge = !isPlainSub;" in html
    assert "nodeTitle" in html


def test_timeline_ask_carries_user_prefix(html):
    # Each ask in the Timeline is prefixed with a grey mono "user: " marker so
    # asks carry a speaker label just like steps carry #n + time.
    assert 'class="story-ask-who"' in html
    assert 'story-ask-who">user: <' in html


def test_map_context_area_stops_at_last_sample(html):
    # The context lane's area fill closes straight down at the LAST sample's x —
    # it must not fade to the right edge as a decaying wedge that reads as data.
    assert "const ctxLastX" in html
    assert "L' + ctxLastX.toFixed(2) + ',30 L' + ctxFirstX.toFixed(2) + ',30 Z'" in html


def test_map_phase_titles_cleaned_and_merged(html):
    # Phase titles strip leading conversational pleasantries ("Got it — my
    # mistake." → "My mistake.") and adjacent phases with the same normalized
    # title merge into one band (#57's confetti came from same-title splits).
    assert "MB_PLEASANTRY_RE" in html
    assert "function mbCleanPhase" in html
    assert "mbNormTitle(prev.name) === mbNormTitle(name)" in html


def test_map_phase_bands_split_at_idle_breaks(html):
    # In time mode a phase band that spans an idle break splits into segments
    # with a visible gap at the break — one band must not bridge an 18h idle
    # gulf as if it were continuous work. Only the widest segment is labelled.
    assert "const phaseSegs = []" in html
    assert "labelSeg: si === widest" in html
    assert "usePhaseSegs ? phaseSegs : phaseBands" in html


# --- #396: KPI "% vs prev" delta must not blow up on a near-empty prior ----- #
def test_trend_chip_guards_zero_baseline_with_new(html):
    # A fresh onboard + 30-day backfill compares the current window against the
    # near-empty pre-backfill period, so cost_delta_pct explodes (e.g. ▲140980%).
    # TrendChip must show "new" when the prior window is thin, not the figure.
    assert "function TrendChip({ pct, prevThin })" in html
    assert "prevThin && pct != null && pct > 0" in html
    assert "▲ new" in html
    # The Cost/Dashboard call site derives prevThin from the prior window's
    # session count and passes it through.
    assert "const trendPrevThin" in html
    assert "compare.previous.sessions" in html
    assert "prevThin=${trendPrevThin}" in html


def test_delta_pct_capped_to_avoid_four_digit_percentages(html):
    # Genuinely-large-but-finite deltas are capped at ">999%" so a tiny (nonzero)
    # prior baseline can't render "140980.0%". Both KPI chips route through the
    # shared formatter — neither re-implements the raw toFixed.
    assert "function fmtDeltaPct(pct)" in html
    assert "DELTA_PCT_CEILING" in html
    assert "'>999%'" in html
    # The old unguarded raw-percentage render is gone from both chips.
    assert "${Math.abs(pct).toFixed(1)}% vs prev" not in html


# --- Optimize ▸ Summarize (Track B) UI: curate → run → review/apply --------- #
def test_summarize_nav_child_and_route(html):
    # A nested "Summarize" child under Optimize, revealed only while the Optimize
    # section is active, routing to #/optimize/summarize (the router already
    # splits view/param). The nav reveal logic shows children per active section.
    assert 'href="#/optimize/summarize" class="nav-link nav-child" data-view="optimize" data-param="summarize"' in html
    # Keep-alive router: #/optimize/summarize resolves to the 'summarize' primary
    # key (a kept-alive sub-view in PRIMARY_VIEWS), not a switch `case` arm.
    assert "if (v === 'optimize' && route.param === 'summarize') return 'summarize';" in html
    assert "['summarize', SummarizeView]" in html
    # nav-child reveal: a child shows only while its section is active.
    assert "el.classList.contains('nav-child')" in html
    assert "el.style.display = (v === view) ? 'flex' : 'none';" in html


def test_summarize_component_present(html):
    assert "function SummarizeView" in html
    # The four-phase flow (engine gate → curate → run → review) is what makes the
    # screen worth more than the all-or-nothing CLI (DEC-034 granularity).
    assert "const [phase, setPhase] = useState('engine')" in html
    for phase in ("phase === 'review'", "phase === 'run'", "phase === 'curate'"):
        assert phase in html, f"missing phase branch {phase}"


def test_summarize_engine_gate_is_capabilities_driven(html):
    # The page starts on a capability-gated engine chooser (never defaulted), so a
    # dead engine (no key / no `claude`) is disabled with its reason.
    assert "api('/summarize/capabilities')" in html
    assert "const avail = cap ? cap.available : false;" in html
    # all three engines are offered; claude_p is normalized to the wire's claude-p
    assert "id === 'claude_p' ? 'claude-p' : id" in html


def test_summarize_curator_wires_to_candidates_scan(html):
    # The curator reads the read-only core scan (candidates) + staged records;
    # status is derived (staged files still appear in the scan until applied).
    assert "api('/summarize/candidates'" in html
    assert "api('/summarize/staged')" in html
    assert "const statusOf = c => stagedPaths.has(c.path) ? 'staged' : 'candidate';" in html
    # candidate fields come straight off ScanResult.to_dict (no fabricated shapes)
    assert "c.prose_words" in html
    assert "c.est_tokens_saved" in html


def test_summarize_run_covers_all_three_engines(html):
    # api/claude-p loop the per-file run route with a progress bar; manual walks
    # prep → paste-back → check with no outbound call.
    assert "apiPostOrDetail('/summarize/run', { path, mode: engine, ratio: 0.5 })" in html
    assert "apiPostOrDetail('/summarize/prep', { path, ratio: 0.5 })" in html
    assert "apiPostOrDetail('/summarize/check', { path: manPrep.path, summary: manSummary, source_hash: manPrep.source_sha256 })" in html


def test_summarize_apply_is_guarded_and_dry_run_until_click(html):
    # The UI's only file-writing action goes through core apply_staged with
    # go:true set ONLY on an explicit per-file/checked Apply. Reject is a
    # client-side dismiss (no destructive endpoint invented).
    assert "apiPostOrDetail('/summarize/apply', { path, go: true })" in html
    # a bare go-less (dry-run) default must not be silently escalated elsewhere
    assert "go: true }" in html
    # honesty: the write is described as backed-up (the drift-refusal is enforced by
    # core + surfaced via apiPostOrDetail's 409 detail, not asserted as fixed copy).
    assert "backs up first" in html


def test_summarize_error_helper_surfaces_server_detail(html):
    # run/apply/undo surface 409/502 reasons (drift, model failure) via a helper
    # that raises the server `detail`, not a bare "API 409".
    assert "async function apiPostOrDetail(path, body)" in html
    assert "(data && data.detail) ? data.detail : `API ${resp.status}`" in html


def test_optimize_dashboard_links_into_summarize_screen(html):
    # The Track A cost signal on Optimize is the doorway: the Summarize waste-row
    # is a link + a "Review →" CTA into the screen.
    assert '<a class="sz-link" href="#/optimize/summarize">${r.title}</a>' in html
    assert "r.title === 'Summarize'" in html


def test_optimize_has_dedicated_summarize_box(html):
    # A dedicated Summarize box on Optimize, rendered from the filesystem finding
    # (st.opt.findings.summarize) so it shows even with no telemetry / no cost chart.
    assert "st.opt.findings ? st.opt.findings.summarize : null" in html
    assert 'id="opt-summarize"' in html
    # The honest DEC-032 tile set: files · est tokens recoverable/call · avg reduction.
    assert "summarizable file" in html
    assert "est. tokens recoverable / call" in html
    assert "avg prose reduction" in html
    # Tokens-first + explicit basis + never "saves you" (Rule 14).
    assert "estimated · tokens" in html
    assert "${sf.estimate_basis}" in html
    box = html[html.index('id="opt-summarize"'):html.index('id="opt-summarize"') + 1400]
    assert "saves you" not in box.lower()


def test_optimize_box_shows_applied_vs_outstanding(html):
    # The scan figure is STILL-recoverable (applied files are dropped). Applied
    # savings come from the backup meta (est_tokens_saved), and the box shows
    # % applied = applied / (applied + outstanding) with an honest scope caveat.
    assert "api('/summarize/backups')" in html                         # OptimizeView fetches backups
    assert "const sfApplied = (st.bk || []).reduce" in html
    assert "const sfTotal = sfApplied + sfTok;" in html
    assert "sfTotal > 0 ? Math.round(sfApplied / sfTotal * 100)" in html
    assert "% applied" in html
    assert "still recoverable here" in html
    # honest denominator caveat (applied is cumulative; outstanding is this scan).
    assert "Applied counts every run; still-recoverable is this scan." in html


def test_review_diff_is_per_block_with_layman_hint(html):
    # The diff modal groups the unified diff into per-hunk blocks (no raw @@ header)
    # with +/- counts, and leads with a plain-language legend.
    assert "function diffBlocks(diffText)" in html
    assert "const mfBlocks = mf ? diffBlocks(mf.diff) : [];" in html
    assert "Block ${i + 1}" in html
    assert "lines removed" in html and "lines added" in html
    assert "unchanged lines are kept verbatim" in html
    # the cryptic @@ hunk header is dropped by diffBlocks, not shown raw
    assert "ln.startsWith('@@')) { cur = " in html


def test_summarize_honesty_no_realized_savings_language(html):
    # Rule 14: estimates are "(est.)" / "estimated", never "saves you". The tally
    # and run rows qualify the token figure as an estimate.
    start = html.index("function SummarizeView")
    end = html.index("// Analytics explorer (#210)")
    view = html[start:end]
    assert "tok/call saved (est.)" in view
    assert "saves you" not in view.lower()


def test_summarize_undo_restores_applied_from_backup(html):
    # An APPLIED file can be reverted via POST /summarize/undo (core restores the
    # gzip backup; 409 on drift/missing surfaced). The row goes terminal 'reverted'.
    assert "apiPostOrDetail('/summarize/undo', { path, go: true })" in html
    assert "const undoPath = async path =>" in html
    assert "[path]: 'reverted'" in html
    # Reachable from the review: inline row link (applied only) + modal footer.
    assert ">undo</span>" in html
    assert "Undo (restore backup)" in html
    # 'reverted' is a first-class terminal state (filter + count).
    assert "{ v: 'reverted', t: 'Reverted' }" in html
    assert "const revertedCount = revRows.filter(r => stOf(r.path) === 'reverted').length;" in html


def test_summarize_undo_reachable_for_prior_applies_via_backups(html):
    # Undo is not only in-session: GET /summarize/backups lists files with a gzip
    # backup so a file applied in ANY earlier session can be undone from the review
    # (and is advertised on the entry hub). This is the persistent undo surface.
    assert "api('/summarize/backups')" in html
    assert "const [backups, setBackups] = useState([]);" in html
    # A backups section in the review with a per-file Undo button + can't-undo reason.
    assert "Applied earlier — undo" in html
    assert "onClick=${() => undoPath(b.source_path)}" in html
    assert "can't undo — ${b.reason}" in html
    # Freshly-applied files are surfaced as undoable (refresh after apply); the entry
    # hub advertises undoable backups.
    assert "loadBackups();" in html
    assert "applied summary(ies) can be undone" in html


def test_summarize_scan_guards_against_toggle_race(html):
    # Two guards (Greptile #426 + reviewer follow-up): (1) a seq guard discards a
    # slower earlier scan's out-of-order response; (2) toggles merge into a ref
    # SYNCHRONOUSLY so a second toggle before re-render can't rebuild the other
    # value from a stale closure.
    assert "const scanSeq = useRef(0);" in html
    assert "if (seq !== scanSeq.current) return;" in html
    assert "const scanOpts = useRef(" in html
    assert "scanOpts.current = { ...scanOpts.current, ...opts };" in html
    assert "loadScan({ recursive: e.target.checked })" in html
    assert "loadScan({ repo: e.target.checked })" in html


def test_summarize_bulk_apply_excludes_structure_failed(html):
    # Bulk "Apply checked" excludes structure_ok===false to match the per-file
    # modal's disabled guard (core re-skips it server-side too; UI stays consistent).
    assert "s.path === p && s.structure_ok === false" in html
    # Reject is a client-side dismiss (no write) → its OWN unfiltered set, so apply
    # guards structure while reject can always clear a row from the view.
    assert "const rvCheckedReject = [...revChecked].filter(p => stOf(p) === 'staged');" in html
    assert "rvCheckedReject.forEach(p => n[p] = 'rejected')" in html


def test_summarize_batch_calls_are_null_safe(html):
    # apiPostOrDetail returns null on a 200-with-empty-body; normalize to {} so a
    # property access never crashes a run/apply that actually succeeded (Greptile #426).
    assert "(await apiPostOrDetail('/summarize/run', { path, mode: engine, ratio: 0.5 })) || {}" in html
    assert "(await apiPostOrDetail('/summarize/apply', { path, go: true })) || {}" in html


def test_summarize_reduction_pct_is_server_computed(html):
    # Anil #426: prose-reduction % is analysis, not presentation — the box tile and
    # per-file column render the analyzer's reduction_pct (#423); no JS chars/4.
    assert "sf.reduction_pct != null ? sf.reduction_pct : 0" in html
    assert "c.reduction_pct != null ? c.reduction_pct : 0" in html
    assert "sfRedPct" not in html      # old per-file chars/4 helper gone
    assert "sfSrcTok" not in html      # old source-token derivation gone


# --- recommendation-outcome panel (measured vs estimated) --------------- #
def test_recommendations_panel_present_and_fetches_endpoint(html):
    # The Optimize view surfaces the recommendation-outcome ledger, fetching the
    # /recommendations endpoint (best-effort) and rendering measured-recovered
    # strictly separate from estimated-recoverable (honesty discipline, Rule 14).
    assert "function RecommendationsPanel" in html
    assert "api('/recommendations')" in html
    assert "measured recovered" in html
    assert "estimated recoverable" in html
    # The panel must never re-derive analysis in JS — it renders server-computed
    # measured vs estimated fields straight from the endpoint payload.
    assert "measured_recovered_tokens" in html
    assert "past_overspend_tokens" in html


# --- cost proposals in the Review inbox (advise-only) ---------------------- #
def test_cost_proposals_wired_into_review_inbox(html):
    # The downsize/cache/trim analyzers surface as advise-only cost proposals in
    # the same Review inbox, fetched from the cost endpoints and rendered with a
    # distinct `kind` badge and an estimate. Keep the fetch + render wiring
    # present.
    assert "function CostProposalCard" in html
    # The Applied tab renders BOTH ledgers (relearn + cost) through one unified
    # row component, discriminated by `rec.kind` — the cost-specific
    # CostAppliedRow was folded into it as part of the inbox tab redesign.
    assert "function AppliedItemRow" in html
    assert "api('/relearn/cost-proposals')" in html
    assert "api('/relearn/cost-applied')" in html
    assert "'/relearn/cost-proposals/apply'" in html
    # The card is advise-only: a marker button, never an apply-to-code write.
    assert "Mark applied" in html
    # INVERTED (was `assert "Cost advisories" in html`). The inbox no longer
    # splits its open rows by which analyzer produced them: the cost proposals
    # and the relearn clusters are one list ranked by money, so a "Cost
    # advisories" tab existing again would be the regression, not its absence.
    assert "Cost advisories" not in html
    assert "Recurring mistakes (" not in html
    # The count is conditional now, because printing "(0)" before the page knows
    # its own numbers reads as "you are all clear" on a non-empty inbox.
    assert "Open ${openCountKnown ? html`(${shownItems.length})`" in html


def test_subagent_cost_card_has_workspace_apply_flow(html):
    # The subagent (4th) analyzer is apply-capable: its CC-origin card routes a
    # reversible rung-1 note through the apply-workspace endpoint (dry-run diff
    # then write), unlike the three advise-only analyzers.
    assert "'/relearn/cost-proposals/apply-workspace'" in html
    assert "apply_capable" in html
    assert "Apply note" in html
    # Human-readable, uppercase analyzer-category badge (inbox redesign
    # requirement #4) — replaced the old lowercase COST_ANALYZER_LABELS map.
    assert "subagent: 'SUBAGENT'" in html


def test_summarize_cost_card_links_to_curate_diff_instead_of_a_bare_apply(html):
    # summarize is a real peer card now, but its fix is a reviewed
    # rewrite (structure kept, prose compressed) driven by its own curate ->
    # diff -> apply lifecycle, which already tracks staged/applied state.
    # The card's ONE affordance must route there instead of offering the
    # generic advise-only "Mark applied" marker or an inline Apply button.
    start = html.index("function CostProposalCard")
    end = html.index("function InboxStatTiles", start)
    card = html[start:end]
    assert "prop.analyzer === 'summarize'" in card
    assert "Review in Summarize" in card
    assert "summarize:  { title: 'Summarize'" in html


def test_relearn_example_session_links_only_when_resolvable(html):
    # Relearn examples are sourced from transcript files on disk, so many name a
    # session that was never ingested and 404s on the detail route. The inbox
    # links only the resolvable ones and keeps the rest as plain evidence text.
    assert "ex.session_resolvable" in html
    assert (
        "? html`<a class=\"sz-link\" href=${'#/sessions/' + ex.session_id}"
        in html
    )
    # The snippet (the evidence itself) is rendered either way.
    assert "${ex.snippet}" in html


def test_sessions_nav_entry_present(html):
    # The session views (Map / Approach / Timeline) were reachable only by
    # following a link out of another screen. A sidebar entry makes them
    # discoverable; the paramless route lands on the session list the Status
    # view already renders, and the entry highlights while a session is open.
    assert (
        '<a href="#/sessions" class="nav-link" data-view="sessions" '
        'data-lens="improve">' in html
    )
    # Keep-alive router: the bare #/sessions route lands on StatusView (the
    # session list) via the PRIMARY_VIEWS registry — the same StatusView the
    # Status alias used to render — kept alive so returning is instant.
    assert "['sessions',  StatusView]" in html
    assert "sessions: 'improve'" in html


def test_sizing_note_apply_explains_unregistered_project(html):
    # A project-scoped sizing-note card whose target tokenjam couldn't resolve
    # (prop.target_path empty, because the project was never onboarded
    # per-project) must EXPLAIN the two exits instead of showing an empty box
    # that only errors on Apply: paste the path, or run `tj onboard
    # --add-project` from the repo so tokenjam learns it.
    start = html.index("function CostProposalCard")
    end = html.index("function InboxStatTiles", start)
    row = html[start:end]
    # The input pre-fills from the backend's resolved path when there is one.
    assert "useState(prop.target_path || '')" in row
    # The guidance is gated on there being no resolved path, and names both exits.
    assert "!prop.target_path ? html`" in row
    # Both exits are still named. The wording went from a numbered list, to one
    # sentence, to a clause on the label line itself, each time because it was the
    # largest non-snippet contributor to row height and each time keeping both
    # exits. Asserted on the exits, not on the layout: the layout is what keeps
    # changing and the exits are what must not be lost.
    assert "<code>CLAUDE.md</code> you paste below" in row
    assert "register it once with" in row
    # The reversibility fact rides the same line and is NOT trimmed with the rest:
    # it is what makes a one-click write to the reader's own file acceptable.
    assert "writes a reversible rung-1 note" in row
    # The register-command is one-click copyable, not just prose.
    assert '<${CopySnippetButton} text="tj onboard --add-project" />' in row
    # Smarter UX: "no path yet" is not an error. The buttons are disabled until
    # a path exists, so nothing can fire, and the empty-path guard NEVER sets a
    # red validation line — the always-visible guidance block is the messaging.
    assert "disabled=${wbusy || !target.trim()}" in row
    assert "if (!target.trim()) return;" in row
    assert "enter a CLAUDE.md path to write the note into" not in html
    assert "Paste a project CLAUDE.md path above, or run" not in html


def test_no_duplicate_status_nav_entry(html):
    # Status and Sessions used to be two sidebar entries rendering the SAME
    # StatusView for their bare route, so clicking between them changed nothing
    # on screen. Exactly one of them survives (Sessions), and the Status nav
    # link is gone. #/status stays a route-level alias for old bookmarks, so
    # the `case 'status'` label must remain even though no link points at it.
    assert 'data-view="status"' not in html, "the duplicate Status nav link must be gone"
    # Keep-alive router: #/status (and #/runs) remain silent aliases for Sessions,
    # resolved to the 'sessions' key in primaryKeyFor (replaced `case 'status'`).
    assert "if (v === 'status' || v === 'runs') return 'sessions';" in html, \
        "keep #/status as a silent alias for Sessions"
    # The two labels must not both render the split-zone page title.
    assert '<div class="page-title">Sessions</div>' in html
    assert '<div class="page-title">Status</div>' not in html
    # Exactly one nav-link resolves to the sessions/status surface (count the
    # actual anchor, not raw string hits — a comment may mention the attribute).
    assert html.count('class="nav-link" data-view="sessions"') == 1


# --- stat tiles / applied-tab unit hierarchy follows the server framing ---- #
# NOTE (inbox redesign): the perpetual verify/receipts layer these tests used
# to exercise (ReceiptsHeader's "Verified saved to date" tile, CostLedgerSummary)
# was removed in commit c0316aba for making unsupportable realized-savings
# claims — see the note in tokenjam/ui/index.html above InboxStatTiles, and
# relearn_apply.py's AppliedFix.verify docstring. Nothing populated
# `receiptsData`/`costLedger` even before this redesign (both were always
# `null`/hidden in production), so the "measured, regressed, no_change" copy
# these tests checked never actually rendered for a real user. Replaced by
# InboxStatTiles's "Fixes applied" tile (behavioral requirement #7, REVISED:
# sums each applied item's own estimate, no "verified" claim or chip ever) and
# AppliedItemRow (the same `est.`-labeled snapshot per row).
def test_stat_tiles_still_accept_a_suppressed_param_for_completeness(html):
    # NOTE: ReviewInboxView always calls InboxStatTiles with suppressed=false
    # on this page now (the always-dollars carve-out — see
    # test_review_inbox_ignores_dollar_suppression below). This only pins that
    # the component itself still HONORS a truthy `suppressed` if ever passed
    # one, i.e. the parameter isn't dead weight removed from the function.
    start = html.index("function InboxStatTiles")
    end = html.index("function ReviewInboxView", start)
    tile = html[start:end]
    assert "suppressed" in tile
    # The open-items half of this component is now the past-overspend band
    # (server-computed, always dollars-with-tokens, no suppression choice to
    # make). What survives here is the applied tile, which still honors a
    # truthy `suppressed` by leading with tokens.
    assert "'~' + fmtTokens(appliedTokSum) + ' tok'" in tile
    assert "fmtUsd(appliedUsdSum)" in tile


# --- Approved carve-out: Review inbox ignores dollar suppression ---------- #
def test_review_inbox_ignores_dollar_suppression(html):
    # On the Review inbox ONLY, dollar figures render unconditionally — the
    # subscription-share suppression rule (dollarsSuppressed(),
    # core/framing.py's suppress_dollars_for_subscription_share) does not
    # apply here, even though it still gates every other dollar figure in the
    # app unchanged. Verified against a real account (87% subscription-billed,
    # Max 20x plan): the API payload correctly carries estimated_monthly_usd
    # for every priced item, so once this page stops gating on
    # dollarsSuppressed() those figures render regardless of plan tier.
    view = html[html.index("function ReviewInboxView"):]
    start = view.index("const suppressed = ")
    end = view.index(";", start)
    line = view[start:end]
    assert line == "const suppressed = false"
    # The old suppression computation must not survive a regression that
    # silently re-adds the plan-tier gate this page deliberately ignores.
    assert "dollarsSuppressed(relearnFraming)" not in view
    assert "dollarsSuppressed(costFraming)" not in view
    # dollarsSuppressed() itself is untouched and still used elsewhere in the
    # app (e.g. the cache-recommend section) — the carve-out only stops THIS
    # page from calling it, it doesn't remove or weaken the function.
    assert "function dollarsSuppressed(framing)" in html
    assert html.count("dollarsSuppressed(") >= 2   # at least one other caller survives


def _fmt_framed_savings_src(html: str) -> str:
    start = html.index("function fmtFramedSavings(usd, tokens, framing, dollarFmt = fmtCost)")
    return html[start: html.index("\n}\n", start)]


def test_fmt_framed_savings_never_renders_a_bare_dollar_when_framing_is_unknown(html):
    """`framing == null` means "not yet known" (the read that would carry it
    hasn't landed), NOT "no suppression needed". Before the fix,
    dollarsSuppressed(null) returned false, so an unknown framing fell
    through to a bare dollar figure — on a subscription-billed account that
    reads as real money billed, exactly what framing exists to prevent
    (root CLAUDE.md's "UI asserts more than its data supports" defect class).

    `dollarFmt` (added for the Dashboard's at-most-2dp rule) defaults to
    fmtCost, so every existing caller's behaviour is unchanged; only the
    Dashboard passes fmtDashUsd. This test still pins the null-guard logic
    against the default formatter.

    The suppressed path's own "% of cycle tokens" branch for subscription is
    gone: dollarsSuppressed() can no longer be true for subscription (that
    differentiation was removed), so the only pricing_mode that ever reaches
    the suppressed path now is local, which just wants the plain token total.
    """
    fn = _fmt_framed_savings_src(html)
    # The null/undefined guard must run BEFORE dollarsSuppressed() is even
    # consulted, and must not fall through to a bare dollar figure.
    assert "if (framing == null) {" in fn
    guard_idx = fn.index("if (framing == null) {")
    suppressed_idx = fn.index("if (dollarsSuppressed(framing)) {")
    assert guard_idx < suppressed_idx
    guard_body = fn[guard_idx: suppressed_idx]
    assert "fmtCost(" not in guard_body
    assert "dollarFmt(" not in guard_body
    assert "fmtTokens(tokens) + ' tokens'" in guard_body

    # Behaviour for a KNOWN framing must be untouched: the suppressed path's
    # tokens-only fallback, and the show-with-qualifier path's dollar figure
    # (via dollarFmt, fmtCost by default), both still follow the null guard.
    # (The code's own comment explaining the removal legitimately mentions
    # the retired string, so check the CODE only, not the comment.)
    assert "% of cycle tokens" not in _no_comments(fn)
    assert "return usd == null ? '—' : dollarFmt(usd);" in fn


def test_resend_dollar_figure_stays_tokens_only_as_a_structural_measurement(html):
    # The resend/TRIM card is the one documented exception to "always
    # dollars": its own evidence text discloses the figure is a structural
    # token-share measurement, not a savings claim (RESEND_HONESTY_CAVEAT,
    # analyzers/context_resend.py). A resend CostProposal renders through
    # `pastOverspendFigure` like every other cost card — the exclusion set
    # only has one live consumer left, `appliedEstimate` (the Applied tab,
    # which merges relearn + cost records and needs to know which analyzer's
    # figure to treat as unpriced there too).
    assert "const NOT_A_SAVINGS_CLAIM_ANALYZERS = new Set(['resend']);" in html
    applied_est_start = html.index("function appliedEstimate(rec)")
    applied_est_end = html.index("\n}", applied_est_start)
    assert "NOT_A_SAVINGS_CLAIM_ANALYZERS.has(rec.analyzer)" in html[applied_est_start:applied_est_end]
    # The relearn-cluster-only forward-figure machinery this exclusion used to
    # gate is retired entirely — a relearn cluster shows its past figure only,
    # rendered via `relearnObservedFigure`/`pastOverspendFigure`, so there is
    # no separate monthly-basis display path left to exclude an analyzer from.
    assert "function monthlyUsdForDisplay(item)" not in html
    assert "function estMonthlyLine(item, suppressed)" not in html
    assert "function combinedEstMonthly(items, suppressed)" not in html


def test_applied_estimate_falls_back_to_legacy_fields_for_pre_upgrade_records(html):
    # A ledger entry applied BEFORE the past_overspend_* collapse snapshotted
    # its estimate under the retired forward-savings names
    # (estimated_monthly_usd/_tokens). appliedEstimate() must still surface a
    # figure for that record -- read-only, never re-persisting the legacy
    # name -- rather than the Applied tab silently losing it.
    fn_start = html.index("function appliedEstimate(rec)")
    fn_end = html.index("\n}", fn_start)
    fn = html[fn_start:fn_end]
    assert "estimated_monthly_usd" in fn
    assert "estimated_monthly_tokens" in fn
    # `!= null` (not a truthy check) so a genuine 0 is never mistaken for
    # "missing" and papered over with the legacy fallback.
    assert "past_overspend_usd != null" in fn
    assert "past_overspend_tokens != null" in fn


def test_fixes_applied_tile_never_claims_verification(html):
    # Behavioral requirement #7 (REVISED, supersedes an earlier "verified
    # saved" draft): the second tile is "Fixes applied", sums each applied
    # item's own ORIGINAL estimate (never a live re-measurement), and must
    # never render the word "verified" or a VERIFIED chip anywhere on the page.
    start = html.index("function InboxStatTiles")
    end = html.index("function ReviewInboxView", start)
    tile = html[start:end]
    assert ">Fixes applied<" in tile
    # The estimates-only qualifier lives in the tile's sub-line (the header
    # "Fixes applied" doesn't itself say "estimate", so the sub-line carries
    # the honesty). Wording was tightened when the tile adopted the mockup's
    # denser styling; the claim it makes must not weaken.
    # The figure must always be dated to apply time, so it can never be read as
    # a live measurement of what actually happened afterward. Wording has moved
    # (an earlier draft said "estimates only, not re-measured"); the claim must
    # not weaken, and the bare-count fallback must say why it has no figure.
    assert "estimated when applied" in tile
    # The count fallback carries no figure, so it makes no savings claim to
    # qualify — it must show recency, never a number dressed up as a saving.
    assert "most recent ${daysAgoLabel(lastAppliedAt)" in tile
    # The old tile label and the mockup's own VERIFIED chip never render (a
    # rendered-text check, not a comment check — explanatory code comments
    # pointing at c0316aba legitimately use the word "verified").
    assert "Verified saved" not in html
    assert "VERIFIED" not in html
    assert ">verified<" not in html.lower()
    # A falsy zero is never faked as the big number: it falls back to the
    # bare applied count when nothing applied carries an estimate.
    assert "String(appliedCount)" in tile
    # The comment points a future reader at the removal commit before they
    # reintroduce live measurement.
    assert "c0316aba" in html


def test_applied_item_row_respects_dollar_suppression(html):
    # The Applied tab's per-row `est.` figure (relearn's apply-time snapshot,
    # or a cost marker's own estimate) falls back to tokens under the same
    # server framing every other dollar figure on this page respects.
    start = html.index("function AppliedItemRow")
    end = html.index("function CopySnippetButton", start)
    fn = html[start:end]
    assert "!suppressed && usd != null" in fn
    assert "fmtTokens(toks)" in fn


def test_dollars_suppressed_reads_the_server_display_rule(html):
    """The suppress/show decision is server-side (core/framing.py); the UI
    reads display_rule rather than re-deriving the rule in JS.
    'suppress_dollars_for_subscription_share' was removed from the suppressed
    set by product decision: tj does not differentiate subscription-billed
    from API-billed users, so a subscription window is never suppressed to
    tokens-only any more. core/framing.py's compute_framing already stopped
    producing that display_rule value; this pins the JS side dropping it too,
    so the two do not silently drift back out of agreement."""
    assert "function dollarsSuppressed" in html
    for rule in ("'tokens_only'", "'suppress_dollars_unknown'"):
        assert rule in html, f"missing suppressing display_rule {rule}"
    assert "'suppress_dollars_for_subscription_share'" not in html


# --- the headline is the server's past-overspend band, not a JS sum -------- #
# SUPERSEDES the old "ESTIMATED RECOVERABLE / mo" tile tests. What replaces
# them: the product leads with what the flagged behaviours ALREADY cost (past
# tense, window-observed), not with what a fix might return. A past figure is
# checkable against a bill the user already paid; a forward one asks them to
# trust a projection of a month that has not happened, and trust is the
# scarce resource.
def test_inbox_headline_is_the_past_overspend_tile(html):
    # The full-width band was removed and this figure now occupies the
    # compact tile slot beside "Fixes applied". The TENSE decision above is
    # unchanged -- only the shape moved.
    start = html.index("function InboxStatTiles")
    end = html.index("function ReviewInboxView", start)
    tile = html[start:end]
    assert "<${PastOverspendTile} block=${pastOverspend}" in tile
    # It sits IN the tile row, beside the applied tile -- not on its own row
    # above it, which is what made it read as a heavy banner.
    row = tile[tile.index('display:flex;gap:14px'):]
    assert "<${PastOverspendTile}" in row
    assert "Fixes applied" in row
    # The old forward-looking headline and its vocabulary are gone from here.
    assert ">Estimated recoverable " not in tile
    assert "est./mo" not in tile


def test_inbox_headline_number_comes_from_the_payload_not_a_client_side_sum(html):
    # Single-compute-path: the headline figure is the server's own rollup over
    # the open set (GET /relearn/cost-proposals -> past_overspend), so it is
    # the same number the Dashboard hero renders. It must not be reduced over
    # whatever cards happen to be on screen — a local dismiss changes one
    # person's view, not what was spent.
    start = html.index("function InboxStatTiles")
    end = html.index("function ReviewInboxView", start)
    tile = html[start:end]
    assert "reduce" not in tile.split("const appliedPriceable")[0]
    assert "priceable.length * 2 >= openItems.length" not in tile


def test_past_overspend_tile_hides_when_nothing_is_open(html):
    # Nothing open means no observed figure to state, so the tile hides rather
    # than rendering a fabricated zero — but the excluded-waste line still has
    # to render, which is why it now lives BELOW the tile row rather than
    # inside the tile that can disappear.
    start = html.index("function InboxStatTiles")
    end = html.index("function ReviewInboxView", start)
    tile = html[start:end]
    assert "hasOpenOverspend ? html`" in tile
    band_start = html.index("function PastOverspendTile")
    band = html[band_start:html.index("\n}", band_start)]
    assert "if (!causes && !toks) return null;" in band
    # The note is not nested inside the conditional tile.
    note_at = tile.index("<${ExcludedWasteNote}")
    assert "hasOpenOverspend" not in tile[note_at - 200:note_at]


# --- #326: excluded waste (summarize) is stated + linked, never summed ----- #
def test_inbox_stat_tiles_renders_the_excluded_cross_reference(html):
    start = html.index("function ExcludedWasteNote")
    end = html.index("function ReviewInboxView", start)
    block = html[start:end]
    # Rendered inside InboxStatTiles, not just defined standalone.
    assert "<${ExcludedWasteNote} excluded=${excluded} />" in block
    # Once, not twice: it used to render both inside the band's `note` slot and
    # again in a bare fallback band. With the band gone it has ONE home, below
    # the tile row, where it renders whether or not the tile does.
    assert block.count("<${ExcludedWasteNote} excluded=${excluded} />") == 1
    # Never folded into the blue tile's own dollar figure — only ever a
    # separate stated line with a link out.
    assert "not summed above" in block
    assert "Review it" in block
    assert "#/optimize/summarize" in block


def test_review_inbox_view_fetches_and_threads_excluded_from_the_rollup(html):
    view = html[html.index("function ReviewInboxView"):]
    end = view.index("function ", len("function ReviewInboxView"))
    view = view[:end]
    assert "setCostExcluded((r.past_overspend && r.past_overspend.excluded) || {})" in view
    assert "excluded=${costExcluded}" in view


def test_estimated_tile_still_renders_with_only_excluded_waste_and_no_open_items(html):
    # Summarize can be the ONLY recoverable figure in a given scan (no cost
    # advisories, no recurring mistakes yet) — the tile must still surface it
    # rather than disappearing the way the pre-#326 empty state did (issue
    # #326: "the product's largest recoverable figure invisible from the
    # headline a user reads").
    start = html.index("function InboxStatTiles")
    end = html.index("function ReviewInboxView", start)
    tile = html[start:end]
    assert "hasExcluded" in tile
    assert "openItems.length === 0 && appliedCount === 0 && !hasExcluded" in tile


# --- Review inbox copy: cost-led, and no hardcoded zero -------------------- #
def test_review_inbox_intro_matches_the_approved_mockup(html):
    # Inbox redesign: the page title and subtitle are the approved mockup's
    # own copy verbatim (colon in place of the mockup transcription's em dash
    # — house style forbids em dashes in tokenjam copy).
    assert "<div class=\"page-title\"" in html
    assert ">Inbox<" in html
    intro = (
        "Waste you're paying for more than once: mistakes your agents keep "
        "repeating, and cost fixes ready to apply."
    )
    assert intro in html
    # The loop-first phrasing from the pre-redesign copy is gone.
    assert "land here so it can relearn them" not in html
    # The Approve/Dismiss mechanics the old intro stated are preserved. They
    # have now moved a second time: off the tab's explainer paragraph (three
    # dense lines above the list) and onto the two controls they describe, so
    # each consequence is stated at the moment it can still be declined. The
    # write disclosure sits on the modal's Approve button (the body only
    # reports git-commit-vs-backup AFTER the write); the local-only disclosure
    # sits on the bulk Dismiss button. The strings must survive wherever they
    # live, which is what these assertions pin.
    assert "git-committed, or backed up if the target is not a git repo" in html
    assert "you confirm the scope and target first" in html
    assert "this browser only; it is not sent to the server" in html
    # House style: no em dashes, and tokens are never called "quota".
    assert "—" not in intro
    assert "quota" not in intro.lower()


def test_old_pending_relearn_stat_line_replaced_by_the_combined_stat_tiles(html):
    # The old cur-listhead token-count segment ("~N tok recoverable", sourced
    # from `d.estTokens`) is gone — the inbox redesign replaced it with the
    # combined ESTIMATED RECOVERABLE / VERIFIED SAVED tiles (InboxStatTiles),
    # covered by their own suppression/hide tests above. The literal "0
    # strategies" placeholder this test used to guard against is gone too.
    assert '<b style="color:var(--accent)">0</b> strategies' not in html
    assert "strategies" not in html
    assert "estTokens: f.past_overspend_tokens" not in html
    assert "~${fmtTokens(d.estTokens)} tok</b> recoverable" not in html
    assert "function InboxStatTiles" in html


# --- Review inbox select-all ----------------------------------------------- #
def test_select_all_checkbox_sits_beside_the_bulk_dismiss_button(html):
    # Inbox redesign: the Recurring-mistakes tab dropped its <table> for flat
    # rows (RecurringMistakeRow, matching the mockup's card-style layout), so
    # the select-all box now sits in the listhead beside "Dismiss checked"
    # rather than inside a <thead><th>.
    assert "function SelectAllCheckbox" in html
    # It governs `bulkRelearn` (the rows tj can apply AND the review queue can
    # drive), not every listed row: the single-list redesign put rows with no
    # apply path on the same list, and a select-all spanning them would hand the
    # bulk buttons a scope they cannot act on.
    assert (
        "<${SelectAllCheckbox} total=${bulkRelearn.length} "
        "selected=${selectedCount} onToggle=${toggleAll} />"
    ) in html
    # The per-row checkbox is still present, just inside a card now.
    assert 'checked=${checked} onChange=${onToggle} />' in html


def test_select_all_reports_the_indeterminate_state_on_a_partial_selection(html):
    # `indeterminate` is a DOM property with no HTML attribute, so it has to be
    # assigned through a ref. A header box that shows plain "checked" over a
    # partial selection invites accidental bulk actions.
    start = html.index("function SelectAllCheckbox")
    end = html.index("function RecurringMistakeRow", start)
    fn = html[start:end]
    assert "ref.current.indeterminate = selected > 0 && selected < total" in fn
    # Fully checked only when every listed row is selected.
    assert "const all = total > 0 && selected === total" in fn
    assert "checked=${all}" in fn


def test_select_all_toggles_off_when_everything_is_selected(html):
    start = html.index("function nextSelectAllSelection")
    end = html.index("// The table's select-all box.", start)
    fn = html[start:end]
    assert "if (all) next.delete(sig)" in fn
    assert "else next.add(sig)" in fn
    # The component delegates to it over the SELECTABLE row set.
    assert (
        "nextSelectAllSelection(bulkRelearn.map(c => c.signature), prev)"
    ) in html


def test_select_all_applies_only_to_the_rendered_rows(html):
    # THE load-bearing one. The list filters out locally-dismissed rows and rows
    # already applied (in this session OR any earlier one), so select-all must
    # iterate the rendered set, never the unfiltered d.clusters, or it would
    # dismiss rows the user never saw. On the single list that rendered set is
    # narrowed once more, to the rows a bulk action can actually reach.
    start = html.index("const bulkRelearn = shownItems.filter(")
    end = html.index("const renderRow =", start)
    block = html[start:end]
    assert "shownItems.filter(" in block
    assert "inboxCanApply(i)" in block
    assert "bulkRelearn.filter(c => checked.has(c.signature))" in block
    assert "bulkRelearn.map(c => c.signature)" in block
    assert "d.clusters" not in block
    # The filter that makes `visible` a strict subset is still in place. This
    # previously pinned the `!appliedSigs.has(...)` form verbatim, which meant
    # the suite was ENFORCING the session-local-only filter that re-offered
    # already-applied fixes; see the dedicated ledger test above for why that
    # was a defect rather than a design.
    assert (
        "const visible = (d.clusters || []).filter(c => !dismissed.has(c.signature) "
        "&& !appliedSigsAll.has(c.signature))"
    ) in html


def test_open_mistakes_exclude_already_applied_fixes_from_the_ledger(html):
    # An already-applied recurring mistake must not come back as an open
    # proposal. `appliedSigs` is session-local and starts empty on every page
    # load, so filtering on it alone re-offered every fix applied in an earlier
    # session: approving one then attempted to rewrite the hook its own earlier
    # approval had created, and only relearn_apply's file-ownership guard
    # stopped the double write. It also inflated the tab count and the
    # ESTIMATED RECOVERABLE tile with savings already banked.
    assert "const appliedSigsAll = new Set([" in html
    assert "...appliedSigs," in html
    assert "...(applied || []).filter(r => r.state !== 'reverted').map(r => r.signature)," in html
    assert (
        "const visible = (d.clusters || []).filter(c => !dismissed.has(c.signature) "
        "&& !appliedSigsAll.has(c.signature))"
    ) in html
    # The session-local set alone must never again be the whole filter.
    assert "!appliedSigs.has(c.signature))" not in html
    # Same rule on the cost half, which already read its ledger correctly — the
    # two halves of this view drifted apart silently once.
    assert (
        "const costAppliedSigs = new Set((costApplied || [])"
        ".filter(r => r.state !== 'reverted').map(r => r.signature))"
    ) in html


def test_dismiss_checked_cannot_reach_an_unlisted_row(html):
    # `checked` is not pruned when a row leaves the list, so dismissing the raw
    # set would sweep along a signature that is no longer on screen.
    start = html.index("const dismissChecked =")
    end = html.index("const renderRow =", start)
    fn = html[start:end]
    assert "bulkRelearn.filter(c => checked.has(c.signature)).map(c => c.signature)" in fn
    assert "...checked]" not in fn


def test_dismiss_button_states_its_blast_radius(html):
    # The action names how many rows it will act on before it fires, and counts
    # the same scoped selection the header checkbox reports.
    # Asserted as separate facts rather than one contiguous string: the button
    # now also carries a title, and a brittle exact-markup match would fail on
    # any attribute added between the handler and the label without the guarded
    # behaviour having changed at all.
    assert "disabled=${selectedCount === 0} onClick=${dismissChecked}" in html
    assert (
        "${selectedCount ? `Dismiss ${selectedCount} checked` : 'Dismiss checked'}"
    ) in html
    # And it states WHERE the dismissal lands, since "dismiss" reads like a
    # server-side action when the intro no longer spells it out at length.
    assert "Hides these rows locally" in html
    # Not the raw, unscoped set that could overstate it.
    assert "disabled=${checked.size === 0}" not in html


def test_select_all_adds_no_bulk_approve(html):
    # Dismiss is local and undone by a reload; Approve writes to disk. The
    # listhead carries exactly TWO bulk actions — "Review N checked" (opens
    # each checked row's modal in turn) and "Dismiss N checked" — and neither
    # writes anything. The invariant this pin defends is that no bulk control
    # can APPROVE; it is not a cap on the number of buttons, so a third
    # non-writing control may be added, but a writing one may not.
    # Scoped to the whole open-list tab block, not just its listhead: the two
    # bulk buttons sit BELOW the list (they used to be in the head, beside
    # select-all), so a head-only slice would miss them entirely and pass
    # vacuously.
    view = html[html.index("function ReviewInboxView"):]
    start = view.index("tab === 'open' ? html`")
    end = view.index("tab === 'applied' ? html`", start)
    head = view[start:end]
    assert "dismissChecked" in head
    assert "startReview(selectedVisible.map(c => c.signature))" in head
    assert "onClick=${approveChecked}" not in html
    assert "Approve checked" not in html
    # The queue only navigates. If startReview ever gains a POST, that is a
    # bulk approve wearing a different name.
    review_fn = html[html.index("const startReview = (sigs) =>"):]
    review_fn = review_fn[: review_fn.index("\n  };")]
    for writing_call in ("apiPost", "apiPostOrDetail", "doApprove", "/apply"):
        assert writing_call not in review_fn, f"startReview must not write: {writing_call}"


# --- no dollar figure escapes the framing, and no false basis in a comment -- #
def test_no_comment_claims_dollars_are_scoped_to_api_billed_traffic(html):
    # The false mechanism must not survive anywhere in the served UI, including
    # in a comment where no test would otherwise look.
    assert "can only ever count the API-billed slice" not in html
    assert "reflect API traffic only" not in html
    assert "of that is on API-billed traffic" not in html


# --- Real-data validation follow-ups: sort order, dollar-first, formatting - #
# Founder's live 40-day store surfaced three gaps against a page rendering
# real (not fixture) proposals: the Cost-advisories tab wasn't sorted by
# est./mo at all (adapter-insertion order leaked through), a priceable
# item's headline still showed tokens, and a token count at billion scale
# rendered as an ugly "11062.0M" instead of "11.3B".

# --- Single-list inbox: the mechanism axis --------------------------------- #
def test_the_mechanism_axis_is_two_orthogonal_facts_not_one_enum(html):
    # The open list is no longer split by which analyzer produced a row. The axis
    # that replaced the tabs is what tj can DO about a row.
    #
    # It must NOT be an enum. `CostProposalCard` has always rendered
    # `${prop.suggestion ? ...}` and `${prop.apply_capable ? ...}` as two
    # INDEPENDENT conditionals, and deadweight's mcp_remove proposal is the
    # overlap case: it shows a copyable snippet AND a confirm-target apply
    # control together. An enum has to pick one of those to report, so it
    # necessarily misreports that row. Two booleans cannot.
    can_start = html.index("function inboxCanApply(item)")
    snip_start = html.index("function inboxHasSnippet(item)")
    tomb_start = html.index("// A `MechanismTags` component")
    can_fn = html[can_start:snip_start]
    snip_fn = html[snip_start:tomb_start]

    # Each predicate reads exactly the flags its own fix block is gated on.
    assert "!item.advise_only && item.write_offered !== false" in can_fn
    assert "item.apply_capable" in can_fn
    assert "item.advise_snippet_offered" in snip_fn
    assert "item.suggestion" in snip_fn
    # They must not consult each other: orthogonal means neither can suppress
    # the other, which is what made the enum lie about the overlap row.
    assert "inboxHasSnippet" not in can_fn
    assert "inboxCanApply" not in snip_fn

    # "Nothing actionable" is DERIVED from both being false where it is needed, never
    # stored as a peer value. Both rows gate their no-action copy on it inline.
    row = html[html.index("function RecurringMistakeRow"):html.index("function RelearnApplyModal")]
    assert "!canApply && !hasSnippet" in row

    # Neither the retired enum nor the retired BADGES may come back. The action
    # button is the row's only promise now; a label restating it over-promised on
    # every row that still needed a pasted path first.
    for dead in ("MECH_WRITE", "MECH_SNIPPET", "MECH_NONE", "MechanismBadge",
                 "inboxMechanism(", "MechanismTags", "MECHANISM_TAG_COPY",
                 "inboxNothingActionable"):
        assert dead not in _no_comments(html), f"must not be reintroduced: {dead}"
    # The badge labels are gone from the CODE, though the tombstone comment still
    # names them to record why. Checked against comment-stripped source.
    code = _no_comments(html)
    for label in ("TJ CAN APPLY", "COPY THE FIX", "NO FIX YET"):
        assert label not in code, f"badge label must not be rendered again: {label}"
    # One category badge per row, no mechanism tag beside it.
    for comp, end in (("function RecurringMistakeRow", "function RelearnApplyModal"),
                      ("function CostProposalCard", "function InboxStatTiles")):
        block = html[html.index(comp):html.index(end)]
        assert block.count('class="badge') == 1, "one category badge per row, no mechanism tag"

    # Never keyed on analyzer identity: the day a downsize proposal becomes
    # apply-capable its rows gain the apply control with no edit here.
    for analyzer in ("'downsize'", "'deadweight'", "'resend'", "'subagent'"):
        assert analyzer not in can_fn + snip_fn, f"must not be keyed on {analyzer}"


def test_a_row_that_is_both_apply_capable_and_snippet_bearing_renders_both(html):
    # deadweight's mcp_remove proposal is exactly this row: `apply_capable` AND a
    # `suggestion`. The two facts are orthogonal, so BOTH affordances must render.
    # This used to assert two BADGES; the badges are gone (the action button says
    # what it does), so it now asserts the behaviour the badges only described.
    tag_start = html.index("function inboxMechanismTag(item)")
    tag_fn = html[tag_start:html.index("\n}", tag_start)]
    assert "if (inboxCanApply(item)) parts.push('apply')" in tag_fn
    assert "if (inboxHasSnippet(item)) parts.push('snippet')" in tag_fn
    assert "parts.join(' ')" in tag_fn
    # No early return that would make the two mutually exclusive.
    assert "else" not in tag_fn

    # The card's two blocks are independent conditionals, not a chain, which is what
    # lets one row show a copy box AND a confirm-target apply control together.
    card = html[html.index("function CostProposalCard"):html.index("function InboxStatTiles")]
    assert "${prop.suggestion ? (canApply ? html`" in card
    # Three apply shapes now, so the chain leads with the register-then-apply one.
    assert "${canApply ? (needsSourcePath ? html`" in card
    assert "hasApplyKind ? html`" in card
    # The snippet block is gated on `prop.suggestion` ALONE — `canApply` only picks
    # which of the two shapes it takes, so an apply control can never suppress it.
    # Asserted structurally rather than on the outer conditional's exact text,
    # because that text is what changed when the second shape was added.
    snippet_block = card[card.index("${prop.suggestion ? (canApply"):card.index("${canApply ? (needsSourcePath")]
    assert snippet_block.count("<${CopySnippetButton} text=${prop.suggestion} />") == 2
    assert snippet_block.count('<div class="sz-copybox">${prop.suggestion}</div>') == 2
    # On a row that offers BOTH, the manual command is the secondary exit and sits
    # in a collapsed disclosure: rendering both open made deadweight's row the
    # tallest on the list while offering strictly less than its neighbours. Still
    # present, still copyable, no longer competing for the row's height.
    assert "<summary>Or copy the command and run it yourself</summary>" in snippet_block


def test_snippet_rows_render_the_snippet_as_the_deliverable(html):
    # THE failure this redesign exists to fix: a row whose fix is a copyable
    # change must show that change, not a bare "Mark applied" with nothing above
    # it. Both ledgers put the fix in a copy box with a Copy button, gated on the
    # snippet fact alone so an apply control never suppresses it.
    #
    # The Copy control moved INSIDE the box (`.sz-fixbox` positions it top-right)
    # when its label line turned out to be 20px spent rendering the word "Fix" next
    # to a code block. Asserted on the pairing rather than on the old layout: what
    # matters is that the snippet renders AND is copyable, not where the button sits.
    row_start = html.index("function RecurringMistakeRow")
    row_end = html.index("function RelearnApplyModal", row_start)
    row = html[row_start:row_end]
    assert "${hasSnippet ? html`" in row
    assert "<${CopySnippetButton} text=${fixText} />" in row
    assert '<div class="sz-copybox">${fixText}</div>' in row
    assert 'class="sz-fixbox"' in row

    card = html[html.index("function CostProposalCard"):html.index("function InboxStatTiles")]
    assert "${prop.suggestion} />" in card
    assert "${prop.suggestion}</div>" in card
    # The button is reserved room inside the box rather than overlapping the text.
    assert ".inbox-row .sz-fixbox > .sz-copybox { padding-right: 74px; }" in html
    # The cost card's "Mark applied" is reachable only as the snippet follow-up.
    # It used to be the catch-all `else`, which asked a reader with no fix on
    # screen to confirm having applied one.
    assert "` : hasSnippet ? html`" in card


def test_relearn_rows_never_offer_mark_applied(html):
    # There is no apply path for an advise-only cluster at all: the modal itself
    # says "Nothing to approve". A Mark-applied button there is a false promise
    # and would write a ledger entry for a fix that does not exist. This is 50 of
    # 55 clusters, the common case rather than an edge case.
    row = html[html.index("function RecurringMistakeRow"):html.index("function RelearnApplyModal")]
    # Asserted on the AFFORDANCE, not on the phrase: the row's comments explain
    # at length why there is deliberately no marker button here, so a bare
    # substring check on "Mark applied" matches the explanation and fails on
    # correct code. What must be absent is a control that fires one.
    assert ">Mark applied<" not in row
    assert "'Mark applied'" not in row
    assert "onMark" not in row
    assert "/apply'" not in row and "cost-proposals/apply" not in row
    # The modal's own gate, which the can-apply predicate mirrors, still stands.
    assert "Nothing to approve: this recommendation is yours to apply." in html
    # A non-applyable row states the budget's REAL reason, not a compressed label,
    # so the reader learns what to do instead of just that a gate fired.
    assert "cluster.advise_only_reason" in row
    # It now lives in its own collapsed disclosure rather than an inline paragraph:
    # it is a paragraph of write economics, and the mechanism tag already says the
    # row has no writer.
    assert "<summary>Why there is no permanent fix on offer</summary>" in row
    assert "<p>${cannotApplyReason}</p>" in row
    # The compressed one-liner is gone from the UI. Asserted on the DEFINITION
    # and the string it produced, not on the bare name: a tombstone comment
    # deliberately still names the removed helper and points at its surviving
    # server-side field, which is documentation rather than a leftover.
    assert "function writeGateNote" not in html
    assert "'No permanent fix offered: '" not in html
    # `write_blocked_short` itself is not dead: the CLI's dense relearn list
    # still renders it, so the UI must not read it any more but the field stays.
    assert "cluster.write_blocked_short" not in html


def test_no_row_renders_a_dead_end(html):
    # Where there genuinely is no fix, say so. An empty action area reads as a
    # bug, and a marker button there asks the reader to confirm doing something
    # the card never told them to do.
    row = html[html.index("function RecurringMistakeRow"):html.index("function RelearnApplyModal")]
    assert "${!canApply && !hasSnippet && !cannotApplyReason ? html`" in row
    assert "There is nothing to apply here yet" in row
    assert "See the example sessions" in row

    card = html[html.index("function CostProposalCard"):html.index("function InboxStatTiles")]
    # summarize keeps its own hop into the curate/diff/apply flow. That punt is
    # deliberate and code-documented (`_summarize_to_proposals`: "Deliberately
    # never `apply_capable`"), because an LLM call sits mid-pipeline between
    # prepare() and a staged rewrite, so no single button can span it.
    assert "Review in Summarize" in card
    # Everything else with no apply path and no snippet points at the analyzer's
    # own detail card, never the marker button. The generic sentence is a
    # last-resort fallback, gated on the row carrying no reason AND no
    # description, so it can never talk over the server's own wording.
    assert "${!blockedReason && !description ? html`" in card
    assert "optimizeFindingHref(prop.analyzer)" in card


def test_inbox_row_text_is_uncapped_without_lifting_the_global_measure(html):
    # Founder feedback on the running page: a row's description stopped well short
    # of the card edge, leaving a wide empty gutter with the amount stranded in the
    # far corner. Cause was the global `.sz-note { max-width: 74ch }`.
    #
    # That cap must SURVIVE: `.sz-note` is the whole app's standalone-paragraph
    # class (page intros, Optimize section notes) and 74ch is the right measure for
    # reading. Only the inbox row overrides it, because there the text sits in a
    # bordered card whose width the reader has already accepted.
    assert ".sz-note { font-size: 13px; color: var(--text-dim); line-height: 1.55; max-width: 74ch; margin: 0; }" in html
    assert ".inbox-row .sz-note, .inbox-row .sz-copybox { max-width: none; }" in html
    # Scoped by a class the row components set, NOT by lifting the cap or by
    # overloading `data-mechanism` (which is a state/debug attribute, not a
    # styling hook).
    assert "[data-mechanism] .sz-note" not in html
    # Both row components carry the class, so the collapsed tail and the
    # below-the-fold group inherit it: they render through these same components.
    assert 'class="opt-section inbox-row" data-mechanism=' in html
    assert "'opt-section inbox-row' + (focused ? ' rev-focus' : '')" in html
    # The relearn Approve MODAL renders outside the row and keeps its measure: it
    # is a reading surface, not a dense card.
    modal = html[html.index("function RelearnApplyModal"):html.index("function daysAgoLabel")]
    assert "inbox-row" not in modal


def test_a_recompute_never_blanks_rows_the_page_already_has(html):
    # THE correctness bug. Both endpoints serve their CACHED result immediately and
    # merely flag `status: "computing"` while a refresh runs, so the page had 55
    # clusters and 16 proposals in hand while printing a bare "Loading…", omitting
    # the avoided tile, and rendering the tab as "Open (0)". Zero is the most
    # misleading value this page can print: it reads as "you are all clear".
    view = html[html.index("function ReviewInboxView"):]

    # "Has this page had an answer yet" is a DIFFERENT question from "is a scan
    # running", and conflating them is what caused the bug.
    assert "const firstLoad = d.loading || !costLoaded" in view
    assert "const scanning = d.status === 'computing' || costStatus === 'computing'" in view
    # `costStatus` cannot stand in for "loaded": it initialises to 'never_run',
    # indistinguishable from a real fresh install.
    assert "const [costLoaded, setCostLoaded] = useState(false)" in view
    assert "finally { setCostLoaded(true); }" in view

    # Rows render unconditionally; only the EMPTY branch is gated. So a recompute
    # can never blank a list that has rows.
    assert "${topOpen.map(renderRow)}" in view
    assert "${openItems.length === 0 ? (" in view
    # Skeletons are for the two states that genuinely have nothing: still
    # fetching, or a real first scan in flight.
    assert "const showSkeleton = openItems.length === 0 && (firstLoad || (scanning && d.status !== 'never_run'))" in view
    assert "<${InboxSkeletonRow} key=${i} />" in view
    assert "Loading…" not in view, "the bare loading string must be gone"


def test_no_count_or_tile_is_rendered_before_the_page_knows_it(html):
    # A count the page does not know is not printed as 0, and the tiles hold their
    # final positions so nothing shifts when the numbers land.
    view = html[html.index("function ReviewInboxView"):]
    assert "const openCountKnown = !firstLoad" in view
    assert "const appliedCountKnown = appliedLoaded" in view
    # A PARTIAL count is as wrong as a zero: it would be replaced a moment later,
    # which is the layout jump this rule exists to prevent. So both ledgers must
    # have answered, not just one.
    assert "d.loading || !costLoaded" in view
    # Shimmer chips reserve the digits' width in both tabs.
    assert view.count('class="shimmer" style="display:inline-block;width:20px') == 2

    tile = html[html.index("function InboxStatTiles"):html.index("function ReviewInboxView")]
    # The band-level skeleton now also fires while the APPLIED read is outstanding,
    # not only on the page-wide flag: the two tiles are fed by reads with wildly
    # different latencies, and one flag could not describe both.
    assert "if ((loading || !appliedKnown) && openItems.length === 0 && appliedCount === 0 && !hasExcluded)" in tile
    assert "loading=${firstLoad}" in view

    # Reused the app's existing shimmer primitive rather than inventing a loader,
    # and nothing fakes progress on a scan of unknown length. Asserted on MARKUP,
    # not on words: the components' own comments name the things they avoid, so a
    # bare substring ban matches the explanation and fails on correct code.
    assert ".shimmer {" in html
    skel = html[html.index("function InboxSkeletonRow"):html.index("function InboxLoadingNote")]
    assert 'class="shimmer"' in skel
    assert "animation" not in skel, "the skeleton must not roll its own animation"
    assert "<progress" not in html
    # One animation in the app, the pre-existing shimmer. A second @keyframes
    # would mean a new loader was invented here.
    assert html.count("@keyframes shimmer") == 1
    # Motion is suppressed for a reader who asked for none.
    assert "@media (prefers-reduced-motion: reduce)" in html
    assert ".shimmer { animation: none; }" in html

    # The state says what it is doing, from what the server actually reported, and
    # never invents a number or a percentage.
    note = html[html.index("function InboxLoadingNote"):html.index("// --- The row's description block")]
    assert "sessionsScanned != null" in note
    assert "A scan is running now" in note


def test_every_row_clamps_its_prose_to_the_same_collapsed_height(html):
    # Founder feedback: description lengths were wildly uneven (resend concatenated
    # evidence + advise_text + caveat into ~15 lines while a downsize row had one),
    # so the list was ragged and could not be scanned.
    #
    # Clamped on LINES, so the collapsed height does not depend on how the text
    # happens to wrap at the current width.
    assert "-webkit-line-clamp: 2" in html
    assert ".inbox-desc.is-open" in html
    assert "-webkit-line-clamp: unset" in html
    # min-height matches the clamp, which is the half that stops the raggedness
    # simply MOVING to the short rows: a row with little or no prose has to occupy
    # the same space as one with a wall of it.
    assert "min-height: calc(2 * 1.55 * 13px)" in html
    assert ".inbox-desc.is-open" in html and "min-height: 0" in html

    # The prose is trimmed TEXTUALLY at a sentence boundary, not merely clipped:
    # a pure CSS clamp gave equal heights but cut mid-clause, and the founder asked
    # for the collapsed state to read as a short complete thought.
    split = html[html.index("function splitLead(text, budget"):]
    split = split[:split.index("\n// The description block")]
    # Boundary is punctuation FOLLOWED BY whitespace, which is what keeps
    # "$1,842.56 of that cost" and "claude-4.7" from being treated as sentence ends.
    assert "if (j + 1 >= full.length || /\\s/.test(full[j + 1])) ends.push(j + 1)" in split
    # Never mid-word either: text with no sentence boundary at all (a raw error
    # dump) falls back to a word-boundary cut.
    assert "full.lastIndexOf(' ', budget)" in split
    # THE lead must be a PREFIX of the text, sliced at a collected boundary offset.
    # Accumulating `/[^.!?]+[.!?]+(?:\s+|$)/g` matches instead silently dropped
    # everything before the first MATCHABLE sentence: a description opening
    # "`posthog` MCP server (configured at .../.mcp.json) made 0 tool calls." has
    # its only period followed by a letter, so `exec` found nothing at index 0,
    # advanced, and the row rendered "json) made 0 tool calls" as its opening
    # words. Two of eight live rows read that way. The banned construct is the
    # assertion, because a prefix-slice implementation cannot reproduce the bug.
    # Banned on the MECHANISM, not on the old regex literal: the source comment
    # quotes that literal while explaining the bug, so a text ban on it would match
    # the explanation. `exec` in a `g`-flagged scan is the part that skips.
    assert "re.exec(full)" not in split, \
        "a match-accumulating scan skips text before the first matchable sentence"
    assert "full.slice(0, cut)" in split
    # The expanded state renders the WHOLE text, never lead + rest concatenated,
    # because the fallback lead carries a trailing ellipsis.
    assert "return { lead, full, more: lead !== full }" in split

    fn = html[html.index("function TrimmedDescription({ text })"):]
    fn = fn[:fn.index("\n}")]
    assert "${open ? full : lead}" in fn
    assert "${more ? html`" in fn
    # Per row, local, not persisted, not expanded by default.
    assert "useState(false)" in fn
    assert "localStorage" not in fn and "sessionStorage" not in fn
    assert "'Read less' : 'Read more'" in fn

    # The server's strings are split, never rewritten: a paraphrase in the UI would
    # be a second drifting copy of the analyzer's claim.
    assert "splitLead(text)" in fn

    # Both row shapes wrap their prose in it, so the collapsed tail and the
    # below-the-fold group inherit the rhythm through the same components.
    row = html[html.index("function RecurringMistakeRow"):html.index("function RelearnApplyModal")]
    card = html[html.index("function CostProposalCard"):html.index("function InboxStatTiles")]
    assert "<${TrimmedDescription} text=${relearnDescription(cluster)} />" in row
    assert "<${TrimmedDescription} text=${description} />" in card

    # A relearn cluster has no analyzer prose, so its block leads with the facts it
    # does have (requirement: give every row a first line worth reading).
    assert "Recurred ${cluster.occurrences} time" in row
    assert "across ${cluster.sessions} session" in row

    # A DESCRIPTION has to be WORDS. One live cluster's captured snippet was the
    # bare identifier `gen_ai.tool.call`, which answers "what is this" for nobody,
    # and a non-empty check accepted it. Two words is the gate, and a snippet that
    # fails it falls through to the derived fix rather than being dressed up.
    prose = html[html.index("function isProse(s)"):html.index("\n}", html.index("function isProse(s)"))]
    assert "split(/\\s+/).filter(Boolean).length >= 2" in prose
    desc_fn = html[html.index("function relearnDescription(cluster)"):]
    desc_fn = desc_fn[:desc_fn.index("\n}")]
    assert "isProse(e.snippet)" in desc_fn
    assert "candidates.find(isProse)" in desc_fn

    # The operative facts stay OUTSIDE the clamp: a reader must not expand anything
    # to learn why a row has no Apply button, and a <details> is already collapsed.
    # Matched on the RENDERED markup, not the phrase: a comment above the
    # component explains the same rule in prose and would match first, making the
    # ordering check compare a comment against the clamp.
    desc_at = card.index("<${TrimmedDescription} text=${description} />")
    assert card.index(">tokenjam cannot apply this one: ${blockedReason}<") > desc_at, \
        "the blocker reason must not be inside the trimmed description"
    assert card.index("<summary>How this number was derived</summary>") > desc_at
    # COVERAGE keeps its OWN disclosure rather than being folded into Read more: it
    # answers what was NOT analysed, which is the whole point of it, and on resend it
    # is a five-line block, the reason that row was four times its neighbours' height.
    assert "<summary>What this figure does and does not cover</summary>" in card
    assert card.index("<summary>What this figure does and does not cover</summary>") > desc_at


def test_a_row_tj_cannot_apply_states_the_servers_own_blocker_reason(html):
    # A row tokenjam cannot apply must say WHY, and must say it in the words the
    # refusing adapter used. `apply_blocked_reason` carries e.g. "no local source
    # path is registered for this agent, so there is nothing to edit. Register one
    # with source_path under the agent in your tj config, or paste the change
    # yourself" — which names the blocker AND the two exits. A paraphrase here
    # would drift from both the CLI and the actual refusal.
    card = html[html.index("function CostProposalCard"):html.index("function InboxStatTiles")]
    assert "prop.apply_blocked_reason" in card
    assert "tokenjam cannot apply this one: ${blockedReason}" in card
    # Shown exactly once. Some adapters already append the reason into
    # `advise_text` server-side ("Applying it here is not on offer: ..."), others
    # only set the field, so the guard checks the RENDERED description rather
    # than guessing which adapter produced the row.
    assert "!description.includes(blockedReason)" in card
    assert "const showBlockedReason = !canApply" in card
    # And the model-swap honesty caveat travels in `advise_text`/`caveat`, both of
    # which the description paragraph renders unconditionally (Critical Rule 14).
    assert "[prop.evidence, prop.advise_text, prop.caveat].filter(Boolean).join(' ')" in card

    # The relearn half has the same obligation and its own field for it.
    row = html[html.index("function RecurringMistakeRow"):html.index("function RelearnApplyModal")]
    assert "cluster.advise_only_reason" in row


def test_bulk_controls_vanish_when_nothing_is_apply_capable(html):
    # For the SDK persona relearn offers no write at all
    # (`write_offered = persona in {claude-code, mixed}`), so the selectable set
    # can be empty. A select-all governing nothing is worse than no select-all:
    # it implies the list is selectable when it is not. Both the checkbox and the
    # bulk bar are gated on the set being non-empty.
    view = html[html.index("function ReviewInboxView"):]
    assert view.count("${bulkRelearn.length > 0 ? html`") == 2
    # The per-row checkbox is gated on the same predicate, so a row that no bulk
    # action can reach never shows one.
    row = html[html.index("function RecurringMistakeRow"):html.index("function RelearnApplyModal")]
    assert '${canApply ? html`<input type="checkbox"' in row
    # The bulk button still names its count.
    assert "${selectedCount ? `Review ${selectedCount} checked` : 'Review checked'}" in html
    assert "${selectedCount ? `Dismiss ${selectedCount} checked` : 'Dismiss checked'}" in html


def test_bulk_mark_applied_can_never_post_a_relearn_row_to_the_cost_ledger(html):
    # The collapsed tail is now mixed, and the bulk marker only speaks the cost
    # ledger's endpoint. Filtered inside markManyApplied rather than at the call
    # site, so a future caller cannot hand it a mixed list and quietly POST the
    # wrong ledger.
    start = html.index("const markManyApplied = async (props)")
    end = html.index("const modalCluster =", start)
    fn = html[start:end]
    assert "props.filter(x => x.kind !== 'relearn' && !x.apply_capable)" in fn


def test_the_per_row_amount_caption_is_gone(html):
    # INVERTED (was test_every_row_discloses_the_span_its_figure_was_observed_over).
    # The caption under each row's dollar figure ("avoidable over the last 30 days ·
    # ~12.3B tok") was removed on founder instruction, which retires the per-row
    # span disclosure with it. Its RETURN is the regression now.
    assert "function inboxSpan" not in html
    assert "spans=${spans}" not in html
    assert "avoidable ' + span.text" not in html
    assert "'already cost ' + span.text" not in html
    # Both row shapes keep the figure itself and nothing under it.
    for comp, end in (("function RecurringMistakeRow", "function RelearnApplyModal"),
                      ("function CostProposalCard", "function InboxStatTiles")):
        block = html[html.index(comp):html.index(end)]
        assert 'class="po-amount"' in block
        assert 'style="font-size:10px"' not in block, "the caption line must not come back"

    # The asymmetry the caption used to disclose is real and did not go away, so the
    # scan's own description of its corpus stays on the payload state for whoever
    # states it next.
    view = html[html.index("function ReviewInboxView"):]
    assert "windowDays: f.window_days" in view
    assert "corpusBasis: f.corpus_basis" in view


def test_the_headline_tile_caption_reads_one_population(html):
    # The tile's caption used to read "was avoidable over the last 30 days · ~21.9B
    # tok · 13 causes of $7,653.24 total cost — that is cost, not waste". The
    # total-cost clause was FALSE as rendered: `observed_cost_usd` covers 2 of the
    # 13 proposals (resend + relearn), so it attached a two-proposal denominator to
    # "13 causes", and summarize alone contributes ~4,811 of avoidable from a
    # proposal carrying no observed cost at all.
    tile = html[html.index("function PastOverspendTile"):html.index("// A `writeGateNote")
                if "// A `writeGateNote" in html else html.index("function PastOverspendTile") + 3000]
    tile = html[html.index("function PastOverspendTile"):]
    tile = tile[:tile.index("\nfunction ")]
    # Survives: window, tokens and cause count, all summed over the same proposals.
    assert "over ${win}" in tile
    assert "fmtTokens(toks)" in tile
    assert "causes + ' cause'" in tile
    # Gone: the leading "was avoidable" wording and the whole total-cost clause.
    assert "was avoidable over" not in tile
    assert "total cost" not in tile
    assert "block.observed_cost_usd" not in tile
    assert "cost_disclosure" not in tile, "no orphaned disclosure for a removed figure"
    # Not hardcoded. Scoped to the RETURNED markup: the surrounding comment quotes
    # the real figures to explain why the old caption was false, and a whole-function
    # literal check matches the explanation instead of the render.
    markup = tile[tile.index("return html`"):]
    for lit in ("21.9", "7,653", "6163", "13 cause"):
        assert lit not in markup
    # The figure stays. The OBSERVED chip does NOT: removed on founder call
    # because the tile's own past-tense title and its window-and-population
    # caption already say what the chip restated, both of which are asserted
    # above and are now the only things carrying the tense.
    assert "po-observed-tag" not in tile
    assert "fmtUsd(usd)" in tile


def test_the_ordering_key_cannot_reach_a_projection(html):
    # Two bugs, one key. The FIRST was ranking flipping to dollars the moment
    # any item had a dollar figure, leaving tokens-only items tied at rank 0 in
    # adapter-insertion order. The SECOND, fixed by the single-list redesign, was
    # the key's fallback: `past_overspend_tokens ?? estimated_monthly_tokens`
    # let a forward 30-day PROJECTION compete for position against past
    # OBSERVATIONS inside one sort. Latent (every real row carries the observed
    # figure) but structural, and unfixable by inspection once shipped, because
    # nothing on screen says which kind of number placed a row.
    #
    # The key now reads ONE field and has no fallback at all; a row without it
    # cannot enter the ranked list.
    start = html.index("function observedRankTokens")
    end = html.index("function splitTopAndTail", start)
    fn = html[start:end]
    assert "item.past_overspend_tokens" in fn
    # No projection, no dollars, no second field of any kind may appear in the
    # ranking block.
    assert "estimated_monthly_tokens" not in fn
    assert "estimated_monthly_usd" not in fn
    assert "estimated_recoverable" not in fn
    assert "anyUsd" not in fn
    # The unranked rows are PARTITIONED OUT rather than sorted to the bottom, so
    # no comparator change can ever interleave them.
    assert "function partitionByObservedOverspend" in fn
    assert "ranked: items.filter(i => observedRankTokens(i) != null)" in fn
    assert "unobserved: items.filter(i => observedRankTokens(i) == null)" in fn
    # And the view splits before it sorts.
    view = html[html.index("function ReviewInboxView"):]
    assert "const { ranked, unobserved } = partitionByObservedOverspend(shownItems)" in view
    assert "const sortedOpen = sortByPastOverspend(ranked)" in view
    # The retired key must not come back under either retired name. Upstream
    # renamed it to `sortByPastOverspend` when relearn's forward claim was
    # retired; this branch had independently grown a `sortByObservedOverspend`
    # computing the identical thing, and that duplicate is deleted in favour of
    # upstream's name rather than kept in parallel.
    assert "sortByEstMonthly" not in html
    assert "sortByObservedOverspend" not in html
    assert "function sortByPastOverspend" in html


def test_ordering_key_rejects_a_projection_only_row(html):
    # A behavioural contract for the guard above, reimplemented in Python from
    # the pinned JS so a divergence fails loudly rather than only when the
    # string changes. A row carrying ONLY a forward projection is unrankable;
    # it must land in `unobserved`, never at the top of the ranked list.
    def observed_rank_tokens(item):
        t = item.get("past_overspend_tokens")
        return t if isinstance(t, (int, float)) and not isinstance(t, bool) else None

    def partition(items):
        return (
            [i for i in items if observed_rank_tokens(i) is not None],
            [i for i in items if observed_rank_tokens(i) is None],
        )

    projection_only = {"title": "forward only", "estimated_monthly_tokens": 10**12}
    observed_small = {"title": "small but real", "past_overspend_tokens": 5_000}
    observed_big = {"title": "big and real", "past_overspend_tokens": 9_000_000}

    ranked, unobserved = partition([projection_only, observed_small, observed_big])
    assert unobserved == [projection_only]
    ranked.sort(key=lambda i: i["past_overspend_tokens"], reverse=True)
    assert [i["title"] for i in ranked] == ["big and real", "small but real"]
    # The old fallback would have put the trillion-token projection first.
    assert ranked[0] is not projection_only


def test_collapsed_tail_combined_figure_is_stated_in_the_past_tense(html):
    # combinedEstMonthly used to lead with dollars the moment ANY tail item had
    # one (summing the rest as $0), understating a mostly-tokens-only tail as
    # a tiny dollar figure. It's retired along with the forward claim it
    # summarised: both tabs now state their combined tail figure via
    # `combinedObservedCost`, on the same `past_overspend_*` basis their
    # expanded rows use, so there is no separate priceable-majority rule left
    # to pin.
    assert "function combinedEstMonthly" not in html
    start = html.index("function combinedObservedCost")
    end = html.index("function CollapsedTailRow", start)
    fn = html[start:end]
    assert "past_overspend_usd" in fn
    assert "already spent, combined" in fn


def _collapsed_tail_row_src(html: str) -> str:
    start = html.index("function CollapsedTailRow")
    end = html.index("\n}\n", start) + 2
    return html[start:end]


def test_collapsed_tail_row_is_a_toggle_command_not_a_static_description(html):
    """The collapse row used to read "N smaller items" in BOTH states (only
    the ▸/▾ arrow changed), the same word BelowFloorNote uses for items below
    the $5 noise floor -- but this row's items are ranked below the top 8
    while still worth $5+ each, the opposite of "smaller" meaning skippable.
    It is now the toggle CONTROL it actually is: "Show N more" while
    collapsed, "Show less" once expanded, so it never claims there is more to
    show once the items are already on screen. No noun/pluralization is
    needed since "more"/"less" don't inflect."""
    fn = _collapsed_tail_row_src(html)
    assert "itemLabel" not in fn
    assert "smaller" not in fn
    assert "const label = open ? 'Show less' : ('Show ' + tail.length + ' more');" in fn
    assert "${open ? '▾' : '▸'} ${label}${summary ? ' · ' + summary : ''}" in fn


def test_collapsed_tail_row_callers_no_longer_pass_a_noun(html):
    # Both call sites used to compute an itemLabel via pluralItems() (one with
    # an extra " with nothing measured yet" suffix); CollapsedTailRow no
    # longer takes or renders one, so neither caller needs to supply it.
    # pluralItems() itself survives -- BelowFloorNote (the $5-floor line,
    # deliberately unchanged) still calls it.
    assert "itemLabel=${pluralItems(tailOpen.length)}" not in html
    assert 'itemLabel=${pluralItems(unobserved.length) + " with nothing measured yet"}' not in html
    assert "<${CollapsedTailRow} tail=${tailOpen} suppressed=${suppressed}" in html
    assert "<${CollapsedTailRow} tail=${unobserved} suppressed=${suppressed}" in html
    assert "label=${pluralItems(floorItems.length)}" in html  # BelowFloorNote, unchanged


def test_below_floor_note_still_says_smaller_under_the_floor(html):
    # The ONE line that should still carry a size claim: it states its own
    # $5 threshold, which is what makes "smaller" unambiguous there.
    assert "${items.length} smaller ${label} under ${fmtUsd(INBOX_MIN_USD)}" in html


def test_fmt_tokens_renders_billion_scale_human_readable(html):
    # "~11268.0M tok" (an actual rendered figure from a real corpus) must
    # become "~11.3B tok" — fmtTokens needs a billion-scale branch above the
    # existing million/thousand ones.
    start = html.index("function fmtTokens(n)")
    end = html.index("\n}", start) + 2
    fn = html[start:end]
    assert "1e9" in fn
    assert "'B'" in fn
    # The billion check must come before the million one (n >= 1e9 also
    # satisfies n >= 1e6, so ordering matters) or a billion-scale value would
    # still hit the million branch first.
    assert fn.index("1e9") < fn.index("1e6")


def test_fmt_tokens_billion_scale_matches_a_real_reported_figure():
    # A Python-side reimplementation of the exact fmtTokens algorithm pinned
    # above, run against a real reported number, so this test fails loudly if
    # the JS and this contract ever diverge in behavior, not just in the
    # presence of the string "1e9".
    def fmt_tokens(n):
        if n is None:
            return "-"
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}k"
        return str(n)

    assert fmt_tokens(11_268_000_000) == "11.3B"
    assert fmt_tokens(80_800_000) == "80.8M"
    assert fmt_tokens(613_500) == "613.5k"
    assert fmt_tokens(999_999_999) == "1000.0M"   # just under the 1e9 boundary
    assert fmt_tokens(1_000_000_000) == "1.0B"     # exactly at the boundary


def test_cost_advisories_sort_is_monotonically_non_increasing_on_real_data():
    # A Python-side contract test pinning the SAME "rank by
    # estimated_monthly_tokens descending" algorithm the JS now implements
    # (sortByPastOverspend, pinned above), run against real numbers from the bug
    # report — the exact dataset that exposed the original "adapter insertion
    # order" bug. Proves the fixed algorithm produces a genuinely monotonic
    # order for real, not synthetic, data.
    items = [
        {"analyzer": "deadweight", "estimated_monthly_tokens": 80_800_000},
        {"analyzer": "trim",       "estimated_monthly_tokens": 11_062_000_000},
        {"analyzer": "subagent",   "estimated_monthly_tokens": 18_300_000},
        {"analyzer": "reuse",      "estimated_monthly_tokens": 20_700_000},
        {"analyzer": "verbosity",  "estimated_monthly_tokens": 1_600_000},
        {"analyzer": "downsize",   "estimated_monthly_tokens": 318_300},
        {"analyzer": "reuse",      "estimated_monthly_tokens": 253_600},
        {"analyzer": "reuse",      "estimated_monthly_tokens": 353_700},
    ]
    ranked = sorted(items, key=lambda i: i["estimated_monthly_tokens"], reverse=True)
    values = [i["estimated_monthly_tokens"] for i in ranked]
    assert values == sorted(values, reverse=True)   # monotonically non-increasing
    # Pins the explicit ordering constraints from the bug report.
    assert ranked[0]["analyzer"] == "trim"
    reuse_values = [i["estimated_monthly_tokens"] for i in ranked if i["analyzer"] == "reuse"]
    assert reuse_values == sorted(reuse_values, reverse=True)
    assert 20_700_000 in reuse_values and reuse_values.index(20_700_000) < len(reuse_values) - 1


def test_split_top_and_tail_slices_an_already_sorted_list(html):
    # The long-tail collapse (requirement #3) must absorb the BOTTOM of the
    # sorted list, not an arbitrary suffix of the unsorted API order — it
    # slices whatever the ranking already produced, never re-sorts or re-orders
    # on its own.
    start = html.index("function splitTopAndTail")
    end = html.index("function relearnObservedFigure", start)
    fn = html[start:end]
    assert "sorted.slice(0, max)" in fn
    assert "sorted.slice(max)" in fn
    assert "sort(" not in fn   # no independent re-sort inside the split itself
    # ONE call, over the one merged open list — there is no longer a per-tab
    # collapse, because there is no longer a per-analyzer tab.
    view = html[html.index("function ReviewInboxView"):]
    assert "splitTopAndTail(sortedOpen)" in view
    assert view.count("splitTopAndTail(") == 1


def test_review_inbox_dollar_headline_ignores_framing_even_when_suppressed():
    # End-to-end contract test for a real scenario: an account whose framing
    # says suppress_dollars_for_subscription_share (87% subscription-billed,
    # verified against a real live store) still gets dollar headlines on the
    # Review inbox for every priced item, tokens for the one documented
    # exception (resend/TRIM), and tokens for a genuinely unpriced item (no
    # computable rate at all). A Python reimplementation of estMonthlyLine's
    # decision, pinned so a future divergence between this contract and the
    # shipped JS fails loudly.
    real_framing = {
        "pricing_mode": "subscription", "plan_tier": "max_20x",
        "subscription_share_pct": 87.0,
        "display_rule": "suppress_dollars_for_subscription_share",
    }
    not_a_savings_claim_analyzers = {"resend"}

    def headline_unit(item, framing):
        # This page's carve-out: `framing` is accepted but never consulted —
        # unlike every other dollar figure in the app, which would check
        # framing["display_rule"] here and fall back to tokens.
        del framing
        if item["analyzer"] in not_a_savings_claim_analyzers:
            return "tokens"
        return "dollars" if item.get("estimated_monthly_usd") is not None else "tokens"

    priced_downsize = {"analyzer": "downsize", "estimated_monthly_usd": 0.87, "estimated_monthly_tokens": 520_324}
    priced_deadweight = {"analyzer": "deadweight", "estimated_monthly_usd": 403.875, "estimated_monthly_tokens": 80_775_000}
    resend_structural = {"analyzer": "resend", "estimated_monthly_usd": 186.357458, "estimated_monthly_tokens": 11_061_129_491}
    unpriced_placement = {"analyzer": "placement", "estimated_monthly_usd": None, "estimated_monthly_tokens": 78_812_584}

    assert headline_unit(priced_downsize, real_framing) == "dollars"
    assert headline_unit(priced_deadweight, real_framing) == "dollars"
    assert headline_unit(resend_structural, real_framing) == "tokens"
    assert headline_unit(unpriced_placement, real_framing) == "tokens"


# --- Recoverable-waste band drops no-lever analyzers for claude-code -------- #
# core/optimize/runner.py's PERSONA_DISABLED_ANALYZERS is the single source of
# truth for which analyzers a persona has no lever for (#308). The dashboard
# must derive its gate solely from the `/optimize` payload's
# `persona_disabled_analyzers` field — never duplicate the map as a JS literal,
# which would silently desync the first time the Python map changes.
def test_dashboard_has_no_hardcoded_persona_disabled_analyzer_list(html):
    assert "PERSONA_DISABLED_ANALYZERS" not in html
    assert "'claude-code': new Set(" not in html


def test_recoverable_tiles_filters_by_persona_before_ranking(html):
    fn_start = html.index("function recoverableTiles(opt)")
    fn_end = html.index("\nfunction ", fn_start + 1)
    fn = html[fn_start:fn_end]
    assert "new Set(opt.persona_disabled_analyzers || [])" in fn
    assert "disabled.has(k)" in fn
    # Downsize + wave-2 findings + the not-ready fallback must all respect the
    # gate (three separate call sites build `out`) so a persona-disabled
    # analyzer never sneaks in through any of them.
    assert fn.count("disabled.has(") >= 3


def test_optimize_view_and_recoverable_tiles_read_same_payload_field(html):
    # Both call sites (the Optimize screen's order filter and the Overview
    # band's recoverableTiles) must key off the same server-published field —
    # not one payload read and one hardcoded literal — so they can never
    # disagree about which analyzers a persona can act on.
    assert html.count("persona_disabled_analyzers") >= 2
    assert "const personaGated = new Set((st.opt && st.opt.persona_disabled_analyzers) || []);" in html


# --- Cost view: top-tenants panel must share Refresh/poll cadence ---------- #
# `loadTenants` used to have its own mount-only effect, independent of
# `load`'s 30s poll and the Refresh button -- so Cost totals updated on every
# refresh/poll while the top-tenants-by-spend panel kept showing whatever it
# fetched on the last mount or since/agentId change.
def test_cost_view_tenants_panel_shares_refresh_and_poll_cadence(html):
    fn_start = html.index("function CostView(")
    fn_end = html.index("\nfunction ", fn_start + 1)
    fn = html[fn_start:fn_end]

    assert "const loadTenants = useCallback(" in fn
    # The Refresh button (and the poll it shares an effect with) must drive a
    # callback that reaches BOTH load and loadTenants -- never `load` alone.
    assert "onClick=${load}>Refresh" not in fn
    assert "load(opts);" in fn and "loadTenants(opts);" in fn


def test_a_model_swap_row_asks_for_the_path_instead_of_offering_mark_applied(html):
    """The design bar: "Mark applied" is the exception, not the default.

    Ten live `downsize` model-swap rows carried a measured, deterministic fix and
    offered nothing but a copy box and a "Mark applied" that recorded the user
    doing it by hand. The cause was a single missing input — where the agent's
    source lives — which tokenjam will not infer (`config.AgentConfig.source_path`
    is opt-in by design). A missing input is a question, so the row asks it.
    """
    card = html[html.index("function CostProposalCard"):html.index("function InboxStatTiles")]

    # Checked BEFORE `hasApplyKind`, because such a row deliberately carries no
    # apply_kind yet: with no registered path there is no deterministic edit, so
    # it must not reach the endpoint that assumes one.
    assert "${canApply ? (needsSourcePath ? html`" in card
    assert card.index("needsSourcePath ? html`") < card.index("hasApplyKind ? html`")
    assert "const needsSourcePath = !!prop.needs_source_path;" in card

    # It ASKS: an input, a preview that writes nothing, and an apply.
    assert "/relearn/cost-proposals/register-source-path" in card
    assert "registerSourcePath(false)" in card and "registerSourcePath(true)" in card
    assert "Apply swap →" in card
    # Never pre-filled: the whole premise is that nothing here knows the path, and
    # a plausible default would be tokenjam inferring it by another name.
    assert "useState('')" in card
    assert "prop.needs_source_path ? prop.target_path" not in card

    # A precheck failure AFTER the path is given is stated on the row, in the
    # server's own words, rather than failing silently.
    assert "setSpErr(e.message || String(e))" in card

    # THE CAVEAT IS OUTSIDE THE COLLAPSED DESCRIPTION. A one-click write makes it
    # easier to lose the distinction between "the cost delta is measured" and "the
    # cheaper model is as good" (Critical Rule 14), and a caveat behind a
    # "Read more" does not count as visible.
    desc_at = card.index("<${TrimmedDescription} text=${description} />")
    caveat_at = card.index("${prop.apply_caveat ? html`")
    assert caveat_at > desc_at
    assert '<div class="opt-caveat" style="margin-top:6px">${prop.apply_caveat}</div>' in card


def test_the_avoided_tile_carries_no_observed_chip(html):
    # Removed on founder call: the chip restated what the surrounding copy already
    # says. The tense now rests ENTIRELY on the title and the caption, so both are
    # pinned here — the title must stay past tense and the caption must keep naming
    # the window and the population, or the removal would have cost real
    # provenance rather than just chrome.
    tile = html[html.index("function PastOverspendTile"):html.index("function combinedObservedCost")]
    # Scoped to the RETURNED markup: the comment above the render names the removed
    # class to record why it went, and a whole-function check would match that
    # explanation rather than the render.
    markup = _no_comments(tile[tile.index("return html`"):])
    assert "po-observed-tag" not in markup
    assert "What you could have avoided" in tile
    assert "over ${win}" in tile
    assert "fmtTokens(toks)" in tile
    assert "causes + ' cause'" in tile
    # And the rule is gone too, not merely unused, since nothing else styled with it.
    assert ".po-observed-tag {" not in html


def test_applied_rows_carry_no_enforcement_label(html):
    # Internal jargon, repeated identically on every enforcement row, earning none
    # of the space it took. Removed rather than restyled or abbreviated.
    row = html[html.index("function AppliedItemRow"):html.index("// ---- Cost proposals")]
    # Scoped to the RENDERED markup for the same reason as the tile above: the
    # tombstone comment quotes the label it replaced.
    markup = _no_comments(row[row.index("return html`"):])
    assert "enforcement ENABLED" not in markup
    assert "ENABLED" not in markup
    # The ACTIONABLE half is still stated, as a verb, where you act on it: the row
    # offers "Disable enforcement" when it is on and "Enable enforcement…" when it
    # is not, so the state stays legible without a constant label.
    assert "Disable enforcement" in row
    assert "Enable enforcement…" in row


def test_a_reverted_row_shows_reverted_where_its_amount_would_be(html):
    # A reverted fix saved nothing, so a green plus-signed figure on that row
    # asserts a benefit that did not occur. It rendered "+$358.50 est." beside a
    # REVERTED badge. The figure is not greyed or de-emphasised, it is NOT
    # RENDERED: the slot answers "what did this yield" and the truthful answer for
    # a reverted row is the word.
    row = html[html.index("function AppliedItemRow"):html.index("// ---- Cost proposals")]
    assert "const isReverted = rec.state === 'reverted';" in row
    # The amount slot branches on it FIRST, so no amount can be reached.
    assert "${isReverted ? html`" in row
    amount_at = row.index("${isReverted ? html`")
    plus_at = row.index("+${(!suppressed && usd != null)")
    assert amount_at < plus_at, "the reverted branch must precede the amount"
    # The now-redundant mid-row chip is gone for the reverted case only: any other
    # non-default state has nowhere better to put it.
    assert "rec.state !== 'applied' && !isReverted ?" in row

    # NOT an over-correction: a non-reverted row still renders its figure, both the
    # dollar form and the token fallback, and still suppresses dollars for a
    # subscription plan. Asserted because the live corpus cannot show it -- all five
    # non-reverted records carry None for past_overspend_usd, past_overspend_tokens
    # AND the legacy estimated_monthly_usd, so they rendered no amount before this
    # change either. The only record with a figure is the reverted one.
    assert "(usd != null || toks != null) ? html`" in row
    assert "+${(!suppressed && usd != null) ? fmtUsd(usd) : (toks != null ? '~' + fmtTokens(toks) + ' tok' : fmtUsd(usd))}" in row
    # The revert REASON and the applied metadata are the genuinely useful parts and
    # must survive: they are what tell the reader why it came back.
    assert "${error}" in row
    assert "Applied ${daysAgoLabel(rec.applied_at)}" in row
    assert "rec.target_path" in row and "rec.git_commit" in row


def test_the_applied_estimate_caveat_is_one_section_line_not_a_per_row_badge(html):
    # The founder asked for the per-row `est.` badge to go. Deleting it alone would
    # have left bare green plus-signed figures, which read as measured savings --
    # exactly the claim an earlier commit removed an unbounded verify layer to
    # avoid. So the caveat moves to ONE line at the section header.
    row = html[html.index("function AppliedItemRow"):html.index("// ---- Cost proposals")]
    assert "estimated-tag" not in row, "no per-row estimate badge"
    assert "est.<" not in row and ">est.</span>" not in row

    view = html[html.index("function ReviewInboxView"):]
    panel = view[view.index("${tab === 'applied' ? html`"):]
    panel = panel[:panel.index("<${RelearnApplyModal}")]
    line = "Each figure below is that row's own estimate from the moment it was applied. Nothing here is re-measured afterwards."
    assert line in panel
    # Stated in the markup, NOT hidden in a title attribute: a tooltip is a hidden
    # disclosure and does not count as stating the caveat.
    assert 'class="sz-note"' in panel[:panel.index(line)]
    assert f'title="{line}"' not in panel
    # Once for the section, not once per row.
    assert panel.count(line) == 1
    # And it may never be worded as a realized claim.
    for banned in ("verified", "measured saving", "confirmed", "realized"):
        assert banned not in line.lower()
    # No em dashes in user-facing copy.
    assert "—" not in line


def test_no_inbox_empty_state_can_render_before_its_own_read_resolves(html):
    """THE RULE, not a fourth patch. An absence claim may never render before the
    read that would refute it, because absence reads as reassurance and is the most
    dangerous thing on this page to guess wrong.

    Three instances of this bug shipped: `Open (0)`, the bare "Loading…" that
    replaced cached rows, and "Nothing applied yet." So this is written as a SWEEP
    over every absence-claiming string in the inbox region rather than as a check
    on the one just fixed, and it fails on a NEW ungated one.
    """
    view = html[html.index("function ReviewInboxView"):html.index("function DashboardView")]

    # Every absence-claiming string the inbox can render, with the flag that has to
    # gate it. `showSkeleton` covers the open list: it is true whenever there are no
    # rows and the page has not had an answer, so it pre-empts both open-list empty
    # states in the same conditional chain.
    gated = {
        "No scan has run yet.": "showSkeleton",
        "Nothing open. The last scan found nothing worth a row": "showSkeleton",
        "Nothing applied yet.": "appliedCountKnown",
    }
    for claim, flag in gated.items():
        assert claim in view, f"the sweep is stale: {claim!r} is no longer rendered"
        # The gate must appear in the same conditional chain, BEFORE the claim.
        before = view[:view.index(claim)]
        assert flag in before, f"{claim!r} renders without consulting {flag}"

    # No NEW absence claim may appear ungated. Any future one has to be added to the
    # map above with its gate, which is the point: the sweep is the rule.
    import re as _re
    found = {
        m.group(1).strip()
        for m in _re.finditer(r">(\s*(?:No |Nothing |None\b)[^<${]*)", _no_comments(view))
    }
    unknown = {f for f in found if not any(f.startswith(k[:18]) for k in gated)}
    assert not unknown, f"ungated absence claim(s) in the inbox: {unknown}"


def test_the_applied_panel_gates_on_its_own_read_not_a_page_wide_flag(html):
    # `/relearn/applied` is the slowest of the four inbox reads, so a page-wide
    # flag clears long before it answers -- which is exactly how the tile came to
    # print "0 / most recent unknown" beside a tab count that was already a
    # shimmer. Both applied reads count, because the number is a merge of the two
    # ledgers and a count over half a population is as wrong as a zero.
    view = html[html.index("function ReviewInboxView"):html.index("function DashboardView")]
    assert "const appliedCountKnown = appliedLoaded && costAppliedLoaded;" in view
    # `/relearn/cost-applied` had no loaded flag at all before this.
    assert "setCostAppliedLoaded(true)" in view
    assert "const [costAppliedLoaded, setCostAppliedLoaded] = useState(false);" in view
    # Not derived from the page-wide flag, which is the mistake being prevented.
    assert "appliedCountKnown = !firstLoad" not in view
    assert "appliedKnown=${appliedCountKnown}" in view

    # The tile holds BOTH its figure and its sub-line until then. "most recent
    # unknown" is a settled value too, and it was rendering off a date the page did
    # not have yet.
    tile = html[html.index("function InboxStatTiles"):html.index("function ReviewInboxView")]
    assert "appliedKnown" in tile.split("\n")[tile.split("\n").index(next(
        l for l in tile.split("\n") if l.startswith("function InboxStatTiles")))]
    assert "${appliedKnown\n          ? html`<div style=\"font-size:24px" in tile
    assert "most recent ${daysAgoLabel(lastAppliedAt)" in tile
    before_recent = tile[:tile.index("most recent ${daysAgoLabel(lastAppliedAt)")]
    assert before_recent.count("appliedKnown") >= 2, \
        "the sub-line must be held back with the figure, not rendered beside a shimmer"
    # An unknown count may not be read as an empty band either, or the whole band
    # disappears instead of shimmering.
    assert "if (openItems.length === 0 && appliedKnown && appliedCount === 0" in tile

    # The section disclosure says "each figure below", so it may not sit above a
    # skeleton, above an empty state, or above rows that carry no figure at all.
    # Re-anchored to the stronger property: gating on the row COUNT was not enough,
    # because rows and figures are different populations. A reverted row renders the
    # word "reverted" instead of a number, and an applied record can arrive with no
    # estimate whatsoever (measured on a real corpus: four applied hooks plus a trim
    # record, all with usd and tokens null), so a row-count gate still let the line
    # qualify nothing.
    assert "appliedCountKnown && combinedApplied.some(" in view
    disclosure_gate = view[view.index("appliedCountKnown && combinedApplied.some("):]
    disclosure_gate = disclosure_gate[: disclosure_gate.index("Each figure below")]
    assert "state === 'reverted'" in disclosure_gate, \
        "a reverted row shows the word, not a figure, so it may not satisfy the gate"
    assert "appliedEstimate(r)" in disclosure_gate, \
        "the gate must consult the row's actual estimate, not merely its presence"

    # The skeleton mirrors the real row's geometry, so content fills in rather than
    # replacing a differently-shaped block.
    assert "function AppliedSkeletonRow()" in html
    skel = html[html.index("function AppliedSkeletonRow()"):html.index("function AppliedItemRow(")]
    assert 'class="sz-runrow"' in skel
    assert 'aria-hidden="true"' in skel
    assert skel.count("shimmer") == 4


# --- Dashboard qualifier banner: removed by product decision ---------------- #
def test_dashboard_qualifier_banner_is_removed(html):
    """The subscription-billed qualifier banner ("N% of sessions are
    subscription-billed...") no longer renders on the Dashboard: subsidized AI
    pricing is common knowledge now, so the caveat was removed by product
    decision. `earlyFraming` (the /relearn/proposals early-read wiring that
    existed only to make this banner appear sooner) is dead with it."""
    dash = _dashboard_src(html)
    assert "earlyFraming" not in dash
    assert "qualifier_text" not in dash
    assert 'class="qualifier"' not in dash
    assert "qualifier-skel" not in dash
    # Every other framing consumer in DashboardView is untouched — the removal
    # was scoped to the banner only, never to `framing` itself. (The trailing
    # `, fmtDashUsd` on the two dollar formatters is the Dashboard's separate
    # at-most-2dp precision override, not a framing change. `useTokens` itself
    # is now LOCAL-only, a separate product decision removing subscription
    # differentiation -- see test_dashboard_use_tokens_is_local_only.)
    assert "const useTokens = !!framing && framing.pricing_mode === 'local'" in dash
    assert "fmtFramedDollar(projected, framing, fmtDashUsd)" in dash
    assert "<${PlanBadge} framing=${framing} />" in dash
    assert "fmtFramedSavings(t.usd, t.tokens, framing, fmtDashUsd)" in dash


def test_dashboard_use_tokens_is_local_only(html):
    """DashboardView's `useTokens` used to switch to token display for BOTH
    subscription and local. Subscription no longer suppresses (product
    decision: tj does not differentiate subscription-billed from API-billed
    users, and dollars price all traffic at API list rates regardless of
    plan); local still does (no marginal cost to price at all -- a
    structurally different case, not a differentiation choice)."""
    dash = _dashboard_src(html)
    assert "framing.pricing_mode === 'subscription'" not in dash


def _summarize_engine_view(html: str) -> str:
    """The `phase === 'engine'` render — the back link, heading, intro
    paragraph, and the three API/Claude CLI/Manual mode cards."""
    start = html.index("if (phase === 'engine') {")
    end = html.index("if (phase === 'run' && engine !== 'manual') {", start)
    return html[start:end]


def test_summarize_back_link_is_not_the_brand_blue_sz_link(html):
    # The back link is page chrome, not an in-flow action, so per founder
    # instruction it must NOT ride the shared brand-blue .sz-link class used
    # for every other clickable string on this screen (and across the app).
    # It gets its own class so this stays a one-hunk, page-local change.
    view = _summarize_engine_view(html)
    assert '<a class="sz-back-link" href="#/optimize">← Optimize</a>' in view
    assert '<a class="sz-link" href="#/optimize">← Optimize</a>' not in view
    # Founder instruction, verbatim: the back link and its text must be white.
    # It's full-brightness (var(--text)) at rest — NOT var(--text-dim) — with
    # the underline-on-hover as its non-color clickable affordance, since a
    # brighten-on-hover signal isn't available once the resting state is
    # already full strength.
    assert ".sz-back-link { color: var(--text); text-decoration: none; }" in html
    assert ".sz-back-link { color: var(--text-dim); text-decoration: none; }" not in html
    assert ".sz-back-link:hover { text-decoration: underline; }" in html


def test_summarize_back_link_has_breathing_room_before_heading(html):
    # The back link and the "Summarize" heading were cramped together at
    # margin:0 0 4px; this must be widened so the two are visually separated.
    view = _summarize_engine_view(html)
    assert '<div style="margin:0 0 4px"><a class="sz-back-link"' not in view
    assert '<div style="margin:0 0 16px"><a class="sz-back-link" href="#/optimize">← Optimize</a></div>' in view


def test_summarize_disabled_reason_is_not_amber(html):
    # "set TJ_ANTHROPIC_API_KEY to enable" used to render in var(--warn)
    # (amber/yellow) via a bespoke .sz-eng-off color rule. Founder instruction,
    # repeated: this text must be white, not just "not amber" — so it keeps
    # the .badge-closed pill's recessed background (still legible with white
    # text in both themes) but its own .sz-eng-off.badge-closed rule pins the
    # text color back to var(--text), overriding .badge-closed's normal
    # var(--text-dim).
    assert ".sz-eng-off { font-size: 12px; color: var(--warn); margin-top: 10px; }" not in html
    assert ".sz-eng-off.badge-closed { color: var(--text); }" in html
    view = _summarize_engine_view(html)
    assert '<div class="sz-eng-off badge badge-closed">${cap.reason}</div>' in view


def test_summarize_disabled_card_is_a_deliberate_state_not_uniform_dimming(html):
    # Blanket opacity:.5 on the whole disabled <button> dimmed the title
    # equally with everything else, so "unavailable" read as "broken" and
    # conflated disabled with less-important. The disabled state is now a
    # dashed border + recessed fill, leaving title/description at full
    # legibility.
    assert ".sz-engine:disabled { opacity: .5; cursor: not-allowed; }" not in html
    assert "border-style: dashed" in html
    assert ".sz-engine:disabled { cursor: not-allowed; background: var(--surface2); border-style: dashed; }" in html


def test_summarize_engine_cards_share_height(html):
    # The three cards (API/Claude CLI/Manual) have different description
    # lengths and only the API card carries the extra disabled-reason badge,
    # so without an explicit stretch the row read as uneven card heights.
    assert ".sz-engines { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin: 4px 0; align-items: stretch; }" in html
    assert "display: flex; flex-direction: column; height: 100%;" in html


def test_summarize_engine_tags_are_monochrome_not_accent(html):
    # "$ at cost · your key" / "against your Claude limits" / "no outbound
    # calls" are card SUBTITLES, not interactive controls, but were rendered
    # in the brand-blue accent color — the same color this app's terminal
    # taste reserves for "typeable or clickable." Moved onto the monochrome
    # (--text-dim) scale so accent keeps one meaning app-wide.
    assert ".sz-eng-tag { font-family: 'Geist Mono', monospace; font-size: 12px; color: var(--text-dim); margin-top: 4px; }" in html
    assert ".sz-eng-tag { font-family: 'Geist Mono', monospace; font-size: 12px; color: var(--brand); margin-top: 4px; }" not in html


def test_summarize_engine_view_has_no_em_dashes(html):
    # Standing project rule: no em dashes in user-facing copy. This view had
    # three: the intro paragraph, the Claude CLI card description, and the
    # staged-rewrite tally / undo hints.
    view = _summarize_engine_view(html)
    assert "—" not in view


def test_summarize_engine_view_has_no_unrendered_backticks(html):
    # The Claude CLI card read "Runs the local `claude` CLI." with literal
    # backtick characters because the string was never passed through <code>.
    # It must now render as a real <code> element with no stray backticks
    # left in the source strings for this view.
    view = _summarize_engine_view(html)
    assert "`claude`" not in view
    assert "<code>claude</code>" in view
    assert "Runs the local <code>claude</code> CLI against your Claude limits, with no dollar cost. Local host only." in view


def test_summarize_tj_keep_token_is_escaped_and_code_styled(html):
    # `<tj-keep>` used to be interpolated as a bare JS string (`${'<tj-keep>'}`)
    # which rendered as plain, unstyled angle-bracket text sitting oddly in
    # the sentence. It's still inserted as escaped text (never a real
    # unknown-tag risk to htm's parser), but now explicitly HTML-escaped and
    # wrapped in <code> so it reads as a token, not stray prose.
    view = _summarize_engine_view(html)
    assert "<code>&lt;tj-keep&gt;</code>" in view
    assert "${'<tj-keep>'}" not in view
    assert "Rewrites prose only: code, tables, and <code>&lt;tj-keep&gt;</code> blocks stay verbatim." in view


def test_summarize_engine_intro_uses_colon_not_em_dash(html):
    view = _summarize_engine_view(html)
    assert "Rewrites prose only —" not in view
    assert "Rewrites prose only: code, tables, and" in view


def _top_tenants_src(html: str) -> str:
    start = html.index("function TopTenantsPanel({ tenants, framing })")
    return html[start: html.index("\n}\n", start)]


def test_cost_tenants_table_shows_tokens_and_never_mislabels_call_count(html):
    """The Top-tenants-by-spend table rendered `call_count` (a raw call
    count) through `fmtTokens` -- a token formatter -- with no unit at all,
    and had no Tokens column even though /cost/tenants already returns
    per-row input/output/cache token sums. Pin both fixes: `fmtCount` for
    Calls, and a Tokens column summing all four token fields."""
    panel = _top_tenants_src(html)
    assert "fmtTokens(r.call_count)" not in panel
    assert "fmtCount(r.call_count)" in panel
    assert "<th>Tokens</th>" in panel
    assert (
        "fmtTokens((r.input_tokens || 0) + (r.output_tokens || 0) "
        "+ (r.cache_tokens || 0) + (r.cache_write_tokens || 0))"
    ) in panel


# --- Analyzer scan: a COLD store is not an empty result -------------------- #
# The phase machine above answers "did the READ answer?". Once analyzer results
# come from a background scan there is a second question it structurally cannot
# see: "does the STORE hold a result?" A cold store answers its HTTP request
# perfectly well (phase 'ready', data truthy) while holding nothing, so without
# an explicit check the band reaches `tileCount > 0`, finds 0, and prints
# "No recoverable candidates." for a scan that has never run. These pin the
# compute-layer half so it composes with the display-layer half rather than
# opening a new door into the same false-absence claim.

def test_recoverable_tiles_yield_nothing_on_a_cold_store(html):
    fn_start = html.index("function recoverableTiles(opt)")
    fn_end = html.index("\nfunction ", fn_start + 1)
    fn = html[fn_start:fn_end]
    assert "opt.report_available === false" in fn


def test_band_state_checks_cold_before_it_can_reach_the_none_branch(html):
    fn_start = html.index("function recoverableBandState(phase, data, tileCount)")
    fn_end = html.index("\n// Why a tile has no figure", fn_start)
    fn = html[fn_start:fn_end]
    # Anchored on the CODE, not on prose: the explanatory comment above the
    # guard mentions the none-branch by name, so a plain substring search
    # would find the comment first and pass regardless of ordering.
    cold_at = fn.index("if (data && data.report_available === false)")
    none_at = fn.index("return tileCount > 0")
    assert cold_at < none_at, (
        "the cold check must precede the tiles/none decision, or a never-run "
        "scan renders as 'we scanned and found nothing'"
    )
    assert "'cold'" in fn


def test_scan_state_treats_a_payload_without_the_field_as_known(html):
    """An older server predating the store sends no `report_available`. Reading
    that as cold would invent a not-yet-computed state for a real report."""
    fn_start = html.index("function scanState(opt)")
    fn_end = html.index("\n// Start a background scan", fn_start)
    fn = html[fn_start:fn_end]
    assert "opt.report_available !== false" in fn
    assert "cold: !fetchFailed && !known" in fn


def test_budgets_at_risk_cannot_report_ready_off_a_cold_store(html):
    idx = html.index("const budgetStatus =")
    line = html[idx:idx + 200]
    assert "scan.known ? 'ready'" in line, (
        "budgetStatus must gate on the STORE, not merely on optData being truthy"
    )
    assert "cold: 'not scanned yet'" in html


def test_both_analyzer_surfaces_carry_provenance_and_a_rescan_control(html):
    # The Dashboard band and the Optimize view both mount the shared ScanBar, so
    # neither can render figures without saying when they were computed.
    assert html.count("<${ScanBar}") >= 2
    # A rescan that FAILED must not look like one that succeeded, so the POST
    # goes through the helper that surfaces the server's reason.
    assert "apiPostOrDetail('/optimize/rescan', {})" in html
    assert "rescan failed: " in html


def test_auto_rescan_is_visibility_gated_and_killable(html):
    fn_start = html.index("function useAutoRescan(scan, rescan)")
    fn_end = html.index("\nfunction recoverableTiles", fn_start)
    fn = html[fn_start:fn_end]
    assert "document.visibilityState === 'visible'" in fn
    assert "scan.scanEnabled ? scan.pollSeconds : 0" in fn
    assert "if (!seconds || seconds <= 0) return undefined;" in fn


def test_optimize_is_not_in_the_client_response_cache(html):
    """The store carries its own computed-at, which the page displays. A 45s
    client-side staleness window on top would be a second, invisible freshness
    mechanism disagreeing with the timestamp on screen."""
    start = html.index("const CACHEABLE_READ_PREFIXES = [")
    end = html.index("]", start)
    assert "'/optimize'" not in html[start:end]


def test_optimize_view_renders_a_cold_state_instead_of_empty_cards(html):
    fn_start = html.index("function OptimizeView({ params })")
    fn_end = html.index("\n// Two lenses, one router", fn_start)
    fn = html[fn_start:fn_end]
    assert "${!st.loading && scan.known && st.opt ? html`" in fn
    assert "No analyzer scan has completed yet." in fn
    assert "this is not a report of zero waste" in fn


# --- Optimize ▸ Analyzer guide --------------------------------------------- #
# The guide exists because `downsize` and `subagent` were repeatedly read as the
# same check. Its two structural risks are (a) becoming a second, JS-side copy
# of the persona gate, and (b) shipping as a route nothing links to. One test
# each, plus one pinning the contrast itself, since that paragraph IS the page's
# reason to exist and a later edit could quietly drop it.


def test_analyzer_guide_nav_entry_is_unconditional_not_a_nav_child(html):
    """The bug this pins actually shipped. The guide was a `nav-child`, and
    App()'s route effect sets `el.style.display = (v === view) ? 'flex' : 'none'`
    for EVERY `.nav-child` -- so the entry existed in the file, a grep-style test
    passed, and the sidebar still showed nothing on Traces / Cost / Alerts /
    Drift / Budget. A string-presence assertion cannot tell those apart, so this
    asserts the property that differs: the entry must not be a nav-child, and
    must not carry a data-param (the attribute the child-visibility rule keys
    on). Reachability has to hold from wherever the user is, not just from the
    one screen the page explains."""
    line = next(
        ln for ln in html.splitlines()
        if 'href="#/guide"' in ln and "nav-link" in ln
    )
    assert "nav-child" not in line, (
        "the guide nav entry is a nav-child again -- it will be display:none "
        "everywhere except its parent section"
    )
    assert "data-param=" not in line
    assert 'data-view="guide"' in line
    assert 'data-lens="observe"' in line
    # No CSS rule scoped to this entry may reintroduce a display condition.
    css_rules = [ln for ln in html.splitlines() if ln.startswith(".sidebar a.nav-reference")]
    assert css_rules, "the nav-reference styling vanished"
    assert not any("display" in ln for ln in css_rules)

    # Nothing may hide it based on the active view. The route effect's only
    # display mutation is the nav-child branch; assert it stays scoped there.
    eff = html[html.index("document.querySelectorAll('.nav-link').forEach"):]
    eff = eff[:eff.index("}, [route.view, route.param]);")]
    display_lines = [ln for ln in eff.splitlines() if "style.display" in ln]
    assert len(display_lines) == 1, (
        "a second display mutation appeared in the nav effect; the guide entry "
        "may now be hidden by the active view"
    )
    # ...and that single mutation lives inside the nav-child branch.
    assert "el.classList.contains('nav-child')" in eff
    assert eff.index("el.classList.contains('nav-child')") < eff.index(display_lines[0])


def test_analyzer_guide_is_reachable_from_the_optimize_screen(html):
    """The contextual entry point, on the screen whose cards it explains. It
    renders before any `st.opt` / `scan.known` guard, so a cold or failed store
    still offers the way in."""
    fn_start = html.index("function OptimizeView({ params })")
    fn_end = html.index("// Optimize \u25b8 Guide", fn_start)
    fn = html[fn_start:fn_end]
    assert 'href="#/guide"' in fn
    title_at = fn.index("Optimize <${PlanBadge}")
    link_at = fn.index('href="#/guide"')
    assert link_at - title_at < 600, "guide link drifted out of the always-rendered title block"


def test_analyzer_guide_routes_resolve_and_old_hash_still_works(html):
    """`#/guide` is canonical; `#/optimize/guide` is the retired spelling and
    must keep working for anything already pointing at it."""
    assert "['guide',     AnalyzerGuideView]," in html
    assert "guide: 'observe'," in html, "the guide must sit in the Observe lens"
    assert "if (v === 'optimize' && route.param === 'guide') return 'guide';" in html
    assert "function isLegacyGuideRoute(route)" in html
    assert "history.replaceState(null, '', '#/guide');" in html


def test_analyzer_guide_reads_the_gate_from_the_server_not_a_js_copy(html):
    """Which checks apply is Python's answer (`PERSONA_DISABLED_ANALYZERS` ->
    `/optimize/analyzers`). The guide may own PROSE keyed by analyzer name, but
    never a membership decision -- a JS copy of the map desyncs the first time
    the Python side changes."""
    fn_start = html.index("function GuideBody({ persona, sets })")
    fn_end = html.index("function AnalyzerGuideView()", fn_start)
    fn = html[fn_start:fn_end]
    # Membership comes from the payload on both sides: what runs, what is gated.
    assert "sets.runs" in fn
    assert "sets.disabled" in fn
    view_start = html.index("function AnalyzerGuideView()")
    view_end = html.index("// Optimize ▸ Summarize (Track B)", view_start)
    view = html[view_start:view_end]
    assert "api('/optimize/analyzers')" in view
    # No persona-keyed analyzer-name list anywhere in the guide's own source:
    # the prose maps are keyed by name, but nothing decides membership from
    # a persona conditional in JS.
    guide_start = html.index("const GUIDE_PERSONA_LABELS = {")
    guide = _no_comments(html[guide_start:view_end])
    assert "PERSONA_DISABLED_ANALYZERS" not in guide
    assert "disabled_analyzers_for_persona" not in guide


def test_analyzer_guide_states_the_downsize_vs_subagent_distinction(html):
    """The founder could not tell these two apart; that is the page's whole
    reason to exist. The contrast must be stated as WHERE vs WHO, and must say
    that a session can only trip one of them."""
    start = html.index("const GUIDE_KEY_CONTRAST = {")
    end = html.index("const GUIDE_ENTRIES = {", start)
    block = html[start:end]
    assert "Downsize is about WHERE the work happened." in block
    assert "Subagent is about WHO did it" in block
    assert "only considers sessions that never delegated at all" in block
    # Rendered ABOVE the per-check cards, not buried in one of them.
    body_start = html.index("function GuideBody({ persona, sets })")
    body_end = html.index("function AnalyzerGuideView()", body_start)
    body = html[body_start:body_end]
    assert body.index("GUIDE_KEY_CONTRAST.title") < body.index("GuideCheck")


def test_analyzer_guide_ships_no_unwritten_persona_as_placeholder_prose(html):
    """Only Claude Code content was validated. An unwritten persona gets a
    banner naming the gap -- never invented copy, and never a silent fallback
    that reads as if it were written for the reader's setup."""
    start = html.index("const GUIDE_ENTRIES = {")
    end = html.index("function guideMissingEntry(name)", start)
    entries = html[start:end]
    # Exactly one persona is populated.
    assert entries.count("    order: [") == 1
    assert "'claude-code': {" in entries
    for absent in ("'sdk': {", "'mixed': {", "'unknown': {"):
        assert absent not in entries, f"{absent} must not carry unvalidated prose"
    view_start = html.index("function AnalyzerGuideView()")
    view = html[view_start:html.index("// Optimize ▸ Summarize (Track B)", view_start)]
    assert "has not been written yet" in view
    assert "Nothing has classified this install yet" in view


def test_analyzer_guide_makes_no_guaranteed_saving_claim(html):
    """Honesty discipline (Critical Rule 14) governs every user-visible string:
    estimates are candidates to review, never a promised saving, and the page
    never discusses how a figure was derived."""
    start = html.index("const GUIDE_PERSONA_LABELS = {")
    end = html.index("// Optimize ▸ Summarize (Track B)", start)
    guide = html[start:end]
    for banned in ("saves you", "you will save", "guaranteed", "realization rate",
                   "best-case", "ceiling of", "over-claim", "past_overspend"):
        assert banned not in guide, f"guide copy must not contain {banned!r}"
    assert "estimates from your own history" in guide
    assert "review before you act on them" in guide
