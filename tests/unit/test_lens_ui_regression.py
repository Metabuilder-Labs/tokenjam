"""Static regression guards for the Lens UI bug fixes (#126–#129).

The dashboard is a single-file Preact SPA with no JS test runner in the Python
CI job, so these assert the *served source* contains the corrected logic and no
longer contains the buggy patterns. They're intentionally narrow — each pins one
bug's fix so a future edit that reintroduces it fails here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_UI = Path(__file__).parent.parent.parent / "tokenjam" / "ui" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return _UI.read_text(encoding="utf-8")


# --- #126: Downsize typed slot always rendered ----------------------------- #
def test_downsize_section_always_renders(html):
    # The no-candidates branch renders a literal Downsize section id instead of
    # returning null, so the section is never silently dropped.
    assert 'id="opt-downsize"' in html
    assert "No downsize candidates in this window" in html


def test_downsize_is_first_in_optimize_order(html):
    assert "const order = ['downsize', 'cache', 'cache-recommend', 'script', 'trim']" in html


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


def test_overview_error_handling_is_asymmetric(html):
    # /cost is load-bearing: NO .catch, so its failure surfaces the error state.
    # The other five panels keep .catch fallbacks so one failing panel renders
    # empty instead of blanking the Overview. Don't unify these (#124 review).
    assert "api('/cost', { since, group_by: 'day' }).catch" not in html  # no catch on /cost
    assert "api('/cost', { since, group_by: 'day' })," in html           # bare, inside Promise.all
    assert "api('/cost/compare', { since, compare: 'previous' }).catch(() => null)" in html
    assert "api('/optimize', { since, fast: 'true' }).catch(() => null)" in html
    assert "api('/drift').catch(() => ({ agents: [] }))" in html


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
    assert "reuse:    { title: 'Reuse'" in html
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


def test_traces_list_cost_routes_through_framing(html):
    # Traces list COST column must consume the framing block, not raw fmtCost.
    # Per #249 it now goes through fmtPerItemCost (per-item → tokens for
    # subscription/local), not the window-aggregate fmtFramedDollar "% of cycle".
    assert "<td>${fmtCost(t.cost_usd)}</td>" not in html
    assert "${fmtPerItemCost(t.cost_usd, _costVal(t, true), framing)}" in html
    # The screen actually pulls the framing block off the /traces response.
    assert "setFraming(td.framing || null)" in html


def test_traces_list_surfaces_pagination(html):
    assert "total_count" in html
    assert "Showing ${traces.length} of ${totalCount} traces" in html
    assert "Load more" in html
    assert "load({ append: true })" in html
    assert "offset" in html


def test_trace_detail_costs_route_through_framing(html):
    # Waterfall bar label, tooltip line, and the span-detail panel all reframe —
    # no bare per-span fmtCost (the bar label + tooltip both used s.cost_usd).
    assert "fmtCost(s.cost_usd)" not in html
    assert "${fmtCost(sel.cost_usd)}" not in html
    assert "const costFramed = fmtFramedDollar(s.cost_usd, framing)" in html
    # The span-detail panel "Cost" is per-item → fmtPerItemCost (#249).
    assert "${fmtPerItemCost(sel.cost_usd, _costVal(sel, true), framing)}" in html
    # Trace detail pulls the framing block off the /traces/{id} response.
    assert "setFraming(d.framing || null)" in html


# --- #191: suppress raw $ on Status, Optimize & Reuse/script surfaces -------- #
def test_status_card_cost_today_routes_through_framing(html):
    # Status agent cards' "Cost today" must consume the /status framing block,
    # not render raw fmtCost(a.cost_today). Per #249 it's per-item → tokens for
    # subscription/local via fmtPerItemCost (not fmtFramedDollar "% of cycle").
    assert "${fmtCost(a.cost_today)}" not in html
    assert "${fmtPerItemCost(a.cost_today, _costVal(a, true), data.framing)}" in html


def test_optimize_window_comparison_routes_through_framing(html):
    # The window-comparison cost delta must reframe for subscription/local.
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
    # The script cluster table "Avg cost" cell is per-item, so per #260 it routes
    # through fmtPerItemCost (tokens for subscription/local), not the raw $ nor
    # the window-aggregate fmtFramedDollar "% of cycle".
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
    # Stacked chart respects plan-tier framing: subscription/local -> tokens.
    assert "fmtY=${fmtY}" in html  # fmtY = useTokens ? fmtTokens : fmtCost
    assert "const useTokens = !!framing && (framing.pricing_mode === 'subscription' || framing.pricing_mode === 'local')" in html


def test_cache_savings_chart_present(html):
    # #212: cache hit-rate + cumulative captured-vs-recoverable chart.
    assert "function CacheSavingsChart" in html
    assert "function buildCacheSeries" in html
    assert "${CacheSavingsChart}" in html
    # fetched from the dedicated endpoint
    assert "/cost/cache" in html


def test_cache_savings_honesty_framing(html):
    # #246 dropped the "estimated recoverable" overlay from this chart (noise;
    # it lives on Optimize). The cache card now reports MEASURED savings, framed:
    # api → "$X saved", subscription/local → cached-token VOLUME (never raw $).
    assert "fmtCost(cacheResp.total_captured_usd || 0)}</b> saved this window" in html
    assert "fmtTokens(cacheResp.total_captured_tokens || 0)}</b> cached reads this window" in html
    # the recoverable overlay is no longer wired into the cache chart
    assert "fmtFramedDollar(cacheResp.estimated_recoverable_usd" not in html


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
    # Per-analyzer recoverable must reframe (subscription/local → token-share),
    # mirroring the existing recoverable band — not raw fmtCost.
    assert "fmtFramedSavings(r.usd, r.tokens, compFraming)" in html
    # the measured-cost total uses the dollar framing helper, not raw fmtCost
    assert "fmtFramedDollar(st.comp.total_cost_usd" in html
    # plan-tier toggle drives tokens-vs-dollars for the whole surface
    assert "compFraming.pricing_mode === 'subscription' || compFraming.pricing_mode === 'local'" in html


# --- #210: Analytics pivot explorer (subsumes #214 leaderboard + #216) ----- #
def test_analytics_screen_registered(html):
    assert "function AnalyticsView" in html
    assert "case 'analytics': return html`<${AnalyticsView}" in html
    assert 'href="#/analytics"' in html  # sidebar nav link


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
    # the leaderboard preset closes #214; spend-by-model line closes #216
    assert "'leaderboard'" in html
    assert "'spend-by-model'" in html


def test_analytics_url_is_source_of_truth(html):
    # state read from URL params with validators, written back via navigate().
    # navigate() targets `route` (default 'analytics' preserves the standalone
    # screen; the dashboard preview passes route="dashboard").
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
    # spend metric switches to token volume for subscription/local (dollars
    # suppressed); never re-derives the suppression rule — reads framing.
    assert "framing.pricing_mode === 'subscription' || framing.pricing_mode === 'local'" in html
    # The Spend KPI tile reframes via spendTileDisplay (implied-value multiplier
    # for subscription, #262) rather than fmtFramedDollar's "% of cycle".
    assert "spendTileDisplay(kpis.spend, framing)" in html


def test_analytics_leaderboard_has_inline_bars(html):
    # #214: sorted leaderboard with inline magnitude bars (CSS, no chart lib).
    assert "function buildLeaderboard" in html
    assert "lb-fill" in html
    assert ".lb-bar" in html


# --- #215: cost-annotated trace waterfall ---------------------------------- #
def test_trace_waterfall_cost_summary(html):
    # A cost-first trace summary header (total cost + tokens + duration + spans).
    assert "wf-summary" in html
    assert "Total cost" in html
    assert "wfTotalCostFramed" in html
    # A single trace's total is per-item, not a window aggregate — per #249 it
    # routes through fmtPerItemCost (tokens for subscription/local), not the
    # window-level fmtFramedDollar "% of cycle".
    assert "const wfTotalCostFramed = fmtPerItemCost(wfTotalCost, wfTotalInOut, framing)" in html


def test_trace_waterfall_per_span_cost_token_annotation(html):
    # Per-span cost + tokens annotation column with a magnitude bar (not just the
    # hover tooltip), so the timeline reads cost-first.
    assert "wf-cost-bar" in html
    assert "wf-cost-fill" in html
    assert 'class="wf-cost-val"' in html
    assert 'class="wf-cost-tok"' in html
    # tokens summed per span and shown in the annotation
    assert "const spanTokens = s =>" in html
    assert "wf-cost-tok\">${sTok ? fmtTokens(sTok)" in html


def test_trace_waterfall_magnitude_respects_framing(html):
    # The magnitude bar (and summary) read on TOKEN volume when dollars are
    # suppressed (subscription/local) — the suppression decision comes from the
    # server framing block, never re-derived in JS.
    assert "framing.pricing_mode === 'subscription' || framing.pricing_mode === 'local'" in html
    assert "const wfMagOf = s => wfUseTokens ? spanTokens(s) : (s.cost_usd || 0)" in html


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


# --- #229: Overview tiles/headline deep-link into Analytics (pre-filtered) --- #
def test_overview_recoverable_tiles_deeplink_into_analytics(html):
    # Each recoverable-waste analyzer maps to an Analytics slice, and the tiles
    # render that deep-link via analyzerSliceHref (not the old Optimize finding
    # link). The route honors metric/group_by/chart/since, so these are just
    # well-formed Analytics URLs — no new state machine.
    assert "const ANALYZER_ANALYTICS_SLICE = {" in html
    for analyzer in ("downsize", "cache", "script", "reuse", "trim"):
        assert f"{analyzer}:" in html  # each analyzer has a slice mapping
    assert "function analyzerSliceHref(name, since, route = 'analytics')" in html
    # Dashboard tiles drill IN-PLACE (route='dashboard'), not to the Optimize screen.
    assert "analyzerSliceHref(t.name, since, 'dashboard')" in html
    assert "'#/optimize?finding=' + t.name" not in html



def test_analytics_deeplink_helper_exists_and_builds_hash_urls(html):
    # The deep-link helper builds #/analytics?... URLs (offline hash links, no
    # fetch) from a query object, dropping empty values.
    assert "function analyticsHref(q, route = 'analytics')" in html
    assert "return '#/' + route + (s ? '?' + s : '');" in html



def test_dashboard_is_default_landing(html):
    # Empty hash → dashboard, set via the PARSED default (no render-time
    # location.hash redirect — #132 discipline).
    assert "|| 'dashboard'" in html
    assert "|| 'overview'" not in html
    assert "location.hash = '#/dashboard'" not in html  # no hash-assign redirect
    assert "function DashboardView" in html
    assert "case 'dashboard': return html`<${DashboardView}" in html
    assert 'href="#/dashboard"' in html


def test_overview_retired(html):
    # Standalone Overview screen + nav item gone; lingering #/overview links fall
    # through to the Dashboard.
    assert "function OverviewView" not in html
    assert 'href="#/overview"' not in html
    assert "case 'overview':\n    case 'dashboard'" in html


def test_dashboard_embeds_analytics_explorer(html):
    # The hero composes the existing AnalyticsView (route rewired to dashboard,
    # embedded, with the run-rate caption) — not a reimplemented pivot. Standalone
    # #/analytics keeps working: the props are default-preserving.
    assert 'route="dashboard" embedded=${true} kpiCaption=${kpiCaption}' in html
    assert "function AnalyticsView({ params, route = 'analytics', embedded = false, kpiCaption = null })" in html
    # The full-screen explorer nav item stays.
    assert 'href="#/analytics"' in html


def test_dashboard_spend_deduped(html):
    # Spend shown ONCE (explorer's Spend KPI tile + chart); the old separate
    # run-rate headline chart is gone, folded into a caption under the KPI row.
    assert "const kpiCaption = (!d.loading && !d.empty && !d.error && projection" in html
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


def test_dashboard_triage_drills_in_place(html):
    # Recoverable-waste tiles update the embedded explorer via #/dashboard URL
    # state (not a jump to standalone #/analytics).
    assert "analyzerSliceHref(t.name, since, 'dashboard')" in html
    assert "function analyzerSliceHref(name, since, route = 'analytics')" in html
# --- #241: Status screen Cards | List view toggle -------------------------- #
def test_status_view_toggle_exists(html):
    # A segmented Cards | List control on the Status screen, driven by the URL
    # view param. List is the default (#263), omitted from the URL via navigate
    # defaults; Cards carries ?view=cards.
    assert ".view-toggle" in html  # segmented-control CSS
    assert "const DEFAULTS = { view: 'list', agent_id: '' };" in html
    assert "readParam(params, 'view', DEFAULTS.view, v => ['cards', 'list'].includes(v))" in html
    assert "const setView = (v) => navigate('status', { agent_id: agentId, view: v }, DEFAULTS);" in html
    # both toggle buttons present
    assert "onClick=${() => setView('cards')}>Cards</button>" in html
    assert "onClick=${() => setView('list')}>List</button>" in html


def test_status_list_is_default_view(html):
    # #263: List is the default scan view (not Cards).
    assert "const DEFAULTS = { view: 'list', agent_id: '' };" in html
    assert "const DEFAULTS = { view: 'cards', agent_id: '' };" not in html


def test_status_list_table_renderer_exists(html):
    # The List view is a dedicated columned-table renderer, gated behind the
    # view mode (now the default render path).
    assert "function StatusListTable({ agents, framing })" in html
    assert "? StatusListTable({ agents, framing: data.framing })" in html


def test_status_list_table_horizontally_scrolls(html):
    # #263: the list table sits in the scrolling .table-wrap AND carries a
    # min-width so it overflows (and scrolls) on narrow viewports instead of
    # compressing columns and clipping the actions cell.
    assert "<div class=\"table-wrap\"><table class=\"status-list\">" in html
    assert ".table-wrap { overflow-x: auto; }" in html
    assert ".table-wrap table.status-list { min-width: 760px; }" in html


def test_status_list_cost_column_respects_framing(html):
    # The List view's Cost cell goes through fmtPerItemCost exactly like the
    # cards — per-item cost renders as tokens for subscription/local (#249),
    # plan-tier framing is never re-derived in JS (#110/#241).
    assert "<td>${fmtPerItemCost(a.cost_today, _costVal(a, true), framing)}</td>" in html


# --- #249: "% of cycle" is window-level; per-item cost must render as tokens -- #
def test_per_item_cost_helper_renders_tokens_for_subscription_local(html):
    # The per-item formatter: subscription/local → token total (the in+out basis
    # via _costVal), api/unknown → dollars. "% of cycle" (a window aggregate) is
    # never produced at per-item granularity.
    assert "function perItemUsesTokens(framing)" in html
    assert "function fmtPerItemCost(costUsd, tokenTotal, framing)" in html
    assert "if (perItemUsesTokens(framing)) return fmtTokens(tokenTotal || 0) + ' tok';" in html
    # the only "% of cycle" string in the codebase lives in fmtFramedDollar, which
    # per-item surfaces no longer call directly for the row value.
    assert "return fmtFramedDollar(costUsd, framing); // api / unknown → dollars" in html


def test_per_item_cost_surfaces_use_the_helper_not_framed_dollar(html):
    # Every per-item dollar cell — Traces list, Status cards, StatusListTable,
    # span-detail, and the per-trace total — uses fmtPerItemCost, not the
    # window-aggregate fmtFramedDollar. Guards against a regression reintroducing
    # "% of cycle" at per-row granularity (the #249 bug: "466.7% of cycle").
    assert "${fmtPerItemCost(t.cost_usd, _costVal(t, true), framing)}" in html        # traces list
    assert "${fmtPerItemCost(a.cost_today, _costVal(a, true), data.framing)}" in html  # status card
    assert "<td>${fmtPerItemCost(a.cost_today, _costVal(a, true), framing)}</td>" in html  # status list
    assert "${fmtPerItemCost(sel.cost_usd, _costVal(sel, true), framing)}" in html     # span detail
    assert "fmtPerItemCost(wfTotalCost, wfTotalInOut, framing)" in html                # trace total
    # these per-item surfaces must NOT call fmtFramedDollar on the row value
    assert "${fmtFramedDollar(t.cost_usd, framing)}" not in html
    assert "${fmtFramedDollar(a.cost_today, data.framing)}" not in html
    assert "${fmtFramedDollar(sel.cost_usd, framing)}" not in html


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
    # Bars size by cost/token magnitude by default (the only thing that renders
    # on duration-less backfill), with a cost/tokens/duration toggle. Cost-first
    # default; tokens when $ is suppressed.
    assert "const [wfMode, setWfMode] = useState(null)" in html
    assert "const wfDefaultMode = wfUseTokens ? 'tokens' : 'cost'" in html
    assert "const magForMode = s =>" in html
    assert "setWfMode('cost')" in html
    assert "setWfMode('tokens')" in html
    assert "setWfMode('duration')" in html


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
    # Cost-first but plan-tier-safe: the per-span value still routes through the
    # server framing block, never a raw fmtCost (guards #187/#249 regressions).
    assert "const costFramed = fmtFramedDollar(s.cost_usd, framing)" in html
    assert "fmtCost(s.cost_usd)" not in html


# --- #246: cache-savings chart redesign (answer-first, single-axis bars) ---- #
def test_cache_chart_leads_with_answer_headline(html):
    # A plain headline: hit-rate stat + savings this window (not three overlaid
    # series). The card title is "Caching".
    assert '<div class="cache-headline">' in html
    assert "cacheSeries.hitRate.toFixed(0)}%</b> cache hit-rate" in html
    assert "saved this window" in html          # api framing
    assert "cached reads this window" in html    # subscription framing (no raw $)


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
    # multiplier == (% of cycle) / 100 == spend / plan_monthly_usd
    assert "(spendUsd || 0) / framing.plan_monthly_usd" in html
    # the tile no longer renders fmtFramedDollar (the "% of cycle") for spend
    assert "const spendVal = fmtFramedDollar(kpis.spend, framing);" not in html
    assert "const spend = spendTileDisplay(kpis.spend, framing);" in html


def test_analytics_count_tiles_have_thousand_separators(html):
    # Sessions / Events tiles are exact counts with separators ("23,954"), not
    # raw String() integers.
    assert "function fmtCount(n)" in html
    assert "toLocaleString('en-US')" in html
    assert "value: fmtCount(kpis.sessions)" in html
    assert "value: fmtCount(kpis.events)" in html
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
