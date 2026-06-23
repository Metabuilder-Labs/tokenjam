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


def test_overview_chart_is_not_a_click_target(html):
    # The cost hero must no longer be wrapped in an <a class="band-hero">; drill
    # is an explicit link.
    assert 'class="chart-card band-hero"' not in html
    assert 'class="drill-link"' in html
    assert "View Cost details" in html


# --- #178 / #188: chart x-axis tick timezone handling --------------------- #
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


def test_overview_caption_says_not_a_forecast(html):
    # Both screens carry the honesty qualifier now.
    assert html.count("(linear run-rate, not a forecast)") >= 2
    assert "(linear run-rate)<" not in html  # the bare Overview variant is gone


def test_axis_uses_compact_dollar_formatter(html):
    assert "function fmtAxisUsd" in html
    assert "axisFmtY=" in html


# --- #132: first-load lands on Overview (no redirect race) ----------------- #
def test_first_load_defaults_to_overview(html):
    # getRoute defaults to overview; the render-time hash redirect is gone.
    assert "|| 'overview'" in html
    assert "location.hash = '#/overview'" not in html
    assert "history.replaceState(null, '', '#/overview')" in html


# --- #133/#136: chart spans full window + consistent date labels ----------- #
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
    # Trim's not_ready content line read "— not ready"; the em-dash prefix is
    # dropped so the three states share a prefix-free scheme.
    assert "— not ready" not in html
    assert ">Not ready<" in html


def test_recoverable_tile_titles_share_one_weight(html):
    # All non-actionable tiles use the identical bare .rec-name title element,
    # so the at_ceiling (Cache) tile can't bold its title differently. The
    # positive emphasis lives only on the content line (.rec-amount.ok), which
    # is the intended #127 design and must stay.
    assert html.count('<div class="rec-name">${meta.title}</div>') >= 3
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
    assert "<td>${fmtCost(t.cost_usd)}</td>" not in html
    assert "${fmtFramedDollar(t.cost_usd, framing)}" in html
    # The screen actually pulls the framing block off the /traces response.
    assert "setFraming(td.framing || null)" in html


def test_trace_detail_costs_route_through_framing(html):
    # Waterfall bar label, tooltip line, and the span-detail panel all reframe —
    # no bare per-span fmtCost (the bar label + tooltip both used s.cost_usd).
    assert "fmtCost(s.cost_usd)" not in html
    assert "${fmtCost(sel.cost_usd)}" not in html
    assert "const costFramed = fmtFramedDollar(s.cost_usd, framing)" in html
    assert "${fmtFramedDollar(sel.cost_usd, framing)}" in html
    # Trace detail pulls the framing block off the /traces/{id} response.
    assert "setFraming(d.framing || null)" in html


# --- #191: suppress raw $ on Status, Optimize & Reuse/script surfaces -------- #
def test_status_card_cost_today_routes_through_framing(html):
    # Status agent cards' "Cost today" must consume the /status framing block,
    # not render raw fmtCost(a.cost_today).
    assert "${fmtCost(a.cost_today)}" not in html
    assert "${fmtFramedDollar(a.cost_today, data.framing)}" in html


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
    # The script/reuse cluster table "Avg cost" cell must reframe, not raw $.
    assert "${fmtCost(c.avg_cost_usd)}" not in html
    assert "${fmtFramedDollar(c.avg_cost_usd, framing)}" in html


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
    # Rule 14: "captured" is measured; the recoverable gap is "estimated", never
    # "saved". The caption must say estimated/recoverable and not claim "saved".
    assert "estimated recoverable" in html.lower()
    assert "not saved" in html.lower()
    # recoverable dollar figure routes through framing (subscription -> tokens)
    assert "fmtFramedDollar(cacheResp.estimated_recoverable_usd" in html


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
    # state read from URL params with validators, written back via navigate()
    assert "navigate('analytics'" in html
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
    assert "fmtFramedDollar(kpis.spend, framing)" in html


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
    # the total cost routes through the framing helper (not raw fmtCost)
    assert "const wfTotalCostFramed = fmtFramedDollar(wfTotalCost, framing)" in html


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
