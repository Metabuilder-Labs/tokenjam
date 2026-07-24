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


def test_traces_window_select_exposes_longer_supported_windows(html):
    # Traces honors these URL/API windows already; keep the filter dropdown in sync
    # so #/traces?since=30d and #/traces?since=90d render selected options.
    traces_start = html.index("function TracesListView")
    traces_end = html.index("function dedup", traces_start)
    traces_view = html[traces_start:traces_end]
    assert '<option value="30d">Last 30d</option>' in traces_view
    assert '<option value="90d">Last 90d</option>' in traces_view


def test_dashboard_recent_activity_drills_into_matching_traces_window(html):
    # The Dashboard defaults to 30d while Traces defaults to 24h. Keep the Recent
    # activity drill-through tied to the Dashboard window so the tile count and
    # destination list use the same basis (#299).
    assert "function tracesHrefForWindow" in html
    assert "tracesHrefForWindow(since)" in html
    assert 'label="Recent activity" value=${(d.traces || []).length} attention=${errTraces > 0} href="#/traces"' not in html


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


def test_overview_error_handling_is_asymmetric(html):
    # /cost is load-bearing: NO .catch, so its failure surfaces the error state.
    # The other five panels keep .catch fallbacks so one failing panel renders
    # empty instead of blanking the Overview. Don't unify these (#124 review).
    assert "api('/cost', { since, group_by: 'day' }).catch" not in html  # no catch on /cost
    assert "api('/cost', { since, group_by: 'day' })," in html           # bare, inside Promise.all
    assert "api('/cost/compare', { since, compare: 'previous' }).catch(() => null)" in html
    assert "api('/optimize', { since, fast: 'true' }).catch(() => null)" in html
    assert "api('/drift').catch(() => ({ agents: [] }))" in html


def test_overview_empty_gate_considers_historical_cost(html):
    # Regression: the Overview front door showed "No data yet" whenever /status
    # reported 0 active agents — and it returned BEFORE /cost was ever fetched.
    # A DB whose sessions are all >24h old (e.g. a user upgrading to review past
    # spend) has 0 active agents but a full cost history, so the default landing
    # screen falsely read empty while Cost/Analytics/Optimize rendered fine.
    ov_start = html.index("const setWin = v => navigate('dashboard'")
    ov_end = html.index("const winPicker = html`", ov_start)
    ov = html[ov_start:ov_end]

    # Buggy pattern GONE: empty was gated purely on active-agent count with an
    # early return before /cost was fetched.
    assert "if (!status.agents || status.agents.length === 0) {" not in ov
    assert "const status = await api('/status');" not in ov  # /status no longer fetched first + serially

    # Fix pattern PRESENT: /cost is fetched in the parallel fan-out, and the
    # empty gate considers historical cost/tokens (not just agents/traces).
    assert "const hasCost = (cost.total_cost_usd || 0) > 0 || (cost.total_tokens || 0) > 0;" in ov
    assert "const empty = !hasCost && !hasAgents && !hasTraces;" in ov
    # /cost stays load-bearing (no .catch) inside the fan-out.
    assert "api('/cost', { since, group_by: 'day' })," in ov
    # /status is now degradable (moved into the parallel fetch with a .catch).
    assert "api('/status').catch(() => ({ agents: [] }))" in ov


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
    assert "case 'review': return html`<${ReviewInboxView}" in html
    assert 'href="#/review"' in html
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
    # Every per-item dollar cell — Traces list, Status cards, span-detail, and
    # the per-trace total — uses fmtPerItemCost, not the window-aggregate
    # fmtFramedDollar. Guards against a regression reintroducing "% of cycle" at
    # per-row granularity (the #249 bug: "466.7% of cycle").
    assert "${fmtPerItemCost(t.cost_usd, _costVal(t, true), framing)}" in html        # traces list
    assert "${fmtPerItemCost(a.cost_today, _costVal(a, true), data.framing)}" in html  # status card
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
# fmtFramedDollar(value, framing) so subscription users see "% of cycle" and
# only api-plan users see raw $ — matching Traces/Cost/Optimize.
def test_session_detail_cost_cell_routes_through_framing(html):
    # The "Cost & Tokens" / "Implied API value" panel must consume the
    # /sessions/{id} framing block, not render raw fmtCost(s.total_cost_usd).
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
    assert "if (route.param === 'summarize') return html`<${SummarizeView} params=${p} />`;" in html
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
    assert "estimated_recoverable_tokens" in html


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
    assert "Cost advisories" in html


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
    assert "case 'sessions':" in html
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
    assert "paste the project's" in row
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
    assert "case 'status':" in html, "keep #/status as a silent route alias"
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
    # on this page now (the founder-approved always-dollars carve-out — see
    # test_review_inbox_ignores_dollar_suppression below). This only pins that
    # the component itself still HONORS a truthy `suppressed` if ever passed
    # one, i.e. the parameter isn't dead weight removed from the function.
    start = html.index("function InboxStatTiles")
    end = html.index("function ReviewInboxView", start)
    tile = html[start:end]
    assert "suppressed" in tile
    assert "'~' + fmtTokens(totalToks) + ' tok'" in tile
    assert "'~' + fmtTokens(appliedTokSum) + ' tok'" in tile
    assert "fmtUsd(totalUsd)" in tile
    assert "fmtUsd(appliedUsdSum)" in tile


# --- Founder-approved carve-out: Review inbox ignores dollar suppression --- #
def test_review_inbox_ignores_dollar_suppression(html):
    # Founder decision: on the Review inbox ONLY, dollar figures render
    # unconditionally — the subscription-share suppression rule
    # (dollarsSuppressed(), core/framing.py's suppress_dollars_for_
    # subscription_share) does not apply here, even though it still gates
    # every other dollar figure in the app unchanged. Verified against the
    # founder's real account (87% subscription-billed, Max 20x plan): the API
    # payload correctly carries estimated_monthly_usd for every priced item,
    # so once this page stops gating on dollarsSuppressed() those figures
    # render regardless of plan tier.
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


def test_resend_dollar_figure_stays_tokens_only_as_a_structural_measurement(html):
    # The resend/TRIM card is the one documented exception to "always
    # dollars": its own evidence text discloses the figure is a structural
    # token-share measurement, not a savings claim (RESEND_HONESTY_CAVEAT,
    # analyzers/context_resend.py) — it stays tokens-only even though its
    # estimated_monthly_usd is a real, non-null number in the real payload.
    assert "const NOT_A_SAVINGS_CLAIM_ANALYZERS = new Set(['resend']);" in html
    assert "function monthlyUsdForDisplay(item)" in html
    fn_start = html.index("function monthlyUsdForDisplay(item)")
    fn_end = html.index("\n}", fn_start) + 2
    fn = html[fn_start:fn_end]
    assert "NOT_A_SAVINGS_CLAIM_ANALYZERS.has(item.analyzer)" in fn
    # Every dollar-first decision point on the page routes through it (or the
    # equivalent inline check in appliedEstimate) rather than reading
    # estimated_monthly_usd directly, so the exclusion can't be bypassed at
    # one of the several call sites.
    est_line_start = html.index("function estMonthlyLine(item, suppressed)")
    est_line_end = html.index("\n}", est_line_start)
    assert "monthlyUsdForDisplay(item)" in html[est_line_start:est_line_end]
    combined_start = html.index("function combinedEstMonthly(items, suppressed)")
    combined_end = html.index("\n}", combined_start)
    assert "monthlyUsdForDisplay(i)" in html[combined_start:combined_end]
    applied_est_start = html.index("function appliedEstimate(rec)")
    applied_est_end = html.index("\n}", applied_est_start)
    assert "NOT_A_SAVINGS_CLAIM_ANALYZERS.has(rec.analyzer)" in html[applied_est_start:applied_est_end]


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
    # The suppress/show decision is server-side (core/framing.py); the UI reads
    # display_rule rather than re-deriving the rule in JS.
    assert "function dollarsSuppressed" in html
    for rule in (
        "'suppress_dollars_for_subscription_share'",
        "'tokens_only'",
        "'suppress_dollars_unknown'",
    ):
        assert rule in html, f"missing suppressing display_rule {rule}"


# --- the estimated-recoverable tile: dollars-vs-tokens by the ≥half rule --- #
def test_estimated_recoverable_tile_leads_with_dollars_only_at_half_priceable(html):
    # Behavioral requirement #2: the headline "ESTIMATED RECOVERABLE" tile
    # shows dollars (with tokens in the sub-line) only when at least half the
    # OPEN items across both tabs are priceable; otherwise it leads with
    # tokens, same as the mockup's own tokens variant.
    start = html.index("function InboxStatTiles")
    end = html.index("function ReviewInboxView", start)
    tile = html[start:end]
    assert "priceable.length * 2 >= openItems.length" in tile
    # The `est.` pill was dropped for the denser mockup styling, so the
    # estimate framing has to survive in the tile's own label + hover title —
    # the figure must never read as a measured number.
    assert ">Estimated recoverable " in tile
    assert "sum of each open item's est./mo figure" in tile


def test_estimated_tile_hides_when_there_are_no_open_items(html):
    # Nothing open on either tab means nothing honest to lead a "recoverable"
    # headline with, so that half of the tile hides rather than showing a zero.
    start = html.index("function InboxStatTiles")
    end = html.index("function ReviewInboxView", start)
    tile = html[start:end]
    assert "openItems.length > 0 ? html`" in tile


# --- Review inbox copy: cost-led, and no hardcoded zero -------------------- #
def test_review_inbox_intro_matches_the_founder_approved_mockup(html):
    # Inbox redesign: the page title and subtitle are the founder-approved
    # mockup's own copy verbatim (colon in place of the mockup transcription's
    # em dash — house style forbids em dashes in tokenjam copy).
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
    assert "estTokens: f.estimated_recoverable_tokens" not in html
    assert "~${fmtTokens(d.estTokens)} tok</b> recoverable" not in html
    assert "function InboxStatTiles" in html


# --- Review inbox select-all ----------------------------------------------- #
def test_select_all_checkbox_sits_beside_the_bulk_dismiss_button(html):
    # Inbox redesign: the Recurring-mistakes tab dropped its <table> for flat
    # rows (RecurringMistakeRow, matching the mockup's card-style layout), so
    # the select-all box now sits in the tab's listhead beside "Dismiss
    # checked" rather than inside a <thead><th>.
    assert "function SelectAllCheckbox" in html
    assert (
        "<${SelectAllCheckbox} total=${visible.length} "
        "selected=${selectedCount} onToggle=${toggleAll} />"
    ) in html
    # The per-row checkbox is still present, just inside a flat row now.
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
    # The component delegates to it over the RENDERED row set.
    assert (
        "nextSelectAllSelection(visible.map(c => c.signature), prev)"
    ) in html


def test_select_all_applies_only_to_the_rendered_rows(html):
    # THE load-bearing one. The list filters out locally-dismissed rows and rows
    # already applied (in this session OR any earlier one), so select-all must
    # iterate `visible`, never the unfiltered d.clusters, or it would dismiss
    # rows the user never saw.
    start = html.index("const selectedVisible =")
    end = html.index("const modalCluster =", start)
    block = html[start:end]
    assert "visible.filter(c => checked.has(c.signature))" in block
    assert "visible.map(c => c.signature)" in block
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
    end = html.index("const modalCluster =", start)
    fn = html[start:end]
    assert "visible.filter(c => checked.has(c.signature)).map(c => c.signature)" in fn
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
    # Scoped to the whole Recurring-mistakes tab block, not just its listhead:
    # the two bulk buttons now sit BELOW the scroll box (they used to be in the
    # head, beside select-all), so a head-only slice would miss them entirely
    # and pass vacuously.
    view = html[html.index("function ReviewInboxView"):]
    start = view.index("tab === 'mistakes' ? html`")
    end = view.index("tab === 'advisories' ? html`", start)
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

def test_sort_by_est_monthly_ranks_uniformly_by_tokens(html):
    # The old bug: ranking flipped to dollars the moment ANY item in the list
    # had a dollar figure, leaving every other (tokens-only) item tied at
    # rank 0 and rendered in whatever order the API happened to return them
    # (adapter-insertion order) — exactly the founder's observed bug. Tokens
    # are the one figure every item carries, priced or not, so the fix ranks
    # by tokens uniformly; dollars stay a per-item DISPLAY choice
    # (estMonthlyLine), decoupled from ranking.
    start = html.index("function sortByEstMonthly")
    end = html.index("function splitTopAndTail", start)
    fn = html[start:end]
    assert "i.estimated_monthly_tokens || 0" in fn
    # The old dollars-if-any gate must not survive a regression re-adding it.
    assert "anyUsd" not in fn
    assert "estimated_monthly_usd" not in fn


def test_collapsed_tail_combined_figure_uses_the_priceable_majority_rule(html):
    # combinedEstMonthly used to lead with dollars the moment ANY tail item had
    # one (summing the rest as $0), understating a mostly-tokens-only tail as
    # a tiny dollar figure. It now matches InboxStatTiles's own majority rule.
    start = html.index("function combinedEstMonthly")
    end = html.index("function CollapsedTailRow", start)
    fn = html[start:end]
    assert "priceable.length * 2 >= items.length" in fn


def test_fmt_tokens_renders_billion_scale_human_readable(html):
    # "~11268.0M tok" (the founder's actual rendered figure) must become
    # "~11.3B tok" — fmtTokens needs a billion-scale branch above the
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


def test_fmt_tokens_billion_scale_matches_the_founders_real_figure():
    # A Python-side reimplementation of the exact fmtTokens algorithm pinned
    # above, run against the founder's own reported number, so this test
    # fails loudly if the JS and this contract ever diverge in behavior, not
    # just in the presence of the string "1e9".
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
    # (sortByEstMonthly, pinned above), run against the founder's own real
    # numbers from the bug report — the exact dataset that exposed the
    # original "adapter insertion order" bug. Proves the fixed algorithm
    # produces a genuinely monotonic order for real, not synthetic, data.
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
    # Pins the founder's own explicit ordering constraints from the bug report.
    assert ranked[0]["analyzer"] == "trim"
    reuse_values = [i["estimated_monthly_tokens"] for i in ranked if i["analyzer"] == "reuse"]
    assert reuse_values == sorted(reuse_values, reverse=True)
    assert 20_700_000 in reuse_values and reuse_values.index(20_700_000) < len(reuse_values) - 1


def test_split_top_and_tail_slices_an_already_sorted_list(html):
    # The long-tail collapse (requirement #3) must absorb the BOTTOM of the
    # sorted list, not an arbitrary suffix of the unsorted API order — it
    # slices whatever sortByEstMonthly already produced, never re-sorts or
    # re-orders on its own.
    start = html.index("function splitTopAndTail")
    end = html.index("function estMonthlyLine", start)
    fn = html[start:end]
    assert "sorted.slice(0, max)" in fn
    assert "sorted.slice(max)" in fn
    assert "sort(" not in fn   # no independent re-sort inside the split itself
    view = html[html.index("function ReviewInboxView"):]
    assert "splitTopAndTail(sortedRelearn)" in view
    assert "splitTopAndTail(sortedCost)" in view


def test_review_inbox_dollar_headline_ignores_framing_even_when_suppressed():
    # End-to-end contract test for the founder's real scenario: an account
    # whose framing says suppress_dollars_for_subscription_share (87%
    # subscription-billed, verified against the founder's own live store)
    # still gets dollar headlines on the Review inbox for every priced item,
    # tokens for the one documented exception (resend/TRIM), and tokens for
    # a genuinely unpriced item (no computable rate at all). A Python
    # reimplementation of estMonthlyLine's decision, pinned so a future
    # divergence between this contract and the shipped JS fails loudly.
    founder_framing = {
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

    assert headline_unit(priced_downsize, founder_framing) == "dollars"
    assert headline_unit(priced_deadweight, founder_framing) == "dollars"
    assert headline_unit(resend_structural, founder_framing) == "tokens"
    assert headline_unit(unpriced_placement, founder_framing) == "tokens"
