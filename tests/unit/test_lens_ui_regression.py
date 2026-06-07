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
    assert "by end of ${cyc.label}" in html
    assert "over 30 days" not in html  # the circular/undershooting framing is gone


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


# --- Work map: graphical "what did my agent do" tab ------------------------ #
def test_work_map_tab_present_and_default(html):
    # Map is the default session tab and renders before Timeline.
    assert "function WorkMapSection" in html
    assert "function WorkMapNode" in html
    assert "useState('map')" in html
    assert "/sessions/' + sessionId + '/workmap'" in html
    # Tab order: the Map button must appear before the Timeline button.
    map_btn = html.index("setTab('map')")
    story_btn = html.index("setTab('story')")
    assert map_btn < story_btn, "Map tab must render before Timeline"


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


def test_index_html_has_no_nul_bytes():
    # Guards the NUL-byte corruption fixed alongside the work map (it broke
    # `node --check` and made `file` mis-detect the SPA as binary).
    assert b"\x00" not in _UI.read_bytes()
