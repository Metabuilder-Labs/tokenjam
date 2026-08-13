---
description: The Lens single-file SPA — spacing tokens, charts, polling, UI testing — and the offline-first requirement (every dependency vendored, zero render-time external HTTP).
paths:
  - "tokenjam/ui/**"
  - "tests/unit/test_ui_offline.py"
  - "tokenjam/api/app.py"
---

# Web UI rules

### Critical Rule 18 — Web UI must work fully offline

`tokenjam/ui/index.html` is the served dashboard ("TokenJam Lens"; see the package notes below). It is
intentionally a single-file SPA with **zero external HTTP loads at render time**. Preact + hooks + htm
+ **uPlot** are vendored under `tokenjam/ui/vendor/` (ESM via `<script type="importmap">`; uPlot as a
plain `<script>` IIFE global); fonts use system-font fallbacks (no Google Fonts); the favicon is
inlined as a `data:` URL. The FastAPI app mounts `/ui/vendor` as `StaticFiles`. The
`tests/unit/test_ui_offline.py` regression test asserts no render-time external URLs exist anywhere
outside `<a href>` (clickable links to github.com are fine — they only fetch on click) and that
vendored CSS has no external `url()`. If you add a CDN font, script, or stylesheet, that test will
fail. Vendor the asset locally instead. See issue #87 + PR #88.

## `tokenjam/ui/` — Web UI ("TokenJam Lens")

`index.html` is the served dashboard — a **single-file Preact + htm SPA** (no build step, no
TypeScript, no client-side router). "TokenJam Lens" is the **brand only**: it appears in `<title>`,
the sidebar wordmark, and the OpenAPI title, but never in module names, route paths, or config keys.
Screens: **Overview** (the default landing route — a triage front door), Status, Traces, Cost,
Alerts, Drift, Optimize (with **Summarize** and **Rules** sub-views), Budget.

`app.py` reads `index.html` into a module string once at `create_app()` time, so editing it requires
a `tj serve` restart to take effect; tests read the file from disk directly and aren't affected.

- **Offline-first (Critical Rule 18, `.claude/rules/web-ui.md`):** every JS/CSS dep is vendored under `vendor/` — Preact + hooks + htm (ESM via `<script type="importmap">`) and **uPlot** (vendored IIFE global `uPlot` + CSS, pinned in `docs/internal/lens-vendor-versions.md`). No render-time external HTTP. `tests/unit/test_ui_offline.py` enforces this; clickable `<a href>` links are the only allowed external URLs.
- **Single compute path:** the UI reads everything from the REST API and **never re-implements analysis, aggregation, or plan-tier framing in JS** — it consumes the `framing` block (see `core/framing.py`). If the UI needs a number, extend the endpoint; don't compute it client-side. This applies to gating/filter sets too, not just numbers: if a Python-side map decides which analyzers/findings apply to a persona (e.g. `core/optimize/runner.py`'s `PERSONA_DISABLED_ANALYZERS`), the API already publishes the resolved set (`persona_disabled_analyzers` on `/optimize`) — read that field everywhere the gate is needed, never re-declare the map as a second JS literal. Two call sites deriving the same set from two different sources will silently desync the moment either side edits its copy.
- **A headline stat never shares a fetch with a slower panel.** The Dashboard's triage band fetches all of its endpoints in one `Promise.all`, which resolves only when its SLOWEST member does. The past-overspend hero therefore has its own effect reading `/relearn/cost-proposals` (a cheap lookup of an already-stored block, never a recompute), so it paints independently of the band and still renders when the triage fetch fails outright. Found live, not by inspection: the hero sat behind a loading shimmer while its own data had been on the client the whole time.
- **URL is the source of truth for filters:** state lives in the hash + query params (`#/cost?since=7d&group_by=model`); `getRoute()` parses it, `navigate()` writes it back omitting defaults. Window vocabulary matches the CLI (`1h`/`24h`/`7d`/`30d`/`90d` + `YYYY-MM-DD:YYYY-MM-DD`). The default landing route is Overview (empty hash → `getRoute()` returns `overview`; do **not** re-introduce a render-time `location.hash = ...` redirect — it raced the first render, issue #132).
- **Charts:** `SpendChart` wraps uPlot, reads CSS custom properties (`--chart-1..5`) so it re-themes, and has a cursor tooltip. The spend chart spans the **full selected window** with zero-fill: `/api/v1/cost` returns a window-bucketed `series` (hourly buckets for ≤2-day windows, daily otherwise; epoch-second `bucket` keys) plus `series_bucket` + `window_start`/`window_end`, and the UI builds a continuous grid + pins the x-scale to the window (issues #133/#136).
- **Run-rate** is a single linear figure projected to the end of the current calendar cycle (`daily_rate × days-remaining`), captioned "not a forecast". The forecasting boundary is deliberate: linear run-rate only — no EWMA, seasonality, or anomaly detection.
- **Polling:** the Overview auto-refreshes every 30s only while the tab is visible (`document.visibilityState`) and **fetches its endpoints in parallel** via `Promise.all` (the daemon DB layer is concurrency-safe since #124 — per-thread cursors). The error handling is deliberately asymmetric: `/cost` is load-bearing (no `.catch` — its failure surfaces the error state), while every other panel carries a `.catch` fallback so one failing panel renders empty rather than blanking the Overview. Don't unify them. Detail screens refresh on user action.
- **Spacing comes from the `:root` tokens, never from a hand-picked literal.** Every surface here is one of four shapes — a card/section body, a row inside one, a head strip, a table cell — and they share ONE horizontal inset (`--inset-x`) so a table lines up with the panel head above it; the vertical value varies by density (`--pad-card-y` / `--pad-row-y` / `--pad-head-y`), and `--stack-gap` separates stacked containers. Reach for the shared classes before writing a style attribute: `.panel-body` and `.panel-row` inset a panel's children (`.panel` is deliberately padding-free so it can hold a full-bleed table, which is exactly why its children kept being written with an ad-hoc padding, or none at all), `.opt-tile` is an `.opt-section` used as a flex column, and `.empty` + `.compact`/`.err` and `td.empty-cell` cover every not-here state. The failure mode is not ugliness, it is drift: an inline box and the loading skeleton that stands in for it are edited months apart and stop matching, and a re-declared `border`/`border-radius` in a style attribute hides that the class already owned it.
- **A cell with `overflow-wrap:anywhere` + `min-width:0` has a min-content width of ONE CHARACTER,** so under auto table-layout its column is the first thing the browser sacrifices and it collapses into a vertical stack of letters rather than wrapping. Any such column needs an explicit `min-width` floor. The mirror image is a `td` left at the default `white-space:nowrap` holding long text: that one column then dictates the whole table's width, and sibling tables on the same page come out visibly different widths with only one of them overflowing. Cap it and let it wrap; `.table-wrap` absorbs the remainder. Both bite on small, unrelated changes (a few px of extra cell padding was enough), so re-check wide tables after any spacing edit.
- **Testing the UI (no JS runner in CI):** the Python `test` job can't run JS. Guard UI work with tests that **execute** something — `test_lens_dashboard_states.py` and `test_lens_select_all_behaviour.py` extract a pure JS helper out of `index.html` and run it under `node`, and `test_ui_offline.py` asserts file-level invariants (no external hosts, no NUL bytes, and that the module script actually parses). **Do not add static-grep tests that assert on literal substrings of `index.html`.** A large test file of those was deleted: it pinned wording and markup structure rather than behaviour, so ordinary visual iteration broke assertions that guarded nothing, and one broke merely because an identifier was added to a `useEffect` dependency array it used as a slice anchor. If a UI behaviour is worth guarding, extract the logic into a pure function and execute it. When iterating locally, verify visually by running `tj serve` (or a seeded `create_app` + uvicorn on an alt port) and screenshotting with headless Chrome — there is intentionally no Playwright/Cypress.

### Critical Rule 43 — Webfont verification in CSS

A `font-family` name in CSS is not a loaded font; without an `@font-face` rule and a vendored file, the browser silently falls back and the stylesheet reads as valid. Verify a webfont with `document.fonts.check()` in a real browser, never by static CSS inspection.

### Critical Rule 44 — Never write an HTML entity into an `html` template literal

`htm` assigns text content; it does not parse entities. `&amp;`, `&rarr;`, `&#8594;` inside a
template render as those literal characters to the reader. Write the character itself (`&`, `→`),
or a JS escape (`→`). Grep `&[a-z]\+;\|&#[0-9]` inside `html\`` blocks before shipping copy.
Entities in the SERVER-RENDERED sidebar markup are fine: that is real HTML, parsed by the browser.

### Critical Rule 45 — Renaming UI copy: grep the SOURCE form, and run every UI-reading test

Tests assert against `ui/index.html` as text, so they match the source, not the rendered output.
Grepping `tests/` for `"Open ("` misses `assert "Open " in page` and the rename ships red. Grep the
bare words, then run the whole set that reads the file:
`grep -rln "index.html" tests/` — currently nine files, and they are cheap. Repair a broken guard by
repinning it on the new wording AND asserting the old wording is absent, so a revert fails too.
