# Vendored front-end libraries

The local web UI (`tokenjam/ui/index.html`) is offline-first (CLAUDE.md Critical
Rule 18): every JS/CSS dependency is vendored under `tokenjam/ui/vendor/` and
served by the FastAPI `/ui/vendor` StaticFiles mount. No CDN loads at render
time. One line per vendored library below.

| Library | Version | Files | License | Notes |
|---|---|---|---|---|
| Preact | (as vendored) | `preact.js`, `preact-hooks.js` | MIT | ESM, via importmap |
| htm | (as vendored) | `htm.js` | Apache-2.0 | ESM, via importmap |
| uPlot | 1.6.32 | `uplot.js`, `uplot.css` | MIT | IIFE global `uPlot`, loaded via plain `<script>` (issue #112) |

## Bump procedure

1. Download the new release's `dist/` files from the upstream repo
   (uPlot: `dist/uPlot.iife.min.js` + `dist/uPlot.min.css`).
2. Replace the file under `tokenjam/ui/vendor/`, keeping the version-pin header
   comment at the top.
3. Update the version in the table above.
4. Run `pytest tests/unit/test_ui_offline.py` — it asserts no render-time
   external URLs and that the vendored files exist and ship in the wheel.
