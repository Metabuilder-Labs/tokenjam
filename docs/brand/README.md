# TokenJam brand assets

The TokenJam mark is a **weight matrix in brackets** — `[ W₁₁ W₁₂ / W₂₁ W₂₂ ]` — the tensor at the heart of every model. The wordmark is set in **DIN Alternate Bold** (a technical, squared geometric that echoes the matrix motif).

**Convention: black on white.** All brand assets are a single ink color (black) on a white/transparent ground, matching [tokenjam.dev](https://tokenjam.dev). Don't recolor the mark per-surface — keep it monochrome.

## Files

| Asset | Vector source | Raster |
|---|---|---|
| Mark / logo | `tokenjam-icon.svg` | `tokenjam-icon.png` |
| Wordmark | `tokenjam-wordmark.svg` | `tokenjam-wordmark.png` |
| Banner (mark + wordmark lockup) | `tokenjam-banner.svg` | `tokenjam-banner.png` |
| Repo header (logo + tagline + spend visual) | (none) | `tokenjam-repo-header.png` |

The README header uses `tokenjam-repo-header.png`. The SVGs are the editable sources — prefer them for any new surface and rasterize as needed.

## Regenerating the PNGs

The PNGs are rendered from the SVGs with [`rsvg-convert`](https://gitlab.gnome.org/GNOME/librsvg) on a white background, then cropped to content:

```bash
rsvg-convert --background-color=white -w 1600 -o tokenjam-banner.png tokenjam-banner.svg
```

The wordmark stroke is filled (solid). A hollow-outline variant — `fill="none" stroke="#000"` — matches the mark's stroke language and is available if a lighter treatment is ever wanted; the filled cut is the default for legibility at small sizes (PyPI, favicons, social cards).

## In-app mark

The dashboard ("TokenJam Lens") renders the same mark inline in `tokenjam/ui/index.html` (`<svg class="brand-mark">`) using `currentColor`, and as the inlined favicon. If you change the mark here, update those too.
