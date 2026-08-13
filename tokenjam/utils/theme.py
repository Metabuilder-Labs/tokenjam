"""The tj terminal palette — one module, so a colour decision is a one-line edit.

The rules (modelled on Claude Code's own CLI, which reads as calm because it is
disciplined rather than colourful):

* **Ordinary prose carries no colour.** If a line is a sentence, it prints plain.
* **One accent, one meaning.** ``accent`` marks *a string the user can type or
  click* — a command, a path, a config key, a URL. Never decoration, never
  emphasis. If it isn't typeable, it isn't accented.
* **``muted`` (dim) for anything secondary** — explanations, provenance, details
  a user reads only when something looks wrong.
* **``label`` (bold) for structure** — headings and field labels. Structure is
  weight, not colour; the accent is reserved for content the user acts on.
* **Colour past the accent is reserved for genuine state, not emphasis.**
  ``warn`` is for a blocker the user must act on *now*; ``error`` for a real
  failure. Success is deliberately *not* a colour: it is a ``✓`` glyph plus
  bold, because a green line that merely says "this worked" spends a colour on
  the least surprising outcome in the flow.

Why this file exists: Rich's automatic highlighter is off on the shared console
(see :mod:`tokenjam.utils.formatting`). Left on — the default — it repaints
numbers cyan, paths magenta, brackets bold and quoted strings green *inside
prose*, so ``Max 20x plan`` rendered as ``Max 2`` + cyan ``0x`` and every
``~/.config/tj/config.toml`` came out two shades of pink. That accidental
palette, stacked on the deliberate one, is what put eight colours on a single
onboarding screen. With the highlighter off, every colour below is a choice
someone made on purpose.
"""

from __future__ import annotations

from rich.theme import Theme

# Soft periwinkle. Mid-tone on purpose: light enough to read on a dark
# background, dark enough to stay legible on a light one. Rich degrades it to
# the nearest 256-colour slot (~medium_purple) on terminals without truecolor,
# which lands close enough that the flow doesn't change character.
ACCENT = "#8b7fd4"

# Brand orange for the ASCII wordmark only (#643). The banner is a branded
# moment, not typeable content, so it earns its own colour rather than reusing
# the periwinkle accent — the accent stays reserved for commands/paths/URLs the
# user acts on. Rich degrades the truecolor value to the nearest 256-colour slot
# on terminals without truecolor.
BRAND_ORANGE = "#e0922f"

TJ_THEME = Theme(
    {
        # the one accent: things you can type or click
        "accent": ACCENT,
        "url": f"underline {ACCENT}",
        # the wordmark. Brand orange (#643) — the one place a screen carries a
        # dedicated brand colour, since the banner isn't typeable content.
        "brand": f"bold {BRAND_ORANGE}",
        # structure
        "label": "bold",
        "heading": "bold",
        "muted": "dim",
        # genuine state only
        "ok": "bold",  # success is bold + ✓, never green
        # The three completion `✓` status lines at the end of onboarding
        # (`--claude-code`). A deliberate FOUNDER OVERRIDE of the usual
        # "success is a glyph + bold, never green" rule (Anil, 2026-07): the
        # payoff screen leads with three green checks so the "you're wired up"
        # moment reads at a glance. Kept a distinct role from `ok` so the
        # override is scoped to that one screen and `ok` stays weight-only
        # everywhere else.
        "check": "bold green",
        # A CALL-TO-ACTION the user should do next (green = go). Distinct from
        # `ok` success: this is "do this now", not "this worked" — so it earns a
        # colour where success does not (Anil, 2026-07-30).
        "go": "green",
        "warn": "yellow",
        "warn.strong": "bold yellow",
        "error": "red",
        "error.strong": "bold red",
    }
)
