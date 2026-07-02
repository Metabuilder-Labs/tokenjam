# Map board — what this surface is for (direction note)

Status: working direction, 2026-07-02. Grew out of founder dogfooding on real long
sessions (meta tickets #56/#58): the original board was five raw telemetry lanes +
two mechanism controls, and a developer looking at it could not say what it was
telling them. This note pins the job so future changes have a test.

## The job

**"Where did this run's time and money go — and does that spending pattern look
healthy?"** One question. Every element must either answer it directly or be
evidence for it one hover away. Method-judgment ("was the approach good") is
explicitly NOT this surface — that's the Approach tab.

## The reading model (current, post-#56/#58)

1. **Insights strip answers first, in text.** Costliest active stretch (+ share of
   spend), friction (errors · true retries), top delegation by cost, idle share,
   edit footprint. If these look fine the user is done in seconds. Insight is
   never hover-gated (dataviz rule: tooltips enhance, never gate).
2. **Lanes are evidence.** Step mode is the DEFAULT read: evenly spaced by
   tool-call order, it shows the sequence without burst/idle distortion. Time
   mode (one click) localizes spend; cost bars share bucket edges with the tools
   histogram so a spike reads vertically.
3. **Treemap answers "did it converge".** Edit-weighted cards, workspace-relative
   dirs, scratch/temp reads demoted to one muted card.

## What was deliberately purged, and the principle

- **Manual bin-width ladder (Auto·30s/1m/5m/15m/1h)** — bin widths are renderer
  units, not user questions; a manual 1h on a ~40m session collapsed the board
  into one full-width slab. Width is auto-resolved and self-described in the
  cost peak label. *Principle: no control whose options the user cannot map to a
  question they have.*
- **Signature-repeat "retries"** — `is_retry` now requires the previous
  same-signature step to have FAILED. Consecutive successful edits of one file
  are normal work, and marking them saturated the retry encoding into noise
  (27 "retries" on a session with ~1 real one). *Principle: an alarm that is
  always on is off.*

## North star (not yet built)

Question-driven zoom instead of mechanism controls: click an insights chip (or
drag-select a region of the cost lane) → the board zooms to that active-time
window, bin width re-resolves automatically, tools/sub-agents/context re-scope.
The step⇄time toggle stays; everything else about navigation should be "point at
the thing that worries you."

Candidate follow-ups, in value order:
1. Chip → zoom (costliest-stretch chip is the obvious first).
2. Drag-select zoom on any lane, Esc to reset.
3. Phase segmentation quality (meta #57 — merge adjacent identical titles) so the
   phase lane reads as acts, not confetti.
4. Consider folding the tools histogram and cost bars into ONE lane (same buckets
   already) — height = cost, stack color = tool mix — reclaiming vertical space.

## Open questions (founder)

- Step-default: confirmed good after living with it a few days?
- Does the sub-agents lane earn its row on sessions with 0–2 delegations, or
  should it collapse into a chip until there are ≥3?
- Is the phase lane worth its height at all before #57 lands?
