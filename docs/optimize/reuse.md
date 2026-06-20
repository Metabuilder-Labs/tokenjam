# Reuse

Product name: **Reuse**. Internal/CLI name: `reuse`.

```bash
tj optimize reuse
```

Detects clusters of sessions that share a **planning skeleton** — the
structural shape of the agent's first planning output. When the same
"patch release" or "triage" workflow gets re-planned dozens of times,
each run re-buys a plan whose tool sequence and structure were identical;
only a version string or date changed. Reuse names that waste and exports
each skeleton as a reviewable template you can convert into a slash
command, a saved prompt, or a deterministic script.

Reuse never reuses a plan for you. The honesty constraint matches the
other analyzers: it reports "these plans look structurally identical" —
the user reviews the skeleton and decides whether it's safe to template.

## How it works

### The planning call

For each session, Reuse finds the **first LLM call that precedes any
tool call** — the model's opening "here's the plan" turn. Mechanically:
spans are ordered by `start_time`; the most recent LLM span before the
first tool span is the planner. Sessions with no tool calls use their
first LLM span; sessions with no LLM span are skipped.

### Tiered detection

Reuse clusters sessions by a signature, and the signature gets sharper
when more capture is available:

- **Mode 1 — tool-sequence signature (always on).** The ordered tuple of
  tool names following the planning call. Works against any telemetry,
  including raw Claude Code backfills with no capture toggles set.
- **Mode 2 — prompt-prefix hash (`[capture] prompts = true`).** Also
  hashes a variable-stripped prefix of the planning prompt (digits, ISO
  dates, and paths are normalized first, so "release v0.3.4 on 2026-06-15"
  and "release v0.3.5 on 2026-06-17" hash identically). The cluster key
  becomes the intersection of tool-sequence **and** prompt-prefix, which
  splits apart sessions that share a tool sequence but are unrelated tasks.

Mode 2 improves precision and recall without changing the output shape —
just the cluster quality.

### Cluster thresholds

A cluster is surfaced only when it clears all three (module-level
constants in `plan_reuse.py`, easy to tune):

- **≥ 3 sessions** share the signature in the window
- **≥ 200 average planning tokens** (tiny "ok, let me think" outputs aren't a plan)
- **≥ $0.01 cache-reuse recoverable** (already-cheap planning isn't worth surfacing)

Below-threshold clusters are dropped silently.

### Two recoverable numbers per cluster

Each cluster carries two framings, rendered side by side:

| Field | Heuristic | Meaning |
|---|---|---|
| **cache-reuse** | `avg_planning_cost × (repetitions − 1)` | Recoverable going forward by reusing the existing skeleton instead of re-planning. Conservative — you already paid once. |
| **script-replacement** | `avg_planning_cost × repetitions` | Upper bound — replacing every planning call with a deterministic template eliminates all of it. |

The aggregate `estimated_recoverable_usd` (what the Lens Overview tile
reads) uses the conservative cache-reuse number. All dollar figures flow
through `core/framing.py`, so subscription users see token-share framing
and local users see token counts instead of dollars.

## What you see

```text
  Reuse:
     • 1 cluster of repeated planning detected (tool-sequence only — enable
       capture.prompts for finer clustering)
       5× read_file → run_test → git_tag
          recoverable by reusing $0.80  · by scripting $1.00
     structurally repeated planning calls — cache-reuse number assumes future
     re-plans skip the LLM call entirely; review templates before reusing
     ! Structural skeleton match, not a guarantee the plans were
       interchangeable. Review the templates before reusing them.
```

Up to the top 5 clusters by recoverable amount are shown, each with both
numbers. `tj optimize --json` includes the full finding under
`findings["reuse"]` (every cluster field), and `/api/v1/optimize` returns
the same shape.

## Capture requirements

Reuse degrades gracefully — nothing is ever required to get *a* finding:

| Capture flag | Effect |
|---|---|
| *(none)* | Mode 1 runs: clusters from tool-sequence signatures only. A hint nudges you toward richer matching. |
| `[capture] prompts = true` | Mode 2 runs: prompt-prefix hashing narrows and sharpens clusters. |
| `[capture] completions = true` | Required to render the **skeleton text** in `tj report --reuse`. Without it, clusters and numbers still show; the skeleton is replaced by a one-line hint. |

## HTML + Markdown report

```bash
tj report --reuse                 # all agents, 30d window
tj report --reuse my-agent        # scope to one agent
tj report --reuse --since 7d      # custom window
tj report --reuse --no-open       # write files without opening browser
```

Writes to `~/.cache/tokenjam/reports/`:

- `reuse-<timestamp>.html` — a static, offline page (no JS, no CDN) with
  one section per cluster: the tool signature, both recoverable numbers,
  and the skeleton with variable regions highlighted as `{{slot_1}}`,
  `{{slot_2}}`, …
- `reuse-<cluster_id>.md` — one Markdown sidecar per cluster, directly
  copy-paste-usable as a slash command or saved prompt. The filename keys
  off the deterministic `cluster_id`, so re-running overwrites in place
  rather than piling up duplicates.

For Markdown only (no HTML, no browser), use the shortcut:

```bash
tj optimize reuse --export-templates
```

## Performance

A single windowed SQL query against `spans`, then an O(N) Python walk over
the result — no N+1 queries, no new indexes. Pure-Python clustering with
deterministic hashing: **no new dependencies, no learned matcher, no
embedding model** in v0.4. On a typical install (a few hundred thousand
spans, 30-day window) a run is well under a second.

## Confidence

`heuristic`. Reuse is structural detection only — it measures that
planning calls repeat, not that the plans were interchangeable. The
variable-slot highlighting is itself an honesty surface: when slots cover
most of the skeleton, you can see the "match" is mostly placeholders. The
caveat — *"Structural skeleton match … review the templates before
reusing them"* — appears on every user-visible surface (CLI, HTML,
Markdown header).

## See also

- [Downsize](downsize.md) — flag sessions whose shape matches a cheaper-model candidate
- [Cache](cache.md) — measure and improve prompt-cache usage
- [Script](script.md) — find workflows that look like deterministic shell scripts
- [Trim](trim.md) — identify low-significance tokens in captured prompts
