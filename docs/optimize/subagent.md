# Subagent

Product name: **Subagent right-sizing**. Internal/CLI name: `subagent`.

```bash
tj optimize subagent
```

Claude Code spawns subagents (the Task tool), and their turns are stored under
the parent session. Folded into one parent total, a heavy research session's
subagent spend hides where the tokens actually went — on a real session tj
measured 66% of spend across ~147 subagents, invisible above the DB. This
analyzer breaks a window's cost down per subagent and flags two structural
right-sizing candidates.

## Claude Code only

The analyzer groups spans by `(session_id, sub_agent_id)`. `sub_agent_id` is
populated only by the Claude Code backfill path (derived from the on-disk
transcript's `agentId` / `isSidechain` fields) — spans from other runtimes
(Codex, the Python SDK, generic OTLP) carry `NULL` and are silently excluded.
See [agent-capability-matrix.md](../agent-capability-matrix.md) for the full
per-persona capability breakdown; subagent right-sizing has no row there today
because no other persona populates the column yet.

## What it flags

Each `(session, sub_agent_id)` group with `cost_usd >= $0.05` (a noise floor —
trivially small spend isn't worth a recommendation regardless of shape) is
checked against two independent structural criteria:

| Flag | Criteria |
|---|---|
| `over_powered` | Ran on a premium (Fable or Opus tier) model, full stop. Mirrors the [Downsize](downsize.md) heuristic, scoped to one subagent — but, unlike Downsize, does **not** also require small output or few tool calls. A Claude Code Task subagent is a full agent loop (100-400 LLM calls, cache-read in the hundreds of millions of tokens on a heavy session), not one dispatch/one answer, so its cost compounds with both output length AND tool-call count. Requiring either to stay small made the worst offenders (the ones that did the most work) the *least* eligible to be flagged; measured on a real corpus, only 6.5% of premium-tier subagent spend cleared both of the old clauses. |
| `over_provisioned` | Was handed a large context — input + cache-read tokens ≥ 50,000 — yet produced fewer than 2,000 output tokens. The prompt it was dispatched with is likely larger than the task needed. |

A single subagent can carry both flags. Thresholds live as module constants in
`tokenjam/core/optimize/analyzers/subagent_rightsizing.py` (`SMALL_OUTPUT_TOKENS`,
`CONTEXT_HEAVY_TOKENS`, `MIN_FLAG_COST_USD`).

It reads aggregate token counts only — no content capture required.

## Output

The finding reports, for the window:

- `total_subagents` / `sessions_with_subagents` — how many subagents ran and across how many sessions
- `subagent_cost_usd` / `subagent_tokens` and `percent_of_cost` — how much of the window's total cost ran inside subagents at all, before any flagging
- `flagged_cost_usd` — spend concentrated in the flagged (candidate) subagents
- `rows` — the top 25 subagents by cost (aggregates are computed over all subagents in the window; only the rendered/serialized list is capped)
- `flagged` — the top 25 flagged candidates by cost, each carrying its `flags` list

Rendering follows the same plan-tier-aware convention as the rest of `tj optimize`: `api` plans see the dollar share, subscription/local/unknown plans see the token share instead.

## Estimate basis / confidence

`past_overspend_usd` and `past_overspend_tokens` are quantified from two components, each with an independent "contributes nothing rather than invent a number" floor:

- **`over_powered`** — priced at `claude-sonnet-5` (one tier down from Opus/Fable/Mythos) over the exact same token mix the flagged subagent was billed for — a pure model-swap delta, no token-count change. This is a *narrower* step than [Downsize](downsize.md)'s own opus→haiku two-tier jump: Downsize's ladder is tuned for structurally tiny (Sonnet-shaped) whole sessions and can afford the aggressive drop, but this analyzer prices every premium-tier subagent, including full agent-loop ones, so it earns the more defensible one-tier-down target instead.
- **`over_provisioned`** — priced on the context *excess* over the subagent's own dispatch cohort's median (same calling agent + model, at least 5 like-shaped peers), at the cache-read rate (context arrives overwhelmingly as cache reads).

`estimate_confidence` is `"heuristic"` and `estimate_basis` reads:

> over_powered subagents (any premium-tier subagent above the noise floor) priced at claude-sonnet-5 (one tier down, not model_downgrade's two-tier opus-to-haiku jump) over the same tokens, a model-swap delta, structural fit only, plus over_provisioned subagents priced on their context excess over their dispatch cohort's own median (same calling agent + model), at the cache-read rate; no quality validation, review before re-dispatching. No guaranteed saving.

`confidence` on the finding itself is `structural` (Rule 14, honesty discipline) — the mandatory caveat, surfaced in every render mode:

> Candidate-flagging heuristic, not a quality judgment. Review the flagged subagents before changing how you dispatch them or which model they use.

As with [Downsize](downsize.md), the analyzer never claims the flagged subagent's task would have succeeded on a cheaper model or with a smaller prompt — only that its structural shape matches a class worth a closer look.

## See also

- [Downsize](downsize.md) — the same over-powered heuristic, scoped to whole sessions instead of subagents
- [Cache](cache.md) — measure and improve prompt-cache usage
- [Script](script.md) — find workflows that look like deterministic shell scripts
- [Trim](trim.md) — identify low-significance tokens in captured prompts
- [agent-capability-matrix.md](../agent-capability-matrix.md) — why this analyzer is Claude Code-only today
