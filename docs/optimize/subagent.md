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
| `over_powered` | Ran on a premium (Opus-tier — model name contains `opus`) model, produced fewer than 2,000 output tokens, and made 5 or fewer tool calls. Mirrors the [Downsize](downsize.md) heuristic, scoped to one subagent. |
| `over_provisioned` | Was handed a large context — input + cache-read tokens ≥ 50,000 — yet produced fewer than 2,000 output tokens. The prompt it was dispatched with is likely larger than the task needed. |

A single subagent can carry both flags. Thresholds live as module constants in
`tokenjam/core/optimize/analyzers/subagent_rightsizing.py` (`SMALL_OUTPUT_TOKENS`,
`FEW_TOOL_CALLS`, `CONTEXT_HEAVY_TOKENS`, `MIN_FLAG_COST_USD`).

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

Candidate-only in v1 — `estimated_recoverable_usd` and `estimated_recoverable_tokens` are deliberately `None`; the analyzer surfaces the spend sitting in flagged subagents (`flagged_cost_usd`) rather than assert a guaranteed recovery. `estimate_confidence` is `"heuristic"` and `estimate_basis` reads:

> spend concentrated in structurally-flagged subagents (premium model with little output, or large context with little output); review before re-dispatching — no guaranteed saving

`confidence` on the finding itself is `structural` (Rule 14, honesty discipline) — the mandatory caveat, surfaced in every render mode:

> Candidate-flagging heuristic, not a quality judgment. Review the flagged subagents before changing how you dispatch them or which model they use.

As with [Downsize](downsize.md), the analyzer never claims the flagged subagent's task would have succeeded on a cheaper model or with a smaller prompt — only that its structural shape matches a class worth a closer look.

## See also

- [Downsize](downsize.md) — the same over-powered heuristic, scoped to whole sessions instead of subagents
- [Cache](cache.md) — measure and improve prompt-cache usage
- [Script](script.md) — find workflows that look like deterministic shell scripts
- [Trim](trim.md) — identify low-significance tokens in captured prompts
- [agent-capability-matrix.md](../agent-capability-matrix.md) — why this analyzer is Claude Code-only today
