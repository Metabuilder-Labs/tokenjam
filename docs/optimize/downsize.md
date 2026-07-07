# Downsize

Product name: **Downsize**. Internal/CLI name: `downsize`.

```bash
tj optimize downsize
```

Flags sessions whose structural shape — short input (< 5K tokens), short output (< 500 tokens), few tool calls (≤ 5) — matches a class of work where a cheaper model in the same provider family is worth reviewing.

The analyzer **does not** claim quality equivalence. It claims the *shape* of the work matches a class worth a closer look. The caveat is mandatory in every render mode:

> Candidate-flagging heuristic, not a quality judgment. Review the example sessions before changing models.

This caveat is baked into `DowngradeFinding` as a dataclass default (`MODEL_DOWNGRADE_CAVEAT`) so it can't be removed accidentally, and it surfaces in every CLI output mode (default human render, `--json`, MCP tool response).

## Downgrade candidates

The analyzer pairs premium models with cheaper alternatives in the same provider family. Pricing for both sides is resolved at runtime from `pricing/models.toml`; if either model is missing from the pricing table, the candidate is silently skipped — TokenJam will not invent a savings number.

| Provider | Premium → cheaper |
|---|---|
| Anthropic | `claude-opus-4-7` / `claude-opus-4-6` / `claude-sonnet-4-6` / `claude-sonnet-4-5` → `claude-haiku-4-5` |
| OpenAI | `gpt-4o` → `gpt-4o-mini`; `o3` → `o4-mini` |
| Google | `gemini-2-5-pro` → `gemini-2-5-flash` |

## Plan-tier-aware rendering

What the finding renders depends on `SessionRecord.pricing_mode` (see [`docs/architecture.md`](../architecture.md#otel-semconv-extensions-billing_account-and-plan_tier)):

- **`api`** — dollar-denominated. "Would have cost ~$X on the smaller model vs $Y actual" plus projected monthly savings.
- **`subscription`** (Claude Pro / Max, ChatGPT Plus / Team / Enterprise) — token-share framing. "Switching these N sessions to <cheaper model> would have used X% fewer tokens (~Y M tokens of your monthly allocation)". No dollar "spend" claim against a plan the user pays a flat fee for.
- **`local`** (Ollama) — token-only framing. No dollar figures.
- **`unknown`** — finding suppressed entirely; header note explains why and points to `tj onboard --reconfigure`.

JSON output mirrors the same data with top-level `plan` and `pricing_mode` fields plus a `monthly_tokens_freed` field for non-API plans.

## Confidence

`structural`. The downsize finding identifies a structural pattern in the captured data; it does not validate that the cheaper model would produce equivalent output. The mandatory caveat is the honest framing of that limitation.

## See also

- [Cache](cache.md) — measure and improve prompt-cache usage
- [Script](script.md) — find workflows that look like deterministic shell scripts
- [Trim](trim.md) — identify low-significance tokens in captured prompts
- [Subagent](subagent.md) — the same over-powered heuristic, scoped to subagents instead of whole sessions
