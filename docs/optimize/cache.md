# Cache

Product name: **Cache**. Internal/CLI names: `cache` and `cache-recommend`. Two related findings under the same product — both surface prompt-caching opportunities; they differ in what they need and what they recommend.

```bash
tj optimize cache
tj optimize cache-recommend
```

## `cache` — measure current caching

Reads aggregate `input_tokens` and `cache_tokens` from spans in the
window. Computes the share of input bytes served from cache per
`(provider, model)` and flags rows where:

- `input_tokens >= 100,000` over the window (otherwise savings are
  negligible regardless of the ratio), and
- `efficacy = cache_tokens / (input_tokens + cache_tokens) < 30%`.

No content capture required. Works on any data already in TokenJam's DB.

### Per-provider support

| Provider | Support | Notes |
|---|---|---|
| Anthropic | **Full** | `cache_read_tokens` + `cache_creation_tokens` populated by JSONL backfill, OTel patches, and the OTLP logs route. Numerically accurate. |
| OpenAI | Best-effort | OpenAI's prompt caching is implicit; per-call cache-hit data isn't exposed via the SDK consistently. Aggregate efficacy approximates the share of input that *would have* been cache-eligible. The renderer surfaces an explicit caveat. |
| Google Gemini | Best-effort | Some models (1.5 explicit, 2.5 implicit) report cache data, others don't. Confidence varies per model. |
| Bedrock, LiteLLM, Cohere, others | Unsupported in v1 | Surfaced with `support = unsupported`; never flagged. Cache primitives differ across these providers enough that a uniform recommendation would mislead. |

## `cache-recommend` — suggest breakpoint placements (Anthropic-only v1)

Walks captured Anthropic prompts in the window, hashes the first ~2000
characters of each, and flags prefixes shared by ≥3 calls. Each flagged
prefix becomes a candidate for a `cache_control` breakpoint.

Requires `[capture] prompts = true` in your config. Without captured
content the analyzer has no way to see prefix overlap and exits with a
clear hint.

Non-Anthropic spans in the window are counted (`skipped_provider_count`
in JSON output) but not analyzed. Multi-provider boundary detection is a
research project deferred beyond v1.

### Output

Each candidate carries:

- `occurrences` — how many calls share the prefix
- `avg_input_tokens` — average input size on calls that share the prefix
- `estimated_cacheable_tokens` — rough estimate of tokens that would
  become cache reads if the breakpoint were placed
- `sample_chars` — first 120 chars of the prefix for identification

## Confidence

Both findings carry `confidence: structural`. They detect a pattern in
the captured data; they don't validate that enabling caching produces
the same model output (Anthropic's cache is content-addressed, so the
output is identical, but other future providers may not be). For Wave 2
this is the right level.

## See also

- [Downsize](downsize.md) — flag sessions whose shape matches a cheaper-model candidate
- [Script](script.md) — find workflows that look like deterministic shell scripts
- [Trim](trim.md) — identify low-significance tokens in captured prompts
- [Subagent](subagent.md) — per-subagent cost breakdown and right-sizing candidates (Claude Code only)
