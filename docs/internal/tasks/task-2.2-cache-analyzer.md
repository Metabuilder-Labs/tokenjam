# Task 2.2 — Cache analyzer (two findings)

**Wave 2. Dispatch after Wave 1 fully merges. Runs in parallel with Tasks 2.1 and 2.3.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.

## Summary

Third of four optimize analyzers. Splits into two distinct findings (per code-review decision):

- **`cache-efficacy`** — measures current prompt-caching usage. No content capture needed.
- **`cache-recommend`** — suggests `cache_control` breakpoint placements. Anthropic-only in v1. Content capture needed.

Product name: **Cache**. Internal/CLI names: `cache-efficacy` and `cache-recommend`.

## Per-provider scope (locked decision)

### `cache-efficacy`

Reads `cache_read_input_tokens` and `cache_creation_input_tokens` from Claude Code JSONL or equivalent fields.

- **Anthropic: fully supported.** Numerically accurate per-call cache hit data.
- **OpenAI: best-effort.** OpenAI's caching is implicit and per-call cache hit data is not exposed reliably. Output language: "caching supported by provider but per-call data unavailable; aggregate efficacy estimation only."
- **Google Gemini: best-effort, model-dependent.** Document which models surface cache data in the per-provider table.
- **All others (Bedrock, LiteLLM, Cohere, etc.): unsupported in v1.** Show explicit "unsupported" message when invoked against these providers.

### `cache-recommend` (Anthropic-only in v1)

- Walks captured Anthropic prompts.
- Computes prefix hashes at common breakpoint positions (after system, after tools, after project context).
- Identifies stable prefixes; recommends specific `cache_control` placements with savings projection.
- **For non-Anthropic providers:** returns explicitly: "cache-recommend currently supports Anthropic only. Multi-provider boundary detection is a future feature." Do not silently produce zeros.
- Requires `capture.include_content: true`. Fails with clear message if not set.

## Scope

Both findings register as separate analyzer modules. Each gets its own `@register("...")` decoration.

- Savings projection uses cached-read rates from `pricing/models.toml`.
- Both findings: confidence level `structural`.
- Per-provider support documented in a table in `docs/optimize/cache.md`.

## Files touched

- New: `tokenjam/core/optimize/analyzers/cache_efficacy.py` (no content dependency)
- New: `tokenjam/core/optimize/analyzers/cache_recommend.py` (content-dependent, Anthropic-only)
- New: `tests/unit/test_cache_efficacy.py`
- New: `tests/unit/test_cache_recommend.py`
- `docs/optimize/cache.md` (includes per-provider support table)
- `CHANGELOG.md`

## Coordination

- Both analyzers self-register via `@register` decoration; auto-discovery picks them up. **Do not edit `analyzers/__init__.py` or `cmd_optimize.py`.**

## Done-when

- `tj optimize --finding cache-efficacy` runs for any user and reports current caching ratio.
- Output for Anthropic shows numerically accurate per-call data.
- Output for OpenAI/Gemini shows best-effort with explicit accuracy caveats.
- Output for Bedrock/LiteLLM/Cohere shows explicit "unsupported in v1" message.
- `tj optimize --finding cache-recommend` produces specific Anthropic breakpoint recommendations when content capture is enabled.
- `cache-recommend` against non-Anthropic providers returns the "Anthropic only" message.
- `cache-recommend` without `capture.include_content = true` fails with a clear message.
- Per-provider support table is in `docs/optimize/cache.md`.
