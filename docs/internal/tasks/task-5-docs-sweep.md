# Task 5 — Documentation consistency sweep

**Wave 4. Dispatch after Waves 1, 2, and 3 fully merge. One agent. No code changes.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.
- Skim every PR merged in Waves 1–3 to understand what landed.

## Summary

Single-agent cross-cutting consistency pass after all feature work merges. Most analyzer/feature docs were written by individual task agents; this task is the consistency sweep across them.

## Scope

### Cross-cutting checks

- Verify every new analyzer doc references the four-product structure consistently. Product names: **Downsize / Trim / Cache / Script**. Internal names: `model-downgrade / prompt-bloat / cache-efficacy + cache-recommend / workflow-restructure`. Both should appear, with the product name in headings and the internal name in CLI flag examples.
- Confirm partnership-posture framing toward Langfuse / Helicone / Phoenix / LangSmith where applicable. We ingest from them; we don't displace them.
- Cross-link product docs to marketing product pages and vice versa.

### Specific docs to verify exist and are coherent

- `docs/configuration.md` documents the new `[capture]` config section and the precedence relationship with `alerts.include_captured_content`.
- `docs/backfill/overview.md` documents the unified `tj backfill <source>` pattern. Lists all three sources (claude-code, langfuse, helicone, otlp) that landed this sprint.
- `docs/optimize/cache.md` contains the per-provider support table (Anthropic full / OpenAI best-effort / Gemini best-effort / others unsupported).
- `docs/installation.md` documents the `tokenjam[bloat]` optional extra and what it pulls in (~2GB torch + transformers).
- `docs/optimize/trim.md` contains the benchmark numbers Task 2.1 was asked to record.
- `docs/optimize/script.md` documents the signature definition with a worked example.
- `docs/architecture.md` documents the OTel semconv extensions for `billing_account` and `plan_tier`.

### Updates

- Update `docs/roadmap` (or equivalent) to reflect current direction. No specific dates.
- Update top-level `docs/quickstart` to reflect the new positioning (cost optimization, four analyzers, plan-tier-aware framing).
- Add a "What's new" page or banner pointing to the v1.1 CHANGELOG entries.
- Verify `CHANGELOG.md` `## Unreleased` section reads coherently as a single release narrative. If not, reorder and lightly edit (without changing facts).

### Honesty audit

Walk every user-facing string committed this sprint and verify:

- No quality-equivalence claims for downgrade recommendations.
- Dollar figures presented appropriately by plan tier (no "spend" claims against subscription plans).
- "Structural heuristic" caveats present on every Level-1 recommendation.
- Internal vs product names used appropriately (product names in headings; internal names in CLI flags and code identifiers).

## Files touched

- `docs/**` (broad)
- `CHANGELOG.md` (light cleanup; do not change facts)

**No code changes.** If you find a code-level issue (e.g. a missing caveat in a user-facing CLI string), file a follow-up issue rather than fixing in this PR. Keep this task scope-locked to docs.

## Done-when

- Every checklist item above is verified.
- All cross-links between docs work (no broken markdown links).
- The `## Unreleased` CHANGELOG section reads as a coherent release narrative.
- The honesty audit produced no findings, or any findings are filed as follow-up issues.
