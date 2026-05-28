# Task 2.3 — Script analyzer (internal: workflow-restructure)

**Wave 2. Dispatch after Wave 1 fully merges. Runs in parallel with Tasks 2.1 and 2.2.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.

## Summary

Fourth of four optimize analyzers. Cluster sessions by tool-call signature; identify high-confidence deterministic patterns; recommend replacement with deterministic scripts.

User-facing product name: **Script**. Internal/CLI name: **`workflow-restructure`**.

## Signature definition (locked decision)

Signature = ordered list of `(tool_name, arg_shape)` tuples.

`arg_shape` describes which argument **types** are present, not values:

- `file_path` — argument resolves to a path (any string matching common path patterns)
- `command_string` — argument is a shell command or partial command
- `json_object` — argument is a JSON-shaped value
- `string` — generic string argument
- `number`, `boolean` — primitives
- `array` — list-typed argument

Two sessions with identical signatures may have very different argument **values**; that's intentional. The analyzer clusters by structural shape, not by content. This is what lets the v1 thresholds (below) find real signal without firing on file-path differences.

## Scope

- New analyzer module: `tokenjam/core/optimize/analyzers/workflow_restructure.py` with `@register("workflow-restructure")` decoration.
- Tool-call signature extraction from session traces using the locked definition.
- Clustering by exact signature match.
- Determinism scoring:
  - Argument-shape stability (always 100% when signatures match by definition — this is structural).
  - Branching presence (does the agent ever skip a step or insert a step?).
  - Outcome variation (does the user accept the result every time, or flag retries?).
- **High-confidence-only output in v1:** clusters with ≥20 instances, zero observed branching across instances, consistent outcomes (no flagged user retries).
- No automatic application; recommendation only.
- Confidence level: `structural` with explicit "review carefully" framing.
- Output includes: cluster's example sequence, instance count, savings projection, suggested script-replacement template (a stub shell script or pseudocode the user can adapt).

## Files touched

- New: `tokenjam/core/optimize/analyzers/workflow_restructure.py`
- New: `tests/unit/test_workflow_restructure.py` (use a deterministic-cluster fixture; cover the threshold logic and the signature-extraction edge cases)
- `docs/optimize/script.md` — **must document the signature definition explicitly with a worked example.**
- `CHANGELOG.md`

## Coordination

- Self-registers via `@register` decoration; auto-discovery picks it up. **Do not edit `analyzers/__init__.py` or `cmd_optimize.py`.**

## Done-when

- `tj optimize --finding workflow-restructure` on synthetic test data with a deterministic-cluster fixture identifies the cluster and reports the savings projection.
- Output includes example sequence, instance count, savings projection (plan-tier-aware: dollars for `api`, tokens for `subscription` / `unknown`), and a suggested script-replacement template.
- Sessions with branching are not flagged.
- Sessions with fewer than 20 instances are not flagged.
- `docs/optimize/script.md` includes a worked example showing how a session's tool calls map to a `(tool_name, arg_shape)` signature.
