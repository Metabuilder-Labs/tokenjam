# Issue #617 Maintainer Follow-up Implementation Plan

> **For Hermes:** Use the subagent-driven-development skill to implement this plan task-by-task, with a fresh review after each logical task.

**Goal:** Resolve the maintainer's actionable follow-up concerns without reopening unsafe read-time cost rollups or automatic agent-wide schema inference in PR #744.

**Architecture:** Keep PR #744 limited to the two safe helper callers already present. Handle session attribution at ingest/reconciliation time, with an explicit disjointness invariant and one canonical aggregate consumed by every reader. Keep schema enforcement explicit and opt-in; automatic inference remains removed unless the maintainer explicitly approves a separate design.

**Tech Stack:** Python, DuckDB, FastAPI, Click, pytest, Ruff, mypy, GitHub CLI.

---

## Current scope and decision gates

PR #744 already removes the following blocking paths, so they must not be reintroduced into that PR:

- trace-wide `session_token_cost_rollup` at a read boundary;
- raw row-level cost summation as a substitute for logical-call accounting;
- per-candidate archive rollup queries;
- inconsistent rollup behavior across status, session detail, CLI, MCP, and timeline readers;
- automatic `SchemaValidator` activation from an inferred agent-wide schema.

The following decisions must be confirmed in the issue before feature code is written:

1. **Trace attribution policy. Recommended:** explicit `session_id` wins; then conversation identity; then a `parent_span_id` chain that resolves to one session. Do not let a bare shared trace make multiple sessions claim the same cost. If a trace-only fallback is retained for the single-marker case, persist that it was derived so a later ambiguous marker can revoke and reconcile it. Unresolved or ambiguous spans remain visible as unattributed spans rather than being copied into multiple sessions.
2. **Schema inference policy. Recommended:** do not ship automatic inference. Keep validation limited to explicitly declared schema files and preserve the existing nullable storage field/column only as inert compatibility state. If the maintainer wants inference, implement it in a separate, explicitly opt-in follow-up using the design below.

Do not force-push, commit, or update PR #744 as part of this plan without explicit authorization.

---

## Track 0: Confirm scope and freeze the safe PR

### Task 1: Record the current PR baseline

**Objective:** Establish an auditable starting point before follow-up work begins.

**Files:** None.

**Steps:**

1. Verify that the worktree is clean and that the local branch and fork branch for PR #744 point to the same head.
2. Record the PR head, commit count, review decision, and CI state in the issue discussion if an issue update is authorized.
3. State that the blocking rollup and inference findings were removed from #744, not redesigned there.

**Verification:** `git status --short --branch`, `git ls-remote fork refs/heads/fix/617-wire-tested-helpers`, `gh pr view 744 --repo Metabuilder-Labs/tokenjam`, and `gh pr checks 744 --repo Metabuilder-Labs/tokenjam`.

### Task 2: Post the two design questions for maintainer confirmation

**Objective:** Avoid implementing an attribution or enforcement policy that the maintainer has not accepted.

**Files:** GitHub issue #617, only if an external comment is authorized.

**Steps:**

1. Ask whether ambiguous shared-trace spans should be left unattributed or assigned by a parent-span rule.
2. Propose the recommended parent-first, disjoint attribution policy above.
3. Ask whether automatic schema inference should be abandoned permanently or redesigned as an explicit opt-in feature.
4. Do not begin the corresponding production implementation until the answers are recorded.

**Verification:** The issue contains an unambiguous accepted policy for each decision.

---

## Track 1: Safe trace/session cost attribution

This track is a separate follow-up feature. It must preserve the invariant that one logical billed call contributes to at most one session total and that session totals never exceed the deduplicated logical-call total.

### Task 3: Add failing shared-trace regression tests

**Objective:** Exercise the real ingest path rather than a hand-seeded database state.

**Files:**
- Modify: `tests/synthetic/test_ingest.py`
- Modify: `tests/unit/test_ingest_accounting.py` if duplicate-source coverage belongs there

**Steps:**

1. Add a real-pipeline case with two session markers (`w1`, `w2`) on one trace and two cost-bearing spans that carry only the shared trace identity.
2. Assert that no session claims the same unattributed cost span twice and that the session figures sum to the canonical real spend.
3. Add a reverse-arrival case where the cost span arrives before the marker; assert that no permanent orphan session is created solely because of arrival order, or encode the accepted provisional-state behavior if the maintainer selects that option.
4. Add a parent-child case proving that a cost span can be attributed through a unique parent chain without using whole-trace membership.
5. Add an ambiguous-parent/shared-trace case proving that the span is not copied into both sessions.
6. Run each new test before production changes and record the expected failure for the selected policy.

**Verification:** The tests fail for the old resolver behavior and use `IngestPipeline.process()` rather than direct `db.insert_span()` for the attribution scenarios.

### Task 4: Replace first-row trace lookup with an explicit attribution query

**Objective:** Make the database API expose enough information to apply disjoint attribution.

**Files:**
- Modify: `tokenjam/core/db.py:80-116, 3496-3501`
- Modify: `tests/integration/test_storage_backend_parity.py`
- Modify: `tests/unit/test_db_helpers.py`

**Steps:**

1. Replace the `LIMIT 1` semantics of `get_session_id_for_trace()` with a query that returns distinct session identities or add a new `get_session_ids_for_trace()` method and make callers use it.
2. Ensure the query distinguishes explicit session-bearing marker spans from an arbitrary first span.
3. Add the parent-chain lookup required by the accepted policy, with cycle protection and a bounded traversal.
4. Keep all values parameterized and preserve the DuckDB single-writer rules.
5. Add parity coverage for production and in-memory/API-backed implementations.
6. Keep a compatibility wrapper only if existing callers require it; the wrapper must return a value only when the result is unambiguous.

**Verification:** A trace with two session IDs never returns one arbitrarily, a trace with one session ID resolves deterministically, and a missing/cyclic parent chain degrades to no attribution.

### Task 5: Implement ingest-time resolution and reverse-arrival handling

**Objective:** Stop using an arbitrary session for ambiguous traces and prevent arrival order from creating duplicate session ownership.

**Files:**
- Modify: `tokenjam/core/ingest.py:245-330, 488-594`
- Modify: `tokenjam/core/models.py:60-137` only if attribution provenance must be represented explicitly
- Modify: `tokenjam/core/db.py` only if a bounded pending/reassignment operation is needed
- Test: `tests/synthetic/test_ingest.py`

**Steps:**

1. Preserve explicit `session_id` and conversation-based resolution.
2. Resolve by parent ancestry before considering a trace fallback.
3. Apply the approved singleton-trace rule only when the trace has exactly one eligible marker; never attach an unattributed span to multiple sessions.
4. For unresolved trace-only spans, use the approved durable pending/provisional behavior rather than immediately minting an irreversible orphan session.
5. When a marker arrives later, reassign only spans that are eligible under the approved rule, inside one write-locked operation.
6. Reconcile affected session counters after reassignment; do not increment totals once per reassignment and again through a later read-time rollup.
7. Make `_build_or_update_session()` and session-end hooks safe when a span remains unattributed.
8. Preserve the existing behavior for genuinely standalone spans if the accepted design allows it; otherwise document the compatibility change and update the existing standalone-span test.

**Verification:** The new ingest tests pass, two sessions on one trace remain disjoint, reverse arrival is deterministic, and no session-end alert/drift hook runs for a span that has no session.

### Task 6: Make logical-call reconciliation duplicate-aware

**Objective:** Ensure historical repair and reassignment do not turn duplicate live/backfill observations into money twice.

**Files:**
- Modify: `tokenjam/core/db.py:1185-1354, 3069-3107`
- Inspect/use: `tokenjam/core/optimize/accounting.py`
- Modify: `tests/unit/test_ingest_accounting.py`
- Modify: `tests/unit/test_session_cost_drift.py`

**Steps:**

1. Define the canonical logical-call identity and source precedence using the existing accounting helpers.
2. Ensure any reconciliation path selects one billed observation per logical call before computing session totals.
3. Do not use an unconditional raw `SUM(cost_usd)` over all source rows for mixed live/backfill history.
4. Reuse `recompute_session_totals_from_spans()` only for data that has already been canonicalized, or extend it to perform the same duplicate-aware selection.
5. Reconcile all touched sessions with one grouped query rather than one query per session/candidate.
6. Add tests for same-source repeated calls, cross-source duplicate observations, mixed models, zero-token rows, and repriced spans.

**Verification:** `session_cost_drift()` is zero after repair, duplicate observations contribute once, repeated repair is idempotent, and the total is independent of ingest order.

### Task 7: Add an indexed, bounded reassignment/reconciliation operation

**Objective:** Keep the new write-time path safe on large stores.

**Files:**
- Modify: `tokenjam/core/db.py`
- Modify: migration definitions in `tokenjam/core/db.py` by appending only a new migration if an index is required
- Test: `tests/unit/test_db_helpers.py`
- Test: `tests/integration/test_storage_backend_parity.py`

**Steps:**

1. Add an index on `spans(session_id)` only if query plans show it is needed; do not modify existing migrations.
2. Batch trace/session IDs and span IDs under the repository's existing bind-parameter limits.
3. Perform reassignment and aggregate refresh under the existing write lock.
4. Verify that a failed reconciliation leaves no partially reassigned session totals, or document the accepted undercount/retry behavior.
5. Add an idempotence test for rerunning the operation.

**Verification:** The operation remains bounded for a large synthetic trace set and does not create N+1 full-table scans.

### Task 8: Pin reader parity to the canonical session values

**Objective:** Prevent the reader disagreement called out by the maintainer.

**Files:**
- Modify: `tokenjam/api/routes/status.py:124-251, 497-504`
- Modify: `tokenjam/api/routes/sessions.py:89-177, 490-555, session-detail path`
- Modify: `tokenjam/cli/cmd_status.py:75-128`
- Modify: `tokenjam/mcp/server.py:397-523, 602-615`
- Modify: `tokenjam/core/session_timeline.py:108-176`
- Modify: `tokenjam/core/api_backend.py` if the API shim needs a new backend method
- Test: `tests/integration/test_api.py`
- Test: a new focused reader-parity test if the existing API fixture cannot cover all surfaces

**Steps:**

1. Make every session-facing surface read the same canonical `SessionRecord` aggregates or one shared backend method.
2. Do not reintroduce a trace-wide rollup at any read boundary.
3. Keep agent/window totals and per-session totals clearly scoped; do not place span-wide totals beside session totals under the same label.
4. Add one fixture containing shared traces, duplicate-source observations, and unattributed spans.
5. Assert that status, session list/detail, CLI, MCP, and timeline either agree exactly or label their intentionally different scopes.

**Verification:** The same session ID has the same cost and four-bucket token total on every surface, and aggregate totals do not exceed canonical logical spend.

### Task 9: Guard archive performance and error behavior

**Objective:** Ensure archive rendering cannot regress into the rejected OOM/N+1 design.

**Files:**
- Modify: `tokenjam/api/routes/status.py:210-251`
- Test: `tests/integration/test_api.py`
- Test: focused status/archive tests under `tests/unit/` if a dedicated file is created

**Steps:**

1. Keep `_count_archived()` as a count over the same session-column population that `_build_archive()` actually filters; remove any claim that it mirrors a removed span rollup.
2. If future attribution data is needed for archive filtering, obtain it with one grouped query for the candidate set, never one query per candidate.
3. Add a scale fixture or benchmark for the maintainer's 500/20k and 5,000/200k shapes without making the largest corpus a normal fast unit test.
4. Add a targeted DuckDB-error fallback around the new grouped operation, following `_build_sdk_services()`'s honest degradation pattern without a broad exception that hides programming errors.
5. Assert that `archived_total` and the returned archive rows use the same zero-signal predicate.

**Verification:** The normal status route performs a bounded number of queries, the scale benchmark does not approach the reported OOM behavior, and a controlled storage failure returns a documented degraded response instead of a 500.

---

## Track 2: Keep schema enforcement explicit, or redesign it only if approved

### Task 10: Remove stale inference claims from documentation and tests

**Objective:** Make the shipped behavior match the current code after inference was removed from PR #744.

**Files:**
- Modify: `docs/architecture.md:73-87, 205-208`
- Modify: `tests/synthetic/test_schema_validation.py`
- Inspect/update: `tokenjam/core/schema_validator.py:24-108`

**Steps:**

1. State that schema validation runs only for captured tool output when an explicit agent schema file is configured.
2. Remove the architecture section claiming that `genson` infers and persists schemas.
3. Reword the no-schema test to assert that captured output is ignored without an explicit declaration.
4. Add a pipeline-level test with `CaptureConfig(tool_outputs=False)` proving that output is stripped before validation and no validation/alert is produced.
5. Preserve the current default-off capture behavior and the existing sensitive-detail stripping rules.

**Verification:** Documentation, validator docstrings, and tests all describe explicit-only validation; capture-off behavior is covered through `IngestPipeline`, not by manually injecting an attribute into a validator-only fixture.

### Task 11: Preserve compatibility for the retired inferred-schema storage field

**Objective:** Avoid breaking existing DuckDB files while ensuring the retired field cannot activate enforcement.

**Files:**
- Inspect/modify: `tokenjam/core/models.py:342-356`
- Inspect/modify: `tokenjam/core/db.py:521-535, 3206-3232, 3718-3747`
- Test: `tests/unit/test_db_helpers.py`
- Test: `tests/synthetic/test_schema_validation.py`

**Steps:**

1. Keep the nullable `drift_baselines.output_schema_inferred` column inert for the first cleanup pass so old databases remain readable.
2. Ensure `SchemaValidator._get_schema()` never falls back to that field.
3. Add a compatibility test showing an old baseline row can still be read while validation remains skipped without an explicit schema.
4. Do not add a migration to drop the column unless the maintainer requests removal and a compatibility/migration plan is approved.

**Verification:** Existing stores open successfully, no inferred schema is written, and a non-null legacy value cannot activate schema validation.

### Task 12: Prepare the optional inference design only if the maintainer requests it

**Objective:** Define a safe replacement instead of reviving the rejected implementation.

**Files:** New design note in the issue or `docs/architecture.md`; production files only after approval.

**Required design:**

1. Explicit configuration gate defaulting to disabled; no behavior change for agents without opt-in.
2. Schema identity keyed at least by agent and tool, never one schema merged across unrelated tools.
3. JSON-only samples; deterministic handling for strings, arrays, nulls, malformed output, and schema-generation failures.
4. A hard byte/sample cap before buffering; no unbounded `tool_outputs.extend(...)`.
5. Capture permission enforced at the pipeline boundary, so capture-off data cannot be inferred or persisted.
6. Validation errors reduced to safe structural metadata; never place raw output or instance text in alert details unless the existing explicit opt-in for captured alert content applies.
7. Versioned/windowed baseline lifecycle with recomputation/reset behavior and explicit treatment of existing baselines.
8. API/CLI/MCP responses expose the schema source and lifecycle state if users are expected to trust the result.

**Tests required before implementation:** mixed-tool outputs, non-JSON output, capture off, sensitive output, cap enforcement, baseline rebuild/reset, old baseline compatibility, and end-to-end alert sanitization.

**Verification:** No optional inference implementation is merged until the maintainer accepts this contract.

---

## Track 3: Small cleanup items that are actually live

### Task 13: Correct remaining live documentation claims

**Objective:** Remove misleading claims without changing behavior.

**Files:**
- Search/modify: `docs/architecture.md`
- Search/modify: any live docstrings in `tokenjam/api/routes/status.py`, `tokenjam/core/db.py`, and `tokenjam/core/cost.py`

**Steps:**

1. Search for `inferred schema`, `rollup`, `span truth`, and claims that a session total is a raw span sum.
2. Update only claims that describe behavior still present in the current tree.
3. Leave historical repair/accounting documentation intact where it accurately describes `doctor --repair` and canonicalized backfill behavior.
4. Do not retain stale references to deleted `session_token_cost_rollup` or `infer_schema_from_outputs` implementations.

**Verification:** A repository search finds no documentation that says inference is active or that a removed trace-wide rollup exists.

### Task 14: Decide whether to simplify the four-token helper call site

**Objective:** Address the maintainer's non-blocking allocation suggestion only if it can be done without weakening the canonical helper boundary.

**Files:**
- Inspect: `tokenjam/api/routes/sessions.py:520-533`
- Inspect/modify: `tokenjam/core/optimize/accounting.py:79-84`
- Test: `tests/unit/test_cost_accounting_guards.py`

**Steps:**

1. Confirm whether the DuckDB result row can expose named attributes; do not assume it can.
2. If it cannot, avoid a larger helper API change solely to remove one small temporary mapping.
3. If a typed row adapter is already used elsewhere, reuse it and preserve the same four-bucket call-site guard.
4. Keep this as a separate, non-blocking cleanup so it cannot obscure the correctness work.

**Verification:** The call-site test still proves the helper is invoked with input, output, cache-read, and cache-write values, and the implementation does not replace the canonical helper with duplicated arithmetic.

---

## Final verification and delivery

After each logical task, run the smallest focused test. Before any authorized push, run:

```bash
env -u FORCE_COLOR ./.venv/Scripts/python.exe -m pytest \
  tests/synthetic/test_ingest.py \
  tests/synthetic/test_schema_validation.py \
  tests/unit/test_ingest_accounting.py \
  tests/unit/test_session_cost_drift.py \
  tests/unit/test_db_helpers.py \
  tests/integration/test_storage_backend_parity.py \
  tests/integration/test_api.py -q

./.venv/Scripts/python.exe -m ruff check tokenjam tests
./.venv/Scripts/python.exe -m mypy tokenjam
git diff --check origin/main...HEAD
```

Acceptance criteria:

- PR #744 remains limited to the two safe helper integrations unless the user explicitly authorizes a new PR scope.
- No shared trace causes two sessions to claim the same unattributed cost.
- Reverse arrival and duplicate live/backfill observations are deterministic and idempotent.
- No reader performs a raw trace-wide rollup or disagrees with the canonical session aggregate.
- Archive work is grouped/bounded and does not perform candidate-by-candidate full scans.
- Explicit schema validation remains available; capture-off output cannot reach validation.
- Automatic inference is either absent, or separately approved and satisfies every privacy, identity, cap, and lifecycle requirement above.
- Documentation and tests do not describe removed behavior as active.
- Any disclosure follows the repository contribution guidance exactly.
