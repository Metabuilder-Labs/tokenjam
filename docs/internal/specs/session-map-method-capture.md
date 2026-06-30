# Session map — capturing *how* an agent solved a problem

Status: **design / in-progress** · Owner: session-status work (PR #306) · Last updated: 2026-06-30

## 1. Goal

Most agent observability answers **"what did my agent do"** — for a coding agent that is near-tautological
(it did the thing you asked). This feature answers the harder, more useful question: **"how did my agent
*attempt* the problem"** — the *method*: the strategy it chose, the order it reasoned in, what it tried and
abandoned (dead-ends), why it delegated, where it backtracked.

The urgency comes from **ephemeral agents**. A harness spawns subagents, background agents, and child
`claude` sessions in fresh terminals; they finish and are killed. The parent keeps only what they
*returned* (the "what"), never their *method* (the "how"). The single artifact most useful for improving an
agentic system — *how the work was actually attempted* — is exactly the one that dies with the process.

Three surfaces consume this:

- **Timeline** — the deterministic play-by-play of *every* agent tied to a session (incl. dead children).
- **Map** — an at-a-glance board: phase/tool/context/cost over time (lens ①) + a codebase-territory treemap
  (lens ③). The *what/when/where*.
- **Approach** — the *how* drill-in: a per-agent **method spine** (orient → reproduce → locate → hypothesize
  → try → fix → verify), dead-ends struck through, delegations expandable into the child's own spine,
  recursively.

Recursion (expand any delegation into the child's own view) is required in **all three** tabs.

## 2. Capture reality (what the code does today)

A review of the live + backfill + linkage paths (branch `fix/claude-code-session-status`) found **two
independent nesting mechanisms that do not compose into one tree today**:

| | **Scope A — in-session subagents** (Task/Agent sidechains) | **Scope B — child sessions in new terminals** |
|---|---|---|
| Method (narration) source | on-disk JSONL → `core/transcript.py` rebuilds it recursively (depth ≤ 3) | a *separate* session; **no method nesting** |
| Cost/identity | `sub_agent_id` span column, set **only** by `core/backfill.py` from `isSidechain`+`agentId` | own `SessionRecord`; linked only by declared `tokenjam.run_id` / `parent_session_id` |
| Live OTLP path | flat 2-level, **zero subagent identity** (`api/routes/logs.py`) | streams live (global telemetry env) **iff** the spawner tagged it |
| Stitched into parent? | yes, via transcript reconstruction (`_attach_subagents`, `_build_subagent`) | session-level run tree only (`api/routes/runs.py:_build_tree`) — **method never spliced in** |

Key citations: `core/transcript.py` (`MAX_SUBAGENT_DEPTH=3`, `TOTAL_STEP_BUDGET=4000`, recursive
`_build_subagent` with `depth_capped`/`budget_capped`/`cycle` markers); `core/backfill.py:~329`
(`sub_agent_id = record["agentId"] if record["isSidechain"]`); `otel/semconv.py` (`RUN_ID`,
`PARENT_SESSION_ID` resource attrs — **no** `sub_agent_id` on the wire); `api/routes/logs.py` (flat 2-level
synthesis); `mcp/server.py` `setup_harness` boundary: *"no method recovers untagged, uninstrumented work."*

### The load-bearing gap

**Method is never persisted.** `/story` and `/workmap` recompute the Story from on-disk JSONL on *every*
request; nothing writes it to the DB. Claude Code **prunes those transcripts**. So "method survives the
killed agent" is only true *until CC garbage-collects the file* — after which the cost spans remain in the DB
but the *how* is gone. This is the prerequisite for everything else.

Secondary gaps: (2) no unified tree across Scope A + B; (3) cross-terminal linkage is opt-in & lossy;
(4) the live path contributes nothing deep — depth/identity come only from JSONL.

## 3. Plan

Ordered by dependency. UI is **last**, because what we can honestly show is determined by what we can
durably capture.

### M1 — Persist a method snapshot (the unlock)

- **Migration (append-only):** add a `session_story` table — `(session_id, story_json, subtree_json,
  captured_at, source, schema_version)`. Stores the reconstructed Story + subagent subtree so it outlives
  transcript pruning.
- **`core/method_capture.py`:** `capture_session_method(db, session_id, projects_dir, ...)` builds the Story
  via `transcript.py` and upserts the snapshot. Idempotent (re-capture overwrites with the latest, fuller
  read). Pure-ish: reads files, writes one row; no analysis.
- **Wire-in:** capture at session close (`cli/cmd_session_end` / close-signal route) **and** per
  newly-ingested session at the end of `tj backfill claude-code` (`core/backfill.py:ingest_claude_code`,
  `source="backfill"`) — both now wired. This is what makes historical sessions (the ones most likely to be
  pruned later) keep their method. Capture is best-effort and error-tolerant (a missing/pruned transcript
  logs and no-ops — never raises into ingest).
- **Read-through:** `/story` and `/workmap` prefer the persisted snapshot, falling back to live transcript
  reconstruction when none exists. Persisted snapshots make a *killed* agent's method survive a later
  transcript prune.
- **Honesty:** snapshot carries `source` (`live-transcript` vs `backfill`) and the existing
  `depth_capped`/`budget_capped`/`cycle` markers ride through unchanged.

### M2 — Unify the tree + provenance

- Splice the session-level run tree (`runs.py`) into the Story tree so a cross-terminal child's own Story
  (captured per M1) nests under the parent's node.
- Tag every node with **provenance** (`in_session_subagent` | `cross_terminal_child`) and
  **capture_completeness** (`full` | `capped` | `session_level_only` | `unrecoverable`) so the UI never
  claims more than the data supports.

### M3 — UI (Map ①+③, Approach, recursion)

- Replace/augment `WorkMapSection` in `tokenjam/ui/index.html`: the ①+③ board, the Approach method spine,
  and recursive expand-delegation across Map / Timeline / Approach.
- **Honest scope markers:** in-session subagents render their full method; cross-terminal children render as
  session-level until M2 splices their Story; the "method kept" badge is qualified by `capture_completeness`.
- Guard with static-grep regression tests (`tests/unit/test_lens_ui_regression.py`) + `test_ui_offline.py`,
  per the repo's no-JS-runner convention.

## 4. Honesty boundary (Rule 14)

The method spine is reconstructed, never graded. Each line carries a **source**: `agent's words`
(narration / TodoWrite), `structural` (revert / retry / spawn — no judgment), or `distilled` (the existing
opt-in `core/distill.py` LLM title, off by default). Dead-ends are detected structurally (edit→revert,
fail→different-approach). No surface ever asserts an approach was good or bad.

## 5. Explicitly out of scope (for now)

- Capturing a child terminal that was **never instrumented and never tagged** — the code is explicit that
  this is unrecoverable; we surface it as `unrecoverable`, we do not invent it.
- Reconstruction **beyond depth 3 / 4000 steps** — kept capped with honest markers.
- LLM-generated *intent* beyond the opt-in distilled titles — the spine stays deterministic by default.
