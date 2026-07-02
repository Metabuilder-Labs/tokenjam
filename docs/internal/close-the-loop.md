# Close the loop — product decision + shape (#53)

Status: **shipped (minimal shape)** · 2026-07-03

## The gap

tokenjam's **capture half** is strong: `tj trace` gives a run timeline, model /
backend info is auto-captured, everything is local-first. But the half a
recurring class of users calls "the real value" was entirely absent — going from
*"something weird happened"* to *"this is now a repeatable check"*:

1. **Human notes/labels after the fact** on a run — no annotation feature.
2. **Turn a bad run into an eval/regression case** — no evals/datasets concept.
3. **Rerun after changes + keep history of what fixed it or made it worse** —
   no fix-history.

This is squarely Langfuse/promptfoo territory today. It is *adjacent* to existing
machinery — drift baselines, the method snapshots that preserve "how an agent
attempted" work — not a one-off.

## The decision: tokenjam owns a local-first loop **primitive**, not an eval platform

The load-bearing question the ticket posed was: **does tokenjam own evals, or
lean on the Langfuse ingestion path it already has?**

**Answer: tokenjam owns a thin, local-first loop primitive. It does not build an
eval-runner, and it does not push anything to Langfuse.**

Why:

- **Local-first is the product's identity.** "No cloud backend, no signup"
  (README line 1). Pushing runs out to a hosted eval platform to close the loop
  would contradict the whole premise and the users who asked for this (offline /
  local-AI users). The Langfuse integration is **inbound only**
  (`core/ingest_adapters/langfuse.py` pulls external traces *in*); it is not an
  egress path and this feature does not make it one.
- **Owning a full eval/assertion runner is a different product (YAGNI).**
  Automated pass/fail assertions, dataset management, scoring rubrics — that is
  promptfoo/Langfuse's job and a multi-quarter bet. tokenjam's leverage is that
  it *already has the runs*. The cheap, high-value move is a ledger over them.
- **Honesty discipline (Critical Rule 14).** tokenjam describes, it doesn't
  grade. So pass/regress is a **recorded human verdict**, never an automated
  score. This keeps the feature truthful and tiny, and it matches how drift and
  the analyzers already talk ("candidate", "review before…").

## The shape (minimal, shipped)

Three additive tables (migration 16), a pure-domain module, an API surface, a CLI
group, and a Lens "Loop" tab. All local, all offline.

| Concern | Table | What it is |
|---|---|---|
| (a) note/label after the fact | `run_annotations` | append-only human note + optional verdict (`good`/`bad`/`mixed`/`unknown`) keyed to a session (a "run") |
| (b) promote a bad run into a case | `expectations` | a named/described expectation, optionally promoted `origin_session_id` FROM a run |
| (c) fix-history | `expectation_runs` | one row per rerun recorded against an expectation, `outcome` ∈ `pass`/`regress`/`unknown` |

Surfaces (all hit the exact same storage, so they stay consistent):

- **Core:** `tokenjam/core/loop.py` — dataclasses + storage helpers.
- **API:** `tokenjam/api/routes/loop.py` —
  `POST/GET /sessions/{id}/annotations`,
  `POST/GET /expectations`, `GET /expectations/{id}` (with history),
  `POST /expectations/{id}/runs`.
- **CLI:** `tj loop annotate | annotations | expect | expectations | record |
  history`. Dual-path: when `tj serve` holds the DB lock the CLI routes through
  the running server (api_mode), else direct DuckDB.
- **Lens:** a "Loop" tab on the Session Detail view — annotate + verdict, promote
  to expectation, record a rerun as pass/regress, and read the fix-history.

## What is deliberately NOT in scope (yet)

- No automated assertions / scoring — pass/regress is a human verdict.
- No repro-bundle export (the ticket's weaker adjacent gap (e)) — filed separately
  if demand persists.
- No per-run automated "failure reason" diagnosis (gap (d)) — the existing
  alerts + drift already cover part of this; a diagnosis analyzer is its own bet.
- No Traces/Drift-view integration beyond the Session Detail "Loop" tab — the tab
  is where a run is already inspected, so it is the natural first home.

## Done-when check

> A user can label a run in Lens, mark expected behavior, and see a history of
> subsequent runs against that expectation (pass/regress).

Met: verified end-to-end through the real CLI (api_mode), the REST API, and the
Lens "Loop" tab (annotate persisted via the browser; pass/regress history renders
newest-first). See the PR's validation table.
