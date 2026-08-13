---
description: Backfill windowing, transcript dedupe, write-vs-repair invariants, and cross-observer call accounting.
paths:
  - "tokenjam/core/backfill.py"
  - "tokenjam/core/transcript_sync.py"
  - "tokenjam/core/transcript.py"
  - "tokenjam/core/ingest.py"
  - "tokenjam/core/ingest_adapters/**"
  - "tokenjam/core/db.py"
  - "tokenjam/core/optimize/accounting.py"
  - "tokenjam/cli/cmd_backfill.py"
  - "tokenjam/cli/cmd_doctor.py"
---

# Ingest / accounting rules

### Critical Rule 33 — A backfill run scoped by `--since` (or any window) must NEVER run the stale-scheme reconciliation DELETE

A windowed keep-set is structurally incomplete. `ingest_claude_code` builds `keep_by_session[sid]` as
"the complete current-scheme span_id set for this session" and then deletes every
`backfill.claude_code`-tagged span for that session that isn't in it, on the reasoning that only a
pre-v0.5.2 uuid-keyed orphan could be missing. That reasoning holds ONLY on an unbounded pass. `since`
filters files by mtime AND by parsed `ended_at`, so a session straddling the window boundary — a long
conversation whose main transcript is still being appended while its `subagents/*.jsonl` finished days
ago — yields only its in-window files, and the out-of-window siblings' already-ingested spans get
destroyed as if they were stale. Verified directly: a two-file session re-ingested with `--since 2d`
after the subagent file aged out lost that file's span
(`tests/unit/test_transcript_sync.py::test_windowed_catch_up_never_purges_an_out_of_window_sibling_file`).
The guard already excluded `max_sessions` for exactly this reason and the comment even said "its
per-session keep-set may be incomplete" — but it named only the quickstart cap, so the identical
hazard one parameter over went unnoticed for as long as `--since` existed. **Two durable
generalisations.** *(a) When a guard exists because a parameter narrows what a pass observes,
enumerate EVERY parameter that narrows it, not the one that motivated the guard* — the correct
predicate here is `since is not None or max_sessions is not None`, i.e. "is this pass bounded at all",
and writing it that way makes the next bounding parameter fail safe by construction. *(b) A
destructive step justified by "anything absent must be an orphan" is only ever as safe as the
completeness of the set it compares against; before adding one, ask what makes the set complete and
which caller can break that.* The blast radius scales with call frequency: this delete was survivable
while it fired only on a human-typed command, and becomes continuous silent data loss the moment a
daemon job calls the same function on a schedule — which is precisely what continuous ingestion does.

### Critical Rule 34 — Dedupe Claude Code transcripts on the INTERNAL `sessionId`, never on filename or file count

A large share of the `.jsonl` files under `~/.claude/projects` live in nested `subagents/` folders and
carry their PARENT session's `sessionId` internally while having their own filename — measured once on
a real machine, that share was big enough that a filename-keyed count roughly doubled the true session
count. Any reconciliation, count, or gap measurement keyed on the filename therefore over-reports, and
two independent measurements of the same ingest gap once disagreed by an order of magnitude for
exactly this reason before anyone noticed why. `core/transcript_sync.scan_disk_sessions` is the
canonical content-derived scan (it reads only each file's leading records, so it stays cheap even over
a large transcript tree) and returns each session's full file list so callers never need a second
walk. The corollary that makes this more than a counting detail: because siblings share one
`session_id`, ANY per-session operation — the totals recompute, the reconciliation DELETE, a
`GROUP BY session_id` in an analyzer (see Rule 27) — silently spans main-thread and subagent work
unless it explicitly says otherwise.

### Critical Rule 36 — A repair pass downstream of a defective write makes the write's defect invisible; test the WRITE with the repair REMOVED

`recompute_session_totals_from_spans` reconciles a session row to `SUM(spans)` at the end of a
backfill, and for as long as it ran unconditionally, `upsert_session`'s per-file REPLACE semantics
were indistinguishable from accumulate: both produced a correct row by the time anything read it. The
write was still wrong, and every path that never reaches the repair inherited the defect — a windowed
pass, a live ingest, an error that returns early, any future caller. Measured once on a real corpus,
stored session totals and their span sums disagreed, with a session's tokens reading as a fraction of
its spans' — the signature of a row describing only the last file processed. The suite was green
throughout, because every test ran the repair. **The same shape has a second form worth checking for
at the same time: a mechanism with passing unit tests that no production path actually calls.** The
call-identity dedup helpers in `core/optimize/accounting.py` had a full green test module while the
figures on every real cost surface stayed doubled, because the only caller was the test, which
reimplemented the query it was pinning. A test that constructs its own subject proves the subject
works and says nothing about whether anything uses it — assert through the path a user reaches (here
`parse_log_records` and `ingest_claude_code` over real transcript files), not through a
re-implementation. **The rule for both forms: when a write and a repair both target the same
invariant, the regression test for the write must disable the repair; and when a mechanism is meant to
change a user-visible figure, the test must reach it the way production does.**
`tests/unit/test_ingest_accounting.py::test_re_running_the_same_files_does_not_double_the_session` and
`::test_a_multi_file_backfill_adds_up_without_the_repair_pass` do it with
`monkeypatch.delattr(DuckDBBackend, "recompute_session_totals_from_spans")`; the write then has to
hold on its own or the test fails. Two corollaries. *(a) Keep the repair, and rewrite its comment to
say what it is now FOR* — here, rows a destructive reconciliation step moved `SUM(spans)` out from
under, and rows an older build left holding a wrong base for any later accumulation. A repair with no
stated remaining job is the one a later reader deletes as redundant. *(b) A DELTA-shaped write is what
makes accumulation idempotent* — sum over the spans the write actually INSERTED, never over what the
file described, so a re-ingest that inserts nothing adds nothing and a span dropped as a duplicate
contributes to neither side of the invariant.

### Critical Rule 37 — A content fingerprint may collapse two observations ACROSS ingest sources, and never two from the same source

One LLM call reaches the store more than once whenever a session is both observed live and backfilled
from its transcript; each path mints its own `span_id`, so `span_id`-keyed idempotency cannot see the
overlap and every raw `SUM(cost_usd)` prices the call twice. Where no id names the call (Rule 38) the
only key both observers share is its billed shape — `core/optimize/accounting.call_fingerprint`, over
session, model and all four token buckets. That is a weaker claim than an id, and the asymmetry is
what makes it safe to act on at all: **two rows from DIFFERENT observers carrying one shape are one
call seen twice; two rows from the SAME observer carrying one shape are two real calls.** Dropping the
second under-reports spend, which is the quiet-lie-in-the-user's-favour Rule 22 exists to stop, so
suppression is additionally capped at the number of observations the other source actually recorded
(`accounting.duplicate_budget`) instead of firing on any match. **The trap that follows is not
optional: a suppressed observation is never stored, so the store cannot tell you how much of the
budget you have already spent.** Without a running per-call tally the same budget is re-derived and
re-spent on the next arrival, and a genuinely repeated call is silently dropped — precisely the
failure the cap was added to prevent. Both ingest paths therefore carry one
(`IngestPipeline._suppressed_calls`, `backfill.DuplicateScan`), scoped to the run so it also spans the
several files of one session. Anything that suppresses on a derived key needs all three parts: the
cross-source-only restriction, the cap, and the tally.

### Critical Rule 38 — Do not assume a telemetry source carries an id for the thing it describes

Check, then derive the key from stored COLUMNS when it does not. Cross-observer accounting needs a
name for the underlying API call, and the obvious design is to stamp the provider's own response id on
both sides. That works wherever the observer sees the provider's response — `sdk/integrations/anthropic.py`
stamps `gen_ai.response.id`, `core/backfill.py` stamps `tj.call_id` off the assistant message key. It
does not work for the pairing that matters most on this product's dominant corpus: **Claude Code's own
OTel log exporter emits no per-call identifier at all.** Read `ClaudeCodeEvents` in `otel/semconv.py`
for what its `api_request` event actually carries, and confirm there before designing around an id —
the absence is the kind of premise that silently invalidates a whole approach. A stamp only ONE side
of a pair can carry is worse than no stamp: it yields two different keys for one call, so the pairing
never fires and nothing looks broken. **Derive the cross-observer key from columns the store already
holds rather than from an attribute an ingest path promises to write** — that also makes it readable
on rows written before any stamping existed, which is what lets `tj doctor --repair` clean a database
an older build already doubled. Keep stamping the ids that DO exist: they are exact where a derived
key is only strong, and they are what a future exporter revision would let you key on directly. If you
are about to rely on the absence, re-verify the exporter's attribute set rather than trusting this line.
