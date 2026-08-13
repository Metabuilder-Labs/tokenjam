---
description: Analyzer authoring, registry naming, persona gating, gate measurement, environment sensitivity, pipeline-order pitfalls, plus the summarize and rulewrite lifecycles.
paths:
  - "tokenjam/core/optimize/analyzers/**"
  - "tokenjam/core/optimize/registry.py"
  - "tokenjam/core/optimize/runner.py"
  - "tokenjam/core/optimize/cost_proposals.py"
  - "tokenjam/core/summarize/**"
  - "tokenjam/core/rulewrite/**"
---

# Analyzer rules (`core/optimize/`)

Read `.claude/rules/optimize-architecture.md` for the package walkthrough and the dollar-field contract,
and `.claude/rules/optimize-cost-figures.md` for the figure-discipline rules (22, 27, 28, 30, 32, 41).

### Critical Rule 16 — New optimize analyzers self-register

Drop a `.py` file under `tokenjam/core/optimize/analyzers/` with a function decorated
`@register("name")` taking `AnalyzerContext`. Auto-discovery in `analyzers/__init__.py` walks the
directory at import time. `cmd_optimize.py`'s positional `findings` Click choices read from
`ANALYZER_REGISTRY.keys()` at decoration — no edits needed there. If your analyzer depends on (or is
depended on by) another, append it to `ANALYZER_ORDER` in `runner.py` at the right position. Wave-2
analyzers attach their findings to `OptimizeReport.findings[name]` (generic dict); the older
`downsize` (registered name; file is `model_downgrade.py`) and `budget-projection` analyzers retain
typed slots on `OptimizeReport` for backwards compat with `cmd_optimize` and the MCP server.

### Critical Rule 19 — Analyzer registry names ≠ file names

Registry strings (`downsize`, `cache`, `script`, `trim`) are decoupled from Python module filenames
(`model_downgrade.py`, `cache_efficacy.py`, `workflow_restructure.py`, `prompt_bloat.py`). The 0.3.1
rename only changed `@register("...")` strings; file names stayed for git-blame continuity. When
grepping for an analyzer, search both the registry string AND the older file-name keyword.

### Critical Rule 26 — An analyzer with no fix for the window's dominant persona must never RUN

See `PERSONA_DISABLED_ANALYZERS` in `core/optimize/runner.py`. Read that map for why an analyzer is
missing from a report; each entry's inline comment names the gate it fails. Never copy the list into
a doc; a doc copy drifts into a false claim about which analyzers run. Don't conflate the two
"disabled" mechanisms: a **fake skip** still runs and queries, returning an `enabled: False` finding
(`trim`, `cache-recommend` with prerequisites off); a **true skip** drops the analyzer before
dispatch. The persona gate is a TRUE skip — `build_report` subtracts
`disabled_analyzers_for_persona(persona)` right after validation. Three consequences. **(a) The bar
is three gates:** the output ends in a concrete edit to a file or setting this persona controls; the
user is net better off after the fix's own standing cost (a rule in an always-loaded instruction file
is re-sent every future session, forever); and the saving does not come from making the agent terser
or dumber. "Describable in a text field" is not a bridge to a real edit. **(b) Mirror every gate into
`COST_ANALYZERS` (`core/optimize/cost_proposals.py`), an INDEPENDENT second selection surface**
feeding the Review inbox — use `cost_analyzers_for_persona`; `cost_proposals_from_report` gates its
adapter table off the same map. Skip the mirror and a "disabled" analyzer keeps surfacing findings in
Review. **(c) A sub-check another analyzer attaches needs its own gate** — `placement` is produced
inside `downsize`, unregistered, so selection cannot reach it and `model_downgrade.run` consults the
map itself. Adding a persona-disabled name is three edits, not one. Only a positively classified
window may lose an analyzer; a persona with no key disables nothing. Which personas have a key is a
question for the map, never for a doc. A disabled analyzer VANISHES rather than rendering a
placeholder — except a name the user typed, which gets one "not run, and why" line, since silence
after a typed command reads as a bug.

### Critical Rule 29 — Before "fixing" an analyzer that looks inert, measure whether its gate is unreachable or merely selects worthless sessions

The two have opposite fixes. `downsize` was contributing a trivial slice of the Review-inbox rollup,
and the ticket's hypothesis was that its small-session gate (the `SMALL_INPUT_TOKENS` /
`SMALL_OUTPUT_TOKENS` / `SMALL_TOOL_CALLS` ceilings in `model_downgrade.py`) was structurally
unreachable for a Claude Code session. Measured on a real corpus, it was not: **a meaningful minority
of CC sessions cleared the gate** — they just carried a negligible share of main-thread spend, and
only part of even that had a cheaper same-family target at all. The gate fires fine; small sessions
are just cheap. Loosening it (the reflex fix for an "unreachable" gate) would have produced MORE
near-worthless cards, against the standing don't-fill-the-inbox constraint, while changing the dollars
by nothing. The real gap was a question nobody asked: a premium model acting as the DRIVER of a long
undelegated session, where the waste GROWS with session size and so is structurally invisible to any
summed-per-session gate — *the worse the waste, the less eligible the session became.* Two general
lessons. **(a) An inert-analyzer ticket's first deliverable is the gate-clearance measurement, not a
rewrite** — sessions-cleared and share-of-spend are different numbers and only the second one tells
you whether to touch the threshold. **(b) When a metric is summed per session, waste that compounds
with session length inverts the gate.** Check for that inversion before assuming a threshold is
miscalibrated.

### Critical Rule 31 — `summarize` and `deadweight` price the files a WALK found, so the ambient environment still decides the answer

The mechanism has one more step than it used to. Both read their file population out of the ingested
`agent_config_files` table (`core/agent_config`), but that table is populated by the same filesystem
walk they used to run inline — the DB is where the walk's result is kept, never a substitute for it.
So the environment the walk runs in still decides everything below, and measuring from an isolated
harness still silently understates rather than erroring. Two things the DB step adds: a read is
scoped to its own populating pass (`store.select(seen_at=...)`), because a persistent table holds
every root ever scanned and an unscoped read would price a repo the current window never touched; and
a store that cannot persist marks itself `degraded` and answers from its in-memory mirror, because a
table that took nothing would otherwise answer "no config is present" — a positive claim from a pass
that never got to look.

They are exposed through DIFFERENT mechanisms; do not treat them as one case.

- **`deadweight` really does read the session's recorded CWD** (`analyzers/deadweight.py` `_session_cwd`, fed into the per-session loop) to find each repo's MCP config, so it decays as those recorded paths go stale or disappear.
- **`summarize` never sees a session CWD at all.** `core/summarize/candidates.list_candidates` takes no such input; with no explicit `path` it scans the catalog **globals** plus `Path.cwd()` — the tj PROCESS's own working directory, never the corpus's recorded session paths. Its floor is therefore the globals, so moving the CWD barely moves the figure — the candidates under `~/.claude/` resolve either way. It collapses only when the GLOBALS vanish too, which is what **repointing `HOME`** does (exactly what `tests/conftest.py`'s `_tj_isolated_home` does). That, and never a missing recorded CWD, is `summarize`'s real failure mode.

Neither analyzer raises or logs a warning for its own case — a missing file just means fewer
candidates, which reads exactly like "the user already fixed it." **Any measurement of either dollar
figure must run against the real environment: the real `HOME` for `summarize`, plus the working-tree
paths the corpus's sessions were recorded under for `deadweight`** — never an isolated test worktree.
Copy the real DuckDB file to a scratch path first (never open the live file read-write while
`tj serve` holds its lock) rather than repointing `HOME` alone, and hold the cwd IDENTICAL across a
before/after comparison or the delta silently absorbs a different project-scope file set. A figure
measured from an isolated harness is not evidence that a fix is inert; it is evidence the harness
never saw the files.

**That is the WHERE half; the WHEN half fails in the opposite direction — manufacturing movement
instead of suppressing it — and it catches `relearn` too.** Copying the DuckDB pins the span data,
not these analyzers' *other* inputs: `relearn` walks the live transcript tree (`rglob` /
`read_records` / `st_mtime`) and `summarize` the live working tree, and **a session that is measuring
is also a session that is writing to both** — so a before/after whose halves are separated by real
time shows large movement with no code change behind it and reads as a regression. **Run the two
halves BACK-TO-BACK; if that is impossible, measure the SAME commit twice and subtract that drift
before attributing anything to a diff.** Generalising: **a measurement harness that cannot see what
it is measuring returns a plausible NUMBER, not an error.** An isolated worktree understates, a stale
installed copy from another worktree makes every run agree, live files make every run differ — none
of the three announces itself, and all three look like findings. The defence is a control whose
answer you already know: something that MUST move (so a null result proves the harness has eyes) or
MUST NOT (so movement is attributable). Run one before reporting a delta, especially a delta of zero.

### Critical Rule 39 — A flag set UPSTREAM of a pass that rewrites the same field is not a flag, it is a suggestion

And a unit test on the constructor cannot see it. An analyzer correctly set `advisory_only` on a
family and cleared its write offer at construction; a later pass, `_apply_write_budget`, then assigned
`write_offered` from an allocator that knew nothing about the flag and quietly restored the offer.
Every value on the field was individually correct at the moment it was written, and the last writer
won. The unit test passed the whole time because it exercised the CONSTRUCTOR path directly while the
daemon ran the COMPOSED path — the test was structurally incapable of observing the defect, so a
green suite was evidence of nothing. **The fix shape is not a second guard in the later pass** (that
is a race between two writers, which the next pass re-opens); it is to **remove the item from the
INPUT the later pass consumes**, so no decision exists to overwrite. Here the advisory family never
becomes a `WriteCandidate` at all. Two rules follow. *(a) When you set a field that a downstream pass
also writes, either withdraw the row from that pass's input or move the decision into it — never both
write it.* Grep for other assignments to the field before assuming your assignment is the final one.
*(b) A test that constructs the object under test proves the constructor works and says nothing about
the pipeline.* Assert through the composed path a user actually reaches (build the report, run the
budget pass, then read the field), which is the same lesson Critical Rule 36 draws for repair passes.

### Critical Rule 40 — A fixture that is a real domain object can stop being a valid vehicle for what it tests while every assertion still passes

For the wrong reason. Eight fixtures seeded one particular relearn failure family in order to
exercise write-budget behaviour. That family later became advisory-only, so it can no longer
demonstrate ANY budget behaviour at all: it is never offered, never priced, never ranked. The tests
stayed green, because the assertions they made happened to hold vacuously on a family that had left
the population under test. This happened twice on the same branch, with two different families. The
hazard is specific to fixtures that are real domain objects rather than stubs: **changing a domain
object's semantics silently invalidates every test that used it as scaffolding, and the suite reports
success either way.** Three habits. *(a) When you change what a family, analyzer or kind MEANS, grep
the tests for its name and re-read each use — asking not "does this still pass" but "can this fixture
still exhibit the behaviour the test claims to check".* *(b) Prefer a fixture that would FAIL loudly
if it stopped qualifying* — assert the precondition explicitly (`assert cluster.write_offered`) before
asserting the behaviour, so the vehicle's own validity is pinned rather than assumed. *(c) When you
do have to swap the fixture, say in the docstring which object it replaced and why the old one
stopped qualifying* — that note is what stops the next reader from "restoring" it.

### Critical Rule 42 — A gate whose predicate is computed over a set passes VACUOUSLY when that set is empty

`not (missing or duplicated or extra or reordered or malformed)` reads as "structure intact", but
every one of those is derived from the protected-block list, so on an input carrying no protected
blocks each is trivially empty and the gate returns intact having verified NOTHING. It is silent and
indistinguishable from a genuine pass: same return value, same downstream path, no warning. The
asymmetry is what makes it dangerous — **the population least able to satisfy a gate's premise is
routinely the one that most needs the check**, so the vacuous case and the high-risk case are the
same case. A plain-prose instruction file has nothing to protect and is also the input most likely to
hijack its own rewrite. Two requirements. *(a) A gate must demand an AFFIRMATIVE signal that
verification actually happened* — here the model must echo back a per-call nonce envelope
(`wrap.envelope`), which is a thing that can only be present if the response came from the rewrite it
was asked for; markers that merely SURVIVE prove nothing when there were none. *(b) It must FAIL
CLOSED: "nothing to verify against" is a refusal, never a pass.* When adding a check, ask what its
predicate reads and what happens when that is empty; if the answer is "it passes", the check does not
exist for the inputs you care about. The inverted twin is a filter that reads structured fields out of
prose: a record whose FORMAT does not match is invisible to that filter rather than rejected by it,
and is indistinguishable from one correctly excluded. Both failures are silent, and both are found by
asking what the predicate reads rather than what it returns.

### Critical Rule 47 — A lane that renders only in the CLI is unfinished

**A fix must also reach the web Review inbox through `cost_proposals`, not just a CLI renderer, or
it does not exist for most users.** Wire both surfaces in the same change that adds a lane, and
verify through both before calling it done.

## `tokenjam/core/summarize/`

Structure-aware prompt summarization (advisory). Pure domain logic — no `tokenjam.cli` /
`tokenjam.api` imports (delivery's API path lazily imports `httpx`, its lone outbound dependency).

- `detect.py` classifies prose vs. structure (fenced/inline code, tags, templates, tables).
- `candidates.py` (+ `catalog.py` / `estimate.py`) powers the `tj summarize list` scan.
- `wrap.py` is the pure protect→restore algorithm: wrap each structured span behind an id'd
  `<tj-keep>` marker, restore verbatim by id — structure is a hard guarantee.
- `session.py` is the no-scratch `prepare`/`check` lifecycle + staging (re-derives the wrap from the
  live file + a content hash; persists nothing but the staged result).
- `apply.py` writes a staged rewrite back to the file (default dry-run; `--go` writes) behind an
  owner + content-hash + symlink guard, with a gzip backup and `undo`.
- `backup.py` stores the gzipped original + metadata under `~/.tj/summary/backups/`.
- `delivery.py` is the CLI's automated rewrite step — `claude -p` (subprocess, timeout-guarded) or
  the Anthropic API (lazy `httpx` + the user's own `TJ_ANTHROPIC_API_KEY`) — plus the "pays for
  itself" amortization.
- `load_semantics.py` is the single source of truth for how an agent-config file loads (always
  resident vs. frontmatter-now / body-on-invocation); `invocations.py` observes the invocation
  multiplier from Claude Code transcripts. `load_semantics.py`'s split is consumed on the write side
  by `core/rulewrite/delivery.standing_tokens_per_session` — see
  `.claude/rules/optimize-architecture.md`.
- `repo_roots.py` resolves recorded session cwds to repo roots for `core/optimize/rule_placement`.

**Environment sensitivity (Critical Rule 31):** `candidates.list_candidates` takes no session CWD.
With no explicit `path` it scans the catalog **globals** plus `Path.cwd()` — the tj PROCESS's own
working directory, never the corpus's recorded session paths. Its floor is therefore the globals, so
moving the CWD barely moves the figure; it collapses only when the GLOBALS vanish too, which is what
repointing `HOME` does. Any measurement of its dollar figure must run against the real `HOME`.

## `tokenjam/core/rulewrite/`

The ONE rule-write lifecycle every rule-writing analyzer shares: `plan.py` (list), `apply.py`
(stage/check/apply/undo), `store.py` (staging + gzip backups), `types.py` (shapes),
`delivery.py` (the `DeliveryKind` seam). Pure domain — no `tokenjam.cli` / `tokenjam.api` imports.

Reuses `core/summarize/apply`'s guard model and `relearn_apply`'s block renderer rather than
reimplementing either; the store persists rendered OUTPUT, never a recipe, so what a reviewer
approved in the diff is byte-for-byte what apply writes.

Surfaced by `tj rules` (`cli/cmd_rules.py`) and `/api/v1/rules/*`.

`core/optimize/rule_placement.py` is the WHERE half — named that, not `placement`, because
`placement` is already a registered analyzer name for an unrelated question (the Batch API lane,
`analyzers/batch_placement.py`; Critical Rule 19).

`delivery.py` is the delivery-mechanism seam: a `DeliveryKind` owns its own renderer AND its own
pricer, and adding a mechanism is a registration plus two functions — never an edit to the staging,
diff, apply or undo machinery, none of which may name a CLAUDE.md. See
`.claude/rules/optimize-architecture.md` for the full delivery-kind, hook-rails and path-scoped-rule
contracts.
