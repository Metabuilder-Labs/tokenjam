# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**This file is deliberately small.** Only what governs *every* file lives here. Everything else —
architecture detail and path-scoped Critical Rules alike — lives in `.claude/rules/*.md`, each of
which loads only when you touch a file its `paths:` frontmatter matches. The single index is at the
bottom of this file — if you are looking for a rule by number, or for an area's architecture notes,
go there first.

## Project Overview

`tj` (TokenJam) is a local-first, OTel-native **cost-optimization layer** for AI agents (with a full observability stack underneath). No cloud backend, no signup. It captures telemetry from agent runtimes, stores it in a local DuckDB database, and runs the optimize analyzers (the live list is `ANALYZER_REGISTRY` / `ANALYZER_ORDER` in `tokenjam/core/optimize/runner.py` — derive it from there rather than trusting a hand-maintained enumeration) that surface cost-saving candidates from real usage — plus a CLI, local REST API, web UI, and MCP server for querying. Install via `pipx install tokenjam` (recommended — sidesteps PEP 668 on Homebrew Python and Debian 12+/Ubuntu 24+) or `pip install tokenjam` in a venv. Run via `tj <subcommand>`. Requires Python >=3.10.

## Build & Development

```bash
pip install -e ".[dev]"                 # dev install
ruff check tokenjam/                    # line-length=100, target py310
mypy tokenjam/                          # only attr-defined + no-any-return disabled (with justification); see [tool.mypy] in pyproject.toml
# Tests — CI runs all except e2e. Layers: unit = pure logic no I/O <1s; synthetic = span
# injection via factories, zero cost; agents = mock agent scenarios, full SDK path;
# integration = CLI + API. e2e needs TJ_ANTHROPIC_API_KEY and is auto-skipped otherwise.
pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/
pytest tests/unit/test_config.py::test_function_name -v    # one test
cd sdk-ts && npm install && npm test    # TypeScript SDK (independent package)
```

**If a batch of CLI-render tests fails on substring assertions, check for `FORCE_COLOR` in your
shell** — Rich then interleaves ANSI escapes inside the asserted phrases. Re-run with
`env -u FORCE_COLOR python -m pytest ...` before concluding you broke something. Full note, plus the
`~/.tj` isolation fixtures, in `.claude/rules/tests.md`.

## Package dependency boundary

- `tokenjam/core/` is pure domain logic. **Must never import from `tokenjam.cli` or `tokenjam.api`**. CLI and API import from core, not the reverse.
- `tokenjam/otel/semconv.py` is pure constants with no internal imports.
- `sdk-ts/` is fully independent from Python — communicates only via HTTP.

## Working with concurrent agents

When more than one agent is editing this repo in parallel, **each agent must operate in its own git worktree**. A single working directory shares one `HEAD`, so two `git commit` calls from different agents land on whichever branch was checked out last — leading to commits leaking into the wrong PR. We've hit this multiple times.

```bash
git worktree add ../tokenjam-<task> main && cd ../tokenjam-<task> && git checkout -b feat/<task>
git worktree remove ../tokenjam-<task>      # once the PR merges and the branch is deleted
```

Symptom of a missed worktree: `git log` shows a commit on a branch you didn't intend (because another agent's `HEAD` was the checked-out one when your `git commit` ran). If you see this, do **not** force-push — rebase the stray commit off your branch first, and only force-push if you own every commit being rewritten.

`.tj/config.toml` showing as modified or new in `git status` is expected — see Critical Rule 20.

## PR and commit conventions (for any agent producing a PR)

These conventions apply to any agent — feature work, bug fixes, docs, content. Briefs may add task-specific structure but should not contradict these.

### Branch + PR titles

- **Branch names** are slash-separated, kebab-case, prefixed by type:
  - `fix/<issue-or-area>` — bug fixes (e.g. `fix/175-176-cost-framing-backfill-plan`)
  - `feat/<area>` — new features (e.g. `feat/reuse-analyzer-115`)
  - `docs/<area>` — documentation (e.g. `docs/readme-cleanup-v0.4.1`)
  - `chore/<area>` — refactors, renames, infra
  - `release/<X.Y.Z>` — release-cut PRs
- **PR titles** lead with the verb / type and reference issues by number when applicable:
  - `Fix #175, #176: tj cost framing + backfill plan_tier propagation (v0.4.2)` (bug fixes)
  - `[feature] Add Reuse analyzer (#115)` (features)
  - `docs: drop stale CHANGELOG.md + add maintainer contact` (docs)
  - `Bump version to 0.4.1` (release-cut PRs — keep these terse)
- Use **`Closes #N`** in the PR body (not just title) when fixing an issue, so GitHub auto-closes the issue on merge. Multiple `Closes` lines if you're closing several. Do not use the comma form `Closes #1, #2` — GitHub only catches the first; use separate lines.

### Commit messages

- **Subject line** (first line, ≤72 chars): one-line summary in active voice. Reference issues with `#N` when applicable.
- **Body** (after blank line): explain *why* the change is needed, not *what* it changes (the diff shows that). Use full sentences, paragraphs, bullet lists.
- **Trailers** (after another blank line, at the very end):
  - Always include: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` (or the appropriate model identifier)
  - When fixing an externally-reported bug: also include `Co-Authored-By: <reporter-handle> <noreply@github.com>` (e.g. `ashwmu` for the external contributor's reports)
- Use **HEREDOC for multi-line messages** to preserve formatting: `git commit -m "$(cat <<'EOF' ... EOF)"`

### PR body structure

```markdown
[1-2 sentence framing of why this exists]
## Summary                                  — bullets, what changed at a high level
## [Per-issue or per-feature section]       — repeated as needed: symptom, root cause, fix
## Tests / Verification                     — test files added/modified + any live verification
                                              (workflow run URL, screenshot, command output)
## What's NOT in this PR                    — if scope was deliberately limited; why each deferred

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: <reporter-handle> <noreply@github.com>   # if applicable
```

The "What's NOT in this PR" section is load-bearing — it makes the reviewer's job 10x easier when the agent explicitly named what they decided to defer. Use it whenever scope is non-obvious.

### Self-review checklist before requesting review

1. **Tests pass locally.** `pytest tests/unit/ tests/integration/` (or `tests/unit/<file>.py` if narrow).
2. **`ruff check tokenjam/` and `mypy tokenjam/` clean** for any files you touched.
3. **CI on the branch is green** for at least the test-ts job (Python jobs may still be running when you push).
4. **Acceptance criteria from the issue are met** — go through them one by one and verify.
5. **No accidental files in the diff** — `.tj/config.toml`, `.tj-test-data/`, screenshots that were just for debugging, etc.
6. **PR body explains the WHY** — symptom + root cause + fix, not just "fixes the bug."
7. **Honesty discipline preserved.** If the change touches any user-facing string ("recoverable," "estimated," "savings"), verify it matches existing analyzer caveat language. Never silently strengthen claims.

### Scope discipline

- **Do what the brief / issue says, no more.** If you notice an adjacent issue, file it as a separate issue rather than expanding the PR. Reviewers should never have to mentally separate "the fix" from "drive-by cleanup."
- **Exception:** when an adjacent change is functionally required to make the primary fix work (e.g., updating a caller of a function you changed). Note it explicitly in the PR body under "What's also in this PR."
- **When in doubt about scope, ask the master agent before expanding.** A 30-second clarification beats a 30-minute scope review.

### Worker vs master

- **Worker agents do not merge their own PRs.** Open the PR, request review, the master + Anil handle merge.
- **Worker agents do not file follow-up issues unprompted.** If you notice something during your work that's out of scope, mention it in the PR body and let the master decide whether to file.
- **Worker agents do not bump versions.** Release-cut PRs are a separate concern handled by the master / Anil.

## Architecture — where the detail lives

Spans enter from the Python SDK (in-process, via `TjSpanExporter`) or over HTTP
(`POST /api/v1/spans`), both converging at `IngestPipeline.process()` in `tokenjam/core/`. Cost,
alerts and schema validation run as post-ingest hooks; the optimize analyzers run **out of band** on
a background daemon pass and their reports are read from storage by the CLI, the REST API, the Lens
web UI and the MCP server.

There are no per-directory `CLAUDE.md` files. Each area's architecture notes live in that area's
`.claude/rules/*.md`, alongside the Critical Rules governing it, and load on the same `paths:`
trigger — see the index at the bottom of this file.

## Critical Rules

> This list is **"Critical Rule N"** — always cite by full name and file ("Critical Rule 27 in
> `tokenjam/CLAUDE.md`", never a bare "rule 27") to disambiguate from any other numbered list
> elsewhere in this codebase. **Numbers are load-bearing and cited from source and tests — never
> renumber.** 6, 13 and 17 were deleted and their numbers are permanently retired. Only the rules
> listed below are always-on; the rest live in `.claude/rules/` (index follows) with their numbers
> intact. A new rule is APPENDED at the end of whichever list it belongs to, taking the next number
> above the highest in use across this file AND `.claude/rules/` — the numbering is one shared space.

1. **DuckDB only** — never import `sqlite3` or write SQLite-style queries. Use `TIMESTAMPTZ` not `TEXT` for timestamps, `JSON` not `TEXT` for JSON. When extracting dates from `TIMESTAMPTZ` columns, always use `CAST(col AT TIME ZONE 'UTC' AS DATE)` — bare `CAST(col AS DATE)` converts to the local timezone first, causing mismatches with Python's `utcnow().date()`.
7. **Parameterised SQL only** — never use f-string SQL.
9. **Use `utcnow()` for timestamps** — always use `tokenjam.utils.time_parse.utcnow()` instead of `datetime.now()` or `datetime.utcnow()`. It returns timezone-aware UTC datetimes.
14. **`tj optimize` output must never claim quality equivalence** — the `downsize` finding flags structural candidates only. Every user-visible string says "looks like" / "candidate" / "review before switching" — never "safe to downgrade" or "would have worked." The `MODEL_DOWNGRADE_CAVEAT` constant lives on `DowngradeFinding` as a dataclass default so it can't be removed by accident; it must also appear in human-readable CLI output. The same honesty discipline applies to all other analyzers — `cache` ("you're getting X% of available caching"), `cache-recommend` (Anthropic-only, structural prefix detection), `script` ("structural shape matches", "review before replacing with a script"), `trim` ("predicted low-significance regions; review before editing"). `tj optimize --export-config` snippets bake the caveat block into the JSONC output as comments.
20. **`.tj/config.toml` is untracked and must stay that way** — the file contains a live per-install `ingest_secret` and is regenerated by `tj onboard` / `tj serve`. It was committed in error from v0.2.0 through v0.3.5 (leaked secret in git history; see PR #145 + issue #141 finding #6). `.gitignore` covers it, and `tests/unit/test_no_tracked_dev_secrets.py` fails CI if it's re-added to the index. If you see `.tj/config.toml` in your `git status` as modified or new, that's expected — just don't `git add` it.
23. **When removing something that should never have been user-visible, grep the TESTS for it FIRST — a green suite may be ENFORCING the defect, not protecting against it.** An internal tracker id sat in the `tj doctor` MCP-wiring warning, a string the product prints, and `tests/integration/test_cli.py` asserted that the id was PRESENT in `mcp_checks[0]["message"]` — so the suite actively required the leak to be present, and anyone who did the right thing was told by CI that they had broken something. Most people believe CI, which is why it survived so long. A stale comment is entropy; a test asserting the bad state is an ACTIVE DEFENCE of it, and it is invisible until you touch the thing. **The repair is to INVERT the assertion, not to delete it:** the test now asserts the durable content is present AND that no bare `#\d+` remains, so the same pin now defends the correct state. Generalise both halves: before deleting any user-visible string, `grep -rn "<the thing>" tests/`; and when a guard has to go, replace it with its inverse rather than removing coverage. `tests/unit/test_no_tracker_ids_in_printed_strings.py` is the standing version of that inverse for this whole class — it walks every non-docstring string literal under `tokenjam/` and fails on an UNQUALIFIED tracker reference, separating real references from CSS colours and ordinals structurally (markup-shaped literals are skipped; a reference needs two or more digits) rather than by an allowlist, because a guard with exceptions just teaches people to add exceptions. **The sharper half of this lesson, learned by getting it wrong:** the first version of that guard banned EVERY reference, and it immediately pushed a scrub into deleting a correct public issue link out of the spans-column-statistics warning — the reference named the real upstream bug the user had just hit, and removing it destroyed true, useful information. Number-sign references are not the problem. **BARE ones are ambiguous** (they resolve against whatever repo the reader is looking at, and our internal and the public number spaces genuinely collide), and **internal ones must never be public**. Those are two different faults with two different fixes: qualify the first (`Metabuilder-Labs/tokenjam#56`, or a full `https://github.com/...` URL, which is what a printed string should carry since a terminal linkifies nothing and a URL survives being pasted anywhere), and describe-and-delete the second. **Decide which fault you have by EVIDENCE, not by the shape of the token:** open the number and ask whether that artifact actually explains the sentence it sits in. It did for the column-statistics warning, whose public issue is exactly about corrupt spans column statistics, so that one was restored qualified; it did not for the MCP-overhead claim or the schema-self-heal warning, whose numbers name our own queue, so those stayed deleted. A guard that cannot tell those two faults apart will cheerfully destroy good information while looking green.
24. **A surface is reachable only if a USER has a path to it — "the component mounts", "the route resolves", "the code is invoked" and "it is deliberate" each prove nothing.** Static analysis cannot find this class; ruff and mypy stay green through every instance. Four checks, all required. **(a) Check the INVERSE direction:** not "does this nav item resolve to a view" but "does anything link TO this view". **(b) Does the path carry data:** a status literal missing from a hand-written WHERE clause emptied `/api/v1/status` and every surface downstream of it. **(c) Does the destination resolve:** transcript-filename links 404 for un-ingested sessions, so render one only when the session resolves, checked at READ time so cached findings self-correct. **(d) Does the capability have a name a user can type:** an analyzer producing a cost card is invisible while absent from `tj optimize`'s Click choices. Deliberate design and documentation are no defense against invisibility. **The inverse trap is equally real:** a missing `@register` decorator is not evidence of death — `batch_placement` and `downsize_agents` are invoked by `model_downgrade.py`. Ask both directions before deleting anything or declaring anything fine: what can a user actually do?
25. **Removing a feature strands its helper modules — grep every function in the modules it used, not just the files that named it.** When the output-cap hook, its `tj savings` CLI, `core/output_cap.py`, and their tests were deleted, `core/savings_log.py` was left behind untouched: 8 of its 9 functions had zero callers anywhere in the repo, including tests, and its docstring still described `tj hook cap-output`, `tj savings`, and an A/B harness that no longer existed. Only `hooks_dir` was still consumed (by `core/recommendations.py`). The durable rule: after deleting a feature, grep every function in its supporting modules for callers before deciding what happens to the module, and treat a docstring that still describes deleted commands as a live defect, not a stale comment — it actively misleads the next reader. A helper module can legitimately outlive the feature it was built for, but only if it's trimmed to what's still consumed and re-documented for its remaining consumer.
43. **Installed is not resident, and neither gate is visible from the filesystem.** A plugin's skills reach the model only when its key is true in `enabledPlugins` (`~/.claude/settings.json`) AND its install scope in `plugins/installed_plugins.json` covers the session — so a directory tree full of `SKILL.md` can contribute nothing at all. Read both keys; never infer residency from what is on disk, and derive any current count from them rather than from a figure written down anywhere. Price only the `name: description` listing an enabled plugin publishes: skill BODIES arrive on invocation, so counting them answers a different question. That WHICH-PART-of-a-loaded-file axis is `core/summarize/load_semantics.py`'s; this rule is the prior one — whether the file loads at all — and the two compose rather than overlap. Same reason plugin paths stay out of the summarize catalog (`core/summarize/agent_files.toml`): they are third-party files under a versioned cache path, so the next plugin update reverts any edit and a fix offered there regresses silently.

44. **When a surface caps or collapses its output, the escape-hatch command it prints becomes load-bearing — pin every advertised command as INVOCABLE, never merely as present.** Reading a flag off `ctx.obj` is not the same as the subcommand accepting it, so declare the flag where the printed string types it. A capped view whose next-step command errors is worse than the uncapped output it replaced, because the user can no longer reach the data at all; `tests/unit/test_advertised_commands_are_invocable.py` parses the commands out of the RENDERED screens and fails unless each resolves against the real command tree, so extend it whenever a screen prints a new one.

45. **A broad `except Exception` that means "log it and carry on" must ask whether the error was FATAL before it carries on — a DuckDB `FatalException` invalidates the whole database instance, not the connection that raised it.** Every handler written to absorb a per-record failure ("one row lost, use the in-memory view", "one analyzer must not sink the rest", "never crash a background thread") is correct for the errors it was written for and catastrophic for this one, and the difference is invisible at the catch site. After a fatal there are no more rows to skip, only queries that will all fail — so the process keeps serving traffic on a database it can no longer read.
   (a) **The invalidation is per-DATABASE**, so a background job's "own" backend is not isolation — it shares the process's instance with the request path. `duckdb.connect(path)` while ANY handle survives returns the SAME dead instance from DuckDB's per-path cache, so reconnecting without closing recovers nothing; close every connection first and recovery works in-process with no restart.
   (b) **Do not rely on the exception REACHING your handler.** A fatal from an analyzer's write crosses several broad handlers on the way out, and any one of them absorbs it — patching each is a game you lose to the next handler someone writes. Record the fatal where it is RECOGNISED (`note_fatal_db_error`) and have long-running jobs recover off that process-wide record in a `finally`, so recovery is independent of who swallowed what. Classifying at the handlers is still worth doing; it is just not sufficient.
   (c) **A recovery must not itself hand out dead handles.** Hold the connection lock across teardown AND reopen, or a thread arriving in between gets a cursor from a closed connection — the exact failure the recovery exists to prevent.
   **Corollary for surfaces:** a liveness probe that never touches storage cannot distinguish "serving" from "up but unable to read anything", so `/health` must query the database and report unhealthy rather than green.

46. **A check that asks the same question two ways establishes nothing until you prove the two ways take different paths — and nothing beyond the cases it actually compared.** Two distinct failure modes, both of which shipped here before being caught, and both of which look like a green check.
   (a) **Wrong comparison form.** `CAST(col AS VARCHAR)` is a no-op on a column already stored as VARCHAR, so the planner discards it and the index under test serves BOTH sides — the probe compares a damaged index against itself and reports sound whatever the damage. Use `CAST(col AS VARCHAR) || ''`, a real expression for every type; a `GROUP BY` aggregate is the independent third opinion when you need certainty. Before trusting agreement between two forms, confirm they cannot both be answered by the thing being tested.
   (b) **Partial coverage read as proof.** A probe that compares sampled VALUES can only find damage in the entries it looked at. A three-value sample found three of four damaged indexes; the fourth reported clean, and a repair driven by that verdict left the table still raising the fatal on the next write. So report what you PROVED: sound only when every case was compared, and everything else in an explicit not-proven category rather than rounded up to passing. **Any caller that must be correct rather than cheap repairs everything instead of trusting the verdict** — here the full rebuild costs ~1s against a fault that 500s every route, so there was never anything to optimise. Damaged DuckDB ART indexes make reads silently under-report as well as making writes fatal, which is why a blind or over-confident probe here hides wrong analyzer numbers, not just a pending crash.

### The `.claude/rules/` index — every other rule, and every area's architecture

Each file below (paths relative to `.claude/rules/`) loads only when you touch a file its `paths:`
frontmatter matches, and carries both that area's architecture notes and the Critical Rules governing
it. **If you are chasing a rule by number, this table is the index** — the rule keeps its number
inside the file. Rows with no rule number are architecture-only.

| Rules | File | Covers · loads when you touch |
|---|---|---|
| — | `core-architecture.md` | data flow, post-ingest hooks, session continuity, the top-level core modules, `StorageBackend` parity · the `tokenjam/core/*.py` modules it documents, plus `core/export/**`, `core/ingest_adapters/**` |
| — | `core-config-pricing.md` | config discovery + precedence, the pricing engine, plan tiers, pricing modes · `core/config.py`, `core/pricing.py`, `tokenjam/pricing/models.toml` |
| 2 | `config-toml.md` | TOML read/write discipline · `core/config.py`, `core/pricing.py`, onboarding/policy/pricing CLI, any `*.toml` |
| 3, 12 | `sdk.md` | `@watch()`, attribution, transport, bootstrap, provider/framework integrations · `tokenjam/sdk/**`, `examples/single_{provider,framework}/**` |
| 4 | `api.md` | app factory, auth layers, per-route behaviour, concurrency, and the MCP stdio server (an SDK/API surface, not a Claude Code one) · `tokenjam/api/**`, `tokenjam/mcp/**`, `cli/cmd_serve.py` |
| 5 | `alerts.md` | alert dispatch, captured-content stripping · `core/alerts.py`, `core/ingest.py`, `core/drift.py`, `core/schema_validator.py`, `api/routes/alerts.py` |
| 8, 11 | `tests.md` | span factories, OTel provider setup, `~/.tj` isolation, `FORCE_COLOR` · `tests/**` |
| 10 | `otel.md` | exporter, OTLP parsing (one home), semconv constants · `tokenjam/otel/**`, ingest + span/log routes, `sdk/integrations/**` |
| 15 | `release.md` | release, packaging and CI detail · `pyproject.toml`, `sdk-ts/package.json`, `npm-wrapper/**`, `scripts/release.sh`, `Makefile`, `.github/workflows/**` |
| — | `optimize-architecture.md` | the analyzer package, registry strings vs file names, the dollar-field contract, write budget, rule placement, product pages · `core/optimize/**`, `core/fixes/**`, `cli/cmd_optimize.py` |
| 16, 19, 26, 29, 31, 39, 40 | `optimize-analyzers.md` | analyzer authoring + gating, the prompt-summarization lifecycle and load semantics, the shared rule-write stage/apply/undo lifecycle · `core/optimize/analyzers/**`, `core/optimize/{registry,runner,cost_proposals}.py`, `core/summarize/**`, `core/rulewrite/**` |
| 18 | `web-ui.md` | the Lens single-file SPA, spacing tokens, charts, polling, UI testing · `tokenjam/ui/**`, `api/app.py`, `tests/unit/test_ui_offline.py` |
| 21 | `onboarding-dotfiles.md` | managed-block dotfile writes · `cli/cmd_{onboard,uninstall,stop,statusline}.py` |
| 22, 27, 28, 30, 32, 41 | `optimize-cost-figures.md` | dollar/token figure discipline · `core/optimize/**`, `core/framing.py`, `core/cost.py`, `core/fixes/**`, cost/optimize routes + CLI |
| 33, 34, 36, 37, 38 | `ingest-accounting.md` | `core/backfill.py`, `core/transcript*.py`, `core/ingest*`, `core/db.py`, `core/optimize/accounting.py`, `cli/cmd_{backfill,doctor}.py` |
| 35 | `cli-output.md` | every non-obvious command, `no_db_commands`, the data-access seam, the daemon, Codex, terminal-output discipline · `tokenjam/cli/**`, `utils/formatting.py`, `utils/theme.py`, `tokenjam/demo/**` |

Two more path-scoped notes with no rule number: `growth-instrumentation.md` (the weekly traffic
archive, `.github/workflows/traffic-archive.yml` + `growth/**`) and `examples-and-incidents.md`
(`examples/**`, `incidents/**`).

## Further Reading

- [`docs/architecture.md`](docs/architecture.md) — design principles, system overview, SDK internals, alerts, drift, MCP, budget, testing architecture, and the **OTel semconv extensions** section (`tokenjam.billing_account`, `tokenjam.plan_tier`, the `pricing_mode` derivation rules, and why `plan_tier` lives on `SessionRecord` rather than each span).
- [`docs/installation.md`](docs/installation.md) — base install vs optional extras matrix.
- [`docs/configuration.md`](docs/configuration.md) — full TOML config surface, the four `[capture]` toggles, and pricing overrides.
- [`docs/optimize/`](docs/optimize/) — one product page per user-facing analyzer; indexed from `.claude/rules/optimize-architecture.md`.
- [`docs/backfill/overview.md`](docs/backfill/overview.md) — backfill sources with per-adapter modes, field mapping, idempotency, limitations. [`docs/policy/overview.md`](docs/policy/overview.md) — `tj policy list`.
- [`AGENTS.md`](AGENTS.md) — codebase conventions for contributors (referenced from the README).
- `docs/internal/specs/` — canonical specs that production code references long-term; add new ones here when a feature needs a stable, code-referenced source of truth.
- [`docs/internal/release-smoke-checklist.md`](docs/internal/release-smoke-checklist.md) — fresh-install pre-release gate (clean env → `pipx install` → `tj onboard --claude-code` → verify Lens plan badge + `tj optimize` agree → sane session count). Its automated counterpart is `tests/integration/test_first_run_roundtrip_239.py`.
