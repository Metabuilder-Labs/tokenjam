# SDK workload corpus

tokenjam has a large Claude Code corpus and essentially zero SDK data. This
directory builds the missing ground truth: small, runnable, multi-turn
OpenAI-API agent workloads instrumented with tokenjam's own SDK, each
deliberately exhibiting one waste shape `tj optimize` claims to detect,
plus a harness that runs one, ingests the telemetry into a scratch DB, and
reports which analyzers actually fired.

This is a measurement tool, not a demo you run once and forget. Its whole
point is to make "does this analyzer work for SDK users" answerable from
real (or realistically-shaped) data instead of guesswork.

## 0. Install

```bash
pip install -e ".[dev]"   # tokenjam itself, from the repo root
pip install openai        # not a tokenjam dependency; these workloads need it directly
```

`--dry-run` never imports `openai` at all, so it works even without that
second install.

## 1. Set your key

```bash
export OPENAI_API_KEY=sk-...
```

Read from the environment only — never hardcoded, never logged, never
written to disk by anything in this directory. Every workload refuses to
run for real without it (with a clear error) unless you pass `--dry-run`,
which never touches the key or the network at all. See `.env.example` at
the repo root.

## 2. Run a workload

Through the harness (recommended — this is what builds the report):

```bash
python examples/sdk_workloads/runner.py --list
python examples/sdk_workloads/runner.py oversized-model --dry-run       # zero cost, sanity check
python examples/sdk_workloads/runner.py oversized-model                 # real, $2.00 default ceiling
python examples/sdk_workloads/runner.py oversized-model --max-spend 0.50 --model gpt-4o --out report.json
```

Or run a workload script directly (useful while iterating on one, but you
won't get the analyzer report — see step 3):

```bash
python examples/sdk_workloads/oversized_model.py --dry-run
python examples/sdk_workloads/oversized_model.py --max-spend 0.50
```

Every workload and the runner share the same flags:

| Flag | Default | Meaning |
|---|---|---|
| `--dry-run` | off | Zero API calls, zero spend — stubbed client, canned deterministic responses, but the SAME real tokenjam span pipeline (DuckDB, cost engine, optimize analyzers) runs on the result. |
| `--max-spend` | `$2.00` | Hard USD ceiling. Checked BEFORE every real API call using a conservative pre-call cost estimate; the call never happens if it would cross the ceiling. Raise it explicitly if a run needs more. |
| `--model` | workload-specific (`gpt-5.4-mini` for most — the current cheap mini-class model; `gpt-4o` for `oversized-model`, deliberately the legacy premium model `downsize`'s candidate mapping knows about) | Any OpenAI chat-completions model. |

`tool-heavy-chain` also takes `--repeat N` (default 1) to run N independent
sessions with the identical tool-call signature — see its own docstring
for why (the `script` analyzer needs volume a single cheap run can't
provide).

The runner forwards unrecognized flags straight to the workload:

```bash
python examples/sdk_workloads/runner.py tool-heavy-chain --dry-run -- --repeat 20
```

## 3. Read the output

The runner prints, in order:

1. The workload's own live output (each call, each tool step).
2. `[spend-guard] cumulative spend: $X.XXXX across N call(s)` — the
   workload's own running total.
3. A table: every registered `tj optimize` analyzer, whether it ran or was
   skipped (persona-gated for this window), whether it produced a finding,
   and a one-line detail (candidate/example count, `past_overspend_usd`
   when the analyzer carries one).
4. A one-line, source-verified note per analyzer explaining WHY it did or
   didn't fire — read this before trusting a "no finding" row; several
   analyzers need session/call volume no single cheap run provides (noted
   inline, e.g. `script` needs 20+ sessions).
5. Alerts fired in the window (e.g. `retry_loop`), separate from the
   optimize analyzers.
6. `Actual spend recorded by tj's own cost engine` — read directly from
   the scratch DB's `SUM(cost_usd)`, independent of the workload's own
   guard bookkeeping, as a cross-check.
7. The scratch DB path, left on disk for manual inspection
   (`tj status --db <path>`, `duckdb <path>`, etc.) — never auto-deleted.

Pass `--out report.json` to also get the full machine-readable report
(the same shape `report_to_dict()` produces) plus the alert list and the
DB-measured actual spend, for scripted before/after diffing later.

## 4. The workloads

| Workload | Waste shape | Targets | Cost (default model, default settings) |
|---|---|---|---|
| `repeated_prefix.py` | Every call shares one ~5,000-token system prompt, 25 calls. | `cache` analyzer. `cache-recommend` is Anthropic-only and never fires here — see caveat below. | ~$0.10 |
| `growing_context.py` | 3 sessions x 4 turns, each turn re-sending the full accumulated history. | `resend` analyzer. | ~$0.01 |
| `retry_loop.py` | Same tool call retried 5x with byte-identical arguments after a deterministic failure. | `RETRY_LOOP` alert (AlertEngine, not an optimize analyzer). | ~$0.001 |
| `tool_heavy_chain.py` | 9 deterministic tool calls per session (search/fetch/extract/format/save), 2 LLM calls. | Tool-span capture. `script` analyzer needs `--repeat 20+` to clear its volume gate. | ~$0.003 x `--repeat` |
| `oversized_model.py` | `gpt-4o` used for one-word yes/no answers. | `downsize` analyzer (fires on a single qualifying session). | ~$0.0004 |
| `streaming_disconnect.py` | A streaming response abandoned (`generator.close()`) before the usage-bearing final chunk. | Nothing today — demonstrates a real gap in tokenjam's own OpenAI SDK integration. | ~$0.001 |

All well inside the default `--max-spend $2.00` ceiling, even added together.

Every workload's module docstring is the authoritative source for its
exact gate math (thresholds, session counts, token volumes) — read it
before assuming the default run will or won't fire a given analyzer.

Every workload tags its spans with the SDK cost-attribution dimensions
(`tenant_id`, `feature`, `prompt_template_id`/`_version` via
`tokenjam.sdk.attribution`, and `environment` via the
`OTEL_RESOURCE_ATTRIBUTES` process env var) with a distinct, clearly
fictional tenant per workload (Acme, Globex, Initech, Umbrella, Stark,
Wayne), so the corpus also exercises those columns in the Cost view.

## 5. Known caveats when reading the report

- **`cache` "fires" on OpenAI for the wrong reason.** tokenjam's OpenAI
  integration (`tokenjam/sdk/integrations/openai.py`) never reads
  `response.usage.prompt_tokens_details.cached_tokens` — only
  `INPUT_TOKENS`/`OUTPUT_TOKENS` are set. So `cache_tokens` is always
  0/unset for every OpenAI span this SDK has ever captured, and the
  `cache` analyzer's "efficacy" reads as 0% regardless of whether OpenAI's
  own automatic prompt caching was actually hitting. A fired card here is
  evidence of an instrumentation gap, not a real caching problem.
- **`cache-recommend` never fires on pure OpenAI telemetry.** It's
  Anthropic-only in v1 and skips every non-Anthropic span outright,
  independent of prefix stability or call volume.
- **`summarize` and `relearn` don't scope to this workload's telemetry.**
  `summarize` scans on-disk prompt files at the harness's CWD (the
  tokenjam repo itself, when run from the repo root) — a fired card
  reflects the repo's own docs, not anything a workload wrote.  `relearn`
  deliberately ignores the report window and scans the whole scratch DB's
  retention period, so its count reflects everything written to that DB
  so far, not just the run you just watched.
- **`subagent` and `deadweight` are structurally unreachable for SDK
  telemetry**, not just under-triggered: `runner.PERSONA_DISABLED_ANALYZERS`
  drops both from dispatch entirely for the `sdk` persona (no
  `sub_agent_id` concept in generic SDK spans; no on-disk Claude Code
  transcript to read).

## 6. Isolation and safety

- **Scratch home, scratch DB, every run.** The runner writes a throwaway
  `tj.toml` (pointing `[storage] path` at a fresh `telemetry.duckdb`
  under a `tempfile.mkdtemp()` directory) and passes it to the workload
  subprocess via `TJ_CONFIG` (plus `HOME` override as defense in depth).
  Your real `~/.tj` is never opened, read, or written by anything here.
- **A fresh Python process per workload run.** tokenjam's `TracerProvider`
  is a process-global, set-once singleton — reusing one interpreter
  across workloads would silently pin every later run to the first run's
  scratch config. The runner always spawns a subprocess; don't try to
  call a workload's `run()` function from a long-lived process alongside
  another tokenjam SDK user.
- **The spend guard is enforced BEFORE every call**, using a conservative
  pre-call estimate (worst-case `max_tokens` output, chars/4 input
  estimate, tokenjam's own pricing table) — never after the fact. A run
  that would cross the ceiling aborts with a clear message and exit code
  2; the runner still builds a report from whatever telemetry landed
  before the abort.
- **Determinism for a future before/after replay.** Real calls use
  `temperature=0` and a fixed `seed`; every local tool function is a pure,
  deterministic lookup (no randomness, no wall-clock-dependent output);
  `--dry-run` responses are fully canned. The harness doesn't implement
  fix-application/replay diffing yet, but nothing here blocks a future
  script from running a workload, applying a fix, re-running the
  identical workload, and diffing `report.json` / `SUM(cost_usd)` between
  the two runs — `--out` already gives you the JSON to diff.

## 7. Tests

```bash
pytest tests/integration/test_sdk_workload_corpus.py -v
```

Runs entirely with `--dry-run` — zero API calls, `OPENAI_API_KEY` is never
required. Covers: the spend guard actually blocking an over-ceiling call,
the missing-key error path, a dry-run workload producing real spans in a
scratch DB, the retry-loop alert firing, and the runner's report shape
(table + `--out` JSON).
