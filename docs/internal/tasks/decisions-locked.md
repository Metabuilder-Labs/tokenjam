# Decisions locked before dispatch

These come out of three rounds of code review. Every task in this queue assumes them. Do not relitigate inside a task; if something feels wrong, surface it back to Anil before coding.

## Data model

1. **`plan_tier` lives on `SessionRecord` as the canonical plan field.** Distinct from `billing_account`. Spans inherit through the session FK. When analyzers need plan tier, they JOIN through `SessionRecord`.

2. **`billing_account` is provider-only**, not composite. Values: `anthropic`, `openai`, `google`, `bedrock`, `local.ollama`. Does NOT encode plan tier. Plan tier lives on `SessionRecord.plan_tier` only.

3. **`pricing_mode` is a derived Python property on `SessionRecord`**, not a stored column. Branches evaluated **top-to-bottom; first match wins**:
   1. `local` if `billing_account == 'local.ollama'`
   2. `subscription` if `plan_tier in {pro, max_5x, max_20x, plus, team, enterprise}`
   3. `api` if `plan_tier == 'api'`
   4. `unknown` if `plan_tier == 'unknown'`

4. **`ProviderBudget.plan` vs `SessionRecord.plan_tier`:** `ProviderBudget.plan` is the user's declared plan per provider, written by `tj onboard`. `SessionRecord.plan_tier` is set at session creation by reading the relevant `ProviderBudget.plan` for that session's `billing_account`. One source of truth (config), one consumer (session field).

5. **Backfilled sessions default `plan_tier = unknown`.** No silent default. `tj status` surfaces a one-line note when unknown sessions exist. `tj optimize` refuses to render dollar figures for unknown-tier sessions until resolved.

6. **No API-key fingerprint in `billing_account`.** Multi-key disambiguation deferred until someone asks.

## Optimize package & analyzers

7. **`--finding` valid choices are registry-driven.** Click decorator reads from `REGISTRY.keys()` at decoration time. Analyzer tasks don't edit `cmd_optimize.py`; they register themselves in the runner.

8. **Analyzer registration uses auto-discovery, not manual `__init__.py` edits.** `analyzers/__init__.py` walks the directory at import time and imports every `.py` file. Wave 2 agents touch nothing in `__init__.py`.

9. **Workflow-restructure signature definition.** Tool names + structural argument shape (which kinds of args are present: `file_path`, `command_string`, `json_object`, `string`, `number`, `boolean`, `array`), with argument *values* excluded. Conservative thresholds (≥20 instances, 100% pattern stability, zero branching) make this work.

10. **Cache analyzer scope.** `cache-efficacy` ships Anthropic-fully-supported plus best-effort for OpenAI/Google with explicit per-provider documentation. `cache-recommend` is Anthropic-only in v1, documented clearly.

## CLI behavior

11. **No `--apply` flag on `tj optimize export-config`.** Writes a snippet file. User copies manually. TokenJam doesn't sit in the call path.

12. **`tj status` stays non-interactive.** Exit codes preserved for scripts and CI. Any plan-tier issue surfaces as a printed note, never a prompt. Reconfiguration happens only via `tj onboard --reconfigure`.

13. **No new `tj reconfigure` top-level command.** `--reconfigure` is a flag on the existing `tj onboard` command.

14. **No shared `claude_settings.py` writer module.** With `--apply` dropped, the original reason is gone. Onboard handles its own writes.

## Dependencies

15. **`llmlingua` is an optional extra**, not a hard dependency. Installed via `pip install "tokenjam[bloat]"`. Trim analyzer imports inside the function body and fails with a clear message if the extra isn't present.

16. **`pyproject.toml` is the build config.** This repo uses hatchling. There is no `setup.py`.

## Process

17. **No backwards-compatibility migration logic.** Product has no external users yet. New DB columns can be `ALTER TABLE ADD COLUMN ... DEFAULT NULL`; no need to backfill heuristics into existing rows. The dev's local DB will get sane defaults from regular use after the schema lands.

18. **Realistic sprint length: 10–12 days.** Each wave's PRs must merge before the next wave's PRs open.

19. **Where this queue and `docs/internal/specs/v1.1-honest-output.md` differ on rendering language, the embedded spec is canonical.** That spec is committed verbatim by Task 0.5.

## Content stripping (when `capture.include_content = false`)

The following attribute keys are stripped from `spans.attrs` before INSERT (in `IngestPipeline.process()`):

- `prompt_content`
- `completion_content`
- `tool_input`
- `tool_output`
- `gen_ai.tool.output`

Both `tool_output` and `gen_ai.tool.output` are stripped intentionally — the alert system and the schema validator use different naming conventions. Belt-and-suspenders.

Token counts, tool names, model names, timestamps, and structural metadata are unaffected.

## General conventions

- Read `CLAUDE.md` at repo root before starting. Critical rules: DuckDB only, parameterised SQL only, `utcnow()` for timestamps, semconv constants, test factories, `tokenjam/core/` must not import from `cli` or `api`.
- Each task gets its own feature branch off `main` and its own PR.
- Tests required for every new analyzer, adapter, or core change. Use existing test factories.
- Update `CHANGELOG.md` under `## Unreleased` for each task.
- Open a draft PR early so Anil can sanity-check direction before final review.

## Coordination

- **`CHANGELOG.md`**: every PR adds entries under `## Unreleased`. Trivial conflicts; last-in rebases.
- **`cmd_optimize.py`**: registry-driven `--finding` choices means analyzer tasks don't touch it. No conflict surface here.
- **`cmd_backfill.py`**: Tasks 1.2, 3.1, 3.2 each register a subcommand. Different subcommand names mean trivial conflict.
- **No two tasks may modify the same analyzer module simultaneously.** With Task 0's package structure, each analyzer lives in its own file.
