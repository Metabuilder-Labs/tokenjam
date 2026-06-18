# Contributing to TokenJam

TokenJam is MIT licensed and welcomes contributions. It's built by AI coding agents, and contributing with one is first-class — see [Using coding agents](#using-coding-agents) below.

## Good first contributions

New here? These are the lowest-friction ways to land your first PR:

- 🟢 **[Good first issues](https://github.com/Metabuilder-Labs/tokenjam/labels/good%20first%20issue)** — scoped tasks tagged ready for newcomers.
- 💸 **Model pricing** — add a missing model or correct a rate in [`tokenjam/pricing/models.toml`](tokenjam/pricing/models.toml). One file, no issue needed (details below).
- 🔌 **A framework/provider integration** — they follow one clear pattern; `tokenjam/sdk/integrations/anthropic.py` is the reference. Open an issue first to align on approach.

Questions on any of these? Open an issue — we're happy to point you at a good starting spot.

## Claiming an issue

Found an issue you want to work on? **Assign it to yourself** so others know it's taken and nobody duplicates the work. If you don't have permission to self-assign (common if you're not yet a repo collaborator), just leave a quick comment saying you're on it and a maintainer will assign you. If an issue is already assigned — or has a recent "I'll take this" comment — pick another one.

## Getting started

```bash
git clone https://github.com/Metabuilder-Labs/tokenjam
cd tokenjam
pip install -e ".[dev]"   # editable install with dev tools (the MCP server ships in the base install)
pip install anthropic     # for running the toy agent
```

## Running tests

The suite is layered — fastest first. CI runs everything except `e2e`, and **none of these need an API key**:

```bash
pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/
```

- `tests/unit/` — pure logic, no I/O (runs in well under a second)
- `tests/synthetic/` — spans injected via `tests/factories.py`, zero cost
- `tests/agents/` — mock agent scenarios through the full SDK path
- `tests/integration/` — CLI + HTTP API

While iterating, run just one file — or one test:

```bash
pytest tests/unit/test_config.py
pytest tests/unit/test_config.py::test_function_name -v
```

Lint and type-check (both run in CI — please run them before pushing):

```bash
ruff check tokenjam/
mypy tokenjam/
```

The TypeScript SDK is an independent package with its own tests — run these only if you touch `sdk-ts/`:

```bash
cd sdk-ts && npm install && npm test
```

E2e tests hit a real API and **auto-skip unless `TJ_ANTHROPIC_API_KEY` is set** (a full run costs fractions of a cent):

```bash
export TJ_ANTHROPIC_API_KEY="sk-ant-..."
pytest tests/e2e/
```

## Branches and commits

- Branch off `main` with a type-prefixed, kebab-case name: `fix/...`, `feat/...`, `docs/...`, or `chore/...` (e.g. `fix/budget-json-flag`).
- Keep each PR to **one concern** — one feature or fix. If you spot an unrelated problem, open a separate issue rather than expanding the PR.
- Write commit subjects in the imperative ("Add budget `--json` flag"), and reference the issue with `#N` where relevant. Put `Closes #N` in the PR description so the issue auto-closes on merge.

## Before opening a PR

- Run the test suite and make sure it passes: `pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/`
- Run `ruff check tokenjam/` **and** `mypy tokenjam/`, and fix anything they flag
- Keep PRs focused — one feature or fix per PR
- If you're adding a framework integration, open an issue first so the approach can be aligned on before you write code — integrations have a specific pattern (see `tokenjam/sdk/integrations/anthropic.py` as the reference implementation)
- When it's ready, **request `@anilmurty` as the reviewer** (if you can't request reviewers, just @-mention him in the PR description) — that's the signal it's ready for a look
- Make sure CI is green on your branch before requesting review

## Project structure

```
tokenjam/core/              Domain logic — no CLI or HTTP imports allowed here
tokenjam/cli/               Click commands — one file per command
tokenjam/api/               FastAPI routes
tokenjam/sdk/               @watch() decorator and provider/framework patches
tokenjam/otel/              OTel TracerProvider and span exporter wiring
tokenjam/utils/             Formatting, time parsing, ID generation
sdk-ts/src/            TypeScript SDK (@tokenjam/sdk)
tokenjam/pricing/models.toml  Community-maintained model pricing — PRs welcome here
tests/factories.py     Span factory — use this in all synthetic tests, never
                       construct NormalizedSpan directly
```

## Using coding agents

TokenJam is built by AI coding agents, and contributing with one is first-class:

- **Claude Code** — it reads [`CLAUDE.md`](CLAUDE.md) automatically; run `/init` to bring your agent up to speed on the architecture and conventions.
- **Codex / Gemini / other agents** — read [`AGENTS.md`](AGENTS.md): the critical rules (DuckDB not SQLite, TOML not YAML, no CLI imports in `tokenjam/core/`, etc.) plus a pointer to CLAUDE.md for the full guide.

## Pricing table contributions

The file `tokenjam/pricing/models.toml` is intentionally community-maintained. If a model is missing or prices have changed, open a PR with the update — no issue needed, just update the TOML and verify the format matches existing entries. (This is the file the cost engine loads at runtime; there is no separate repo-root copy.)

## Reporting issues

Use GitHub Issues. For bugs, include:

- Python version (`python3 --version`)
- OS
- The command you ran
- The full error output
- Output of `tj doctor` if relevant
