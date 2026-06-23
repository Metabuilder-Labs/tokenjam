# Contributing to TokenJam

TokenJam is MIT licensed and welcomes contributions. The codebase was built using parallel Claude Code agents — the `.claude/` task files are intentionally committed so contributors can use the same workflow.

## Good first contributions

New here? These are the lowest-friction ways to land your first PR:

- 🟢 **[Good first issues](https://github.com/Metabuilder-Labs/tokenjam/labels/good%20first%20issue)** — scoped tasks tagged ready for newcomers.
- 💸 **Model pricing** — add a missing model or correct a rate in [`tokenjam/pricing/models.toml`](tokenjam/pricing/models.toml). One file, no issue needed (details below).
- 🔌 **A framework/provider integration** — they follow one clear pattern; `tokenjam/sdk/integrations/anthropic.py` is the reference. Open an issue first to align on approach.

Questions on any of these? Open an issue — we're happy to point you at a good starting spot.

## Getting started

```bash
git clone https://github.com/Metabuilder-Labs/tokenjam
cd tokenjam
pip install -e ".[dev]"   # editable install with dev tools (the MCP server ships in the base install)
pip install anthropic     # for running the toy agent
```

## Running tests

```bash
# All non-e2e tests (what CI runs)
pytest tests/unit/ tests/synthetic/ tests/agents/ tests/integration/

# Linting and type checking
ruff check tokenjam/
mypy tokenjam/

# E2e tests (requires real API key — costs fractions of a cent)
export TJ_ANTHROPIC_API_KEY="sk-ant-..."
pytest tests/e2e/
```

## Before opening a PR

- Run the full test suite and ensure it passes
- Run `ruff check tokenjam/` and fix any lint errors
- If you're adding a framework integration, open an issue first so the approach can be aligned on before you write code — integrations have a specific pattern (see `tokenjam/sdk/integrations/anthropic.py` as the reference implementation)
- Keep PRs focused — one feature or fix per PR

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

## Using Claude Code

This project was built using parallel Claude Code agents. The `.claude/` directory contains the original task files. If you're using Claude Code to contribute:

- Read `AGENTS.md` at the repo root before starting — it contains the critical rules (DuckDB not SQLite, TOML not YAML, no CLI imports in `tokenjam/core/`, etc.)
- The task files in `.claude/` show how the codebase was structured and are useful context for larger contributions

## Pricing table contributions

The file `tokenjam/pricing/models.toml` is intentionally community-maintained. If a model is missing or prices have changed, open a PR with the update — no issue needed, just update the TOML and verify the format matches existing entries. (This is the file the cost engine loads at runtime; there is no separate repo-root copy.)

## Reporting issues

Use GitHub Issues. For bugs, include:

- Python version (`python3 --version`)
- OS
- The command you ran
- The full error output
- Output of `tj doctor` if relevant
