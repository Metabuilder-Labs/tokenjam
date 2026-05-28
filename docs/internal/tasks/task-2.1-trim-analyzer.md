# Task 2.1 — Trim analyzer (internal: prompt-bloat)

**Wave 2. Dispatch after Wave 1 fully merges. Runs in parallel with Tasks 2.2 and 2.3.**

## Prerequisites

- Read `CLAUDE.md`, `docs/internal/tasks/decisions-locked.md`.

## Summary

Second of four optimize analyzers. Use LLMLingua-2 (BERT-class, runs on CPU) to score token significance in user prompts; identify bloat regions; render an HTML report.

User-facing product name: **Trim**. Internal/CLI name: **`prompt-bloat`**.

## Dependency handling (locked decision)

- **`llmlingua` is an optional extra.** Defined in `pyproject.toml` as `tokenjam[bloat]`. Base `pip install tokenjam` does NOT pull torch.
- The analyzer imports `llmlingua` **inside the analysis function body** (not at module top). On `ImportError`, fail with this message:
  ```
  Trim analyzer requires extra dependencies.

  Install with: pip install "tokenjam[bloat]"

  (This pulls in PyTorch and transformers, ~2GB. Optional because most
  users don't need this analyzer.)
  ```
- The analyzer **self-registers in the runner registry even without the extra installed**; the import error fires only when the analysis function actually runs. This way the `--finding` choices list always includes `prompt-bloat` and users discover it naturally.

## Capture dependency

- Requires `capture.include_content: true` (config key from Task 0).
- If not set, the analyzer exits with a clear message pointing the user at how to enable it.

## LLMLingua-2 model handling

- Model downloads on first analyzer use, not on package install. Cache to `~/.cache/tokenjam/models/llmlingua-2-bert/`.
- Download is explicit. Before the download, print: `Downloading LLMLingua-2 model (~110MB on first use)...`
- Offline behavior: if no cached model and no network, fail with a clear message rather than crashing.
- `tj serve` does NOT download the model. Only loaded when an analysis is actually requested.
- **Benchmark on an M1 MacBook before merging.** Document typical latency for a 4K-token prompt in `docs/optimize/trim.md`.

## Scope

- New analyzer module: `tokenjam/core/optimize/analyzers/prompt_bloat.py` with `@register("prompt-bloat")` decoration.
- HTML report rendering: new CLI command `tj report --bloat <agent_id>` that opens a local HTML file showing high-significance tokens bold, low-significance tokens dimmed. Use `webbrowser.open()` on the local file.
- JSON output in `--json` mode.
- Confidence level: `structural` only (no replay validation for this analyzer).
- Update `pyproject.toml` with the `tokenjam[bloat]` extras definition.

## Files touched

- New: `tokenjam/core/optimize/analyzers/prompt_bloat.py`
- New: `tokenjam/cli/cmd_report.py`
- `tokenjam/cli/main.py` (register `cmd_report`)
- `pyproject.toml` (add `[project.optional-dependencies] bloat = ["llmlingua>=..."]`)
- New: `tests/unit/test_prompt_bloat.py` — **mock the LLMLingua-2 model so CI doesn't download 110MB.** Patch the import at the function-body site.
- `docs/optimize/trim.md` (includes benchmark numbers and the product/internal name table)
- `docs/installation.md` (document `tokenjam[bloat]` extra and what it pulls in)
- `CHANGELOG.md`

## Coordination

- Self-registers via `@register` decoration; auto-discovery in `analyzers/__init__.py` picks it up. **Do not edit `analyzers/__init__.py`, `cmd_optimize.py`, or any other analyzer file.**
- New `cmd_report.py` registered in `cli/main.py` — Wave 2 tasks 2.2 and 2.3 don't touch `main.py`. Small rebase if conflicts.

## Done-when

- `pip install tokenjam` does NOT pull torch (verify by checking installed packages in a clean venv).
- `pip install "tokenjam[bloat]"` does pull torch.
- `tj optimize --finding prompt-bloat` is listed as an option even without the extra installed.
- Running it without the extra produces the friendly ImportError message.
- Running it without `capture.include_content = true` produces a clear "enable content capture" message.
- Running it with both prerequisites met on a project with captured content produces a bloat finding.
- `tj report --bloat <agent_id>` opens an HTML visualization in the user's browser.
- Benchmark numbers for a 4K-token prompt on an M1 MacBook are documented in `docs/optimize/trim.md`.
- Tests pass without downloading the model (mock).
