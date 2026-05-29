# Installation

TokenJam ships as a Python package on PyPI and a TypeScript SDK on npm. Pick the install that matches what you need.

## Base install

```bash
pip install tokenjam
```

This is enough for the CLI (`tj`), local REST API (`tj serve`), the four out-of-box optimize analyzers that don't need ML models, and every native SDK integration except LLMLingua-based Trim. Requires Python ≥ 3.10.

After install, run:

```bash
tj onboard            # generic
tj onboard --claude-code
tj onboard --codex
```

to generate a config, optionally wire up Claude Code or Codex telemetry, and start the background daemon. See [`docs/configuration.md`](configuration.md) for the full config surface.

## Optional extras

TokenJam keeps heavyweight ML dependencies, framework adapters, and the MCP server out of the base install. Add them when you need them:

| Extra | What it pulls in | Why it's optional |
|---|---|---|
| `tokenjam[mcp]` | `fastmcp` | Only needed for the Claude Code / Codex MCP server (`tj mcp`). Pulled by `tj onboard --claude-code` automatically when invoked through the documented one-liner. |
| `tokenjam[bloat]` | `llmlingua>=0.2`, transitively PyTorch + transformers (~2GB) | The Trim analyzer (`tj optimize trim`) scores token significance with LLMLingua-2. Most users don't run it; keeping torch out of the base install means `pip install tokenjam` stays small and fast on machines that don't have a GPU/CPU build of torch already. |
| `tokenjam[langchain]` | `langchain>=0.2` | Convenience pin for `patch_langchain()`; you can also install langchain yourself. |
| `tokenjam[crewai]` | `crewai>=0.28` | Convenience pin for `patch_crewai()`. |
| `tokenjam[autogen]` | `pyautogen>=0.2` | Convenience pin for `patch_autogen()`. |
| `tokenjam[litellm]` | `litellm>=1.40` | Convenience pin for `patch_litellm()`. |
| `tokenjam[dev]` | `pytest`, `pytest-asyncio`, `httpx`, `ruff`, `mypy` | For working on TokenJam itself. |

Combine multiple extras:

```bash
pip install "tokenjam[mcp,bloat]"
```

### Bloat extra details

`tokenjam[bloat]` is the largest extra — LLMLingua-2 transitively pulls in PyTorch and Hugging Face transformers, roughly 2GB on disk. On first run the analyzer downloads a ~110MB BERT-class classifier model under `~/.cache/tokenjam/models/` (override via `TOKENJAM_MODEL_CACHE`); subsequent runs are offline-capable.

If you run `tj optimize trim` without the extra installed, the analyzer self-registers and exits with a clear hint pointing at this install command — nothing in the base install crashes from its absence.

See [`docs/optimize/trim.md`](optimize/trim.md) for performance numbers, capture requirements, and what the analyzer actually reports.

## TypeScript SDK

```bash
npm install @tokenjam/sdk
```

The TypeScript SDK is independent of the Python package. It emits OTLP spans over HTTP to `POST /api/v1/spans` on a running `tj serve`. See the SDK README under `sdk-ts/` for the full API.

## Verifying the install

```bash
tj doctor
```

`tj doctor` checks config validity, DB connectivity, ingest secret presence, daemon health, and (when applicable) alert-channel reachability. Exit code 0 = clean, 1 = warnings, 2 = errors.
