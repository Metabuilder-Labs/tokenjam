# AGENTS.md — TokenJam

> **AI coding agents working in this repo should read [`CLAUDE.md`](./CLAUDE.md) instead.**
> CLAUDE.md is the maintained source of truth and is kept up to date by the agents
> working in this codebase. This file exists only for tools that look specifically
> for AGENTS.md (Codex, Gemini Code, etc.).

## Quick rules for agents that don't read CLAUDE.md

1. Storage is **DuckDB** — never `import sqlite3`, never write SQLite queries
2. Config is **TOML** — `tomllib.load()` requires `open(path, "rb")` binary mode
3. `tj/core/` has **no CLI or HTTP imports** — pure domain logic only
4. `@watch()` alone does **not** create LLM spans — requires `patch_anthropic()` etc.
5. Never use unicode bullet characters in output — Rich handles formatting
6. Tests use `InMemoryBackend` and `tests/factories.py` — never construct `NormalizedSpan` directly

**Read [CLAUDE.md](./CLAUDE.md) for the full architecture, development guide, and current rules.**