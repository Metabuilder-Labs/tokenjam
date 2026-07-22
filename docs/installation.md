# Installation

TokenJam ships as a Python package on PyPI and a TypeScript SDK on npm. Pick the install that matches what you need.

## Zero-install first run (no-commitment preview)

Want to look before you install? This prints a read-only report; **no install, no config, no
daemon**; over the logs you already have. The recommended way to actually run TokenJam is the base
install plus `tj onboard` (below):

```bash
npx tokenjam
```

Bare `npx tokenjam` reads your existing Claude Code sessions from
`~/.claude/projects/*.jsonl` (the same files [ccusage](https://github.com/ryoppippi/ccusage)
reads) into a throwaway in-memory database and prints your quota composition
(re-reading context vs. net-new work) plus a session timeline. Nothing is
written to disk and no background process starts.

**How the launcher resolves:** `npx tokenjam` runs the
[`tokenjam` npm wrapper](https://www.npmjs.com/package/tokenjam); a thin launcher
that shells out to the Python CLI via the first available runner (`uvx` → `pipx run` →
an installed `tj`). Any explicit args pass straight through, so `npx tokenjam optimize`,
`npx tokenjam onboard`, etc. all work; `npx tokenjam onboard` detects the ephemeral
runner and offers to convert itself into a permanent install. (The PyPI package
`tokenjam` also ships a `tj` command directly, so once installed you can run
`uvx --from tokenjam tj <command>` or an installed `tj <command>` the same way.)

**Requirements:** a Python runner; `uv` (recommended) or `pipx`. `npx tokenjam` prints
install guidance if neither is present.

When you're ready for live capture, the local dashboard, and the zero-token
statusline, install the full CLI (below) and run `tj onboard`.

## Base install

```bash
pipx install tokenjam
```

This is the recommended install path on **all platforms**. `pipx` automatically creates an isolated venv for the `tj` CLI, which means:

- It works on macOS with Homebrew Python (which refuses `pip install` into its managed environment by default; [PEP 668](https://peps.python.org/pep-0668/)).
- It works on Debian 12+ / Ubuntu 24+ (same PEP 668 enforcement).
- It doesn't pollute your system Python or any project's venv.

Don't have `pipx`? Install it with one of:

| Platform | Command |
|---|---|
| macOS | `brew install pipx` |
| Debian / Ubuntu | `apt install pipx` |
| Windows | `py -m pip install --user pipx` |
| Anywhere else | `python3 -m pip install --user pipx` |

Then ensure pipx's bin dir is on your `PATH` with `pipx ensurepath`.

### Alternative: pip in a venv

If you prefer plain pip (or need to install into an existing project venv):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install tokenjam
```

Either path is enough for the CLI (`tj`), local REST API (`tj serve`), the four out-of-box optimize analyzers that don't need ML models, and every native SDK integration except LLMLingua-based Trim. Requires Python ≥ 3.10.

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
| `tokenjam[mcp]` | nothing (no-op alias) | **No longer needed.** `fastmcp` moved into the base install in v0.3.5, so the Claude Code / Codex MCP server (`tj mcp`) works on a plain `pipx install tokenjam`. The extra is kept as an empty no-op so old `pipx install 'tokenjam[mcp]'` commands still succeed; it pulls nothing extra. |
| `tokenjam[bloat]` | `llmlingua>=0.2`, transitively PyTorch + transformers (~2GB) | The Trim analyzer (`tj optimize trim`) scores token significance with LLMLingua-2. Most users don't run it; keeping torch out of the base install keeps it small and fast on machines that don't have a GPU/CPU build of torch already. |
| `tokenjam[langchain]` | `langchain>=0.2` | Convenience pin for `patch_langchain()`; you can also install langchain yourself. |
| `tokenjam[crewai]` | `crewai>=0.28` | Convenience pin for `patch_crewai()`. |
| `tokenjam[autogen]` | `pyautogen>=0.2` | Convenience pin for `patch_autogen()`. |
| `tokenjam[litellm]` | `litellm>=1.40` | Convenience pin for `patch_litellm()`. |
| `tokenjam[dev]` | `pytest`, `pytest-asyncio`, `httpx`, `ruff`, `mypy` | For working on TokenJam itself. |

Combine multiple extras:

```bash
pipx install 'tokenjam[bloat,litellm]'
```

### Bloat extra details

`tokenjam[bloat]` is the largest extra; LLMLingua-2 transitively pulls in PyTorch and Hugging Face transformers, roughly 2GB on disk. On first run the analyzer downloads a ~110MB BERT-class classifier model under `~/.cache/tokenjam/models/` (override via `TOKENJAM_MODEL_CACHE`); subsequent runs are offline-capable.

If you run `tj optimize trim` without the extra installed, the analyzer self-registers and exits with a clear hint pointing at this install command; nothing in the base install crashes from its absence.

See [`docs/optimize/trim.md`](optimize/trim.md) for performance numbers, capture requirements, and what the analyzer actually reports.

## Upgrading

```bash
pipx upgrade tokenjam          # if you installed via pipx (recommended)
pip install --upgrade tokenjam # if you're in a pip + venv setup
```

After upgrading:

1. Restart the daemon to pick up the new code: `tj stop && tj serve &`
2. DB migrations apply automatically on the next `tj` invocation; no manual step required
3. Verify with `tj --version`

PyPI's CDN occasionally lags ~1–2 min after a release. If `pipx upgrade` reports "already at the latest version" but the reported `tj --version` is older than what's on the [releases page](https://github.com/Metabuilder-Labs/tokenjam/releases), wait a minute and retry.

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

To confirm telemetry is actually flowing (not just that setup completed), use `tj onboard --verify` (polls for the first span right after setup) or run `tj ping` any time afterward (emits one labeled test span and reports where it landed). If you've already set up and just restarted Claude Code / Codex, `tj onboard --claude-code --verify-only` (or `--codex`) skips the wizard entirely and only re-polls for a live span; no config rewrite, no summary replay. Note that backfilled history doesn't count as live: verification waits for a *new* span, so a fresh install with only backfilled sessions will report "no telemetry yet" until the restarted runtime emits one (`tj doctor` distinguishes the two). All are covered in the [CLI reference](cli-reference.md). Running onboarding unattended (CI, Docker, a script)? Use the non-interactive flags (`--plan`, `--budget`, `--no-daemon`, `--project`) documented in the [CLI reference](cli-reference.md) to skip every prompt.

**`--no-daemon` installs:** verification needs a read path to the database that doesn't compete with a concurrent writer; DuckDB is single-writer. With the daemon running, verification reads over HTTP and there's no conflict. Without it, the poller opens the DuckDB file directly, which only works if nothing else is writing to it at that moment. If your agent (or the SDK) is trying to write concurrently, you'll see `Can't verify yet — the database is locked (is tj serve running?)…` followed by `Start tj serve, then run tj doctor to check`; that's lock contention, not a failure of onboarding itself. Two ways around it: start `tj serve` temporarily (`tj serve &`, verify, then `tj stop`), or run `tj ping`, which proves interception through its own local proof-of-life exporter and works whether or not the daemon is up.
