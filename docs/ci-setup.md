# Non-interactive / CI setup

`tj onboard` is an interactive wizard by default, but every prompt has a flag
that skips it — so onboarding can run unattended in CI, a Docker harness, or
any script with no TTY attached. This page gives the exact promptless
invocation for each persona, notes what a Docker harness needs, and states
the environment expectations.

## The flags that matter

| Flag | Skips | Applies to |
|---|---|---|
| `--plan <tier>` | The "how do you pay?" plan-tier prompt | all personas (`api`/`pro`/`max_5x`/`max_20x` for Anthropic, `api`/`plus`/`team`/`enterprise` for OpenAI) |
| `--budget <usd>` | The daily-budget prompt | all personas (`0` disables the alert threshold) |
| `--project <name>` | The project-name prompt | `--claude-code` only |
| `--no-daemon` | Background daemon install (which otherwise runs unconditionally, not a prompt, but you generally want this off in ephemeral CI runners) | all personas |
| `--verify` | N/A — opts *into* the post-setup poll instead of the interactive "verify now?" confirm | all personas (see [`tj onboard --verify` and `--no-daemon`](installation.md#verifying-the-install) for the daemon caveat) |

Without `--plan` or `--budget`, the wizard calls `click.prompt(...)` unconditionally —
it does not check for a TTY first. In a script or CI runner with no stdin to
read, that prompt either hangs waiting for input or aborts on EOF. Passing both
flags is required for a clean non-interactive run on every persona; `--project`
is additionally required for `--claude-code` specifically. The plan flag also
skips a second, conditional prompt (a monthly API spend-ceiling question) that
only fires for `plan == "api"` — passing any `--plan` value bypasses it
regardless of which tier you choose.

## Per-persona invocation

### Bare / SDK

```bash
tj onboard --plan api --budget 5.00 --no-daemon
```

Writes `.tj/config.toml` in the current directory. Prints the
`patch_anthropic()` / `@watch()` snippet for whichever provider SDK / agent
framework it detects in the project (falls back to a generic Anthropic
snippet if nothing is detected) — instrument your agent with that snippet
separately; onboarding doesn't modify your application code.

### Claude Code

```bash
tj onboard --claude-code --plan max_20x --budget 5.00 --project my-project --no-daemon
```

Writes the shared global config (`~/.config/tj/config.toml`), OTLP exporter
vars into `~/.claude/settings.json`, and the zero-token statusline — all
non-destructively (an existing `statusLine` or other `env` keys are
preserved, never clobbered). Claude Code must be restarted for the new
`settings.json` env vars to take effect; that's a Claude Code constraint, not
something `tj onboard` can do for you in the same process.

### Codex

```bash
tj onboard --codex --plan enterprise --budget 5.00 --no-daemon
```

Writes an `[otel]` block to `~/.codex/config.toml`. Codex hardcodes
`service.name=codex_exec` in its binary, so onboarding is one-time and
global — there's no per-project `--project` flag for this persona.

## Re-running is idempotent

Re-running any of the invocations above with the same flags is safe: it
re-writes the same config values (no duplicate agent entries, no duplicate
`env` keys, no duplicate `~/.zshrc`/`~/.codex/config.toml` blocks — each is
matched and replaced in place, not appended again) and still produces zero
prompts.

## Docker harness notes

`tj onboard --claude-code` and `tj onboard --codex` write **two** parallel
endpoint configs so both native and containerized runs pick up telemetry
automatically:

- `~/.claude/settings.json` / `~/.codex/config.toml` — `127.0.0.1:<port>`, for
  Claude Code / Codex running natively on the host.
- `~/.zshrc` (created if absent, under a `# tj harness observability` marker)
  — `host.docker.internal:<port>`, for agent sessions launched inside a
  container that can't reach the host's loopback address directly.

If your harness's container is invoked non-interactively (no login shell, so
`~/.zshrc` is never sourced), export the same four vars directly in the
container/compose environment instead of relying on the `.zshrc` block:

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_LOGS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/json
export OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:<port>
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <ingest_secret>"
```

`<port>` defaults to `7391`; `<ingest_secret>` is printed by `tj onboard` and
also readable from `[security] ingest_secret` in the written config.

## Environment expectations

- `HOME` must resolve to a real, writable directory — onboarding writes to
  `~/.config/tj/`, `~/.claude/`, `~/.codex/`, and `~/.zshrc` under it. In a
  from-scratch container, set `HOME` explicitly if it isn't already.
- `TJ_CONFIG` (or `--config`) overrides config discovery for every other `tj`
  command afterward — set it if you wrote the config somewhere non-standard.
- `tj doctor` (exit 0 = clean, 1 = warnings, 2 = errors) is the standard
  post-onboard health gate for a CI step; `tj onboard --verify` / `tj ping`
  additionally confirm telemetry is *flowing*, not just that config was
  written correctly — see [Verifying the install](installation.md#verifying-the-install).
