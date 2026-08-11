---
paths:
  - "tokenjam/mcp/**"
  - "tokenjam/cli/cmd_onboard.py"
  - "tokenjam/cli/cmd_mcp.py"
  - "tokenjam/cli/cmd_stop.py"
  - "tokenjam/cli/cmd_uninstall.py"
---

# MCP server, daemon, and coding-agent onboarding

> Moved verbatim out of `CLAUDE.md` so it loads only when you touch these files.
> Cross-cutting rules stay in `CLAUDE.md`.

- **`tokenjam/mcp/server.py`**: FastMCP stdio server exposing observability data to Claude Code (plus the summarize tools — `list_summarize_candidates`, `summarize_prep`, `summarize_check`, `summarize_apply`, `summarize_undo`; see `core/summarize/`). Uses either a read-only DuckDB connection or HTTP proxy to `tj serve`. Initialized via `init()` from `cmd_mcp.py`.

## Daemon (launchd / systemd)

`tj onboard` (and `tj onboard --claude-code` / `--codex`) installs a background daemon that runs `tj serve` on login:
- **macOS**: `~/Library/LaunchAgents/com.tokenjam.serve.plist` — loaded via `launchctl load`. Logs at `/tmp/tj-serve.{out,err}`.
- **Linux**: `~/.config/systemd/user/tokenjam.service` — enabled via `systemctl --user enable --now tokenjam`.
- **Other**: skipped with a notice; user runs `tj serve` manually.

Reinstall behavior: `--claude-code` and `--codex` onboard check `_daemon_already_running()` (launchctl list / systemctl is-active) and skip reinstall when the daemon is up unless `--force` is passed. This avoids spurious "Background Items Added" prompts on macOS during second-project onboards. The launchd path always uses `launchctl unload -w` then `launchctl load -w` — the `-w` flag clears any Disabled=true entry from the launchd database (`tj stop` writes Disabled=true via `launchctl unload -w`), without which a subsequent plain `launchctl load` is a silent no-op. Use `tj stop` to halt the daemon, `tj uninstall` to remove unit files. `tj stop` also sweeps for any orphan foreground `tj serve` processes (e.g. from a manual `tj serve &`) so it reliably frees port 7391.

Before every DB write, onboard calls `_stop_serve_for_db_write()` (`cmd_onboard.py`), which stops the daemon via `stop_tj_serve()` (`cmd_stop.py`) and feeds the result into `stopped_for_db`; `_finish_onboard_serve()` then forces a restart (`need_restart = secret_rotated or plan_changed or stopped_for_db`), bypassing the already-running skip above. For that skip to ever actually fire, `stop_tj_serve()` must report "stopped" ONLY when something was genuinely loaded/active — it checks `launchctl list <label>` (or `systemctl --user is-active`) before unloading/disabling, since `launchctl unload -w` and `systemctl disable --now` both return 0 even against a plist/unit that was never loaded. Without that check, `stopped_for_db` is true on effectively every onboard run and the daemon reinstalls/restarts every time, even with nothing changed.

`tj serve` writes its resolved config path to `~/.local/share/tj/server.state` at startup. This is informational — onboarding flows (`--claude-code` and `--codex`) always write to the global config, so server.state is not used for secret-sync.

**Ephemeral-cache guard on the daemon unit itself (`_daemon_program_args`):** `npx tokenjam onboard` → `uvx --from tokenjam tj onboard` may still be running from a throwaway `uvx`/`pipx run` cache when it reaches daemon install — `_maybe_guard_ephemeral_runner()` (`cmd_onboard.py`, #120) offers a persistent install at the top of onboard but doesn't force one (declined, or non-interactive). `_resolve_tj_binary()`'s fallback would then resolve to a path like `~/.cache/uv/archive-v0/<hash>/bin/tj`, which `uv cache prune`/`uv cache clean` (routine maintenance) deletes outright, silently killing the daemon on next load and freezing it on whatever version was resolved at onboard time. `_daemon_program_args()` detects that (`_is_ephemeral_path`) and instead points ProgramArguments/ExecStart at the stable `uvx`/`pipx` shim itself (`uvx --from tokenjam tj --config ... serve`), so launchd/systemd re-resolves `tj` on every start rather than a path that can vanish. If no durable entrypoint exists at all, it warns and skips the install rather than writing a cache path.

## MCP Server

**The MCP is an SDK / API surface, not a Claude Code / Codex one.** It puts tj *in the request path* — the right place for SDK / API integrations doing real-time enforcement/policy/budgets. It is deliberately **not** wired for Claude Code / Codex subscription users: an in-loop MCP is a per-turn token burden on them (an A/B against a no-tj control measured a **materially higher** model-weighted token count; the figure itself is deliberately not restated here — it lives in the shipped product string that quotes it, and in the test pinning that string). Those users get tj **out-of-band**: the zero-token statusline (`tj statusline`, wired by `tj onboard --claude-code`) plus OTel telemetry ingest. `tj mcp` still works for anyone who invokes it; onboarding just no longer defaults CC/Codex users into it.

`tj mcp` starts a FastMCP stdio server. The connection mode is chosen at startup by `cmd_mcp.py`:
1. If `tj serve` is reachable on `config.api.{host,port}`, MCP proxies to it via HTTP (live ingest visible).
2. Otherwise it tries to spawn `tj serve` in the background and waits for the port up to `_start_and_wait`'s `timeout` default (`cmd_mcp.py`).
3. If neither works, it falls back to a **read-only DuckDB connection** — read tools still work, but newly ingested spans won't appear until restart.
4. If no config file is found, `init()` is skipped and tools return a no-config sentinel.

SDK / API users who want the in-loop tools can wire it manually: `claude mcp add tj --scope user -- tj mcp`. The `--claude-code` and `--codex` onboard flows **no longer** register the MCP (they wire the out-of-band statusline / OTel instead), and a re-onboard retires any tj-managed `[mcp_servers.tj]` block a previous version wrote to `~/.codex/config.toml`.

## Codex CLI Integration

`tj onboard --codex` writes an `[otel]` block to `~/.codex/config.toml` (out-of-band telemetry only). It does **not** register the tj MCP for Codex — Codex has no statusline surface, so tj stays fully out-of-band via OTel + the `tj` CLI; a re-onboard retires any `[mcp_servers.tj]` block a previous version wrote. Notes:
- Codex hardcodes `service.name=codex_exec` in its binary and silently ignores `[otel.resource]`, so onboarding does **not** write that block — all Codex traces land under the `codex_exec` agent ID regardless of project. Onboarding is one-time global, not per-project.
- Codex emits OTLP **logs** (not spans) to `/v1/logs`. `tokenjam/api/routes/logs.py` converts Codex events (`sse_event`, `user_prompt`, `tool_decision`, `tool_result`, `api_request`) into normalized spans for cost/drift/alerting. Event name is read from `attrs["event.name"]` when the OTLP body is empty (Codex schema quirk); epoch `timeUnixNano=0` falls back to `attrs["event.timestamp"]` ISO-8601. The `/v1/logs` endpoint also silently accepts `resourceSpans`/`resourceMetrics` because Codex's exporter reuses one endpoint for all signal types.
- Re-running `tj onboard --codex` is a no-op only when `[otel]` is already present in `~/.codex/config.toml` (the check is `[otel]`-only now); a re-onboard also strips any legacy `[mcp_servers.tj]` block, since tj is out-of-band for Codex. Re-onboarding either Codex or Claude Code cross-syncs the ingest secret into the other's config if it's already configured.
