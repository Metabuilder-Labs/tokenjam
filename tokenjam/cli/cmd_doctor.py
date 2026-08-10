from __future__ import annotations

import json
import platform
from pathlib import Path

import click
import duckdb
from rich.markup import escape

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.cli.tj_status import TjCommand
from tokenjam.core.config import load_config, resolve_config_path
from tokenjam.core.data_span import MIN_PLAUSIBLE_YEAR
from tokenjam.core.db import SPANS_INDEXES
from tokenjam.core.server_state import find_own_serve_pid
from tokenjam.utils.formatting import console, display_path


@click.command("doctor", cls=TjCommand, status_message="Running health checks…")
@json_option
@click.option(
    "--repair",
    is_flag=True,
    help="Attempt to fix issues that have a known repair path (e.g. rebuild the "
         "spans table when DuckDB column statistics are corrupt: "
         "https://github.com/Metabuilder-Labs/tokenjam/issues/56).",
)
@click.pass_context
def cmd_doctor(ctx: click.Context, output_json_flag: bool, repair: bool) -> None:
    """Health-check your tj setup."""
    output_json = resolve_output_json(ctx, output_json_flag)
    config = ctx.obj["config"]
    checks: list[dict] = []

    # 1. Config file found and valid
    checks.append(_check_config(ctx.obj.get("config_path_override")))

    # 2. DuckDB file writable
    checks.append(_check_db(config))

    # 3. Ingest secret set
    checks.append(_check_ingest_secret(config))

    # 3b. Daemon lifecycle — the unit must be registered, server.state must
    #     name a live serve process, and one machine must not run several.
    checks.extend(_check_daemon_lifecycle(config))

    # 3c. OTLP endpoint reachability — resolve, connect, authenticate. Replaces
    #     the per-invocation stderr warning that used to fire on a static file
    #     diff while nothing was listening at all.
    checks.append(_check_otlp_endpoint(config))

    # 4. Prometheus configured
    checks.append(_check_prometheus(config))

    # 5. Schema validation vs capture
    checks.append(_check_schema_vs_capture(config))

    # 6. Drift configured but inactive
    checks.append(_check_drift_inactive(config, ctx.obj["db"]))

    # 6b. Prompt capture off — trim / cache-recommend / reuse degraded
    checks.append(_check_capture_prompts(config))

    # 7. Webhook URL security
    checks.extend(_check_webhook_security(config))

    # 8. Webhook domain allowlist
    checks.extend(_check_webhook_allowlist(config))

    # 9. DuckDB spans column-statistics corruption (issue #56)
    spans_stats_check = _check_spans_stats(ctx.obj["db"])
    checks.append(spans_stats_check)

    # 9b. Secondary-index integrity on spans — a fault the statistics check
    #     above cannot see, and one that makes the retention DELETE unreliable.
    checks.append(_check_spans_indexes(ctx.obj["db"]))

    # 9c. Every OTHER explicit index, swept. 9b covers spans specifically
    #     (including absent ones); this one asks the same divergence question
    #     of the whole catalogue, because the fault is not confined to spans.
    checks.append(_check_index_divergence(ctx.obj["db"]))

    # 10. Live-span staleness — flags a stalled OTLP connection (issue #179)
    checks.append(_check_span_staleness(ctx.obj["db"]))

    # 10b. Background daemon liveness — is `tj serve` actually running, not
    #      just installed. Computed once and reused by the corpus-freshness
    #      and transcript-gap checks below so all three agree on whether
    #      anything is currently positioned to self-heal a gap.
    daemon_alive = find_own_serve_pid() is not None
    checks.append(_check_daemon_liveness(daemon_alive))

    # 10c. Corpus freshness — newest ingested session vs the configured
    #      ingest cadence. FAILS (not warns) when nothing is live to close
    #      the gap on its own, since that's silent, unbounded data loss
    #      rather than a transient lag.
    checks.append(_check_corpus_freshness(config, ctx.obj["db"], daemon_alive))

    # 11. Proxy base-URL wiring consistency (issue #219)
    checks.append(_check_proxy_wiring(config))

    # 12. MCP server wiring (issue #285)
    checks.append(_check_mcp_wiring(config))

    # 13. Schema integrity — recorded-but-unlanded migration (issue #55)
    checks.append(_check_schema_integrity(ctx.obj["db"]))

    # 14. Claude Code statusline wiring (issue #59) — the zero-token surface
    checks.append(_check_statusline_wiring(config))

    # 15. Onboarded-but-silent — first-signal diagnosis (issue #80)
    checks.append(_check_onboarding_first_signal(config, ctx.obj["db"]))

    # 16. Transcript ingest completeness — on-disk sessions never ingested,
    #     still recoverable only until Claude Code prunes the transcript.
    checks.append(_check_transcript_ingest_gap(config, ctx.obj["db"], daemon_alive))

    # 17. Cost integrity — sessions.total_cost_usd vs SUM(spans.cost_usd)
    checks.append(_check_cost_integrity(ctx.obj["db"]))

    # 18. Duplicate call ingest — one LLM call stored twice, once per observer
    checks.append(_check_duplicate_call_ingest(ctx.obj["db"]))

    # 18b. Timestamp sentinels an older build wrote — a one-shot cleanup.
    checks.append(_check_sentinel_timestamps(ctx.obj["db"]))

    # 19. Retention — the analysis span, what it keeps, and what the last run
    #     actually deleted. A delete of user history has to be readable after
    #     the fact, not inferable by diffing two measurements days apart.
    checks.append(_check_retention(config, ctx.obj["db"]))

    # 20. Subagent type linkage — sub_agent_id spans with no resolved
    #     sub_agent_type, the join key `subagent_rightsizing` and cost
    #     attribution need.
    checks.append(_check_unresolved_subagent_type(ctx.obj["db"]))

    # 21. Captured content backfill gap — [capture] is on, but a slice of
    #     backfilled history predates that and never got the content.
    checks.append(_check_content_capture_backfill_gap(config, ctx.obj["db"]))

    if output_json:
        click.echo(json.dumps(checks, default=str))
    else:
        for c in checks:
            _print_check(c)

    # --repair: attempt fixes for any check that exposed a repair_action
    if repair:
        _attempt_repairs(checks, ctx.obj["db"], output_json, config)

    has_errors = any(c["level"] == "error" for c in checks)
    has_warnings = any(c["level"] == "warning" for c in checks)
    if has_errors:
        ctx.exit(2)
    elif has_warnings:
        ctx.exit(1)
    else:
        ctx.exit(0)


def _check_config(override: str | None = None) -> dict:
    """Report which config file is live.

    Takes the invocation's own `--config` value so `tj --config PATH doctor`
    reports PATH: an explicit override never reaches the environment, so a
    bare rediscovery here would name a different file while every other check
    — which reads the config loaded from PATH — describes that one.
    """
    try:
        path = resolve_config_path(override)
        if path is None:
            return {"name": "Config file", "level": "error",
                    "message": "No config file found. Run `tj onboard` to create one."}
        load_config(str(path))
        return {"name": "Config file", "level": "ok",
                "message": f"Found and valid: {display_path(path)}"}
    except Exception as e:
        return {"name": "Config file", "level": "error",
                "message": f"Config parse error: {e}"}


def _check_db(config: object) -> dict:
    """
    Verify the DuckDB file is writable. The daemon legitimately holds
    the write lock when running — that's the recommended operating mode
    after `tj onboard`. Detect a lock-conflict error and downgrade to
    informational rather than flagging it as ✗ (#68 §4).
    """
    try:
        from pathlib import Path
        db_path = Path(config.storage.path).expanduser()
        conn = duckdb.connect(str(db_path))
        conn.close()
        return {"name": "DuckDB writable", "level": "ok",
                "message": f"Database accessible: {display_path(db_path)}"}
    except Exception as e:
        # DuckDB raises "Could not set lock on file ... Conflicting lock
        # is held in ... PID N" when another process (the daemon, in the
        # common case) has the DB open in write mode. That's the expected
        # operating state — surface as info, not error.
        err_msg = str(e).lower()
        if "conflicting lock" in err_msg or "could not set lock" in err_msg:
            return {
                "name": "DuckDB writable",
                "level": "info",
                "message": (
                    "Skipped — DB write lock held by another process "
                    "(typically tj serve). This is the expected operating "
                    "state when the daemon is running."
                ),
            }
        return {"name": "DuckDB writable", "level": "error",
                "message": f"Cannot open database: {e}"}


def _check_ingest_secret(config: object) -> dict:
    if not config.security.ingest_secret:
        return {"name": "Ingest secret", "level": "warning",
                "message": "No ingest secret set. API ingest endpoint is unprotected."}

    # Beyond "is a secret set": is it the SAME secret every config on this
    # machine's search path would resolve to? A project-local .tj/config.toml
    # and the global ~/.config/tj/config.toml can carry different secrets for
    # one store — the SDK reads one, a daemon started from a different cwd
    # reads the other, and span pushes 401 silently with no error surfaced
    # anywhere else (see `core/config.find_diverged_secret_config`).
    import sys
    from typing import cast

    # tomllib is stdlib only from 3.11+; pyproject.toml declares >=3.10, and
    # this file already carries this exact guard at `_check_mcp_wiring`
    # (~:1247) for the same reason — matching it here, not inventing a
    # second spelling.
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]

    from tokenjam.core.config import TjConfig, active_config_path, find_diverged_secret_config

    active_path = active_config_path(cast("TjConfig", config))
    if active_path is not None:
        try:
            with open(active_path, "rb") as f:
                active_raw = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            active_raw = None
        if active_raw is not None:
            diverged = find_diverged_secret_config(active_path, active_raw)
            if diverged is not None:
                other_path, _active_secret, _other_secret = diverged
                return {
                    "name": "Ingest secret",
                    "level": "warning",
                    "message": (
                        f"ingest_secret differs between {active_path} and "
                        f"{other_path} — whichever process reads the other "
                        f"file will 401 every span it tries to push. Align "
                        f"them (copy one secret into the other config) or "
                        f"delete the unused config."
                    ),
                }

    return {"name": "Ingest secret", "level": "ok",
            "message": "Ingest secret is configured."}


def _check_daemon_lifecycle(config: object) -> list[dict]:
    """Report the three lifecycle facts that files alone cannot prove."""
    from typing import cast

    from tokenjam.core.config import TjConfig
    from tokenjam.core.server_state import (
        inspect_daemon_unit,
        is_pid_alive,
        is_serve_process,
        list_serve_processes,
        read_server_state,
        server_state_path,
    )

    typed_config = cast("TjConfig", config)

    checks: list[dict] = []
    unit = inspect_daemon_unit()
    if unit.manager is None:
        checks.append({
            "name": "Daemon service",
            "level": "info",
            "message": "This platform has no managed TokenJam daemon; run `tj serve` manually.",
        })
    elif unit.error is not None:
        checks.append({
            "name": "Daemon service",
            "level": "warning",
            "message": f"Could not inspect the {unit.manager} unit: {unit.error}.",
        })
    elif not unit.installed:
        checks.append({
            "name": "Daemon service",
            "level": "info",
            "message": f"No {unit.manager} unit is installed. Run `tj onboard` to install it.",
        })
    elif not unit.loaded:
        action = "loaded" if unit.manager == "launchd" else "enabled"
        checks.append({
            "name": "Daemon service",
            "level": "warning",
            "message": (
                f"The {unit.manager} unit exists at {unit.path} but is not {action}; "
                "it will not start or restart automatically. Re-run `tj onboard`."
            ),
        })
    elif unit.manager == "systemd" and unit.active is False:
        checks.append({
            "name": "Daemon service",
            "level": "warning",
            "message": (
                f"The systemd unit is enabled at {unit.path} but is not active. "
                "Re-run `tj onboard` or inspect `systemctl --user status tokenjam`."
            ),
        })
    else:
        service_state = "loaded" if unit.manager == "launchd" else "enabled and active"
        checks.append({
            "name": "Daemon service",
            "level": "ok",
            "message": f"The {unit.manager} unit is {service_state}: {unit.path}",
        })

    processes = list_serve_processes()
    if processes is None:
        checks.append({
            "name": "Daemon instances",
            "level": "info",
            "message": "Could not inspect the process table for tj serve instances.",
        })
    elif len(processes) > 1:
        pids = ", ".join(str(process.pid) for process in processes)
        checks.append({
            "name": "Daemon instances",
            "level": "warning",
            "message": (
                f"Multiple tj serve instances are running (PIDs {pids}). Stop the "
                "extra foreground/venv instances, then run `tj onboard` so one "
                "managed daemon owns the configured endpoint."
            ),
        })
    elif len(processes) == 1:
        checks.append({
            "name": "Daemon instances",
            "level": "ok",
            "message": f"One tj serve instance is running (PID {processes[0].pid}).",
        })
    else:
        level = "warning" if unit.installed and unit.loaded else "info"
        message = (
            "The daemon unit is registered but no tj serve process is running. "
            "Re-run `tj onboard` and inspect the daemon logs."
            if level == "warning"
            else "No tj serve process is running."
        )
        checks.append({"name": "Daemon instances", "level": level, "message": message})

    state_path = server_state_path()
    if not state_path.exists():
        checks.append({
            "name": "Server state",
            "level": "info",
            "message": f"No server state file found at {state_path}.",
        })
        return checks

    server_state = read_server_state(state_path)
    if server_state is None:
        checks.append({
            "name": "Server state",
            "level": "warning",
            "message": f"The server state file at {state_path} is malformed or incomplete.",
        })
    elif not is_pid_alive(server_state.pid):
        checks.append({
            "name": "Server state",
            "level": "warning",
            "message": (
                f"The server state file points at dead PID {server_state.pid}"
                f"{f' on port {server_state.port}' if server_state.port is not None else ''}. "
                "It is stale; start the managed daemon with `tj onboard`."
            ),
        })
    elif not is_serve_process(server_state.pid):
        checks.append({
            "name": "Server state",
            "level": "warning",
            "message": (
                f"The server state file points at PID {server_state.pid}, but that PID is not "
                "a tj serve process. Start the managed daemon with `tj onboard`."
            ),
        })
    else:
        port_mismatch = (
            server_state.port is not None and server_state.port != typed_config.api.port
        )
        # A live daemon on the right port but a DIFFERENT config file is not
        # healthy: it's validating the wrong database and ingest secret. Port
        # alone can't see this — two configs can legitimately share a port
        # across separate installs/worktrees.
        config_mismatch = (
            server_state.config_path is not None
            and typed_config.config_path is not None
            and server_state.config_path != str(typed_config.config_path)
        )
        if port_mismatch or config_mismatch:
            details = []
            if port_mismatch:
                details.append(
                    f"port {server_state.port} (config expects {typed_config.api.port})"
                )
            if config_mismatch:
                details.append(
                    f"config {server_state.config_path} "
                    f"(this process resolved {typed_config.config_path})"
                )
            checks.append({
                "name": "Server state",
                "level": "warning",
                "message": (
                    "The live server state doesn't match this install's config: "
                    f"{'; '.join(details)}. Re-run `tj onboard` to reconcile the daemon."
                ),
            })
        else:
            checks.append({
                "name": "Server state",
                "level": "ok",
                "message": (
                    f"server.state points at live tj serve PID {server_state.pid}"
                    f"{f' on port {server_state.port}' if server_state.port is not None else ''}."
                ),
            })
    return checks


#: How long a reachability probe waits before calling the endpoint dead. Short
#: on purpose: doctor is interactive, and a hung TCP connect is the failure
#: mode being diagnosed, not something to sit through.
OTLP_PROBE_TIMEOUT_S = 2.0


def _configured_otlp_endpoint() -> tuple[str | None, str, str | None]:
    """The OTLP endpoint an agent session on this machine actually exports to.

    Returns ``(endpoint, source, secret)``. The authority is the tj-managed
    block in ``~/.zshrc``, because that is what a Claude Code session inherits;
    reading only the config would report the endpoint tj INTENDED rather than
    the one already written to the user's shell profile, which is precisely the
    fault this check exists to catch. ``endpoint`` is None when no managed block
    is installed.
    """
    import re
    from pathlib import Path

    from tokenjam.cli.cmd_onboard import _ZSHRC_OTEL_END, _ZSHRC_OTEL_START

    try:
        text = (Path.home() / ".zshrc").read_text()
    except OSError:
        return None, "shell profile", None
    block = re.search(
        rf"{re.escape(_ZSHRC_OTEL_START)}\n(.*?){re.escape(_ZSHRC_OTEL_END)}",
        text,
        flags=re.DOTALL,
    )
    if block is None:
        return None, "shell profile", None
    body = block.group(1)
    endpoint = re.search(r"^export OTEL_EXPORTER_OTLP_ENDPOINT=(\S+)$", body, re.M)
    secret = re.search(
        r'^export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer ([^"]+)"$', body, re.M
    )
    return (
        endpoint.group(1) if endpoint else None,
        "~/.zshrc",
        secret.group(1) if secret else None,
    )


def _check_otlp_endpoint(config: object) -> dict:
    """Does the OTLP endpoint agents export to actually answer?

    A telemetry pipeline that drops spans silently is the defect this replaces
    a per-invocation static warning with: the endpoint written to the shell
    profile was a container-only hostname that does not resolve on a host, so
    every span failed at DNS and only the delayed transcript backfill survived.
    Nothing anywhere reported it.

    So this resolves the host, opens a TCP connection, and — when something
    answers — authenticates with the secret that block carries, reporting a
    real verdict at each layer rather than a claim derived from file contents.
    """
    import socket
    from urllib.parse import urlparse

    name = "OTLP endpoint"
    endpoint, source, block_secret = _configured_otlp_endpoint()
    if endpoint is None:
        return {"name": name, "level": "info",
                "message": "No tj-managed OTel block in ~/.zshrc. Run "
                           "`tj onboard --claude-code` to configure Claude Code "
                           "telemetry."}

    parsed = urlparse(endpoint)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return {"name": name, "level": "error",
                "message": f"Endpoint in {source} is not a usable URL: "
                           f"{endpoint}. Re-run `tj onboard --claude-code` to "
                           f"rewrite it."}

    try:
        socket.getaddrinfo(host, port)
    except OSError:
        return {
            "name": name, "level": "error",
            "message": (
                f"{endpoint} ({source}) does not resolve: the hostname "
                f"{host} has no address on this machine, so every span an "
                f"agent session emits is dropped before it leaves the "
                f"process. Re-run `tj onboard --claude-code` to rewrite the "
                f"endpoint."
            ),
        }

    try:
        with socket.create_connection((host, port), timeout=OTLP_PROBE_TIMEOUT_S):
            pass
    except OSError as e:
        expected = config.api.port
        mismatch = (
            f" The daemon binds port {expected}, and this endpoint names "
            f"{port}."
            if port != expected else ""
        )
        return {
            "name": name, "level": "warning",
            "message": (
                f"{endpoint} ({source}) resolves but nothing is listening "
                f"({e.strerror or e}).{mismatch} Start the daemon with "
                f"`tj serve`, then re-run `tj doctor`."
            ),
        }

    # Something answers. Does it accept the secret the shell block signs with?
    if not block_secret:
        return {"name": name, "level": "ok",
                "message": f"{endpoint} ({source}) is reachable."}

    import httpx

    try:
        resp = httpx.get(
            f"{parsed.scheme}://{host}:{port}/api/v1/status",
            headers={"Authorization": f"Bearer {block_secret}"},
            timeout=OTLP_PROBE_TIMEOUT_S,
        )
    except httpx.RequestError as e:
        return {"name": name, "level": "warning",
                "message": f"{endpoint} ({source}) accepted a connection but "
                           f"the status probe failed: {e}."}

    if resp.status_code == 401:
        return {
            "name": name, "level": "error",
            "message": (
                f"{endpoint} ({source}) is reachable but rejected the ingest "
                f"secret in that block (401), so every span it pushes is "
                f"discarded. Re-run `tj onboard --claude-code` to rewrite the "
                f"block with the secret the daemon uses."
            ),
        }
    if resp.status_code >= 400:
        return {"name": name, "level": "warning",
                "message": f"{endpoint} ({source}) answered HTTP "
                           f"{resp.status_code} to the status probe."}
    return {"name": name, "level": "ok",
            "message": f"{endpoint} ({source}) is reachable and accepted the "
                       f"configured ingest secret."}


def _check_prometheus(config: object) -> dict:
    if config.export.prometheus.enabled:
        return {"name": "Prometheus", "level": "ok",
                "message": "Prometheus export enabled."}
    return {"name": "Prometheus", "level": "info",
            "message": "Prometheus export disabled."}


def _check_proxy_wiring(config: object) -> dict:
    """Flag orphaned proxy base-URL wiring (#219).

    The dangerous state: an agent's provider base-URL points at the proxy port
    but the proxy is disabled — traffic would hit a dead listener. Reuses the
    proxy wiring helper so the check and `tj proxy` agree.
    """
    from tokenjam.proxy.wiring import find_orphaned_wiring, proxy_base_url
    try:
        orphaned = find_orphaned_wiring(config)
    except Exception:  # best-effort; never fail doctor on this
        orphaned = []
    if orphaned:
        return {
            "name": "Proxy wiring", "level": "warning",
            "message": (
                f"{', '.join(orphaned)} point at the tj proxy "
                f"({proxy_base_url(config)}) but the proxy is disabled. Agent "
                "traffic would hit a dead port. Run `tj proxy enable` to start "
                "the listener or `tj proxy disable` to remove the wiring."
            ),
        }
    if getattr(config.proxy, "enabled", False):
        return {"name": "Proxy wiring", "level": "ok",
                "message": f"Proxy enabled on {proxy_base_url(config)} (suggest mode)."}
    return {"name": "Proxy wiring", "level": "ok",
            "message": "Proxy disabled; no orphaned base-URL wiring."}


def _check_schema_vs_capture(config: object) -> dict:
    has_schema = any(
        ac.output_schema for ac in config.agents.values()
    )
    if has_schema and not config.capture.tool_outputs:
        return {"name": "Schema vs capture", "level": "warning",
                "message": "Agent has output_schema but capture.tool_outputs is false. "
                           "Schema validation will have no data to validate."}
    return {"name": "Schema vs capture", "level": "ok",
            "message": "Schema and capture settings are consistent."}


def _check_capture_prompts(config: object) -> dict:
    """`capture.prompts` defaults on; flag when it's off so the user (running
    a config that predates the default, or one that turned it off on
    purpose) knows `trim` and `cache-recommend` produce no findings and
    `reuse` never reaches its prompt-prefix mode without it — not that
    they're broken."""
    if getattr(config.capture, "prompts", False):
        return {"name": "Prompt capture", "level": "ok",
                "message": "capture.prompts is on: trim, cache-recommend, "
                           "and reuse's prompt-prefix mode have data."}
    return {"name": "Prompt capture", "level": "info",
            "message": "capture.prompts is off. `tj optimize trim` and "
                       "`cache-recommend` will produce no findings, and "
                       "`reuse` stays on tool-sequence-only clustering. Set "
                       "capture.prompts = true under [capture] in your "
                       "config to enable them (stored locally in your "
                       "telemetry DB, never sent anywhere)."}


def _check_drift_inactive(config: object, db: object) -> dict:
    """Report drift-baseline progress for any agent that hasn't reached threshold yet.

    Drift detection is enabled by default, so brand-new agents (0–9 sessions)
    would otherwise trip a warning on every `tj doctor` run — pure noise,
    since collection-in-progress is the expected state. Downgraded to `info`
    so the user can see which agents are still building a baseline without
    treating it as a problem.
    """
    in_progress: list[str] = []
    for agent_id, ac in config.agents.items():
        if not ac.drift.enabled:
            continue
        count = db.get_completed_session_count(agent_id)
        if count < ac.drift.baseline_sessions:
            in_progress.append(f"{agent_id} ({count}/{ac.drift.baseline_sessions})")
    if in_progress:
        return {"name": "Drift detection", "level": "info",
                "message": "Collecting baseline: " + ", ".join(in_progress)}
    return {"name": "Drift detection", "level": "ok",
            "message": "Drift detection status is consistent."}


def _check_webhook_security(config: object) -> list[dict]:
    results = []
    for ch in config.alerts.channels:
        url = ch.url or ch.webhook_url
        if url and not url.startswith("https://") and not _is_local_url(url):
            results.append({
                "name": "Webhook security",
                "level": "warning",
                "message": f"Non-HTTPS, non-local webhook URL: {url}",
            })
    if not results:
        results.append({"name": "Webhook security", "level": "ok",
                        "message": "All webhook URLs are secure or local."})
    return results


def _check_webhook_allowlist(config: object) -> list[dict]:
    allowed = config.security.webhook_allowed_domains
    if not allowed:
        return []
    results = []
    for ch in config.alerts.channels:
        url = ch.url or ch.webhook_url
        if url:
            from urllib.parse import urlparse
            domain = urlparse(url).hostname
            if domain and domain not in allowed:
                results.append({
                    "name": "Webhook allowlist",
                    "level": "error",
                    "message": f"Webhook domain '{domain}' not in allowed list.",
                })
    return results


def _check_spans_stats(db: object) -> dict:
    """Detect DuckDB v1.5.x spans column-statistics corruption (issue #56).

    When stats are corrupt, `WHERE trace_id = X` returns 0 rows but the data
    is still there (visible via `LIKE`). The fix is to rebuild the table.

    Uses the already-open connection on `db` rather than opening a second
    one — DuckDB rejects mixing read-only and read-write connections to the
    same file from the same process.
    """
    from tokenjam.core.db import check_spans_stats_corruption

    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": "Spans column statistics", "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        corrupt = check_spans_stats_corruption(conn)
    except duckdb.Error as e:
        return {"name": "Spans column statistics", "level": "info",
                "message": f"Skipped — could not run canary query: {e}"}
    if corrupt:
        return {
            "name": "Spans column statistics",
            "level": "warning",
            "message": "DuckDB column statistics on the spans table are corrupt "
                       "— trace-detail queries (`tj traces <id>`, dashboard "
                       "trace view) will return no spans. Run `tj doctor "
                       "--repair` to rebuild the table (data is preserved). "
                       "Background: "
                       "https://github.com/Metabuilder-Labs/tokenjam/issues/56.",
            "repair_action": "rebuild_spans",
        }
    return {"name": "Spans column statistics", "level": "ok",
            "message": "Column statistics are consistent."}


def _check_spans_indexes(db: object) -> dict:
    """Detect a missing or under-reporting secondary index on `spans`.

    An ERROR, not a warning, and deliberately: the retention job's `DELETE FROM
    spans WHERE start_time < ?` maintains every index inside its own
    transaction, so a damaged one can abort the statement part-way and leave it
    reporting fewer rows removed than it matched. A deletion of user history
    whose extent nobody can state afterwards is not a degraded read path, it is
    an unsafe write path, and the column-statistics check next door cannot see
    it — that one asks only about `trace_id`'s row-group statistics and reports
    OK on either fault below.
    """
    from tokenjam.core.db import check_spans_index_corruption

    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": "Spans index integrity", "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        faults = check_spans_index_corruption(conn)
    except duckdb.Error as e:
        return {"name": "Spans index integrity", "level": "info",
                "message": f"Skipped — could not probe the indexes: {e}"}
    if faults:
        detail = "; ".join(f"{name} {reason}" for name, reason in faults)
        return {
            "name": "Spans index integrity",
            "level": "error",
            "message": f"Secondary index damage on the spans table — {detail}. "
                       f"Queries served by these indexes return incomplete "
                       f"results, and a DELETE (retention cleanup) can abort "
                       f"part-way through. Run `tj doctor --repair` to rebuild "
                       f"them from the table (no data is moved).",
            "repair_action": "rebuild_spans_indexes",
        }
    return {"name": "Spans index integrity", "level": "ok",
            "message": f"All {len(SPANS_INDEXES)} secondary indexes on spans "
                       f"agree with the table."}


def _check_index_divergence(db: object) -> dict:
    """Sweep EVERY explicit index for disagreement with its table.

    One cause, two harms, which is why this is an error and not a note:

    * **Reads silently under-report.** A `WHERE col = ...` served by a damaged
      index returns fewer rows than the table holds, with no error at all, so
      analyzers understate whatever they read through it.
    * **Writes are fatal.** A statement that must REMOVE an index entry raises
      a DuckDB `FatalException`, which invalidates the whole database instance
      and every connection in the process.

    The sweep covers every explicit index because the fault has been seen on
    more than one table. It reports what it can PROVE: an index is called sound
    only when every distinct value was compared, and anything less is reported
    as not proven rather than counted as passing. `--repair` therefore rebuilds
    every index rather than only the ones named here -- see the repair handler.
    """
    from tokenjam.core.db import check_index_divergence, explicit_indexes

    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": "Index integrity", "level": "info",
                "message": "Skipped \u2014 CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        faults, unproven = check_index_divergence(conn)
        total = len(explicit_indexes(conn))
    except duckdb.Error as e:
        return {"name": "Index integrity", "level": "info",
                "message": f"Skipped \u2014 could not probe the indexes: {e}"}

    if faults:
        detail = "; ".join(f"{name} on {table} {reason}" for name, table, reason in faults)
        caveat = (
            f" A further {len(unproven)} index(es) could not be proven sound; "
            f"the repair rebuilds every index, not only the ones listed here."
            if unproven else ""
        )
        return {
            "name": "Index integrity",
            "level": "error",
            "message": f"Index damage \u2014 {detail}. Queries served by these "
                       f"indexes return incomplete results, and the next write "
                       f"that removes an entry will invalidate the database and "
                       f"500 every route. Run `tj doctor --repair` to rebuild "
                       f"every index from its table (no rows are read or moved)."
                       f"{caveat}",
            "repair_action": "rebuild_diverged_indexes",
        }
    if unproven:
        listed = "; ".join(f"{name} ({reason})" for name, _t, reason in unproven)
        return {
            "name": "Index integrity",
            "level": "warn",
            "message": f"No divergence found, but only {total - len(unproven)} of "
                       f"{total} explicit indexes could be PROVEN sound. Not "
                       f"proven: {listed}. `tj doctor --repair` rebuilds every "
                       f"index regardless, which is the only way to be certain.",
            "repair_action": "rebuild_diverged_indexes",
        }
    return {"name": "Index integrity", "level": "ok",
            "message": f"All {total} explicit indexes compared against every "
                       f"distinct value in their tables and agree."}


def _check_retention(config: object, db: object) -> dict:
    """Report the analysis span, the retention derived from it, and the last delete.

    A deletion of the user's own history that leaves no account of itself is
    discoverable only by measuring the store twice, days apart, and diffing —
    which is how eight weeks of the oldest history went missing unnoticed. The
    ledger makes it a fact on disk; this makes it a fact somebody sees.

    Never an error: retention doing its job is not a fault. What would be a
    fault is a config whose retention undercuts its own span, and that cannot
    happen — `core/analysis_span.retention_days_for` raises it — so this reports
    when the clamp fired rather than warning about a state that no longer exists.
    """
    from tokenjam.core.analysis_span import (
        retention_days_for,
        retention_was_raised_to_span,
        span_label,
    )

    storage = getattr(config, "storage", None)
    if storage is None:
        return {"name": "Retention", "level": "info",
                "message": "Skipped — no storage config."}

    span = span_label(storage)
    kept = retention_days_for(storage)
    kept_text = (
        "deletion is disabled" if kept is None else f"history is kept for {kept} days"
    )
    clamp = ""
    if retention_was_raised_to_span(storage):
        clamp = (
            " Storage retention is set shorter than the span and has been raised "
            "to match — nothing tj analyzes can be deleted underneath it."
        )

    conn = getattr(db, "conn", None)
    last = ""
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT ran_at, spans_deleted, sessions_deleted, oldest_kept "
                "FROM retention_events ORDER BY ran_at DESC LIMIT 1"
            ).fetchone()
        except duckdb.Error:
            row = None
        if row is None:
            last = " No retention run has been recorded yet."
        else:
            ran_at, spans_deleted, sessions_deleted, oldest_kept = row
            last = (
                f" Last run {ran_at:%Y-%m-%d %H:%M} UTC removed {spans_deleted} "
                f"span(s) and {sessions_deleted} session(s)"
            )
            last += (
                f"; oldest surviving span {oldest_kept:%Y-%m-%d}." if oldest_kept
                else "; no dated spans remain."
            )

    return {"name": "Retention", "level": "ok",
            "message": f"Analyzing {span}, so {kept_text}.{clamp}{last}"}


def _check_sentinel_timestamps(db: object) -> dict:
    """Find rows an older build stamped with an epoch sentinel instead of a time.

    Ingest can no longer produce one — a record with no observed time is
    rejected at the boundary — so this is a one-shot cleanup for corpora already
    written, offered here rather than as a SQL snippet somebody has to be told
    about. A warning, not an error: the rows are inert until something takes a
    naive `MIN()` over them, at which point a single row reports a span decades
    wide and crushes every derived rate to zero.
    """
    from tokenjam.core.db import count_sentinel_timestamp_rows

    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": "Timestamp sentinels", "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        found = count_sentinel_timestamp_rows(conn)
    except duckdb.Error as e:
        return {"name": "Timestamp sentinels", "level": "info",
                "message": f"Skipped — could not scan for sentinels: {e}"}
    if found:
        detail = ", ".join(f"{count} in {table}" for table, count in sorted(found.items()))
        return {
            "name": "Timestamp sentinels",
            "level": "warning",
            "message": f"Rows dated before {MIN_PLAUSIBLE_YEAR} — {detail}. These "
                       f"carry no real observed time and a single one makes a "
                       f"naive MIN() report a span decades wide. Run `tj doctor "
                       f"--repair` to delete them.",
            "repair_action": "purge_timestamp_sentinels",
        }
    return {"name": "Timestamp sentinels", "level": "ok",
            "message": "No rows carry a placeholder timestamp."}


def _check_schema_integrity(db: object) -> dict:
    """Detect a recorded-but-unlanded migration (missing columns #55 / tables #382).

    `run_migrations` keys on the version integer alone, so a version renumbered
    or recorded under an older definition can be marked applied while its
    `ADD COLUMN` / `CREATE TABLE` never ran — leaving `spans`/`sessions` missing a
    column the code writes on every ingest (DuckDB Binder Error, silently dropped,
    blank Status page — #55) or a peripheral table absent so a policy-audit /
    session-label / session-story / loop write raises (#382). `DuckDBBackend` now
    self-heals both on open (`ensure_expected_columns` + `ensure_expected_tables`),
    so this normally reports OK; a WARN means the live DB was opened without that
    path (or self-heal failed) and needs `tj doctor --repair`.

    Uses the already-open connection on `db` (same pattern as the spans-stats
    canary) — DuckDB rejects a second read-write connection to the same file.
    """
    from tokenjam.core.db import missing_expected_columns, missing_expected_tables

    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": "Schema integrity", "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        missing_cols = missing_expected_columns(conn)
        missing_tables = missing_expected_tables(conn)
    except duckdb.Error as e:
        return {"name": "Schema integrity", "level": "info",
                "message": f"Skipped — could not inspect schema: {e}"}
    if missing_cols or missing_tables:
        parts = []
        if missing_cols:
            parts.append(
                "missing column(s) the code writes on ingest: "
                f"{', '.join(missing_cols)}"
            )
        if missing_tables:
            parts.append(
                "missing table(s) the code writes on non-ingest paths: "
                f"{', '.join(missing_tables)}"
            )
        return {
            "name": "Schema integrity",
            "level": "warning",
            "message": (
                f"Live schema has {'; '.join(parts)}. These were recorded as "
                "migrated but never landed; affected writes fail (dropped ingest "
                "/ blank Status page, or a raised error on a peripheral path). "
                "Run `tj doctor --repair` to reconcile (idempotent, data "
                "preserved)."
            ),
            "repair_action": "heal_schema",
        }
    return {"name": "Schema integrity", "level": "ok",
            "message": "All expected columns and tables present."}


def _check_cost_integrity(db: object) -> dict:
    """Flag sessions whose stored cost disagrees with their spans' sum.

    ``sessions.total_cost_usd`` and ``SUM(spans.cost_usd)`` are two figures the
    UI can show side by side — a session card next to a span-derived total — so
    they have to agree, and ``recompute_session_totals_from_spans`` already
    names the span sum as the source of truth. Historical rows can still hold a
    stale total from a path that moved one side without the other; this surfaces
    that instead of leaving the user to notice a 3.8% discrepancy between two
    screens. The repair is the existing idempotent recompute, so no cost is
    invented and no span is touched.
    """
    from tokenjam.core.db import session_cost_drift

    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": "Cost integrity", "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        count, total_drift, worst = session_cost_drift(conn)
    except duckdb.Error as e:
        return {"name": "Cost integrity", "level": "info",
                "message": f"Skipped — could not inspect cost totals: {e}"}
    if not count:
        return {"name": "Cost integrity", "level": "ok",
                "message": "Every session's stored cost matches the sum of its spans."}
    biggest = ""
    if worst:
        session_id, stored, span_sum = worst[0]
        biggest = (
            f" Largest: session {session_id} stores ${stored:,.2f} against "
            f"${span_sum:,.2f} of spans."
        )
    return {
        "name": "Cost integrity",
        "level": "warning",
        "message": (
            f"{count} session(s) store a total that disagrees with the sum of "
            f"their spans, ${total_drift:,.2f} apart in all. Cost shown per "
            f"session and cost derived from spans will not match."
            f"{biggest} Run `tj doctor --repair` to reconcile each session to "
            f"its spans (idempotent, spans untouched)."
        ),
        "repair_action": "heal_session_costs",
    }


def _check_duplicate_call_ingest(db: object) -> dict:
    """Flag LLM calls stored once per observer instead of once.

    A session that ran while `tj serve` was up is observed live AND again when
    its transcript is backfilled. Each path mints its own span_id, so the
    store's span_id-keyed idempotency never sees the overlap and every cost
    figure prices the call twice. Both ingest paths suppress this at write time
    now, so a DB filled by a current build reports nothing here; a DB filled by
    an older one still carries the doubled rows and reports a cost that is
    simply too high. The repair drops the redundant observations and
    reconciles the affected sessions, both idempotent.
    """
    from tokenjam.core.db import duplicate_call_observations

    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": "Duplicate call ingest", "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        spans, redundant_usd, worst = duplicate_call_observations(conn)
    except duckdb.Error as e:
        return {"name": "Duplicate call ingest", "level": "info",
                "message": f"Skipped — could not inspect span provenance: {e}"}
    if not spans:
        return {"name": "Duplicate call ingest", "level": "ok",
                "message": "No LLM call is stored more than once."}
    biggest = ""
    if worst:
        session_id, session_spans, session_usd = worst[0]
        biggest = (
            f" Largest: session {session_id} carries {session_spans} redundant "
            f"span(s) worth ${session_usd:,.2f}."
        )
    return {
        "name": "Duplicate call ingest",
        "level": "warning",
        "message": (
            f"{spans} span(s) restate an LLM call another ingest path already "
            f"recorded, inflating reported cost by ${redundant_usd:,.2f}."
            f"{biggest} Run `tj doctor --repair` to drop the restatements and "
            f"reconcile the affected sessions."
        ),
        "repair_action": "drop_duplicate_calls",
    }


def _check_unresolved_subagent_type(db: object) -> dict:
    """Flag Claude Code subagent spans with no resolved `sub_agent_type`.

    `sub_agent_id` (migration 14) and `sub_agent_type` (migration 19, the
    stable per-dispatch-agent-definition identity `subagent_rightsizing` and
    the cost-attribution proposals key on) are both derived from the on-disk
    transcript by `core/backfill.py`. A span tagged with the former but not
    the latter is one of: a row inserted before migration 19 shipped (the
    common case, and the one this check's repair actually fixes — a normal
    `tj backfill claude-code` re-derives it from the sidecar and overlays it
    onto the existing row), a dispatch whose transcript/sidecar Claude Code has
    since pruned (unrecoverable), or a deliberate `_PER_DISPATCH_TASK_KINDS`
    carve-out (not a gap — see that constant's docstring). The count here
    can't tell those apart; the repair narrows it to whatever the first case
    still leaves.
    """
    from tokenjam.core.db import unresolved_subagent_type_stats

    name = "Subagent type linkage"
    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": name, "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        spans, cost_usd = unresolved_subagent_type_stats(conn)
    except duckdb.Error as e:
        return {"name": name, "level": "info",
                "message": f"Skipped — could not inspect subagent spans: {e}"}
    if not spans:
        return {"name": name, "level": "ok",
                "message": "Every subagent-dispatch span has a resolved type."}
    return {
        "name": name,
        "level": "warning",
        "message": (
            f"{spans} subagent span(s) (${cost_usd:,.2f}) carry a "
            f"sub_agent_id with no resolved sub_agent_type — per-agent-type "
            f"cost breakdowns undercount them. Run `tj doctor --repair` to "
            f"re-derive what the on-disk transcripts still support; some may "
            f"remain unresolved because Claude Code has since pruned the "
            f"transcript or sidecar."
        ),
        "repair_action": "resolve_subagent_types",
    }


def _check_content_capture_backfill_gap(config: object, db: object) -> dict:
    """Warn loudly when `[capture]` is ON but backfilled spans still have no
    content — the other side of `_check_capture_prompts` above, which only
    flags capture being OFF. A row backfilled BEFORE the user turned capture
    on never gets the content on its own (a plain re-run only ever inserted
    NEW spans until `bulk_overlay_span_attrs` grew a content-merge half), so
    silently having neither check cover this case reads as "capture is on,
    so it must be working" when in fact a slice of history never got it.
    """
    capture = getattr(config, "capture", None)
    prompts_on = bool(getattr(capture, "prompts", False))
    completions_on = bool(getattr(capture, "completions", False))
    tool_inputs_on = bool(getattr(capture, "tool_inputs", False))
    name = "Captured content backfill"
    if not (prompts_on or completions_on or tool_inputs_on):
        return {"name": name, "level": "info",
                "message": "Skipped — no [capture] toggle is on."}
    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": name, "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        from tokenjam.core.db import (
            missing_captured_content_stats,
            missing_captured_tool_input_stats,
        )

        llm_missing = missing_captured_content_stats(
            conn, prompts=prompts_on, completions=completions_on,
        )
        tool_missing = missing_captured_tool_input_stats(conn) if tool_inputs_on else 0
    except duckdb.Error as e:
        return {"name": name, "level": "info",
                "message": f"Skipped — could not inspect span content: {e}"}
    total_missing = llm_missing + tool_missing
    if not total_missing:
        return {"name": name, "level": "ok",
                "message": "Every backfilled span has the content [capture] "
                           "toggles say it should."}
    parts = []
    if llm_missing:
        parts.append(f"{llm_missing} LLM span(s) missing prompt/completion content")
    if tool_missing:
        parts.append(f"{tool_missing} tool span(s) missing tool_input")
    return {
        "name": name,
        "level": "warning",
        "message": (
            f"{'; '.join(parts)} — likely backfilled before [capture] was "
            f"turned on. Run `tj doctor --repair` to re-derive it from the "
            f"on-disk transcripts still present; some may remain unresolved "
            f"if Claude Code has since pruned them."
        ),
        "repair_action": "backfill_missing_content",
    }


# Spans older than this are treated as a stalled connection. Claude Code /
# Codex flush their OTLP exporter on a short interval while running, so during
# any active session the newest span is minutes old at most. A 6h gap means
# either nothing has run (benign) or — the issue #179 failure mode — a running
# agent is still exporting to a stale endpoint after `tj onboard` rewrote it.
# 6h is wide enough to not nag overnight/weekend gaps, tight enough to catch a
# same-day reconfigure-then-keep-working session.
_SPAN_STALENESS_THRESHOLD_HOURS = 6


def _check_span_staleness(db: object) -> dict:
    """Warn when telemetry has stopped flowing despite spans existing (#179).

    After `tj onboard --claude-code` rewrites the OTLP endpoint/secret, an
    already-running Claude Code (or Codex) instance keeps exporting to the old
    endpoint, so today's spans silently never arrive. The user sees a flat
    chart and concludes "TokenJam isn't tracking anything." This check compares
    the newest span's `start_time` to wall-clock and nudges a restart when the
    gap exceeds the threshold.

    Queries `db.conn` directly with parameterised SQL — `StorageBackend` has no
    "newest span time" method, and `cmd_doctor` already accesses `db.conn` for
    the spans-stats canary.
    """
    from datetime import timezone

    from tokenjam.utils.time_parse import utcnow

    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": "Live-span freshness", "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        row = conn.execute("SELECT MAX(start_time) FROM spans").fetchone()
    except duckdb.Error as e:
        return {"name": "Live-span freshness", "level": "info",
                "message": f"Skipped — could not query span timestamps: {e}"}

    newest = row[0] if row else None
    if newest is None:
        # No spans at all — nothing to be stale. A genuinely empty DB is a
        # pre-onboard state, not a stalled connection; don't false-warn.
        return {"name": "Live-span freshness", "level": "info",
                "message": "No spans recorded yet — nothing to check."}

    # DuckDB TIMESTAMPTZ normally yields a tz-aware datetime; guard the naive
    # case (assume UTC) so the subtraction never raises.
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)

    age_hours = (utcnow() - newest).total_seconds() / 3600.0
    if age_hours > _SPAN_STALENESS_THRESHOLD_HOURS:
        return {
            "name": "Live-span freshness",
            "level": "warning",
            "message": (
                f"Latest span is {age_hours:.1f}h old (threshold "
                f"{_SPAN_STALENESS_THRESHOLD_HOURS}h) — if Claude Code or Codex "
                "is running, restart it so it picks up the current OTLP endpoint "
                "(or check your OTLP endpoint / `tj serve`)."
            ),
        }
    return {"name": "Live-span freshness", "level": "ok",
            "message": f"Most recent span is {age_hours:.1f}h old."}


# macOS/Linux paths a background daemon installs itself under (see
# `cmd_onboard._install_launchd` / `_install_systemd`). Presence means the
# user opted into continuous background capture — used only to phrase the
# liveness message; the liveness question itself is answered by
# `find_own_serve_pid()`, which works the same whether `tj serve` is running
# as a managed daemon or was started by hand.
def _daemon_install_path() -> Path | None:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library/LaunchAgents/com.tokenjam.serve.plist"
    if system == "Linux":
        return Path.home() / ".config/systemd/user/tokenjam.service"
    return None


def _daemon_liveness_hint() -> str:
    system = platform.system()
    if system == "Darwin":
        return ("Check with `launchctl print gui/$(id -u)/com.tokenjam.serve`, then "
                "re-register with `tj onboard --claude-code --reconfigure` or run "
                "`tj serve` manually.")
    if system == "Linux":
        return ("Check with `systemctl --user status tokenjam`, then re-register with "
                "`tj onboard --claude-code --reconfigure` or run `tj serve` manually.")
    return "Run `tj serve` manually."


def _check_daemon_liveness(daemon_alive: bool) -> dict:
    """Is `tj serve` actually running right now — not just installed.

    `find_own_serve_pid()` reads the state file `tj serve`'s own lifespan
    writes after binding its port, so this is true liveness (a real running
    process), independent of HOW it was started — managed daemon or a
    manually foregrounded `tj serve` both count. A background install that
    exists on disk but isn't currently loaded/running is the exact failure
    mode that motivated this check: on a real machine the launchd plist was
    present with `RunAtLoad`/`KeepAlive` both true and no `Disabled` key, yet
    `launchctl list` showed nothing loaded and there was no error trace —
    ingestion had simply stopped, silently, with no signal anywhere else.
    """
    name = "Background daemon liveness"
    if daemon_alive:
        return {"name": name, "level": "ok", "message": "`tj serve` is running."}

    install_path = _daemon_install_path()
    if install_path is None or not install_path.exists():
        return {
            "name": name,
            "level": "info",
            "message": ("No background daemon installed and `tj serve` is not "
                        "running — run `tj serve` manually, or `tj onboard` to "
                        "install one."),
        }
    return {
        "name": name,
        "level": "error",
        "message": (
            f"A background daemon is installed ({install_path}) but `tj serve` "
            f"is not running. Ingestion, alerts, and the web UI are all silently "
            f"paused — Claude Code prunes its own on-disk transcripts, so every "
            f"hour this stays down is history moving closer to being unrecoverable. "
            f"{_daemon_liveness_hint()}"
        ),
    }


def _check_corpus_freshness(config: object, db: object, daemon_alive: bool) -> dict:
    """Compare the newest ingested `sessions.started_at` to wall-clock,
    against a threshold derived from the daemon's own `[ingest]
    interval_minutes` cadence (`core/ingest_freshness.py`, shared with `tj
    status`'s advisory line so the two can't disagree on what "stale" means).

    Wall-clock age alone can't tell "ingestion is lagging" apart from "the
    user simply hasn't run Claude Code since" — both look identical from the
    `sessions` table. So a stale-by-age result is only PROVISIONAL: it's
    confirmed as a real gap by checking for on-disk transcript data newer
    than what's ingested (the same reconciliation `_check_transcript_ingest_gap`
    below uses), and downgraded to healthy when there is none. Severity for a
    confirmed gap depends on whether anything is currently positioned to
    close it on its own: with the daemon down, a real gap can only get
    staler, so this FAILS; with the daemon up, it's a WARNING — the next
    scheduled catch-up may well close it before anyone needs to act.
    """
    name = "Corpus freshness"
    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": name, "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    interval_minutes = getattr(getattr(config, "ingest", None), "interval_minutes", 30)
    try:
        from tokenjam.core.ingest_freshness import corpus_freshness
        from tokenjam.utils.time_parse import utcnow

        freshness = corpus_freshness(conn, interval_minutes, now=utcnow())
    except duckdb.Error as e:
        return {"name": name, "level": "info",
                "message": f"Skipped — could not query session timestamps: {e}"}

    if freshness.newest_session_at is None:
        return {"name": name, "level": "info",
                "message": "No sessions recorded yet — nothing to check."}
    if not freshness.is_stale:
        return {"name": name, "level": "ok",
                "message": f"Newest session started {freshness.age_hours:.1f}h ago."}

    # Stale by age — confirm there's actually un-ingested data waiting,
    # rather than the user just being idle, before calling it a gap.
    try:
        from tokenjam.core.transcript_sync import reconcile_claude_code

        report = reconcile_claude_code(db, since=freshness.newest_session_at)
    except Exception as e:  # never let a health check crash `tj doctor`
        return {"name": name, "level": "info",
                "message": f"Newest session started {freshness.age_hours:.1f}h ago; "
                           f"couldn't check on-disk transcripts to confirm this is "
                           f"a real ingestion gap: {e}"}

    if not report.verified:
        # Same "can't confidently render either way" case _check_transcript_
        # ingest_gap guards against (#642) — the anti-join never ran, so the
        # missing-count below would be meaningless.
        return {"name": name, "level": "info",
                "message": f"Newest session started {freshness.age_hours:.1f}h ago; "
                           "couldn't verify against on-disk transcripts to confirm "
                           "this is a real ingestion gap."}

    if report.missing_count == 0:
        return {
            "name": name,
            "level": "ok",
            "message": (
                f"Newest session started {freshness.age_hours:.1f}h ago, but "
                "there's no newer on-disk Claude Code data waiting to be "
                "ingested — you just haven't run it since."
            ),
        }

    level = "warning" if daemon_alive else "error"
    remedy = (
        " `tj serve` is running, so this should close on its own by the next "
        "scheduled catch-up; run `tj backfill claude-code` to close it now."
        if daemon_alive else
        " The background daemon is not running (see the liveness check above) "
        "— nothing is ingesting until it's back. Run `tj backfill claude-code` "
        "to close the gap once it is."
    )
    return {
        "name": name,
        "level": level,
        "message": (
            f"Newest session started {freshness.age_hours:.1f}h ago (expected "
            f"roughly every {interval_minutes}m), and {report.missing_count} "
            f"newer on-disk session(s) aren't ingested yet.{remedy}"
        ),
    }


def _check_transcript_ingest_gap(config: object, db: object, daemon_alive: bool = True) -> dict:
    """Surface on-disk Claude Code sessions that were never ingested.

    The live OTLP path drops any session whose shell lacked the telemetry env
    vars or that ran while `tj serve` was down/unreachable — Claude Code's
    exporter has no retry and no buffer. The transcript survives on disk, but
    only until Claude Code prunes it (~30 days), after which the session is
    unrecoverable. `_check_span_staleness` above catches "nothing is arriving
    right now"; this catches the subtler steady-state case where *most* sessions
    arrive and a slice silently doesn't.

    Scoped to a recent window so the check stays fast and only reports sessions
    that are still recoverable — anything older than the rotation horizon is
    gone regardless of what we say about it.

    Severity mirrors `_check_corpus_freshness`: a gap is only a WARNING while
    the daemon is up and `auto_catch_up` is on — both conditions under which
    the gap can plausibly close on its own on the next scheduled pass. Either
    one being false means nothing is going to fix this without a human
    running `tj backfill claude-code`, which is a FAIL, not a nag.
    """
    from datetime import timedelta

    from tokenjam.core.transcript_sync import CLAUDE_CODE_ROTATION_DAYS
    from tokenjam.utils.time_parse import utcnow

    name = "Transcript ingest completeness"
    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": name, "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        from tokenjam.core.transcript_sync import reconcile_claude_code

        since = utcnow() - timedelta(days=CLAUDE_CODE_ROTATION_DAYS)
        report = reconcile_claude_code(db, since=since)
    except Exception as e:  # never let a health check crash `tj doctor`
        return {"name": name, "level": "info",
                "message": f"Skipped — could not compare transcripts: {e}"}

    if not report.root.exists():
        return {"name": name, "level": "info",
                "message": f"No Claude Code transcripts found at {report.root}."}
    if report.missing_count == 0:
        return {"name": name, "level": "ok",
                "message": (f"All {report.disk_sessions} on-disk session(s) in the last "
                            f"{CLAUDE_CODE_ROTATION_DAYS}d are ingested.")}

    days_left = report.days_until_rotation()
    horizon = (f" Oldest is ~{days_left:.0f} day(s) from being pruned."
               if days_left is not None else "")
    auto = getattr(getattr(config, "ingest", None), "auto_catch_up", False)
    self_heals = daemon_alive and auto
    if self_heals:
        remedy = (" `tj serve` will catch up automatically; run `tj backfill claude-code` "
                  "to close it now.")
    elif not daemon_alive:
        remedy = (" The background daemon is not running, so nothing will close this "
                  "automatically — run `tj backfill claude-code`.")
    else:
        remedy = (" Automatic catch-up is off ([ingest] auto_catch_up) — run "
                  "`tj backfill claude-code`.")
    return {
        "name": name,
        "level": "warning" if self_heals else "error",
        "message": (
            f"{report.missing_count} on-disk session(s) are not ingested."
            f"{horizon}{remedy}"
        ),
    }


def _attempt_repairs(checks: list[dict], db: object, output_json: bool, config: object = None) -> None:
    """Run repair actions for any check that flagged one."""
    from tokenjam.core.db import (
        SESSION_COST_DRIFT_TOLERANCE_USD,
        check_spans_index_corruption,
        ensure_expected_columns,
        ensure_expected_tables,
        purge_sentinel_timestamp_rows,
        check_index_divergence,
        repair_explicit_indexes,
        repair_spans_indexes,
        repair_spans_stats,
        session_cost_drift,
    )

    conn = getattr(db, "conn", None)
    for c in checks:
        action = c.get("repair_action")
        if not action:
            continue
        if action == "heal_schema":
            if conn is None:
                if not output_json:
                    console.print(
                        "  [yellow]Repair skipped — CLI is using the HTTP API "
                        "fallback. Stop `tj serve` and retry so doctor has "
                        "direct DB access.[/yellow]"
                    )
                continue
            try:
                added = ensure_expected_columns(conn)
                created = ensure_expected_tables(conn)
            except duckdb.Error as e:
                if not output_json:
                    console.print(
                        f"  [red]Schema repair failed — {e}. If the database is "
                        f"locked, stop `tj serve` and retry.[/red]"
                    )
                continue
            if not output_json:
                repaired = added + [f"{t} (table)" for t in created]
                summary = ", ".join(repaired) if repaired else "nothing (already healthy)"
                console.print(
                    f"  [green]Schema reconciled — added {summary}.[/green]"
                )
            continue
        if action == "heal_session_costs":
            recompute = getattr(db, "recompute_session_totals_from_spans", None)
            if conn is None or recompute is None:
                if not output_json:
                    console.print(
                        "  [yellow]Repair skipped — CLI is using the HTTP API "
                        "fallback. Stop `tj serve` and retry so doctor has "
                        "direct DB access.[/yellow]"
                    )
                continue
            try:
                # Re-read the drifted set here rather than trusting the check's
                # truncated `worst` list, which is capped for display.
                count, _total, _worst = session_cost_drift(conn)
                drifted = [
                    str(row[0])
                    for row in conn.execute(
                        """
                        SELECT s.session_id
                        FROM sessions AS s
                        LEFT JOIN (
                            SELECT session_id, SUM(cost_usd) AS span_cost
                            FROM spans WHERE session_id IS NOT NULL
                            GROUP BY session_id
                        ) AS agg ON agg.session_id = s.session_id
                        WHERE ABS(COALESCE(agg.span_cost, 0.0)
                                  - COALESCE(s.total_cost_usd, 0.0)) > $1
                        """,
                        [SESSION_COST_DRIFT_TOLERANCE_USD],
                    ).fetchall()
                ]
                recompute(drifted)
                after, _after_total, _ = session_cost_drift(conn)
            except duckdb.Error as e:
                if not output_json:
                    console.print(
                        f"  [red]Cost repair failed — {e}. If the database is "
                        f"locked, stop `tj serve` and retry.[/red]"
                    )
                continue
            if not output_json:
                console.print(
                    f"  [green]Session costs reconciled — {count - after} of "
                    f"{count} session(s) now match their spans.[/green]"
                )
            continue
        if action == "drop_duplicate_calls":
            from tokenjam.core.db import purge_duplicate_call_observations

            recompute = getattr(db, "recompute_session_totals_from_spans", None)
            if conn is None or recompute is None:
                if not output_json:
                    console.print(
                        "  [yellow]Repair skipped — CLI is using the HTTP API "
                        "fallback. Stop `tj serve` and retry so doctor has "
                        "direct DB access.[/yellow]"
                    )
                continue
            try:
                deleted, sessions = purge_duplicate_call_observations(conn)
                # The delete moved SUM(spans) under rows nobody rewrote, so the
                # affected sessions have to be reconciled or this repair simply
                # trades one disagreement for another.
                if sessions:
                    recompute(sessions)
            except duckdb.Error as e:
                if not output_json:
                    console.print(
                        f"  [red]Duplicate-call repair failed — {e}. If the "
                        f"database is locked, stop `tj serve` and retry.[/red]"
                    )
                continue
            if not output_json:
                console.print(
                    f"  [green]Dropped {deleted} restated observation(s) across "
                    f"{len(sessions)} session(s); their totals were "
                    f"reconciled.[/green]"
                )
            continue
        if action == "resolve_subagent_types":
            from tokenjam.core.backfill import (
                CLAUDE_CODE_PROJECTS_ROOT,
                ingest_claude_code,
            )
            from tokenjam.core.db import unresolved_subagent_type_stats

            if conn is None:
                if not output_json:
                    console.print(
                        "  [yellow]Repair skipped — CLI is using the HTTP API "
                        "fallback. Stop `tj serve` and retry so doctor has "
                        "direct DB access.[/yellow]"
                    )
                continue
            try:
                # A normal (non-`--reingest`) backfill: it inserts anything
                # newly on disk AND, as of this repair, overlays
                # sub_agent_id/sub_agent_type onto existing rows it can now
                # resolve (see `_dedup_new_spans`'s overlay_candidates /
                # `bulk_overlay_span_attrs`). Unbounded (no `--since`) so
                # it reaches every row this check counted, matching how the
                # transcript-ingest-gap repair above is also unbounded.
                subagent_repair_result = ingest_claude_code(
                    db, root=CLAUDE_CODE_PROJECTS_ROOT, config=config,
                )
                still_unresolved, still_unresolved_usd = unresolved_subagent_type_stats(conn)
            except Exception as e:  # backfill can raise for many reasons (OS, JSON, DB)
                if not output_json:
                    console.print(
                        f"  [red]Subagent-type repair failed — {e}.[/red]"
                    )
                continue
            if not output_json:
                tail = (
                    f" {still_unresolved} span(s) (${still_unresolved_usd:,.2f}) "
                    "remain unresolved — their transcript or sidecar has likely "
                    "been pruned by Claude Code, or they're a per-dispatch "
                    "label that's correctly untyped."
                    if still_unresolved else " None remain unresolved."
                )
                console.print(
                    f"  [green]Resolved sub_agent_type on "
                    f"{subagent_repair_result.spans_retagged} existing "
                    f"span(s).[/green]{tail}"
                )
            continue
        if action == "backfill_missing_content":
            from tokenjam.core.backfill import (
                CLAUDE_CODE_PROJECTS_ROOT,
                ingest_claude_code,
            )
            from tokenjam.core.db import (
                missing_captured_content_stats,
                missing_captured_tool_input_stats,
            )

            if conn is None:
                if not output_json:
                    console.print(
                        "  [yellow]Repair skipped — CLI is using the HTTP API "
                        "fallback. Stop `tj serve` and retry so doctor has "
                        "direct DB access.[/yellow]"
                    )
                continue
            capture = getattr(config, "capture", None)
            prompts_on = bool(getattr(capture, "prompts", False))
            completions_on = bool(getattr(capture, "completions", False))
            tool_inputs_on = bool(getattr(capture, "tool_inputs", False))
            try:
                # Same underlying mechanism as "resolve_subagent_types" — a
                # normal (non-`--reingest`) backfill, unbounded — the
                # content-overlay half of `bulk_overlay_span_attrs` is
                # capture-config-aware (see `_dedup_new_spans`'s
                # `capture_on` gate), so this one pass re-derives content
                # from the transcripts still on disk using whatever
                # [capture] toggles are ACTUALLY on right now.
                content_repair_result = ingest_claude_code(
                    db, root=CLAUDE_CODE_PROJECTS_ROOT, config=config,
                )
                still_missing = missing_captured_content_stats(
                    conn, prompts=prompts_on, completions=completions_on,
                ) + (missing_captured_tool_input_stats(conn) if tool_inputs_on else 0)
            except Exception as e:  # backfill can raise for many reasons (OS, JSON, DB)
                if not output_json:
                    console.print(
                        f"  [red]Content-backfill repair failed — {e}.[/red]"
                    )
                continue
            if not output_json:
                tail = (
                    f" {still_missing} span(s) remain missing content — "
                    "their transcript has likely been pruned by Claude Code."
                    if still_missing else " None remain missing."
                )
                # spans_retagged counts every overlay this pass made (content
                # AND any newly-resolvable sub_agent_type together — one
                # shared counter), so it is an upper bound on the content
                # portion specifically, not an exact count of it.
                console.print(
                    f"  [green]Overlaid {content_repair_result.spans_retagged} "
                    f"existing span(s) (content and/or subagent type)."
                    f"[/green]{tail}"
                )
            continue
        if action == "purge_timestamp_sentinels":
            if conn is None:
                if not output_json:
                    console.print(
                        "  [yellow]Repair skipped — CLI is using the HTTP API "
                        "fallback. Stop `tj serve` and retry so doctor has "
                        "direct DB access.[/yellow]"
                    )
                continue
            try:
                removed = purge_sentinel_timestamp_rows(conn)
            except duckdb.Error as e:
                if not output_json:
                    console.print(
                        f"  [red]Sentinel purge failed — {e}. If the database is "
                        f"locked, stop `tj serve` and retry.[/red]"
                    )
                continue
            if not output_json:
                detail = ", ".join(
                    f"{count} from {table}" for table, count in sorted(removed.items())
                ) or "nothing"
                console.print(f"  [green]Deleted {detail}.[/green]")
            continue
        if action == "rebuild_spans_indexes":
            if conn is None:
                if not output_json:
                    console.print(
                        "  [yellow]Repair skipped — CLI is using the HTTP API "
                        "fallback. Stop `tj serve` and retry so doctor has "
                        "direct DB access.[/yellow]"
                    )
                continue
            try:
                repair_spans_indexes(conn)
                remaining = check_spans_index_corruption(conn)
            except duckdb.Error as e:
                if not output_json:
                    console.print(
                        f"  [red]Index rebuild failed — {e}. If the database is "
                        f"locked, stop `tj serve` and retry.[/red]"
                    )
                continue
            if not output_json:
                if remaining:
                    # Rebuilding from the table cannot leave an index disagreeing
                    # with it, so anything still failing is a fault one layer
                    # down — say so rather than reporting a repair that did not
                    # take.
                    console.print(
                        f"  [red]Rebuilt, but {len(remaining)} index(es) still "
                        f"disagree with the table — the damage is below the "
                        f"index layer. Run `tj doctor --repair` again to rebuild "
                        f"the table itself.[/red]"
                    )
                else:
                    console.print(
                        f"  [green]Rebuilt {len(SPANS_INDEXES)} secondary "
                        f"index(es) on spans; no rows were moved.[/green]"
                    )
            continue
        if action == "rebuild_spans":
            if conn is None:
                if not output_json:
                    console.print(
                        "  [yellow]Repair skipped — CLI is using the HTTP API "
                        "fallback. Stop `tj serve` and retry so doctor has "
                        "direct DB access.[/yellow]"
                    )
                continue
            try:
                before_row = conn.execute("SELECT COUNT(*) FROM spans").fetchone()
                repair_spans_stats(conn)
                after_row = conn.execute("SELECT COUNT(*) FROM spans").fetchone()
                before = before_row[0] if before_row else 0
                after = after_row[0] if after_row else 0
            except duckdb.Error as e:
                if not output_json:
                    console.print(
                        f"  [red]Repair failed — {e}. If the database is locked, "
                        f"stop `tj serve` and retry.[/red]"
                    )
                continue
            if not output_json:
                console.print(
                    f"  [green]Spans table rebuilt — {before} rows preserved "
                    f"(verified: {after}).[/green]"
                )
            continue
        if action == "rebuild_diverged_indexes":
            if conn is None:
                if not output_json:
                    console.print(
                        "  [yellow]Repair skipped \u2014 CLI is using the HTTP API "
                        "fallback. Stop `tj serve` and retry so doctor has "
                        "direct DB access.[/yellow]"
                    )
                continue
            try:
                # EVERY index, not only the ones the probe flagged. The probe
                # compares particular values, so a clean verdict from it is not
                # proof -- on a real damaged database a sampled sweep missed one
                # of four, and repairing only its verdict left the table still
                # raising the fatal. Measured at ~1.1s for all fourteen indexes
                # of a 3.9GB database, so there is nothing to be clever about.
                rebuilt = repair_explicit_indexes(conn)
                still_diverged, _ = check_index_divergence(conn)
            except duckdb.Error as e:
                if not output_json:
                    console.print(
                        f"  [red]Repair failed \u2014 {e}. If the database is locked, "
                        f"stop `tj serve` and retry.[/red]"
                    )
                continue
            if not output_json:
                if still_diverged:
                    detail = "; ".join(
                        f"{n} on {t} {r}" for n, t, r in still_diverged
                    )
                    console.print(
                        f"  [red]Rebuilt {len(rebuilt)} index(es) but divergence "
                        f"remains \u2014 {detail}.[/red]"
                    )
                else:
                    console.print(
                        f"  [green]Rebuilt {len(rebuilt)} damaged index(es) from "
                        f"their tables \u2014 {', '.join(rebuilt)}. No rows read or "
                        f"moved; re-probe is clean.[/green]"
                    )


def _is_local_url(url: str) -> bool:
    from urllib.parse import urlparse
    hostname = urlparse(url).hostname
    return hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0") if hostname else False


def _check_mcp_wiring(config: object) -> dict:
    """Check if the tj MCP server is wired into Claude Code or Codex."""
    import sys
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]
    from pathlib import Path

    codex_config_path = Path.home() / ".codex" / "config.toml"
    claude_config_path = Path.home() / ".claude.json"

    has_codex = False
    has_claude = False

    # Check Codex config
    if codex_config_path.exists():
        try:
            with open(codex_config_path, "rb") as f:
                codex_data = tomllib.load(f)
            has_codex = bool(codex_data.get("mcp_servers", {}).get("tj"))
        except Exception:
            pass

    # Check Claude Code global config
    if claude_config_path.exists():
        try:
            with open(claude_config_path, "r", encoding="utf-8") as f:
                claude_data = json.load(f)
            has_claude = bool(claude_data.get("mcpServers", {}).get("tj"))
        except Exception:
            pass

    # Check project-level configs in current directory
    has_project = False
    for name in (".mcp.json", ".claude.json"):
        project_path = Path.cwd() / name
        if project_path.exists():
            try:
                with open(project_path, "r", encoding="utf-8") as f:
                    proj_data = json.load(f)
                if bool(proj_data.get("mcpServers", {}).get("tj")):
                    has_project = True
                    break
            except Exception:
                pass

    # The MCP is an SDK / API surface, not a Claude Code / Codex one (#59): an
    # in-loop MCP is a per-turn quota burden on subscription users, so its
    # ABSENCE for a coding agent is the correct state, never a warning. A
    # registration found in Claude Code, Codex, or project scope is exactly
    # the quota-tax footgun this check exists to catch (`tj onboard` hasn't
    # written any of these since #59 — see cmd_onboard.py's Codex path,
    # which actively retires a legacy block) — flag it, don't green-check it.
    if has_codex or has_claude or has_project:
        found_locations = []
        removal_hints = []
        if has_codex:
            found_locations.append("Codex (global)")
            removal_hints.append(
                "`tj onboard --codex --reconfigure` retires the legacy "
                "[mcp_servers.tj] block"
            )
        if has_claude:
            found_locations.append("Claude Code (global)")
            removal_hints.append("`claude mcp remove tj --scope user` to deregister")
        if has_project:
            found_locations.append("project scope")
            removal_hints.append(
                "remove the `tj` entry from .mcp.json / .claude.json in this project"
            )
        return {
            "name": "MCP wiring",
            "level": "warning",
            "message": (
                f"MCP server registered in {', '.join(found_locations)} — an "
                "in-loop MCP is a per-turn token tax on subscription users "
                "(+36% measured), not the recommended surface for "
                "Claude Code / Codex. Remove it: " + "; ".join(removal_hints) + ". "
                "The MCP is meant for SDK / API integrations only."
            ),
        }

    return {
        "name": "MCP wiring",
        "level": "info",
        "message": (
            "MCP server not registered — expected for Claude Code / Codex, which "
            "use tj out-of-band (statusline + OTel). The MCP is for SDK / API "
            "users; wire it with `claude mcp add tj --scope user -- tj mcp` only "
            "if you want the in-loop tools (it costs per-turn quota)."
        ),
    }


def _claude_code_context(config: object) -> bool:
    """True when there's positive evidence the user runs tj *with* Claude Code.

    Two signals, both scoped to THIS user's tj setup rather than the machine as a
    whole:
      * a Claude Code home dir (``~/.claude``) exists, or
      * a ``claude-code-*`` agent is present in config (created by
        ``tj onboard --claude-code`` and its backfill).

    Deliberately NOT a signal: a ``claude`` binary on ``$PATH``. The CLI can be
    installed machine-wide while a given user onboards tj SDK-only, so keying the
    missing-statusline warning off the binary flipped a correct, fully-configured
    SDK install's ``tj doctor`` to exit 1 (#105). (Uses the simple ``~/.claude`` +
    agent-prefix heuristics; the core/framing persona helpers land on a later
    branch and aren't available here.)
    """
    from pathlib import Path

    if (Path.home() / ".claude").exists():
        return True
    agent_ids = list(getattr(config, "agents", {}) or {})
    return any(a.startswith("claude-code-") for a in agent_ids)


def _check_statusline_wiring(config: object) -> dict:
    """Check whether the zero-token tj statusline is wired into Claude Code (#59).

    The statusline is tj's out-of-band Claude Code surface. When there's a Claude
    Code context (see :func:`_claude_code_context`) but the statusLine isn't tj's,
    nudge the user to onboard (warning); a foreign statusLine is fine (we never
    clobber it) and reported as info. On a machine with no Claude Code context —
    a pure-SDK install — the check is purely informational so ``tj doctor`` still
    exits 0 (#105).
    """
    from pathlib import Path

    settings_path = Path.home() / ".claude" / "settings.json"
    statusline: dict | None = None
    if settings_path.exists():
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sl = data.get("statusLine")
            if isinstance(sl, dict):
                statusline = sl
        except Exception:
            pass

    cmd = statusline.get("command", "") if statusline else ""
    if isinstance(cmd, str) and "tj statusline" in cmd:
        return {
            "name": "Statusline wiring",
            "level": "ok",
            "message": "tj statusline wired into Claude Code (zero token cost).",
        }

    if statusline is not None:
        return {
            "name": "Statusline wiring",
            "level": "info",
            "message": (
                "A non-tj statusLine is set in ~/.claude/settings.json (left "
                "untouched). Set its command to `tj statusline` to see tj's "
                "re-read / quota line."
            ),
        }
    if _claude_code_context(config):
        return {
            "name": "Statusline wiring",
            "level": "warning",
            "message": (
                "tj statusline not wired into Claude Code. Run `tj onboard "
                "--claude-code` for the zero-token re-read / quota line."
            ),
        }
    return {
        "name": "Statusline wiring",
        "level": "info",
        "message": (
            "Claude Code not detected — this looks like an SDK-only setup. "
            "`tj onboard --claude-code` wires the zero-token tj statusline if "
            "you also use Claude Code."
        ),
    }


def _tj_statusline_wired() -> bool:
    """True when ~/.claude/settings.json wires the tj statusline (a Claude Code
    onboarding marker)."""
    from pathlib import Path

    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        return False
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        sl = data.get("statusLine")
        cmd = sl.get("command", "") if isinstance(sl, dict) else ""
        return isinstance(cmd, str) and "tj statusline" in cmd
    except Exception:
        return False


def _detect_onboarded_persona(config: object) -> str:
    """Best-guess the onboarding persona from config, for a tailored cause.

    Falls back to ``"sdk"`` — the fail-open path where a silent config is most
    likely — when no Claude Code / Codex marker is present.
    """
    agent_ids = list(getattr(config, "agents", {}) or {})
    if any(a.startswith("claude-code-") for a in agent_ids):
        return "claude-code"
    if "codex_exec" in agent_ids:
        return "codex"
    if _tj_statusline_wired():
        return "claude-code"
    return "sdk"


def _check_onboarding_first_signal(config: object, db: object) -> dict:
    """Diagnose the onboarded-but-zero-spans silent case (#80).

    Two personas can finish onboarding "successfully" yet never emit a span: the
    Claude Code path only starts telemetry after a restart, and the fail-open SDK
    path swallows a typo'd agent_id / missing patch / dead daemon silently. When
    the DB has zero *live* spans but a config exists, surface the likely
    per-persona cause instead of leaving the user staring at a blank Status page.

    Live vs backfill (#102): a bare `COUNT(*)` counts backfilled history too, so
    a Claude-Code user who onboarded and got 25 backfilled spans reads "telemetry
    is flowing" here while `tj onboard --verify` (which waits for a *new*, live
    span) says "no telemetry yet" — the two directly contradict each other on the
    same on-disk state. Backfill spans carry `attributes.source = 'backfill.*'`,
    so we split the count and only treat *live* spans as "flowing"; a backfill-only
    DB gets the honest "restart and it will appear" message instead.

    Reported at **info** level (not warning) in the silent cases: doctor has no
    onboarding timestamp, so it can't tell "just onboarded seconds ago" (expected
    empty) from "onboarded long ago and silent" (the failure). Flagging every
    fresh setup as a warning would be a false alarm and flip a clean `tj doctor`
    to exit 1. The info line still carries the actionable cause. Complements
    ``_check_span_staleness`` (which owns spans-exist-but-stale); this one owns the
    no-live-spans case.
    """
    conn = getattr(db, "conn", None)
    if conn is None:
        return {"name": "Onboarding signal", "level": "info",
                "message": "Skipped — CLI is running through the HTTP API "
                           "fallback (stop `tj serve` to access the DB directly)."}
    try:
        row = conn.execute(
            """
            SELECT
              COUNT(*) AS total,
              COUNT(*) FILTER (
                WHERE json_extract_string(attributes, '$.source') LIKE 'backfill.%'
              ) AS backfill
            FROM spans
            """
        ).fetchone()
    except duckdb.Error as e:
        return {"name": "Onboarding signal", "level": "info",
                "message": f"Skipped — could not count spans: {e}"}

    total = int(row[0]) if row and row[0] is not None else 0
    backfill = int(row[1]) if row and row[1] is not None else 0
    live = total - backfill

    if live > 0:
        note = f" ({backfill} backfilled)" if backfill else ""
        return {"name": "Onboarding signal", "level": "ok",
                "message": f"{live} live span(s) recorded{note} — telemetry is flowing."}

    if backfill > 0:
        # Onboarded, history backfilled, but nothing live since — the exact state
        # where `--verify` correctly reports "no telemetry yet". Say why, and what
        # to do, instead of a reassuring raw count.
        return {
            "name": "Onboarding signal",
            "level": "info",
            "message": (
                f"{backfill} backfilled session span(s) present, but no LIVE "
                "telemetry since onboarding — restart Claude Code (or your agent "
                "runtime) and new activity will appear here."
            ),
        }

    from tokenjam.core.onboard_verify import not_confirmed_cause

    persona = _detect_onboarded_persona(config)
    return {
        "name": "Onboarding signal",
        "level": "info",
        "message": (
            "Onboarded but no spans have been recorded yet. "
            + not_confirmed_cause(persona)
        ),
    }


def _print_check(check: dict) -> None:
    level = check["level"]
    icons = {"ok": "[green]\u2713[/green]", "warning": "[yellow]\u26a0[/yellow]",
             "error": "[red]\u2717[/red]", "info": "[blue]i[/blue]"}
    icon = icons.get(level, "?")
    # Check names/messages are plain text, not Rich markup \u2014 a literal
    # bracketed value inside one (e.g. the `[mcp_servers.tj]` TOML section
    # header) would otherwise be parsed as a markup tag and stripped.
    console.print(f"  {icon}  {escape(check['name'])}: {escape(check['message'])}")
