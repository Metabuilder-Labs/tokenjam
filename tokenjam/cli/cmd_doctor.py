from __future__ import annotations

import json

import click
import duckdb

from tokenjam.core.config import find_config_file, load_config
from tokenjam.utils.formatting import console


@click.command("doctor")
@click.option("--json", "output_json", is_flag=True)
@click.option(
    "--repair",
    is_flag=True,
    help="Attempt to fix issues that have a known repair path (e.g. rebuild the "
         "spans table when DuckDB column statistics are corrupt — see issue #56).",
)
@click.pass_context
def cmd_doctor(ctx: click.Context, output_json: bool, repair: bool) -> None:
    """Run health checks on tj configuration and environment."""
    config = ctx.obj["config"]
    checks: list[dict] = []

    # 1. Config file found and valid
    checks.append(_check_config())

    # 2. DuckDB file writable
    checks.append(_check_db(config))

    # 3. Ingest secret set
    checks.append(_check_ingest_secret(config))

    # 4. Prometheus configured
    checks.append(_check_prometheus(config))

    # 5. Schema validation vs capture
    checks.append(_check_schema_vs_capture(config))

    # 6. Drift configured but inactive
    checks.append(_check_drift_inactive(config, ctx.obj["db"]))

    # 7. Webhook URL security
    checks.extend(_check_webhook_security(config))

    # 8. Webhook domain allowlist
    checks.extend(_check_webhook_allowlist(config))

    # 9. DuckDB spans column-statistics corruption (issue #56)
    spans_stats_check = _check_spans_stats(ctx.obj["db"])
    checks.append(spans_stats_check)

    # 10. Live-span staleness — flags a stalled OTLP connection (issue #179)
    checks.append(_check_span_staleness(ctx.obj["db"]))

    # 11. Proxy base-URL wiring consistency (issue #219)
    checks.append(_check_proxy_wiring(config))

    # 12. MCP server wiring (issue #285)
    checks.append(_check_mcp_wiring(config))

    if output_json:
        click.echo(json.dumps(checks, default=str))
    else:
        for c in checks:
            _print_check(c)

    # --repair: attempt fixes for any check that exposed a repair_action
    if repair:
        _attempt_repairs(checks, ctx.obj["db"], output_json)

    has_errors = any(c["level"] == "error" for c in checks)
    has_warnings = any(c["level"] == "warning" for c in checks)
    if has_errors:
        ctx.exit(2)
    elif has_warnings:
        ctx.exit(1)
    else:
        ctx.exit(0)


def _check_config() -> dict:
    try:
        path = find_config_file()
        if path is None:
            return {"name": "Config file", "level": "error",
                    "message": "No config file found. Run `tj onboard` to create one."}
        load_config(str(path))
        return {"name": "Config file", "level": "ok",
                "message": f"Found and valid: {path}"}
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
                "message": f"Database accessible: {db_path}"}
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
    if config.security.ingest_secret:
        return {"name": "Ingest secret", "level": "ok",
                "message": "Ingest secret is configured."}
    return {"name": "Ingest secret", "level": "warning",
            "message": "No ingest secret set. API ingest endpoint is unprotected."}


def _check_prometheus(config: object) -> dict:
    if config.export.prometheus.enabled:
        return {"name": "Prometheus", "level": "ok",
                "message": f"Enabled on port {config.export.prometheus.port}"}
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
    except Exception:  # noqa: BLE001 — best-effort; never fail doctor on this
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
                       "See issue #56.",
            "repair_action": "rebuild_spans",
        }
    return {"name": "Spans column statistics", "level": "ok",
            "message": "Column statistics are consistent."}


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


def _attempt_repairs(checks: list[dict], db: object, output_json: bool) -> None:
    """Run repair actions for any check that flagged one."""
    from tokenjam.core.db import repair_spans_stats

    conn = getattr(db, "conn", None)
    for c in checks:
        action = c.get("repair_action")
        if not action:
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
    import shutil
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

    if has_codex or has_claude or has_project:
        found_locations = []
        if has_codex:
            found_locations.append("Codex (global)")
        if has_claude:
            found_locations.append("Claude Code (global)")
        if has_project:
            found_locations.append("project scope")
        return {
            "name": "MCP wiring",
            "level": "ok",
            "message": f"MCP server registered in {', '.join(found_locations)}.",
        }

    # If not registered anywhere, decide level based on agent presence
    codex_present = bool(shutil.which("codex")) or (Path.home() / ".codex").exists()
    claude_present = bool(shutil.which("claude")) or (Path.home() / ".claude").exists()

    if codex_present or claude_present:
        agents = []
        if claude_present:
            agents.append("Claude Code")
        if codex_present:
            agents.append("Codex")
        return {
            "name": "MCP wiring",
            "level": "warning",
            "message": (
                f"MCP server not registered in {', '.join(agents)}. Coding agents "
                "won't be able to query TokenJam telemetry. Run `tj onboard --claude-code` "
                "or `tj onboard --codex` to register it."
            ),
        }

    return {
        "name": "MCP wiring",
        "level": "info",
        "message": (
            "MCP server is not registered in Claude Code or Codex. Run `tj onboard "
            "--claude-code` or `tj onboard --codex` if using a coding agent."
        ),
    }


def _print_check(check: dict) -> None:
    level = check["level"]
    icons = {"ok": "[green]\u2713[/green]", "warning": "[yellow]\u26a0[/yellow]",
             "error": "[red]\u2717[/red]", "info": "[blue]i[/blue]"}
    icon = icons.get(level, "?")
    console.print(f"  {icon}  {check['name']}: {check['message']}")
