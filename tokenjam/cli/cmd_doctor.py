from __future__ import annotations

import json

import click
import duckdb
from rich.markup import escape

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.config import find_config_file, load_config
from tokenjam.utils.formatting import console, display_path


@click.command("doctor")
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
    """Run health checks on tj configuration and environment."""
    output_json = resolve_output_json(ctx, output_json_flag)
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

    # 6b. Prompt capture off — trim / cache-recommend / reuse degraded
    checks.append(_check_capture_prompts(config))

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

    # 13. Schema integrity — recorded-but-unlanded migration (issue #55)
    checks.append(_check_schema_integrity(ctx.obj["db"]))

    # 14. Claude Code statusline wiring (issue #59) — the zero-token surface
    checks.append(_check_statusline_wiring(config))

    # 15. Onboarded-but-silent — first-signal diagnosis (issue #80)
    checks.append(_check_onboarding_first_signal(config, ctx.obj["db"]))

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
    from tokenjam.core.db import (
        ensure_expected_columns,
        ensure_expected_tables,
        repair_spans_stats,
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
        return "claude_code"
    if "codex_exec" in agent_ids:
        return "codex"
    if _tj_statusline_wired():
        return "claude_code"
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
