from __future__ import annotations

import json

import click
import duckdb

from tj.core.config import find_config_file, load_config
from tj.utils.formatting import console


@click.command("doctor")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def cmd_doctor(ctx: click.Context, output_json: bool) -> None:
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

    if output_json:
        click.echo(json.dumps(checks, default=str))
    else:
        for c in checks:
            _print_check(c)

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
    try:
        from pathlib import Path
        db_path = Path(config.storage.path).expanduser()
        conn = duckdb.connect(str(db_path))
        conn.close()
        return {"name": "DuckDB writable", "level": "ok",
                "message": f"Database accessible: {db_path}"}
    except Exception as e:
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
    for agent_id, ac in config.agents.items():
        if ac.drift.enabled:
            count = db.get_completed_session_count(agent_id)
            if count < ac.drift.baseline_sessions:
                return {"name": "Drift detection", "level": "warning",
                        "message": f"Agent '{agent_id}' has drift enabled but only "
                                   f"{count}/{ac.drift.baseline_sessions} baseline sessions."}
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


def _is_local_url(url: str) -> bool:
    from urllib.parse import urlparse
    hostname = urlparse(url).hostname
    return hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0") if hostname else False


def _print_check(check: dict) -> None:
    level = check["level"]
    icons = {"ok": "[green]\u2713[/green]", "warning": "[yellow]\u26a0[/yellow]",
             "error": "[red]\u2717[/red]", "info": "[blue]i[/blue]"}
    icon = icons.get(level, "?")
    console.print(f"  {icon}  {check['name']}: {check['message']}")
