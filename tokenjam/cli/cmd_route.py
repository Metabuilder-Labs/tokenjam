"""`tj route` — compile advisory router configs from the downsize findings.

The OSS routing exit ramp (issue #207): `tj route export --target ccr|litellm`
writes a router config the user drops into their *own* router by hand. It
mirrors the `tj optimize --export-config` (claude_code) doctrine exactly —
writes ONLY to `~/.config/tokenjam/exports/`, never touches an external config,
and embeds the evidence level + derivation window + derived-at + the
MODEL_DOWNGRADE_CAVEAT in every export. Rules are advisory L1 ("structural
match, review before applying"), never "safe" (CLAUDE.md Critical Rule 14).

`tj route export --check` diffs the current findings against the last export and
flags staleness, without writing anything.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import click

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.framing import dominant_plan, plan_tier_mix, pricing_mode_for
from tokenjam.core.optimize import OptimizeReport, build_report, report_from_dict
from tokenjam.utils.formatting import console
from tokenjam.utils.time_parse import parse_since, utcnow

_TARGETS = ("ccr", "litellm")
_EXT = {"ccr": "jsonc", "litellm": "yaml"}


def _exports_dir() -> Path:
    return Path.home() / ".config" / "tokenjam" / "exports"


@click.group("route")
def cmd_route() -> None:
    """Compile advisory router configs from TokenJam's downsize findings."""


@cmd_route.command("export")
@click.option("--target", type=click.Choice(_TARGETS), default=None,
              help="Router config to emit (ccr = claude-code-router, "
                   "litellm = LiteLLM router).")
@click.option("--check", is_flag=True, default=False,
              help="Report whether the last export is stale vs current "
                   "findings. Writes nothing.")
@click.option("--agent", default=None, help="Scope to a specific agent_id.")
@click.option("--since", default="30d",
              help="Lookback window for the findings (e.g. 7d, 30d).")
@json_option
@click.pass_context
def cmd_route_export(
    ctx: click.Context,
    target: str | None,
    check: bool,
    agent: str | None,
    since: str,
    output_json_flag: bool,
) -> None:
    """Write (or --check) an advisory router config from the downsize findings."""
    output_json = resolve_output_json(ctx, output_json_flag)
    if not check and target is None:
        raise click.UsageError("Pass --target ccr|litellm (or --check).")

    downgrade, pricing_mode, plan_tier, since_label, until_label = _resolve_finding(
        ctx, agent=agent, since=since,
    )

    if check:
        _run_check(target, downgrade, output_json)
        return

    body, ext = _render(
        target, downgrade=downgrade, pricing_mode=pricing_mode,
        plan_tier=plan_tier, since=since_label, until=until_label, agent_id=agent,
    )

    out_dir = _exports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    out_path = out_dir / f"{target}-{today}.{ext}"
    out_path.write_text(body)
    _write_manifest(target, downgrade, str(out_path), since_label, until_label)

    if output_json:
        click.echo(json.dumps({
            "target": target,
            "path": str(out_path),
            "plan_tier": plan_tier,
            "pricing_mode": pricing_mode,
            "rule_count": len(downgrade.suggestions) if downgrade else 0,
        }, default=str))
        return

    n = len(downgrade.suggestions) if downgrade else 0
    console.print(f"[green]✓[/green] {target} config written to [bold]{out_path}[/bold] "
                  f"({n} advisory rule{'s' if n != 1 else ''}).")
    console.print(
        "\nOpen the file and translate the recommendations into your own router "
        "config by hand.\n[dim]TokenJam does not enforce these rules — they are "
        "advisory (evidence level L1), never \"safe\". The export is a "
        "recommendation, not an active routing config.[/dim]"
    )


# ---------------------------------------------------------------------------
# Finding resolution (mirrors cmd_optimize's dual direct-conn / API-shim path)
# ---------------------------------------------------------------------------

def _resolve_finding(ctx, *, agent, since):
    """Return (downgrade, pricing_mode, plan_tier, since_label, until_label)."""
    db = ctx.obj.get("db") if ctx.obj else None
    config = ctx.obj.get("config") if ctx.obj else None
    if db is None or config is None:
        raise click.ClickException("route requires a database connection.")

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc
    until_dt = utcnow()
    since_label = since
    until_label = until_dt.date().isoformat()

    conn = getattr(db, "conn", None)
    if conn is None:
        # API-shim path: daemon holds the DB lock.
        from tokenjam.core.api_backend import ApiBackend
        if not isinstance(db, ApiBackend):
            raise click.ClickException(
                "route requires either a direct DuckDB connection or a running "
                "tj serve at the configured api.{host,port}."
            )
        try:
            report_dict = db.fetch_optimize_report(
                since=since, agent_id=agent, findings=["downsize"],
            )
        except Exception as exc:
            raise click.ClickException(
                f"Failed to fetch optimize report from tj serve: {exc}"
            ) from exc
        report: OptimizeReport = report_from_dict(report_dict)
        plan_mix = report_dict.get("plan_tier_mix") or {}
    else:
        report = build_report(
            db=db, config=config, since=since_dt, until=until_dt,
            agent_id=agent, findings=["downsize"],
        )
        plan_mix = plan_tier_mix(conn, since_dt, until_dt, agent)

    dominant = dominant_plan(plan_mix)
    return report.downgrade, pricing_mode_for(dominant), dominant, since_label, until_label


def _render(target, *, downgrade, pricing_mode, plan_tier, since, until, agent_id):
    if target == "ccr":
        from tokenjam.core.export.ccr import render_ccr_config
        body = render_ccr_config(
            downgrade=downgrade, pricing_mode=pricing_mode, plan_tier=plan_tier,
            since=since, until=until, agent_id=agent_id,
        )
    elif target == "litellm":
        from tokenjam.core.export.litellm import render_litellm_config
        body = render_litellm_config(
            downgrade=downgrade, pricing_mode=pricing_mode, plan_tier=plan_tier,
            since=since, until=until, agent_id=agent_id,
        )
    else:  # pragma: no cover — Click's Choice() constrains this
        raise click.ClickException(f"Unknown export target: {target}")
    return body, _EXT[target]


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def _fingerprint(downgrade) -> str:
    """Stable hash of the finding content that an export encodes."""
    payload = {
        "suggestions": dict(sorted(downgrade.suggestions.items())) if downgrade else {},
        "candidate_sessions": downgrade.candidate_sessions if downgrade else 0,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _manifest_path(target: str) -> Path:
    return _exports_dir() / f"{target}.manifest.json"


def _write_manifest(target, downgrade, path, since, until) -> None:
    _manifest_path(target).write_text(json.dumps({
        "target": target,
        "path": path,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "since": since,
        "until": until,
        "fingerprint": _fingerprint(downgrade),
    }, indent=2))


def _run_check(target, downgrade, output_json) -> None:
    """Report staleness for the requested target (or all targets)."""
    targets = [target] if target else list(_TARGETS)
    current = _fingerprint(downgrade)
    results = []
    for t in targets:
        mpath = _manifest_path(t)
        if not mpath.exists():
            results.append({"target": t, "status": "no_export"})
            continue
        try:
            manifest = json.loads(mpath.read_text())
        except (OSError, json.JSONDecodeError):
            results.append({"target": t, "status": "no_export"})
            continue
        stale = manifest.get("fingerprint") != current
        results.append({
            "target": t,
            "status": "stale" if stale else "current",
            "last_export": manifest.get("generated_at"),
        })

    if output_json:
        click.echo(json.dumps({"results": results}, default=str))
        return

    for r in results:
        if r["status"] == "no_export":
            console.print(f"[dim]{r['target']}:[/dim] no export yet — run "
                          f"[bold]tj route export --target {r['target']}[/bold].")
        elif r["status"] == "stale":
            console.print(f"[yellow]{r['target']}: STALE[/yellow] — findings changed "
                          f"since the last export ({r['last_export']}). "
                          f"Re-run [bold]tj route export --target {r['target']}[/bold].")
        else:
            console.print(f"[green]{r['target']}: up to date[/green] "
                          f"(last export {r['last_export']}).")
