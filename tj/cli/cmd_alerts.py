import json

import click

from tj.core.models import AlertFilters, AlertType, Severity
from tj.utils.formatting import console, make_table, severity_colour
from tj.utils.time_parse import parse_since


@click.command("alerts")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("--since", default="24h", help="Time window (e.g. 1h, 24h, 7d)")
@click.option(
    "--severity",
    type=click.Choice(["critical", "warning", "info"]),
    default=None,
    help="Filter by minimum severity",
)
@click.option("--type", "alert_type", default=None, help="Filter by alert type")
@click.option("--unread", is_flag=True, help="Show only unacknowledged alerts")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.pass_context
def cmd_alerts(
    ctx: click.Context,
    agent: str | None,
    since: str,
    severity: str | None,
    alert_type: str | None,
    unread: bool,
    output_json: bool,
) -> None:
    """Show alert history."""
    db = ctx.obj["db"]
    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc
    filters = AlertFilters(
        agent_id=agent,
        since=since_dt,
        severity=Severity(severity) if severity else None,
        type=AlertType(alert_type) if alert_type else None,
        unread=unread,
    )
    alerts = db.get_alerts(filters)

    if output_json:
        click.echo(json.dumps(
            [
                {
                    "alert_id": a.alert_id,
                    "fired_at": a.fired_at.isoformat(),
                    "type": a.type.value,
                    "severity": a.severity.value,
                    "title": a.title,
                    "detail": a.detail,
                    "agent_id": a.agent_id,
                    "session_id": a.session_id,
                    "span_id": a.span_id,
                    "acknowledged": a.acknowledged,
                    "suppressed": a.suppressed,
                }
                for a in alerts
            ],
            default=str,
        ))
        return

    if not alerts:
        console.print("[dim]No alerts found for the given filters.[/dim]")
        return

    critical_count = sum(1 for a in alerts if a.severity == Severity.CRITICAL)
    warning_count = sum(1 for a in alerts if a.severity == Severity.WARNING)
    console.print(
        f"[bold]Alerts \u2014 last {since}[/bold]   "
        f"({len(alerts)} total: {critical_count} critical, {warning_count} warning)"
    )

    table = make_table("TIME", "SEVERITY", "TYPE", "AGENT", "DETAIL")
    for a in alerts:
        time_str = a.fired_at.strftime("%H:%M:%S")
        colour = severity_colour(a.severity.value)
        detail_msg = a.detail.get("message", a.title)
        if len(detail_msg) > 60:
            detail_msg = detail_msg[:57] + "..."
        table.add_row(
            time_str,
            f"[{colour}]{a.severity.value.upper()}[/]",
            a.type.value,
            a.agent_id or "-",
            detail_msg,
        )
    console.print(table)
