import click
import json
from tj.core.models import CostFilters
from tj.utils.formatting import console, make_table, format_cost, format_tokens
from tj.utils.time_parse import parse_since


@click.command("cost")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("--since", default="7d", help="Time window (e.g. 1h, 7d, 2026-03-01)")
@click.option("--group-by", "group_by",
              type=click.Choice(["agent", "model", "day", "tool"]),
              default="day")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def cmd_cost(ctx: click.Context, agent: str | None, since: str,
             group_by: str, output_json: bool) -> None:
    """Show cost breakdown by agent, model, day, or tool."""
    db = ctx.obj["db"]
    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc
    filters = CostFilters(
        agent_id=agent,
        since=since_dt,
        group_by=group_by,
    )
    rows = db.get_cost_summary(filters)
    total = sum(r.cost_usd for r in rows)

    if output_json:
        click.echo(json.dumps({
            "rows": [vars(r) for r in rows],
            "total_cost_usd": total,
        }, default=str))
        return

    if not rows:
        console.print("[dim]No cost data found for the given filters.[/dim]")
        return

    if group_by == "day":
        table = make_table("DATE", "AGENT", "MODEL", "TOKENS IN", "TOKENS OUT", "COST")
        for r in rows:
            table.add_row(
                r.group,
                r.agent_id or "-",
                r.model or "-",
                format_tokens(r.input_tokens),
                format_tokens(r.output_tokens),
                format_cost(r.cost_usd),
            )
    elif group_by == "agent":
        table = make_table("AGENT", "MODEL", "TOKENS IN", "TOKENS OUT", "COST")
        for r in rows:
            table.add_row(
                r.group,
                r.model or "-",
                format_tokens(r.input_tokens),
                format_tokens(r.output_tokens),
                format_cost(r.cost_usd),
            )
    elif group_by == "model":
        table = make_table("MODEL", "TOKENS IN", "TOKENS OUT", "COST")
        for r in rows:
            table.add_row(
                r.group,
                format_tokens(r.input_tokens),
                format_tokens(r.output_tokens),
                format_cost(r.cost_usd),
            )
    elif group_by == "tool":
        table = make_table("TOOL", "COST")
        for r in rows:
            table.add_row(
                r.group,
                format_cost(r.cost_usd),
            )

    if group_by == "day":
        table.add_row("", "", "", "", "[bold]TOTAL[/bold]", f"[bold]{format_cost(total)}[/bold]")
    elif group_by == "agent":
        table.add_row("", "", "", "[bold]TOTAL[/bold]", f"[bold]{format_cost(total)}[/bold]")
    elif group_by == "model":
        table.add_row("", "", "[bold]TOTAL[/bold]", f"[bold]{format_cost(total)}[/bold]")
    elif group_by == "tool":
        table.add_row("[bold]TOTAL[/bold]", f"[bold]{format_cost(total)}[/bold]")

    console.print(table)
