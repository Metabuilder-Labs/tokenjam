from __future__ import annotations

import json

import click

from tj.utils.formatting import console, make_table
from tj.utils.time_parse import parse_since


@click.command("tools")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("--since", default="24h", help="Time window (e.g. 1h, 7d)")
@click.option("--name", "tool_name", default=None, help="Filter to specific tool")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def cmd_tools(ctx: click.Context, agent: str | None, since: str,
              tool_name: str | None, output_json: bool) -> None:
    """Show tool call summary."""
    db = ctx.obj["db"]
    agent_filter = agent or ctx.obj.get("agent")
    since_dt = parse_since(since)

    rows = db.get_tool_calls(agent_filter, since_dt, tool_name)

    if output_json:
        click.echo(json.dumps(rows, default=str))
        return

    if not rows:
        console.print("[dim]No tool calls found for the given filters.[/dim]")
        return

    table = make_table("TOOL", "AGENT", "CALLS", "AVG DUR")
    for r in rows:
        call_count = r["call_count"]
        total_dur = r["total_duration_ms"]
        avg_dur = f"{total_dur / call_count:.0f}ms" if call_count > 0 else "-"
        table.add_row(
            r["tool_name"],
            r.get("agent_id") or "-",
            str(call_count),
            avg_dur,
        )
    console.print(table)
