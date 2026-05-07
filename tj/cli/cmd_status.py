from __future__ import annotations

import json

import click

from tj.core.models import AlertFilters
from tj.utils.formatting import console, format_cost, format_tokens, status_icon
from tj.utils.time_parse import utcnow


@click.command("status")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def cmd_status(ctx: click.Context, agent: str | None, output_json: bool) -> None:
    """Show agent status overview."""
    db = ctx.obj["db"]
    agent_filter = agent or ctx.obj.get("agent")

    # Get all agents from recent sessions
    if agent_filter:
        agent_ids = [agent_filter]
    elif hasattr(db, "conn"):
        # Direct DB access
        rows = db.conn.execute(
            "SELECT DISTINCT agent_id FROM sessions WHERE agent_id IS NOT NULL ORDER BY agent_id"
        ).fetchall()
        agent_ids = [r[0] for r in rows]
    else:
        # API mode — discover agents from recent traces
        from tj.core.models import TraceFilters
        traces = db.get_traces(TraceFilters(limit=100))
        agent_ids = sorted({t.agent_id for t in traces if t.agent_id})

    if not agent_ids:
        if output_json:
            click.echo(json.dumps({"agents": [], "has_active_alerts": False}))
        else:
            console.print("[dim]No agents found. Run an instrumented agent first.[/dim]")
        return

    has_active_alerts = False
    agents_data = []

    for aid in agent_ids:
        session = None
        if hasattr(db, "conn"):
            # Direct DB access
            sessions = db.get_completed_sessions(aid, limit=1)
            active_rows = db.conn.execute(
                "SELECT * FROM sessions WHERE agent_id = $1 AND status = 'active' "
                "ORDER BY started_at DESC LIMIT 1",
                [aid],
            ).fetchall()
            if active_rows:
                cols = [d[0] for d in db.conn.description]
                from tj.core.db import _row_to_session
                session = _row_to_session(active_rows[0], cols)
            elif sessions:
                session = sessions[0]
        else:
            # API mode — limited session info
            sessions = db.get_completed_sessions(aid, limit=1)
            if sessions:
                session = sessions[0]

        today_cost = db.get_daily_cost(aid, utcnow().date())

        # Budget from config: per-agent overrides defaults
        config = ctx.obj["config"]
        agent_config = config.agents.get(aid)
        if agent_config and agent_config.budget.daily_usd is not None:
            daily_limit = agent_config.budget.daily_usd
        elif hasattr(config, "defaults") and config.defaults.budget.daily_usd is not None:
            daily_limit = config.defaults.budget.daily_usd
        else:
            daily_limit = None

        # Active alerts
        alerts = db.get_alerts(AlertFilters(agent_id=aid, unread=True, limit=50))
        active_alerts = [a for a in alerts if not a.acknowledged and not a.suppressed]
        if active_alerts:
            has_active_alerts = True

        agent_data = {
            "agent_id": aid,
            "status": session.status if session else "idle",
            "session_id": session.session_id if session else None,
            "cost_today": today_cost,
            "daily_limit": daily_limit,
            "input_tokens": session.input_tokens if session else 0,
            "output_tokens": session.output_tokens if session else 0,
            "tool_call_count": session.tool_call_count if session else 0,
            "error_count": session.error_count if session else 0,
            "active_alerts": len(active_alerts),
        }
        agents_data.append(agent_data)

        if not output_json:
            _print_agent_status(agent_data, active_alerts, session)

    if output_json:
        click.echo(json.dumps({
            "agents": agents_data,
            "has_active_alerts": has_active_alerts,
        }, default=str))

    ctx.exit(1 if has_active_alerts else 0)


def _print_agent_status(data: dict, active_alerts: list, session: object | None) -> None:
    status = data["status"]
    icon = status_icon(status)
    style = "green" if status == "active" else "dim"

    duration_str = ""
    if session and hasattr(session, "duration_seconds") and session.duration_seconds:
        secs = int(session.duration_seconds)
        mins, s = divmod(secs, 60)
        duration_str = f"   ({mins}m {s}s)"

    console.print(f"[{style}]{icon}[/] [bold]{data['agent_id']}[/bold]   "
                  f"{status}{duration_str}")
    console.print()

    cost_str = format_cost(data["cost_today"])
    if data["daily_limit"]:
        cost_str += f" / {format_cost(data['daily_limit'])} limit"
    console.print(f"  Cost today:     {cost_str}")

    in_tok = format_tokens(data["input_tokens"])
    out_tok = format_tokens(data["output_tokens"])
    console.print(f"  Tokens:         {in_tok} in / {out_tok} out")

    tool_str = str(data["tool_call_count"])
    if data["error_count"]:
        tool_str += f"  ({data['error_count']} failed)"
    console.print(f"  Tool calls:     {tool_str}")

    if data["session_id"]:
        console.print(f"  Active session: {data['session_id']}")

    console.print()
    for alert in active_alerts:
        from tj.utils.formatting import severity_colour
        colour = severity_colour(alert.severity.value)
        console.print(f"  [{colour}]{alert.title}[/]")

    if not active_alerts:
        console.print("  [green]No active alerts[/green]")

    console.print()
