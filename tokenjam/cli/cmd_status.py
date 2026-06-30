from __future__ import annotations

import json

import click

from tokenjam.core.models import AlertFilters
from tokenjam.utils.formatting import console, format_cost, format_tokens, status_icon
from tokenjam.utils.time_parse import utcnow


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
        # Local DB mode
        agent_ids = db.get_distinct_agent_ids()
    else:
        # API mode — discover agents from recent traces
        from tokenjam.core.models import TraceFilters
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
            # Local DB mode — prefer the live active session, fall back to the
            # most recent completed one.
            sessions = db.get_completed_sessions(aid, limit=1)
            session = db.get_active_session(aid)
            if session is None and sessions:
                session = sessions[0]
        else:
            # API mode — limited session info
            sessions = db.get_completed_sessions(aid, limit=1)
            if sessions:
                session = sessions[0]

        # Active (compute) time = sum of span durations; distinct from the
        # wall-clock duration_seconds, which spans days for resumed sessions
        # (issue #147).
        active_seconds = None
        if session is not None and hasattr(db, "conn"):
            active_seconds = db.get_session_active_seconds(session.session_id)

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
            "duration_seconds": session.duration_seconds if session else None,
            "active_seconds": active_seconds,
        }
        agents_data.append(agent_data)

        if not output_json:
            _print_agent_status(agent_data, active_alerts, session)

    # Count sessions with plan_tier='unknown' so the user knows to reconfigure.
    # Informational only — exit code stays driven by alert state.
    unknown_count = 0
    if hasattr(db, "conn"):
        try:
            unknown_count = db.count_unknown_plan_tier_sessions()
        except Exception:
            unknown_count = 0

    if output_json:
        click.echo(json.dumps({
            "agents": agents_data,
            "has_active_alerts": has_active_alerts,
            "unknown_plan_tier_sessions": unknown_count,
        }, default=str))
    elif unknown_count > 0:
        console.print(
            f"[dim]Note: {unknown_count} session(s) have unknown plan tier. "
            f"Run [bold]tj onboard --claude-code --reconfigure[/bold] "
            f"(or [bold]--codex[/bold]) to set it.[/dim]"
        )

    ctx.exit(1 if has_active_alerts else 0)


def _fmt_dur(seconds: float | None, *, coarse: bool = False) -> str:
    """Human duration. coarse=True caps at days/hours for long wall-clock spans."""
    if seconds is None:
        return "-"
    secs = int(seconds)
    if coarse and secs >= 3600:
        mins = secs // 60
        d, rem = divmod(mins, 1440)
        h, m = divmod(rem, 60)
        return f"{d}d {h}h" if d else f"{h}h {m}m"
    mins, s = divmod(secs, 60)
    if mins >= 60:
        h, m = divmod(mins, 60)
        return f"{h}h {m}m"
    return f"{mins}m {s}s"


def _print_agent_status(data: dict, active_alerts: list, session: object | None) -> None:
    status = data["status"]
    icon = status_icon(status)
    style = "green" if status == "active" else "dim"

    console.print(f"[{style}]{icon}[/] [bold]{data['agent_id']}[/bold]   "
                  f"{status}")
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

    # Active = compute time (Σ span durations); Elapsed = wall-clock, which can
    # span days for resumed sessions (issue #147).
    if data.get("active_seconds") is not None or data.get("duration_seconds") is not None:
        parts = []
        if data.get("active_seconds") is not None:
            parts.append(f"active {_fmt_dur(data['active_seconds'])}")
        if data.get("duration_seconds") is not None:
            parts.append(f"[dim]elapsed {_fmt_dur(data['duration_seconds'], coarse=True)}[/dim]")
        console.print(f"  Duration:       {' · '.join(parts)}")

    if data["session_id"]:
        console.print(f"  Active session: {data['session_id']}")

    console.print()
    for alert in active_alerts:
        from tokenjam.utils.formatting import severity_colour
        colour = severity_colour(alert.severity.value)
        console.print(f"  [{colour}]{alert.title}[/]")

    if not active_alerts:
        console.print("  [green]No active alerts[/green]")

    console.print()
