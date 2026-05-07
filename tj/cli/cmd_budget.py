from __future__ import annotations

from pathlib import Path

import click

from tj.core.config import (
    AgentConfig,
    find_config_file,
    resolve_effective_budget,
    validate_budget_value,
    write_config,
)
from tj.utils.formatting import console


@click.command("budget")
@click.option("--agent", default=None, help="Agent ID to target (omit for global defaults)")
@click.option("--daily", "daily_usd", type=float, default=None,
              help="Daily budget in USD (0 = remove limit)")
@click.option("--session", "session_usd", type=float, default=None,
              help="Per-session budget in USD (0 = remove limit)")
@click.pass_context
def cmd_budget(
    ctx: click.Context,
    agent: str | None,
    daily_usd: float | None,
    session_usd: float | None,
) -> None:
    """View or set cost budgets for agents."""
    config = ctx.obj["config"]
    writing = daily_usd is not None or session_usd is not None

    if not writing:
        _show_budgets(config, ctx.obj.get("db"))
        return

    # Write mode — find config file on disk
    config_path_str = find_config_file()
    if config_path_str is None:
        raise click.ClickException(
            "No config file found. Run 'tj onboard' to create one."
        )

    if agent:
        if agent not in config.agents:
            config.agents[agent] = AgentConfig()
        budget = config.agents[agent].budget
        scope = f"agent '{agent}'"
    else:
        budget = config.defaults.budget
        scope = "global defaults"

    try:
        if daily_usd is not None:
            budget.daily_usd = validate_budget_value(daily_usd, "daily_usd")
        if session_usd is not None:
            budget.session_usd = validate_budget_value(session_usd, "session_usd")
    except ValueError as e:
        raise click.ClickException(str(e))

    write_config(config, Path(config_path_str))

    console.print(f"[green]\u2713[/green] Budget updated for {scope}")
    if daily_usd is not None:
        val = f"${daily_usd:.2f}" if daily_usd > 0 else "no limit"
        console.print(f"  Daily:   {val}")
    if session_usd is not None:
        val = f"${session_usd:.2f}" if session_usd > 0 else "no limit"
        console.print(f"  Session: {val}")


def _show_budgets(config, db) -> None:
    from rich.table import Table

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Scope")
    table.add_column("Daily", justify="right")
    table.add_column("Session", justify="right")

    def _fmt(val: float | None) -> str:
        return f"${val:.2f}" if val is not None else "[dim]no limit[/dim]"

    def _fmt_effective(raw: float | None, effective: float | None) -> str:
        if raw is not None:
            return f"${raw:.2f}"
        if effective is not None:
            return f"[dim]${effective:.2f}[/dim] [dim](default)[/dim]"
        return "[dim]no limit[/dim]"

    table.add_row(
        "[bold]defaults[/bold]",
        _fmt(config.defaults.budget.daily_usd),
        _fmt(config.defaults.budget.session_usd),
    )

    # Merge agent IDs from config + DB-observed agents
    agent_ids = set(config.agents)
    if db is not None and hasattr(db, "conn"):
        rows = db.conn.execute(
            "SELECT DISTINCT agent_id FROM sessions ORDER BY agent_id"
        ).fetchall()
        agent_ids |= {r[0] for r in rows}

    for agent_id in sorted(agent_ids):
        agent_cfg = config.agents.get(agent_id)
        eff = resolve_effective_budget(agent_id, config)
        raw_daily = agent_cfg.budget.daily_usd if agent_cfg else None
        raw_session = agent_cfg.budget.session_usd if agent_cfg else None
        table.add_row(
            agent_id,
            _fmt_effective(raw_daily, eff.daily_usd),
            _fmt_effective(raw_session, eff.session_usd),
        )

    if not agent_ids:
        table.add_row("[dim]no agents configured[/dim]", "", "")

    console.print(table)
