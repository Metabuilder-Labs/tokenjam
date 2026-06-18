from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import click

from tokenjam.core.config import (
    AgentConfig,
    find_config_file,
    resolve_effective_budget,
    validate_budget_value,
    write_config,
)
from tokenjam.utils.formatting import console


class _BudgetRow(TypedDict):
    scope: str
    agent_id: str | None
    daily_usd: float | None
    session_usd: float | None
    effective_daily_usd: float | None
    effective_session_usd: float | None


@click.command("budget")
@click.option("--agent", default=None, help="Agent ID to target (omit for global defaults)")
@click.option("--daily", "daily_usd", type=float, default=None,
              help="Daily budget in USD (0 = remove limit)")
@click.option("--session", "session_usd", type=float, default=None,
              help="Per-session budget in USD (0 = remove limit)")
@click.option("--json", "output_json", is_flag=True,
              help="Emit machine-readable JSON.")
@click.pass_context
def cmd_budget(
    ctx: click.Context,
    agent: str | None,
    daily_usd: float | None,
    session_usd: float | None,
    output_json: bool,
) -> None:
    """View or set cost budgets for agents."""
    config = ctx.obj["config"]
    writing = daily_usd is not None or session_usd is not None

    if not writing:
        _show_budgets(config, ctx.obj.get("db"), output_json)
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

    if output_json:
        effective = resolve_effective_budget(agent, config) if agent else config.defaults.budget
        click.echo(json.dumps({
            "scope": "agent" if agent else "defaults",
            "agent_id": agent,
            "daily_usd": budget.daily_usd,
            "session_usd": budget.session_usd,
            "effective_daily_usd": effective.daily_usd,
            "effective_session_usd": effective.session_usd,
        }))
        return

    console.print(f"[green]\u2713[/green] Budget updated for {scope}")
    if daily_usd is not None:
        val = f"${daily_usd:.2f}" if daily_usd > 0 else "no limit"
        console.print(f"  Daily:   {val}")
    if session_usd is not None:
        val = f"${session_usd:.2f}" if session_usd > 0 else "no limit"
        console.print(f"  Session: {val}")


def _budget_rows(config, db) -> list[_BudgetRow]:
    rows: list[_BudgetRow] = [{
        "scope": "defaults",
        "agent_id": None,
        "daily_usd": config.defaults.budget.daily_usd,
        "session_usd": config.defaults.budget.session_usd,
        "effective_daily_usd": config.defaults.budget.daily_usd,
        "effective_session_usd": config.defaults.budget.session_usd,
    }]

    # Merge agent IDs from config + DB-observed agents. Two paths:
    # - Direct DB: pull distinct agent_ids from sessions table.
    # - API shim (daemon up, no db.conn): derive from recent traces.
    #   Without this fallback, `tj budget` silently misses every agent
    #   that hasn't been declared in tj.toml when the daemon is running
    #   (#68 §11). Mirrors the pattern already used by cmd_status.
    agent_ids = set(config.agents)
    if db is not None:
        if hasattr(db, "conn"):
            rows_from_db = db.conn.execute(
                "SELECT DISTINCT agent_id FROM sessions ORDER BY agent_id"
            ).fetchall()
            agent_ids |= {r[0] for r in rows_from_db if r[0]}
        else:
            try:
                from tokenjam.core.models import TraceFilters
                traces = db.get_traces(TraceFilters(limit=200))
                agent_ids |= {t.agent_id for t in traces if t.agent_id}
            except Exception:
                # API call failed (server down, transient error) —
                # render what we have from config rather than crash.
                pass

    for agent_id in sorted(agent_ids):
        agent_cfg = config.agents.get(agent_id)
        eff = resolve_effective_budget(agent_id, config)
        rows.append({
            "scope": "agent",
            "agent_id": agent_id,
            "daily_usd": agent_cfg.budget.daily_usd if agent_cfg else None,
            "session_usd": agent_cfg.budget.session_usd if agent_cfg else None,
            "effective_daily_usd": eff.daily_usd,
            "effective_session_usd": eff.session_usd,
        })

    return rows


def _show_budgets(config, db, output_json: bool = False) -> None:
    rows = _budget_rows(config, db)
    if output_json:
        click.echo(json.dumps(rows))
        return

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

    defaults = rows[0]
    table.add_row(
        "[bold]defaults[/bold]",
        _fmt(defaults["daily_usd"]),
        _fmt(defaults["session_usd"]),
    )

    for row in rows[1:]:
        table.add_row(
            str(row["agent_id"]),
            _fmt_effective(row["daily_usd"], row["effective_daily_usd"]),
            _fmt_effective(row["session_usd"], row["effective_session_usd"]),
        )

    if len(rows) == 1:
        table.add_row("[dim]no agents configured[/dim]", "", "")

    console.print(table)
