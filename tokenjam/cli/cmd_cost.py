import click
import json
from tokenjam.core.cost import compute_cost_diff
from tokenjam.core.models import CostFilters
from tokenjam.utils.formatting import console, make_table, format_cost, format_tokens
from tokenjam.utils.time_parse import parse_since, utcnow


@click.command("cost")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("--since", default="7d", help="Time window (e.g. 1h, 7d, 2026-03-01)")
@click.option("--group-by", "group_by",
              type=click.Choice(["agent", "model", "day", "tool"]),
              default="day")
@click.option("--compare", "compare", default=None,
              help="Compare to a prior window. Accepts 'previous', 'last-week', "
                   "'last-month', 'last-7d', 'last-30d', or 'YYYY-MM-DD:YYYY-MM-DD'.")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def cmd_cost(ctx: click.Context, agent: str | None, since: str,
             group_by: str, compare: str | None, output_json: bool) -> None:
    """Show cost breakdown by agent, model, day, or tool."""
    db = ctx.obj["db"]
    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    # --compare branch: surface a diff against the prior window. This runs
    # before the regular grouped report and exits — the two outputs are
    # different shapes and combining them would be cluttered.
    if compare:
        if not hasattr(db, "conn"):
            raise click.ClickException(
                "--compare requires a direct DuckDB connection. Stop tj serve first."
            )
        until_dt = utcnow()
        try:
            diff = compute_cost_diff(db, since_dt, until_dt, compare, agent_id=agent)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--compare'") from exc
        if output_json:
            click.echo(json.dumps(_diff_to_dict(diff), default=str))
        else:
            _render_diff(diff)
        return

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


def _arrow(delta: float) -> str:
    if delta > 0:
        return "[red]▲[/red]"
    if delta < 0:
        return "[green]▼[/green]"
    return "·"


def _pct_str(pct: float | None) -> str:
    if pct is None:
        return "[dim](—)[/dim]"
    sign = "+" if pct >= 0 else ""
    return f"({sign}{pct:.1f}%)"


def _render_diff(diff) -> None:
    """
    Human-readable diff renderer. Plan-tier-aware rendering is deferred —
    this v1 renders dollar deltas; tj optimize already handles the
    subscription / local reframing in its own renderer.
    """
    cur = diff.current
    prev = diff.previous
    cur_days = max((cur.until - cur.since).days, 1)
    prev_days = max((prev.until - prev.since).days, 1)
    console.print(
        f"\n[bold]Current  ({cur.since.date()} → {cur.until.date()}, "
        f"{cur_days}d):[/bold]  "
        f"{cur.sessions} sessions, {format_tokens(cur.total_tokens)} tokens, "
        f"{format_cost(cur.total_cost_usd)}"
    )
    console.print(
        f"[bold]Previous ({prev.since.date()} → {prev.until.date()}, "
        f"{prev_days}d):[/bold] "
        f"{prev.sessions} sessions, {format_tokens(prev.total_tokens)} tokens, "
        f"{format_cost(prev.total_cost_usd)}"
    )
    console.print()
    console.print(
        f"  Cost delta:   {_arrow(diff.cost_delta_usd)} "
        f"[bold]{format_cost(abs(diff.cost_delta_usd))}[/bold] "
        f"{_pct_str(diff.cost_delta_pct)}"
    )
    console.print(
        f"  Token delta:  {_arrow(diff.tokens_delta)} "
        f"[bold]{format_tokens(abs(diff.tokens_delta))}[/bold] "
        f"{_pct_str(diff.tokens_delta_pct)}"
    )

    if diff.by_agent:
        console.print()
        console.print("  [bold]Top shifts by agent:[/bold]")
        for entry in diff.by_agent:
            console.print(
                f"    {_arrow(entry['delta'])} {entry['group']:<24} "
                f"{format_cost(entry['previous_cost'])} → "
                f"{format_cost(entry['current_cost'])}  "
                f"[dim]({'+' if entry['delta'] >= 0 else ''}"
                f"{format_cost(entry['delta'])})[/dim]"
            )

    if diff.by_model:
        console.print()
        console.print("  [bold]Top shifts by model:[/bold]")
        for entry in diff.by_model:
            console.print(
                f"    {_arrow(entry['delta'])} {entry['group']:<32} "
                f"{format_cost(entry['previous_cost'])} → "
                f"{format_cost(entry['current_cost'])}  "
                f"[dim]({'+' if entry['delta'] >= 0 else ''}"
                f"{format_cost(entry['delta'])})[/dim]"
            )

    console.print()


def _diff_to_dict(diff) -> dict:
    """JSON serialiser for CostDiff. Mirrors _render_diff's data."""
    def _wt(w):
        return {
            "since": w.since.isoformat(),
            "until": w.until.isoformat(),
            "sessions": w.sessions,
            "input_tokens": w.input_tokens,
            "output_tokens": w.output_tokens,
            "cache_tokens": w.cache_tokens,
            "total_tokens": w.total_tokens,
            "total_cost_usd": w.total_cost_usd,
        }
    return {
        "current": _wt(diff.current),
        "previous": _wt(diff.previous),
        "cost_delta_usd": diff.cost_delta_usd,
        "cost_delta_pct": diff.cost_delta_pct,
        "tokens_delta": diff.tokens_delta,
        "tokens_delta_pct": diff.tokens_delta_pct,
        "by_agent": diff.by_agent,
        "by_model": diff.by_model,
    }
