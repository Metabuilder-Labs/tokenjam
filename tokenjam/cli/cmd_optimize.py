"""`tj optimize` — surface cost-saving candidates and budget projections."""
from __future__ import annotations

import json

import click

from tokenjam.core.optimize import (
    MODEL_DOWNGRADE_CAVEAT,
    BudgetProjection,
    DowngradeFinding,
    OptimizeReport,
    build_report,
    report_to_dict,
)
from tokenjam.utils.formatting import (
    console,
    format_cost,
    format_tokens,
)
from tokenjam.utils.time_parse import parse_since, utcnow


@click.command("optimize")
@click.option("--agent", default=None, help="Scope to a specific agent_id.")
@click.option("--since", default="30d", help="Window for analysis (default 30d).")
@click.option("--only", type=click.Choice(["model", "budget"]), default=None,
              help="Run only one analyzer.")
@click.option("--budget", "budget_provider", default=None,
              help="Scope budget projection to a single provider (e.g. anthropic).")
@click.option("--budget-usd", type=float, default=None,
              help="Override the configured budget for this run.")
@click.option("--json", "output_json", is_flag=True,
              help="Emit machine-readable JSON.")
@click.pass_context
def cmd_optimize(
    ctx: click.Context,
    agent: str | None,
    since: str,
    only: str | None,
    budget_provider: str | None,
    budget_usd: float | None,
    output_json: bool,
) -> None:
    """Analyze recent usage for cost-saving candidates and budget exposure."""
    db = ctx.obj.get("db")
    config = ctx.obj.get("config")
    if db is None or config is None:
        raise click.ClickException("optimize requires a database connection.")

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    until_dt = utcnow()

    # If tj serve is running it holds the write-lock, and main.py hands us an
    # API-proxy backend that has no .conn. Open the DB ourselves read-only so
    # optimize works alongside a running server.
    conn = getattr(db, "conn", None)
    if conn is None:
        import duckdb
        from pathlib import Path as _Path
        db_path = str(_Path(config.storage.path).expanduser())
        try:
            ro_conn = duckdb.connect(db_path, read_only=True)
        except Exception as exc:
            raise click.ClickException(
                f"Could not open {db_path} read-only: {exc}"
            ) from exc

        class _RoShim:
            def __init__(self, c):
                self.conn = c
        db = _RoShim(ro_conn)
        conn = ro_conn
    row = conn.execute(
        "SELECT COUNT(*) FROM spans WHERE model IS NOT NULL"
    ).fetchone()
    if not row or not row[0]:
        if output_json:
            click.echo(json.dumps({
                "error": "no_data",
                "message": "No span data available — let TokenJam run for a few "
                           "days, or `tj backfill claude-code` if you use Claude Code.",
            }))
        else:
            console.print(
                "[yellow]No usage data found.[/yellow] "
                "[dim]Let TokenJam run for a few days, or — if you use "
                "Claude Code — try [bold]tj backfill claude-code[/bold] to "
                "ingest historical sessions.[/dim]"
            )
        return

    report = build_report(
        db=db,
        config=config,
        since=since_dt,
        until=until_dt,
        agent_id=agent,
        only=only,
        budget_provider_filter=budget_provider,
        budget_usd_override=budget_usd,
    )

    if output_json:
        click.echo(json.dumps(report_to_dict(report), default=str))
        return

    _render_report(report, agent=agent)


# ---------------------------------------------------------------------------
# Human-readable renderer
# ---------------------------------------------------------------------------

def _render_report(report: OptimizeReport, agent: str | None) -> None:
    w = report.window
    scope_tag = f", {agent}" if agent else ""
    days_int = max(int(round(w.days)), 1)
    console.print(
        f"\nAnalyzing [bold]{w.sessions}[/bold] sessions, "
        f"[bold]{format_tokens(w.total_tokens)}[/bold] tokens, "
        f"[bold]{format_cost(w.total_cost_usd)}[/bold] spend "
        f"(last {days_int}d{scope_tag})…\n"
    )

    if w.sessions == 0:
        console.print("[dim]No sessions in window.[/dim]")
        return

    for note in report.notes:
        console.print(f"  [yellow]![/yellow] {note}")
    if report.notes:
        console.print()

    if report.downgrade is not None:
        _render_downgrade(report.downgrade)
        console.print()

    for proj in report.budgets:
        _render_budget(proj)
        console.print()

    if report.downgrade is None and not report.budgets:
        console.print(
            "[dim]No candidates flagged in this window. Either spend is small or "
            "all sessions already use a cost-effective model.[/dim]"
        )


def _render_downgrade(d: DowngradeFinding) -> None:
    console.print(
        f"  [bold]① Model downgrade:[/bold] "
        f"{d.percent_of_sessions:.0f}% of sessions match a smaller-model "
        f"candidate shape"
    )
    console.print(
        f"     • {d.candidate_sessions} of {d.total_sessions} sessions matched "
        f"structural heuristics"
    )
    console.print(
        f"     • Would have cost ~{format_cost(d.alternative_cost_usd)} on the "
        f"smaller model vs {format_cost(d.actual_cost_usd)} actual (in window)"
    )
    console.print(
        f"     • Projected savings if pattern holds: "
        f"[bold]{format_cost(d.monthly_savings_usd)}/mo[/bold]"
    )
    if d.suggestions:
        pairs = ", ".join(f"{k} → {v}" for k, v in d.suggestions.items())
        console.print(f"     • Pattern: [dim]{pairs}[/dim]")

    if d.examples:
        console.print()
        console.print("     [dim]Examples:[/dim]")
        for ex in d.examples:
            dur = f"{ex.duration_seconds:.1f}s" if ex.duration_seconds else "—"
            console.print(
                f"       [dim]{ex.trace_id[:8]}..[/dim]  "
                f"{ex.tool_calls} tool calls   {dur}   "
                f"{format_cost(ex.cost_usd)}  ({ex.model})"
            )
    console.print()
    console.print(
        f"     [yellow]![/yellow] [italic]{MODEL_DOWNGRADE_CAVEAT}[/italic]"
    )


def _render_budget(p: BudgetProjection) -> None:
    headline = f"  [bold]② Budget projection ({p.provider}, " \
               f"{format_cost(p.budget_usd)}/cycle):[/bold] "

    if not p.over_budget:
        unused_pct = max(0, int(round(100 * (1 - p.projected_cycle_total / p.budget_usd))))
        console.print(headline + "comfortably within budget")
        console.print(
            f"     Run rate "
            f"[bold]{format_cost(p.monthly_run_rate_usd)}/mo[/bold] — "
            f"{unused_pct}% of cycle budget unused."
        )
        return

    console.print(headline + "[red]projected to exceed cycle budget[/red]")
    console.print(
        f"     • Monthly run rate: "
        f"[bold]{format_cost(p.monthly_run_rate_usd)}[/bold] "
        f"({p.monthly_run_rate_usd / p.budget_usd:.1f}× the budget)"
    )
    if p.exhaustion_date:
        console.print(
            f"     • At current pace, budget exhausted on "
            f"[bold]{p.exhaustion_date.strftime('%Y-%m-%d')}[/bold] "
            f"({p.days_until_exhaustion:.1f} day(s) from now)"
        )
    console.print(f"     • Days remaining in cycle: {p.days_remaining:.0f}")
    console.print(
        f"     • Projected cycle total: "
        f"{format_cost(p.projected_cycle_total)}, "
        f"overage: [red]{format_cost(p.projected_overage_usd)}[/red]"
    )
    if p.downgrade_run_rate_usd is not None and p.downgrade_run_rate_usd < p.monthly_run_rate_usd:
        console.print(
            f"     • With model-downgrade pattern: run rate drops to "
            f"[bold]{format_cost(p.downgrade_run_rate_usd)}/mo[/bold]"
        )
    if p.applies_to_services:
        console.print(
            f"     [dim]Counted services: {', '.join(p.applies_to_services)}[/dim]"
        )
