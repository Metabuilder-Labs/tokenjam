"""`tj savings` — show tokens the `tj hook cap-output` hook reclaimed.

Reads the append-only JSONL sink (never the DB) and reports cumulative reclaimed
tokens — the ACTION half of tj's measure→act→prove loop (`tj context` measures
where quota goes; this proves what the hook clawed back).

All figures are char/4 ESTIMATES and are labelled as such (honesty discipline,
CLAUDE.md Rule 14) — never "saved you", always "estimated reclaimed".
"""
from __future__ import annotations

import json as json_mod

import click

from tokenjam.core.recommendations import (
    read_outcomes,
    recommendations_path,
    summarize_outcomes,
)
from tokenjam.core.savings_log import read_savings, savings_path, summarize_savings


@click.command("savings")
@click.option("--session", "session_id", default=None,
              help="Scope to a single session_id.")
@click.option("--json", "output_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def cmd_savings(ctx: click.Context, session_id: str | None, output_json: bool) -> None:
    """Show tokens reclaimed by the output-trim hook (estimated)."""
    config = ctx.obj.get("config")
    if config is None:
        raise click.ClickException("savings requires a config.")

    events = read_savings(config, session_id=session_id)
    summary = summarize_savings(events)

    outcomes = read_outcomes(config)
    rec_summary = summarize_outcomes(outcomes)

    if output_json:
        click.echo(json_mod.dumps({
            "sink": str(savings_path(config)),
            "summary": summary,
            "recommendations_sink": str(recommendations_path(config)),
            "recommendations": rec_summary,
        }, indent=2))
        return

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console(no_color=ctx.obj.get("no_color", False))

    if summary["trims"] == 0:
        console.print(Panel(
            Text.from_markup(
                "No trims recorded yet.\n\n"
                "The [bold]tj hook cap-output[/bold] PostToolUse hook trims bloated "
                "tool outputs before they enter context. Install it with "
                "[bold]tj onboard --claude-code[/bold], then run a session with a "
                "verbose test/build or a wide search.\n\n"
                f"[dim]sink: {savings_path(config)}[/dim]"
            ),
            title="tj savings", border_style="cyan",
        ))
        _render_recommendations(console, config, rec_summary)
        return

    saved = summary["saved_tok_est"]
    orig = summary["orig_tok_est"]
    pct = (saved / orig * 100) if orig else 0.0

    head = Text()
    head.append("~", style="bold green")
    head.append(f"{saved:,}", style="bold green")
    head.append(" tokens estimated reclaimed", style="green")
    head.append(f"  across {summary['trims']} trim(s)\n", style="dim")
    head.append(f"~{summary['saved_today_tok_est']:,} today", style="cyan")
    head.append(
        f"   ·   {pct:.0f}% of the {orig:,} tokens those outputs would have cost",
        style="dim",
    )

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Tool")
    table.add_column("Trims", justify="right")
    table.add_column("~Reclaimed (est. tok)", justify="right")
    for tool, bt in sorted(
        summary["by_tool"].items(), key=lambda kv: kv[1]["saved_tok_est"], reverse=True
    ):
        table.add_row(tool, str(bt["trims"]), f"{bt['saved_tok_est']:,}")

    console.print(Panel(head, title="tj savings", border_style="cyan"))
    console.print(table)
    console.print(
        Text(f"sink: {savings_path(config)}   (estimates; char/4)", style="dim")
    )

    _render_recommendations(console, config, rec_summary)


def _render_recommendations(console, config, rec_summary: dict) -> None:
    """Render the recommendation-outcome panel — measured-recovered kept
    strictly separate from estimated-recoverable (honesty discipline, Rule 14).

    Nothing recorded yet → a short prompt, so the panel never lies with zeros.
    """
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    rows = rec_summary.get("rows") or []
    if not rows:
        console.print(Panel(
            Text.from_markup(
                "No recommendation outcomes recorded yet.\n\n"
                "tokenjam records the actions it can observe directly — a "
                "[bold]tj summarize apply --go[/bold] and a "
                "[bold]tj optimize --export-config claude-code[/bold] — then, a "
                "week or so after an export, measures whether your model mix "
                "actually shifted off the recommended premium models.\n\n"
                f"[dim]sink: {recommendations_path(config)}[/dim]"
            ),
            title="recommendations acted on", border_style="magenta",
        ))
        return

    est_usd = rec_summary["estimated_recoverable_usd"]
    est_tok = rec_summary["estimated_recoverable_tokens"]
    meas_usd = rec_summary["measured_recovered_usd"]
    meas_tok = rec_summary["measured_recovered_tokens"]

    head = Text()
    head.append("Estimated recoverable", style="bold yellow")
    head.append(
        f"   ~{est_tok:,} tok"
        + (f"  (~${est_usd:,.2f})" if est_usd else "")
        + f"   across {rec_summary['actions_recorded']} recorded action(s)\n",
        style="yellow",
    )
    head.append("Measured recovered", style="bold green")
    head.append(
        f"      {meas_tok:,} tok"
        + (f"  (${meas_usd:,.2f})" if meas_usd else "")
        + f"   ·   {rec_summary['adopted']} adopted / {rec_summary['ignored']} ignored\n",
        style="green",
    )
    head.append(
        "measured = observed shift in real spans after a recommendation "
        "(correlation, not proof tokenjam caused it); estimated = projected at "
        "recommendation time.",
        style="dim",
    )

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("When")
    table.add_column("What")
    table.add_column("Status")
    table.add_column("Recovered / est.", justify="right")
    for o in rows[:20]:
        when = str(o.get("ts", ""))[:10]
        kind = str(o.get("kind", "?")).replace("_", " ")
        status = str(o.get("status", ""))
        if o.get("measured"):
            tok = int(o.get("recovered_tokens", 0) or 0)
            cell = f"[green]{tok:,} tok measured[/green]" if tok else "[dim]0 (ignored)[/dim]"
        else:
            tok = int(o.get("estimated_tokens", 0) or 0)
            cell = f"[yellow]~{tok:,} tok est.[/yellow]"
        table.add_row(when, kind, status, cell)

    console.print()
    console.print(Panel(head, title="recommendations acted on", border_style="magenta"))
    console.print(table)
    console.print(
        Text(f"sink: {recommendations_path(config)}", style="dim")
    )
