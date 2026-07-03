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

    if output_json:
        click.echo(json_mod.dumps({
            "sink": str(savings_path(config)),
            "summary": summary,
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
