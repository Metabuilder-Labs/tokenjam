"""`tj backfill` — ingest historical agent session logs into the local DB."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import click

from tokenjam.core.backfill import (
    CLAUDE_CODE_PROJECTS_ROOT,
    BackfillResult,
    ingest_claude_code,
)
from tokenjam.utils.formatting import console, format_cost
from tokenjam.utils.time_parse import utcnow


@click.group("backfill")
def cmd_backfill() -> None:
    """Backfill historical session data from local agent logs."""


@cmd_backfill.command("claude-code")
@click.option("--root", "root_path", default=None,
              help=f"Override Claude Code projects root (default {CLAUDE_CODE_PROJECTS_ROOT}).")
@click.option("--since-days", type=int, default=None,
              help="Only ingest sessions whose end time is within the last N days.")
@click.option("--quiet", is_flag=True, help="Suppress per-session progress output.")
@click.pass_context
def claude_code(ctx: click.Context, root_path: str | None, since_days: int | None,
                quiet: bool) -> None:
    """Ingest Claude Code session logs from ~/.claude/projects/."""
    db = ctx.obj.get("db")
    if db is None:
        raise click.ClickException("backfill requires a database connection.")

    root = Path(root_path).expanduser() if root_path else CLAUDE_CODE_PROJECTS_ROOT
    if not root.exists():
        console.print(f"[yellow]No Claude Code logs found at {root}.[/yellow]")
        console.print(
            "[dim]This is normal if Claude Code hasn't been used on this "
            "machine yet — backfill will be useful once it has.[/dim]"
        )
        return

    since = None
    if since_days is not None and since_days > 0:
        since = utcnow() - timedelta(days=since_days)

    state = {"last_print": 0}

    def progress(parsed, result: BackfillResult) -> None:
        if quiet:
            return
        if result.sessions_seen - state["last_print"] >= 1:
            console.print(
                f"  [dim]({result.sessions_seen} sessions, "
                f"{result.spans_ingested} new spans, "
                f"{format_cost(result.total_cost_usd)} total)[/dim]",
                end="\r",
            )
            state["last_print"] = result.sessions_seen

    console.print(f"Backfilling Claude Code sessions from {root} …")
    result = ingest_claude_code(db, root=root, since=since, progress=progress)

    if not quiet:
        # Clear the carriage-return line
        console.print(" " * 80, end="\r")

    if result.sessions_seen == 0:
        console.print(
            "[yellow]No sessions found.[/yellow] "
            "[dim]Use Claude Code for a while, then re-run.[/dim]"
        )
        return

    days_span = None
    if result.earliest and result.latest:
        days_span = (result.latest - result.earliest).days

    parts = [
        f"Backfilled [bold]{result.sessions_ingested}[/bold] of "
        f"{result.sessions_seen} sessions",
    ]
    if days_span is not None:
        parts.append(f"over {days_span} day{'s' if days_span != 1 else ''}")
    if result.project_count:
        parts.append(f"from {result.project_count} project"
                     f"{'s' if result.project_count != 1 else ''}")
    parts.append(f"({format_cost(result.total_cost_usd)} total spend)")
    console.print("[green]✓[/green] " + ", ".join(parts) + ".")

    if result.spans_skipped_existing:
        console.print(
            f"  [dim]Skipped {result.spans_skipped_existing} spans already "
            f"present (idempotent re-run).[/dim]"
        )
    if result.files_failed:
        console.print(
            f"  [yellow]Warning: {result.files_failed} session(s) failed to "
            f"parse — sample errors:[/yellow]"
        )
        for err in result.sample_errors:
            console.print(f"    [dim]{err}[/dim]")
    if days_span is not None and days_span < 7:
        console.print(
            "  [dim]Less than 7 days of history available — `tj optimize` will "
            "flag thin-data projections.[/dim]"
        )
