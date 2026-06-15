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
from tokenjam.core.ingest_adapters.helicone import ingest_helicone
from tokenjam.core.ingest_adapters.langfuse import ingest_langfuse
from tokenjam.core.ingest_adapters.otlp import ingest_otlp
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
@click.option("--reingest", is_flag=True,
              help="Update spans already in the DB in place (never duplicated): "
                   "re-tags sub_agent_id on pre-column history AND backfills "
                   "captured content (message text / tool_input) onto existing "
                   "spans when [capture] was enabled after they were first "
                   "ingested. Run this after turning on [capture].")
@click.pass_context
def claude_code(ctx: click.Context, root_path: str | None, since_days: int | None,
                quiet: bool, reingest: bool) -> None:
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
    # Pass config so backfilled sessions carry the declared plan tier (#176).
    result = ingest_claude_code(
        db, root=root, since=since, progress=progress, reingest=reingest,
        config=ctx.obj.get("config"),
    )

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

    if result.spans_retagged:
        console.print(
            f"  [dim]Re-tagged {result.spans_retagged} existing spans "
            f"(sub_agent_id refreshed).[/dim]"
        )
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


@cmd_backfill.command("langfuse")
@click.option("--source-url", default=None,
              help="Live Langfuse base URL (e.g. https://cloud.langfuse.com). "
                   "Reads /api/public/observations with --api-key Bearer auth.")
@click.option("--source-file", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Local JSON file containing a Langfuse observations dump. "
                   "Accepts a bare list, {\"data\": [...]} envelope, or NDJSON.")
@click.option("--api-key", default=None, help="Langfuse public API key (Bearer).")
@click.option("--since", default=None,
              help="Only ingest observations newer than this. Accepts '30d', "
                   "'24h', or an ISO-8601 timestamp.")
@click.pass_context
def langfuse(ctx: click.Context, source_url: str | None, source_file: str | None,
             api_key: str | None, since: str | None) -> None:
    """Ingest Langfuse observations from a live API or a JSON dump."""
    db = ctx.obj.get("db")
    if db is None:
        raise click.ClickException("backfill requires a database connection.")
    if (source_url is None) == (source_file is None):
        raise click.UsageError("Provide exactly one of --source-url or --source-file.")

    since_dt = None
    if since:
        from tokenjam.utils.time_parse import parse_since
        try:
            since_dt = parse_since(since)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    source_label = source_url or source_file
    console.print(f"Ingesting Langfuse observations from {source_label} …")
    try:
        result = ingest_langfuse(
            db,
            source_url=source_url,
            source_file=source_file,
            api_key=api_key,
            since=since_dt,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    console.print(
        f"[green]✓[/green] Read [bold]{result['observations_read']}[/bold] "
        f"observation(s); wrote [bold]{result['spans_written']}[/bold] new "
        f"span(s); skipped [bold]{result['spans_skipped']}[/bold] already "
        f"present."
    )


@cmd_backfill.command("helicone")
@click.option("--source-url", default=None,
              help="Live Helicone base URL (e.g. https://api.helicone.ai). "
                   "POSTs /v1/request/query with --api-key Bearer auth.")
@click.option("--source-file", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Local JSON file containing a Helicone records dump. "
                   "Accepts a bare list, {\"data\": [...]} envelope, or NDJSON.")
@click.option("--api-key", default=None, help="Helicone API key (Bearer).")
@click.option("--since", default=None,
              help="Only ingest records newer than this. Accepts '30d', "
                   "'24h', or an ISO-8601 timestamp.")
@click.pass_context
def helicone(ctx: click.Context, source_url: str | None, source_file: str | None,
             api_key: str | None, since: str | None) -> None:
    """Ingest Helicone request records from a live API or a JSON dump."""
    db = ctx.obj.get("db")
    if db is None:
        raise click.ClickException("backfill requires a database connection.")
    if (source_url is None) == (source_file is None):
        raise click.UsageError("Provide exactly one of --source-url or --source-file.")

    since_dt = None
    if since:
        from tokenjam.utils.time_parse import parse_since
        try:
            since_dt = parse_since(since)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    source_label = source_url or source_file
    console.print(f"Ingesting Helicone records from {source_label} …")
    try:
        result = ingest_helicone(
            db,
            source_url=source_url,
            source_file=source_file,
            api_key=api_key,
            since=since_dt,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    console.print(
        f"[green]✓[/green] Read [bold]{result['records_read']}[/bold] "
        f"record(s); wrote [bold]{result['spans_written']}[/bold] new "
        f"span(s); skipped [bold]{result['spans_skipped']}[/bold] already "
        f"present."
    )


@cmd_backfill.command("otlp")
@click.option("--source-url", default=None,
              help="HTTP(S) URL to an OTLP JSON dump (GET-fetched). For "
                   "live push-style OTLP ingestion, point your collector "
                   "at the tj serve endpoint instead.")
@click.option("--source-file", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Local OTLP JSON file to ingest. Accepts a single "
                   "{\"resourceSpans\": [...]} envelope or NDJSON with one "
                   "envelope per line.")
@click.option("--since", default=None,
              help="Only ingest spans newer than this. Accepts '30d', "
                   "'24h', or an ISO-8601 timestamp.")
@click.pass_context
def otlp(ctx: click.Context, source_url: str | None, source_file: str | None,
         since: str | None) -> None:
    """Ingest a raw OTLP JSON dump from a live endpoint or a file."""
    db = ctx.obj.get("db")
    if db is None:
        raise click.ClickException("backfill requires a database connection.")
    if (source_url is None) == (source_file is None):
        raise click.UsageError("Provide exactly one of --source-url or --source-file.")

    since_dt = None
    if since:
        from tokenjam.utils.time_parse import parse_since
        try:
            since_dt = parse_since(since)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    source_label = source_url or source_file
    console.print(f"Ingesting OTLP spans from {source_label} …")
    try:
        result = ingest_otlp(
            db,
            source_url=source_url,
            source_file=source_file,
            since=since_dt,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    console.print(
        f"[green]✓[/green] Saw [bold]{result['spans_seen']}[/bold] span(s); "
        f"wrote [bold]{result['spans_written']}[/bold] new; "
        f"skipped [bold]{result['spans_skipped']}[/bold] already present; "
        f"rejected [bold]{result['spans_rejected']}[/bold]."
    )
