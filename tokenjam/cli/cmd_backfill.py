"""`tj backfill` — ingest historical agent session logs into the local DB."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import click

from tokenjam.cli.backfill_progress import backfill_progress
from tokenjam.core.backfill import (
    CLAUDE_CODE_PROJECTS_ROOT,
    count_claude_code_sessions_in_scope,
    ingest_claude_code,
)
from tokenjam.core.ingest_adapters.codex import (
    CODEX_SESSIONS_ROOT,
    ingest_codex,
)
from tokenjam.core.ingest_adapters.helicone import ingest_helicone
from tokenjam.core.ingest_adapters.langfuse import ingest_langfuse
from tokenjam.core.ingest_adapters.otlp import ingest_otlp
from tokenjam.utils.formatting import console, format_cost
from tokenjam.utils.time_parse import parse_since, utcnow


@click.group("backfill")
def cmd_backfill() -> None:
    """Import past sessions from your agents."""


@cmd_backfill.command("claude-code")
@click.option("--root", "root_path", default=None,
              help=f"Override Claude Code projects root (default {CLAUDE_CODE_PROJECTS_ROOT}).")
@click.option("--since", "since_value", default=None,
              help="Only ingest sessions since a window like 30d, 7d, or YYYY-MM-DD.")
@click.option("--since-days", type=int, default=None,
              help="Deprecated alias for --since Nd.")
@click.option("--quiet", is_flag=True, help="Suppress per-session progress output.")
@click.option("--reingest", is_flag=True,
              help="Update spans already in the DB in place (never duplicated): "
                   "re-tags sub_agent_id on pre-column history AND backfills "
                   "captured content (message text / tool_input) onto existing "
                   "spans when [capture] was enabled after they were first "
                   "ingested. Run this after turning on [capture].")
@click.pass_context
def claude_code(ctx: click.Context, root_path: str | None, since_value: str | None,
                since_days: int | None, quiet: bool, reingest: bool) -> None:
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
    if since_value and since_days is not None:
        raise click.UsageError("Use either --since or --since-days, not both.")
    if since_value:
        try:
            since = parse_since(since_value)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--since'") from exc
    elif since_days is not None:
        if since_days <= 0:
            raise click.BadParameter("amount must be > 0", param_hint="'--since-days'")
        since = utcnow() - timedelta(days=since_days)

    # Cheap pre-count (stat() only, no parsing) so the shared progress counter
    # can show "N/total" rather than a bare running count (#443).
    total_in_scope = count_claude_code_sessions_in_scope(root=root, since=since)

    console.print(f"Backfilling Claude Code sessions from {root} …")
    # Pass config so backfilled sessions carry the declared plan tier (#176).
    with backfill_progress(total_in_scope, quiet=quiet) as progress:
        result = ingest_claude_code(
            db, root=root, since=since, progress=progress,
            config=ctx.obj.get("config"), reingest=reingest,
        )

    if result.sessions_seen == 0:
        console.print(
            "[yellow]No sessions found.[/yellow] "
            "[dim]Use Claude Code for a while, then re-run.[/dim]"
        )
        return

    days_span = None
    if result.earliest and result.latest:
        days_span = (result.latest - result.earliest).days

    total = result.sessions_total
    parts = [
        f"Backfilled [bold]{result.sessions_new}[/bold] new "
        f"({result.sessions_existing} already present) · "
        f"[bold]{total}[/bold] total session{'s' if total != 1 else ''}",
    ]
    if days_span is not None:
        parts.append(f"over {days_span} day{'s' if days_span != 1 else ''}")
    if result.project_count:
        parts.append(f"from {result.project_count} project"
                     f"{'s' if result.project_count != 1 else ''}")
    parts.append(f"({format_cost(result.total_cost_usd)} total spend)")
    console.print("[green]✓[/green] " + ", ".join(parts) + ".")

    # Make the conversations-vs-sessions distinction explicit when they differ
    # (Claude Code writes multiple JSONL files per session) so the smaller
    # `sessions` count doesn't read as data loss (#238).
    if result.conversations_seen != total:
        console.print(
            f"  [dim]Parsed {result.conversations_seen} conversation files "
            f"into {total} session{'s' if total != 1 else ''}.[/dim]"
        )

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


@cmd_backfill.command("status")
@click.option("--root", "root_path", default=None,
              help=f"Override Claude Code projects root (default {CLAUDE_CODE_PROJECTS_ROOT}).")
@click.option("--since", "since_value", default=None,
              help="Only compare sessions since a window like 30d, 7d, or YYYY-MM-DD.")
@click.option("--json", "output_json", is_flag=True, help="Emit the report as JSON.")
@click.option("--limit", type=int, default=10, show_default=True,
              help="How many missing sessions to list.")
@click.pass_context
def status(ctx: click.Context, root_path: str | None, since_value: str | None,
           output_json: bool, limit: int) -> None:
    """Show which on-disk Claude Code sessions are NOT yet ingested.

    Claude Code prunes its own transcripts after roughly 30 days, so a session
    that never made it into the DB is on a clock: once the file is gone it is
    unrecoverable. This surfaces the gap while the source still exists.

    Sessions are compared by the transcript's INTERNAL `sessionId`, not by
    filename — roughly half the `.jsonl` files under the projects root live in
    nested `subagents/` folders and carry their PARENT's session id, so a
    filename-based count wildly over-reports the gap.
    """
    import json as _json

    from tokenjam.core.transcript_sync import reconcile_claude_code

    db = ctx.obj.get("db")
    if db is None:
        raise click.ClickException("backfill status requires a database connection.")

    root = Path(root_path).expanduser() if root_path else None
    since = None
    if since_value:
        try:
            since = parse_since(since_value)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    report = reconcile_claude_code(db, root=root, since=since)

    if output_json:
        console.print_json(_json.dumps(report.to_dict()))
        return

    if not report.root.exists():
        console.print(f"[yellow]No Claude Code logs found at {report.root}.[/yellow]")
        return

    console.print(
        f"Compared [bold]{report.files_scanned}[/bold] transcript files at "
        f"{report.root} → [bold]{report.disk_sessions}[/bold] distinct session"
        f"{'s' if report.disk_sessions != 1 else ''} "
        f"([dim]deduped on internal sessionId[/dim])."
    )

    if not report.verified:
        # The disk-vs-DB anti-join never ran (the daemon holds the write-lock
        # and its HTTP shim lookup failed). The ingested/missing/skipped buckets
        # are all 0 by default here — printing them would falsely claim
        # "everything is ingested" (#642). Be honest instead.
        console.print(
            f"  [yellow]![/yellow] Couldn't verify ingestion status — "
            f"found [bold]{report.disk_sessions}[/bold] on-disk session"
            f"{'s' if report.disk_sessions != 1 else ''} but the tj daemon was "
            f"unreachable, so the DB comparison could not run."
        )
        console.print(
            "  [dim]Try `tj stop` then re-run `tj backfill status`, or check "
            "that `tj serve` is healthy.[/dim]"
        )
        return

    console.print(
        f"  [bold]{report.ingested_sessions}[/bold] already ingested · "
        f"[bold]{report.missing_count}[/bold] missing · "
        f"{report.skipped_empty_count} skipped (no assistant turn)."
    )

    if report.files_unreadable:
        console.print(
            f"  [dim]{report.files_unreadable} file(s) had no readable session id.[/dim]"
        )

    if not report.missing:
        console.print("[green]✓[/green] Every on-disk session with usage is ingested.")
        return

    days_left = report.days_until_rotation()
    if days_left is not None and days_left <= 7:
        console.print(
            f"[red]![/red] The oldest missing session is ~{days_left:.0f} day(s) from "
            f"being pruned by Claude Code — after that it is unrecoverable."
        )
    elif days_left is not None:
        console.print(
            f"  [dim]Oldest missing session stays recoverable for ~{days_left:.0f} "
            f"more day(s).[/dim]"
        )

    for session in report.missing[:limit]:
        started = (
            session.started_at.strftime("%Y-%m-%d %H:%M")
            if session.started_at else "unknown"
        )
        console.print(f"    [dim]{session.session_id[:8]}  {started}  {session.project}[/dim]")
    if report.missing_count > limit:
        console.print(f"    [dim]… and {report.missing_count - limit} more.[/dim]")

    console.print(
        "  Run [bold]tj backfill claude-code[/bold] to ingest them now, or start "
        "[bold]tj serve[/bold] — it catches up automatically."
    )


@cmd_backfill.command("codex")
@click.option("--root", "root_path", default=None,
              help=f"Override Codex sessions root (default {CODEX_SESSIONS_ROOT}).")
@click.option("--since", "since_value", default=None,
              help="Only ingest sessions since a window like 30d, 7d, or YYYY-MM-DD.")
@click.pass_context
def codex(ctx: click.Context, root_path: str | None,
          since_value: str | None) -> None:
    """Ingest Codex CLI session logs from ~/.codex/sessions/."""
    db = ctx.obj.get("db")
    if db is None:
        raise click.ClickException("backfill requires a database connection.")

    root = Path(root_path).expanduser() if root_path else CODEX_SESSIONS_ROOT
    if not root.exists():
        console.print(f"[yellow]No Codex logs found at {root}.[/yellow]")
        console.print(
            "[dim]This is normal if the Codex CLI hasn't been used on this "
            "machine yet — backfill will be useful once it has.[/dim]"
        )
        return

    since = None
    if since_value:
        try:
            since = parse_since(since_value)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    console.print(f"Backfilling Codex sessions from {root} …")
    try:
        result = ingest_codex(
            db, root=root, since=since, config=ctx.obj.get("config"),
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    if result["sessions_seen"] == 0:
        console.print(
            "[yellow]No sessions found.[/yellow] "
            "[dim]Use the Codex CLI for a while, then re-run.[/dim]"
        )
        return

    console.print(
        f"[green]✓[/green] Saw [bold]{result['sessions_seen']}[/bold] "
        f"session(s); wrote [bold]{result['sessions_written']}[/bold] new "
        f"session(s), [bold]{result['spans_written']}[/bold] new span(s); "
        f"skipped [bold]{result['spans_skipped']}[/bold] already present."
    )
    if result["sessions_failed"]:
        console.print(
            f"  [yellow]Warning: {result['sessions_failed']} session(s) "
            f"failed to parse.[/yellow]"
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
