from __future__ import annotations

import json

import click

from tj.core.models import NormalizedSpan, TraceFilters
from tj.utils.formatting import console, format_cost, make_table
from tj.utils.time_parse import parse_since


@click.command("traces")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("--since", default="24h", help="Time window (e.g. 1h, 7d)")
@click.option("--limit", default=50, type=int)
@click.option("--type", "span_type", default=None, help="Filter by span name/type")
@click.option("--status", default=None, type=click.Choice(["ok", "error"]))
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def cmd_traces(ctx: click.Context, agent: str | None, since: str, limit: int,
               span_type: str | None, status: str | None, output_json: bool) -> None:
    """List recent traces."""
    db = ctx.obj["db"]
    agent_filter = agent or ctx.obj.get("agent")
    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc
    filters = TraceFilters(
        agent_id=agent_filter,
        since=since_dt,
        span_name=span_type,
        status=status,
        limit=limit,
    )
    traces = db.get_traces(filters)

    if output_json:
        click.echo(json.dumps([
            {
                "trace_id": t.trace_id,
                "agent_id": t.agent_id,
                "name": t.name,
                "start_time": t.start_time.isoformat() if t.start_time else None,
                "duration_ms": t.duration_ms,
                "cost_usd": t.cost_usd,
                "status_code": t.status_code,
                "span_count": t.span_count,
            }
            for t in traces
        ], default=str))
        return

    if not traces:
        console.print("[dim]No traces found for the given filters.[/dim]")
        return

    table = make_table("TRACE ID", "AGENT", "TYPE", "DUR", "COST", "STATUS")
    for t in traces:
        dur = f"{t.duration_ms:.0f}ms" if t.duration_ms else "-"
        cost = format_cost(t.cost_usd) if t.cost_usd else "-"
        status_style = "red" if t.status_code == "error" else ""
        table.add_row(
            t.trace_id[:12] + "...",
            t.agent_id or "-",
            t.name,
            dur,
            cost,
            f"[{status_style}]{t.status_code}[/]" if status_style else t.status_code,
        )
    console.print(table)


@click.command("trace")
@click.argument("trace_id")
@click.option("--json", "output_json", is_flag=True)
@click.pass_context
def cmd_trace(ctx: click.Context, trace_id: str, output_json: bool) -> None:
    """Show span waterfall for a single trace."""
    db = ctx.obj["db"]
    spans = db.get_trace_spans(trace_id)

    # Support prefix matching (like git short hashes)
    if not spans and len(trace_id) < 32:
        if hasattr(db, "conn"):
            rows = db.conn.execute(
                "SELECT DISTINCT trace_id FROM spans WHERE trace_id LIKE $1 LIMIT 2",
                [trace_id + "%"],
            ).fetchall()
            if len(rows) == 1:
                trace_id = rows[0][0]
                spans = db.get_trace_spans(trace_id)
            elif len(rows) > 1:
                console.print(f"[red]Ambiguous prefix '{trace_id}' — matches "
                              f"{len(rows)} traces. Use more characters.[/red]")
                return

    if not spans:
        console.print(f"[dim]No spans found for trace {trace_id}[/dim]")
        return

    if output_json:
        click.echo(json.dumps([
            {
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "name": s.name,
                "kind": s.kind.value,
                "status_code": s.status_code.value,
                "start_time": s.start_time.isoformat() if s.start_time else None,
                "duration_ms": s.duration_ms,
                "provider": s.provider,
                "model": s.model,
                "tool_name": s.tool_name,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "cost_usd": s.cost_usd,
            }
            for s in spans
        ], default=str))
        return

    # Build parent->children map for tree rendering
    children: dict[str | None, list[NormalizedSpan]] = {}
    for s in spans:
        children.setdefault(s.parent_span_id, []).append(s)

    # Find root spans (no parent or parent not in this trace)
    span_ids = {s.span_id for s in spans}
    roots = [s for s in spans if s.parent_span_id is None
             or s.parent_span_id not in span_ids]

    for root in roots:
        _print_span_tree(root, children, prefix="", is_last=True)


def _print_span_tree(span: NormalizedSpan, children: dict[str | None, list[NormalizedSpan]],
                     prefix: str, is_last: bool) -> None:
    connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "
    dur = f"{span.duration_ms:.0f}ms" if span.duration_ms else ""
    cost = format_cost(span.cost_usd) if span.cost_usd else ""

    parts = [f"[bold]{span.name}[/bold]"]
    if dur:
        parts.append(f"[dim]{dur}[/dim]")
    if span.model:
        parts.append(f"[cyan]{span.model}[/cyan]")
    if span.tool_name:
        parts.append(f"[magenta]{span.tool_name}[/magenta]")
    if cost:
        parts.append(cost)
    if span.status_code.value == "error":
        parts.append("[red]ERROR[/red]")

    line = " ".join(parts)
    console.print(f"{prefix}{connector}{line}")

    child_spans = children.get(span.span_id, [])
    for i, child in enumerate(child_spans):
        child_prefix = prefix + ("   " if is_last else "\u2502  ")
        _print_span_tree(child, children, child_prefix, i == len(child_spans) - 1)
