from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import click

from tj.core.models import TraceFilters
from tj.utils.formatting import console
from tj.utils.time_parse import parse_since


@click.command("export")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("--since", default="7d", help="Time window (e.g. 1h, 7d)")
@click.option("--format", "fmt",
              type=click.Choice(["json", "csv", "otlp", "openevals"]),
              default="json")
@click.option("--output", "output_path", default=None, help="Output file path (stdout if omitted)")
@click.pass_context
def cmd_export(ctx: click.Context, agent: str | None, since: str,
               fmt: str, output_path: str | None) -> None:
    """Export spans in various formats."""
    db = ctx.obj["db"]
    agent_filter = agent or ctx.obj.get("agent")
    filters = TraceFilters(
        agent_id=agent_filter,
        since=parse_since(since),
        limit=10000,
    )
    traces = db.get_traces(filters)

    if fmt == "json":
        output = _export_json(db, traces)
    elif fmt == "csv":
        output = _export_csv(db, traces)
    elif fmt == "otlp":
        _export_otlp(ctx, db, traces)
        return
    elif fmt == "openevals":
        output = _export_openevals(db, traces)
    else:
        console.print(f"[red]Unknown format: {fmt}[/red]")
        return

    if output_path:
        Path(output_path).write_text(output)
        console.print(f"[green]Exported to {output_path}[/green]")
    else:
        click.echo(output)


def _export_json(db: object, traces: list) -> str:
    lines = []
    seen_traces: set[str] = set()
    for t in traces:
        if t.trace_id in seen_traces:
            continue
        seen_traces.add(t.trace_id)
        spans = db.get_trace_spans(t.trace_id)
        for s in spans:
            lines.append(json.dumps({
                "span_id": s.span_id,
                "trace_id": s.trace_id,
                "agent_id": s.agent_id,
                "name": s.name,
                "start_time": s.start_time.isoformat() if s.start_time else None,
                "end_time": s.end_time.isoformat() if s.end_time else None,
                "duration_ms": s.duration_ms,
                "cost_usd": s.cost_usd,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "status_code": s.status_code.value,
                "provider": s.provider,
                "model": s.model,
                "tool_name": s.tool_name,
                "attributes": s.attributes,
            }, default=str))
    return "\n".join(lines)


def _export_csv(db: object, traces: list) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "span_id", "trace_id", "agent_id", "name", "start_time",
        "duration_ms", "cost_usd", "input_tokens", "output_tokens", "status_code",
    ])
    seen_traces: set[str] = set()
    for t in traces:
        if t.trace_id in seen_traces:
            continue
        seen_traces.add(t.trace_id)
        spans = db.get_trace_spans(t.trace_id)
        for s in spans:
            writer.writerow([
                s.span_id, s.trace_id, s.agent_id, s.name,
                s.start_time.isoformat() if s.start_time else "",
                s.duration_ms, s.cost_usd,
                s.input_tokens, s.output_tokens, s.status_code.value,
            ])
    return buf.getvalue()


_KIND_MAP = {"internal": 1, "server": 2, "client": 3, "producer": 4, "consumer": 5}


def _export_otlp(ctx: click.Context, db: object, traces: list) -> None:
    config = ctx.obj["config"]
    if not config.export.otlp.enabled:
        console.print("[red]OTLP export is not enabled. Set export.otlp.enabled = true "
                      "in config.[/red]")
        return

    import httpx
    endpoint = config.export.otlp.endpoint.rstrip("/") + "/v1/traces"
    headers = dict(config.export.otlp.headers)
    headers.setdefault("Content-Type", "application/json")

    seen_traces: set[str] = set()
    succeeded = 0
    failed = 0
    for t in traces:
        if t.trace_id in seen_traces:
            continue
        seen_traces.add(t.trace_id)
        spans = db.get_trace_spans(t.trace_id)
        payload = {
            "resourceSpans": [{
                "resource": {"attributes": [
                    {"key": "service.name", "value": {"stringValue": "tokenjam"}},
                ]},
                "scopeSpans": [{
                    "spans": [
                        {
                            "traceId": s.trace_id,
                            "spanId": s.span_id,
                            "name": s.name,
                            "kind": _KIND_MAP.get(s.kind.value, 1) if s.kind else 1,
                            "startTimeUnixNano": str(int(s.start_time.timestamp() * 1e9))
                            if s.start_time else "0",
                            "endTimeUnixNano": str(int(s.end_time.timestamp() * 1e9))
                            if s.end_time else "0",
                        }
                        for s in spans
                    ],
                }],
            }],
        }
        resp = httpx.post(endpoint, json=payload, headers=headers)
        if resp.status_code >= 400:
            console.print(f"[red]OTLP export failed for trace {t.trace_id}: "
                          f"{resp.status_code}[/red]")
            failed += 1
        else:
            succeeded += 1

    console.print(f"[green]Exported {succeeded} traces to {endpoint}[/green]"
                  + (f" ({failed} failed)" if failed else ""))


def _export_openevals(db: object, traces: list) -> str:
    results = []
    seen_traces: set[str] = set()
    for t in traces:
        if t.trace_id in seen_traces:
            continue
        seen_traces.add(t.trace_id)
        spans = db.get_trace_spans(t.trace_id)
        messages = []
        for s in spans:
            if s.attributes.get("gen_ai.prompt.content"):
                messages.append({
                    "role": "user",
                    "content": s.attributes["gen_ai.prompt.content"],
                })
            if s.tool_name:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"name": s.tool_name}],
                })
                if s.attributes.get("gen_ai.tool.output"):
                    messages.append({
                        "role": "tool",
                        "content": s.attributes["gen_ai.tool.output"],
                    })
            if s.attributes.get("gen_ai.completion.content"):
                messages.append({
                    "role": "assistant",
                    "content": s.attributes["gen_ai.completion.content"],
                })
        results.append({
            "trace_id": t.trace_id,
            "agent_id": t.agent_id,
            "messages": messages,
        })
    return json.dumps(results, default=str, indent=2)
