"""
Incident: "Why did my agent just spend $47 on a hello world?"

The model silently escalated from cheap Haiku to expensive Opus mid-chain.
With print(), you see "response received". OCW shows cost per model, per call.

No API keys required.
"""
from __future__ import annotations

AGENT_ID = "demo-surprise-cost"
DESCRIPTION = "Agent silently burns budget on expensive models — OCW tracks every dollar"

_PRINT_SIMULATION = """\
[agent] Starting document analysis...
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[llm] Response received (200 OK)
[agent] Task complete.
"""

# (model, provider, input_tokens, output_tokens)
_CALLS = [
    ("claude-haiku-4-5",  "anthropic", 2_000,  500),
    ("claude-haiku-4-5",  "anthropic", 3_000,  800),
    ("claude-sonnet-4-6", "anthropic", 15_000, 3_000),
    ("claude-sonnet-4-6", "anthropic", 18_000, 4_000),
    ("claude-opus-4-6",   "anthropic", 40_000, 6_000),
    ("claude-opus-4-6",   "anthropic", 45_000, 7_000),
    ("claude-opus-4-6",   "anthropic", 38_000, 5_500),
    ("claude-sonnet-4-6", "anthropic", 12_000, 2_500),
]


def run() -> None:
    """
    Inject multi-model LLM calls showing escalating costs.
    Renders Rich panels, or JSON if --json was passed to `ocw demo`.
    """
    import click

    ctx = click.get_current_context(silent=True)
    output_json = ctx.params.get("output_json", False) if ctx else False

    env, result, model_breakdown = _execute()

    if output_json:
        import json as json_mod
        click.echo(json_mod.dumps({
            "scenario": "surprise-cost",
            "agent_id": result.agent_id,
            "span_count": result.span_count,
            "alert_count": result.alert_count,
            "alert_types": result.alert_types,
            "alerts": [{"type": t} for t in result.alert_types],
            "total_cost_usd": result.total_cost_usd,
            "trace_count": result.trace_count,
            "model_breakdown": model_breakdown,
        }, indent=2))
    else:
        from tj.utils.formatting import console
        _render(console, result, model_breakdown)


def _execute():
    """Run the scenario logic, return (env, result, model_breakdown)."""
    from datetime import timedelta

    from tj.core.models import NormalizedSpan, SpanKind, SpanStatus
    from tj.demo.env import DemoEnvironment
    from tj.utils.ids import new_span_id, new_trace_id, new_uuid
    from tj.utils.time_parse import utcnow

    env = DemoEnvironment()
    now = utcnow()
    conv_id = new_uuid()
    trace_id = new_trace_id()

    for i, (model, provider, inp, out) in enumerate(_CALLS):
        t = now + timedelta(seconds=i * 4)
        env.process(NormalizedSpan(
            span_id=new_span_id(),
            trace_id=trace_id,
            name="gen_ai.llm.call",
            kind=SpanKind.CLIENT,
            status_code=SpanStatus.OK,
            start_time=t,
            end_time=t + timedelta(seconds=3),
            duration_ms=3000,
            agent_id=AGENT_ID,
            conversation_id=conv_id,
            provider=provider,
            model=model,
            input_tokens=inp,
            output_tokens=out,
        ))

    result = env.build_result(AGENT_ID)
    rows = env.db.conn.execute(
        "SELECT model, SUM(cost_usd), COUNT(*) FROM spans "
        "WHERE agent_id = $1 AND model IS NOT NULL "
        "GROUP BY model ORDER BY SUM(cost_usd) DESC",
        [AGENT_ID],
    ).fetchall()
    model_breakdown = [
        {"model": r[0], "cost_usd": round(r[1] or 0, 6), "calls": r[2]}
        for r in rows
    ]
    return env, result, model_breakdown


def _render(console, result, model_breakdown) -> None:
    from io import StringIO

    from rich import box
    from rich.console import Console as RichConsole
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console.print()
    console.print(Panel.fit(
        "[bold]Incident: Why did my agent just spend $47 on a hello world?[/bold]\n"
        "Scenario: [cyan]surprise-cost[/cyan]",
        border_style="red",
    ))
    console.print()
    console.print(Panel(
        _PRINT_SIMULATION,
        title="[yellow]What print() shows[/yellow]",
        border_style="yellow",
    ))

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Calls", justify="right")
    table.add_column("Cost (USD)", justify="right", style="red")
    for row in model_breakdown:
        table.add_row(row["model"], str(row["calls"]), f"${row['cost_usd']:.4f}")

    buf = StringIO()
    tmp = RichConsole(file=buf, highlight=False)
    tmp.print(table)
    tmp.print(Text(f"Total session cost: ${result.total_cost_usd:.4f}", style="bold red"))
    tmp.print("\n[dim]In your real agent:[/dim]")
    tmp.print("  ocw cost --by model    [dim]# per-model spend[/dim]")
    tmp.print("  ocw cost               [dim]# daily breakdown[/dim]")

    console.print(Panel(buf.getvalue(), title="[green]What OCW reveals[/green]", border_style="green"))
    console.print()
