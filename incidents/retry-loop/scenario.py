"""
Incident: "Your agent isn't flaky. You're blind."

A tool call silently fails and retries itself in a loop. The agent keeps
calling the same broken tool, burning time and money. With print(), you see
"tool called" over and over. OCW sees the loop pattern and fires an alert.

No API keys required.
"""
from __future__ import annotations

AGENT_ID = "demo-retry-loop"
DESCRIPTION = "Agent stuck retrying a failing tool — OCW detects the loop and fires an alert"

_PRINT_SIMULATION = """\
[agent] Starting task...
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[tool] search_knowledge_base called
[tool] search_knowledge_base returned: null
[agent] Retrying...
"""


def run() -> None:
    """
    Inject a retry-loop pattern directly into IngestPipeline + InMemoryBackend.
    Renders Rich panels, or JSON if --json was passed to `ocw demo`.
    """
    import click

    ctx = click.get_current_context(silent=True)
    output_json = ctx.params.get("output_json", False) if ctx else False

    env, result = _execute()

    if output_json:
        import json as json_mod
        click.echo(json_mod.dumps({
            "scenario": "retry-loop",
            "agent_id": result.agent_id,
            "span_count": result.span_count,
            "alert_count": result.alert_count,
            "alert_types": result.alert_types,
            "alerts": [{"type": t} for t in result.alert_types],
            "total_cost_usd": result.total_cost_usd,
            "trace_count": result.trace_count,
        }, indent=2))
    else:
        from tj.utils.formatting import console
        _render(console, result)


def _execute():
    """Run the scenario logic, return (env, result)."""
    from datetime import timedelta

    from tj.core.models import NormalizedSpan, SpanKind, SpanStatus
    from tj.demo.env import DemoEnvironment
    from tj.utils.ids import new_span_id, new_trace_id, new_uuid
    from tj.utils.time_parse import utcnow

    env = DemoEnvironment()
    now = utcnow()
    conv_id = new_uuid()
    trace_id = new_trace_id()

    # Open a session
    env.process(NormalizedSpan(
        span_id=new_span_id(),
        trace_id=trace_id,
        name="gen_ai.invoke_agent",
        kind=SpanKind.INTERNAL,
        status_code=SpanStatus.OK,
        start_time=now,
        agent_id=AGENT_ID,
        conversation_id=conv_id,
    ))

    # 5 identical failing tool calls — triggers RETRY_LOOP (threshold: 4 in last 6 spans)
    tool_name = "search_knowledge_base"
    for i in range(5):
        t = now + timedelta(seconds=i * 3)
        env.process(NormalizedSpan(
            span_id=new_span_id(),
            trace_id=trace_id,
            name="gen_ai.tool.call",
            kind=SpanKind.INTERNAL,
            status_code=SpanStatus.ERROR,
            start_time=t,
            end_time=t + timedelta(milliseconds=300),
            duration_ms=300,
            agent_id=AGENT_ID,
            conversation_id=conv_id,
            tool_name=tool_name,
            status_message="connection timeout",
        ))

    return env, env.build_result(AGENT_ID)


def _render(console, result) -> None:
    from rich.panel import Panel

    console.print()
    console.print(Panel.fit(
        "[bold]Incident: Your agent isn't flaky. You're blind.[/bold]\n"
        "Scenario: [cyan]retry-loop[/cyan]",
        border_style="red",
    ))
    console.print()
    console.print(Panel(
        _PRINT_SIMULATION,
        title="[yellow]What print() shows[/yellow]",
        border_style="yellow",
    ))

    alert_str = "\n".join(
        f"  [red]ALERT[/red] {t}" for t in result.alert_types
    ) or "  (none)"
    ocw_output = (
        f"[bold]Spans ingested:[/bold] {result.span_count}\n"
        f"[bold]Traces:[/bold] {result.trace_count}\n\n"
        f"[bold]Alerts fired:[/bold]\n{alert_str}\n\n"
        "[dim]In your real agent:[/dim]\n"
        "  ocw alerts          [dim]# see the retry_loop alert[/dim]\n"
        "  ocw traces          [dim]# see the loop pattern[/dim]"
    )
    console.print(Panel(ocw_output, title="[green]What OCW reveals[/green]", border_style="green"))
    console.print()
