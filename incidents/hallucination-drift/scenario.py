"""
Incident: "My agent worked yesterday. Today it's possessed."

Behavioral drift: the agent's token usage and tool patterns have shifted
significantly. With print(), you notice "outputs look different." OCW
measures the deviation with Z-scores and fires a DRIFT_DETECTED alert.

No API keys required.
"""
from __future__ import annotations

AGENT_ID = "demo-hallucination-drift"
DESCRIPTION = "Agent behavior shifts unexpectedly — OCW catches statistical drift and fires an alert"

_PRINT_SIMULATION = """\
[agent] Session 1... output looks reasonable
[agent] Session 2... output looks reasonable
[agent] Session 3... output looks reasonable
[agent] Session 4... output looks reasonable
[agent] Session 5... output looks reasonable
[agent] Session 6... output looks... different?
[agent] Hmm, that response was longer than usual.
[agent] But hey, it completed successfully. Moving on.
"""

_BASELINE_SESSIONS = 5  # reduced from default 10 for a fast demo


def run() -> None:
    """
    Build baseline from 5 normal sessions, then fire an anomalous session
    to trigger DRIFT_DETECTED. Renders Rich panels, or JSON if --json passed.
    """
    import click

    ctx = click.get_current_context(silent=True)
    output_json = ctx.params.get("output_json", False) if ctx else False

    env, result = _execute()

    if output_json:
        import json as json_mod
        click.echo(json_mod.dumps({
            "scenario": "hallucination-drift",
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

    from tj.core.config import AgentConfig, DriftConfig
    from tj.core.models import NormalizedSpan, SpanKind, SpanStatus
    from tj.demo.env import DemoEnvironment
    from tj.utils.ids import new_span_id, new_trace_id, new_uuid
    from tj.utils.time_parse import utcnow

    agent_cfg = AgentConfig(
        drift=DriftConfig(
            enabled=True,
            baseline_sessions=_BASELINE_SESSIONS,
            token_threshold=2.0,
            tool_sequence_diff=0.4,
        )
    )
    env = DemoEnvironment(agent_configs={AGENT_ID: agent_cfg})

    def _run_session(input_tokens: int, output_tokens: int, tools: list) -> None:
        conv_id = new_uuid()
        trace_id = new_trace_id()
        t = utcnow()

        for j in range(2):
            tok_in = input_tokens // 2 if j == 0 else input_tokens - input_tokens // 2
            tok_out = output_tokens // max(len(tools), 1)
            env.process(NormalizedSpan(
                span_id=new_span_id(),
                trace_id=trace_id,
                name="gen_ai.llm.call",
                kind=SpanKind.CLIENT,
                status_code=SpanStatus.OK,
                start_time=t + timedelta(seconds=j),
                end_time=t + timedelta(seconds=j + 1),
                duration_ms=1000,
                agent_id=AGENT_ID,
                conversation_id=conv_id,
                provider="anthropic",
                model="claude-haiku-4-5",
                input_tokens=tok_in,
                output_tokens=tok_out,
            ))

        for k, tool_name in enumerate(tools):
            env.process(NormalizedSpan(
                span_id=new_span_id(),
                trace_id=trace_id,
                name="gen_ai.tool.call",
                kind=SpanKind.INTERNAL,
                status_code=SpanStatus.OK,
                start_time=t + timedelta(seconds=k + 2),
                end_time=t + timedelta(seconds=k + 2, milliseconds=100),
                duration_ms=100,
                agent_id=AGENT_ID,
                conversation_id=conv_id,
                tool_name=tool_name,
            ))

        end_t = t + timedelta(seconds=len(tools) + 3)
        env.process(NormalizedSpan(
            span_id=new_span_id(),
            trace_id=trace_id,
            name="invoke_agent",
            kind=SpanKind.INTERNAL,
            status_code=SpanStatus.OK,
            start_time=t,
            end_time=end_t,
            duration_ms=(end_t - t).total_seconds() * 1000,
            agent_id=AGENT_ID,
            conversation_id=conv_id,
        ))

    # 5 consistent baseline sessions
    for _ in range(_BASELINE_SESSIONS):
        _run_session(input_tokens=1_000, output_tokens=200, tools=["search", "summarize"])

    # 1 anomalous session: 50x tokens, completely different tools → DRIFT_DETECTED
    _run_session(
        input_tokens=50_000,
        output_tokens=10_000,
        tools=["fetch_url", "parse_html", "extract_entities", "classify", "store_results"],
    )

    return env, env.build_result(AGENT_ID)


def _render(console, result) -> None:
    from rich.panel import Panel

    console.print()
    console.print(Panel.fit(
        "[bold]Incident: My agent worked yesterday. Today it's possessed.[/bold]\n"
        "Scenario: [cyan]hallucination-drift[/cyan]",
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
        f"[bold]Sessions:[/bold] {_BASELINE_SESSIONS} baseline + 1 anomalous\n"
        f"[bold]Spans ingested:[/bold] {result.span_count}\n\n"
        f"[bold]Alerts fired:[/bold]\n{alert_str}\n\n"
        "[bold]The anomalous session had:[/bold]\n"
        "  Input tokens: 50,000 vs baseline ~1,000 (Z-score: inf)\n"
        "  Tool sequence: 5 new tools never seen in baseline\n\n"
        "[dim]In your real agent:[/dim]\n"
        "  ocw drift               [dim]# Z-scores and baseline stats[/dim]\n"
        "  ocw alerts              [dim]# see the drift_detected alert[/dim]"
    )
    console.print(Panel(ocw_output, title="[green]What OCW reveals[/green]", border_style="green"))
    console.print()
