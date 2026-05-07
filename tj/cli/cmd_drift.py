"""tj drift — show behavioral drift baselines and Z-scores."""
from __future__ import annotations

import json as json_mod

import click
from rich.table import Table

from tj.core.drift import evaluate_drift
from tj.utils.formatting import console


@click.command("drift")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("--json", "output_json", is_flag=True, help="JSON output")
@click.pass_context
def cmd_drift(ctx: click.Context, agent: str | None, output_json: bool) -> None:
    """Show drift baselines and Z-scores for recent sessions."""
    db = ctx.obj["db"]
    config = ctx.obj["config"]
    agent_filter = agent or ctx.obj.get("agent")

    # Discover agents with baselines
    if agent_filter:
        agent_ids = [agent_filter]
    elif hasattr(db, "conn"):
        rows = db.conn.execute(
            "SELECT DISTINCT agent_id FROM drift_baselines ORDER BY agent_id"
        ).fetchall()
        agent_ids = [r[0] for r in rows]
    else:
        agent_ids = []

    if not agent_ids:
        if output_json:
            click.echo(json_mod.dumps({"agents": [], "drifted": False}))
        else:
            console.print(
                "[dim]No drift baselines found. "
                "Need at least 10 completed sessions to build a baseline.[/dim]"
            )
        ctx.exit(0)
        return

    all_results = []
    any_drifted = False

    for aid in agent_ids:
        baseline = db.get_baseline(aid)
        if baseline is None:
            continue

        sessions = db.get_completed_sessions(aid, limit=1)
        if not sessions:
            continue
        latest = sessions[0]

        agent_cfg = config.agents.get(aid)
        threshold = agent_cfg.drift.token_threshold if agent_cfg else 2.0
        seq_threshold = agent_cfg.drift.tool_sequence_diff if agent_cfg else 0.4

        result = evaluate_drift(
            session=latest,
            baseline=baseline,
            config_threshold=threshold,
            sequence_diff_threshold=seq_threshold,
            db=db,
        )

        if result.drifted:
            any_drifted = True

        agent_data = {
            "agent_id": aid,
            "baseline_sessions": baseline.sessions_sampled,
            "drifted": result.drifted,
            "violations": [
                {
                    "dimension": v.dimension,
                    "z_score": v.z_score,
                    "expected": v.expected,
                    "observed": v.observed,
                    "detail": v.detail,
                }
                for v in result.violations
            ],
            "metrics": _build_metrics(baseline, latest, result, threshold),
        }
        all_results.append(agent_data)

        if not output_json:
            _print_drift_table(aid, baseline, latest, result, threshold, seq_threshold)

    if output_json:
        click.echo(json_mod.dumps(
            {"agents": all_results, "drifted": any_drifted},
            default=str,
        ))

    ctx.exit(1 if any_drifted else 0)


def _build_metrics(baseline, session, result, threshold: float) -> list[dict]:
    """Return per-dimension metric dicts for JSON output."""
    from tj.core.drift import z_score

    violated_dims = {v.dimension for v in result.violations}
    metrics = []

    def _add(dimension: str, mean, stddev, current) -> None:
        if mean is None or stddev is None:
            return
        z = z_score(float(current), float(mean), float(stddev))
        metrics.append({
            "dimension": dimension,
            "baseline_mean": mean,
            "baseline_stddev": stddev,
            "current_value": current,
            "z_score": z,
            "status": "DRIFT" if dimension in violated_dims else "ok",
        })

    _add("input_tokens", baseline.avg_input_tokens, baseline.stddev_input_tokens,
         session.input_tokens)
    _add("output_tokens", baseline.avg_output_tokens, baseline.stddev_output_tokens,
         session.output_tokens)
    if session.duration_seconds is not None:
        _add("session_duration", baseline.avg_session_duration_s,
             baseline.stddev_session_duration, session.duration_seconds)
    _add("tool_call_count", baseline.avg_tool_call_count, baseline.stddev_tool_call_count,
         session.tool_call_count)

    # tool_sequence is special (Jaccard, no z-score)
    if "tool_sequence" in violated_dims:
        seq_viol = next((v for v in result.violations if v.dimension == "tool_sequence"), None)
        if seq_viol:
            metrics.append({
                "dimension": "tool_sequence",
                "baseline_mean": None,
                "baseline_stddev": None,
                "current_value": seq_viol.observed,
                "z_score": None,
                "status": "DRIFT",
            })

    return metrics


def _print_drift_table(aid, baseline, session, result, threshold: float, seq_threshold: float = 0.4) -> None:
    """Render a Rich table for a single agent's drift state."""
    from tj.core.drift import z_score

    violated_dims = {v.dimension for v in result.violations}
    status_label = "[bold red]DRIFTED[/bold red]" if result.drifted else "[green]ok[/green]"

    console.print()
    console.print(
        f"[bold]Agent:[/bold] {aid}  |  "
        f"[bold]Baseline:[/bold] {baseline.sessions_sampled} sessions  |  "
        f"[bold]Status:[/bold] {status_label}"
    )
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Dimension", style="dim")
    table.add_column("Baseline")
    table.add_column("Current")
    table.add_column("Z-Score", justify="right")
    table.add_column("Status")

    def _z_color(z: float | None) -> str:
        if z is None:
            return "--"
        az = abs(z)
        if az < 1.0:
            return f"[green]{z:.2f}[/green]"
        if az <= threshold:
            return f"[yellow]{z:.2f}[/yellow]"
        return f"[red]{z:.2f}[/red]"

    def _status_cell(dimension: str) -> str:
        if dimension in violated_dims:
            return "[bold red]DRIFT[/bold red]"
        return "[green]ok[/green]"

    def _add_row(dimension: str, mean, stddev, current, fmt_baseline: str, fmt_current: str) -> None:
        z = z_score(float(current), float(mean), float(stddev)) if (mean is not None and stddev is not None) else None
        table.add_row(dimension, fmt_baseline, fmt_current, _z_color(z), _status_cell(dimension))

    if baseline.avg_input_tokens is not None and baseline.stddev_input_tokens is not None:
        _add_row(
            "input_tokens",
            baseline.avg_input_tokens, baseline.stddev_input_tokens, session.input_tokens,
            f"{baseline.avg_input_tokens:,.0f} +/- {baseline.stddev_input_tokens:,.0f}",
            f"{session.input_tokens:,}",
        )
    if baseline.avg_output_tokens is not None and baseline.stddev_output_tokens is not None:
        _add_row(
            "output_tokens",
            baseline.avg_output_tokens, baseline.stddev_output_tokens, session.output_tokens,
            f"{baseline.avg_output_tokens:,.0f} +/- {baseline.stddev_output_tokens:,.0f}",
            f"{session.output_tokens:,}",
        )
    if (
        session.duration_seconds is not None
        and baseline.avg_session_duration_s is not None
        and baseline.stddev_session_duration is not None
    ):
        _add_row(
            "session_duration",
            baseline.avg_session_duration_s, baseline.stddev_session_duration,
            session.duration_seconds,
            f"{baseline.avg_session_duration_s:.1f}s +/- {baseline.stddev_session_duration:.1f}s",
            f"{session.duration_seconds:.1f}s",
        )
    if baseline.avg_tool_call_count is not None and baseline.stddev_tool_call_count is not None:
        _add_row(
            "tool_call_count",
            baseline.avg_tool_call_count, baseline.stddev_tool_call_count, session.tool_call_count,
            f"{baseline.avg_tool_call_count:.0f} +/- {baseline.stddev_tool_call_count:.0f}",
            str(session.tool_call_count),
        )

    # Tool sequence row (Jaccard, no z-score)
    seq_viol = next((v for v in result.violations if v.dimension == "tool_sequence"), None)
    if seq_viol:
        table.add_row(
            "tool_sequence",
            seq_viol.expected or "",
            seq_viol.observed or "",
            "--",
            "[bold red]DRIFT[/bold red]",
        )
    elif baseline.common_tool_sequences:
        min_sim = 1.0 - seq_threshold
        table.add_row("tool_sequence", f"similarity >= {min_sim:.2f}", "--", "--", "[green]ok[/green]")

    console.print(table)
