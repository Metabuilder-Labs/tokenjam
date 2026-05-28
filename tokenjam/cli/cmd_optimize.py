"""`tj optimize` — surface cost-saving candidates and budget projections."""
from __future__ import annotations

import json

import click

from tokenjam.core.optimize import (
    ANALYZER_REGISTRY,
    MODEL_DOWNGRADE_CAVEAT,
    BudgetProjection,
    DowngradeFinding,
    OptimizeReport,
    build_report,
    report_to_dict,
)
from tokenjam.otel.semconv import SUBSCRIPTION_PLAN_TIERS
from tokenjam.utils.formatting import (
    console,
    format_cost,
    format_tokens,
)
from tokenjam.utils.time_parse import parse_since, utcnow


# Subscription plan label + flat monthly fee. Used in the implied-API-value
# header line. Keys must match SessionRecord.plan_tier values. Plans whose
# fee is contract-priced (enterprise, team variants) have no fee here — the
# header line skips the multiplier in that case.
_PLAN_LABEL_AND_FEE: dict[str, tuple[str, float | None]] = {
    "pro":        ("Pro plan",         20.0),
    "max_5x":     ("Max 5x plan",     100.0),
    "max_20x":    ("Max 20x plan",    200.0),
    "plus":       ("ChatGPT Plus",     20.0),
    "team":       ("ChatGPT Team",      None),
    "enterprise": ("ChatGPT Enterprise", None),
}


def _pricing_mode_for(plan_tier: str) -> str:
    """Mirror SessionRecord.pricing_mode without needing an instance."""
    if plan_tier == "local":
        return "local"
    if plan_tier in SUBSCRIPTION_PLAN_TIERS:
        return "subscription"
    if plan_tier == "api":
        return "api"
    return "unknown"


def _dominant_plan(plan_mix: dict[str, int]) -> str:
    """
    Pick the rendering mode from the plan-tier mix.

    - If plan_mix is empty (e.g. spans inserted without sessions in test
      fixtures), default to 'api' — the historical rendering mode. Real
      users always have a populated sessions table because IngestPipeline
      creates sessions before writing spans.
    - If any non-unknown plan_tier is present, return the most common one.
    - Otherwise return 'unknown' (the caller handles dollar suppression).
    """
    if not plan_mix:
        return "api"
    known = {k: v for k, v in plan_mix.items() if k != "unknown"}
    if not known:
        return "unknown"
    return max(known.items(), key=lambda kv: kv[1])[0]


@click.command("optimize")
@click.option("--agent", default=None, help="Scope to a specific agent_id.")
@click.option("--since", default="30d", help="Window for analysis (default 30d).")
@click.option(
    "--finding",
    "findings",
    multiple=True,
    type=click.Choice(sorted(ANALYZER_REGISTRY.keys())),
    help="Run only the named analyzer(s). Repeatable. Default: run all.",
)
@click.option("--budget", "budget_provider", default=None,
              help="Scope budget projection to a single provider (e.g. anthropic).")
@click.option("--budget-usd", type=float, default=None,
              help="Override the configured budget for this run.")
@click.option("--compare", "compare", default=None,
              help="Surface a window-cost diff against a prior period. Accepts "
                   "'previous', 'last-week', 'last-month', 'last-7d', "
                   "'last-30d', or 'YYYY-MM-DD:YYYY-MM-DD'. Analyzers still "
                   "run against the current window only.")
@click.option("--export-config", "export_target", default=None,
              type=click.Choice(["claude-code"]),
              help="Write the current recommendations to a snippet file the "
                   "user can merge into their routing config manually. Does "
                   "not modify any file outside the TokenJam config directory.")
@click.option("--json", "output_json", is_flag=True,
              help="Emit machine-readable JSON.")
@click.pass_context
def cmd_optimize(
    ctx: click.Context,
    agent: str | None,
    since: str,
    findings: tuple[str, ...],
    budget_provider: str | None,
    budget_usd: float | None,
    compare: str | None,
    export_target: str | None,
    output_json: bool,
) -> None:
    """Analyze recent usage for cost-saving candidates and budget exposure."""
    db = ctx.obj.get("db")
    config = ctx.obj.get("config")
    if db is None or config is None:
        raise click.ClickException("optimize requires a database connection.")

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    until_dt = utcnow()

    # If tj serve is running it holds the write-lock, and main.py hands us an
    # API-proxy backend that has no .conn. Open the DB ourselves read-only so
    # optimize works alongside a running server.
    conn = getattr(db, "conn", None)
    if conn is None:
        import duckdb
        from pathlib import Path as _Path
        db_path = str(_Path(config.storage.path).expanduser())
        try:
            ro_conn = duckdb.connect(db_path, read_only=True)
        except Exception as exc:
            raise click.ClickException(
                f"Could not open {db_path} read-only: {exc}"
            ) from exc

        class _RoShim:
            def __init__(self, c):
                self.conn = c
        db = _RoShim(ro_conn)
        conn = ro_conn
    row = conn.execute(
        "SELECT COUNT(*) FROM spans WHERE model IS NOT NULL"
    ).fetchone()
    if not row or not row[0]:
        if output_json:
            click.echo(json.dumps({
                "error": "no_data",
                "message": "No span data available — let TokenJam run for a few "
                           "days, or `tj backfill claude-code` if you use Claude Code.",
            }))
        else:
            console.print(
                "[yellow]No usage data found.[/yellow] "
                "[dim]Let TokenJam run for a few days, or — if you use "
                "Claude Code — try [bold]tj backfill claude-code[/bold] to "
                "ingest historical sessions.[/dim]"
            )
        return

    report = build_report(
        db=db,
        config=config,
        since=since_dt,
        until=until_dt,
        agent_id=agent,
        findings=list(findings) if findings else None,
        budget_provider_filter=budget_provider,
        budget_usd_override=budget_usd,
    )

    # Query plan-tier mix in this window. Used for both the unknown-tier note
    # (0.4b) and dominant-plan rendering (when Task 1.1 ships the full
    # subscription reframing this stays useful as the signal).
    plan_mix = _plan_tier_mix(conn, since_dt, until_dt, agent)

    dominant = _dominant_plan(plan_mix)
    pricing_mode = _pricing_mode_for(dominant)

    # --export-config branch: write the snippet to disk and exit. Skips
    # the normal rendering path. The user reads the snippet file and
    # copies the routing block into their routing layer manually.
    if export_target:
        _export_snippet(
            report.downgrade, dominant, pricing_mode,
            target=export_target, agent_id=agent,
            output_json=output_json,
        )
        return

    # Optional period comparison. Independent of the analyzer findings —
    # surfaces a window-cost diff at the top so the user can see trend
    # before reading the recommendations.
    cost_diff = None
    if compare:
        from tokenjam.core.cost import compute_cost_diff
        try:
            cost_diff = compute_cost_diff(db, since_dt, until_dt, compare, agent_id=agent)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="'--compare'") from exc

    if output_json:
        payload = report_to_dict(report)
        payload["plan_tier_mix"] = plan_mix
        payload["plan"] = dominant
        payload["pricing_mode"] = pricing_mode
        if cost_diff is not None:
            from tokenjam.cli.cmd_cost import _diff_to_dict
            payload["compare"] = _diff_to_dict(cost_diff)
        # For subscription/local users, the dollar fields on the downgrade
        # finding mislead — surface the token-share fields instead. Don't
        # remove actual_cost_usd / alternative_cost_usd; those are useful
        # raw data. Just zero out the savings_usd projection and add
        # monthly_tokens_freed alongside.
        d = payload.get("downgrade") or {}
        if d and pricing_mode in {"subscription", "local"}:
            d["monthly_tokens_freed"] = d.get("monthly_tokens_in_candidates", 0)
            if pricing_mode == "local":
                d["monthly_savings_usd"] = 0
        click.echo(json.dumps(payload, default=str))
        return

    _render_report(
        report, agent=agent, plan_mix=plan_mix,
        dominant_plan=dominant, pricing_mode=pricing_mode,
    )
    if cost_diff is not None:
        from tokenjam.cli.cmd_cost import _render_diff
        console.print("\n[bold]Window comparison[/bold]")
        _render_diff(cost_diff)


def _plan_tier_mix(conn, since, until, agent_id: str | None) -> dict[str, int]:
    """Count sessions by plan_tier inside the analysis window."""
    clauses = ["started_at >= $1", "started_at < $2"]
    params: list = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT COALESCE(plan_tier, 'unknown'), COUNT(*) FROM sessions "
        f"WHERE {where} GROUP BY 1",
        params,
    ).fetchall()
    return {str(r[0]): int(r[1]) for r in rows}


# ---------------------------------------------------------------------------
# Human-readable renderer
# ---------------------------------------------------------------------------

def _render_report(
    report: OptimizeReport,
    agent: str | None,
    plan_mix: dict[str, int] | None = None,
    dominant_plan: str = "unknown",
    pricing_mode: str = "unknown",
) -> None:
    w = report.window
    scope_tag = f", {agent}" if agent else ""
    days_int = max(int(round(w.days)), 1)
    plan_mix = plan_mix or {}
    unknown_count = plan_mix.get("unknown", 0)
    total_sessions = sum(plan_mix.values()) or w.sessions
    all_unknown = total_sessions > 0 and unknown_count == total_sessions

    # ----- Header -----
    if all_unknown:
        console.print(
            f"\nAnalyzing [bold]{w.sessions}[/bold] sessions, "
            f"[bold]{format_tokens(w.total_tokens)}[/bold] tokens "
            f"(last {days_int}d{scope_tag})\n"
            f"[dim]All sessions have unknown plan tier; dollar figures suppressed. "
            f"Run [bold]tj onboard --reconfigure[/bold] to set your plan.[/dim]\n"
        )
    elif pricing_mode == "subscription":
        label, fee = _PLAN_LABEL_AND_FEE.get(dominant_plan, (dominant_plan, None))
        plan_suffix = f", ${fee:.0f}/mo flat" if fee else ""
        console.print(
            f"\nAnalyzing [bold]{w.sessions}[/bold] sessions, "
            f"[bold]{format_tokens(w.total_tokens)}[/bold] tokens this cycle "
            f"([bold]{label}[/bold]{plan_suffix})…"
        )
        if fee and w.total_cost_usd > 0:
            multiplier = w.total_cost_usd / fee
            console.print(
                f"[dim]Implied API value: "
                f"[bold]{format_cost(w.total_cost_usd)}[/bold] — about "
                f"{multiplier:.1f}× your plan cost.[/dim]\n"
            )
        else:
            console.print(
                f"[dim]Implied API value: "
                f"[bold]{format_cost(w.total_cost_usd)}[/bold] "
                f"(what this usage would cost at API list prices).[/dim]\n"
            )
    elif pricing_mode == "local":
        console.print(
            f"\nAnalyzing [bold]{w.sessions}[/bold] sessions, "
            f"[bold]{format_tokens(w.total_tokens)}[/bold] tokens "
            f"(last {days_int}d{scope_tag})\n"
            f"[dim]Local inference — no marginal cost.[/dim]\n"
        )
    else:
        # api mode (default current behavior)
        console.print(
            f"\nAnalyzing [bold]{w.sessions}[/bold] sessions, "
            f"[bold]{format_tokens(w.total_tokens)}[/bold] tokens, "
            f"[bold]{format_cost(w.total_cost_usd)}[/bold] spend "
            f"(last {days_int}d{scope_tag})…\n"
        )
        if unknown_count > 0:
            console.print(
                f"[dim]Note: {unknown_count} of {total_sessions} sessions have "
                f"unknown plan tier; dollar figures may overstate actual cost "
                f"for those. Run [bold]tj onboard --reconfigure[/bold] to "
                f"resolve.[/dim]\n"
            )

    if w.sessions == 0:
        console.print("[dim]No sessions in window.[/dim]")
        return

    for note in report.notes:
        console.print(f"  [yellow]![/yellow] {note}")
    if report.notes:
        console.print()

    # ----- Model-downgrade body -----
    if report.downgrade is not None:
        _render_downgrade(
            report.downgrade,
            pricing_mode=("unknown" if all_unknown else pricing_mode),
        )
        console.print()

    # ----- Budget projection -----
    # Subscription users don't have a dollar-denominated budget projection;
    # the [budget.<provider>] section may exist as a self-imposed soft
    # ceiling, but rendering it as a hard cap would mislead. Suppress in
    # subscription/local/unknown modes — surface only in api mode.
    if pricing_mode == "api":
        for proj in report.budgets:
            _render_budget(proj)
            console.print()

    if (
        report.downgrade is None
        and not (pricing_mode == "api" and report.budgets)
    ):
        console.print(
            "[dim]No candidates flagged in this window. Either spend is small or "
            "all sessions already use a cost-effective model.[/dim]"
        )


def _render_downgrade(d: DowngradeFinding, pricing_mode: str = "api") -> None:
    """
    Render the model-downgrade finding for the given pricing mode.

    - api:          dollar-denominated savings (current behavior)
    - subscription: token-share framing — "candidate sessions are X% of your
                    cycle's tokens; routing them to {alt} frees that share
                    against your plan cap"
    - local:        token-only framing for capacity planning
    - unknown:      structural-only, no savings figures
    """
    console.print(
        f"  [bold]① Model downgrade:[/bold] "
        f"{d.percent_of_sessions:.0f}% of sessions match a smaller-model "
        f"candidate shape"
    )
    console.print(
        f"     • {d.candidate_sessions} of {d.total_sessions} sessions matched "
        f"structural heuristics"
    )

    if pricing_mode == "unknown":
        console.print(
            "     • Structural shape matches a cheaper-model candidate class "
            "(savings figures suppressed — plan tier unknown)"
        )
    elif pricing_mode == "subscription":
        console.print(
            f"     • Those sessions hold "
            f"[bold]~{d.percent_of_tokens:.0f}%[/bold] of this cycle's tokens "
            f"({format_tokens(d.candidate_tokens)} of "
            f"{format_tokens(d.window_total_tokens)})"
        )
        console.print(
            f"     • Routing them to a smaller model would free that share "
            f"against your plan's allocation "
            f"(~{format_tokens(d.monthly_tokens_in_candidates)}/mo at this rate)"
        )
    elif pricing_mode == "local":
        console.print(
            f"     • Those sessions consumed "
            f"[bold]{format_tokens(d.candidate_tokens)}[/bold] tokens "
            f"({d.percent_of_tokens:.0f}% of the window)"
        )
        console.print(
            "     • [dim]Relevant for capacity planning if you switch this "
            "workload to API-billed inference.[/dim]"
        )
    else:  # api
        console.print(
            f"     • Would have cost ~{format_cost(d.alternative_cost_usd)} on the "
            f"smaller model vs {format_cost(d.actual_cost_usd)} actual (in window)"
        )
        console.print(
            f"     • Projected savings if pattern holds: "
            f"[bold]{format_cost(d.monthly_savings_usd)}/mo[/bold]"
        )
    if d.suggestions:
        pairs = ", ".join(f"{k} → {v}" for k, v in d.suggestions.items())
        console.print(f"     • Pattern: [dim]{pairs}[/dim]")

    if d.examples:
        console.print()
        console.print("     [dim]Examples:[/dim]")
        for ex in d.examples:
            dur = f"{ex.duration_seconds:.1f}s" if ex.duration_seconds else "—"
            console.print(
                f"       [dim]{ex.trace_id[:8]}..[/dim]  "
                f"{ex.tool_calls} tool calls   {dur}   "
                f"{format_cost(ex.cost_usd)}  ({ex.model})"
            )
    console.print()
    console.print(
        f"     [yellow]![/yellow] [italic]{MODEL_DOWNGRADE_CAVEAT}[/italic]"
    )


def _render_budget(p: BudgetProjection) -> None:
    headline = f"  [bold]② Budget projection ({p.provider}, " \
               f"{format_cost(p.budget_usd)}/cycle):[/bold] "

    if not p.over_budget:
        unused_pct = max(0, int(round(100 * (1 - p.projected_cycle_total / p.budget_usd))))
        console.print(headline + "comfortably within budget")
        console.print(
            f"     Run rate "
            f"[bold]{format_cost(p.monthly_run_rate_usd)}/mo[/bold] — "
            f"{unused_pct}% of cycle budget unused."
        )
        return

    console.print(headline + "[red]projected to exceed cycle budget[/red]")
    console.print(
        f"     • Monthly run rate: "
        f"[bold]{format_cost(p.monthly_run_rate_usd)}[/bold] "
        f"({p.monthly_run_rate_usd / p.budget_usd:.1f}× the budget)"
    )
    if p.exhaustion_date:
        console.print(
            f"     • At current pace, budget exhausted on "
            f"[bold]{p.exhaustion_date.strftime('%Y-%m-%d')}[/bold] "
            f"({p.days_until_exhaustion:.1f} day(s) from now)"
        )
    console.print(f"     • Days remaining in cycle: {p.days_remaining:.0f}")
    console.print(
        f"     • Projected cycle total: "
        f"{format_cost(p.projected_cycle_total)}, "
        f"overage: [red]{format_cost(p.projected_overage_usd)}[/red]"
    )
    if p.downgrade_run_rate_usd is not None and p.downgrade_run_rate_usd < p.monthly_run_rate_usd:
        console.print(
            f"     • With model-downgrade pattern: run rate drops to "
            f"[bold]{format_cost(p.downgrade_run_rate_usd)}/mo[/bold]"
        )
    if p.applies_to_services:
        console.print(
            f"     [dim]Counted services: {', '.join(p.applies_to_services)}[/dim]"
        )


def _export_snippet(
    downgrade,
    dominant_plan: str,
    pricing_mode: str,
    *,
    target: str,
    agent_id: str | None,
    output_json: bool,
) -> None:
    """
    Write a routing-config snippet for the requested target and print a
    pointer to the file. No file outside ~/.config/tokenjam/exports/ is
    touched — the user merges the snippet manually.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    if target == "claude-code":
        from tokenjam.core.export.claude_code import render_claude_code_snippet
        body = render_claude_code_snippet(
            downgrade=downgrade,
            pricing_mode=pricing_mode,
            plan_tier=dominant_plan,
            agent_id=agent_id,
        )
        ext = "json"
    else:
        # Click's Choice() already constrained this; defensive only.
        raise click.ClickException(f"Unknown export target: {target}")

    out_dir = Path.home() / ".config" / "tokenjam" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    out_path = out_dir / f"{target}-{today}.{ext}"
    out_path.write_text(body)

    if output_json:
        click.echo(json.dumps({
            "target": target,
            "path": str(out_path),
            "plan_tier": dominant_plan,
            "pricing_mode": pricing_mode,
        }, default=str))
        return

    console.print(
        f"[green]✓[/green] Snippet written to [bold]{out_path}[/bold]."
    )
    if target == "claude-code":
        console.print(
            "\nOpen the file and copy the routing block into your "
            "[bold].claude/settings.json[/bold] or your routing layer of "
            "choice (LiteLLM router config, framework code, etc.).\n"
            "\n[dim]TokenJam does not enforce these rules. The snippet is "
            "a recommendation, not an active routing config.[/dim]"
        )
