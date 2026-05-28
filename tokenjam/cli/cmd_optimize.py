"""`tj optimize` — surface cost-saving candidates and budget projections."""
from __future__ import annotations

import json

import click
from rich.markup import escape as _rich_escape

from tokenjam.core.optimize import (
    ANALYZER_REGISTRY,
    MODEL_DOWNGRADE_CAVEAT,
    BudgetProjection,
    DowngradeFinding,
    OptimizeReport,
    build_report,
    report_from_dict,
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

    # Two paths depending on whether the daemon holds the DB lock.
    #
    # Local DB available (no daemon, or we got handed a real DuckDBBackend) →
    # build the report locally using db.conn directly. Fastest, no HTTP.
    #
    # Daemon up (main.py handed us an ApiBackend because DuckDB refused to
    # open) → fetch the report from /api/v1/optimize. Previously this path
    # tried to open the DB read-only, but DuckDB blocks read-only attaches
    # while another process holds the write lock — `tj optimize` failed with
    # "Could not set lock on file" any time the daemon was up. See issue
    # #68 §12.
    conn = getattr(db, "conn", None)
    report: OptimizeReport
    plan_mix: dict[str, int]
    if conn is None:
        # API-shim path
        from tokenjam.core.api_backend import ApiBackend
        if not isinstance(db, ApiBackend):
            raise click.ClickException(
                "optimize requires either a direct DuckDB connection or a "
                "running tj serve at the configured api.{host,port}."
            )
        try:
            report_dict = db.fetch_optimize_report(
                since=since,
                agent_id=agent,
                findings=list(findings) if findings else None,
                budget_provider=budget_provider,
                budget_usd=budget_usd,
            )
        except Exception as exc:
            raise click.ClickException(
                f"Failed to fetch optimize report from tj serve: {exc}"
            ) from exc

        if report_dict.get("error") == "no_data":
            if output_json:
                click.echo(json.dumps(report_dict))
            else:
                console.print(
                    "[yellow]No usage data found.[/yellow] "
                    "[dim]Let TokenJam run for a few days, or — if you use "
                    "Claude Code — try [bold]tj backfill claude-code[/bold] to "
                    "ingest historical sessions.[/dim]"
                )
            return

        report = report_from_dict(report_dict)
        # Plan-tier mix is included in the /api/v1/optimize payload as of
        # #68 §12 follow-up #29, so the CLI can render subscription /
        # local / unknown framings correctly under daemon mode.
        plan_mix = report_dict.get("plan_tier_mix") or {}
    else:
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
    cost_diff_dict = None  # populated under API-shim mode
    if compare:
        if conn is None:
            # API-shim path: fetch from /api/v1/cost/compare. Result is a
            # dict (not a CostDiff dataclass) so we render it via
            # _render_diff_dict instead of _render_diff.
            if hasattr(db, "fetch_cost_compare"):
                try:
                    cost_diff_dict = db.fetch_cost_compare(
                        since=since, compare=compare, agent_id=agent,
                    )
                except Exception as exc:
                    raise click.ClickException(
                        f"Failed to fetch --compare from tj serve: {exc}"
                    ) from exc
            else:
                console.print(
                    "[yellow]Note:[/yellow] [dim]--compare is not supported "
                    "via this backend. Continuing without comparison.[/dim]\n"
                )
        else:
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
        elif cost_diff_dict is not None:
            payload["compare"] = cost_diff_dict
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
    elif cost_diff_dict is not None:
        from tokenjam.cli.cmd_cost import _render_diff_dict
        console.print("\n[bold]Window comparison[/bold]")
        _render_diff_dict(cost_diff_dict)


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

    # ----- Wave-2 findings (cache-efficacy, cache-recommend,
    # workflow-restructure, prompt-bloat) — these attach to report.findings
    # rather than typed slots. Dispatch to a per-finding renderer; ignore
    # any unknown finding names (forward-compatible with future analyzers).
    rendered_any_wave2 = False
    for name, finding in (report.findings or {}).items():
        renderer = _FINDING_RENDERERS.get(name)
        if renderer is None:
            continue
        renderer(finding, pricing_mode=pricing_mode)
        console.print()
        rendered_any_wave2 = True

    # Only show the catch-all when truly nothing rendered. The earlier
    # condition missed the Wave-2 findings dict, so cache-efficacy /
    # cache-recommend / workflow-restructure / prompt-bloat were silently
    # falling into "No candidates flagged" (issue #68 §15).
    rendered_any = (
        report.downgrade is not None
        or (pricing_mode == "api" and bool(report.budgets))
        or rendered_any_wave2
    )
    if not rendered_any:
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
        # The per-example cost figure has the same honesty problem as the
        # top-level savings line: in non-api modes we either don't know whether
        # the number is real spend (unknown), or we know it isn't (subscription
        # users on flat fees, local users with zero marginal cost). Drop the
        # column rather than leak the dollar value into a context where we've
        # explicitly suppressed it above. (issue #68 §14)
        for ex in d.examples:
            dur = f"{ex.duration_seconds:.1f}s" if ex.duration_seconds else "—"
            if pricing_mode == "api":
                console.print(
                    f"       [dim]{ex.trace_id[:8]}..[/dim]  "
                    f"{ex.tool_calls} tool calls   {dur}   "
                    f"{format_cost(ex.cost_usd)}  ({ex.model})"
                )
            else:
                console.print(
                    f"       [dim]{ex.trace_id[:8]}..[/dim]  "
                    f"{ex.tool_calls} tool calls   {dur}   "
                    f"({ex.model})"
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


# ---------------------------------------------------------------------------
# Wave-2 finding renderers
# ---------------------------------------------------------------------------
# These render the findings attached to OptimizeReport.findings (the generic
# dict keyed by analyzer name). _FINDING_RENDERERS at the bottom maps each
# analyzer's registration name to the function that renders its finding.
#
# Each renderer takes (finding, pricing_mode=str) and prints to the global
# `console`. Adding a new analyzer: add a renderer here and an entry in the
# dispatch table. cmd_optimize._render_report iterates report.findings and
# calls into here.

def _render_cache_efficacy(finding, *, pricing_mode: str = "api") -> None:
    """
    Render the cache-efficacy finding — current caching-ratio table per
    (provider, model). When any rows are flagged, surface them prominently;
    otherwise show the full table dimmed so the user sees the underlying
    data even when no recommendation is warranted.
    """
    console.print("  [bold]Cache efficacy:[/bold]")
    if not finding.rows:
        console.print(
            "     [dim]No LLM spans with provider/model in this window.[/dim]"
        )
        return

    flagged = list(finding.flagged) if finding.flagged else []
    if flagged:
        console.print(
            f"     • [bold]{len(flagged)}[/bold] (provider, model) "
            f"row{'s' if len(flagged) != 1 else ''} flagged below the "
            f"30% efficacy threshold at ≥100K input tokens:"
        )
        for r in flagged:
            console.print(
                f"       [bold]{r.provider}/{r.model}[/bold]  "
                f"{r.efficacy*100:.0f}% efficacy  "
                f"({format_tokens(r.input_tokens)} input / "
                f"{format_tokens(r.cache_tokens)} cache)"
            )
        console.print()

    console.print("     [dim]All (provider, model) usage in window:[/dim]")
    for r in finding.rows:
        caveat = ""
        if r.support == "best_effort":
            caveat = " [dim](best-effort)[/dim]"
        elif r.support == "unsupported":
            caveat = " [dim](unsupported)[/dim]"
        flag_marker = "[yellow]![/yellow] " if r.flagged else "  "
        console.print(
            f"     {flag_marker}{r.provider}/{r.model}  "
            f"[dim]efficacy[/dim] {r.efficacy*100:.0f}%  "
            f"[dim]input[/dim] {format_tokens(r.input_tokens)}  "
            f"[dim]cache[/dim] {format_tokens(r.cache_tokens)}"
            f"{caveat}"
        )


def _render_cache_recommend(finding, *, pricing_mode: str = "api") -> None:
    """
    Render the cache-recommend finding — Anthropic-only v1 breakpoint
    candidates. When the analyzer is disabled (capture.prompts off), surface
    the hint instead of an empty table.
    """
    console.print("  [bold]Cache recommend:[/bold]")
    if not finding.enabled:
        # Hint includes the install / config instruction from the analyzer.
        # _rich_escape because the hint contains TOML section names like
        # `[capture]` which Rich would otherwise interpret as a style tag
        # and silently strip from the output.
        if finding.hint:
            console.print(f"     [dim]{_rich_escape(finding.hint)}[/dim]")
        else:
            console.print(
                "     [dim]Disabled. Set [bold]capture.prompts = true[/bold] "
                "in tj.toml to run this analyzer.[/dim]"
            )
        return

    if not finding.candidates:
        msg = "     [dim]No stable prefixes shared across ≥3 Anthropic calls"
        if finding.skipped_provider_count:
            msg += (
                f". Skipped {finding.skipped_provider_count} non-Anthropic "
                f"span(s) — multi-provider support is a future feature."
            )
        msg += ".[/dim]"
        console.print(msg)
        return

    console.print(
        f"     • [bold]{len(finding.candidates)}[/bold] prefix candidate"
        f"{'s' if len(finding.candidates) != 1 else ''} for "
        f"[bold]cache_control[/bold] placement:"
    )
    for c in finding.candidates:
        sample = c.sample_chars.replace("\n", " ")[:80]
        if len(c.sample_chars) > 80:
            sample = sample[:77] + "..."
        console.print(
            f"       [dim]{c.prefix_hash[:8]}..[/dim]  "
            f"{c.occurrences}× shared  "
            f"~{format_tokens(c.estimated_cacheable_tokens)} cacheable/call  "
            f"[dim]({format_tokens(int(c.avg_input_tokens))} avg input)[/dim]"
        )
        console.print(f"           [dim italic]{sample}[/dim italic]")

    if finding.skipped_provider_count:
        console.print(
            f"     [dim]Note: {finding.skipped_provider_count} non-Anthropic "
            f"span(s) skipped — multi-provider support is a future feature.[/dim]"
        )


def _render_workflow_restructure(finding, *, pricing_mode: str = "api") -> None:
    """
    Render the workflow-restructure (Script) finding — clusters of sessions
    matching the same (tool_name, arg_shape) signature.
    """
    console.print("  [bold]Workflow restructure:[/bold]")
    if not finding.clusters:
        if finding.sessions_examined == 0:
            console.print(
                "     [dim]No tool spans in this window.[/dim]"
            )
        else:
            console.print(
                f"     [dim]Examined {finding.sessions_examined} session"
                f"{'s' if finding.sessions_examined != 1 else ''}; "
                f"no clusters above threshold (≥20 identical signatures, "
                f"zero branching).[/dim]"
            )
        if finding.degraded:
            console.print(
                "     [dim]Clustering ran in tool-names-only mode "
                "(capture.tool_inputs = false). Enable to "
                "cluster by argument shape too.[/dim]"
            )
        return

    note = ""
    if finding.degraded:
        note = " [dim](tool-names-only — enable capture.tool_inputs for "\
               "finer clustering)[/dim]"
    console.print(
        f"     • [bold]{len(finding.clusters)}[/bold] deterministic-pattern "
        f"cluster{'s' if len(finding.clusters) != 1 else ''} found{note}"
    )
    for c in finding.clusters:
        # Build a compact signature preview
        sig_preview = " → ".join(
            f"{step['tool']}({','.join(step.get('args', [])) or '-'})"
            for step in c.signature
        )
        if len(sig_preview) > 100:
            sig_preview = sig_preview[:97] + "..."
        dur = (
            f"{c.avg_duration_seconds:.1f}s avg"
            if c.avg_duration_seconds else "—"
        )
        console.print(
            f"       [bold]{c.instances}×[/bold] {sig_preview}  "
            f"[dim]({dur})[/dim]"
        )
        if pricing_mode == "api" and c.avg_cost_usd > 0:
            console.print(
                f"          [dim]avg session cost {format_cost(c.avg_cost_usd)}; "
                f"replacing with a deterministic script would eliminate it.[/dim]"
            )
    if finding.caveat:
        console.print(f"     [yellow]![/yellow] [italic]{finding.caveat}[/italic]")


def _render_prompt_bloat(finding, *, pricing_mode: str = "api") -> None:
    """
    Render the prompt-bloat (Trim) finding — LLMLingua-2 token-significance
    summary. When the analyzer is disabled (either capture off or extra
    not installed), surface the hint.
    """
    console.print("  [bold]Prompt bloat:[/bold]")
    if not finding.enabled:
        if finding.hint:
            # Escape Rich markup — hints can contain TOML section names
            # (`[capture]`) or bracketed install hints (`tokenjam[bloat]`).
            console.print(f"     [dim]{_rich_escape(finding.hint)}[/dim]")
        else:
            console.print(
                "     [dim]Disabled. See "
                "[bold]docs/optimize/trim.md[/bold] for install + capture "
                "requirements.[/dim]"
            )
        return

    if not finding.per_prompt:
        console.print(
            f"     [dim]Scanned {finding.prompts_scored} prompt"
            f"{'s' if finding.prompts_scored != 1 else ''}; "
            f"skipped {finding.prompts_skipped}. No bloat regions above "
            f"the minimum-length threshold.[/dim]"
        )
        return

    pct = (
        finding.total_bloat_chars / finding.total_chars * 100.0
        if finding.total_chars > 0 else 0.0
    )
    console.print(
        f"     • Scored [bold]{finding.prompts_scored}[/bold] prompt"
        f"{'s' if finding.prompts_scored != 1 else ''}: "
        f"[bold]{pct:.1f}%[/bold] of chars in flagged regions "
        f"([bold]{finding.total_bloat_chars}[/bold] / "
        f"{finding.total_chars})"
    )
    console.print("     • Top prompts by bloat volume:")
    for p in finding.per_prompt[:5]:
        sample = p.sample_chars.replace("\n", " ")[:80]
        if len(p.sample_chars) > 80:
            sample = sample[:77] + "..."
        console.print(
            f"       [dim]{p.agent_id}[/dim]  "
            f"[bold]{p.bloat_chars}[/bold] bloat / {p.prompt_chars} chars  "
            f"[dim]~{p.estimated_token_reduction} tokens trimmable[/dim]"
        )
        console.print(f"           [dim italic]{sample}[/dim italic]")
    console.print(
        "     [dim]For per-prompt highlights run: "
        "[bold]tj report --bloat[/bold][/dim]"
    )


# Dispatch table — analyzer registration name → renderer.
_FINDING_RENDERERS = {
    "cache-efficacy":       _render_cache_efficacy,
    "cache-recommend":      _render_cache_recommend,
    "workflow-restructure": _render_workflow_restructure,
    "prompt-bloat":         _render_prompt_bloat,
}
