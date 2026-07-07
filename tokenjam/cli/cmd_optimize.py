"""`tj optimize` — surface cost-saving candidates and budget projections."""
from __future__ import annotations

import json
from typing import Any

import click
from rich.markup import escape as _rich_escape

from tokenjam.core.framing import (
    PLAN_LABEL_AND_FEE,
    agent_persona_mix,
    config_declared_plan,
    dominant_persona,
    dominant_plan,
    plan_tier_mix,
    pricing_mode_for,
)
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
from tokenjam.utils.formatting import (
    console,
    format_cost,
    format_tokens,
)
from tokenjam.utils.time_parse import parse_since, utcnow

# Plan-tier framing helpers (PLAN_LABEL_AND_FEE, pricing_mode_for,
# dominant_plan, config_declared_plan, plan_tier_mix) live in
# tokenjam.core.framing — the single source shared with cmd_tokenmaxx and the
# REST API. See issue #110. agent_persona_mix / dominant_persona (also
# framing.py) classify the window's dominant user (Claude Code subscriber vs
# SDK/API developer) so the downsize finding's call-to-action matches the
# levers that persona actually has — see #97.


@click.command("optimize")
@click.argument(
    "findings",
    nargs=-1,
    type=click.Choice(sorted(ANALYZER_REGISTRY.keys())),
)
@click.option("--agent", default=None, help="Scope to a specific agent_id.")
@click.option("--since", default="30d", help="Window for analysis (default 30d).")
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
@click.option("--export-templates", "export_templates", is_flag=True, default=False,
              help="Reuse only: write per-cluster Markdown skeletons to the "
                   "reports directory without opening the HTML report.")
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
    export_templates: bool,
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

    # If user passed --compare last-7d / last-30d / last-week, override
    # --since so the analysis window matches the comparison period (#71
    # finding 5). Without this, `tj optimize --compare last-7d` would do
    # 30d-vs-30d (because --since defaults to 30d), while `tj cost` did
    # 7d-vs-7d — same flag, two shapes.
    if compare:
        from tokenjam.core.cost import override_since_for_compare
        since_dt = override_since_for_compare(compare, since_dt, until_dt)
        since = f"{(until_dt - since_dt).days}d"

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
        # Agent-persona mix (#97) — same daemon-mode plumbing as plan_mix
        # above, so the downsize CTA matches persona whether or not the
        # daemon is up.
        agent_mix = report_dict.get("agent_persona_mix") or {}
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

        plan_mix = plan_tier_mix(conn, since_dt, until_dt, agent)
        agent_mix = agent_persona_mix(conn, since_dt, until_dt, agent)

    dominant = dominant_plan(plan_mix)
    pricing_mode = pricing_mode_for(dominant)
    declared_plan = config_declared_plan(config)
    persona = dominant_persona(agent_mix, declared_plan=declared_plan)

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

    # --export-templates branch: write the Reuse Markdown skeletons and exit,
    # without rendering the HTML report. Needs direct DB access (to fetch the
    # planning completion text), so it's local-mode only.
    if export_templates:
        _export_reuse_templates(report, conn=conn, config=config, agent=agent)
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
        payload["agent_persona_mix"] = agent_mix
        payload["persona"] = persona
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
            # Zero the misleading dollar projection for BOTH flat-fee
            # subscription tiers and local (zero-marginal-cost) inference —
            # matching the token-share framing the human renderer applies.
            d["monthly_savings_usd"] = 0
        click.echo(json.dumps(payload, default=str))
        return

    _render_report(
        report, agent=agent, plan_mix=plan_mix,
        dominant_plan=dominant, pricing_mode=pricing_mode,
        declared_plan=declared_plan,
        requested=list(findings) if findings else None,
        persona=persona,
    )
    if cost_diff is not None:
        from tokenjam.cli.cmd_cost import _render_diff
        console.print("\n[bold]Window comparison[/bold]")
        _render_diff(cost_diff)
    elif cost_diff_dict is not None:
        from tokenjam.cli.cmd_cost import _render_diff_dict
        console.print("\n[bold]Window comparison[/bold]")
        _render_diff_dict(cost_diff_dict)


# ---------------------------------------------------------------------------
# Finding ranking (#97) — order the numbered slots by reclaimable share of
# the window's tokens, instead of ANALYZER_ORDER (registry order).
# ---------------------------------------------------------------------------

# Minimum estimated-recoverable-token share of the window for a finding to
# occupy a numbered slot. Below this, ranking by share is noise — a finding
# whose analyzer merely happened to run first shouldn't outrank one that's
# actually reclaiming a meaningful fraction of the window. Findings below the
# threshold still render, just collapsed into the "Minor findings" pointer
# list instead of a numbered slot.
DE_MINIMIS_SHARE = 0.01

# Display labels for the "Minor findings" collapsed pointer list — must match
# the header text each renderer prints in its numbered form.
_MINOR_FINDING_LABELS = {
    "downsize":        "Model downgrade",
    "cache":           "Cache efficacy",
    "cache-recommend": "Cache recommend",
    "script":          "Workflow restructure",
    "reuse":           "Reuse",
    "trim":            "Prompt bloat",
    "subagent":        "Subagent right-sizing",
}


def _numbered_marker(n: int) -> str:
    """Circled-digit marker for a ranked finding's slot (①, ②, … ⑳)."""
    if 1 <= n <= 20:
        return chr(0x2460 + n - 1)
    return f"({n})"  # defensive — no report should ever have this many


def _reclaimable_share(finding: Any, window_total_tokens: int) -> float | None:
    """Estimated-recoverable-tokens share of the window, for ranking.

    Returns ``None`` — not 0.0 — when the finding has no quantified estimate
    at all (analyzer disabled, no candidates, or cache-recommend, which
    recommends a cache_control placement rather than a token count). Those
    findings still render in full (they're not "de-minimis", they're
    "unranked") — only a finding with a real-but-tiny share collapses into
    the Minor findings pointer list. Conflating the two would hide an
    analyzer's own diagnostic empty-state message (e.g. "no tool spans in
    this window") behind a generic pointer.
    """
    tokens = getattr(finding, "estimated_recoverable_tokens", None)
    if tokens is None or window_total_tokens <= 0:
        return None
    return max(float(tokens), 0.0) / window_total_tokens


def _rank_findings(
    report: OptimizeReport, requested: list[str] | None,
) -> list[tuple[str, float | None]]:
    """Rank findings with something to show by reclaimable token share
    (largest first; unranked findings — no quantified estimate — sort last).
    Ties fall back to ANALYZER_ORDER for determinism.
    """
    window_tokens = report.window.total_tokens
    order = ["downsize", *_FINDING_RENDERERS.keys()]
    order_index = {name: i for i, name in enumerate(order)}

    items: list[tuple[str, float | None]] = []
    # Render an explicit "no candidates" empty state when the downsize
    # analyzer ran but found nothing — the Optimize web tab does this
    # (PR #130 / issue #126) and the CLI used to silently skip the section,
    # which makes reviewers think the analyzer didn't run. Skip the empty
    # state when the user asked for a different positional subset
    # (`tj optimize cache` shouldn't mention downsize at all).
    downsize_was_requested = (not requested) or ("downsize" in requested)
    if report.downgrade is not None:
        items.append(("downsize", _reclaimable_share(report.downgrade, window_tokens)))
    elif downsize_was_requested:
        items.append(("downsize", None))

    for name, finding in (report.findings or {}).items():
        if name not in _FINDING_RENDERERS:
            continue
        items.append((name, _reclaimable_share(finding, window_tokens)))

    items.sort(key=lambda item: (
        item[1] is None,
        -(item[1] or 0.0),
        order_index.get(item[0], len(order)),
    ))
    return items


# ---------------------------------------------------------------------------
# Human-readable renderer
# ---------------------------------------------------------------------------

def _render_report(
    report: OptimizeReport,
    agent: str | None,
    plan_mix: dict[str, int] | None = None,
    dominant_plan: str = "unknown",
    pricing_mode: str = "unknown",
    declared_plan: str | None = None,
    requested: list[str] | None = None,
    persona: str = "unknown",
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
            f"Run [bold]tj onboard --claude-code --reconfigure[/bold] "
            f"(or [bold]--codex[/bold]) to set your plan.[/dim]\n"
        )
    elif pricing_mode == "subscription":
        label, fee = PLAN_LABEL_AND_FEE.get(dominant_plan, (dominant_plan, None))
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
                f"for those. Run [bold]tj onboard --claude-code --reconfigure[/bold] "
                f"(or [bold]--codex[/bold]) to "
                f"resolve.[/dim]\n"
            )

    # Surface a divergence note when the user has reconfigured to a new plan
    # but historical sessions still reflect the previous plan. Honest framing:
    # show the data as it was actually generated, but flag that future
    # sessions will be costed differently (#71 finding 1).
    if (
        declared_plan
        and declared_plan != dominant_plan
        and declared_plan in PLAN_LABEL_AND_FEE  # only flag subscription deltas
    ):
        label, _ = PLAN_LABEL_AND_FEE[declared_plan]
        console.print(
            f"[dim]Note: your config declares "
            f"[bold]{label}[/bold] but historical sessions ran under "
            f"a different plan — rendering reflects what actually ran. "
            f"New sessions will use the configured plan.[/dim]\n"
        )

    if w.sessions == 0:
        console.print("[dim]No sessions in window.[/dim]")
        return

    for note in report.notes:
        console.print(f"  [yellow]![/yellow] {note}")
    if report.notes:
        console.print()

    # ----- Findings, ranked by reclaimable share of the window's tokens -----
    # Findings used to render in ANALYZER_ORDER (registry order), so a
    # nothing-burger could occupy the top numbered slot just because its
    # analyzer happened to run first — e.g. "① Model downgrade: 28% of
    # sessions match" when those sessions held ~0% of the window's tokens
    # (#97). Rank by estimated_recoverable_tokens / window.total_tokens
    # instead. Three buckets:
    #   major    — real, meaningful share: numbered slot, full render.
    #   unranked — no quantified estimate at all (disabled / no candidates):
    #              full render (own empty-state message), unnumbered — same
    #              as the historical behavior, so diagnostic detail ("no tool
    #              spans in this window") never disappears.
    #   minor    — real but de-minimis share: collapsed to a one-line pointer
    #              so it can't crowd out a finding that actually matters.
    ranked = _rank_findings(report, requested)
    major = [item for item in ranked if item[1] is not None and item[1] >= DE_MINIMIS_SHARE]
    unranked = [item for item in ranked if item[1] is None]
    minor = [item for item in ranked if item[1] is not None and item[1] < DE_MINIMIS_SHARE]

    def _render_finding(name: str, marker: str) -> None:
        if name == "downsize":
            if report.downgrade is not None:
                _render_downgrade(
                    report.downgrade,
                    pricing_mode=("unknown" if all_unknown else pricing_mode),
                    persona=persona,
                    marker=marker,
                )
            else:
                console.print(
                    f"{_finding_header(marker, 'Model downgrade:')} "
                    "[dim]no candidates in this window — sessions don't match "
                    "the smaller-model shape (small input/output, few tool "
                    "calls).[/dim]"
                )
        else:
            _FINDING_RENDERERS[name](
                report.findings[name], pricing_mode=pricing_mode, marker=marker,
            )

    for slot, (name, _share) in enumerate(major, start=1):
        _render_finding(name, _numbered_marker(slot))
        console.print()

    for name, _share in unranked:
        _render_finding(name, "")
        console.print()

    # ----- Budget projection -----
    # Not part of the reclaimable-share ranking above — it's a forward-looking
    # cap/overage exposure, not a recoverable-tokens finding, so it always
    # renders in its own section rather than competing for a numbered slot.
    # Subscription users don't have a dollar-denominated budget projection;
    # the [budget.<provider>] section may exist as a self-imposed soft
    # ceiling, but rendering it as a hard cap would mislead. Suppress in
    # subscription/local/unknown modes — surface only in api mode.
    if pricing_mode == "api":
        for proj in report.budgets:
            _render_budget(proj)
            console.print()

    # ----- Minor findings -----
    # De-minimis-share findings stay visible (never silently dropped — the
    # honesty discipline forbids a quiet skip) but collapsed to a one-line
    # pointer instead of a full render, so a near-zero finding can't crowd
    # out the ones that actually matter.
    if minor:
        console.print(
            f"  [dim]Minor findings (< {DE_MINIMIS_SHARE * 100:.0f}% of window "
            f"tokens):[/dim]"
        )
        for name, share in minor:
            # `minor` is filtered to `item[1] is not None` above; the
            # assertion below just narrows the type for mypy.
            assert share is not None
            label = _MINOR_FINDING_LABELS.get(name, name)
            if name == "downsize":
                # The exact scenario this ranking fixes (#97): the analyzer
                # found session-shape candidates, but they hold a negligible
                # share of the window's tokens — say so plainly rather than
                # "no candidates" (that empty state is the `unranked` bucket,
                # which only holds report.downgrade is None).
                assert report.downgrade is not None
                console.print(
                    f"     [dim]• {label} — "
                    f"{report.downgrade.percent_of_sessions:.0f}% of sessions "
                    f"match, but only ~{share * 100:.1f}% of window tokens. "
                    f"Run [bold]tj optimize downsize[/bold] for detail.[/dim]"
                )
            else:
                console.print(
                    f"     [dim]• {label} — ~{share * 100:.1f}% of window "
                    f"tokens. Run [bold]tj optimize {name}[/bold] for "
                    f"detail.[/dim]"
                )
        console.print()

    rendered_any = bool(major) or bool(unranked) or bool(minor) or (
        pricing_mode == "api" and bool(report.budgets)
    )
    if not rendered_any:
        console.print(
            "[dim]No candidates flagged in this window. Either spend is small or "
            "all sessions already use a cost-effective model.[/dim]"
        )


def _sampling_ci_suffix(d: DowngradeFinding) -> str:
    """Sampling-confidence suffix for the savings line (#308).

    Renders " (n=42, 95% CI $Y–$Z)" so a 5-session projection visibly differs
    from a 500-session one. This is SAMPLING confidence — how much usage the
    estimate rests on — NOT a claim the model swap is safe (the
    MODEL_DOWNGRADE_CAVEAT governs that). The CI bounds are None when n < 2.
    """
    if d.n_sessions <= 0:
        return ""
    if d.ci_low is None or d.ci_high is None:
        # Too few sessions to bracket the projection — surface n alone so the
        # thinness is still visible, without inventing an interval.
        return f"  [dim](n={d.n_sessions} sessions; too few to bracket)[/dim]"
    return (
        f"  [dim](n={d.n_sessions} sessions, 95% CI "
        f"{format_cost(d.ci_low)}–{format_cost(d.ci_high)}/mo — sampling "
        f"confidence, not a safety claim)[/dim]"
    )


def _render_downgrade(
    d: DowngradeFinding,
    pricing_mode: str = "api",
    persona: str = "unknown",
    marker: str = "①",
) -> None:
    """
    Render the downsize finding for the given pricing mode.

    - api:          dollar-denominated savings (current behavior)
    - subscription: token-share framing — "candidate sessions are X% of your
                    cycle's tokens; routing them to {alt} frees that share
                    against your plan cap"
    - local:        token-only framing for capacity planning
    - unknown:      structural-only, no savings figures

    `persona` picks the call-to-action at the bottom (#97) — see
    `_render_downgrade_cta`.
    """
    console.print(
        f"  [bold]{marker} Model downgrade:[/bold] "
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
            f"{_sampling_ci_suffix(d)}"
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
    if d.bench_command:
        _render_downgrade_cta(d.bench_command, persona)


def _render_downgrade_cta(bench_command: str, persona: str) -> None:
    """
    Persona-aware call-to-action for the downsize finding (#97).

    A Claude Code subscription user can't pass `--original`/`--candidate` to
    pick a model per request — their real levers are exporting a routing
    config, right-sizing subagents, and `/compact`. `tokenjam-bench` (which
    runs the swap directly against a provider API key) is the right CTA for
    an SDK/API developer, and only a secondary note for anyone who also runs
    SDK agents. A mixed window shows both, labeled. Persona "sdk" and the
    defensive "unknown" fallback both get the original, unchanged CTA.
    """
    console.print()
    if persona == "claude-code":
        console.print(
            "     [bold]Candidate only — review before routing:[/bold]"
        )
        console.print(
            "       tj route export --target ccr   "
            "[dim]# or --target litellm[/dim]"
        )
        console.print(
            "       tj optimize subagent            "
            "[dim]# right-size subagent models/context[/dim]"
        )
        console.print(
            "       /compact                        "
            "[dim]# trim context mid-session[/dim]"
        )
        console.print()
        console.print(
            "     [dim]If you also run SDK agents against these models:[/dim]"
        )
        console.print("       [dim]pip install tokenjam-bench[/dim]")
        for line in bench_command.split("\n"):
            console.print(f"       [dim]{line}[/dim]")
    elif persona == "mixed":
        console.print(
            "     [bold]Candidate only — review before routing:[/bold]"
        )
        console.print("     [dim]Claude Code sessions:[/dim]")
        console.print(
            "       tj route export --target ccr   "
            "[dim]# or --target litellm[/dim]"
        )
        console.print("       tj optimize subagent")
        console.print("       /compact")
        console.print("     [dim]SDK sessions:[/dim]")
        console.print("       pip install tokenjam-bench")
        for line in bench_command.split("\n"):
            console.print(f"       {line}")
    else:  # persona in {"sdk", "unknown"} — unchanged original CTA
        console.print(
            "     [bold]Candidate only — prove it holds before switching:[/bold]"
        )
        console.print("       pip install tokenjam-bench")
        for line in bench_command.split("\n"):
            console.print(f"       {line}")


def _render_budget(p: BudgetProjection) -> None:
    headline = f"  [bold]Budget projection ({p.provider}, " \
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
            f"     • With downsize pattern: run rate drops to "
            f"[bold]{format_cost(p.downgrade_run_rate_usd)}/mo[/bold]"
        )
    if p.applies_to_services:
        console.print(
            f"     [dim]Counted services: {', '.join(p.applies_to_services)}[/dim]"
        )


def _export_reuse_templates(report, *, conn, config, agent: str | None) -> None:
    """
    Write the Reuse analyzer's Markdown skeletons to the reports directory and
    print pointers. The `tj optimize reuse --export-templates` shortcut — same
    sidecars `tj report --reuse` writes, minus the HTML/browser.
    """
    from tokenjam import __version__
    from tokenjam.cli.cmd_report import _report_dir
    from tokenjam.core.export.reuse_report import export_templates

    if conn is None:
        raise click.ClickException(
            "--export-templates needs direct database access. Stop the daemon "
            "with `tj stop` and re-run."
        )
    finding = report.findings.get("reuse")
    if finding is None or not finding.clusters:
        console.print(
            "[dim]No repeated planning detected — nothing to export. Try a "
            "longer [bold]--since[/bold].[/dim]"
        )
        return

    now = utcnow()  # tz-aware UTC (Rule 9)
    paths = export_templates(
        finding, conn=conn, config=config, out_dir=_report_dir(),
        version=__version__, generated_at_iso=now.isoformat(),
    )
    if not paths:
        console.print(
            "[yellow]No skeletons written.[/yellow] [dim]Enable "
            "[bold]capture.completions[/bold] so the planning text is "
            "available to render.[/dim]"
        )
        return
    console.print(
        f"[green]✓[/green] Wrote [bold]{len(paths)}[/bold] Reuse skeleton"
        f"{'s' if len(paths) != 1 else ''}:"
    )
    for p in paths:
        console.print(f"  [dim]{p}[/dim]")


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
        ext = "jsonc"
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
# Each renderer takes (finding, pricing_mode=str, marker=str) and prints to
# the global `console`. `marker` is the numbered slot assigned by
# `_rank_findings` (e.g. "②") — empty when the finding rendered outside the
# ranked list. Adding a new analyzer: add a renderer here, an entry in the
# dispatch table, and a label in `_MINOR_FINDING_LABELS`. cmd_optimize.
# _render_report ranks report.findings by reclaimable share and calls in here.

def _finding_header(marker: str, label: str) -> str:
    """Bold header line for a ranked finding, e.g. '② Cache efficacy:'."""
    prefix = f"{marker} " if marker else ""
    return f"  [bold]{prefix}{label}[/bold]"


def _render_cache_efficacy(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the cache finding — current caching-ratio table per
    (provider, model). When any rows are flagged, surface them prominently;
    otherwise show the full table dimmed so the user sees the underlying
    data even when no recommendation is warranted.
    """
    console.print(_finding_header(marker, "Cache efficacy:"))
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


def _render_cache_recommend(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the cache-recommend finding — Anthropic-only v1 breakpoint
    candidates. When the analyzer is disabled (capture.prompts off), surface
    the hint instead of an empty table.
    """
    console.print(_finding_header(marker, "Cache recommend:"))
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


def _render_workflow_restructure(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the script (Script) finding — clusters of sessions
    matching the same (tool_name, arg_shape) signature.
    """
    console.print(_finding_header(marker, "Workflow restructure:"))
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


def _render_prompt_bloat(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the trim (Trim) finding — LLMLingua-2 token-significance
    summary. When the analyzer is disabled (either capture off or extra
    not installed), surface the hint.
    """
    console.print(_finding_header(marker, "Prompt bloat:"))
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
        "[bold]tj report --trim[/bold][/dim]"
    )


def _render_reuse(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the reuse (Reuse) finding — clusters of sessions whose planning
    skeleton repeats. Two recoverable numbers per cluster: cache-reuse (reuse
    the existing skeleton) and script-replacement (replace every planning call
    with a deterministic template). Framed per pricing mode.
    """
    console.print(_finding_header(marker, "Reuse:"))
    if not finding.clusters:
        console.print(
            "     [dim]No repeated planning detected above threshold "
            "(≥3 sessions sharing a skeleton).[/dim]"
        )
        if finding.hint:
            console.print(f"     [dim]{_rich_escape(finding.hint)}[/dim]")
        return

    mode_note = (
        " [dim](tool-sequence only — enable capture.prompts for finer "
        "clustering)[/dim]"
        if finding.capture_mode == "tool_sequence_only"
        else ""
    )
    console.print(
        f"     • [bold]{len(finding.clusters)}[/bold] cluster"
        f"{'s' if len(finding.clusters) != 1 else ''} of repeated planning "
        f"detected{mode_note}"
    )

    for c in finding.clusters[:5]:
        sig_preview = " → ".join(c.tool_signature) if c.tool_signature else "(no tools)"
        if len(sig_preview) > 100:
            sig_preview = sig_preview[:97] + "..."
        console.print(
            f"       [bold]{c.repetitions}×[/bold] {sig_preview}"
        )
        # Recoverable framing. api → dollars; subscription/local → tokens;
        # unknown → dollars with the standard overstate qualifier.
        if pricing_mode in ("subscription", "local"):
            cache_str = f"~{format_tokens(c.cache_reuse_recoverable_tokens)} tokens"
            script_str = f"~{format_tokens(c.script_replacement_recoverable_tokens)} tokens"
        else:
            cache_str = format_cost(c.cache_reuse_recoverable_usd)
            script_str = format_cost(c.script_replacement_recoverable_usd)
        console.print(
            f"          [dim]recoverable by reusing[/dim] [bold]{cache_str}[/bold]  "
            f"[dim]· by scripting[/dim] {script_str}"
        )
        if pricing_mode == "unknown":
            console.print(
                "          [dim]figures may overstate — run "
                "[bold]tj onboard --claude-code --reconfigure[/bold] "
                "(or [bold]--codex[/bold])[/dim]"
            )

    if finding.estimate_basis:
        console.print(f"     [dim]{finding.estimate_basis}[/dim]")
    if finding.clusters:
        console.print(
            f"     [yellow]![/yellow] [italic]{finding.clusters[0].caveat}[/italic]"
        )


def _render_subagent(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the subagent right-sizing finding — how much of the window's cost ran
    inside subagents, plus the structurally-flagged candidates (over-powered
    model / over-provisioned context).
    """
    console.print(_finding_header(marker, "Subagent right-sizing:"))
    if not finding.total_subagents:
        console.print(
            "     [dim]No subagent (Task-tool) activity in this window.[/dim]"
        )
        return

    pct = finding.percent_of_cost * 100
    # Dollars only for api-billed users; subscription / local / unknown plans
    # see token-share instead (matches the report-wide suppression convention).
    if pricing_mode == "api":
        share = format_cost(finding.subagent_cost_usd)
    else:
        share = f"{format_tokens(finding.subagent_tokens)} tokens"
    console.print(
        f"     • [bold]{finding.total_subagents}[/bold] subagent"
        f"{'s' if finding.total_subagents != 1 else ''} across "
        f"[bold]{finding.sessions_with_subagents}[/bold] session"
        f"{'s' if finding.sessions_with_subagents != 1 else ''} — "
        f"[bold]{pct:.0f}%[/bold] of window cost ({share})"
    )

    flagged = list(finding.flagged) if finding.flagged else []
    if not flagged:
        console.print(
            "     [dim]No right-sizing candidates above thresholds.[/dim]"
        )
    else:
        suffix = (
            f" ([bold]{format_cost(finding.flagged_cost_usd)}[/bold] of spend)"
            if pricing_mode == "api"
            else ""
        )
        console.print(
            f"     • [yellow]{len(flagged)}[/yellow] right-sizing candidate"
            f"{'s' if len(flagged) != 1 else ''}{suffix}:"
        )
        for r in flagged[:10]:
            cost_str = (
                f"  {format_cost(r.cost_usd)}"
                if pricing_mode == "api"
                else ""
            )
            console.print(
                f"       [dim]{r.session_id[:8]}…/{r.sub_agent_id[:10]}[/dim]  "
                f"[bold]{r.model}[/bold]  "
                f"[dim]in[/dim] {format_tokens(r.input_tokens)} "
                f"[dim]cache[/dim] {format_tokens(r.cache_tokens)} "
                f"[dim]out[/dim] {format_tokens(r.output_tokens)} "
                f"[dim]· {r.tool_calls} tools[/dim]{cost_str}"
            )
            console.print(f"           [yellow]→[/yellow] {', '.join(r.flags)}")

    console.print(f"     [yellow]![/yellow] [italic]{finding.caveat}[/italic]")


# Dispatch table — analyzer registration name → renderer.
_FINDING_RENDERERS = {
    "cache":       _render_cache_efficacy,
    "cache-recommend":      _render_cache_recommend,
    "script": _render_workflow_restructure,
    "reuse":        _render_reuse,
    "trim":         _render_prompt_bloat,
    "subagent":     _render_subagent,
}
