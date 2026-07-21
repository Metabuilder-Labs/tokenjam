"""`tj optimize` — surface cost-saving candidates and budget projections."""
from __future__ import annotations

import json
from typing import Any, NoReturn

import click
from rich.markup import escape as _rich_escape

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.framing import (
    PLAN_LABEL_AND_FEE,
    Framing,
    agent_persona_mix,
    config_declared_plan,
    dominant_persona,
    dominant_plan,
    plan_tier_mix,
    pricing_mode_for,
    render_savings,
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
# levers that persona actually has — see #97. Framing / render_savings are
# the plan-tier-aware dollar-vs-token-share rendering rule itself; _render_resend
# is the one renderer in this file that feeds its recoverable figure through
# it instead of hand-branching on pricing_mode, so it can't silently drift
# from the rule cost_proposal_verbs.py already applies to the same figures.

# `placement` (batch-placement candidates) is a check that rides along inside
# `downsize`'s registry entry rather than being its own registered analyzer
# (see analyzers/batch_placement.py's module docstring and model_downgrade.py
# `run()`) — there is deliberately only one execution path for it. Without an
# alias here it had no typeable name at all: Click's Choice() only accepted
# ANALYZER_REGISTRY keys, so `tj optimize placement` was rejected before
# reaching the analyzer layer even though the finding already had a renderer
# and reached --json and the web tab (anti-pattern #24 — a surface reachable
# only as a side effect of another command isn't reachable at all). Typing
# it now runs `downsize` under the hood (never a second, standalone pass) and
# the rendered report shows the placement card without also surfacing the
# downsize card the user didn't ask for — see `_rank_findings`.
_PLACEMENT_FINDING_NAME = "placement"
_PLACEMENT_ANALYZER = "downsize"


def _resolve_analyzer_names(requested: list[str] | None) -> list[str] | None:
    """Translate CLI-facing finding names to registry analyzer names.

    `placement` isn't a registered analyzer — asking for it means "run the
    analyzer that produces it" (`downsize`). Order-preserving de-dup so
    `tj optimize placement downsize` (or the reverse) still runs `downsize`
    exactly once.
    """
    if requested is None:
        return None
    return list(dict.fromkeys(
        _PLACEMENT_ANALYZER if name == _PLACEMENT_FINDING_NAME else name
        for name in requested
    ))


@click.command("optimize")
@click.argument(
    "findings",
    nargs=-1,
    type=click.Choice(sorted({*ANALYZER_REGISTRY.keys(), _PLACEMENT_FINDING_NAME})),
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
@click.option("--validate", "validate_finding", default=None,
              type=click.Choice(["downsize"]),
              help="Re-run the finding's candidate vs the recorded baseline on a "
                   "small sample of your OWN captured calls (your API key) and "
                   "report the MEASURED token/cost delta + a quality check. "
                   "Requires [capture] prompts = true. Spends real money — you "
                   "confirm the cost estimate first.")
@click.option("--samples", "samples", type=int, default=None,
              help="Number of recorded calls to re-run under --validate "
                   "(default 5, max 20).")
@click.option("--yes", "-y", "assume_yes", is_flag=True, default=False,
              help="Skip the --validate cost-estimate confirmation prompt.")
@json_option
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
    validate_finding: str | None,
    samples: int | None,
    assume_yes: bool,
    output_json_flag: bool,
) -> None:
    """Analyze recent usage for cost-saving candidates and budget exposure."""
    output_json = resolve_output_json(ctx, output_json_flag)
    db = ctx.obj.get("db")
    config = ctx.obj.get("config")
    if db is None or config is None:
        raise click.ClickException("optimize requires a database connection.")

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    until_dt = utcnow()

    # --validate branch: turn an estimate into a MEASURED result by re-running
    # the finding's candidate vs the recorded baseline on a sample of the user's
    # own captured calls (issue #477). Self-contained early exit — it needs raw
    # span attributes + a live provider call, so it requires a direct DuckDB
    # connection (the read-only serve shim can't expose captured prompt content)
    # and never runs the normal analyzer/report path.
    if validate_finding:
        _run_validate(
            db, config,
            finding=validate_finding,
            since_dt=since_dt, until_dt=until_dt,
            agent_id=agent, samples=samples, assume_yes=assume_yes,
            output_json=output_json,
        )
        return

    # If user passed --compare last-7d / last-30d / last-week, override
    # --since so the analysis window matches the comparison period (#71
    # finding 5). Without this, `tj optimize --compare last-7d` would do
    # 30d-vs-30d (because --since defaults to 30d), while `tj cost` did
    # 7d-vs-7d — same flag, two shapes.
    if compare:
        from tokenjam.core.cost import override_since_for_compare
        since_dt = override_since_for_compare(compare, since_dt, until_dt)
        since = f"{(until_dt - since_dt).days}d"

    # The names the user actually typed (kept for rendering/JSON below) vs the
    # names the analyzer layer understands (`placement` resolved to `downsize`
    # — see `_resolve_analyzer_names`). Both API-shim and local paths below
    # must run against `analyzer_findings`: the server-side route validates
    # against the same ANALYZER_REGISTRY and would reject a raw "placement".
    requested = list(findings) if findings else None
    analyzer_findings = _resolve_analyzer_names(requested)

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
                findings=analyzer_findings,
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
            findings=analyzer_findings,
            budget_provider_filter=budget_provider,
            budget_usd_override=budget_usd,
        )

        plan_mix = plan_tier_mix(conn, since_dt, until_dt, agent)
        agent_mix = agent_persona_mix(conn, since_dt, until_dt, agent)

        # Opportunistic adoption detection: with a direct DuckDB connection in
        # hand, resolve any ripe past config exports into measured
        # adopted/ignored outcomes — but only when the daemon is actually
        # down. Holding a direct `conn` here means our own `open_db()` won a
        # lock-free open; it does NOT guarantee `tj serve` isn't concurrently
        # running (e.g. a narrow startup/shutdown window), and a daemon that
        # *is* up already runs this same detection server-side on every
        # /api/v1/recommendations read. Without an explicit check, both sides
        # could race to resolve the same ripe export and each append a
        # `downsize_adoption` record for it. Probe the daemon's HTTP API
        # (same reachability check `main.py` uses on a DB-lock failure) and
        # skip when it answers, so only one side ever runs detection for a
        # given invocation. Fail-safe — never break optimize.
        try:
            from tokenjam.core.api_backend import probe_api
            from tokenjam.core.recommendations import detect_downsize_adoption
            api_key = config.api.auth.api_key if config.api.auth.enabled else None
            daemon_up = probe_api(config.api.host, config.api.port, api_key) is not None
            if not daemon_up:
                detect_downsize_adoption(conn, config)
        except Exception:
            pass

        # Opportunistic cost-proposal refresh: until now the ONLY producer of
        # the cost-proposal store (core.optimize.cost_proposals
        # .recompute_cost_proposals) was the web Review inbox's manual
        # refresh button — a pure-CLI user who never runs `tj serve` plus the
        # web UI would never have a cost proposal computed at all, so `tj
        # relearn cost-proposals` would sit permanently empty regardless of
        # how good its renderer is. Piggyback the same recompute here so a
        # plain `tj optimize` run keeps that store warm too.
        # `recompute_cost_proposals` already never raises (it returns [] on
        # failure), so a broken window here degrades to a stale/empty
        # cost-proposals list, never a broken `tj optimize`.
        try:
            from tokenjam.core.optimize.cost_proposals import recompute_cost_proposals
            recompute_cost_proposals(db, config, agent_id=agent)
        except Exception:
            pass

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
            config=config, since=since_dt, until=until_dt,
            window_days=(until_dt - since_dt).total_seconds() / 86400.0,
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

    # Cost-proposal count (downsize/cache/trim/subagent/... fixes, each with a
    # copy-pasteable snippet) — read regardless of output mode so both the
    # JSON payload and the human footer below can point at `tj relearn
    # cost-proposals` instead of leaving findings with nowhere to go.
    from tokenjam.core.optimize import relearn_proposals
    cost_proposal_count = len(relearn_proposals.list_cost_proposals(config))

    if output_json:
        payload = report_to_dict(report)
        payload["plan_tier_mix"] = plan_mix
        payload["plan"] = dominant
        payload["pricing_mode"] = pricing_mode
        payload["agent_persona_mix"] = agent_mix
        payload["persona"] = persona
        payload["cost_proposals_available"] = cost_proposal_count
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
        requested=requested,
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

    # Findings above are diagnoses; this is where they go. Until now nothing
    # in `tj optimize`'s output pointed anywhere — the fix for e.g. a `cache`
    # or `deadweight` finding lived only in the web Review inbox's cost-proposal
    # cards (core.optimize.cost_proposals), never named from the terminal.
    if cost_proposal_count:
        console.print(
            f"[dim]{cost_proposal_count} cost fix"
            f"{'es' if cost_proposal_count != 1 else ''} available, each with "
            f"a copy-pasteable snippet: run [bold]tj relearn "
            f"cost-proposals[/bold].[/dim]"
        )


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

# Findings that must NEVER be collapsed into the "Minor findings" pointer by
# token share. `relearn` is a recurring-failure-cluster finding, not a
# token-reclamation one — its `estimated_recoverable_tokens` is a soft
# occurrence×heuristic estimate for the Lens inbox, not a real fraction of the
# window. Ranking it by that share let a heavy `--since 365d` window (huge
# denominator) push real clusters below DE_MINIMIS_SHARE and hide them behind a
# "~0.0% of window tokens" pointer — the same "nothing found" failure as the
# empty-state bug. These findings always render in full (the `unranked`
# bucket): their own detail when populated, their own empty-state when not.
_ALWAYS_FULL_FINDINGS = {"relearn"}

# Display labels for the "Minor findings" collapsed pointer list — must match
# the header text each renderer prints in its numbered form.
_MINOR_FINDING_LABELS = {
    "downsize":        "Model downgrade",
    "cache":           "Cache efficacy",
    "cache-recommend": "Cache recommend",
    "resend":          "Context resend",
    "script":          "Workflow restructure",
    "reuse":           "Reuse",
    "trim":            "Prompt bloat",
    "subagent":        "Subagent right-sizing",
    "relearn":         "Relearn",
    "verbosity":       "Verbosity",
    "deadweight":      "Deadweight",
    "placement":       "Batch placement",
    "summarize":       "Summarize",
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
    # which makes reviewers think the analyzer didn't run. Skip the section
    # entirely (empty state or full) when the user asked for a different
    # positional subset (`tj optimize cache` shouldn't mention downsize at
    # all). This also covers `tj optimize placement`: that alias resolves to
    # running the `downsize` analyzer (see `_resolve_analyzer_names`), so
    # `report.downgrade` is populated even though the user never typed
    # "downsize" — without this guard its card would leak into a report the
    # user only asked to see `placement` in.
    downsize_was_requested = (not requested) or ("downsize" in requested)
    if downsize_was_requested:
        if report.downgrade is not None:
            items.append(("downsize", _reclaimable_share(report.downgrade, window_tokens)))
        else:
            items.append(("downsize", None))

    for name, finding in (report.findings or {}).items():
        if name not in _FINDING_RENDERERS:
            continue
        # A non-token finding (e.g. relearn) is forced into the unranked bucket
        # so its clusters always render in full — a large window denominator
        # must not collapse them into the de-minimis pointer list.
        share = (
            None if name in _ALWAYS_FULL_FINDINGS
            else _reclaimable_share(finding, window_tokens)
        )
        items.append((name, share))

    items.sort(key=lambda item: (
        item[1] is None,
        -(item[1] or 0.0),
        order_index.get(item[0], len(order)),
    ))
    return items


# ---------------------------------------------------------------------------
# Human-readable renderer
# ---------------------------------------------------------------------------

def _run_validate(
    db: Any,
    config: Any,
    *,
    finding: str,
    since_dt,
    until_dt,
    agent_id: str | None,
    samples: int | None,
    assume_yes: bool,
    output_json: bool,
) -> None:
    """Empirically validate a finding on a sample of the user's own calls (#477).

    Honesty (Rule 14): every figure is framed "measured on a sample of N calls".
    We NEVER emit "certified"/"guaranteed" — that vocabulary is reserved for a
    separate paid layer. Gates: prompt capture must be on; a live provider key
    must be present; the user confirms the up-front cost estimate before we spend.
    """
    import os

    from tokenjam.core.optimize.validate import (
        ANTHROPIC_KEY_ENV,
        DEFAULT_SAMPLE_SIZE,
        MAX_SAMPLE_SIZE,
        AnthropicProviderClient,
        collect_downsize_samples,
        estimate_sample_cost,
        result_to_dict,
        run_validation,
    )

    def _fail(message: str) -> NoReturn:
        if output_json:
            click.echo(json.dumps({"error": "validate_precondition", "message": message}))
        else:
            console.print(f"[red]{_rich_escape(message)}[/red]")
        raise click.exceptions.Exit(1)

    # --validate needs raw captured attributes + a live call, so it must run
    # against a direct DuckDB connection (the serve shim can't expose prompt
    # content). Bail cleanly if the daemon holds the lock.
    conn = getattr(db, "conn", None)
    if conn is None:
        _fail(
            "tj optimize --validate needs direct database access and can't run "
            "while tj serve holds the lock. Stop the daemon (tj stop) and retry."
        )

    # Gate 1: prompt capture must be on. Capture defaults on, so this only
    # fires when it's been explicitly turned off. Actionable message + the
    # exact config hint either way.
    if not getattr(config.capture, "prompts", False):
        _fail(
            "tj optimize --validate re-runs your recorded prompts, which requires "
            "prompt capture. It is currently off in your config. Enable it "
            "under [capture]:\n\n    [capture]\n    prompts = true\n\n"
            "then let a few captured calls accumulate and try again."
        )

    # Resolve + bound the sample size.
    k = DEFAULT_SAMPLE_SIZE if samples is None else samples
    if k < 1:
        _fail("--samples must be at least 1.")
    if k > MAX_SAMPLE_SIZE:
        _fail(f"--samples is capped at {MAX_SAMPLE_SIZE} (cost-bounded).")

    sampled = collect_downsize_samples(conn, since_dt, until_dt, agent_id, k)
    if not sampled:
        _fail(
            "No captured calls match the downsize candidate shape in this window. "
            "Either there are no downsize candidates with captured prompts yet, or "
            "capture was enabled after these calls ran. Widen --since or let more "
            "captured calls accumulate."
        )

    # Gate 2: a live provider key. (v1 is Anthropic-only — the downsize candidates
    # we replay are same-family Claude models.)
    api_key = os.environ.get(ANTHROPIC_KEY_ENV)
    if not api_key:
        _fail(
            f"tj optimize --validate makes live API calls with your own key. Set "
            f"{ANTHROPIC_KEY_ENV} in your environment and try again."
        )

    # Gate 3: up-front cost estimate + confirmation before spending real money.
    est = estimate_sample_cost(sampled)
    n = len(sampled)
    if not output_json and not assume_yes:
        console.print(
            f"[bold]Validating '{finding}' on {n} of your recorded calls.[/bold]\n"
            f"[dim]This re-runs each call twice (baseline + candidate) through the "
            f"real API with your key. Estimated cost ceiling: "
            f"[bold]{format_cost(est)}[/bold].[/dim]"
        )
        if not click.confirm("Proceed and spend this?", default=False):
            console.print("[dim]Cancelled — nothing was spent.[/dim]")
            return

    client = AnthropicProviderClient(api_key)
    result = run_validation(sampled, client, finding=finding)

    if output_json:
        click.echo(json.dumps(result_to_dict(result)))
        return

    _render_validation(result)


def _render_validation(result: Any) -> None:
    """Render a ValidationResult. Honesty (Rule 14): lead with the sample size
    and the 'measured on a sample' framing; the quality line is the point."""
    tok_pct = result.tokens_delta_pct
    cost_pct = result.cost_delta_pct
    tok_pct_str = f" ({tok_pct:+.0f}%)" if tok_pct is not None else ""
    cost_pct_str = f" ({cost_pct:+.0f}%)" if cost_pct is not None else ""

    console.print(
        f"\n[bold]Measured on a sample of {result.sample_size} of your recorded "
        f"calls[/bold] [dim](finding: {result.finding})[/dim]"
    )
    console.print(
        f"  Tokens:  {format_tokens(result.baseline_tokens)} -> "
        f"{format_tokens(result.candidate_tokens)}{tok_pct_str}"
    )
    console.print(
        f"  Cost:    {format_cost(result.baseline_cost_usd)} -> "
        f"{format_cost(result.candidate_cost_usd)}{cost_pct_str}"
    )
    console.print(
        f"  Quality: preserved "
        f"[bold]{result.quality_preserved}/{result.sample_size}[/bold] "
        f"[dim](exact-match on output)[/dim]"
    )
    console.print(f"\n[dim]{_rich_escape(result.caveat)}[/dim]")


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
        console.print(f"  [yellow]![/yellow] {_rich_escape(note)}")
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
        elif name == "resend":
            # Persona-branched fix (compaction vs cache_control), same reason
            # downsize gets `persona` above — see `_render_resend_fix`. Called
            # directly rather than through `_FINDING_RENDERERS[name]`, same as
            # `_render_downgrade` above: that dict's value type is inferred
            # from every renderer sharing it, so it only advertises the
            # (finding, pricing_mode, marker) signature common to all of
            # them — a call through it with `persona=` is a real mypy error
            # (call-arg), not a false positive, since nothing in the dict's
            # type says entry "resend" specifically accepts that kwarg.
            _render_resend(
                report.findings[name], pricing_mode=pricing_mode, marker=marker,
                persona=persona,
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
    config=None,
    since=None,
    until=None,
    window_days: float = 0.0,
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

    # Record the export in the recommendation-outcome ledger, stashing the
    # downsize baseline so post-hoc adoption detection can later measure whether
    # the recommended premium models' usage actually dropped. Fail-safe.
    if config is not None and since is not None and until is not None:
        try:
            from tokenjam.core.recommendations import record_config_export
            provider = None
            suggestions = getattr(downgrade, "suggestions", None) or {}
            if suggestions:
                from tokenjam.core.optimize.analyzers.model_downgrade import (
                    DOWNGRADE_CANDIDATES,
                )
                for prov, mapping in DOWNGRADE_CANDIDATES.items():
                    if any(m in mapping for m in suggestions):
                        provider = prov
                        break
            record_config_export(
                config, target=target, export_path=str(out_path),
                downgrade=downgrade, pricing_mode=pricing_mode, provider=provider,
                since=since, until=until, window_days=window_days,
            )
        except Exception:
            pass

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
    (provider, model), followed by the root-caused per-agent candidates
    behind it (A1 uncached / A2 thrash / A3 lookback miss, see
    `_render_cache_root_causes`). When any ratio rows are flagged, surface
    them prominently; otherwise show the full table dimmed so the user sees
    the underlying data even when no recommendation is warranted. The ratio
    table only measures the (provider, model) efficacy gap; the root-cause
    section is what actually carries a ready cache_control_snippet.
    """
    console.print(_finding_header(marker, "Cache efficacy:"))
    has_root_cause = bool(
        finding.uncached_agents or finding.thrash_agents or finding.lookback_miss_agents
    )
    if not finding.rows and not has_root_cause:
        console.print(
            "     [dim]No LLM spans with provider/model in this window.[/dim]"
        )
        return

    if finding.rows:
        flagged = list(finding.flagged) if finding.flagged else []
        if flagged:
            # Effective thresholds, not the historical hardcoded 30%/100K: a user
            # who has lowered [optimize] cache_efficacy_threshold / min_cache_input_tokens
            # must see the bar they actually configured, not the old default.
            console.print(
                f"     • [bold]{len(flagged)}[/bold] (provider, model) "
                f"row{'s' if len(flagged) != 1 else ''} flagged below the "
                f"{finding.efficacy_threshold * 100:.0f}% efficacy threshold at "
                f"≥{format_tokens(finding.min_input_tokens)} input tokens:"
            )
            for r in flagged:
                console.print(
                    f"       [bold]{r.provider}/{r.model}[/bold]  "
                    f"{r.efficacy*100:.0f}% efficacy  "
                    f"({format_tokens(r.input_tokens)} input / "
                    f"{format_tokens(r.cache_tokens)} cache)"
                )
            # Diagnosis and remedy live under two different finding keys — this
            # renderer only measures the ratio; the actual cache_control breakpoint
            # candidates come from `cache-recommend`. Point there explicitly so a
            # user reading `cache` alone doesn't miss the fix.
            console.print(
                "     [yellow]→[/yellow] Run [bold]tj optimize cache-recommend[/bold] "
                "for concrete cache_control breakpoint candidates."
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

    console.print()
    _render_cache_root_causes(finding, pricing_mode=pricing_mode)


def _render_cache_root_causes(finding, *, pricing_mode: str) -> None:
    """
    Render the three root-caused per-agent candidates behind the ratio table
    above (see `_classify_a1` / `_classify_a2` / `_classify_a3` in
    analyzers/cache_efficacy.py):

    - A1 uncached agents: caching never attempted at all (zero cache reads
      AND zero cache writes on every call) despite a prefix large enough to
      matter.
    - A2 cache thrash: caching attempted regularly, but more was spent
      writing the prefix than was ever recovered reading it back. The card
      branches on cause — "ttl" (calls land more than five minutes apart, so
      the default 5-minute write keeps expiring before reuse) versus
      "instability" (calls land close together, so a TTL expiry doesn't
      explain it — the prefix itself is likely changing between calls).
    - A3 lookback miss: recurring cache misses that directly follow a long,
      tool-heavy turn — the shape of Anthropic's 20-block breakpoint
      lookback limit. Weakest-confidence of the three; an agent only lands
      here when A1/A2 don't already explain its waste.

    Classification is mutually exclusive per agent (uncached beats thrash
    beats lookback — see `_compute_root_cause_candidates`). Unlike the ratio
    table above, every candidate here carries a ready `cache_control_snippet`
    — the same data `cost_proposals.py` turns into the A1/A2/A3 cost
    proposals.
    """
    uncached = finding.uncached_agents
    thrash = finding.thrash_agents
    lookback = finding.lookback_miss_agents
    if not uncached and not thrash and not lookback:
        console.print(
            f"     [dim]No agent group cleared the "
            f"≥{finding.min_calls_for_root_cause} calls threshold for "
            f"root-cause classification. Lower "
            f"\\[optimize] min_calls_for_root_cause in tj.toml to classify "
            f"smaller agent groups.[/dim]"
        )
        return

    console.print("     [dim]Root-caused agent candidates:[/dim]")

    if uncached:
        n = len(uncached)
        console.print(
            f"     • [bold]{n}[/bold] agent{'s' if n != 1 else ''} never "
            f"attempt caching [dim](zero cache reads, zero cache writes, "
            f"prefix large enough to matter)[/dim]:"
        )
        for c in uncached[:5]:
            console.print(
                f"       [bold]{c.agent_id}[/bold]  {c.provider}/{c.model}  "
                f"{c.calls} call{'s' if c.calls != 1 else ''} / "
                f"{c.sessions} session{'s' if c.sessions != 1 else ''}  "
                f"[dim]~{format_tokens(c.assumed_prefix_tokens)} assumed prefix[/dim]"
            )
            if pricing_mode == "api":
                if c.estimated_recoverable_usd is not None:
                    console.print(
                        f"           [dim]≈[/dim] "
                        f"[green]{format_cost(c.estimated_recoverable_usd)}[/green] "
                        f"estimated recoverable over this window"
                    )
                else:
                    console.print(
                        "           [dim]no dollar figure: no priced rate "
                        f"observed for {c.model or 'this model'}[/dim]"
                    )
            console.print("           [dim]cache_control:[/dim]")
            console.print(
                c.cache_control_snippet, markup=False, highlight=False, soft_wrap=True,
            )
        if n > 5:
            console.print(f"       [dim]… and {n - 5} more.[/dim]")

    if thrash:
        n = len(thrash)
        console.print(
            f"     • [bold]{n}[/bold] agent{'s' if n != 1 else ''} "
            f"thrashing the cache [dim](writing more than is ever read "
            f"back)[/dim]:"
        )
        for c in thrash[:5]:
            console.print(
                f"       [bold]{c.agent_id}[/bold]  {c.provider}/{c.model}  "
                f"read:write [bold]{c.read_write_ratio:.2f}[/bold]  "
                f"[dim]({format_tokens(c.cache_read_tokens)} read / "
                f"{format_tokens(c.cache_write_tokens)} write, {c.calls} "
                f"calls, gap p50 {c.inter_call_gap_p50_minutes:.1f} min)[/dim]"
            )
            if c.cause == "ttl":
                if c.ttl_worth_it:
                    console.print(
                        "           [dim]cause: calls land more than 5 min "
                        "apart, so the default 5-minute cache write keeps "
                        "expiring — the 1-hour TTL is estimated to pay off "
                        "at this cadence[/dim]"
                    )
                else:
                    console.print(
                        "           [dim]cause: calls land more than 5 min "
                        "apart, but the 1-hour TTL's write premium doesn't "
                        "clear at this cadence — caching not worth it "
                        "here[/dim]"
                    )
            else:
                console.print(
                    "           [dim]cause: calls land close enough "
                    "together that TTL expiry doesn't explain it — the "
                    "prefix itself is likely changing between calls[/dim]"
                )
            if pricing_mode == "api":
                if c.estimated_recoverable_usd is not None:
                    console.print(
                        f"           [dim]≈[/dim] "
                        f"[green]{format_cost(c.estimated_recoverable_usd)}[/green] "
                        f"wasted writing this prefix over this window"
                    )
                else:
                    console.print(
                        "           [dim]no dollar figure: the recommended "
                        "fix would not recover it[/dim]"
                    )
            console.print("           [dim]cache_control:[/dim]")
            console.print(
                c.cache_control_snippet, markup=False, highlight=False, soft_wrap=True,
            )
        if n > 5:
            console.print(f"       [dim]… and {n - 5} more.[/dim]")

    if lookback:
        n = len(lookback)
        console.print(
            f"     • [bold]{n}[/bold] agent{'s' if n != 1 else ''} hitting "
            f"the 20-block lookback limit [dim](long tool-heavy turns "
            f"pushing the prior breakpoint out of range)[/dim]:"
        )
        for c in lookback[:5]:
            console.print(
                f"       [bold]{c.agent_id}[/bold]  {c.provider}/{c.model}  "
                f"{c.miss_count} miss{'es' if c.miss_count != 1 else ''}  "
                f"[dim](avg {c.avg_prior_turn_blocks:.0f} blocks in the "
                f"prior turn)[/dim]"
            )
            if pricing_mode == "api":
                if c.estimated_recoverable_usd is not None:
                    console.print(
                        f"           [dim]≈[/dim] "
                        f"[green]{format_cost(c.estimated_recoverable_usd)}[/green] "
                        f"estimated recoverable over this window"
                    )
                else:
                    console.print(
                        "           [dim]no dollar figure: no priced rate "
                        f"observed for {c.model or 'this model'}[/dim]"
                    )
            console.print("           [dim]cache_control:[/dim]")
            console.print(
                c.cache_control_snippet, markup=False, highlight=False, soft_wrap=True,
            )
        if n > 5:
            console.print(f"       [dim]… and {n - 5} more.[/dim]")

    if pricing_mode != "api":
        console.print(
            "     [dim]This plan doesn't bill per token, so no dollar "
            "figures are shown for these candidates; the counts above still "
            "show the caching opportunity.[/dim]"
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
        msg = (
            f"     [dim]No stable prefixes shared across "
            f"≥{finding.min_prefix_occurrences} Anthropic calls"
        )
        if finding.skipped_provider_count:
            msg += (
                f". Skipped {finding.skipped_provider_count} non-Anthropic "
                f"span(s) — multi-provider support is a future feature."
            )
        msg += (
            ". Lower [bold]\\[optimize] min_prefix_occurrences[/bold] in "
            "tj.toml to see prefixes shared across fewer calls.[/dim]"
        )
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
        # Subscription/local plans don't pay per token — no dollar lever to
        # show. On api, a candidate can still have no priced rate for its
        # model, in which case we say why rather than print a $0.00.
        if pricing_mode == "api":
            if c.estimated_recoverable_usd is not None:
                console.print(
                    f"           [dim]≈[/dim] [green]{format_cost(c.estimated_recoverable_usd)}[/green] "
                    f"estimated over this window [dim](model {c.model})[/dim]"
                )
            else:
                console.print(
                    f"           [dim]no dollar figure: no priced rate observed "
                    f"for {c.model or 'this model'}[/dim]"
                )
        console.print(f"           [dim italic]{sample}[/dim italic]")

    if pricing_mode == "api" and finding.estimated_recoverable_usd is not None:
        console.print(
            f"     • [green]~{format_cost(finding.estimated_recoverable_usd)}[/green] "
            f"estimated recoverable across these candidates [dim](reads after "
            f"the first occurrence, minus one cache write per prefix)[/dim]"
        )
    elif pricing_mode != "api":
        console.print(
            "     [dim]This plan doesn't bill per token, so no dollar figure "
            "is shown; the token counts above still show the caching "
            "opportunity.[/dim]"
        )
    else:
        console.print(
            "     [dim]No dollar figure: no priced Anthropic model rate was "
            "observed for these candidates.[/dim]"
        )

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
                f"no clusters above threshold (≥{finding.min_cluster_instances} "
                f"identical signatures, zero branching). Lower "
                f"\\[optimize] min_cluster_instances in tj.toml to see "
                f"smaller clusters.[/dim]"
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
            f"skipped {finding.prompts_skipped}. No region scored below the "
            f"{finding.significance_threshold:.2f} significance threshold "
            f"ran long enough to flag. Raise \\[optimize] "
            f"trim_significance_threshold in tj.toml to flag more "
            f"borderline text as bloat.[/dim]"
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
        console.print(f"           [dim italic]{_rich_escape(sample)}[/dim italic]")
        # Provenance (read-only, see prompt_bloat.py's module docstring): most
        # prompts end unattributed — that's the conservative, expected outcome,
        # not a gap — so this block only prints when a catalog file actually
        # cleared the verbatim-containment bar. `trim` never edits the file; the
        # pointer below is a navigation hint into `summarize`, which owns editing.
        #
        # This gate is also the deliberate persona split, not an accident of
        # the provenance check: source_path is only ever set when the prompt
        # verbatim-contains a catalog file (CLAUDE.md, AGENTS.md, ...), which
        # by definition means a harness-shaped workspace `summarize` can act
        # on. A pure-SDK caller (no catalog file in play) never gets a
        # source_path and so never sees this pointer — the flagged-text
        # section below is that caller's whole, complete answer: there's
        # nothing degraded about it, it's the only thing to point at since
        # they construct the prompt themselves rather than editing a file.
        if p.source_path:
            console.print(
                f"           [dim]Attributed to [bold]{_rich_escape(p.source_path)}[/bold] "
                f"({_rich_escape(p.source_basis)})[/dim]"
            )
            console.print(
                f"           [dim]Review it: [bold]tj summarize list "
                f"{_rich_escape(p.source_path)}[/bold][/dim]"
            )
        # The flagged text itself, not just the bloat percentage — a user
        # can't act on "38% low-signal" alone; they need to see what to cut.
        regions = p.regions[:3]
        if regions:
            console.print("           [dim]Flagged text:[/dim]")
            for r in regions:
                text = _rich_escape(r.sample_chars.replace("\n", " ").strip())
                console.print(f"             [dim]·[/dim] [italic]{text}…[/italic] "
                              f"[dim]({r.char_length} chars)[/dim]")
            if len(p.regions) > 3:
                console.print(
                    f"             [dim]… and {len(p.regions) - 3} more region(s).[/dim]"
                )
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
            f"     [dim]No repeated planning detected above threshold "
            f"(≥{finding.min_repetitions} sessions sharing a skeleton). "
            f"Lower \\[optimize] min_reuse_repetitions in tj.toml to see "
            f"smaller clusters.[/dim]"
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
            f"     [dim]No right-sizing candidates above thresholds "
            f"(structural shape checks, plus a "
            f"{format_cost(finding.min_flag_cost_usd)} minimum flagged spend). "
            f"Lower \\[optimize] min_flag_cost_usd in tj.toml to flag "
            f"cheaper subagents.[/dim]"
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

    # Quantified estimate (#101): the over_powered model-swap delta — what earns
    # this finding its ranked slot. Dollars for api-billed users; the token quota
    # otherwise (same category discipline as the peer analyzers). Honest
    # "estimated recoverable" framing only; the caveat below still governs.
    if finding.estimated_recoverable_tokens is not None:
        if pricing_mode == "api" and finding.estimated_recoverable_usd is not None:
            recov = format_cost(finding.estimated_recoverable_usd)
        else:
            recov = f"{format_tokens(finding.estimated_recoverable_tokens)} tokens"
        console.print(
            f"     • [green]~{recov}[/green] estimated recoverable "
            f"[dim](over_powered subagents at their cheaper same-family model)[/dim]"
        )

    console.print(f"     [yellow]![/yellow] [italic]{finding.caveat}[/italic]")


def _render_relearn(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the relearn finding — recurring failure clusters the self-improve
    loop tracks (blockers an agent silently re-hits across sessions). Was
    missing from _FINDING_RENDERERS entirely: the text view fell
    through to the generic "No candidates flagged" empty state even when
    --json carried dozens of clusters, because _rank_findings drops any
    finding name not present in this dispatch table.
    """
    console.print(_finding_header(marker, "Relearn:"))
    if not finding.clusters:
        console.print(
            f"     [dim]Scanned {finding.sessions_scanned} session"
            f"{'s' if finding.sessions_scanned != 1 else ''}; "
            f"no recurring failure clusters above threshold "
            f"(≥{finding.min_sessions} sessions sharing a signature). Lower "
            f"\\[optimize] min_recurring_sessions in tj.toml to see smaller "
            f"clusters.[/dim]"
        )
        return

    console.print(
        f"     • [bold]{len(finding.clusters)}[/bold] relearn "
        f"cluster{'s' if len(finding.clusters) != 1 else ''} found — "
        f"recurring blockers this agent silently re-hits"
    )
    for c in finding.clusters[:10]:
        console.print(
            f"       [bold]{c.signature}[/bold]  "
            f"{c.occurrences} occurrence{'s' if c.occurrences != 1 else ''} / "
            f"{c.sessions} session{'s' if c.sessions != 1 else ''}  "
            f"[dim](rung {c.rung})[/dim]"
        )
    if len(finding.clusters) > 10:
        console.print(f"       [dim]… and {len(finding.clusters) - 10} more.[/dim]")
    console.print(
        "     [dim]Review + apply fixes in the Lens Review inbox, or see "
        "full detail with [bold]tj optimize relearn --json[/bold].[/dim]"
    )
    if finding.caveat:
        console.print(f"     [yellow]![/yellow] [italic]{finding.caveat}[/italic]")


def _render_verbosity(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the verbosity finding — sessions whose OUTPUT tokens run high vs
    the per-task-shape median (the like-for-like baseline). The least-grounded
    analyzer: candidate framing only, recoverable shown as a soft estimate.
    """
    console.print(_finding_header(marker, "Verbosity:"))
    if not finding.candidates:
        if finding.sessions_examined == 0:
            console.print("     [dim]No LLM spans in this window.[/dim]")
        else:
            console.print(
                f"     [dim]Examined {finding.sessions_examined} session"
                f"{'s' if finding.sessions_examined != 1 else ''} across "
                f"{finding.cohorts_examined} task-shape cohort"
                f"{'s' if finding.cohorts_examined != 1 else ''} (≥"
                f"{finding.min_cohort_sessions} sessions each); no session's "
                f"output ran high enough vs its cohort median to flag. Lower "
                f"\\[optimize] min_cohort_sessions in tj.toml to consider "
                f"smaller cohorts.[/dim]"
            )
        return

    shown = len(finding.candidates)
    total = finding.total_candidates or shown
    more = f" [dim](showing top {shown} of {total})[/dim]" if total > shown else ""
    console.print(
        f"     • [bold]{total}[/bold] high-verbosity "
        f"candidate{'s' if total != 1 else ''} "
        f"[dim](output well above the per-task-shape median)[/dim]{more}"
    )
    for c in finding.candidates:
        # Recoverable framing: dollars only for api-billed users; otherwise the
        # over-baseline token figure (the same category discipline as peers).
        if pricing_mode == "api" and c.recoverable_usd:
            recov = f"~{format_cost(c.recoverable_usd)} at output rates"
        else:
            recov = f"~{format_tokens(c.over_baseline_tokens)} output tokens"
        ratio = (
            f", {c.output_input_ratio}× input"
            if c.output_input_ratio is not None else ""
        )
        console.print(
            f"       [dim]{c.session_id}[/dim]  "
            f"[bold]{format_tokens(c.output_tokens)}[/bold] out "
            f"[dim](vs {format_tokens(c.baseline_output_tokens)} median, "
            f"{c.over_baseline_multiple}×{ratio})[/dim] → {recov} above baseline"
        )
    if finding.suggested_max_tokens:
        console.print(
            f"     [dim]Remedy to review (not applied): add a terse "
            f"system-prompt line and/or a max_tokens cap near "
            f"~{format_tokens(finding.suggested_max_tokens)}, then measure with "
            f"[bold]tj optimize --validate[/bold].[/dim]"
        )
    if finding.caveat:
        console.print(f"     [yellow]![/yellow] [italic]{finding.caveat}[/italic]")


def _render_summarize(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the summarize finding — catalog prompt files (CLAUDE.md / AGENTS.md /
    globals) whose prose could be summarized. Registered and runs like every
    other analyzer, but had no entry in `_FINDING_RENDERERS`, so it was
    silently dropped from plain-text `tj optimize` output and only reachable
    via `--json`.

    Tokens-only by design (see core/optimize/analyzers/summarize.py):
    `estimated_recoverable_usd` is intentionally None — there's no per-file
    call telemetry to amortize a dollar figure over — so this renderer never
    fabricates one, only the per-call token reduction.
    """
    console.print(_finding_header(marker, "Summarize:"))
    if not finding.candidates:
        console.print(
            "     [dim]No catalog prompt files (CLAUDE.md / AGENTS.md / "
            "globals) with summarizable prose found.[/dim]"
        )
        return

    tokens = finding.estimated_recoverable_tokens or 0
    console.print(
        f"     • [bold]{finding.files}[/bold] file{'s' if finding.files != 1 else ''} "
        f"summarizable, ~[bold]{format_tokens(tokens)}[/bold] per call "
        f"[dim](aggregate {finding.reduction_pct}% prose reduction)[/dim]"
    )
    for c in finding.candidates[:5]:
        console.print(
            f"       [dim]{c.path}[/dim]  [dim]({c.scope})[/dim]  "
            f"~{format_tokens(c.est_tokens_saved)} saved  "
            f"[dim]{c.reduction_pct}% reduction[/dim]"
        )
    if len(finding.candidates) > 5:
        console.print(f"       [dim]… and {len(finding.candidates) - 5} more.[/dim]")
    console.print(
        "     [yellow]→[/yellow] Run [bold]tj summarize list[/bold] to review, "
        "then [bold]tj summarize prep <path>[/bold] to generate a rewrite."
    )
    if finding.caveat:
        console.print(f"     [yellow]![/yellow] [italic]{finding.caveat}[/italic]")


def _render_deadweight(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the deadweight finding — configured MCP servers whose schemas are
    injected into every session and never invoked. Same class of bug as the
    relearn renderer above: absent from _FINDING_RENDERERS, the text view fell
    through to the generic empty state even when --json carried dead servers,
    because _rank_findings drops any finding name missing from that table.
    """
    console.print(_finding_header(marker, "Deadweight:"))
    if not finding.sessions_scanned:
        console.print(
            "     [dim]No Claude Code sessions in this window.[/dim]"
        )
        return
    if not finding.configured_servers:
        console.print(
            f"     [dim]Scanned {finding.sessions_scanned} session"
            f"{'s' if finding.sessions_scanned != 1 else ''}; no MCP server is "
            f"configured, so nothing is being injected.[/dim]"
        )
        return

    if not finding.dead_servers:
        # _rich_escape: the analyzer's own note names the config key in
        # bracket form ("Lower [optimize] min_sessions_deadweight..."), which
        # Rich would otherwise parse as an unknown style tag and silently
        # drop from the printed line.
        for note in finding.notes:
            console.print(f"     [dim]{_rich_escape(note)}[/dim]")
        if not finding.notes:
            console.print(
                f"     [dim]All {finding.configured_servers} configured MCP "
                f"server{'s' if finding.configured_servers != 1 else ''} were "
                f"invoked at least once in this window.[/dim]"
            )
    else:
        n = len(finding.dead_servers)
        console.print(
            f"     • [bold]{n}[/bold] dead MCP server{'s' if n != 1 else ''} of "
            f"[bold]{finding.configured_servers}[/bold] configured "
            f"[dim](schemas injected every session, never called)[/dim]"
        )
        for s in finding.dead_servers:
            console.print(
                f"       [bold]{s.name}[/bold] [dim]({s.scope} · {s.source})[/dim]  "
                f"present in {s.sessions_present} session"
                f"{'s' if s.sessions_present != 1 else ''}, "
                f"[yellow]{s.invocations}[/yellow] invocations"
            )
            # Dollars only when a priced model was actually observed for this
            # server. None means no rate was available, and printing $0.00
            # there would read as "this costs nothing".
            if pricing_mode == "api" and s.estimated_tax_usd_90d is not None:
                tax = (
                    f"~{format_tokens(s.estimated_tax_tokens_90d)} tokens / "
                    f"{format_cost(s.estimated_tax_usd_90d)} over 90 days "
                    f"[dim](estimated, priced at {s.priced_model})[/dim]"
                )
            else:
                tax = (
                    f"~{format_tokens(s.estimated_tax_tokens_90d)} tokens over "
                    f"90 days [dim](estimated; no priced model observed for "
                    f"this server, so no dollar figure)[/dim]"
                )
            console.print(f"          [dim]tax[/dim] {tax}")
            if s.tax_construction:
                console.print(f"          [dim]{s.tax_construction}[/dim]")
            console.print(f"          [yellow]→[/yellow] {s.fix}")

    # C2 context tax: every always-injected content source, dead or alive. Kept
    # to the top rows so it stays a pointer rather than a second report.
    if finding.tax_table:
        console.print(
            "     [dim]Always-injected context per session (estimated):[/dim]"
        )
        for row in finding.tax_table[:5]:
            console.print(
                f"       [dim]{row.source}[/dim]  "
                f"~{format_tokens(row.avg_tokens_per_session)}/session "
                f"[dim]× {row.sessions} session"
                f"{'s' if row.sessions != 1 else ''} = "
                f"{format_tokens(row.total_tokens_window)} "
                f"({row.tag})[/dim]"
            )
        if len(finding.tax_table) > 5:
            console.print(
                f"       [dim]… and {len(finding.tax_table) - 5} more source(s). "
                f"Full detail with [bold]tj optimize deadweight --json[/bold].[/dim]"
            )

    if finding.estimate_basis:
        console.print(f"     [dim]{finding.estimate_basis}[/dim]")
    if finding.caveat:
        console.print(f"     [yellow]![/yellow] [italic]{finding.caveat}[/italic]")


def _cadence_phrase(seconds: float) -> str:
    """A median inter-start gap as something a person reads at a glance."""
    if seconds >= 86400:
        return f"~{seconds / 86400:.1f}d"
    if seconds >= 3600:
        return f"~{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"~{seconds / 60:.0f}m"
    return f"~{seconds:.0f}s"


def _render_placement(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the batch-placement finding — unattended, cadence-regular workloads
    whose shape allows a Batch API discussion. Third finding of this shape to
    ship without a text-view renderer (relearn, then deadweight): absent from
    _FINDING_RENDERERS it reaches the web tab and --json but falls through to
    the generic empty state in the CLI.

    The estimate here is a PRICE difference on the same tokens, not tokens
    freed, so the token figure is labelled as the size of the affected
    workload and never as "recoverable". The Batch API's flat discount is an
    api-billed lever, so subscription and local plans are told that plainly
    rather than shown a dollar figure that cannot apply to them.
    """
    console.print(_finding_header(marker, "Batch placement:"))
    if not finding.candidates:
        console.print(
            f"     [dim]No unattended, cadence-regular workloads in this "
            f"window (≥{finding.min_sessions_for_cadence} sessions on a "
            f"regular cadence, ≥{format_cost(finding.min_group_cost_usd)} "
            f"window spend). Lower \\[optimize] min_sessions_for_cadence / "
            f"min_group_cost_usd in tj.toml to consider smaller "
            f"workloads.[/dim]"
        )
        return

    n = len(finding.candidates)
    console.print(
        f"     • [bold]{n}[/bold] workload{'s' if n != 1 else ''} "
        f"{'fit' if n != 1 else 'fits'} the batch "
        f"shape [dim](regular cadence, no human turn after the first model "
        f"call)[/dim]: [bold]{finding.percent_of_window_cost:.1f}%[/bold] of "
        f"window cost"
    )
    for c in finding.candidates[:10]:
        console.print(
            f"       [bold]{c.agent_id}[/bold]  "
            f"{c.sessions} session{'s' if c.sessions != 1 else ''} every "
            f"{_cadence_phrase(c.median_gap_seconds)} "
            f"[dim](cadence spread {c.gap_cv:.2f})[/dim]  "
            f"{format_tokens(c.tokens)} tokens"
        )
        if pricing_mode == "api":
            console.print(
                f"          [dim]spend[/dim] {format_cost(c.cost_usd)} "
                f"[dim]→ at the batch rate[/dim] "
                f"[green]{format_cost(c.cost_usd - c.estimated_batch_saving_usd)}[/green] "
                f"[dim](a difference of "
                f"{format_cost(c.estimated_batch_saving_usd)}, estimated)[/dim]"
            )
    if len(finding.candidates) > 10:
        console.print(
            f"       [dim]… and {len(finding.candidates) - 10} more.[/dim]"
        )

    if pricing_mode == "api" and finding.estimated_recoverable_usd is not None:
        console.print(
            f"     • [green]~{format_cost(finding.estimated_recoverable_usd)}[/green] "
            f"estimated price difference over this window "
            f"[dim](the same work, billed at the Batch API's flat rate)[/dim]"
        )
    else:
        console.print(
            "     [dim]The Batch API's discount is an api-billed price lever, "
            "so no dollar figure is shown for this plan. The workload sizes "
            "above still say how much work fits the shape.[/dim]"
        )

    if finding.estimate_basis:
        console.print(f"     [dim]{finding.estimate_basis}[/dim]")
    if finding.friction:
        console.print(f"     [yellow]![/yellow] [italic]{finding.friction}[/italic]")


def _render_resend(
    finding, *, pricing_mode: str = "api", marker: str = "", persona: str = "unknown",
) -> None:
    """
    Render the resend finding — structural context re-send: how much of each
    turn's prompt was already sent, unchanged, in an earlier turn. Registered
    and running since it landed (see analyzers/context_resend.py), but with no
    renderer at all: `tj optimize` showed nothing for the product's headline
    waste category (this repo's own HAL-corpus benchmark measured 93.8% of
    prompt tokens re-sent) even though the finding already reached --json and
    the web tab. Same class of gap as `relearn` and `deadweight` before their
    renderers landed.

    `repeat_share` is a measured token-share, not a savings claim (Rule 14 /
    anti-pattern #22): it is shown even when `estimated_recoverable_*` is
    suppressed below, and `finding.caveat` renders verbatim every time this
    prints, never paraphrased.
    """
    console.print(_finding_header(marker, "Context resend:"))
    if finding.repeat_share is None:
        # Below the data threshold (too few sessions/turns) — empty-state
        # discipline: never a bare "nothing found", always the reason.
        for note in finding.notes:
            console.print(f"     [dim]{_rich_escape(note)}[/dim]")
        if not finding.notes:
            console.print("     [dim]No LLM turns in this window.[/dim]")
        return

    console.print(
        f"     • [bold]{finding.repeat_share * 100:.1f}%[/bold] of prompt "
        f"tokens across [bold]{finding.sessions_examined}[/bold] session"
        f"{'s' if finding.sessions_examined != 1 else ''} "
        f"({finding.turns_examined} turns, "
        f"{finding.multi_turn_sessions} multi-turn) were already sent in an "
        f"earlier turn [dim](conservative lower bound)[/dim]"
    )
    if finding.repeat_share_median is not None and finding.repeat_share_p90 is not None:
        console.print(
            f"       [dim]per-session median[/dim] "
            f"{finding.repeat_share_median * 100:.1f}%  [dim]p90[/dim] "
            f"{finding.repeat_share_p90 * 100:.1f}%"
        )
    console.print(
        f"       [dim]{format_tokens(finding.repeat_tokens)} repeat tokens "
        f"of {format_tokens(finding.prompt_tokens_total)} total prompt "
        f"tokens[/dim]"
    )

    if finding.examples:
        console.print()
        console.print("     [dim]Heaviest sessions:[/dim]")
        for ex in finding.examples[:5]:
            console.print(
                f"       [dim]{ex.session_id[:12]}[/dim]  {ex.turns} turns  "
                f"[bold]{ex.repeat_share * 100:.0f}%[/bold] repeat  "
                f"{format_tokens(ex.repeat_tokens)} tokens  "
                f"[dim]({ex.provider}/{ex.model})[/dim]"
            )

    # The "why": reuse `tj context`'s own recurring-inclusion rendering (the
    # tag-per-kind lookup), rather than re-deriving a second copy of that
    # translation table that could drift from the established card.
    if finding.recurring_examples:
        from tokenjam.cli.cmd_context import _INCLUSION_LABELS
        console.print()
        console.print("     [dim]Why (recurring inclusions):[/dim]")
        for r in finding.recurring_examples[:5]:
            tag = _INCLUSION_LABELS.get(r.inclusion_type, "repeat")
            # `_rich_escape` around the bracketed tag itself: "[file]" reads
            # as Rich markup (an unknown style tag) if left unescaped inside
            # a string Rich otherwise parses, which silently ate the tag.
            console.print(
                f"       [cyan]{_rich_escape(f'[{tag}]')}[/cyan] "
                f"[bold]{_rich_escape(r.target)}[/bold]  "
                f"×{r.occurrences} ({r.sessions} sessions)"
            )
            console.print(f"          [green]→[/green] {_rich_escape(r.fix)}")
    else:
        for note in finding.notes:
            console.print(f"     [dim]{_rich_escape(note)}[/dim]")

    # Recoverable figure: fed through framing.render_savings rather than a
    # hand-rolled pricing_mode branch, so it can't quietly disagree with the
    # same rule cost_proposal_verbs.py applies to every other recoverable
    # figure. Framed against this finding's OWN denominator (prompt_tokens_total,
    # not the window's four-token-type total) since that's the basis
    # repeat_share itself is measured against.
    framing = Framing(pricing_mode=pricing_mode, window_total_tokens=finding.prompt_tokens_total)
    recoverable = render_savings(
        finding.estimated_recoverable_usd, finding.estimated_recoverable_tokens, framing,
    )
    console.print()
    if recoverable != "—":
        console.print(f"     • [green]~{recoverable}[/green] estimated recoverable")
    elif pricing_mode == "api":
        console.print(
            "     [dim]No dollar figure: no priced example session for the "
            "cache_control lever.[/dim]"
        )
    if finding.estimate_basis:
        console.print(f"     [dim]{finding.estimate_basis}[/dim]")

    console.print(f"     [yellow]![/yellow] [italic]{finding.caveat}[/italic]")
    _render_resend_fix(finding, persona)


def _render_resend_fix(finding, persona: str) -> None:
    """
    Persona-aware fix for the resend finding, mirroring `_render_downgrade_cta`
    (#97): `fix_compaction` is the agent-harness lever (a Claude Code
    subscriber's actual lever — they can't set `cache_control` on someone
    else's harness); `fix_cache_control` is the SDK-adoption lever, a
    ready-to-paste snippet that is empty whenever no priced example produced
    one. Unlike the downgrade CTA's `bench_command` (always present),
    `fix_cache_control` can be empty, so the safe default for a "mixed" or
    "unknown" window is compaction first (always non-empty, and its "start a
    fresh session" clause is meaningful for any agent loop, not just Claude
    Code) with the cache_control snippet offered second when one exists.
    """
    console.print()
    if persona == "sdk" and finding.fix_cache_control:
        console.print("     [bold]Fix (cache_control adoption):[/bold]")
        console.print(
            finding.fix_cache_control, markup=False, highlight=False, soft_wrap=True,
        )
    elif persona == "mixed":
        console.print(
            "     [bold]Fix — pick the lever that matches the traffic:[/bold]"
        )
        console.print("     [dim]Agent-harness sessions:[/dim]")
        console.print(f"       {finding.fix_compaction}")
        if finding.fix_cache_control:
            console.print("     [dim]SDK sessions:[/dim]")
            console.print(
                finding.fix_cache_control, markup=False, highlight=False, soft_wrap=True,
            )
    else:  # persona in {"claude-code", "unknown"}, or "sdk" with no snippet
        console.print(f"     [bold]Fix:[/bold] {finding.fix_compaction}")
        if finding.fix_cache_control:
            console.print(
                "     [dim]If you also run SDK agents against these models, "
                "the cache_control lever applies too:[/dim]"
            )
            console.print(
                finding.fix_cache_control, markup=False, highlight=False, soft_wrap=True,
            )


# Dispatch table — analyzer registration name → renderer.
_FINDING_RENDERERS = {
    "cache":       _render_cache_efficacy,
    "cache-recommend":      _render_cache_recommend,
    "resend":      _render_resend,
    "script": _render_workflow_restructure,
    "reuse":        _render_reuse,
    "trim":         _render_prompt_bloat,
    "subagent":     _render_subagent,
    "relearn":      _render_relearn,
    "verbosity":    _render_verbosity,
    "deadweight":   _render_deadweight,
    "placement":    _render_placement,
    "summarize":    _render_summarize,
}
