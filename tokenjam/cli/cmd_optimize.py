"""`tj optimize` — surface cost-saving candidates and budget projections."""
from __future__ import annotations

import json
from typing import Any, NoReturn

import click
from rich.markup import escape as _rich_escape
from rich.padding import Padding
from rich.table import Table

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.cli.tj_status import TjCommand, tj_status
from tokenjam.core.optimize.analyzers.deadweight import UNUSED_RECENCY_WINDOW_DAYS
from tokenjam.core.optimize.types import DEGRADED_CAPTURE_MODES
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
    ALWAYS_FULL_FINDINGS as _ALWAYS_FULL_FINDINGS,
    ANALYZER_REGISTRY,
    MODEL_DOWNGRADE_CAVEAT,
    BudgetProjection,
    DowngradeFinding,
    OptimizeReport,
    build_report,
    disabled_analyzers_for_persona as _disabled_analyzers,
    rank_findings as _rank_findings_core,
    reclaimable_share as _reclaimable_share,  # noqa: F401 — re-exported for tests
    report_from_dict,
    report_to_dict,
)
from tokenjam.core.rulewrite.delivery import delivery_label
from tokenjam.utils.formatting import (
    console,
    format_cost,
    format_tokens,
    make_table,
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


def _guard_export_templates(selected: set[str], persona: str) -> None:
    """Fail `--export-templates` when `reuse` will not produce a finding.

    Two ways it cannot, and the order matters. The persona skip gate
    (`PERSONA_DISABLED_ANALYZERS`, Critical Rule 26) drops `reuse` inside
    `build_report` even when the user names it explicitly, so on a gated
    window "add `reuse` to the finding list" names a command that cannot
    work — check the gate first and say so. The disabled set is read from
    `disabled_analyzers_for_persona`, never re-declared here.

    Both cases previously fell through to `_export_reuse_templates`, which
    printed "No repeated planning detected" — a claim about the DATA, made
    when the analyzer that would have looked never ran (#578).
    """
    if "reuse" in _disabled_analyzers(persona):
        raise click.ClickException(
            "--export-templates exports the reuse finding, and reuse did not "
            f"run on this window: it is skipped for the {persona} persona, "
            "so it never queried your data and nothing was measured to "
            "export. `tj optimize reuse` prints the same note. If you also "
            "run SDK or API agents, scope the window to "
            "one of them: tj --agent <agent_id> optimize reuse "
            "--export-templates."
        )
    if "reuse" not in selected:
        raise click.ClickException(
            "--export-templates requires the reuse finding. Run "
            "`tj optimize reuse --export-templates`, or include "
            "`reuse` in the finding list."
        )


#: No class-level `status_message`: the `--validate` branch (below) has its
#: own confirmation prompt, and a live spinner colliding with a blocking
#: stdin read corrupts both. `tj_status` is called manually below, scoped to
#: just the report fetch/build — the one stretch that's actually silent and
#: has no prompt anywhere in it.
@click.command("optimize", cls=TjCommand)
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
@click.option("-v", "--verbose", "verbose_flag", is_flag=True, default=False,
              help="Print every finding card in full instead of the scoreboard.")
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
    verbose_flag: bool,
    output_json_flag: bool,
) -> None:
    """Find cost-saving opportunities."""
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

    # --export-templates only makes sense when the reuse analyzer runs.
    # findings=None runs every registered analyzer (including reuse).
    selected_analyzers = (
        set(analyzer_findings)
        if analyzer_findings is not None
        else set(ANALYZER_REGISTRY.keys())
    )

    # The one truly slow, silent stretch in this command: fetching or
    # building the report (analyzer sweep can run to a couple of minutes on
    # a large corpus) plus the two opportunistic background passes below it.
    # --validate (above) never reaches here, so its confirmation prompt is
    # never live at the same time as this spinner.
    with tj_status("Scanning your sessions…", ctx):
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

            # The daemon no longer runs analyzers on a request — it serves the
            # report its background scan stored (`core.optimize.report_store`).
            # A cold store is NOT an empty report: say "not computed yet" rather
            # than rendering a report full of zeros that reads as "no waste".
            if report_dict.get("report_available") is False:
                _echo_scan_not_ready(report_dict, output_json)
                return

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
            # Daemon mode learns the persona only from the payload — `report.persona`
            # is the value the runner actually gated on, so read it rather than
            # re-deriving one here. Checked before the export branch below, whose
            # "stop the daemon and re-run" advice would be false when the real
            # blocker is the persona gate.
            if export_templates:
                _guard_export_templates(selected_analyzers, report.persona or "unknown")
            # Plan-tier mix is included in the /api/v1/optimize payload as of
            # #68 §12 follow-up #29, so the CLI can render subscription /
            # local / unknown framings correctly under daemon mode.
            plan_mix = report_dict.get("plan_tier_mix") or {}
            # Agent-persona mix (#97) — same daemon-mode plumbing as plan_mix
            # above, so the downsize CTA matches persona whether or not the
            # daemon is up.
            agent_mix = report_dict.get("agent_persona_mix") or {}
        else:
            # Resolved before build_report — and before the no-data return below,
            # so an unusable `--export-templates` still fails loudly on an empty
            # window instead of exiting 0 — using the same window read
            # build_report makes for itself (`agent_persona_mix` /
            # `dominant_persona` are the one derivation both sides call).
            agent_mix = agent_persona_mix(conn, since_dt, until_dt, agent)
            if export_templates:
                _guard_export_templates(
                    selected_analyzers,
                    dominant_persona(agent_mix, declared_plan=config_declared_plan(config)),
                )

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

            # The Review inbox reads this finding from relearn_store rather
            # than recomputing it on every request. A direct CLI run has just
            # computed the full finding, so publish that exact result before
            # rendering it. In daemon mode the CLI only reads a stored report,
            # and analyzer subsets that omit relearn leave the cache untouched.
            relearn_finding = (report.findings or {}).get("relearn")
            if relearn_finding is not None:
                from tokenjam.core.optimize import relearn_store

                relearn_store.write_cache(relearn_finding, config=config)

            plan_mix = plan_tier_mix(conn, since_dt, until_dt, agent)

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
            # `recompute_cost_proposals` already never raises — it returns a
            # `CostRecomputeResult` whose `status` says whether it built anything
            # (`ready`), stood aside for a scan cycle or a concurrent recompute
            # (`declined`), or blew up (`failed`). This call is opportunistic
            # upkeep of someone else's store, not a figure `tj optimize` renders,
            # so it acts on none of those: a broken window here degrades to a
            # stale/empty cost-proposals list, never a broken `tj optimize`.
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
        # The names the persona gate dropped, so a --json consumer can tell
        # "ran, found nothing" from "not run for this persona" — same
        # distinction the /optimize route exposes. The two JSON surfaces must
        # not drift: a scripted consumer reading the CLI would otherwise see an
        # absent finding with no way to know why.
        payload["persona_disabled_analyzers"] = sorted(
            _disabled_analyzers(report.persona or "unknown")
        )
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

    # Three views over one renderer set (see `_render_scoreboard`). The card
    # path is reached deliberately — by naming an analyzer, or by asking for
    # everything with -v — rather than being the default nobody chose.
    # `-v` is byte-for-byte what a bare `tj optimize` used to print, which is
    # also the migration path for the tests that pin that output.
    # Either position turns it on. The scoreboard's Next block advertises
    # `tj optimize -v`, which parsed as "No such option" while the flag lived
    # only on the group: reading `ctx.obj` is not the same as accepting the
    # flag. Both copies are booleans meaning the same thing, so OR is the
    # entire resolution rule.
    verbose = verbose_flag or bool(ctx.obj.get("verbose"))
    if verbose or requested:
        _render_report(
            report, agent=agent, plan_mix=plan_mix,
            dominant_plan=dominant, pricing_mode=pricing_mode,
            declared_plan=declared_plan,
            requested=requested,
            persona=persona,
        )
    else:
        _render_scoreboard(
            report, agent=agent, plan_mix=plan_mix,
            dominant_plan=dominant, pricing_mode=pricing_mode,
            declared_plan=declared_plan,
            cost_proposal_count=cost_proposal_count,
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
    # Card path only: the scoreboard names the same command in its Next block,
    # so printing it here too would say it twice.
    if cost_proposal_count and (verbose or requested):
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
# token-reclamation one — its `past_overspend_tokens` is a soft
# occurrence×heuristic estimate for the Lens inbox, not a real fraction of the
# window. Ranking it by that share let a heavy `--since 365d` window (huge
# denominator) push real clusters below DE_MINIMIS_SHARE and hide them behind a
# "~0.0% of window tokens" pointer — the same "nothing found" failure as the
# empty-state bug. These findings always render in full (the `unranked`
# bucket): their own detail when populated, their own empty-state when not.
# `_ALWAYS_FULL_FINDINGS` is imported from core so CLI and API cannot drift.

# Display labels for the "Minor findings" collapsed pointer list — must match
# the header text each renderer prints in its numbered form.
_MINOR_FINDING_LABELS = {
    "downsize":        "Downsize",
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
    "stream-usage":    "Streaming usage gap",
}


def _numbered_marker(n: int) -> str:
    """Circled-digit marker for a ranked finding's slot (①, ②, … ⑳)."""
    if 1 <= n <= 20:
        return chr(0x2460 + n - 1)
    return f"({n})"  # defensive — no report should ever have this many


def _echo_scan_not_ready(payload: dict, output_json: bool) -> None:
    """Report that the daemon's analyzer scan has not produced a result yet.

    Deliberately NOT an empty report and never a set of zeros: "the scan has
    not completed" and "the scan found nothing" are different claims, and only
    one of them is true here. `status` distinguishes a never-run store from one
    whose only attempts errored, so the user is told which.
    """
    status = payload.get("status") or "never_run"
    if output_json:
        click.echo(json.dumps({
            "error": "scan_not_ready",
            "status": status,
            "last_error": payload.get("last_error"),
            "message": "The tj serve daemon has not stored an analyzer report yet.",
        }))
        return
    if status == "computing":
        console.print(
            "[yellow]Analyzer scan is running.[/yellow] "
            "[dim]tj serve computes the report in the background; "
            "re-run this in a moment.[/dim]"
        )
    elif status == "error":
        console.print(
            "[yellow]The last analyzer scan failed[/yellow] "
            f"[dim]({payload.get('last_error') or 'no detail recorded'}). "
            "Nothing has been computed yet — this is not a report of "
            "zero waste.[/dim]"
        )
    else:
        console.print(
            "[yellow]No analyzer report has been computed yet.[/yellow] "
            "[dim]tj serve scans in the background on startup and on a "
            "schedule. Press Rescan in the web UI, or stop the daemon "
            "([bold]tj stop[/bold]) to run the analyzers locally.[/dim]"
        )


def _rank_findings(
    report: OptimizeReport, requested: list[str] | None,
) -> list[tuple[str, float | None]]:
    """Rank findings with something to show by reclaimable token share."""
    return _rank_findings_core(
        report,
        requested,
        known_names=_FINDING_RENDERERS,
        always_full=_ALWAYS_FULL_FINDINGS,
    )


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


def _format_plan_multiplier(multiplier: float) -> str:
    """Render implied-API-value / plan-fee multiplier for subscription users."""
    if multiplier < 0.1:
        return "<0.1×"
    return f"{multiplier:.1f}×"


#: The analyzers whose evidence is the filesystem rather than the span table
#: (docs/configuration.md, "Which filesystem the analyzers read") — the ones a
#: skipped scan actually costs the reader.
_FILESYSTEM_SCAN_ANALYZERS = ("deadweight", "relearn", "summarize")


def _filesystem_scan_note(report: OptimizeReport) -> str | None:
    """The "these did not look" line for a skipped filesystem scan, or None.

    Hardcoding the three names made the note assert something false for a
    persona whose gate had already dropped some of them: `deadweight` and
    `summarize` are gated off for `sdk`, so naming them under a SCOPE reason
    blames the scope for an absence the persona gate caused. Names are filtered
    against the gate, and a skip that cost this reader nothing says nothing.
    """
    reason = report.filesystem_scan_skipped_reason
    if not reason:
        return None
    disabled = _disabled_analyzers(report.persona or "unknown")
    names = [n for n in _FILESYSTEM_SCAN_ANALYZERS if n not in disabled]
    if not names:
        return None
    joined = f"{', '.join(names[:-1])} and {names[-1]}" if len(names) > 1 else names[0]
    return f"  [dim]{joined} did not run: {_rich_escape(str(reason))}.[/dim]"


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
                f"{_format_plan_multiplier(multiplier)} your plan cost.[/dim]\n"
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

    # Said out loud, because the alternative is analyzers rendering as
    # "nothing found" when the truth is that they never looked (root
    # anti-pattern 22). See `core/optimize/scope.py`.
    scan_note = _filesystem_scan_note(report)
    if scan_note:
        console.print(scan_note)
        console.print()

    # An analyzer the user typed by name that this persona's skip gate dropped
    # would otherwise render as silence, which reads as a broken command. The
    # gate is deliberately invisible when it fires on the default (unnamed)
    # selection — those analyzers simply do not exist for this user — but a
    # name someone typed deserves an answer.
    if requested:
        gated = sorted(
            set(requested) & _disabled_analyzers(report.persona or "unknown")
        )
        if gated:
            # The REASON is persona-specific, and one sentence cannot carry
            # both: a `claude-code` window loses analyzers whose lever lives on
            # the harness's side of the line, while an `sdk` window loses ones
            # whose INPUT it structurally does not have (a Claude Code
            # transcript, a populated `sub_agent_id`, an agent instruction
            # file). Stating the wrong one is worse than terse. Per-analyzer
            # reasons live in `PERSONA_DISABLED_ANALYZERS`.
            reason = (
                "No fix for these exists inside an interactive coding-agent session"
                if (report.persona or "") == "claude-code"
                else "These read an input an SDK/API window does not have"
            )
            console.print(
                f"  [dim]Not run: {', '.join(gated)}. {reason}, so they are "
                f"skipped rather than reported as findings you cannot act "
                f"on.[/dim]\n"
            )

    # ----- Findings, ranked by reclaimable share of the window's tokens -----
    # Findings used to render in ANALYZER_ORDER (registry order), so a
    # nothing-burger could occupy the top numbered slot just because its
    # analyzer happened to run first — e.g. "① Model downgrade: 28% of
    # sessions match" when those sessions held ~0% of the window's tokens
    # (#97). Rank by past_overspend_tokens / window.total_tokens
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
                    f"{_finding_header(marker, 'Downsize:')} "
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
        elif name == "cache-recommend":
            # Persona-gated cache_control snippet, same reason resend/downsize
            # get `persona` above — see `_render_cache_recommend`. Called
            # directly for the same mypy reason documented on the `resend`
            # branch: `_FINDING_RENDERERS`'s inferred call signature doesn't
            # include `persona`.
            _render_cache_recommend(
                report.findings[name], pricing_mode=pricing_mode, marker=marker,
                persona=persona,
            )
        elif name == "cache":
            # Same persona gate as `cache-recommend` just above, applied to
            # the A1/A2/A3 root-caused candidates nested inside this finding
            # — see `_render_cache_root_causes` / `_render_cache_control_or_
            # no_lever`. Same mypy reason for the direct call.
            _render_cache_efficacy(
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


# ---------------------------------------------------------------------------
# Scoreboard — the default `tj optimize` view
# ---------------------------------------------------------------------------
# `_render_report` above prints every finding card in full: candidate lists,
# verbatim caveats, methodology paragraphs. That is the right screen once you
# have chosen an area to work on, and the wrong one as an opening screen —
# several pages of justified prose in which the headline numbers, the
# per-area findings and the next command all sit at equal weight.
#
# So the card path is now reached deliberately rather than by default:
#
#   tj optimize            → the scoreboard below (never calls a card renderer)
#   tj optimize <area>     → that one card, in full, renderer untouched
#   tj optimize -v         → byte-for-byte what `tj optimize` printed before
#
# The one-line summaries live HERE, in the CLI layer, rather than as a new
# `summary` field on every analyzer's finding dataclass: they are a
# presentation concern of this one screen, and the analyzers must stay the
# single source of the prose that has to render verbatim.
#
# Honesty rails, which are the whole reason this is a summary and not a
# rewrite (Critical Rule 14 in tokenjam/CLAUDE.md, root anti-pattern 22):
#
#   * A `caveat` / `estimate_basis` / `coverage_note` is NEVER paraphrased
#     into a summary line. The scoreboard carries a pointer at the card that
#     prints them verbatim, and nothing else.
#   * A finding with no priced figure shows `—` in RECOVERABLE. Never `0`,
#     never blank: zero reads as "no waste", which is the opposite claim.
#   * An analyzer that ran and found nothing gets no row, and the
#     `N analyzers · M findings` header line carries the did-it-run signal so
#     silence is still not ambiguous.


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _summarize_downsize(f: Any) -> tuple[str, str] | None:
    if not f.candidate_sessions:
        return None
    return _plural(f.candidate_sessions, "downsize candidate"), "which sessions"


def _summarize_cache(f: Any) -> tuple[str, str] | None:
    flagged = list(f.flagged) if f.flagged else []
    root_caused = (
        len(f.uncached_agents or []) + len(f.thrash_agents or [])
        + len(f.lookback_miss_agents or [])
    )
    if not flagged and not root_caused:
        return None
    if flagged:
        line = (
            f"{_plural(len(flagged), 'model')} below "
            f"{f.efficacy_threshold * 100:.0f}% cache efficacy"
        )
    else:
        line = _plural(root_caused, "cache root-cause candidate")
    return line, "why"


def _summarize_cache_recommend(f: Any) -> tuple[str, str] | None:
    candidates = list(f.candidates) if f.candidates else []
    if not candidates:
        return None
    return _plural(len(candidates), "uncached repeated prefix"), "which prefixes"


def _summarize_resend(f: Any) -> tuple[str, str] | None:
    if f.repeat_share is None:
        return None
    share = f"{f.repeat_share * 100:.0f}%"
    return f"{share} of prompt tokens re-sent", f"why {share}"


def _summarize_script(f: Any) -> tuple[str, str] | None:
    clusters = list(f.clusters) if f.clusters else []
    if not clusters:
        return None
    return _plural(len(clusters), "scriptable repeated workflow"), "which workflows"


def _summarize_reuse(f: Any) -> tuple[str, str] | None:
    clusters = list(f.clusters) if f.clusters else []
    if not clusters:
        return None
    return _plural(len(clusters), "repeated plan cluster"), "which plans"


def _summarize_trim(f: Any) -> tuple[str, str] | None:
    per_prompt = list(f.per_prompt) if f.per_prompt else []
    if not per_prompt:
        return None
    return _plural(len(per_prompt), "prompt with low-significance text"), "which regions"


def _summarize_subagent(f: Any) -> tuple[str, str] | None:
    flagged = list(f.flagged) if f.flagged else []
    if not flagged:
        return None
    return _plural(len(flagged), "over-powered candidate"), "which subagents"


def _summarize_relearn(f: Any) -> tuple[str, str] | None:
    clusters = list(f.clusters) if f.clusters else []
    if not clusters:
        return None
    return _plural(len(clusters), "recurring blocker"), "which blockers"


def _summarize_verbosity(f: Any) -> tuple[str, str] | None:
    candidates = list(f.candidates) if f.candidates else []
    if not candidates:
        return None
    total = f.total_candidates or len(candidates)
    return _plural(total, "high-output candidate"), "which sessions"


def _summarize_deadweight(f: Any) -> tuple[str, str] | None:
    unused = list(f.unused_servers) if f.unused_servers else []
    unused_plugins = list(getattr(f, "unused_plugins", None) or [])
    total = len(unused) + len(unused_plugins)
    if not total:
        return None
    parts = []
    if unused:
        parts.append(_plural(len(unused), "MCP server"))
    if unused_plugins:
        parts.append(_plural(len(unused_plugins), "plugin"))
    return f"{' + '.join(parts)} injected, never invoked", "which servers/plugins"


def _summarize_placement(f: Any) -> tuple[str, str] | None:
    candidates = list(f.candidates) if f.candidates else []
    if not candidates:
        return None
    return _plural(len(candidates), "batch-placement candidate"), "which jobs"


def _summarize_summarize(f: Any) -> tuple[str, str] | None:
    candidates = list(f.candidates) if f.candidates else []
    if not candidates:
        return None
    return _plural(len(candidates), "summarizable prompt file"), "which files"


def _summarize_stream_usage(f: Any) -> tuple[str, str] | None:
    if not f.call_sites:
        return None
    return (
        f"{f.streams_missing_usage} of {f.streams_observed} streams reported no usage",
        "which call sites",
    )


# Dispatch table — analyzer registration name → one-line summarizer. Mirrors
# `_FINDING_RENDERERS`; a summarizer returns None when its analyzer ran and
# found nothing, which is how a clean analyzer earns no row.
_FINDING_SUMMARIES = {
    "downsize":      _summarize_downsize,
    "cache":         _summarize_cache,
    "cache-recommend": _summarize_cache_recommend,
    "resend":        _summarize_resend,
    "script":        _summarize_script,
    "reuse":         _summarize_reuse,
    "trim":          _summarize_trim,
    "subagent":      _summarize_subagent,
    "relearn":       _summarize_relearn,
    "verbosity":     _summarize_verbosity,
    "deadweight":    _summarize_deadweight,
    "placement":     _summarize_placement,
    "summarize":     _summarize_summarize,
    "stream-usage":  _summarize_stream_usage,
}

# Findings whose headline figure is NOT a recoverable amount, so the
# RECOVERABLE column must stay `—` for them however well-priced they are.
# `stream-usage` carries `undercounted_usd`: spend that already happened and
# was never recorded. A data-quality number sitting in a savings column is
# read as a saving.
_UNPRICED_IN_SCOREBOARD = {"stream-usage"}


def _scoreboard_recoverable(name: str, finding: Any, framing: Framing) -> str:
    """RECOVERABLE cell for one row — `—` whenever there is no priced figure.

    `render_savings` already returns the `—` marker for a missing figure, so
    an absent estimate can never surface as `0` or an empty cell.
    """
    if name in _UNPRICED_IN_SCOREBOARD:
        return "—"
    return render_savings(
        getattr(finding, "past_overspend_usd", None),
        getattr(finding, "past_overspend_tokens", None),
        framing,
    )


def _scoreboard_sort_value(name: str, finding: Any, framing: Framing) -> float | None:
    """The numeric behind the RECOVERABLE cell, for ordering the rows.

    Deliberately mirrors `_scoreboard_recoverable` field for field, including
    which figure each pricing mode renders (`local` shows tokens, everything
    else dollars). A sort key read from a different field than the column
    displays produces exactly the defect this replaces: a table that looks
    ranked and is not.

    `None` for anything the column renders as the null marker, so an unpriced
    row sorts to the bottom instead of being treated as a zero-value row.
    """
    if name in _UNPRICED_IN_SCOREBOARD:
        return None
    if framing.pricing_mode == "local":
        tokens = getattr(finding, "past_overspend_tokens", None)
        return None if tokens is None else float(tokens)
    usd = getattr(finding, "past_overspend_usd", None)
    return None if usd is None else float(usd)


def _render_scoreboard(
    report: OptimizeReport,
    agent: str | None,
    plan_mix: dict[str, int] | None = None,
    dominant_plan: str = "unknown",
    pricing_mode: str = "unknown",
    declared_plan: str | None = None,
    cost_proposal_count: int = 0,
) -> None:
    """The default `tj optimize` screen: header, findings table, Next block."""
    w = report.window
    scope_tag = f", {agent}" if agent else ""
    days_int = max(int(round(w.days)), 1)
    plan_mix = plan_mix or {}
    unknown_count = plan_mix.get("unknown", 0)
    total_sessions = sum(plan_mix.values()) or w.sessions
    all_unknown = total_sessions > 0 and unknown_count == total_sessions

    counts = f"[bold]{w.sessions}[/bold] sessions · [bold]{format_tokens(w.total_tokens)}[/bold] tokens"

    # ----- Header -----
    if all_unknown:
        console.print(f"\n  {counts} (last {days_int}d{scope_tag})")
        console.print(
            "  [dim]All sessions have unknown plan tier; dollar figures "
            "suppressed. Run [accent]tj onboard --claude-code --reconfigure"
            "[/accent] to set your plan.[/dim]"
        )
    elif pricing_mode == "subscription":
        label, fee = PLAN_LABEL_AND_FEE.get(dominant_plan, (dominant_plan, None))
        plan_suffix = f" (${fee:.0f}/mo)" if fee else ""
        console.print(f"\n  {counts} · [bold]{label}[/bold]{plan_suffix}")
        if fee and w.total_cost_usd > 0:
            multiplier = w.total_cost_usd / fee
            console.print(
                f"  [dim]Implied API value [bold]{format_cost(w.total_cost_usd)}"
                f"[/bold], about {_format_plan_multiplier(multiplier)} your "
                f"plan cost.[/dim]"
            )
        else:
            console.print(
                f"  [dim]Implied API value [bold]{format_cost(w.total_cost_usd)}"
                f"[/bold] (what this usage would cost at API list prices).[/dim]"
            )
    elif pricing_mode == "local":
        console.print(f"\n  {counts} (last {days_int}d{scope_tag})")
        console.print("  [dim]Local inference; no marginal cost.[/dim]")
    else:
        console.print(
            f"\n  {counts} · [bold]{format_cost(w.total_cost_usd)}[/bold] spend "
            f"(last {days_int}d{scope_tag})"
        )
        if unknown_count > 0:
            console.print(
                f"  [dim]{unknown_count} of {total_sessions} sessions have "
                f"unknown plan tier; dollar figures may overstate actual cost "
                f"for those. Run [accent]tj onboard --claude-code --reconfigure"
                f"[/accent] to resolve.[/dim]"
            )

    if (
        declared_plan
        and declared_plan != dominant_plan
        and declared_plan in PLAN_LABEL_AND_FEE
    ):
        label, _ = PLAN_LABEL_AND_FEE[declared_plan]
        console.print(
            f"  [dim]Your config declares [bold]{label}[/bold] but historical "
            f"sessions ran under a different plan; rendering reflects what "
            f"actually ran.[/dim]"
        )

    if w.sessions == 0:
        console.print("\n  [dim]No sessions in window.[/dim]")
        return

    # ----- Rows -----
    ranked = _rank_findings(report, requested=None)
    framing = Framing(
        pricing_mode=("unknown" if all_unknown else pricing_mode),
        window_total_tokens=w.total_tokens,
    )
    rows: list[tuple[str, str, str, str]] = []
    # `_rank_findings` orders by reclaimable TOKEN share, which is the right
    # order for the card path it was written for (it also drives the
    # de-minimis pointer collapsing there) and the wrong one here: this table
    # publishes a RECOVERABLE column, and a table sorted by a quantity it does
    # not show reads as unsorted — a $48 row landing above a $296 one. So the
    # rows are re-sorted on the figure the column actually prints, with the
    # token-share rank kept only as a deterministic tie-break.
    ordered: list[tuple[float | None, int, str, str, str, str]] = []
    for rank, (name, _share) in enumerate(ranked):
        summarize = _FINDING_SUMMARIES.get(name)
        if summarize is None:
            continue
        finding = report.downgrade if name == "downsize" else report.findings.get(name)
        if finding is None:
            continue
        summary = summarize(finding)
        if summary is None:
            continue
        line, why = summary
        ordered.append((
            _scoreboard_sort_value(name, finding, framing), rank,
            name, line, _scoreboard_recoverable(name, finding, framing), why,
        ))

    # An unpriced row sorts last and keeps its null marker: it is a finding
    # with no figure, not a finding worth nothing.
    ordered.sort(key=lambda r: (r[0] is None, -(r[0] or 0.0), r[1]))
    rows = [(name, line, recoverable, why) for _v, _r, name, line, recoverable, why in ordered]
    # Rows carrying an actual figure. Drives the overlap disclosure below,
    # which only has something to say once two figures sit in one column.
    priced_rows = sum(1 for value, *_rest in ordered if value is not None)

    console.print(
        f"  [dim]{_plural(len(ranked), 'analyzer')} · "
        f"{_plural(len(rows), 'finding')}[/dim]\n"
    )

    # Said out loud rather than left as quiet absences: an analyzer that
    # never looked is not an analyzer that found nothing (root anti-pattern 22).
    for note in report.notes:
        console.print(f"  [warn]![/warn] {_rich_escape(note)}")
    scan_note = _filesystem_scan_note(report)
    if scan_note:
        console.print(scan_note)
    if report.notes or scan_note:
        console.print()

    if rows:
        table = make_table("ANALYZERS", "FINDING", "RECOVERABLE")
        for name, line, recoverable, _why in rows:
            # The recoverable figure is weight, not colour: the accent role
            # means "a string you can type" and nothing else (Critical Rule 35
            # in tokenjam/CLAUDE.md).
            table.add_row(name, _rich_escape(line), f"[label]{recoverable}[/label]")
        console.print(table)
        # A column of dollar figures invites the reader to add it up, and the
        # sum would be wrong: the analyzers price overlapping angles on the
        # SAME sessions (`downsize` deliberately excludes `sub_agent_id IS NOT
        # NULL` spans because `subagent` already prices the identical swap over
        # them), so summing them sums waste measured twice. Same reasoning and
        # substantially the same wording as `_recoverable_overlap_note` in
        # api/routes/cost.py, condensed for a terminal; that route's
        # `recoverable_additive: False` is the machine-readable form of it.
        # This is why the screen prints no total: the top row, being the
        # largest single lever, is not a sum of anything and is the one figure
        # that is honest standing alone.
        if priced_rows >= 2:
            # Padded rather than prefixed with two spaces: this is the one
            # note here long enough to wrap, and a hand-written prefix indents
            # only the first line, dropping the continuation to column 0.
            console.print(Padding(
                f"[dim]These {priced_rows} estimates are computed from "
                f"overlapping angles on the same sessions, so they do not add "
                f"up to an amount you could recover. The largest single line "
                f"is the one to act on first.[/dim]",
                (0, 0, 0, 2),
            ))
        console.print(
            "  [dim]Estimates carry method notes and caveats. See "
            "[accent]tj optimize <analyzer>[/accent].[/dim]\n"
        )
    else:
        console.print(
            "  [dim]No candidates flagged in this window. Either spend is small "
            "or your sessions already use a cost-effective shape.[/dim]\n"
        )

    # ----- Next -----
    # Literal runnable commands, not advice. A finding with nowhere to go is
    # a diagnosis the user cannot act on.
    next_lines: list[tuple[str, str]] = []
    if cost_proposal_count:
        next_lines.append((
            "tj relearn cost-proposals",
            f"{cost_proposal_count} copy-paste fix"
            f"{'' if cost_proposal_count == 1 else 'es'}",
        ))
    if rows:
        top_name, _line, _rec, top_why = rows[0]
        next_lines.append((f"tj optimize {top_name}", top_why))
    next_lines.append(("tj optimize -v", "every finding in full"))

    width = max(len(cmd) for cmd, _ in next_lines)
    for i, (cmd, hint) in enumerate(next_lines):
        prefix = "  Next:  " if i == 0 else "         "
        console.print(f"{prefix}[accent]{cmd:<{width}}[/accent]   [dim]{hint}[/dim]")


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

    The driver-role case leads when it fired: it is the primary case, and on a
    coding-agent corpus it carries essentially all of the analyzer's dollars.
    The tiny-session block below it is skipped entirely when no session matched
    it, so the header never reports "0% of sessions" over a real finding.
    """
    if d.driver_sessions:
        _render_driver_role(d, pricing_mode, marker)
        marker = " "
    if d.candidate_sessions <= 0:
        # No tiny-session candidates, so there is no `bench_command` and no
        # model swap for the CTA to talk about: the driver-role block above
        # carries its own fix, which is a CLAUDE.md rule rather than a swap.
        return

    console.print(
        f"  [bold]{marker} Downsize:[/bold] "
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
            f"smaller model vs {format_cost(d.actual_cost_usd)} actual (in window), "
            f"{format_tokens(d.candidate_tokens)} of {format_tokens(d.window_total_tokens)} "
            f"window tokens"
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


def _render_driver_role(
    d: DowngradeFinding, pricing_mode: str, marker: str,
) -> None:
    """The primary case: a premium model drove undelegated work inline.

    Dollars are suppressed outside `api` pricing exactly as the tiny-session
    block suppresses them — the token figure carries the finding instead, since
    a flat-rate plan's re-read tail is real quota even when it is not a bill.
    """
    swaps = ", ".join(f"{m} → {alt}" for m, alt in sorted(d.driver_substitutes.items()))
    console.print(
        f"  [bold]{marker} Model role:[/bold] a premium model drove "
        f"{d.driver_sessions} of {d.total_sessions} sessions inline, without "
        f"dispatching a single worker"
    )
    console.print(
        f"     • [bold]{format_tokens(d.driver_tail_tokens)}[/bold] tokens were "
        f"re-read purely because that work stayed in the main thread"
    )
    if pricing_mode == "api":
        console.print(
            f"     • Routing it to a worker would have saved "
            f"[bold]{format_cost(d.driver_recoverable_usd)}[/bold] in the window "
            f"({format_cost(d.driver_offload_usd)} of re-reads + "
            f"{format_cost(d.driver_tier_usd)} of tier difference)"
        )
    if swaps:
        console.print(f"     • Suggested worker tier: [dim]{swaps}[/dim]")
    console.print(
        "     • [dim]Your own thread stays on the premium model — what moves is "
        "the context-heavy work, not the driver.[/dim]"
    )
    console.print()


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


def _usd_with_tokens(usd: float, tokens: int | None) -> str:
    """`format_cost(usd)` with a parenthetical token figure when available.

    Shared by the api-mode "estimated recoverable" lines so a dollar figure
    never renders alone when its own past_overspend_tokens counterpart is
    right there on the same candidate.
    """
    if tokens is not None:
        return f"{format_cost(usd)} (~{format_tokens(tokens)} tokens)"
    return format_cost(usd)


def _render_prose(
    text: str, *, marker: str = "", style: str | None = None, escape: bool = True,
    indent: int = 5,
) -> None:
    """Render an analyzer prose field (`caveat`, `estimate_basis`, `notes`,
    `coverage_note`, and their siblings — `friction`, `measurement_note`,
    `accounting_note`) so every line hangs indented under the first
    paragraph's text, not just the first line.

    These fields were rewritten from single 40-90 word run-on paragraphs into
    short sentences grouped into paragraphs separated by a literal blank line
    (`"\\n\\n"`), which the web dashboard already renders correctly. A
    hand-written prefix like `f"     [yellow]![/yellow] [italic]{caveat}[/italic]"`
    only indents line 1 — rich reflows everything after it, both the
    soft-wrapped continuation of a long sentence and every paragraph after a
    blank line, starting back at column 0 and visually detaching it from the
    marker it belongs to.

    Built on a borderless two-column `Table.grid` (elsewhere in this file,
    `_render_scoreboard`'s `priced_rows` note reaches for `Padding` to solve
    the same class of problem for an unmarked block): the marker column is
    `no_wrap` so it never reflows and fixes a left column, and rich aligns
    every line of the prose column — wrapped or blank-line-broken alike —
    under that same column start (root anti-pattern 30: pin the indent to
    every line the renderer produces, not just the first one it happened to
    be written for).

    `marker` is one-time leading markup (e.g. `"[yellow]![/yellow]"` for a
    caveat, `""` for a plain dim line) rendered once at `indent` spaces;
    continuation paragraphs align under the TEXT, not under the marker.
    `style` wraps the prose itself (`"italic"`, `"dim"`). `escape` must match
    each call site's PRE-EXISTING `_rich_escape` usage — several of these
    fields (caveat, estimate_basis) print unescaped today, and this helper
    only fixes indentation, not that.
    """
    body = _rich_escape(text) if escape else text
    content = f"[{style}]{body}[/{style}]" if style else body
    lead = " " * indent + (f"{marker} " if marker else "")
    grid = Table.grid(padding=0)
    grid.add_column(no_wrap=True)
    grid.add_column()
    grid.add_row(lead, content)
    console.print(grid)


def _render_cache_control_or_no_lever(snippet: str, persona: str) -> None:
    """Persona-gated `cache_control` snippet render, shared by every
    cache-family CLI renderer that prints one (`cache`'s A1/A2/A3
    root-caused candidates below, `cache-recommend`'s prefix candidates).

    A `cache_control` fix is an edit to the raw Anthropic API request, a
    lever a Claude Code session never has — the harness constructs that
    request, not the user. Mirrors
    `cost_proposals._persona_gated_cache_fields`: `persona == "claude-code"`
    swaps the snippet for the honest no-lever explanation (imported from
    `cost_proposals` so the CLI never drifts from the web copy); every
    other persona, including "unknown", keeps the actionable snippet — for
    cache advice the risky direction is under-offering a real fix, not
    over-offering one (the opposite default from `_persona_gated_write_fields`).
    """
    if persona == "claude-code":
        from tokenjam.core.optimize.cost_proposals import CACHE_NO_LEVER_TEXT
        console.print(f"           [dim]{_rich_escape(CACHE_NO_LEVER_TEXT)}[/dim]")
        return
    console.print("           [dim]cache_control:[/dim]")
    console.print(snippet, markup=False, highlight=False, soft_wrap=True)


def _render_cache_efficacy(
    finding, *, pricing_mode: str = "api", marker: str = "", persona: str = "unknown",
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
    _render_cache_root_causes(finding, pricing_mode=pricing_mode, persona=persona)


def _render_cache_root_causes(
    finding, *, pricing_mode: str, persona: str = "unknown",
) -> None:
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
    proposals, gated by the same `persona` rule those proposals apply (see
    `_render_cache_control_or_no_lever`): a claude-code window gets the
    honest no-lever explanation instead of a request edit it can't make.
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
                if c.past_overspend_usd is not None:
                    console.print(
                        f"           [dim]≈[/dim] "
                        f"[green]{_usd_with_tokens(c.past_overspend_usd, c.past_overspend_tokens)}[/green] "
                        f"estimated recoverable over this window"
                    )
                else:
                    console.print(
                        "           [dim]no dollar figure: no priced rate "
                        f"observed for {c.model or 'this model'}[/dim]"
                    )
            _render_cache_control_or_no_lever(c.cache_control_snippet, persona)
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
                if c.past_overspend_usd is not None:
                    console.print(
                        f"           [dim]≈[/dim] "
                        f"[green]{_usd_with_tokens(c.past_overspend_usd, c.past_overspend_tokens)}[/green] "
                        f"wasted writing this prefix over this window"
                    )
                else:
                    console.print(
                        "           [dim]no dollar figure: the recommended "
                        "fix would not recover it[/dim]"
                    )
            _render_cache_control_or_no_lever(c.cache_control_snippet, persona)
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
                if c.past_overspend_usd is not None:
                    console.print(
                        f"           [dim]≈[/dim] "
                        f"[green]{_usd_with_tokens(c.past_overspend_usd, c.past_overspend_tokens)}[/green] "
                        f"estimated recoverable over this window"
                    )
                else:
                    console.print(
                        "           [dim]no dollar figure: no priced rate "
                        f"observed for {c.model or 'this model'}[/dim]"
                    )
            _render_cache_control_or_no_lever(c.cache_control_snippet, persona)
        if n > 5:
            console.print(f"       [dim]… and {n - 5} more.[/dim]")

    if pricing_mode != "api":
        console.print(
            "     [dim]This plan doesn't bill per token, so no dollar "
            "figures are shown for these candidates; the counts above still "
            "show the caching opportunity.[/dim]"
        )


def _render_cache_recommend(
    finding, *, pricing_mode: str = "api", marker: str = "", persona: str = "unknown",
) -> None:
    """
    Render the cache-recommend finding — Anthropic-only v1 breakpoint
    candidates. When the analyzer is disabled (capture.prompts off), surface
    the hint instead of an empty table.

    Each candidate's `cache_control_snippet` is an edit to the raw Anthropic
    API request — a lever a Claude Code session never has, since the harness
    constructs that request, not the user. Gated by the same rule
    `cost_proposals._persona_gated_cache_fields` applies to the Review-inbox
    proposal built from this same finding: `persona == "claude-code"` swaps
    the snippet for the honest no-lever explanation (imported from
    `cost_proposals` so the CLI never drifts from the web copy); every other
    persona, including "unknown", still gets the snippet — for cache advice
    the risky direction is under-offering a real fix, not over-offering one.
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
            if c.past_overspend_usd is not None:
                console.print(
                    f"           [dim]≈[/dim] "
                    f"[green]{_usd_with_tokens(c.past_overspend_usd, c.past_overspend_tokens)}[/green] "
                    f"estimated over this window [dim](model {c.model})[/dim]"
                )
            else:
                console.print(
                    f"           [dim]no dollar figure: no priced rate observed "
                    f"for {c.model or 'this model'}[/dim]"
                )
        console.print(f"           [dim italic]{sample}[/dim italic]")
        _render_cache_control_or_no_lever(c.cache_control_snippet, persona)

    if pricing_mode == "api" and finding.past_overspend_usd is not None:
        console.print(
            f"     • [green]~{_usd_with_tokens(finding.past_overspend_usd, finding.past_overspend_tokens)}"
            f"[/green] estimated recoverable across these candidates [dim](reads "
            f"after the first occurrence, minus one cache write per prefix)[/dim]"
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
            f"[dim]({dur}, ~{format_tokens(c.avg_tokens)} avg tokens)[/dim]"
        )
        if pricing_mode == "api" and c.avg_cost_usd > 0:
            console.print(
                f"          [dim]avg session cost {format_cost(c.avg_cost_usd)}; "
                f"replacing with a deterministic script would eliminate it.[/dim]"
            )
    if finding.caveat:
        _render_prose(finding.caveat, marker="[yellow]![/yellow]", style="italic", escape=False)


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
        trim_cost = (
            f" (~{format_cost(p.estimated_cost_reduction_usd)})"
            if pricing_mode == "api" and p.estimated_cost_reduction_usd is not None
            else ""
        )
        console.print(
            f"       [dim]{p.agent_id}[/dim]  "
            f"[bold]{p.bloat_chars}[/bold] bloat / {p.prompt_chars} chars  "
            f"[dim]~{p.estimated_token_reduction} tokens trimmable{trim_cost}[/dim]"
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

    # `capture_mode` states what clustering ACTUALLY ran on (measured per
    # window), so the degrade is named without asserting a cause the finding
    # cannot know — the accompanying hint carries the remedy.
    if finding.capture_mode == "tool_sequence_only":
        mode_note = (
            " [dim](tool-sequence only — no prompt text was captured for "
            "these calls)[/dim]"
        )
    elif finding.capture_mode == "mixed_prompt_prefix":
        mode_note = (
            " [dim](mixed basis — only some of these calls carried prompt "
            "text; the rest matched on tool sequence alone)[/dim]"
        )
    else:
        mode_note = ""
    console.print(
        f"     • [bold]{len(finding.clusters)}[/bold] cluster"
        f"{'s' if len(finding.clusters) != 1 else ''} of repeated planning "
        f"detected{mode_note}"
    )
    if finding.capture_mode in DEGRADED_CAPTURE_MODES and finding.hint:
        console.print(f"     [dim]{_rich_escape(finding.hint)}[/dim]")

    for c in finding.clusters[:5]:
        sig_preview = " → ".join(c.tool_signature) if c.tool_signature else "(no tools)"
        if len(sig_preview) > 100:
            sig_preview = sig_preview[:97] + "..."
        console.print(
            f"       [bold]{c.repetitions}×[/bold] {sig_preview}"
        )
        # Recoverable framing. api/unknown → dollars + tokens; subscription/local
        # → tokens only (dollars suppressed, Critical Rule 22 — a flat-fee/local
        # plan has no marginal dollar figure to realise).
        if pricing_mode in ("subscription", "local"):
            cache_str = f"~{format_tokens(c.cache_reuse_recoverable_tokens)} tokens"
            script_str = f"~{format_tokens(c.script_replacement_recoverable_tokens)} tokens"
        else:
            cache_str = (
                f"{format_cost(c.cache_reuse_recoverable_usd)} "
                f"(~{format_tokens(c.cache_reuse_recoverable_tokens)} tokens)"
            )
            script_str = (
                f"{format_cost(c.script_replacement_recoverable_usd)} "
                f"(~{format_tokens(c.script_replacement_recoverable_tokens)} tokens)"
            )
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
        _render_prose(finding.estimate_basis, style="dim", escape=False)
    if finding.clusters:
        _render_prose(
            finding.clusters[0].caveat, marker="[yellow]![/yellow]", style="italic",
            escape=False,
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
    if finding.past_overspend_tokens is not None:
        if pricing_mode == "api" and finding.past_overspend_usd is not None:
            recov = (
                f"{format_cost(finding.past_overspend_usd)} "
                f"({format_tokens(finding.past_overspend_tokens)} tokens)"
            )
        else:
            recov = f"{format_tokens(finding.past_overspend_tokens)} tokens"
        console.print(
            f"     • [green]~{recov}[/green] estimated recoverable "
            f"[dim](over_powered subagents at their cheaper same-family model)[/dim]"
        )

    _render_prose(finding.caveat, marker="[yellow]![/yellow]", style="italic", escape=False)


def _relearn_past_overspend(cluster, *, pricing_mode: str = "api") -> str:
    """What this cluster ALREADY COST, as a short display string.

    Named for the field it reads. It was ``_relearn_observed_cost``, which named
    a ``CostProposal`` field it never touched and which no longer exists at all;
    a helper named after the wrong field is how the next reader concludes the
    purge was incomplete.

    Reads ``past_overspend_*`` straight off the cluster and derives nothing:
    the figure is ungated on purpose (a cluster with no fix template in our
    library, and a cluster whose rule is uneconomic to keep, both still cost
    real money) and no future maintenance cost is netted out of it.

    Dollars alongside tokens for api-billed users; a subscription user sees the
    token figure only, never a price they were never charged (Critical Rule 22).
    Empty string when the cluster carries neither field — a cache written
    before they existed — so the caller omits the segment rather than
    printing a confident zero.
    """
    usd = getattr(cluster, "past_overspend_usd", None)
    tokens = getattr(cluster, "past_overspend_tokens", None)
    if pricing_mode == "api" and usd:
        if tokens:
            return f"{format_cost(usd)} (~{format_tokens(tokens)} tok)"
        return format_cost(usd)
    if tokens:
        return f"~{format_tokens(tokens)} tok"
    return ""


def _render_relearn_gate_summary(clusters) -> None:
    """One line naming how many clusters have no apply path at all (advisory
    families, or a workspace/persona that has no write surface). Without it
    the list reads as though every cluster is actionable, when a few carry no
    apply path of their own — never a finding that those failures were
    harmless, which is the reading a bare count invites.
    """
    gated = [c for c in clusters if getattr(c, "advise_only", False)]
    if not gated:
        return
    console.print(
        f"     [dim]{len(gated)} of {len(clusters)} have no permanent-rule "
        f"apply path of their own (no workspace to write into, or the "
        f"harness already self-corrects). Each still cost what is shown "
        f"above: the gap is in what we can act on, not a finding that the "
        f"failure was harmless.[/dim]"
    )


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
    # Lead each row with what the recurrence ALREADY COST, never a fix-gated
    # forward claim — a cluster's cost stands whether or not it has a
    # permanent-rule apply path (CLAUDE.md anti-pattern 32b).
    for c in finding.clusters[:10]:
        cost = _relearn_past_overspend(c, pricing_mode=pricing_mode)
        console.print(
            f"       [bold]{c.signature}[/bold]  "
            f"{c.occurrences} occurrence{'s' if c.occurrences != 1 else ''} / "
            f"{c.sessions} session{'s' if c.sessions != 1 else ''}  "
            f"[dim]({delivery_label(getattr(c, 'delivery', ''))})[/dim]"
            + (f"  [bold]{cost}[/bold] [dim]already spent[/dim]" if cost else "")
        )
        if getattr(c, "advise_only", False):
            console.print(
                "         [dim]no permanent-rule apply path — see the "
                "example sessions for what to change by hand[/dim]"
            )
    if len(finding.clusters) > 10:
        console.print(f"       [dim]… and {len(finding.clusters) - 10} more.[/dim]")
    _render_relearn_gate_summary(finding.clusters)
    console.print(
        "     [dim]Review + apply fixes in the Lens Review inbox, or see "
        "full detail with [bold]tj optimize relearn --json[/bold].[/dim]"
    )
    if finding.caveat:
        _render_prose(finding.caveat, marker="[yellow]![/yellow]", style="italic", escape=False)


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
        # Recoverable framing: dollars plus tokens for api-billed users;
        # otherwise the over-baseline token figure only (the same category
        # discipline as peers — Critical Rule 22).
        if pricing_mode == "api" and c.recoverable_usd:
            recov = (
                f"~{format_cost(c.recoverable_usd)} "
                f"(~{format_tokens(c.over_baseline_tokens)} tokens) at output rates"
            )
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
        _render_prose(finding.caveat, marker="[yellow]![/yellow]", style="italic", escape=False)


def _render_summarize(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the summarize finding — catalog prompt files (CLAUDE.md / AGENTS.md /
    globals) whose prose could be summarized. Registered and runs like every
    other analyzer, but had no entry in `_FINDING_RENDERERS`, so it was
    silently dropped from plain-text `tj optimize` output and only reachable
    via `--json`.

    The per-file line is the one-time, per-CALL reduction (`file_reduction_
    tokens`); `past_overspend_tokens` and `past_overspend_usd` are both
    per-WINDOW, priced by how each file actually LOADS: always-on text (a
    CLAUDE.md, a rules file, a skill/command/agent's frontmatter) on every call
    of every loading session, and a skill/command/agent BODY only on the
    occasions it was observed being invoked (see
    core/optimize/analyzers/summarize.py). The window line only appears when the
    analyzer could observe those counts — it is never fabricated from a default
    rate — and it goes through `render_savings` so a subscription/local plan
    sees the same framing every other analyzer gives it.
    """
    console.print(_finding_header(marker, "Summarize:"))
    if not finding.candidates:
        console.print(
            "     [dim]No catalog prompt files (CLAUDE.md / AGENTS.md / "
            "globals) with summarizable prose found.[/dim]"
        )
        return

    file_reduction_tokens = getattr(finding, "file_reduction_tokens", None) or 0
    console.print(
        f"     • [bold]{finding.files}[/bold] file{'s' if finding.files != 1 else ''} "
        f"summarizable, ~[bold]{format_tokens(file_reduction_tokens)}[/bold] per call "
        f"[dim](aggregate {finding.reduction_pct}% prose reduction)[/dim]"
    )
    recoverable_usd = getattr(finding, "past_overspend_usd", None)
    window_tokens = finding.past_overspend_tokens or 0
    if recoverable_usd:
        savings = render_savings(
            recoverable_usd, window_tokens, Framing(pricing_mode=pricing_mode),
        )
        if savings != "—":
            console.print(
                f"       [green]~{savings}[/green] across the "
                f"[bold]{finding.sessions_examined}[/bold] session(s) in this "
                f"window: always-on text on every call; skill / command / agent "
                f"bodies only when invoked"
            )
            if getattr(finding, "invocations_observed", False):
                console.print(
                    f"       [dim]{finding.invocations_total:,} invocation(s) "
                    f"observed across {finding.transcripts_examined:,} "
                    f"transcript(s).[/dim]"
                )
    for c in finding.candidates[:5]:
        # An on-demand file's per-call figure is only realized on the calls that
        # follow an invocation, so the class is named rather than left implied.
        load_class = getattr(c, "load_class", "always")
        loads = "always-on" if load_class == "always" else f"on demand: {load_class}"
        console.print(
            f"       [dim]{c.path}[/dim]  [dim]({c.scope}, {loads})[/dim]  "
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
        _render_prose(finding.caveat, marker="[yellow]![/yellow]", style="italic", escape=False)


def _render_deadweight_plugins(finding, *, pricing_mode: str = "api") -> None:
    """The plugin half of the deadweight finding.

    Renders the RESIDENT count against the installed count, always, because the
    gap is the story: on a real machine most of what is installed is switched
    off or scoped to one project and costs nothing, and a reader shown only a
    dollar figure has no way to know which of those they are looking at.

    Three-state gating per plugin (never a number with no evidence behind
    it): `unused` gets the priced row + disable arrow; `partial_use_no_fix`
    names which components are unused and says plainly no fix is available
    (enable/disable is whole-plugin only); `insufficient_history` renders as
    "not enough history yet" — never a number, never a fix arrow. A plugin
    with nothing measurable (no components, or only unmeasured MCP servers)
    never reaches ANY of these three rows — see `PluginDeadweight.unused`'s
    vacuous-truth guard — so a `$0` row with a disable arrow cannot render.
    """
    plugins = list(getattr(finding, "plugins", []) or [])
    if not plugins:
        return
    resident = int(getattr(finding, "plugins_resident", 0) or 0)
    console.print(
        f"     [dim]Plugins: {resident} of {len(plugins)} installed are resident "
        f"(the rest are disabled or scoped to one project and cost "
        f"nothing).[/dim]"
    )
    for plugin in getattr(finding, "unused_plugins", []) or []:
        # Guaranteed by the analyzer's own vacuous-truth guard, but re-asserted
        # here rather than trusted blindly: a $0 row must never carry a
        # disable arrow (Part C). If this ever fires it is a bug upstream,
        # not a row worth printing.
        if not plugin.estimated_tax_tokens_window:
            continue
        console.print(
            f"       [bold]{plugin.name}[/bold] [dim]({plugin.skills} skill"
            f"{'s' if plugin.skills != 1 else ''}, {plugin.agents} agent"
            f"{'s' if plugin.agents != 1 else ''} listed every session)[/dim]"
        )
        if pricing_mode == "api" and plugin.estimated_tax_usd_window is not None:
            tax = (
                f"~{format_tokens(plugin.estimated_tax_tokens_window)} tokens / "
                f"{format_cost(plugin.estimated_tax_usd_window)} in this window "
                f"[dim](estimated, priced at {plugin.priced_model})[/dim]"
            )
        else:
            tax = (
                f"~{format_tokens(plugin.estimated_tax_tokens_window)} tokens in "
                f"this window [dim](estimated; no priced model observed, so no "
                f"dollar figure)[/dim]"
            )
        console.print(f"          [dim]tax[/dim] {tax}")
        if plugin.tax_construction:
            console.print(f"          [dim]{_rich_escape(plugin.tax_construction)}[/dim]")
        if plugin.fix:
            console.print(f"          [yellow]→[/yellow] {_rich_escape(plugin.fix)}")
    for plugin in getattr(finding, "plugins", []) or []:
        if plugin.partial_use_no_fix:
            console.print(
                f"       [bold]{plugin.name}[/bold] [dim](some components used, "
                f"some not)[/dim]"
            )
            console.print(f"          [dim]{_rich_escape(plugin.fix)}[/dim]")
    insufficient = [p for p in plugins if p.resident and p.insufficient_history]
    if insufficient:
        names = ", ".join(p.name for p in insufficient)
        console.print(
            f"     [dim]Not enough session history yet to say whether "
            f"{names} {'is' if len(insufficient) == 1 else 'are'} used "
            f"(need {UNUSED_RECENCY_WINDOW_DAYS} days).[/dim]"
        )


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
        # The plugin lane is INDEPENDENT of the MCP one and must still render
        # here: a user with no MCP servers at all can be paying for an enabled
        # plugin in every session, and returning early would make the whole
        # lane invisible to exactly that user (Critical Rule 24 — a capability
        # nobody has a path to does not exist).
        _render_deadweight_plugins(finding, pricing_mode=pricing_mode)
        if finding.coverage_note:
            _render_prose(finding.coverage_note, marker="[yellow]![/yellow]", style="dim")
        return

    if not finding.unused_servers:
        # _rich_escape: the analyzer's own note may name a config key in
        # bracket form, which Rich would otherwise parse as an unknown style
        # tag and silently drop from the printed line.
        for note in finding.notes:
            _render_prose(note, style="dim")
        if not finding.notes:
            console.print(
                f"     [dim]All {finding.configured_servers} configured MCP "
                f"server{'s' if finding.configured_servers != 1 else ''} were "
                f"invoked at least once in this window.[/dim]"
            )
        insufficient = [s for s in finding.servers if s.insufficient_history]
        if insufficient:
            names = ", ".join(s.name for s in insufficient)
            console.print(
                f"     [dim]Not enough session history yet to say whether "
                f"{names} {'is' if len(insufficient) == 1 else 'are'} used "
                f"(need {UNUSED_RECENCY_WINDOW_DAYS} days).[/dim]"
            )
    else:
        n = len(finding.unused_servers)
        console.print(
            f"     • [bold]{n}[/bold] unused MCP server{'s' if n != 1 else ''} of "
            f"[bold]{finding.configured_servers}[/bold] configured "
            f"[dim](schemas injected every session, nothing fired in the last "
            f"{UNUSED_RECENCY_WINDOW_DAYS} days)[/dim]"
        )
        for s in finding.unused_servers:
            console.print(
                f"       [bold]{s.name}[/bold] [dim]({s.scope} · {s.source})[/dim]  "
                f"present in {s.sessions_present} session"
                f"{'s' if s.sessions_present != 1 else ''}, "
                f"[yellow]{s.invocations}[/yellow] invocations"
            )
            # Dollars only when a priced model was actually observed for this
            # server. None means no rate was available, and printing $0.00
            # there would read as "this costs nothing".
            if pricing_mode == "api" and s.estimated_tax_usd_window is not None:
                tax = (
                    f"~{format_tokens(s.estimated_tax_tokens_window)} tokens / "
                    f"{format_cost(s.estimated_tax_usd_window)} in this window "
                    f"[dim](estimated, priced at {s.priced_model})[/dim]"
                )
            else:
                tax = (
                    f"~{format_tokens(s.estimated_tax_tokens_window)} tokens in "
                    f"this window [dim](estimated; no priced model observed for "
                    f"this server, so no dollar figure)[/dim]"
                )
            console.print(f"          [dim]tax[/dim] {tax}")
            if s.tax_construction:
                console.print(f"          [dim]{s.tax_construction}[/dim]")
            console.print(f"          [yellow]→[/yellow] {s.fix}")

    _render_deadweight_plugins(finding, pricing_mode=pricing_mode)

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
        _render_prose(finding.estimate_basis, style="dim", escape=False)
    # The MEASUREMENT coverage note, beside the figure it qualifies. Without it
    # the terminal showed a priced dollar total for the servers that could be
    # measured and said nothing about the ones excluded — an undisclosed FLOOR
    # rendered as a total. The number was honest; the presentation was not.
    if getattr(finding, "measurement_note", ""):
        _render_prose(finding.measurement_note, marker="[yellow]![/yellow]", style="dim")
    if finding.coverage_note:
        _render_prose(finding.coverage_note, marker="[yellow]![/yellow]", style="dim")
    if finding.caveat:
        _render_prose(finding.caveat, marker="[yellow]![/yellow]", style="italic", escape=False)


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

    if pricing_mode == "api" and finding.past_overspend_usd is not None:
        console.print(
            f"     • [green]~{_usd_with_tokens(finding.past_overspend_usd, finding.past_overspend_tokens)}"
            f"[/green] estimated price difference over this window (the token count "
            f"is the size of the affected workload, not tokens freed) "
            f"[dim](the same work, billed at the Batch API's flat rate)[/dim]"
        )
    else:
        console.print(
            "     [dim]The Batch API's discount is an api-billed price lever, "
            "so no dollar figure is shown for this plan. The workload sizes "
            "above still say how much work fits the shape.[/dim]"
        )

    if finding.estimate_basis:
        _render_prose(finding.estimate_basis, style="dim", escape=False)
    if finding.friction:
        _render_prose(finding.friction, marker="[yellow]![/yellow]", style="italic", escape=False)


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
            _render_prose(note, style="dim")
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
            _render_prose(note, style="dim")

    # Recoverable figure: fed through framing.render_savings rather than a
    # hand-rolled pricing_mode branch, so it can't quietly disagree with the
    # same rule cost_proposal_verbs.py applies to every other recoverable
    # figure. Framed against this finding's OWN denominator (prompt_tokens_total,
    # not the window's four-token-type total) since that's the basis
    # repeat_share itself is measured against.
    framing = Framing(pricing_mode=pricing_mode, window_total_tokens=finding.prompt_tokens_total)
    recoverable = render_savings(
        finding.past_overspend_usd, finding.past_overspend_tokens, framing,
    )
    console.print()
    if recoverable != "—":
        console.print(f"     • [green]~{recoverable}[/green] estimated recoverable")
    elif pricing_mode == "api":
        console.print(
            "     [dim]No dollar figure: no priced example session for the "
            "cache_control lever.[/dim]"
        )
    if recoverable == "—" and getattr(finding, "compaction_avoidable_tokens", None):
        # The offload lever's token and dollar figures degrade together
        # (Critical Rule 28 corollary a), so when the pair is absent this is
        # the only token estimate left — and it prices a different, wider
        # lever, which is why it is labelled as that lever rather than shown
        # as the missing recoverable figure.
        console.print(
            f"     [dim]Compaction lever (separate estimate, no dollar figure): "
            f"~{format_tokens(finding.compaction_avoidable_tokens)} tokens "
            f"avoidable across every session with repeat volume.[/dim]"
        )
    if finding.estimate_basis:
        _render_prose(finding.estimate_basis, style="dim", escape=False)

    _render_prose(finding.caveat, marker="[yellow]![/yellow]", style="italic", escape=False)
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
    elif persona == "sdk":
        # No priced cache_control example: `finding.fix_compaction` is
        # `/compact`, a Claude Code interactive command an SDK caller cannot
        # run. Fall back to the persona-neutral call-site instruction instead
        # of falling through to the claude-code branch below.
        from tokenjam.core.optimize.analyzers.context_resend import (
            RESEND_SDK_TRIM_FIX,
        )
        console.print(f"     [bold]Fix:[/bold] {RESEND_SDK_TRIM_FIX}")
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
    else:  # persona in {"claude-code", "unknown"}
        # LEADS WITH THE DURABLE FIX, same as the Review inbox card for this
        # same finding. This branch used to print `fix_compaction` — so the
        # card led with the offload rule while the CLI led with `/compact`, for
        # one finding, and the CLI led with the weaker one. `COMPACTION_FIX`'s
        # own text disclaims it ("never fixes the pattern going forward — treat
        # it as immediate relief for an already-full session, not the durable
        # fix"), so the CLI was leading with something the constant itself says
        # is not the fix. Rendered through the SAME helper the card builds its
        # advise from, so the two cannot drift into different wordings again.
        from tokenjam.core.optimize.cost_proposals import compound_offload_fix

        console.print(
            f"     [bold]Fix:[/bold] "
            f"{compound_offload_fix({}, finding.fix_subagent_offload, finding.fix_rightsize)}"
        )
        if finding.fix_compaction:
            # Same secondary position the card gives it: immediate relief for
            # an already-full session, explicitly not the durable fix.
            console.print(
                f"     [dim]Immediate relief in an already-full session: "
                f"{finding.fix_compaction}[/dim]"
            )
        if finding.fix_cache_control:
            console.print(
                "     [dim]If you also run SDK agents against these models, "
                "the cache_control lever applies too:[/dim]"
            )
            console.print(
                finding.fix_cache_control, markup=False, highlight=False, soft_wrap=True,
            )


def _render_stream_usage(
    finding, *, pricing_mode: str = "api", marker: str = "",
) -> None:
    """
    Render the streaming usage gap — streamed calls whose token counts the
    provider never reported, so their spend is missing from every total.

    This is the one finding on this screen that is NOT a saving. The figure it
    carries is spend that already happened and was never recorded, so the copy
    below never says "recoverable" and the accounting note is printed verbatim
    rather than summarised: a data-quality number sitting among savings
    numbers is read as a saving unless it says otherwise every time.
    """
    console.print(_finding_header(marker, "Streaming usage gap:"))
    if not finding.call_sites:
        console.print(f"     [dim]{_rich_escape(finding.hint)}[/dim]" if finding.hint
                      else "     [dim]Every observed stream reported its token "
                           "usage — no measurement gap in this window.[/dim]")
        return

    console.print(
        f"     • [bold]{finding.streams_missing_usage}[/bold] of "
        f"[bold]{finding.streams_observed}[/bold] streamed calls closed without a "
        f"usage payload [dim](content was produced; no token counts were "
        f"reported)[/dim]"
    )
    if pricing_mode == "api" and finding.undercounted_usd is not None:
        console.print(
            f"     [dim]unrecorded spend[/dim] ~"
            f"{format_tokens(finding.undercounted_tokens)} tokens / "
            f"{format_cost(finding.undercounted_usd)} "
            f"[dim](estimated — see basis below)[/dim]"
        )
    elif finding.undercounted_tokens is not None:
        console.print(
            f"     [dim]unrecorded spend[/dim] ~"
            f"{format_tokens(finding.undercounted_tokens)} tokens "
            f"[dim](no dollar figure on this pricing mode)[/dim]"
        )

    for site in finding.call_sites[:5]:
        # Escape the analyzer-supplied values, never the markup around them:
        # a model id or agent name can contain brackets Rich would eat as a
        # style tag, but escaping the whole line prints the tags literally.
        label = _rich_escape(f"{site.provider}/{site.model or 'unknown model'}")
        agent = f" [dim]({_rich_escape(site.agent_id)})[/dim]" if site.agent_id else ""
        console.print(
            f"       [bold]{label}[/bold]{agent}  "
            f"{site.affected_calls} call"
            f"{'s' if site.affected_calls != 1 else ''} across "
            f"{site.sessions} session{'s' if site.sessions != 1 else ''}"
        )
        console.print(f"          [dim]{_rich_escape(site.derivation)}[/dim]")
        console.print(f"          [yellow]→[/yellow] {_rich_escape(site.remediation)}")
        console.print(
            site.remediation_snippet, markup=False, highlight=False, soft_wrap=True,
        )
    if len(finding.call_sites) > 5:
        console.print(
            f"       [dim]… and {len(finding.call_sites) - 5} more call site(s). "
            f"Full detail with [bold]tj optimize stream-usage --json[/bold].[/dim]"
        )

    if finding.estimate_basis:
        _render_prose(finding.estimate_basis, style="dim")
    _render_prose(finding.accounting_note, style="dim")


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
    "stream-usage": _render_stream_usage,
}
