"""``tj quota-audit`` — the retroactive premium-tier quota audit over Claude Code sessions.

The accountability companion to ``opusplan`` / ``/model`` (issue #5). Those are
forward-looking and say nothing about session history; nothing answers the
backward-looking question "which of my PAST premium (Opus/Fable) sessions were
Sonnet-shaped?". This command does: it runs the structural downsize heuristic
(:func:`tokenjam.core.optimize.analyzers.model_downgrade.audit_opus_quota`)
retroactively, scoped to premium-tier sessions (Fable + Opus), and reports:

  * the headline — **% of your premium (Opus/Fable) quota that went to
    Sonnet-shaped sessions** (a retrospective behaviour mirror in premium token
    share — the tokens are already spent, so nothing is "reclaimed"; and never a
    dollar "saving" — the subscription majority is on a flat fee, so dollar
    framing mis-targets them, see
    ``research/evidence/subscription-vs-cost-framing.md``);
  * the specific example sessions to **spot-check**;
  * an optional tuned routing-config export (``--export-config claude-code``).

Framing is an *audit* (quota language, not dollars) and the honesty caveat —
"candidates to spot-check, never safe-to-downgrade" — is always visible. The
audit computes purely from already-backfilled token/model metadata; it does NOT
depend on captured content (#3).

Needs a direct DuckDB connection (it aggregates per-session token/model
metadata the API shim does not expose at this grain).
"""
from __future__ import annotations

import json
from typing import Any

import click

from tokenjam.cli.data_access import resolve_data_access
from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.framing import Framing
from tokenjam.core.model_tiers import PREMIUM_TIER_LABEL
from tokenjam.core.optimize.types import (
    DowngradeFinding,
    OpusQuotaAudit,
    audit_to_dict,
)
from tokenjam.utils.formatting import console, format_cost, format_tokens
from tokenjam.utils.time_parse import parse_since


@click.command("quota-audit")
@click.option("--agent", default=None, help="Filter to a specific agent_id.")
@click.option("--since", default="30d",
              help="Window for the audit (e.g. 7d, 30d, 2026-03-01). Default 30d.")
@click.option("--export-config", "export_target", default=None,
              type=click.Choice(["claude-code"]),
              help="Write a tuned routing snippet (currently claude-code) under "
                   "the TokenJam config directory. Does not modify any external "
                   "config — you merge it manually.")
@json_option
@click.pass_context
def cmd_quota_audit(ctx: click.Context, agent: str | None, since: str,
                    export_target: str | None, output_json_flag: bool) -> None:
    """Audit your Opus quota: which past Opus sessions were Sonnet-shaped?"""
    output_json = resolve_output_json(ctx, output_json_flag)
    db = ctx.obj.get("db")
    config = ctx.obj.get("config")
    agent = agent or ctx.obj.get("agent")
    if db is None or config is None:
        raise click.ClickException("quota-audit requires a database connection.")

    # Validate the window up-front so the direct and serve paths agree on the
    # error message.
    try:
        parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    # One seam, two backends: a direct DuckDB connection when no daemon runs,
    # else the compute routed through the running `tj serve` (which holds the DB
    # write lock). No `hasattr(db, "conn")` sniffing — the seam owns that choice.
    data = resolve_data_access(ctx)
    audit, framing = data.quota_audit(since=since, agent_id=agent)

    if export_target:
        _export_snippet(audit, framing, target=export_target, agent_id=agent,
                        output_json=output_json)
        return

    if output_json:
        payload = audit_to_dict(audit)
        payload["framing"] = framing.to_dict()
        click.echo(json.dumps(payload, default=str))
        return

    _render(audit, framing, since=since)


# ───────────────────────────── rendering ──────────────────────────────────


def _headline_nudge(audit: OpusQuotaAudit, framing: Framing) -> Any:
    """Persona-specific takeaway under the measured misallocation headline.

    - API-billed with pricing data: the retrospective counterfactual in dollars
      — the same work priced at the suggested cheaper tiers, labelled as
      already-billed so it never reads as a future "saving".
    - API-billed with NO pricing data (every candidate's model is absent from
      the pricing table, so ``actual_cost_usd == 0``): a neutral routing prompt.
      API billing has no rolling quota window, so the subscription "next window"
      language below would be actively wrong for these users.
    - Subscription / local (and the unknown-plan default): a habit nudge — that
      quota share is still available for hard problems next window.
    """
    from rich.text import Text

    if framing.pricing_mode == "api":
        if audit.actual_cost_usd > 0:
            line = Text("\n→ ", style="dim")
            line.append(
                f"Same work at the suggested tiers ≈ "
                f"{format_cost(audit.alternative_cost_usd)} vs your "
                f"{format_cost(audit.actual_cost_usd)} actual "
                "(retrospective — already billed).",
                style="dim",
            )
            return line
        # API mode but no pricing data — stay neutral, never subscription quota
        # language (there is no rolling window to hold that share for).
        line = Text("\n→ ", style="dim")
        line.append(
            "Review these sessions before adjusting your routing.",
            style="dim",
        )
        return line
    # subscription / local / unknown — no dollars; behaviour-mirror habit nudge.
    line = Text("\n→ ", style="green")
    line.append(
        "That share stays available for hard problems next window. Use "
        "`/model sonnet` for mechanical work and right-size your subagents.",
        style="green",
    )
    return line


def _render(audit: OpusQuotaAudit, framing: Framing, *, since: str) -> None:
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    if not audit.has_opus:
        console.print(
            f"\n[yellow]No premium ({PREMIUM_TIER_LABEL}) sessions found in this "
            "window.[/yellow]\n"
            f"[dim]The quota audit only inspects premium-tier ({PREMIUM_TIER_LABEL}) "
            "sessions (where quota burn is acute). Run "
            "[bold]tj onboard --claude-code[/bold] to ingest your Claude Code "
            "sessions, then re-run [bold]tj quota-audit[/bold].[/dim]\n"
        )
        return

    sections: list[Any] = []

    headline = Text()
    headline.append("Premium quota audit", style="bold")
    headline.append(
        f"  ·  {audit.opus_sessions} premium session"
        f"{'s' if audit.opus_sessions != 1 else ''} "
        f"({format_tokens(audit.opus_tokens)} {PREMIUM_TIER_LABEL} tokens, "
        f"last {since})",
        style="dim",
    )
    sections.append(headline)
    sections.append(Text(""))

    if audit.candidate_sessions == 0:
        clean = Text()
        clean.append("0% misallocated", style="bold green")
        clean.append(
            " — none of your premium sessions match the Sonnet-shaped structure "
            "(small input/output, few tool calls).",
            style="",
        )
        sections.append(clean)
    else:
        # The headline is a retrospective behaviour mirror: what SHARE of premium
        # quota already WENT to Sonnet-shaped sessions. It is misallocated, not
        # "reclaimable" — those tokens are spent. Quota share, never dollars for
        # the subscription majority (audit framing).
        big = Text()
        big.append(f"~{audit.percent_quota_misallocated:.0f}% ", style="bold red")
        big.append(
            f"of your premium ({PREMIUM_TIER_LABEL}) quota went to "
            "Sonnet-shaped work",
            style="bold",
        )
        # The number is a single labelled ESTIMATE (founder D1/D3): the "estimate"
        # tag + a wide bootstrap CI (resampled segments) so it never reads as a
        # settled figure. Below 2 segments there's no spread to bracket.
        est = Text("\n", style="dim")
        est.append(f"{audit.segment_estimate_confidence}", style="italic yellow")
        if audit.segment_ci_low is not None and audit.segment_ci_high is not None:
            est.append(
                f" · 95% CI {audit.segment_ci_low:.0f}–{audit.segment_ci_high:.0f}%",
                style="dim",
            )
        else:
            est.append(
                " · single stretch — interval too wide to bracket",
                style="dim",
            )
        est.append(
            f" · from {audit.segment_count} Sonnet-shaped "
            f"stretch{'es' if audit.segment_count != 1 else ''} across "
            f"{audit.candidate_sessions} of {audit.opus_sessions} premium "
            f"session{'s' if audit.opus_sessions != 1 else ''}",
            style="dim",
        )
        big.append(est)
        sections.append(big)

        detail = Text()
        detail.append("\nSonnet-shaped premium tokens:  ", style="dim")
        detail.append(format_tokens(audit.candidate_tokens), style="bold")
        detail.append(
            f"  of {format_tokens(audit.opus_tokens)} total", style="dim"
        )
        sections.append(detail)

        # Persona-gated takeaway. The measured mirror above is shown to everyone;
        # this is the actionable framing for the user's billing reality.
        sections.append(_headline_nudge(audit, framing))

        if audit.suggestions:
            pairs = ", ".join(
                f"{k} → {v}" for k, v in sorted(audit.suggestions.items())
            )
            pat = Text("\nPattern: ", style="dim")
            pat.append(pairs, style="dim")
            sections.append(pat)

    # ── Example sessions to spot-check. ──
    if audit.examples:
        sections.append(Text(""))
        ex_header = Text("Sessions to spot-check", style="bold")
        sections.append(ex_header)
        for ex in audit.examples:
            dur = f"{ex.duration_seconds:.0f}s" if ex.duration_seconds else "—"
            line = Text("  · ", style="dim")
            sid = ex.session_id or ex.trace_id
            line.append(f"{sid[:12]}", style="bold")
            line.append(
                f"  {ex.model}  "
                f"in {format_tokens(ex.input_tokens)} / "
                f"out {format_tokens(ex.output_tokens)} · "
                f"{ex.tool_calls} tool call{'s' if ex.tool_calls != 1 else ''} · "
                f"{dur}",
                style="dim",
            )
            sections.append(line)
        nudge = Text(
            "    → Open these in Claude Code and judge whether the smaller "
            "model would have sufficed before changing your routing.",
            style="green",
        )
        sections.append(nudge)

    panel_body = Group(*sections)
    console.print()
    console.print(Panel(
        panel_body,
        title="[bold]TokenJam Premium Quota Audit[/bold]",
        title_align="left",
        border_style="dim",
        padding=(1, 2),
    ))

    # Qualifier banner (plan-tier framing) + the mandatory honesty caveat.
    if framing.qualifier_text:
        console.print(f"  [dim]{framing.qualifier_text}[/dim]")
    console.print(f"  [yellow]![/yellow] [italic]{audit.caveat}[/italic]")
    if audit.candidate_sessions > 0:
        console.print(
            "  [dim]Export a tuned routing snippet with "
            "[bold]tj quota-audit --export-config claude-code[/bold].[/dim]"
        )
    console.print()


def _export_snippet(audit: OpusQuotaAudit, framing: Framing, *,
                    target: str, agent_id: str | None,
                    output_json: bool) -> None:
    """Write a routing snippet tuned to the audit's candidate patterns.

    Reuses the downsize snippet generator by adapting the audit into a
    ``DowngradeFinding`` shim carrying the observed model→alt suggestions and
    the misallocated token figure. No file outside the TokenJam config directory
    is touched — the user merges the snippet manually.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    from tokenjam.core.export.claude_code import render_claude_code_snippet

    shim = DowngradeFinding(
        candidate_sessions=audit.candidate_sessions,
        total_sessions=audit.opus_sessions,
        actual_cost_usd=audit.actual_cost_usd,
        alternative_cost_usd=audit.alternative_cost_usd,
        monthly_savings_usd=round(
            max(audit.actual_cost_usd - audit.alternative_cost_usd, 0.0), 2
        ),
        percent_of_sessions=audit.percent_sessions,
        examples=[],
        suggestions=dict(audit.suggestions),
        candidate_tokens=audit.candidate_tokens,
        window_total_tokens=audit.opus_tokens,
        percent_of_tokens=audit.percent_quota_misallocated,
        monthly_tokens_in_candidates=audit.candidate_tokens,
    )

    body = render_claude_code_snippet(
        downgrade=shim,
        pricing_mode=framing.pricing_mode,
        plan_tier=framing.plan_tier,
        agent_id=agent_id,
    )

    out_dir = Path.home() / ".config" / "tokenjam" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    out_path = out_dir / f"quota-audit-{target}-{today}.jsonc"
    out_path.write_text(body)

    if output_json:
        click.echo(json.dumps({
            "target": target,
            "path": str(out_path),
            "plan_tier": framing.plan_tier,
            "pricing_mode": framing.pricing_mode,
            "percent_quota_misallocated": audit.percent_quota_misallocated,
            # DEPRECATED alias (see audit_to_dict) — kept one release.
            "percent_quota_reclaimable": audit.percent_quota_misallocated,
        }, default=str))
        return

    console.print(
        f"[green]✓[/green] Routing snippet written to [bold]{out_path}[/bold]."
    )
    console.print(
        "\n[dim]These are spot-check candidates, not a verdict. Test on a few "
        "real sessions before trusting the rules broadly. TokenJam does not "
        "enforce them — you merge the block into your routing layer.[/dim]"
    )
