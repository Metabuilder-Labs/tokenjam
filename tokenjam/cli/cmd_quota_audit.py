"""``tj quota-audit`` — the retroactive Opus quota audit over Claude Code sessions.

The accountability companion to ``opusplan`` / ``/model`` (issue #5). Those are
forward-looking and say nothing about session history; nothing answers the
backward-looking question "which of my PAST Opus sessions were Sonnet-shaped?".
This command does: it runs the structural downsize heuristic
(:func:`tokenjam.core.optimize.analyzers.model_downgrade.audit_opus_quota`)
retroactively, scoped to Opus sessions, and reports:

  * the headline — **% of your Opus quota reclaimable from Sonnet-shaped
    sessions** (Opus token share, never a dollar "saving" — the subscription
    majority is on a flat fee; dollar framing mis-targets them, see
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
from tokenjam.core.framing import Framing
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
@click.option("--json", "output_json", is_flag=True,
              help="Emit machine-readable JSON.")
@click.pass_context
def cmd_quota_audit(ctx: click.Context, agent: str | None, since: str,
                    export_target: str | None, output_json: bool) -> None:
    """Audit your Opus quota: which past Opus sessions were Sonnet-shaped?"""
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


def _render(audit: OpusQuotaAudit, framing: Framing, *, since: str) -> None:
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    if not audit.has_opus:
        console.print(
            "\n[yellow]No Opus sessions found in this window.[/yellow]\n"
            "[dim]The quota audit only inspects Opus-family sessions (where "
            "quota burn is acute). Run [bold]tj onboard --claude-code[/bold] to "
            "ingest your Claude Code sessions, then re-run "
            "[bold]tj quota-audit[/bold].[/dim]\n"
        )
        return

    sections: list[Any] = []

    headline = Text()
    headline.append("Opus quota audit", style="bold")
    headline.append(
        f"  ·  {audit.opus_sessions} Opus session"
        f"{'s' if audit.opus_sessions != 1 else ''} "
        f"({format_tokens(audit.opus_tokens)} Opus tokens, last {since})",
        style="dim",
    )
    sections.append(headline)
    sections.append(Text(""))

    if audit.candidate_sessions == 0:
        clean = Text()
        clean.append("0% reclaimable", style="bold green")
        clean.append(
            " — none of your Opus sessions match the Sonnet-shaped structure "
            "(small input/output, few tool calls).",
            style="",
        )
        sections.append(clean)
    else:
        # The headline: quota share, never dollars (audit framing).
        big = Text()
        big.append(f"~{audit.percent_quota_reclaimable:.0f}% ", style="bold red")
        big.append("of your Opus quota is reclaimable", style="bold")
        big.append(
            f"\nfrom {audit.candidate_sessions} of {audit.opus_sessions} Opus "
            f"session{'s' if audit.opus_sessions != 1 else ''} "
            f"({audit.percent_sessions:.0f}%) that are Sonnet-shaped",
            style="dim",
        )
        sections.append(big)

        detail = Text()
        detail.append("\nReclaimable Opus tokens:  ", style="dim")
        detail.append(format_tokens(audit.candidate_tokens), style="bold")
        detail.append(
            f"  of {format_tokens(audit.opus_tokens)} total", style="dim"
        )
        # Secondary implied-dollar calibration for API users only — never the
        # headline. Suppressed for subscription / local / unknown plans.
        if (
            framing.pricing_mode == "api"
            and audit.actual_cost_usd > 0
        ):
            detail.append("\nImplied API value:        ", style="dim")
            detail.append(
                f"{format_cost(audit.actual_cost_usd)}", style="bold"
            )
            detail.append(
                f" → {format_cost(audit.alternative_cost_usd)} on the smaller "
                f"model (calibration only)",
                style="dim",
            )
        sections.append(detail)

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
        title="[bold]TokenJam Opus Quota Audit[/bold]",
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
    the reclaimable token figure. No file outside the TokenJam config directory
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
        percent_of_tokens=audit.percent_quota_reclaimable,
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
            "percent_quota_reclaimable": audit.percent_quota_reclaimable,
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
