"""
`tj tokenmaxx` — the shareable quota/efficiency card.

Designed as a screenshottable artifact for social posts ("quota Wrapped"),
reframed for June-2026 sentiment: the cultural tide turned from *tokenmaxxing*
(brag about spend) to *tokenminimizing* (brag about efficiency). A spend-brag
card now repels; an efficiency/quota card aligns — and it's quota-native, so it
works for the subscription majority who have no dollar spend at all.

The card leads with the **context-composition** headline from the #4 diagnostic:
what fraction of your quota went to *overhead* (re-reading history, CLAUDE.md,
accumulated tool output) versus *real work* (uncached input + output). The hook
is "how lean is your context?"; the payoff is "here's what you reclaimed and how
to reclaim more (`tj context` / `tj optimize`)".

Framing is quota-native (mirrors `tj quota-audit`, #5): the headline is always a
token share / "% of cycle tokens"; dollar figures are demoted to a secondary
API-only line and suppressed for subscription / local / unknown plans. No dollar
spend-brag for subscription users.

Honesty discipline (CLAUDE.md Rule 14): the efficiency number is a *measured*
token share, never a guaranteed saving. Re-read tokens are real billed quota
(cache reads at a reduced rate, not free).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import click

from tokenjam.cli.data_access import resolve_data_access
from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.context_diagnostic import ContextDiagnostic
from tokenjam.core.framing import Framing
from tokenjam.utils.formatting import console, format_tokens
from tokenjam.utils.time_parse import parse_since


# Efficiency-tier table — keyed on the *overhead* (re-read) share of the
# window, low-to-high. Lower overhead = leaner context = the thing to brag
# about in the tokenminimizing era. The labels celebrate efficiency, not spend.
# Order matters: classify walks low overhead → high overhead; first match wins.
@dataclass(frozen=True)
class Tier:
    max_overhead: float  # upper bound (inclusive) of re-read share for this tier
    label: str
    emoji: str
    quip: str


# Boundaries chosen against the diagnostic's HIGH_REREAD_SHARE = 0.80 signal:
# steady-state CC turns drift well past 80% overhead once history + CLAUDE.md
# grow, so the leaner bands are the achievement.
_TIERS: list[Tier] = [
    Tier(0.30, "TokenMinimizer",   "🧘", "Surgical context. Teach the rest of us."),
    Tier(0.50, "LeanOperator",     "🌿", "Tight loop. Most of your quota does real work."),
    Tier(0.70, "SteadyState",      "⚖️",  "Normal drift. A `/compact` habit keeps it lean."),
    Tier(0.85, "ContextHeavy",     "🪨", "Overhead is winning. Run `tj context`."),
    Tier(1.01, "QuotaSink",        "🕳️",  "Re-reading is eating your quota. Run `tj context`."),
]


def _classify(overhead_share: float) -> Tier:
    """Pick an efficiency tier from the overhead (re-read) share.

    Walks tiers lean → heavy; the first tier whose ``max_overhead`` the share
    falls at or below wins. The final tier's bound is >1.0 so every share lands.
    """
    for tier in _TIERS:
        if overhead_share <= tier.max_overhead:
            return tier
    return _TIERS[-1]


@click.command("tokenmaxx")
@click.option("--agent", default=None, help="Filter to a specific agent_id.")
@click.option("--since", default="30d", help="Window for analysis (default 30d).")
@click.option("--weekly", is_flag=True,
              help="Weekly 'quota Wrapped' recap mode (last 7 days, recap copy).")
@json_option
@click.pass_context
def cmd_tokenmaxx(ctx: click.Context, agent: str | None, since: str,
                  weekly: bool, output_json_flag: bool) -> None:
    """Your quota/efficiency card: how lean is your context? (screenshottable)"""
    output_json = resolve_output_json(ctx, output_json_flag)
    db = ctx.obj.get("db")
    config = ctx.obj.get("config")
    agent = agent or ctx.obj.get("agent")
    if db is None or config is None:
        raise click.ClickException("tokenmaxx requires a database connection.")

    # --weekly is a recap preset: a 7-day window with recap framing. An explicit
    # --since still wins if the user narrows it themselves.
    if weekly and since == "30d":
        since = "7d"

    # Validate the window up-front so the direct and serve paths give the same
    # error message.
    try:
        parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    # One seam, two backends: a direct DuckDB connection when no daemon runs,
    # else the compute routed through the running `tj serve` (which holds the DB
    # write lock). No `hasattr(db, "conn")` sniffing — the seam owns that choice.
    data = resolve_data_access(ctx)
    diag, framing = data.context_diagnostic(since=since, agent_id=agent)

    overhead_share = diag.reread_share
    work_share = (
        diag.total_work_tokens / diag.total_tokens if diag.total_tokens else 0.0
    )
    tier = _classify(overhead_share)

    if output_json:
        click.echo(json.dumps({
            "since": since,
            "weekly": weekly,
            "sessions": diag.sessions,
            "turns": diag.turns,
            "overhead_share": round(overhead_share, 4),
            "work_share": round(work_share, 4),
            "total_reread_tokens": diag.total_reread_tokens,
            "total_work_tokens": diag.total_work_tokens,
            "total_tokens": diag.total_tokens,
            "tier": tier.label,
            "tier_emoji": tier.emoji,
            "tier_quip": tier.quip,
            "plan_tier": framing.plan_tier,
            "pricing_mode": framing.pricing_mode,
            "plan_label": framing.plan_label,
        }, default=str))
        return

    _render(
        diag=diag, framing=framing, tier=tier,
        overhead_share=overhead_share, work_share=work_share,
        since=since, weekly=weekly,
    )


# ───────────────────────────── helpers ────────────────────────────────────

def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _quota_share(tokens: int, framing: Framing) -> str:
    """Render a token figure as a quota share on a subscription plan.

    Mirrors ``cmd_context._quota_share``: subscription users see "X% of cycle
    tokens" against the window total — the quota-native headline. API / local /
    unknown users see the raw token count (dollars surfaced separately, if at
    all).
    """
    total = framing.window_total_tokens
    if framing.pricing_mode == "subscription" and total > 0:
        return f"{100.0 * tokens / total:.1f}% of cycle tokens"
    return f"{format_tokens(tokens)} tokens"


# ───────────────────────────── rendering ──────────────────────────────────

def _render(
    *, diag: ContextDiagnostic, framing: Framing, tier: Tier,
    overhead_share: float, work_share: float, since: str, weekly: bool,
) -> None:
    """
    Big-headline render. A clean PNG-screenshot artifact: a bordered Panel with
    the efficiency tier callout up top, the overhead-vs-work composition in the
    middle, the reclaimed figure, and the action line at the bottom. The share
    prompt sits OUTSIDE the panel. Readable without any surrounding context.
    """
    if not diag.has_data:
        console.print(
            "\n[yellow]No Claude Code turns found in this window.[/yellow]\n"
            "[dim]Run [bold]tj onboard --claude-code[/bold] to ingest your "
            "existing Claude Code sessions, then re-run "
            "[bold]tj tokenmaxx[/bold].[/dim]\n"
        )
        return

    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    window_label = "this week" if weekly else f"last {since}"

    # ── Tier callout — the headline + a non-dim quip with `tj context` /
    #    `tj optimize` recolored green-bold where they appear. ──
    headline = Text()
    headline.append(f"{tier.emoji} ", style="")
    headline.append("You're a ", style="bold")
    headline.append(tier.label, style="bold")
    headline.append(".", style="bold")

    quip_text = _highlight_commands(tier.quip)

    # ── Composition block — overhead vs real work, quota-framed. ──
    body = Text()
    body.append(f"{_pct(overhead_share)} ", style="bold red")
    body.append("of your quota went to ")
    body.append("overhead", style="bold")
    body.append(" — re-reading history, CLAUDE.md, tool output.", style="dim")
    body.append("\n")
    body.append(f"{_pct(work_share)} ", style="bold green")
    body.append("did ")
    body.append("real work", style="bold")
    body.append(" — uncached input + output.", style="dim")

    # Quota-native breakdown (subscription → "% of cycle"; else token counts).
    body.append("\n\nOverhead:  ", style="dim")
    body.append(_quota_share(diag.total_reread_tokens, framing), style="bold")
    body.append(f"  ({format_tokens(diag.total_reread_tokens)} cache reads)",
                style="dim")
    body.append("\nReal work: ", style="dim")
    body.append(_quota_share(diag.total_work_tokens, framing), style="bold")
    body.append(f"  ({format_tokens(diag.total_work_tokens)} tokens)",
                style="dim")
    body.append(
        f"\n\nAcross {diag.sessions} sessions, {diag.turns} turns ({window_label}).",
        style="dim",
    )

    # Secondary implied-dollar calibration for API users ONLY — never the
    # headline, suppressed for subscription / local / unknown (mirrors #5).
    if framing.pricing_mode == "api" and diag.total_cost_usd > 0:
        body.append("\nImplied API value: ", style="dim")
        body.append(f"${diag.total_cost_usd:,.2f}", style="bold")
        body.append(" over the window (calibration only)", style="dim")

    # ── Action line — what you can reclaim, never a guaranteed saving. ──
    action = Text("💡 ")
    if diag.compact_candidates:
        reclaimable = sum(c.reread_tokens for c in diag.compact_candidates)
        action.append(_quota_share(reclaimable, framing), style="bold")
        action.append(" sits in ")
        action.append(f"{len(diag.compact_candidates)}", style="bold")
        action.append(" compactable session"
                      f"{'s' if len(diag.compact_candidates) != 1 else ''}. Run ")
        action.append("tj context", style="bold green")
        action.append(" to see where to reclaim it.")
    elif overhead_share >= 0.50:
        action.append("Run ")
        action.append("tj context", style="bold green")
        action.append(" to see exactly which files are re-read every session.")
    else:
        action.append("Lean context. Run ")
        action.append("tj optimize", style="bold green")
        action.append(" for the full savings report.")

    # ── Prove line — nudge to verify the savings hold, never "guaranteed". ──
    prove = Text("🔬 ")
    prove.append("Prove your savings hold: ", style="dim")
    prove.append("pip install tokenjam-bench", style="bold green")

    panel_body = Group(
        headline,
        Text(""),
        Align.left(quip_text),
        Text(""),
        body,
        Text(""),
        action,
        prove,
    )

    title = ("[bold]TokenJam Quota Wrapped — Weekly Recap[/bold]" if weekly
             else "[bold]TokenJam Quota / Efficiency Card[/bold]")
    console.print()
    console.print(Panel(
        panel_body,
        title=title,
        title_align="left",
        border_style="dim",
        padding=(1, 2),
    ))

    # Qualifier banner (plan-tier framing) + honesty caveat, below the panel.
    if framing.qualifier_text:
        console.print(f"  [dim]{framing.qualifier_text}[/dim]")
    console.print(f"  [dim]{diag.caveat}[/dim]")

    # Share prompt — outside the panel, teal, points at the brand handle.
    console.print(
        "  [cyan]Share your card: screenshot the above and tag "
        "[bold]@tokenjamdev[/bold][/cyan]"
    )
    console.print()


def _highlight_commands(quip: str):
    """Recolor backtick-wrapped `tj context` / `tj optimize` tokens green-bold.

    Rich doesn't auto-parse backticks, so we split on the known command tokens
    and re-style them. Any other backticked token is left as plain text.
    """
    from rich.text import Text

    text = Text()
    remaining = quip
    commands = ("`tj context`", "`tj optimize`", "`/compact`")
    while remaining:
        # Find the earliest command token in the remaining string.
        idx, hit = min(
            ((remaining.find(c), c) for c in commands if c in remaining),
            default=(-1, ""),
        )
        if idx < 0:
            text.append(remaining)
            break
        if idx:
            text.append(remaining[:idx])
        text.append(hit.strip("`"), style="bold green")
        remaining = remaining[idx + len(hit):]
    return text
