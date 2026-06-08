"""
`tj tokenmaxx` — single-shot "how hard are you tokenmaxxing?" command.

Designed as a shareable, screenshottable artifact for social posts.

Reads the last 30 days of spend (via the existing cost-summary path, so it
works under both direct-DB and API-shim modes), classifies it into a tier,
computes a plan-multiplier when a subscription plan is declared in config,
and runs the downsize analyzer for the headline savings figure.

Honesty discipline: the tier names are intentionally ironic — escalating
labels for higher spend — but every tier output is paired with the downsize
savings figure on the next line, so the command is always actionable, not
just a flex. Score is the hook; savings is the payoff.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import click

from tokenjam.utils.formatting import console, format_cost
from tokenjam.utils.time_parse import parse_since, utcnow


# Tier table — monthly USD spend thresholds → (label, one-liner).
# Calibrated for individual coding-agent users. Order matters: walked
# bottom-up, first matching threshold wins.
@dataclass(frozen=True)
class Tier:
    threshold: float
    label: str
    emoji: str
    quip: str


_TIERS: list[Tier] = [
    Tier(0,    "TokenSipper",     "💧", "Are you even using AI?"),
    Tier(50,   "TokenModerator",  "🥱", "Mostly reasonable. Try harder."),
    Tier(200,  "TokenMaxxer",     "💸", "You're paying Anthropic's rent."),
    Tier(500,  "TokenChad",       "🔥", "You're paying their interns' rent too."),
    Tier(1500, "TokenGigaChad",   "🔥🔥", "Touch grass. Then run `tj optimize`."),
]


def _classify(monthly_spend: float) -> Tier:
    """Walk tiers high-to-low; first threshold the spend exceeds wins."""
    for tier in reversed(_TIERS):
        if monthly_spend >= tier.threshold:
            return tier
    return _TIERS[0]


@click.command("tokenmaxx")
@click.option("--since", default="30d", help="Window for analysis (default 30d).")
@click.option("--json", "output_json", is_flag=True,
              help="Emit machine-readable JSON.")
@click.pass_context
def cmd_tokenmaxx(ctx: click.Context, since: str, output_json: bool) -> None:
    """How hard are you TokenMaxxing? Find out in one command."""
    db = ctx.obj.get("db")
    config = ctx.obj.get("config")
    if db is None or config is None:
        raise click.ClickException("tokenmaxx requires a database connection.")

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    until_dt = utcnow()
    window_days = max((until_dt - since_dt).days, 1)

    # Pull spend + downsize savings via the same paths optimize uses, so this
    # works whether the daemon is up (API shim) or not (direct DuckDB).
    spend_usd, savings_usd, sessions = _fetch(db, config, since_dt, until_dt, since)

    # Normalise to monthly (the tier thresholds are monthly).
    monthly_spend = spend_usd * (30.0 / window_days)

    # Plan multiplier — only for subscription users with a declared plan.
    declared_plan = _config_declared_plan(config)
    plan_info = _plan_label_and_fee(declared_plan)
    multiplier = None
    if plan_info and plan_info[1]:
        multiplier = monthly_spend / plan_info[1]

    tier = _classify(monthly_spend)

    if output_json:
        click.echo(json.dumps({
            "window_days": window_days,
            "sessions": sessions,
            "spend_usd": round(spend_usd, 2),
            "monthly_spend_usd": round(monthly_spend, 2),
            "tier": tier.label,
            "tier_emoji": tier.emoji,
            "tier_quip": tier.quip,
            "plan_tier": declared_plan,
            "plan_label": plan_info[0] if plan_info else None,
            "plan_monthly_usd": plan_info[1] if plan_info else None,
            "plan_multiplier": round(multiplier, 1) if multiplier else None,
            "downsize_monthly_savings_usd": round(savings_usd, 2),
        }))
        return

    _render(
        tier=tier, spend_usd=spend_usd, monthly_spend=monthly_spend,
        window_days=window_days, sessions=sessions,
        plan_info=plan_info, multiplier=multiplier,
        savings_usd=savings_usd,
    )


# ───────────────────────────── data fetching ──────────────────────────────

def _fetch(db, config, since_dt, until_dt, since_str) -> tuple[float, float, int]:
    """
    Returns (spend_usd, monthly_savings_usd, sessions) over the window.

    Two code paths to handle the daemon-up case where db is an ApiBackend
    without a `.conn` attribute. The optimize report payload already includes
    the window total + downsize finding, so we can fetch it in one round trip
    rather than running a second cost query.
    """
    from tokenjam.core.optimize import build_report

    conn = getattr(db, "conn", None)
    if conn is None:
        # API-shim path
        from tokenjam.core.api_backend import ApiBackend
        if not isinstance(db, ApiBackend):
            raise click.ClickException(
                "tokenmaxx requires either a direct DuckDB connection or a "
                "running tj serve at the configured api.{host,port}."
            )
        report_dict = db.fetch_optimize_report(
            since=since_str,
            findings=["downsize"],
        )
        if report_dict.get("error") == "no_data":
            return 0.0, 0.0, 0
        window = report_dict.get("window") or {}
        downgrade = report_dict.get("downgrade") or {}
        return (
            float(window.get("total_cost_usd") or 0.0),
            float(downgrade.get("monthly_savings_usd") or 0.0),
            int(window.get("sessions") or 0),
        )

    # Direct-DB path
    row = conn.execute(
        "SELECT COUNT(*) FROM spans WHERE model IS NOT NULL"
    ).fetchone()
    if not row or not row[0]:
        return 0.0, 0.0, 0

    report = build_report(
        db=db, config=config,
        since=since_dt, until=until_dt,
        findings=["downsize"],
    )
    spend = float(report.window.total_cost_usd or 0.0)
    sessions = int(report.window.sessions or 0)
    savings = 0.0
    if report.downgrade is not None:
        savings = float(report.downgrade.monthly_savings_usd or 0.0)
    return spend, savings, sessions


# ───────────────────────────── helpers ────────────────────────────────────

def _config_declared_plan(config) -> str | None:
    """Mirror cmd_optimize._config_declared_plan."""
    budgets = getattr(config, "budgets", None) or {}
    for provider in sorted(budgets.keys()):
        plan = getattr(budgets[provider], "plan", None)
        if plan:
            return str(plan)
    return None


def _plan_label_and_fee(plan_tier: str | None) -> tuple[str, float | None] | None:
    """
    Mirror cmd_optimize._PLAN_LABEL_AND_FEE without importing it (avoid
    circular import risk and keep tokenmaxx independently testable).
    """
    if plan_tier is None:
        return None
    table: dict[str, tuple[str, float | None]] = {
        "pro":        ("Pro plan",          20.0),
        "max_5x":     ("Max 5x plan",      100.0),
        "max_20x":    ("Max 20x plan",     200.0),
        "plus":       ("ChatGPT Plus",      20.0),
        "team":       ("ChatGPT Team",       None),
        "enterprise": ("ChatGPT Enterprise", None),
    }
    return table.get(plan_tier)


# ───────────────────────────── rendering ──────────────────────────────────

def _render(
    *, tier: Tier, spend_usd: float, monthly_spend: float,
    window_days: int, sessions: int,
    plan_info: tuple[str, float | None] | None,
    multiplier: float | None,
    savings_usd: float,
) -> None:
    """Big-headline render. Designed to be a clean screenshot artifact."""
    if sessions == 0:
        console.print(
            "\n[yellow]No usage data found.[/yellow]\n"
            "[dim]Run [bold]tj onboard --claude-code[/bold] to ingest your existing "
            "Claude Code sessions, or wait for new spans to land.[/dim]\n"
        )
        return

    # Banner — the headline shareable line.
    console.print()
    console.print(f"  {tier.emoji} [bold]You're a {tier.label}.[/bold]")
    console.print(f"     [dim italic]\"{tier.quip}\"[/dim italic]")
    console.print()

    # Spend breakdown — what produced the tier.
    actual_label = f"last {window_days}d" if window_days < 30 else "last 30d"
    if window_days == 30:
        console.print(
            f"  [bold]{format_cost(spend_usd)}[/bold] in {actual_label} "
            f"across [bold]{sessions}[/bold] sessions."
        )
    else:
        console.print(
            f"  [bold]{format_cost(spend_usd)}[/bold] in {actual_label} "
            f"across [bold]{sessions}[/bold] sessions "
            f"(≈ [bold]{format_cost(monthly_spend)}/mo[/bold] at this rate)."
        )

    # Plan multiplier — the punchline.
    if plan_info and multiplier:
        plan_label, fee = plan_info
        console.print(
            f"  That's [bold]{multiplier:.1f}×[/bold] your "
            f"[bold]{plan_label}[/bold] cost ({format_cost(fee)}/mo flat)."
        )
    elif plan_info:
        plan_label, _ = plan_info
        console.print(f"  Plan: [bold]{plan_label}[/bold].")

    console.print()

    # The action — the part that makes this a tool, not a flex.
    if savings_usd > 0:
        console.print(
            f"  💡 [bold]{format_cost(savings_usd)}/mo[/bold] of that looks "
            f"recoverable. Run [bold]tj optimize[/bold] to see candidates."
        )
    else:
        console.print(
            "  💡 No obvious savings flagged yet — run [bold]tj optimize[/bold] "
            "for the full report once you have more data."
        )
    console.print()

    # Subtle share prompt.
    console.print(
        "  [dim]Share your tier: screenshot the above and post with #tokenmaxx[/dim]"
    )
    console.print()
