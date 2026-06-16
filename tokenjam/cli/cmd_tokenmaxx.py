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

from tokenjam.utils.formatting import console, format_cost  # noqa: F401  (kept for back-compat imports)
from tokenjam.utils.time_parse import parse_since, utcnow


# Tier table — multiplier thresholds (×plan-fee) → (label, one-liner).
# The `threshold` field is interpreted as a multiplier when the user has a
# subscription plan declared, or as an absolute USD/mo amount when they
# don't (API users, no plan set). The fallback thresholds in _SPEND_TIERS
# below mirror the multiplier ladder calibrated against the Max-5x plan
# ($100/mo) so the tier names mean roughly the same thing across paths.
# Order matters: classify walks high-to-low; first match wins.
@dataclass(frozen=True)
class Tier:
    threshold: float
    label: str
    emoji: str
    quip: str


_TIERS: list[Tier] = [
    Tier(0,  "TokenSipper",      "💧",      "Are you even using AI?"),
    Tier(1,  "TokenModerator",   "🥱",      "Mostly reasonable. Try harder."),
    Tier(4,  "TokenMaxxer",      "💸",      "You're paying Anthropic's rent."),
    Tier(10, "TokenSuperMaxxer", "🔥",      "You're paying their interns' rent too."),
    Tier(20, "TokenMegaMaxxer",  "🔥🔥",    "Touch grass. Then run `tj optimize`."),
    Tier(50, "TokenGigaMaxxer",  "🔥🔥🔥",  "Anthropic's CFO knows your name."),
]

# Absolute USD/mo fallback for users without a subscription plan (API users).
# Calibrated against Max-5x = $100/mo: each threshold is the multiplier × $100.
# This way a $400/mo API user and a 4× Pro/Max-5x/Max-20x user both end up
# in TokenMaxxer — the tier name reflects "shocking spend" in either world.
_SPEND_TIER_THRESHOLDS_USD: list[float] = [0, 100, 400, 1000, 2000, 5000]


def _classify(monthly_spend: float, multiplier: float | None = None) -> Tier:
    """
    Pick a tier. Prefer the multiplier path when the user has a subscription
    plan with a declared fee; fall back to absolute monthly spend when not.
    Walks tiers high-to-low; first matching threshold wins.
    """
    if multiplier is not None:
        for tier in reversed(_TIERS):
            if multiplier >= tier.threshold:
                return tier
        return _TIERS[0]
    # API / no-plan path — map onto the same tier labels via absolute USD.
    for tier, threshold_usd in zip(reversed(_TIERS), reversed(_SPEND_TIER_THRESHOLDS_USD)):
        if monthly_spend >= threshold_usd:
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

    tier = _classify(monthly_spend, multiplier)

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
    """Return the user's declared subscription plan tier.

    Checks the active config first; if no `[budget.<provider>].plan` is set
    (common when running from a project dir whose `.tj/config.toml` has no
    `[budget]` section), falls back to peeking at the global config at
    `~/.config/tj/config.toml`. Without this fallback, tokenmaxx silently
    rendered api-pricing framing in subdirectories even when the user had
    set their plan globally via `tj onboard`. Issue #106.
    """
    budgets = getattr(config, "budgets", None) or {}
    for provider in sorted(budgets.keys()):
        plan = getattr(budgets[provider], "plan", None)
        if plan:
            return str(plan)

    # Active config has no plan — peek at the global config file directly.
    try:
        import sys
        from pathlib import Path
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-redef]
        global_path = Path.home() / ".config" / "tj" / "config.toml"
        if not global_path.exists():
            return None
        with open(global_path, "rb") as f:
            raw = tomllib.load(f)
        budget_block = raw.get("budget") or {}
        for provider in sorted(budget_block.keys()):
            plan = (budget_block[provider] or {}).get("plan")
            if plan:
                return str(plan)
    except (OSError, Exception):  # noqa: BLE001
        return None
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

def _fmt_spend(usd: float) -> str:
    """Spend / savings: always 2 decimals — readable, screenshot-friendly.

    The default `format_cost` helper uses 4 decimals (precision for the cost
    engine internals); for the tokenmaxx social artifact we want $4044.57,
    not $4044.5774.
    """
    return f"${usd:.2f}"


def _fmt_fee(usd: float) -> str:
    """Plan fee: drop the decimals when the fee is a round dollar.

    Anthropic / OpenAI subscription tiers (Pro $20, Max-5x $100, Max-20x $200)
    are all whole-dollar — `$100.00` reads worse than `$100`. Falls back to
    2 decimals for anything fractional.
    """
    if usd == int(usd):
        return f"${int(usd)}"
    return f"${usd:.2f}"


def _render(
    *, tier: Tier, spend_usd: float, monthly_spend: float,
    window_days: int, sessions: int,
    plan_info: tuple[str, float | None] | None,
    multiplier: float | None,
    savings_usd: float,
) -> None:
    """
    Big-headline render. Designed to be a clean screenshot artifact:
    bordered Panel with a heading, the tier callout up top, the spend
    breakdown in the middle, the action line at the bottom, and the
    share prompt OUTSIDE the panel.
    """
    if sessions == 0:
        console.print(
            "\n[yellow]No usage data found.[/yellow]\n"
            "[dim]Run [bold]tj onboard --claude-code[/bold] to ingest your existing "
            "Claude Code sessions, or wait for new spans to land.[/dim]\n"
        )
        return

    # Build the inside-the-panel content as one rich Text/markup string.
    # Rich Panel doesn't have native sub-spacing primitives, so we hand-pad
    # with newlines to get the visual rhythm we want in the screenshot.
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align

    # Tier callout — the headline, plus a larger non-dim quip (no quotes)
    # with `tj optimize` highlighted green-bold when it appears in the quip.
    headline = Text()
    headline.append(f"{tier.emoji} ", style="")
    headline.append("You're a ", style="bold")
    headline.append(tier.label, style="bold")
    headline.append(".", style="bold")

    quip_text = Text()
    # Walk the quip and recolor any `tj optimize` backtick-wrapped token green.
    # Rich doesn't auto-parse backticks, so we do it manually with split.
    parts = tier.quip.split("`tj optimize`")
    for i, p in enumerate(parts):
        if p:
            quip_text.append(p, style="")
        if i < len(parts) - 1:
            quip_text.append("tj optimize", style="bold green")

    # The spend / multiplier block — same content as before but with the
    # cleaner formatters and a slightly tighter line structure.
    body = Text()
    actual_label = f"last {window_days}d" if window_days < 30 else "last 30d"
    body.append(_fmt_spend(spend_usd), style="bold")
    body.append(f" in {actual_label} across ")
    body.append(str(sessions), style="bold")
    body.append(" sessions.")
    if window_days != 30:
        body.append("  (≈ ", style="dim")
        body.append(_fmt_spend(monthly_spend), style="bold")
        body.append("/mo at this rate)", style="dim")

    if plan_info and multiplier:
        plan_label, fee = plan_info
        body.append("\nThat's ")
        body.append(f"{multiplier:.1f}×", style="bold")
        body.append(" your ")
        body.append(plan_label, style="bold")
        body.append(f" cost ({_fmt_fee(fee)}/mo flat).")
    elif plan_info:
        plan_label, _ = plan_info
        body.append("\nPlan: ")
        body.append(plan_label, style="bold")
        body.append(".")

    # Action line — savings recoverable, or fall through to "no obvious
    # savings yet". `tj optimize` rendered green-bold either way so the
    # eye lands on the verb.
    action = Text("💡 ")
    if savings_usd > 0:
        action.append(_fmt_spend(savings_usd) + "/mo", style="bold")
        action.append(" of that looks recoverable. Run ")
        action.append("tj optimize", style="bold green")
        action.append(" to see candidates.")
    else:
        action.append("No obvious savings flagged yet — run ")
        action.append("tj optimize", style="bold green")
        action.append(" for the full report once you have more data.")

    # Compose with deliberate vertical spacing.
    panel_body = Group(
        headline,
        Text(""),            # blank line under headline
        Align.left(quip_text),
        Text(""),            # blank line under quip
        body,
        Text(""),            # blank line before action
        action,
    )

    console.print()
    console.print(Panel(
        panel_body,
        title="[bold]TokenJam TokenMaxxing Report[/bold]",
        title_align="left",
        border_style="dim",
        padding=(1, 2),
    ))

    # Share prompt — outside the panel, teal, points at the brand handle so
    # the social mechanic routes to a real account we can amplify from.
    console.print(
        "  [cyan]Share your tier: screenshot the above and tag "
        "[bold]@tokenjamdev[/bold][/cyan]"
    )
    console.print()
