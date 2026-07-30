from __future__ import annotations

import json
from datetime import timedelta

import click

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.models import Alert, AlertFilters
from tokenjam.utils.formatting import console, format_cost, format_tokens, status_icon
from tokenjam.utils.time_parse import utcnow

#: Window the recoverable teaser looks back over, matching `tj optimize`'s
#: own `--since` default so the figure means the same thing in both places.
_TEASER_WINDOW_DAYS = 30

#: Below this, "$X recoverable" reads as noise rather than a real pointer —
#: stay silent instead of printing a sub-dollar figure.
_TEASER_MIN_USD = 1.0


@click.command("status")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@json_option
@click.pass_context
def cmd_status(ctx: click.Context, agent: str | None, output_json_flag: bool) -> None:
    """Show agent status overview."""
    output_json = resolve_output_json(ctx, output_json_flag)
    db = ctx.obj["db"]
    agent_filter = agent or ctx.obj.get("agent")

    # Get all agents from recent sessions
    if agent_filter:
        agent_ids = [agent_filter]
    elif hasattr(db, "conn"):
        # Local DB mode
        agent_ids = db.get_distinct_agent_ids()
    else:
        # API mode — discover agents from recent traces
        from tokenjam.core.models import TraceFilters
        traces = db.get_traces(TraceFilters(limit=100))
        agent_ids = sorted({t.agent_id for t in traces if t.agent_id})

    if not agent_ids:
        if output_json:
            click.echo(json.dumps({"agents": [], "has_active_alerts": False}))
        else:
            console.print("[dim]No agents found. Run an instrumented agent first.[/dim]")
        return

    has_active_alerts = False
    agents_data = []

    for aid in agent_ids:
        session = None
        if hasattr(db, "conn"):
            # Local DB mode — prefer the live active session, fall back to the
            # most recent completed one.
            sessions = db.get_completed_sessions(aid, limit=1)
            session = db.get_active_session(aid)
            if session is None and sessions:
                session = sessions[0]
        else:
            # API mode — limited session info
            sessions = db.get_completed_sessions(aid, limit=1)
            if sessions:
                session = sessions[0]

        # Active (compute) time = sum of span durations; distinct from the
        # wall-clock duration_seconds, which spans days for resumed sessions
        # (issue #147).
        active_seconds = None
        if session is not None and hasattr(db, "conn"):
            active_seconds = db.get_session_active_seconds(session.session_id)

        today_cost = db.get_daily_cost(aid, utcnow().date())

        # Budget from config: per-agent overrides defaults
        config = ctx.obj["config"]
        agent_config = config.agents.get(aid)
        if agent_config and agent_config.budget.daily_usd is not None:
            daily_limit = agent_config.budget.daily_usd
        elif hasattr(config, "defaults") and config.defaults.budget.daily_usd is not None:
            daily_limit = config.defaults.budget.daily_usd
        else:
            daily_limit = None

        # Active alerts
        alerts = db.get_alerts(AlertFilters(agent_id=aid, unread=True, limit=50))
        active_alerts = [a for a in alerts if not a.acknowledged and not a.suppressed]
        if active_alerts:
            has_active_alerts = True

        agent_data = {
            "agent_id": aid,
            "status": session.status if session else "idle",
            "session_id": session.session_id if session else None,
            "cost_today": today_cost,
            "daily_limit": daily_limit,
            "input_tokens": session.input_tokens if session else 0,
            "output_tokens": session.output_tokens if session else 0,
            "tool_call_count": session.tool_call_count if session else 0,
            "error_count": session.error_count if session else 0,
            "active_alerts": len(active_alerts),
            "duration_seconds": session.duration_seconds if session else None,
            "active_seconds": active_seconds,
        }
        agents_data.append(agent_data)

        if not output_json:
            _print_agent_status(agent_data, active_alerts, session)

    # Count sessions with plan_tier='unknown' so the user knows to reconfigure.
    # Informational only — exit code stays driven by alert state.
    unknown_count = 0
    if hasattr(db, "conn"):
        try:
            unknown_count = db.count_unknown_plan_tier_sessions()
        except Exception:
            unknown_count = 0

    if output_json:
        click.echo(json.dumps({
            "agents": agents_data,
            "has_active_alerts": has_active_alerts,
            "unknown_plan_tier_sessions": unknown_count,
        }, default=str))
    else:
        if unknown_count > 0:
            console.print(
                f"[dim]Note: {unknown_count} session(s) have unknown plan tier. "
                f"Run [bold]tj onboard --claude-code --reconfigure[/bold] "
                f"(or [bold]--codex[/bold]) to set it.[/dim]"
            )
        teaser = _recoverable_teaser(db, ctx.obj.get("config"))
        if teaser:
            console.print(teaser)

    ctx.exit(1 if has_active_alerts else 0)


def _recoverable_teaser(db, config) -> str | None:
    """One-line `tj optimize` pointer for `tj status` — nothing in status,
    doctor, statusline or the banner ever mentioned optimize existed.

    Reuses the same recoverable-savings contract every analyzer already
    carries (`past_overspend_usd`, #111) rather than inventing a new
    figure, scoped to `COST_ANALYZERS` — the analyzers that actually feed the
    cost/apply rail (`cost_proposals.py`).

    Reports the LARGEST single analyzer's estimate rather than a sum across
    analyzers — mirroring `largest_recoverable_usd` in `api/routes/cost.py`.
    The analyzers price OVERLAPPING angles on the same spans (a cache fix and
    a downsize swap can both price the same call's waste), so summing them
    would double-count and print an inflated, non-additive headline; the
    single largest entry is honest standalone because it isn't a sum of
    anything (see `_recoverable_overlap_note` in `api/routes/cost.py` for the
    full rationale). Silent (returns None) whenever the figure wouldn't mean
    anything: no direct DB connection (daemon holds the lock), no usage, or a
    sub-$1 figure. Dollars are shown regardless of billing mode (product
    decision: no differentiated messaging between subscription and API
    users).
    """
    conn = getattr(db, "conn", None)
    if conn is None or config is None:
        return None
    try:
        from tokenjam.core.optimize import build_report
        from tokenjam.core.optimize.cost_proposals import COST_ANALYZERS

        since_dt = utcnow() - timedelta(days=_TEASER_WINDOW_DAYS)
        until_dt = utcnow()

        report = build_report(
            db=db, config=config, since=since_dt, until=until_dt,
            findings=list(COST_ANALYZERS),
        )
        estimates: list[tuple[float, int]] = []
        if report.downgrade is not None:
            usd = report.downgrade.past_overspend_usd
            if usd is not None:
                estimates.append((usd, getattr(report.downgrade, "past_overspend_tokens", None) or 0))
        for finding in (report.findings or {}).values():
            usd = getattr(finding, "past_overspend_usd", None)
            if usd is not None:
                estimates.append((usd, getattr(finding, "past_overspend_tokens", None) or 0))
        largest_usd, largest_tokens = max(estimates, default=(0.0, 0))

        if largest_usd < _TEASER_MIN_USD:
            return None
        return (
            f"[dim]{format_cost(largest_usd)} / {format_tokens(largest_tokens)} tokens "
            f"recoverable (largest single fix): run "
            f"[bold]tj optimize[/bold][/dim]"
        )
    except Exception:
        # Never let a teaser computation break `tj status` itself.
        return None


def _fmt_dur(seconds: float | None, *, coarse: bool = False) -> str:
    """Human duration. coarse=True caps at days/hours for long wall-clock spans."""
    if seconds is None:
        return "-"
    secs = int(seconds)
    if coarse and secs >= 3600:
        mins = secs // 60
        d, rem = divmod(mins, 1440)
        h, m = divmod(rem, 60)
        return f"{d}d {h}h" if d else f"{h}h {m}m"
    mins, s = divmod(secs, 60)
    if mins >= 60:
        h, m = divmod(mins, 60)
        return f"{h}h {m}m"
    return f"{mins}m {s}s"


def _cost_line(data: dict) -> str:
    """Render the 'Cost today' line (#96): always the raw dollar figure.

    Previously reframed to a "% of cycle" share for subscription/local plans
    via `core/framing.render_dollar`. Removed by product decision: dollars
    are always legitimate and tj no longer differentiates its rendering
    between subscription and API users.

    `daily_limit` is a user-configured DAILY dollar cap (`budget.daily_usd`);
    shown as its literal dollar amount with a `/day` suffix.
    """
    cost_str = format_cost(data["cost_today"])
    if data["daily_limit"]:
        cost_str += f" / {format_cost(data['daily_limit'])}/day limit"
    return cost_str


def _dedupe_alerts(alerts: list[Alert]) -> list[tuple[Alert, int]]:
    """Collapse repeat alerts of the same type into one (alert, count) pair (#96).

    The alert engine's dedup state (CooldownTracker, `_failure_rate_fired`) is
    in-memory only and resets on process restart, so the DB can legitimately
    hold multiple rows for the same (type, agent) pair — the query itself
    (`get_alerts`) is a plain `SELECT`, no join, so there's no fanout to fix
    there. Collapse at render time instead of dropping information: keep the
    first (most recent, since `alerts` is fired_at DESC) occurrence of each
    type and note how many fired.
    """
    first_seen: dict[str, Alert] = {}
    counts: dict[str, int] = {}
    order: list[str] = []
    for alert in alerts:
        key = alert.type.value
        if key not in first_seen:
            first_seen[key] = alert
            counts[key] = 0
            order.append(key)
        counts[key] += 1
    return [(first_seen[key], counts[key]) for key in order]


def _print_agent_status(data: dict, active_alerts: list, session: object | None) -> None:
    status = data["status"]
    icon = status_icon(status)
    style = "green" if status == "active" else "dim"

    console.print(f"[{style}]{icon}[/] [bold]{data['agent_id']}[/bold]   "
                  f"{status}")
    console.print()

    console.print(f"  Cost today:     {_cost_line(data)}")

    in_tok = format_tokens(data["input_tokens"])
    out_tok = format_tokens(data["output_tokens"])
    console.print(f"  Tokens:         {in_tok} in / {out_tok} out")

    tool_str = str(data["tool_call_count"])
    if data["error_count"]:
        tool_str += f"  ({data['error_count']} failed)"
    console.print(f"  Tool calls:     {tool_str}")

    # Active = compute time (Σ span durations); Elapsed = wall-clock, which can
    # span days for resumed sessions (issue #147).
    if data.get("active_seconds") is not None or data.get("duration_seconds") is not None:
        parts = []
        if data.get("active_seconds") is not None:
            parts.append(f"active {_fmt_dur(data['active_seconds'])}")
        if data.get("duration_seconds") is not None:
            parts.append(f"[dim]elapsed {_fmt_dur(data['duration_seconds'], coarse=True)}[/dim]")
        console.print(f"  Duration:       {' · '.join(parts)}")

    if data["session_id"]:
        console.print(f"  Active session: {data['session_id']}")

    console.print()
    for alert, count in _dedupe_alerts(active_alerts):
        from tokenjam.utils.formatting import severity_colour
        colour = severity_colour(alert.severity.value)
        suffix = f" ×{count}" if count > 1 else ""
        console.print(f"  [{colour}]{alert.title}{suffix}[/]")

    if not active_alerts:
        console.print("  [green]No active alerts[/green]")

    console.print()
