from __future__ import annotations

import json

import click
from rich.markup import escape

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.cli.tj_status import TjCommand
from tokenjam.core.models import Alert, AlertFilters
from tokenjam.utils.formatting import (
    console,
    format_cost,
    format_tokens,
    print_capped_table,
    status_icon,
)
from tokenjam.utils.time_parse import utcnow

#: Below this, "$X recoverable" reads as noise rather than a real pointer —
#: stay silent instead of printing a sub-dollar figure.
_TEASER_MIN_USD = 1.0

#: What the teaser says when the analyzer report store has never been written.
#: A COLD store is not zero and not "nothing to recover" — it carries no figure
#: at all, so this line names the command that produces one and states no
#: quantity. `tj optimize` runs the scan in the foreground and prints the real
#: figures; the daemon writes the store on its own schedule.
_COLD_TEASER = (
    "[dim]Recoverable waste has not been scanned yet: run "
    "[bold]tj optimize[/bold].[/dim]"
)


@click.command("status", cls=TjCommand, status_message="Scanning your sessions…")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("-v", "--verbose", "verbose_flag", is_flag=True, default=False,
              help="Print every agent's full card instead of the capped table.")
@json_option
@click.pass_context
def cmd_status(
    ctx: click.Context, agent: str | None, verbose_flag: bool, output_json_flag: bool,
) -> None:
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
    # (agent_data, active_alerts, session) per agent, collected rather than
    # printed inside the loop: the overview ranks agents against each other,
    # so nothing can be rendered until every agent has been read.
    entries: list[tuple[dict, list, object | None]] = []

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
        entries.append((agent_data, active_alerts, session))

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
        # Two views over the same data. The cards are reached deliberately —
        # by naming an agent, or by asking for all of them with -v — rather
        # than being the default nobody chose: one ~9-line card per tracked
        # agent_id is a screen whose length grows with every project
        # directory tj has ever seen, with the totals stranded at the bottom.
        # Either position turns it on. `-v` is declared on the group AND on
        # this command because the screen advertises `tj status -v`, which is
        # what a user types; reading only the group's copy made the advertised
        # string exit 2 with "No such option". They cannot conflict: both are
        # booleans meaning the same thing, so OR is the whole resolution rule
        # and there is no precedence question to get wrong.
        verbose = verbose_flag or bool(ctx.obj.get("verbose"))
        if verbose or agent_filter:
            for agent_data, active_alerts, session in entries:
                _print_agent_status(agent_data, active_alerts, session)
        else:
            _print_overview(entries)

        if unknown_count > 0:
            console.print(
                f"[dim]Note: {unknown_count} session(s) have unknown plan tier. "
                f"Run [bold]tj onboard --claude-code --reconfigure[/bold] "
                f"(or [bold]--codex[/bold]) to set it.[/dim]"
            )
        teaser = _recoverable_teaser(ctx.obj.get("config"))
        if teaser:
            console.print(teaser)
        freshness_note = _ingest_freshness_note(ctx.obj.get("config"), db)
        if freshness_note:
            console.print(freshness_note)

    ctx.exit(1 if has_active_alerts else 0)


def _recoverable_teaser(config) -> str | None:
    """One-line `tj optimize` pointer for `tj status` — nothing in status,
    doctor, statusline or the banner ever mentioned optimize existed.

    READS THE STORE, NEVER RUNS AN ANALYZER. This used to call `build_report`
    inline on every plain `tj status`, which dispatched the whole cost
    analyzer set (including the deliberately unbounded `relearn` scan) over
    the entire corpus and cost about a minute of CPU before the command
    returned. That is the identical failure `core/optimize/report_store.py`
    was written to end on the HTTP side ("no request path runs an analyzer");
    a CLI command that a user types to read a status table is the same kind of
    path. Analyzer runs happen at daemon boot, on the daemon's schedule, on a
    user Rescan, or when the user explicitly types `tj optimize` — and they
    all land in the store this reads.

    Reuses the same recoverable-savings contract every analyzer already
    carries (`past_overspend_usd`) rather than inventing a new figure, scoped
    to `COST_ANALYZERS` — the analyzers that actually feed the cost/apply rail
    (`cost_proposals.py`). The store holds every analyzer's finding, so the
    scoping is applied here on the read rather than by restricting the scan.

    Reports the LARGEST single analyzer's estimate rather than a sum across
    analyzers — mirroring `largest_recoverable_usd` in `api/routes/cost.py`.
    The analyzers price OVERLAPPING angles on the same spans (a cache fix and
    a downsize swap can both price the same call's waste), so summing them
    would double-count and print an inflated, non-additive headline; the
    single largest entry is honest standalone because it isn't a sum of
    anything (see `_recoverable_overlap_note` in `api/routes/cost.py` for the
    full rationale).

    A COLD store renders `_COLD_TEASER`: a command to run, never a number and
    never a zero. `$0.00 recoverable` off a scan that never ran reads as "you
    have no waste", which is a reassurance the data does not support (root
    anti-pattern 22). Otherwise silent (returns None) whenever the figure
    wouldn't mean anything: no config, or a sub-$1 figure. Dollars are shown
    regardless of billing mode (product decision: no differentiated messaging
    between subscription and API users).
    """
    if config is None:
        return None
    try:
        from tokenjam.core.optimize import report_store
        from tokenjam.core.optimize.cost_proposals import cost_analyzers_for_persona

        report = report_store.stored_report(config)
        if report is None:
            return _COLD_TEASER
        # PERSONA-SCOPED membership, not the raw `COST_ANALYZERS` tuple. The
        # daemon's pass now dispatches the union across personas so one stored
        # report answers for either side of the dashboard's persona picker
        # (`runner.build_report`'s `personas`), which means a claude-code
        # window's report can carry findings that persona has no lever for.
        # Teasing the largest of THOSE would point a user at a fix they cannot
        # apply.
        cost_analyzers = set(cost_analyzers_for_persona(
            str(getattr(report, "persona", "") or "unknown"),
        ))
        estimates: list[tuple[float, int]] = []
        if report.downgrade is not None:
            usd = report.downgrade.past_overspend_usd
            if usd is not None:
                estimates.append((usd, getattr(report.downgrade, "past_overspend_tokens", None) or 0))
        for name, finding in (report.findings or {}).items():
            if name not in cost_analyzers:
                continue
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


def _ingest_freshness_note(config: object, db: object) -> str | None:
    """One-line nudge when nothing has been ingested for far longer than the
    configured `[ingest] interval_minutes` cadence.

    Shares its staleness math with `tj doctor`'s "Corpus freshness" check
    (`core/ingest_freshness.py`) so the two surfaces can't silently disagree
    about what "stale" means. Three-state gating (root anti-pattern 22):
    silent when there's no config/DB to read (not-yet-known — never a
    fabricated claim), silent when the corpus genuinely isn't stale
    (known-and-fine — nothing worth a line), and a line ONLY once it's known
    to actually be stale. This is purely advisory — unlike `tj doctor`, `tj
    status` never fails on it, since a quiet corpus is also the everyday
    state of a user who simply hasn't run an agent in a while.
    """
    if config is None:
        return None
    try:
        if not hasattr(db, "conn"):
            return None
        from tokenjam.core.ingest_freshness import corpus_freshness

        interval_minutes = getattr(getattr(config, "ingest", None), "interval_minutes", 30)
        freshness = corpus_freshness(db.conn, interval_minutes, now=utcnow())
    except Exception:
        # Never let an advisory note break `tj status` itself.
        return None
    if freshness.newest_session_at is None or not freshness.is_stale:
        return None
    return (
        f"[warn]⚠ Last session ingested {freshness.age_hours:.0f}h ago[/warn] "
        f"(expected roughly every {interval_minutes}m) — run "
        f"[bold]tj doctor[/bold] to check the background daemon."
    )


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


# ---------------------------------------------------------------------------
# Overview — the default `tj status` view
# ---------------------------------------------------------------------------
# `_print_agent_status` below prints one ~9-line card per agent. That is the
# right screen for ONE agent and the wrong one for all of them: the agent_id
# set grows monotonically (roughly one per project directory tj has ever
# seen), so the card list is a screen whose length tracks how long the user
# has been running tj, with the only cross-agent totals stranded at the very
# bottom, past everything the reader has already scrolled through. It is worst
# for the heaviest user, who is exactly the one with something to find in it.
#
#   tj status              → the overview below: totals, capped table, trailer
#   tj status --agent <id> → that one agent's card, in full, renderer untouched
#   tj status -v           → byte-for-byte what a bare run printed before
#
# Rows are ranked by RECENCY (most recent session activity first, live
# sessions above everything), not by cost. Cost today is $0.00 for nearly
# every tracked agent on a normal day — a cost ranking would put the handful
# of agents the user actually worked in today in an arbitrary order below a
# tie of hundreds of zeroes. Recency is defined for every agent that has ever
# run and orders the list the way the user thinks about it.


#: Width the AGENT column is elided to. Agent ids run long (they are derived
#: from project directories) and share a leading `claude-code-` run, so a
#: whole screen of them truncated on the RIGHT renders as identical rows.
_AGENT_COL_WIDTH = 34


def _elide_left(text: str, width: int) -> str:
    """Shorten from the LEFT, keeping the distinguishing tail."""
    if len(text) <= width:
        return text
    return "…" + text[-(width - 1):]


def _last_activity(session: object | None):
    """When this agent was last doing something, or None if never recorded."""
    if session is None:
        return None
    return getattr(session, "ended_at", None) or getattr(session, "started_at", None)


def _fmt_age(when) -> str:
    """Coarse relative age. `-` when unknown, never a fabricated zero."""
    if when is None:
        return "-"
    try:
        secs = int((utcnow() - when).total_seconds())
    except TypeError:
        return "-"
    if secs < 0:
        secs = 0
    if secs < 60:
        return "just now"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def _print_overview(entries: list[tuple[dict, list, object | None]]) -> None:
    """Totals first, then the top agents by recency, then the way to the rest."""
    total_today = sum(d["cost_today"] or 0.0 for d, _a, _s in entries)
    active = sum(1 for d, _a, _s in entries if d["status"] == "active")
    alerting = sum(1 for d, _a, _s in entries if d["active_alerts"])

    header = (
        f"\n  [bold]{len(entries)}[/bold] agent{'' if len(entries) == 1 else 's'} · "
        f"[bold]{active}[/bold] active · [bold]{format_cost(total_today)}[/bold] today"
    )
    console.print(header)
    if alerting:
        console.print(
            f"  [warn]![/warn] {alerting} agent{'' if alerting == 1 else 's'} "
            f"with active alerts."
        )
    console.print()

    # Live sessions first, then most recent activity. An agent with no
    # recorded session sorts last on `-inf` rather than being handed an
    # invented timestamp.
    def sort_key(entry: tuple[dict, list, object | None]):
        data, _alerts, session = entry
        when = _last_activity(session)
        return (
            0 if data["status"] == "active" else 1,
            -(when.timestamp() if when is not None else float("-inf")),
            data["agent_id"],
        )

    rows = []
    for data, _alerts, session in sorted(entries, key=sort_key):
        # An agent_id is user data (it is derived from a project directory),
        # so it can carry `[...]` that Rich would eat as a style tag.
        #
        # `_cost_line`'s budget suffix and the token counts are deliberately
        # NOT here: the daily-limit suffix triples the column's width, and
        # per-session token counts cover a different population than a
        # today-scoped dollar column sitting beside them (root anti-pattern
        # 22b). Both live on the card, one command away.
        rows.append((
            escape(_elide_left(data["agent_id"], _AGENT_COL_WIDTH)),
            data["status"],
            format_cost(data["cost_today"]),
            str(data["active_alerts"]) if data["active_alerts"] else "-",
            _fmt_age(_last_activity(session)),
        ))

    hidden = print_capped_table(
        ("AGENT", "STATUS", "COST TODAY", "ALERTS", "LAST ACTIVE"),
        rows,
        more_command="tj status --agent <id>",
        more_lead="See one with",
        noun="agent",
        summary="Ranked by most recent activity.",
    )
    if hidden:
        console.print("  [dim]All of them: [accent]tj status -v[/accent].[/dim]")
    console.print()


def _print_agent_status(data: dict, active_alerts: list, session: object | None) -> None:
    status = data["status"]
    icon = status_icon(status)
    # Named roles, never a raw colour (Critical Rule 35 in tokenjam/CLAUDE.md).
    # "active" is weight, not green: a colour spent on the least surprising
    # outcome stops meaning anything.
    style = "ok" if status == "active" else "muted"

    console.print(f"[{style}]{icon}[/] [bold]{escape(data['agent_id'])}[/bold]   "
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
        console.print("  [muted]No active alerts[/muted]")

    console.print()
