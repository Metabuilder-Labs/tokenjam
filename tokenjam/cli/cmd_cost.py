import click
import json
from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.cost import compute_cost_diff
from tokenjam.core.models import CostFilters
from tokenjam.utils.formatting import console, make_table, format_cost, format_tokens
from tokenjam.utils.time_parse import parse_since, utcnow


@click.command("cost")
@click.option("--agent", default=None, help="Filter to specific agent_id")
@click.option("--since", default="7d", help="Time window (e.g. 1h, 7d, 2026-03-01)")
@click.option("--group-by", "group_by",
              type=click.Choice(["agent", "model", "day", "tool"]),
              default="day")
@click.option("--compare", "compare", default=None,
              help="Compare to a prior window. Accepts 'previous', 'last-week', "
                   "'last-month', 'last-7d', 'last-30d', or 'YYYY-MM-DD:YYYY-MM-DD'.")
@json_option
@click.pass_context
def cmd_cost(ctx: click.Context, agent: str | None, since: str,
             group_by: str, compare: str | None, output_json_flag: bool) -> None:
    """Show cost breakdown by agent, model, day, or tool."""
    output_json = resolve_output_json(ctx, output_json_flag)
    db = ctx.obj["db"]
    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc

    # --compare branch: surface a diff against the prior window. This runs
    # before the regular grouped report and exits — the two outputs are
    # different shapes and combining them would be cluttered.
    if compare:
        if hasattr(db, "conn"):
            until_dt = utcnow()
            from tokenjam.core.cost import override_since_for_compare
            since_dt = override_since_for_compare(compare, since_dt, until_dt)
            try:
                diff = compute_cost_diff(db, since_dt, until_dt, compare, agent_id=agent)
            except ValueError as exc:
                raise click.BadParameter(str(exc), param_hint="'--compare'") from exc
            if output_json:
                click.echo(json.dumps(_diff_to_dict(diff), default=str))
            else:
                _render_diff(diff, _compare_framing(ctx, db, since_dt, until_dt, agent, diff))
            return
        # API-shim path: fetch the diff from tj serve. The endpoint
        # mirrors compute_cost_diff's output schema, so we can pass
        # the dict to _render_diff_from_dict (or just emit it as JSON).
        if hasattr(db, "fetch_cost_compare"):
            try:
                diff_dict = db.fetch_cost_compare(
                    since=since, compare=compare, agent_id=agent,
                )
            except Exception as exc:
                raise click.ClickException(
                    f"Failed to fetch comparison from tj serve: {exc}"
                ) from exc
            if output_json:
                click.echo(json.dumps(diff_dict, default=str))
            else:
                _render_diff_dict(diff_dict)
            return
        raise click.ClickException(
            "--compare requires a direct DuckDB connection or a running tj serve."
        )

    filters = CostFilters(
        agent_id=agent,
        since=since_dt,
        group_by=group_by,
    )
    rows = db.get_cost_summary(filters)
    total = sum(r.cost_usd for r in rows)
    total_in = sum(r.input_tokens for r in rows)
    total_out = sum(r.output_tokens for r in rows)

    if output_json:
        click.echo(json.dumps({
            "rows": [vars(r) for r in rows],
            "total_cost_usd": total,
        }, default=str))
        return

    if not rows:
        console.print("[dim]No cost data found for the given filters.[/dim]")
        return

    # Plan-tier framing for the COST column (#175). Same source the API and
    # `tj optimize` / `tj cost --compare` use: subscription users get token-share
    # framing instead of raw dollars they never paid, and a qualifier note is
    # surfaced above the table. API users are byte-identical to before.
    until_dt = utcnow()
    framing = _cost_framing(ctx, db, since, since_dt, until_dt, agent,
                            total, total_in + total_out)
    note = _cost_note(framing)
    if note:
        console.print(f"[dim]{note}[/dim]")

    def _cost(value: float) -> str:
        return _cost_cell(value, framing)

    # CACHE R / CACHE W columns make cost reconcilable from the shown tokens —
    # cache-write is often the dominant driver and was previously invisible (#17).
    cache_r = sum(r.cache_tokens for r in rows)
    cache_w = sum(r.cache_write_tokens for r in rows)
    if group_by == "day":
        table = make_table("DATE", "AGENT", "MODEL", "TOKENS IN", "TOKENS OUT", "CACHE R", "CACHE W", "COST")
        for r in rows:
            table.add_row(
                r.group, r.agent_id or "-", r.model or "-",
                format_tokens(r.input_tokens), format_tokens(r.output_tokens),
                format_tokens(r.cache_tokens), format_tokens(r.cache_write_tokens),
                _cost(r.cost_usd),
            )
        table.add_row("", "", "", format_tokens(total_in), format_tokens(total_out),
                      format_tokens(cache_r), format_tokens(cache_w),
                      f"[bold]{_cost(total)}[/bold]")
    elif group_by == "agent":
        table = make_table("AGENT", "MODEL", "TOKENS IN", "TOKENS OUT", "CACHE R", "CACHE W", "COST")
        for r in rows:
            table.add_row(
                r.group, r.model or "-",
                format_tokens(r.input_tokens), format_tokens(r.output_tokens),
                format_tokens(r.cache_tokens), format_tokens(r.cache_write_tokens),
                _cost(r.cost_usd),
            )
        table.add_row("", "", format_tokens(total_in), format_tokens(total_out),
                      format_tokens(cache_r), format_tokens(cache_w),
                      f"[bold]{_cost(total)}[/bold]")
    elif group_by == "model":
        table = make_table("MODEL", "TOKENS IN", "TOKENS OUT", "CACHE R", "CACHE W", "COST")
        for r in rows:
            table.add_row(
                r.group,
                format_tokens(r.input_tokens), format_tokens(r.output_tokens),
                format_tokens(r.cache_tokens), format_tokens(r.cache_write_tokens),
                _cost(r.cost_usd),
            )
        table.add_row("", format_tokens(total_in), format_tokens(total_out),
                      format_tokens(cache_r), format_tokens(cache_w),
                      f"[bold]{_cost(total)}[/bold]")
    elif group_by == "tool":
        # Tool grouping has no token dimension — cost only.
        table = make_table("TOOL", "COST")
        for r in rows:
            table.add_row(r.group, _cost(r.cost_usd))
        table.add_row("[bold]TOTAL[/bold]", f"[bold]{_cost(total)}[/bold]")

    console.print(table)


def _cost_framing(ctx, db, since, since_dt, until_dt, agent, total_cost, total_tokens):
    """Plan-tier Framing for the bare `tj cost` table (#175), from the shared
    core/framing module — the same source the API and `tj cost --compare` use.

    Direct-conn path computes it locally; the API-shim path (daemon holds the
    DB lock, the common real-world case) reuses the `framing` block the
    /api/v1/cost response already carries. Returns None when neither is
    available → API-style raw-dollar rendering, unchanged."""
    config = ctx.obj.get("config") if ctx.obj else None
    conn = getattr(db, "conn", None)
    if config is not None and conn is not None:
        from tokenjam.core.framing import (
            WindowSummary,
            compute_framing,
            plan_determination_mix,
        )
        # Window-INDEPENDENT mix for the framing DECISION — matching the API
        # (`api/routes/cost.py`) so the CLI direct-conn path and Lens render
        # identical units + qualifier for the same DB (#197). The pricing mode
        # is a property of the user's plan, not the selected window; only the
        # totals below stay window-scoped.
        mix = plan_determination_mix(conn, agent)
        return compute_framing(config, WindowSummary(
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
            sessions=sum(mix.values()),
            plan_tier_mix=mix,
        ))
    if hasattr(db, "fetch_cost_framing"):
        try:
            fr = db.fetch_cost_framing(since=since, agent_id=agent)
        except Exception:
            return None
        if fr:
            from tokenjam.core.framing import Framing
            try:
                return Framing(**fr)
            except TypeError:
                return None  # server framing schema drift → fall back to dollars
    return None


def _cost_note(framing) -> str | None:
    """The honesty note printed above the COST table (#175).

    Prefers the framing's own qualifier_text (set for unknown plan tier and
    mixed subscription/API windows); otherwise emits a concise mode note for
    pure subscription / local so the reframed COST column is never unexplained.
    API → None (no note; byte-identical to the pre-framing output)."""
    if framing is None:
        return None
    if framing.qualifier_text:
        return framing.qualifier_text
    if framing.pricing_mode == "subscription":
        return ("Subscription plan — flat-fee billing; COST shown as share of "
                "your monthly plan, not dollars spent.")
    if framing.pricing_mode == "local":
        return "Local inference — no marginal cost; dollar figures suppressed."
    return None


def _cost_cell(value: float, framing) -> str:
    """Render one COST cell, plan-tier-aware (#175). API/unknown keep the
    historical format_cost output byte-for-byte; subscription shows token-share
    ("X% of cycle") and local suppresses to "—" via core/framing.render_dollar.
    The qualifier note (surfaced above the table) carries the honesty caveat for
    the subscription-with-API-mix and unknown cases."""
    if framing is not None and framing.pricing_mode in ("subscription", "local"):
        from tokenjam.core.framing import render_dollar
        return render_dollar(value, framing)
    return format_cost(value)


def _compare_framing(ctx, db, since_dt, until_dt, agent, diff):
    """Build the plan-tier Framing for the current comparison window from the
    shared core/framing module (#120) — same source the API and tj optimize use.
    Returns None if config/conn is unavailable (→ api-style rendering)."""
    config = ctx.obj.get("config") if ctx.obj else None
    conn = getattr(db, "conn", None)
    if config is None or conn is None:
        return None
    from tokenjam.core.framing import (
        WindowSummary,
        compute_framing,
        plan_determination_mix,
    )
    # Window-INDEPENDENT mix for the framing DECISION (#197) — same basis as the
    # bare `tj cost` table and the API. The compare window's totals/deltas stay
    # window-scoped; only the tokens-vs-dollars framing decision is plan-wide.
    mix = plan_determination_mix(conn, agent)
    return compute_framing(config, WindowSummary(
        total_cost_usd=diff.current.total_cost_usd,
        total_tokens=diff.current.total_tokens,
        sessions=sum(mix.values()),
        plan_tier_mix=mix,
    ))


def _arrow(delta: float) -> str:
    if delta > 0:
        return "[red]▲[/red]"
    if delta < 0:
        return "[green]▼[/green]"
    return "·"


def _pct_str(pct: float | None) -> str:
    if pct is None:
        return "[dim](—)[/dim]"
    sign = "+" if pct >= 0 else ""
    return f"({sign}{pct:.1f}%)"


# Plan-tier framing for the comparison view (#120). The mode decision comes from
# core/framing.compute_framing — this only chooses which lines to render. In
# subscription / local modes dollar deltas aren't marginal, so we suppress the
# dollar lines and compare token usage; unknown keeps dollars with a qualifier;
# api is byte-identical to the pre-framing output.
def _suppresses_dollars(mode: str) -> bool:
    return mode in ("subscription", "local")


def _diff_note(mode: str, qualifier_text: str | None = None) -> str | None:
    if mode == "subscription":
        return ("Subscription plan — dollar deltas omitted (flat-fee billing); "
                "comparing token usage.")
    if mode == "local":
        return "Local inference — no marginal cost; comparing token usage."
    if mode == "unknown":
        return qualifier_text or (
            "Plan tier unknown — figures may overstate actual cost. "
            "Run `tj onboard --claude-code --reconfigure` (or `--codex`)."
        )
    return None  # api → no note (byte-identical)


def _render_diff(diff, framing=None) -> None:
    """
    Human-readable diff renderer. Plan-tier-aware (#120): subscription/local
    suppress dollar deltas and compare token usage; unknown adds a qualifier;
    api output is unchanged. The mode comes from core/framing.compute_framing.
    """
    mode = getattr(framing, "pricing_mode", "api") if framing is not None else "api"
    suppress = _suppresses_dollars(mode)
    note = _diff_note(mode, getattr(framing, "qualifier_text", None))
    if note:
        console.print(f"[dim]{note}[/dim]")

    cur = diff.current
    prev = diff.previous
    cur_days = max((cur.until - cur.since).days, 1)
    prev_days = max((prev.until - prev.since).days, 1)
    cur_cost = "" if suppress else f", {format_cost(cur.total_cost_usd)}"
    prev_cost = "" if suppress else f", {format_cost(prev.total_cost_usd)}"
    console.print(
        f"\n[bold]Current  ({cur.since.date()} → {cur.until.date()}, "
        f"{cur_days}d):[/bold]  "
        f"{cur.sessions} sessions, {format_tokens(cur.total_tokens)} tokens"
        f"{cur_cost}"
    )
    console.print(
        f"[bold]Previous ({prev.since.date()} → {prev.until.date()}, "
        f"{prev_days}d):[/bold] "
        f"{prev.sessions} sessions, {format_tokens(prev.total_tokens)} tokens"
        f"{prev_cost}"
    )
    console.print()
    if not suppress:
        console.print(
            f"  Cost delta:   {_arrow(diff.cost_delta_usd)} "
            f"[bold]{format_cost(abs(diff.cost_delta_usd))}[/bold] "
            f"{_pct_str(diff.cost_delta_pct)}"
        )
    console.print(
        f"  Token delta:  {_arrow(diff.tokens_delta)} "
        f"[bold]{format_tokens(abs(diff.tokens_delta))}[/bold] "
        f"{_pct_str(diff.tokens_delta_pct)}"
    )

    # Per-agent / per-model shifts are dollar-denominated, so they're suppressed
    # alongside the dollar deltas in flat-fee / local modes.
    if not suppress and diff.by_agent:
        console.print()
        console.print("  [bold]Top shifts by agent:[/bold]")
        for entry in diff.by_agent:
            console.print(
                f"    {_arrow(entry['delta'])} {entry['group']:<24} "
                f"{format_cost(entry['previous_cost'])} → "
                f"{format_cost(entry['current_cost'])}  "
                f"[dim]({'+' if entry['delta'] >= 0 else ''}"
                f"{format_cost(entry['delta'])})[/dim]"
            )

    if not suppress and diff.by_model:
        console.print()
        console.print("  [bold]Top shifts by model:[/bold]")
        for entry in diff.by_model:
            console.print(
                f"    {_arrow(entry['delta'])} {entry['group']:<32} "
                f"{format_cost(entry['previous_cost'])} → "
                f"{format_cost(entry['current_cost'])}  "
                f"[dim]({'+' if entry['delta'] >= 0 else ''}"
                f"{format_cost(entry['delta'])})[/dim]"
            )

    console.print()


def _diff_to_dict(diff) -> dict:
    """JSON serialiser for CostDiff. Mirrors _render_diff's data."""
    def _wt(w):
        return {
            "since": w.since.isoformat(),
            "until": w.until.isoformat(),
            "sessions": w.sessions,
            "input_tokens": w.input_tokens,
            "output_tokens": w.output_tokens,
            "cache_tokens": w.cache_tokens,
            "total_tokens": w.total_tokens,
            "total_cost_usd": w.total_cost_usd,
        }
    return {
        "current": _wt(diff.current),
        "previous": _wt(diff.previous),
        "cost_delta_usd": diff.cost_delta_usd,
        "cost_delta_pct": diff.cost_delta_pct,
        "tokens_delta": diff.tokens_delta,
        "tokens_delta_pct": diff.tokens_delta_pct,
        "by_agent": diff.by_agent,
        "by_model": diff.by_model,
    }


def _render_diff_dict(d: dict) -> None:
    """
    Render the cost-diff dict format returned by `/api/v1/cost/compare`.

    Same visual structure as `_render_diff` but operates on the JSON shape
    directly so cmd_cost doesn't have to round-trip through CostDiff
    dataclasses when fetched from the API (#68 §12 follow-up).
    """
    from datetime import datetime
    cur = d.get("current") or {}
    prev = d.get("previous") or {}

    def _pd(s: str | None):
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None

    cur_since = _pd(cur.get("since"))
    cur_until = _pd(cur.get("until"))
    prev_since = _pd(prev.get("since"))
    prev_until = _pd(prev.get("until"))
    cur_days = max((cur_until - cur_since).days, 1) if (cur_since and cur_until) else 1
    prev_days = max((prev_until - prev_since).days, 1) if (prev_since and prev_until) else 1

    # Plan-tier framing from the /api/v1/cost/compare response block (#120).
    fr = d.get("framing") or {}
    mode = fr.get("pricing_mode", "api")
    suppress = _suppresses_dollars(mode)
    note = _diff_note(mode, fr.get("qualifier_text"))
    if note:
        console.print(f"[dim]{note}[/dim]")

    cur_cost = "" if suppress else f", {format_cost(float(cur.get('total_cost_usd', 0)))}"
    prev_cost = "" if suppress else f", {format_cost(float(prev.get('total_cost_usd', 0)))}"
    console.print(
        f"\n[bold]Current  "
        f"({cur_since.date() if cur_since else '?'} → "
        f"{cur_until.date() if cur_until else '?'}, {cur_days}d):[/bold]  "
        f"{cur.get('sessions', 0)} sessions, "
        f"{format_tokens(int(cur.get('total_tokens', 0)))} tokens"
        f"{cur_cost}"
    )
    console.print(
        f"[bold]Previous "
        f"({prev_since.date() if prev_since else '?'} → "
        f"{prev_until.date() if prev_until else '?'}, {prev_days}d):[/bold] "
        f"{prev.get('sessions', 0)} sessions, "
        f"{format_tokens(int(prev.get('total_tokens', 0)))} tokens"
        f"{prev_cost}"
    )
    console.print()

    cost_delta = float(d.get("cost_delta_usd", 0))
    tokens_delta = int(d.get("tokens_delta", 0))
    cost_pct = d.get("cost_delta_pct")
    tokens_pct = d.get("tokens_delta_pct")
    if not suppress:
        console.print(
            f"  Cost delta:   {_arrow(cost_delta)} "
            f"[bold]{format_cost(abs(cost_delta))}[/bold] "
            f"{_pct_str(cost_pct)}"
        )
    console.print(
        f"  Token delta:  {_arrow(tokens_delta)} "
        f"[bold]{format_tokens(abs(tokens_delta))}[/bold] "
        f"{_pct_str(tokens_pct)}"
    )

    by_agent = d.get("by_agent") or [] if not suppress else []
    by_model = d.get("by_model") or [] if not suppress else []
    if by_agent:
        console.print("\n  [bold]Top shifts by agent:[/bold]")
        for entry in by_agent:
            delta = float(entry.get("delta", 0))
            console.print(
                f"    {_arrow(delta)} {str(entry.get('group', '')):<24} "
                f"{format_cost(float(entry.get('previous_cost', 0)))} → "
                f"{format_cost(float(entry.get('current_cost', 0)))}  "
                f"[dim]({'+' if delta >= 0 else ''}{format_cost(delta)})[/dim]"
            )
    if by_model:
        console.print("\n  [bold]Top shifts by model:[/bold]")
        for entry in by_model:
            delta = float(entry.get("delta", 0))
            console.print(
                f"    {_arrow(delta)} {str(entry.get('group', '')):<32} "
                f"{format_cost(float(entry.get('previous_cost', 0)))} → "
                f"{format_cost(float(entry.get('current_cost', 0)))}  "
                f"[dim]({'+' if delta >= 0 else ''}{format_cost(delta)})[/dim]"
            )
    console.print()
