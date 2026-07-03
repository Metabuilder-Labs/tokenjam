"""``tj context`` — the context-cost diagnostic over Claude Code sessions.

The validated Claude-Code wedge (issue #4). Renders the pure-logic diagnostic
from :mod:`tokenjam.core.context_diagnostic` as a screenshottable terminal card:

  * per-turn context composition — what share of each turn is *re-reading* prior
    context (cache reads: conversation history, CLAUDE.md, accumulated tool
    output) vs. *net-new work*;
  * recurring inclusions — the same file re-read across many sessions, each with
    a concrete ``@file`` / CLAUDE.md structural fix (capture-gated);
  * compact candidates — sessions where a ``/compact`` reclaims the most quota.

Headline numbers are framed in QUOTA terms for subscription (Pro / Max) users —
token-share and "% of cycle" via :mod:`tokenjam.core.framing` (the single source
of truth for plan-tier-aware rendering). Dollars are a secondary calibration
signal for API users, never the headline.

Needs the raw ``attributes`` column for recurring-inclusion detection, which
the API shim does not expose row-by-row. When ``tj serve`` holds the DuckDB
write lock, the diagnostic is instead computed **server-side** (the daemon owns
the direct connection) and fetched over the ``/api/v1/context`` shim, so this
launch-hero command renders with the daemon up rather than telling the user to
stop it (#63).
"""
from __future__ import annotations

import json
from typing import Any

import click

from tokenjam.core.context_diagnostic import (
    INCLUSION_FILE_READ,
    INCLUSION_PROMPT,
    INCLUSION_SEARCH,
    INCLUSION_TOOL_OUTPUT,
    ContextDiagnostic,
    compute_context_diagnostic,
    diagnostic_from_dict,
    diagnostic_to_dict,
)
from tokenjam.core.framing import (
    Framing,
    WindowSummary,
    compute_framing,
    plan_determination_mix,
)
from tokenjam.utils.formatting import console, format_tokens
from tokenjam.utils.time_parse import parse_since, utcnow

# Short tags shown before each recurring inclusion so the kind is obvious.
_INCLUSION_LABELS = {
    INCLUSION_FILE_READ: "file",
    INCLUSION_SEARCH: "search",
    INCLUSION_PROMPT: "prompt",
    INCLUSION_TOOL_OUTPUT: "output",
}


@click.command("context")
@click.option("--agent", default=None, help="Filter to a specific agent_id.")
@click.option("--since", default="30d",
              help="Window for analysis (e.g. 7d, 30d, 2026-03-01). Default 30d.")
@click.option("--json", "output_json", is_flag=True,
              help="Emit machine-readable JSON.")
@click.pass_context
def cmd_context(ctx: click.Context, agent: str | None, since: str,
                output_json: bool) -> None:
    """Diagnose where your Claude Code quota goes: re-reading vs. real work."""
    db = ctx.obj.get("db")
    config = ctx.obj.get("config")
    agent = agent or ctx.obj.get("agent")
    if db is None or config is None:
        raise click.ClickException("context requires a database connection.")

    # Validate the window up-front so both paths give the same error message.
    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc
    until_dt = utcnow()

    conn = getattr(db, "conn", None)
    if conn is None:
        # API-shim path: `tj serve` holds the DuckDB write lock. The diagnostic
        # reads the raw `attributes` column, which the shim can't expose
        # row-by-row and DuckDB won't let us open read-only alongside the
        # writer — so the daemon (which owns the connection) computes it and we
        # render the returned payload, instead of refusing to run on the exact
        # command the launch drives people to (#63).
        _render_via_serve(db, since=since, agent=agent, output_json=output_json)
        return

    capture = getattr(config, "capture", None)
    tool_inputs_captured = bool(getattr(capture, "tool_inputs", False))
    prompts_captured = bool(getattr(capture, "prompts", False))
    tool_outputs_captured = bool(getattr(capture, "tool_outputs", False))

    diag = compute_context_diagnostic(
        conn,
        since_dt,
        until_dt,
        agent_id=agent,
        tool_inputs_captured=tool_inputs_captured,
        prompts_captured=prompts_captured,
        tool_outputs_captured=tool_outputs_captured,
    )

    framing = _framing_for(conn, config, diag, agent)

    # The ACTION half of the measure→act→prove loop: what the `tj hook
    # cap-output` PostToolUse hook reclaimed (read from the append-only sink,
    # never the DB). Estimated (char/4).
    from tokenjam.core.savings_log import read_savings, summarize_savings
    reclaimed = summarize_savings(read_savings(config))

    if output_json:
        payload = diagnostic_to_dict(diag)
        payload["framing"] = framing.to_dict()
        payload["reclaimed"] = reclaimed
        click.echo(json.dumps(payload, default=str))
        return

    _render(diag, framing, since=since, reclaimed=reclaimed)


def _render_via_serve(
    db, *, since: str, agent: str | None, output_json: bool,
) -> None:
    """Render the diagnostic fetched from a running ``tj serve`` (#63).

    The daemon holds the DuckDB write lock, so the CLI can't read the raw
    ``attributes`` column directly. It fetches the server-computed diagnostic
    (``diagnostic_to_dict`` + the ``framing`` block) over ``/api/v1/context``,
    reconstructs the dataclasses, and renders exactly as the direct path does.
    """
    from tokenjam.core.api_backend import ApiBackend

    if not isinstance(db, ApiBackend):
        raise click.ClickException(
            "tj context needs either a direct DuckDB connection or a running "
            "tj serve at the configured api.{host,port}."
        )
    try:
        payload = db.fetch_context_diagnostic(since=since, agent_id=agent)
    except Exception as exc:  # noqa: BLE001 — surface any HTTP/transport error
        raise click.ClickException(
            f"Failed to fetch the context diagnostic from tj serve: {exc}"
        ) from exc

    if output_json:
        # The server already merged the `framing` block into the payload;
        # emit it verbatim so `--json` is byte-identical across paths.
        click.echo(json.dumps(payload, default=str))
        return

    diag = diagnostic_from_dict(payload)
    framing = _framing_from_dict(payload.get("framing"))
    _render(diag, framing, since=since)


def _framing_from_dict(data: dict | None) -> Framing:
    """Reconstruct a :class:`Framing` from a serialized ``framing`` block.

    Mirrors the reconstruction in ``cmd_cost._cost_framing``: ``Framing`` is a
    flat dataclass, so ``Framing(**data)`` round-trips ``to_dict()``. A missing
    block or a server-side schema drift degrades to the neutral default (which
    renders raw token counts) rather than raising.
    """
    if not data:
        return Framing()
    try:
        return Framing(**data)
    except TypeError:
        return Framing()


def _framing_for(
    conn, config, diag: ContextDiagnostic, agent: str | None,
) -> Framing:
    """Plan-tier framing for the diagnostic's window.

    Plan determination is window-INDEPENDENT (per #177) — the user's pricing
    mode is a property of their plan, not the selected window. Only the totals
    (tokens / cost) are window-scoped, so "% of cycle tokens" reads truthfully.
    """
    mix = plan_determination_mix(conn, agent)
    summary = WindowSummary(
        total_cost_usd=diag.total_cost_usd,
        total_tokens=diag.total_tokens,
        sessions=diag.sessions,
        plan_tier_mix=mix,
    )
    return compute_framing(config, summary)


# ───────────────────────────── rendering ──────────────────────────────────

def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _quota_share(tokens: int, framing: Framing) -> str:
    """Render a token figure as a quota share when on a subscription plan.

    Subscription users see "X% of cycle tokens" against the window total — the
    quota-native headline the research calls for. API / unknown users see the
    token count (dollars are surfaced separately as the secondary signal).
    """
    total = framing.window_total_tokens
    if framing.pricing_mode == "subscription" and total > 0:
        return f"{100.0 * tokens / total:.1f}% of cycle tokens"
    return f"{format_tokens(tokens)} tokens"


def _render(diag: ContextDiagnostic, framing: Framing, *, since: str,
            reclaimed: dict | None = None) -> None:
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text

    if not diag.has_data:
        console.print(
            "\n[yellow]No Claude Code turns found in this window.[/yellow]\n"
            "[dim]Run [bold]tj onboard --claude-code[/bold] to ingest your "
            "existing sessions, then re-run [bold]tj context[/bold].[/dim]\n"
        )
        return

    # ── Headline: the re-read-vs-work composition, quota-framed. ──
    headline = Text()
    headline.append("Context composition", style="bold")
    headline.append(f"  ·  {diag.sessions} sessions, {diag.turns} turns "
                    f"(last {since})", style="dim")

    reread = Text()
    reread.append(f"{_pct(diag.reread_share)} ", style="bold red")
    reread.append("of your tokens went to ", style="")
    reread.append("re-reading context", style="bold")
    reread.append(" (history, CLAUDE.md, tool output)", style="dim")

    work = Text()
    work_share = (
        diag.total_work_tokens / diag.total_tokens if diag.total_tokens else 0.0
    )
    work.append(f"{_pct(work_share)} ", style="bold green")
    work.append("went to ", style="")
    work.append("net-new work", style="bold")
    work.append(" (uncached input + output)", style="dim")

    breakdown = Text()
    breakdown.append("\nRe-read:    ", style="dim")
    breakdown.append(_quota_share(diag.total_reread_tokens, framing), style="bold")
    breakdown.append(f"  ({format_tokens(diag.total_reread_tokens)} cache reads)",
                     style="dim")
    # Named overhead source: prompt-cache MISS (cache-creation), #11. Shown only
    # when present so default/zero-cache-write output stays clean.
    if diag.total_cache_miss_tokens > 0:
        breakdown.append("\nCache-miss: ", style="dim")
        breakdown.append(_quota_share(diag.total_cache_miss_tokens, framing),
                         style="bold yellow")
        breakdown.append(
            f"  ({format_tokens(diag.total_cache_miss_tokens)} cache writes, "
            "billed at a premium)", style="dim")
    breakdown.append("\nNew work:   ", style="dim")
    breakdown.append(_quota_share(diag.total_work_tokens, framing), style="bold")
    breakdown.append(f"  ({format_tokens(diag.total_work_tokens)} tokens)",
                     style="dim")

    # Secondary: implied dollars for API users / calibration. Never headline.
    if framing.pricing_mode in ("api", "unknown") and diag.total_cost_usd > 0:
        breakdown.append("\nImplied $: ", style="dim")
        breakdown.append(f"${diag.total_cost_usd:,.2f}", style="bold")
        breakdown.append(" over the window", style="dim")

    # The action-proof line: tokens the output-trim hook clawed back (estimated).
    if reclaimed and reclaimed.get("trims", 0) > 0:
        breakdown.append("\nReclaimed:  ", style="dim")
        breakdown.append(
            f"~{format_tokens(reclaimed['saved_tok_est'])} est.", style="bold green")
        breakdown.append(
            f"  (tj cap-output trimmed {reclaimed['trims']} outputs"
            f"; ~{format_tokens(reclaimed.get('saved_today_tok_est', 0))} today)",
            style="dim")

    sections: list[Any] = [headline, Text(""), reread, work, breakdown]

    # ── Recurring inclusions (capture-gated, multi-kind). ──
    sections.append(Text(""))
    rec_header = Text("Recurring inclusions", style="bold")
    sections.append(rec_header)
    any_capture = (
        diag.tool_inputs_captured
        or diag.prompts_captured
        or diag.tool_outputs_captured
    )
    if diag.recurring:
        for r in diag.recurring[:5]:
            line = Text("  · ", style="dim")
            line.append(f"[{_INCLUSION_LABELS.get(r.inclusion_type, 'repeat')}] ",
                        style="cyan")
            line.append(r.target, style="bold")
            line.append(f"  ×{r.occurrences} ({r.sessions} sessions)",
                        style="dim")
            sections.append(line)
            fix = Text("    → ", style="green")
            fix.append(r.fix, style="dim")
            sections.append(fix)
    elif not any_capture:
        sections.append(Align.left(Text(
            "  Needs content capture (`[capture] tool_inputs / prompts / "
            "tool_outputs = true`) — see note below.",
            style="dim yellow",
        )))
    else:
        sections.append(Align.left(Text(
            "  None recurring across enough sessions/turns yet.", style="dim",
        )))

    # ── Compact candidates. ──
    sections.append(Text(""))
    comp_header = Text("Compact candidates", style="bold")
    sections.append(comp_header)
    if diag.compact_candidates:
        for c in diag.compact_candidates[:3]:
            line = Text("  · ", style="dim")
            line.append(f"session {c.session_id[:12]}", style="bold")
            line.append(f"  {_pct(c.reread_share)} re-read, "
                        f"{format_tokens(c.reread_tokens)} cache "
                        f"({c.turns} turns)", style="dim")
            sections.append(line)
        sections.append(Align.left(Text(
            "    → `/compact` mid-session (or start fresh) to reclaim that quota.",
            style="green",
        )))
    else:
        sections.append(Align.left(Text(
            "  No session crossed the compact threshold.", style="dim",
        )))

    panel_body = Group(*sections)
    console.print()
    console.print(Panel(
        panel_body,
        title="[bold]TokenJam Context Diagnostic[/bold]",
        title_align="left",
        border_style="dim",
        padding=(1, 2),
    ))

    # Qualifier banner (plan-tier framing) + honesty caveat, below the panel.
    if framing.qualifier_text:
        console.print(f"  [dim]{framing.qualifier_text}[/dim]")
    console.print(f"  [dim]{diag.caveat}[/dim]")
    for note in diag.notes:
        console.print(f"  [yellow]Note:[/yellow] [dim]{note}[/dim]")
    console.print()
