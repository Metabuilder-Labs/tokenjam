"""``tj quickstart`` — the zero-install, zero-config first run (issue #6).

The 15-second time-to-first-value path that lets a brand-new user — reached via
``npx tokenjam`` / ``uvx tj`` with **no** pip env, **no** daemon, **no** onboarding —
see where their Claude Code quota actually goes, straight from the JSONL files
ccusage already reads (``~/.claude/projects/*.jsonl``).

Design (what makes it "zero-setup"):

  * It opens a **transient in-memory** DuckDB (``InMemoryBackend``) — nothing is
    written to ``~/.tj``, no config is read or written, no daemon is started or
    contacted. Each run re-reads the JSONL fresh.
  * It backfills the on-disk Claude Code sessions into that transient DB via the
    existing :func:`tokenjam.core.backfill.ingest_claude_code` parser, then runs
    the same two read-only views the paid-deeper path exposes:
      - quota composition (re-reading vs. net-new work) from
        :mod:`tokenjam.core.context_diagnostic` (issue #4's engine, reused);
      - a session timeline from :mod:`tokenjam.core.session_timeline`.
  * The output **leads with reads-your-local-logs + added-value framing** —
    "reads your ~/.claude session logs; here's where your quota actually goes" —
    then ends on the opt-in "go deeper" pointer to ``tj onboard`` (daemon /
    statusline / live capture).

Because it manages its own transient DB, ``quickstart`` is registered in
``no_db_commands`` so the CLI never opens the on-disk DB or trips the daemon's
write lock for it.

Honesty discipline (CLAUDE.md Rule 14): every figure here is a *measured* token
share re-derived from the JSONL, never a projected saving.
"""
from __future__ import annotations

import glob as _glob
import json as _json
import re as _re
from pathlib import Path

import click

from tokenjam.cli.cmd_statusline import REREAD_WARN, format_status_line
from tokenjam.core.backfill import (
    CLAUDE_CODE_PROJECTS_ROOT,
    ingest_claude_code,
)
from tokenjam.core.context_diagnostic import compute_context_diagnostic
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.session_timeline import (
    SessionTimeline,
    TimelineSession,
    compute_session_timeline,
    timeline_to_dict,
)
from tokenjam.core.usage import AssistantUsage, iter_cumulative_usage
from tokenjam.utils.formatting import console, format_cost, format_tokens
from tokenjam.utils.time_parse import parse_since, utcnow

# First-run cap (#13): on a large ~/.claude history a full backfill into the
# transient DB blows past the <30s time-to-first-value goal. We cap the headline
# to the most-recent N sessions (bounded work, well under 30s even on thousands
# of sessions) and disclose the cap; `--full` lifts it for the complete picture.
# ~300 sessions keeps the slowest plausible session shapes comfortably in budget.
DEFAULT_MAX_SESSIONS = 300

# "Substantial" floor for the statusline live-preview (#120-adjacent): the
# most-recent session is only worth previewing the nudge on if it actually ran
# long enough to feel like a real session, not a two-turn smoke test. Below
# this we fall back to the largest recent session that crossed the threshold.
PREVIEW_MIN_TURNS = 20


@click.command("quickstart")
@click.option("--since", default="30d",
              help="Window for analysis (e.g. 7d, 30d, 2026-03-01). Default 30d.")
@click.option("--root", "root_path", default=None,
              help=f"Override Claude Code projects root (default {CLAUDE_CODE_PROJECTS_ROOT}).")
@click.option("--full", is_flag=True,
              help=f"Process the full history (default caps at the most-recent "
                   f"{DEFAULT_MAX_SESSIONS} sessions for a fast first run).")
@click.option("--json", "output_json", is_flag=True,
              help="Emit machine-readable JSON.")
@click.pass_context
def cmd_quickstart(ctx: click.Context, since: str, root_path: str | None,
                   full: bool, output_json: bool) -> None:
    """Zero-setup first run: where your Claude Code quota actually goes.

    Reads the same ~/.claude/projects/*.jsonl files ccusage does — no pip env,
    no daemon, no onboarding. On a large history the first run caps at the
    most-recent sessions for speed (use `--full` for everything). Run
    `tj onboard` afterwards to go deeper (live capture, the dashboard, and the
    zero-token statusline).
    """
    from pathlib import Path

    root = Path(root_path).expanduser() if root_path else CLAUDE_CODE_PROJECTS_ROOT
    if not root.exists():
        _render_no_logs(root, output_json)
        return

    try:
        since_dt = parse_since(since)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--since'") from exc
    until_dt = utcnow()

    # Transient in-memory DB — nothing persisted, no config, no daemon.
    max_sessions = None if full else DEFAULT_MAX_SESSIONS
    db = InMemoryBackend()
    result = ingest_claude_code(db, root=root, since=since_dt,
                                max_sessions=max_sessions)

    if result.sessions_ingested == 0:
        _render_no_sessions(result, since, output_json)
        return

    diag = compute_context_diagnostic(db.conn, since_dt, until_dt)
    timeline = compute_session_timeline(db.conn)

    if output_json:
        from tokenjam.core.context_diagnostic import diagnostic_to_dict
        payload = {
            "quota_composition": diagnostic_to_dict(diag),
            "session_timeline": timeline_to_dict(timeline),
            "backfill": {
                "sessions_ingested": result.sessions_ingested,
                "spans_ingested": result.spans_ingested,
                "project_count": result.project_count,
                "total_cost_usd": round(result.total_cost_usd, 6),
                "limit_reached": result.limit_reached,
                "max_sessions": max_sessions,
            },
        }
        click.echo(_json.dumps(payload, default=str))
        return

    _render(diag, timeline, since=since,
            limit_reached=result.limit_reached, max_sessions=max_sessions,
            root=root)


# ───────────────────────────── rendering ──────────────────────────────────

def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_no_logs(root, output_json: bool) -> None:
    if output_json:
        click.echo(_json.dumps({"error": "no_claude_code_logs", "root": str(root)}))
        return
    console.print(
        f"\n[yellow]No Claude Code logs found at {root}.[/yellow]\n"
        "[dim]tj quickstart reads your ~/.claude/projects/*.jsonl session logs. "
        "This is normal if Claude Code hasn't run on this machine "
        "yet — use it for a session, then re-run [bold]tj quickstart[/bold].[/dim]\n"
    )


def _render_no_sessions(result, since: str, output_json: bool) -> None:
    if output_json:
        click.echo(_json.dumps({"error": "no_sessions_in_window", "since": since}))
        return
    console.print(
        f"\n[yellow]No Claude Code sessions in the last {since}.[/yellow]\n"
        "[dim]Try a wider window, e.g. [bold]tj quickstart --since 90d[/bold].[/dim]\n"
    )


def _render(diag, timeline, *, since: str,
            limit_reached: bool = False, max_sessions: int | None = None,
            root: Path | None = None) -> None:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich.text import Text

    # ── Lead: reads-your-local-logs + added-value framing. ──
    console.print()
    lead = Text()
    lead.append("TokenJam reads your ", style="dim")
    lead.append("~/.claude/projects/*.jsonl", style="bold")
    lead.append(" session logs — and shows you ", style="dim")
    lead.append("where your quota actually goes", style="bold")
    lead.append(".", style="dim")
    console.print(lead)

    # Honest disclosure when the first-run cap truncated the history (#13). This
    # must read as scoping, NOT as "this is your whole history" — so we say so up
    # front and point at the full-picture escape hatches.
    if limit_reached and max_sessions is not None:
        note = Text()
        note.append("Showing your most-recent ", style="yellow")
        note.append(f"{max_sessions} sessions", style="bold yellow")
        note.append(" for a fast first run — run ", style="yellow")
        note.append("tj quickstart --full", style="bold")
        note.append(" (or ", style="yellow")
        note.append("tj context", style="bold")
        note.append(") for your full history.", style="yellow")
        console.print(note)

    # ── Quota composition (reuses the issue-#4 diagnostic engine). ──
    sections: list = []
    head = Text()
    head.append("Quota composition", style="bold")
    scope = "most-recent " if limit_reached else "last "
    head.append(f"  ·  {diag.sessions} sessions, {diag.turns} turns "
                f"({scope}{since if not limit_reached else f'{max_sessions}'})",
                style="dim")
    sections.append(head)
    sections.append(Text(""))

    reread = Text()
    reread.append(f"{_pct(diag.reread_share)} ", style="bold red")
    reread.append("of your tokens went to ", style="")
    reread.append("re-reading context", style="bold")
    reread.append(" (history, CLAUDE.md, tool output)", style="dim")
    sections.append(reread)

    work_share = (
        diag.total_work_tokens / diag.total_tokens if diag.total_tokens else 0.0
    )
    work = Text()
    work.append(f"{_pct(work_share)} ", style="bold green")
    work.append("went to ", style="")
    work.append("net-new work", style="bold")
    work.append(" (uncached input + output)", style="dim")
    sections.append(work)

    detail = Text()
    detail.append("\nRe-read:   ", style="dim")
    detail.append(f"{format_tokens(diag.total_reread_tokens)} tokens", style="bold")
    detail.append("  (cache reads)", style="dim")
    detail.append("\nNew work:  ", style="dim")
    detail.append(f"{format_tokens(diag.total_work_tokens)} tokens", style="bold")
    sections.append(detail)

    if diag.compact_candidates:
        sections.append(Text(""))
        ch = Text("Biggest reclaim opportunities", style="bold")
        sections.append(ch)
        for c in diag.compact_candidates[:3]:
            line = Text("  · ", style="dim")
            line.append(f"session {c.session_id[:12]}", style="bold")
            line.append(f"  {_pct(c.reread_share)} re-read, "
                        f"{format_tokens(c.reread_tokens)} cache "
                        f"({c.turns} turns)", style="dim")
            sections.append(line)
        sections.append(Text(
            "    → /compact mid-session (or start fresh) to reclaim that quota.",
            style="green",
        ))

    console.print(Panel(
        Group(*sections),
        title="[bold]Where your quota goes[/bold]",
        title_align="left",
        border_style="dim",
        padding=(1, 2),
    ))

    # ── Statusline live preview (self-contained; omits silently if no
    # candidate session ever crosses the nudge threshold). ──
    _render_statusline_preview(timeline, root)

    # ── Session timeline. ──
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                  title="Session timeline (most recent)", title_justify="left",
                  title_style="bold")
    table.add_column("When")
    table.add_column("Project")
    table.add_column("Tokens", justify="right")
    table.add_column("Re-read", justify="right")
    table.add_column("", justify="left")  # token-share bar

    max_tokens = max((s.total_tokens for s in timeline.sessions), default=0)
    for s in timeline.sessions:
        when = s.started_at.strftime("%m-%d %H:%M") if s.started_at else "—"
        bar = _bar(s.total_tokens, max_tokens)
        table.add_row(
            when,
            s.project[:22],
            format_tokens(s.total_tokens),
            _pct(s.reread_share),
            bar,
        )
    console.print(table)

    summary = Text()
    summary.append("Totals: ", style="dim")
    summary.append(f"{timeline.total_sessions} sessions", style="bold")
    summary.append(f" across {timeline.project_count} project"
                   f"{'s' if timeline.project_count != 1 else ''}, ", style="dim")
    summary.append(f"{format_tokens(timeline.total_tokens)} tokens", style="bold")
    if timeline.total_cost_usd > 0:
        summary.append(f"  ·  implied API value {format_cost(timeline.total_cost_usd)}",
                       style="dim")
    console.print(summary)

    # ── Honesty caveat + opt-in "go deeper" pointer. ──
    console.print(f"  [dim]{diag.caveat}[/dim]")
    console.print()
    deeper = Text()
    deeper.append("Go deeper", style="bold")
    deeper.append(" — install and set up live capture, the local dashboard, "
                  "and the zero-token statusline:", style="dim")
    console.print(deeper)
    console.print()
    console.print(Text("  pipx install tokenjam && tj onboard", style="bold cyan"))
    console.print()


def _bar(value: int, maximum: int, width: int = 16) -> str:
    """A simple proportional bar for the timeline (no unicode bullets)."""
    if maximum <= 0:
        return ""
    filled = max(1, round(width * value / maximum)) if value > 0 else 0
    return "[cyan]" + ("█" * filled) + "[/cyan]" + ("─" * (width - filled))


# ─────────────────────── statusline live preview ───────────────────────────
#
# `tj quickstart` is a read-only, one-shot report; `tj statusline` is the live
# product it upsells (a zero-token Claude Code statusline, updated every turn).
# This section renders — using the SAME `format_status_line` formatter the
# live statusline calls — the line the user's own most-recent substantial
# session would have shown at the exact turn its re-read share crossed the
# nudge threshold. Forward-looking framing only: it previews the LIVE
# experience, it is not advice about the (already-ended) session shown.


def _display_model_name(raw: str | None) -> str:
    """Best-effort human display name for a raw transcript `model` id.

    The live statusline gets a ready-made `display_name` from Claude Code's
    hook payload (see `cmd_statusline._model_name`); this preview only has the
    raw JSONL `message.model` string (e.g. `claude-opus-4-8-20260115`), so it
    reconstructs the same "Family X.Y" shape. Falls back to the raw string for
    shapes it doesn't recognize (e.g. `<synthetic>`).
    """
    if not raw:
        return "?"
    stripped = _re.sub(r"-\d{8}$", "", raw)  # trailing -YYYYMMDD build stamp
    m = _re.match(r"^claude-([a-z]+)-([\d-]+)$", stripped)
    if m:
        family, version = m.groups()
        return f"{family.capitalize()} {version.replace('-', '.')}"
    m = _re.match(r"^claude-([a-z]+)$", stripped)
    if m:
        return m.group(1).capitalize()
    if _re.match(r"^[a-z]+$", stripped):
        return stripped.capitalize()
    return raw


def _transcript_path_for(session_id: str, root: Path) -> str | None:
    """Resolve a session id to its on-disk transcript path under `root`.

    Same glob shape as the live statusline's `find_transcript` fallback, but
    honors quickstart's own `--root` override instead of hardcoding
    `~/.claude/projects` — quickstart already ingested from `root`, so the
    preview must look in the same place.
    """
    pattern = str(root / "**" / f"{session_id}.jsonl")
    hits = _glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


class _PreviewCandidate:
    """One timeline session's preview-selection scoring (see `_select_preview_session`)."""

    __slots__ = ("session", "turns", "crossing")

    def __init__(self, session: TimelineSession, turns: int,
                 crossing: tuple[int, str | None, AssistantUsage] | None) -> None:
        self.session = session
        self.turns = turns
        self.crossing = crossing


def _walk_for_preview(path: str) -> tuple[int, tuple[int, str | None, AssistantUsage] | None]:
    """Walk one transcript once: return `(total_turns, first_threshold_crossing)`.

    `first_threshold_crossing` is `(turn_index, model, cumulative_usage)` for
    the first turn whose cumulative re-read %% reaches the live statusline's
    nudge threshold (`REREAD_WARN`), or None if the session never crosses it.
    Reuses `core.usage.iter_cumulative_usage` — the exact cumulative walk the
    live statusline's own numbers are built from — so this can't show a figure
    the real statusline wouldn't have shown at that point. Never raises: an
    unreadable transcript degrades to "no candidate" (0, None).
    """
    turns = 0
    crossing: tuple[int, str | None, AssistantUsage] | None = None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for turn_index, model, usage in iter_cumulative_usage(fh):
                turns = turn_index
                if crossing is None:
                    total = usage.total
                    reread_pct = (100.0 * usage.cache_read_tokens / total) if total else 0.0
                    if reread_pct >= REREAD_WARN:
                        crossing = (turn_index, model, usage)
    except Exception:
        return 0, None
    return turns, crossing


def _select_preview_session(
    timeline: SessionTimeline, root: Path,
) -> _PreviewCandidate | None:
    """Pick the session to preview: the most-recent substantial session that
    crossed the nudge threshold; if none is substantial enough, the largest
    (by turns) that still crossed it; if none ever crossed it, None.
    """
    candidates: list[_PreviewCandidate] = []
    for session in timeline.sessions:  # already most-recent-first
        path = _transcript_path_for(session.session_id, root)
        if not path:
            continue
        turns, crossing = _walk_for_preview(path)
        if crossing is not None:
            candidates.append(_PreviewCandidate(session, turns, crossing))

    if not candidates:
        return None
    substantial = [c for c in candidates if c.turns >= PREVIEW_MIN_TURNS]
    if substantial:
        return substantial[0]  # most-recent among the substantial ones
    return max(candidates, key=lambda c: c.turns)


def _render_statusline_preview(timeline: SessionTimeline, root: Path | None) -> None:
    """"What you'd see live" preview section — self-contained; prints nothing
    when there is no session to preview (no history, no readable transcript,
    or no session ever crossed the nudge threshold)."""
    if root is None or not timeline.sessions:
        return

    from rich.text import Text

    picked = _select_preview_session(timeline, root)
    if picked is None:
        return

    turn_index, model_raw, usage = picked.crossing
    total = usage.total
    reread_pct = (100.0 * usage.cache_read_tokens / total) if total else 0.0
    line = format_status_line(_display_model_name(model_raw), total, reread_pct)

    console.print()
    intro = Text()
    intro.append("With the statusline installed, ", style="dim")
    intro.append(f"session {picked.session.session_id[:12]}", style="bold")
    intro.append(f" would have shown this at turn {turn_index}:", style="dim")
    console.print(intro)
    console.print()
    console.print(Text(f"  {line}", style="bold"))
    console.print()
    outro = Text()
    outro.append("That's live, every turn, for zero model tokens — ", style="dim")
    outro.append("tj onboard", style="bold cyan")
    outro.append(" sets it up.", style="dim")
    console.print(outro)
    console.print()
