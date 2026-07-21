"""``tj session-story`` — turn-by-turn reconstruction of *how* a Claude Code
session attempted its task, with per-subagent attribution.

A thin CLI view over the deterministic core reconstruction (``core/transcript``
+ ``core/method_spine``). It surfaces the agent's own narration + literal tool
calls folded into ordered *moves* (``delegate`` / ``dead_end`` / ``verify`` /
``act``), and — for every ``delegate`` move — the spawned subagent's own mandate,
a factual tool-category tally over what it DID, and its recursively-rendered
method spine one indent level deeper.

Honesty discipline (CLAUDE.md Rule 14): the four move kinds are the ONLY intent
this surfaces — all structurally determinable. It never claims a move was
"good"/"bad"/"wasted", and it renders a capped/un-expanded delegation as exactly
that (``not expanded``) rather than inventing moves it never captured. This is a
behavioral view — what a subagent read/edited/searched/delegated/verified — so it
carries no cost or dollar figures (that framing lives in ``tj cost`` / the Lens).

Session resolution mirrors the two-path pattern the rest of the CLI uses:

  * Direct DuckDB (``db.conn``): resolve ``--last`` via a ``sessions`` query and
    build the story straight from the on-disk transcript, falling back to the
    persisted method snapshot (``core/method_capture``) when the transcript was
    pruned.
  * API-shim (``ApiBackend``, when ``tj serve`` holds the write lock): the daemon
    already did the reconstruction + snapshot fallback, so we fetch the ready
    ``{"available", ...}`` payload over HTTP and render it verbatim.
"""
from __future__ import annotations

import json
from typing import Any

import click
from rich.markup import escape

from tokenjam.cli.json_option import json_option, resolve_output_json
from tokenjam.core.method_capture import load_session_method
from tokenjam.core.method_spine import build_method_spine
from tokenjam.core.transcript import build_session_story, resolve_projects_root
from tokenjam.core.workmap import (
    _BASH_TOOLS,
    _FILE_TOOLS,
    _SEARCH_TOOLS,
    _SPAWN_TOOLS,
    _WEB_TOOLS,
)
from tokenjam.utils.formatting import console

#: The whole of the "substantial" heuristic: a session is worth auto-selecting
#: for the story view once it ran at least this many tool calls. Kept
#: deliberately minimal (any real session clears it) — see the task brief.
MIN_SUBSTANTIAL_TOOL_CALLS = 1

#: Stable user-facing reason when a session has no on-disk CC transcript AND no
#: persisted snapshot (mirrors ``api/routes/sessions._NO_TRANSCRIPT_REASON``).
_NO_TRANSCRIPT_REASON = (
    "No on-disk transcript for this session "
    "(SDK session, or transcript pruned)."
)

#: Per-move-kind display colour. ``act`` is the uncoloured default.
_KIND_STYLE = {
    "delegate": "cyan",
    "dead_end": "red",
    "verify": "green",
    "act": "",
}


@click.command("session-story")
@click.option("--session", "session_id", default=None,
              help="Reconstruct a specific session id (default: the most recent "
                   "substantial session).")
@click.option("--last", is_flag=True,
              help="Reconstruct the most recent substantial session "
                   "(the default when no --session is given).")
@json_option
@click.pass_context
def cmd_session_story(ctx: click.Context, session_id: str | None,
                      last: bool, output_json_flag: bool) -> None:
    """Show turn-by-turn HOW a Claude Code session attempted its task.

    Reconstructs the session's method — its ordered moves and, for each
    delegation, the subagent's mandate + what it did — from the on-disk
    transcript (or a persisted snapshot when the transcript was pruned). With no
    ``--session``, auto-selects the most recent substantial session.
    """
    output_json = resolve_output_json(ctx, output_json_flag)
    db = ctx.obj.get("db")
    if db is None:
        raise click.ClickException("session-story requires a database connection.")

    conn = getattr(db, "conn", None)
    if conn is None:
        _run_via_serve(db, session_id=session_id, output_json=output_json)
        return

    # ── Direct DuckDB path. ──
    if session_id is None:
        session_id = _resolve_last_session_direct(conn)
    if session_id is None:
        _emit_unavailable("no sessions found", output_json)
        return

    projects_root = resolve_projects_root(None)
    story = build_session_story(
        session_id, projects_root=projects_root, include_subagents=True
    )
    from_snapshot = False
    if story is None:
        snapshot = load_session_method(db, session_id)
        if snapshot and snapshot.get("story"):
            story = snapshot["story"]
            from_snapshot = True
    if not story:
        _emit_unavailable(_NO_TRANSCRIPT_REASON, output_json, session_id=session_id)
        return

    if output_json:
        _emit_story_json(story, session_id=session_id, from_snapshot=from_snapshot)
        return

    render_session_story(story, session_id=session_id, from_snapshot=from_snapshot)


def _run_via_serve(db: Any, *, session_id: str | None, output_json: bool) -> None:
    """Render the story fetched from a running ``tj serve``.

    The daemon holds the DuckDB write lock, so the CLI can't read the transcript
    file or the snapshot table directly here — the daemon already did the
    reconstruction + fallback, so we ask it for the ready payload and render it.
    """
    from tokenjam.core.api_backend import ApiBackend

    if not isinstance(db, ApiBackend):
        raise click.ClickException(
            "tj session-story needs either a direct DuckDB connection or a "
            "running tj serve at the configured api.{host,port}."
        )

    if session_id is None:
        try:
            session_id = db.find_last_substantial_session(MIN_SUBSTANTIAL_TOOL_CALLS)
        except Exception as exc:  # noqa: BLE001 — surface any HTTP/transport error
            raise click.ClickException(
                f"Failed to list sessions from tj serve: {exc}"
            ) from exc
    if session_id is None:
        _emit_unavailable("no sessions found", output_json)
        return

    try:
        payload = db.fetch_session_story(session_id, subagents=True)
    except Exception as exc:  # noqa: BLE001 — surface any HTTP/transport error
        raise click.ClickException(
            f"Failed to fetch the session story from tj serve: {exc}"
        ) from exc

    if not payload.get("available"):
        reason = payload.get("reason") or _NO_TRANSCRIPT_REASON
        _emit_unavailable(reason, output_json, session_id=session_id)
        return

    from_snapshot = bool(payload.get("from_snapshot"))
    if output_json:
        _emit_story_json(payload, session_id=session_id, from_snapshot=from_snapshot)
        return

    render_session_story(payload, session_id=session_id, from_snapshot=from_snapshot)


def _resolve_last_session_direct(conn: Any) -> str | None:
    """Most-recent substantial session id from the direct DuckDB connection."""
    row = conn.execute(
        "SELECT session_id FROM sessions WHERE tool_call_count >= $1 "
        "ORDER BY started_at DESC LIMIT 1",
        [MIN_SUBSTANTIAL_TOOL_CALLS],
    ).fetchone()
    return row[0] if row and row[0] else None


# ───────────────────────────── JSON output ─────────────────────────────────

def _emit_unavailable(reason: str, output_json: bool,
                      *, session_id: str | None = None) -> None:
    """Honest "no story" state — exit 0 on both paths, never an exception."""
    if output_json:
        payload: dict[str, Any] = {"available": False, "reason": reason}
        if session_id is not None:
            payload["session_id"] = session_id
        click.echo(json.dumps(payload, default=str))
        return
    console.print(
        f"\n[yellow]{escape(reason)}[/yellow]\n"
        "[dim]Run [bold]tj onboard --claude-code[/bold] to ingest your Claude "
        "Code sessions, then re-run [bold]tj session-story[/bold].[/dim]\n"
    )


def _emit_story_json(story: dict[str, Any], *, session_id: str,
                     from_snapshot: bool) -> None:
    payload = {
        "session_id": session_id,
        "available": True,
        "task": story.get("task") or "",
        "outcome": story.get("outcome") or "",
        "spine": build_method_spine(story),
        "from_snapshot": from_snapshot,
    }
    click.echo(json.dumps(payload, default=str))


# ───────────────────────────── rendering ──────────────────────────────────

def render_session_story(
    story: dict[str, Any],
    *,
    session_id: str,
    from_snapshot: bool = False,
    max_moves: int | None = None,
) -> None:
    """Render a session's method spine turn-by-turn (Rich).

    ``max_moves`` caps the number of TOP-LEVEL moves shown (compact mode, used by
    the ``tj quickstart`` teaser) and appends a "+K more" trailer; nested
    subagent spines always render in full. In compact mode the footer summary is
    skipped so the teaser stays short.
    """
    spine = build_method_spine(story)
    compact = max_moves is not None

    _render_header(session_id, story, compact=compact)
    _render_moves(spine, depth=0, max_moves=max_moves)
    if not compact:
        _render_footer(spine, from_snapshot)


def _render_header(session_id: str, story: dict[str, Any], *, compact: bool) -> None:
    short_id = session_id[:12]
    task = (story.get("task") or "").strip()
    header = f"[bold]session {escape(short_id)}[/bold]"
    if task:
        header += f"  {escape(task)}"
    console.print(header)
    if not compact:
        outcome = (story.get("outcome") or "").strip()
        if outcome:
            console.print(f"[dim]outcome:[/dim] {escape(outcome)}")


def _render_moves(
    moves: list[dict[str, Any]], depth: int, max_moves: int | None = None,
) -> None:
    """Render an ordered list of moves at ``depth``; cap top-level when asked."""
    shown = moves if max_moves is None else moves[:max_moves]
    for i, move in enumerate(shown, 1):
        _render_move(move, i, depth)
    if max_moves is not None and len(moves) > max_moves:
        extra = len(moves) - max_moves
        console.print(
            f"    [dim]… +{extra} more — see full detail with "
            "`tj session-story`[/dim]"
        )


def _render_move(move: dict[str, Any], index: int, depth: int) -> None:
    indent = "  " * depth
    kind = move.get("kind") or "act"
    style = _KIND_STYLE.get(kind, "")
    kind_text = f"[{style}]{kind}[/{style}]" if style else kind
    label = escape((move.get("label") or "").strip())
    console.print(f"{indent}  {index}. {kind_text}  {label}")

    if kind == "delegate":
        for deleg in move.get("delegations") or []:
            _render_delegation(deleg, depth)


def _render_delegation(deleg: dict[str, Any], depth: int) -> None:
    indent = "  " * (depth + 1)
    name = deleg.get("name") or (deleg.get("agent_id") or "")[:8] or "subagent"
    task = escape((deleg.get("task") or "").strip())
    sub_line = f"{indent}    [cyan]{escape(str(name))}[/cyan]"
    if task:
        sub_line += f": [dim]{task}[/dim]"
    console.print(sub_line)

    capped = deleg.get("capped")
    if capped:
        console.print(f"{indent}    [dim](not expanded: {escape(str(capped))})[/dim]")
        return

    sub_spine = deleg.get("spine") or []
    console.print(f"{indent}    [dim]{_tally_evidence(sub_spine)}[/dim]")
    _render_moves(sub_spine, depth=depth + 2)


def _tally_evidence(spine: list[dict[str, Any]]) -> str:
    """Factual tool-category tally over a spine's top-level move evidence.

    E.g. ``6 moves — 4 reads, 1 edit, 1 bash``. Categorizes each evidence entry
    by the same tool-category sets ``method_spine`` uses. Descriptive only.
    """
    reads = edits = searches = commands = spawns = webs = others = 0
    for move in spine:
        for ev in move.get("evidence") or []:
            name = ev.get("name")
            if name == "Read":
                reads += 1
            elif name in _FILE_TOOLS:
                edits += 1
            elif name in _SEARCH_TOOLS:
                searches += 1
            elif name in _BASH_TOOLS:
                commands += 1
            elif name in _SPAWN_TOOLS:
                spawns += 1
            elif name in _WEB_TOOLS:
                webs += 1
            else:
                others += 1

    parts: list[str] = []
    if reads:
        parts.append(f"{reads} read{'s' if reads != 1 else ''}")
    if edits:
        parts.append(f"{edits} edit{'s' if edits != 1 else ''}")
    if searches:
        parts.append(f"{searches} search{'es' if searches != 1 else ''}")
    if commands:
        parts.append(f"{commands} bash")
    if spawns:
        parts.append(f"{spawns} delegation{'s' if spawns != 1 else ''}")
    if webs:
        parts.append(f"{webs} web fetch{'es' if webs != 1 else ''}")
    if others:
        parts.append(f"{others} other{'s' if others != 1 else ''}")

    n = len(spine)
    head = f"{n} move{'s' if n != 1 else ''}"
    return f"{head} — {', '.join(parts)}" if parts else head


def _count_spine(spine: list[dict[str, Any]]) -> dict[str, int]:
    """Totals over the whole (recursive) spine: moves / delegations /
    dead-ends / verifies. A small local count, not a reconstruction."""
    counts = {"moves": 0, "delegations": 0, "dead_ends": 0, "verifies": 0}

    def walk(moves: list[dict[str, Any]]) -> None:
        for move in moves:
            counts["moves"] += 1
            kind = move.get("kind")
            if kind == "dead_end":
                counts["dead_ends"] += 1
            elif kind == "verify":
                counts["verifies"] += 1
            for deleg in move.get("delegations") or []:
                counts["delegations"] += 1
                walk(deleg.get("spine") or [])

    walk(spine)
    return counts


def _render_footer(spine: list[dict[str, Any]], from_snapshot: bool) -> None:
    c = _count_spine(spine)
    summary = (
        f"{c['moves']} moves, {c['delegations']} delegations, "
        f"{c['dead_ends']} dead-ends, {c['verifies']} verifies"
    )
    console.print(f"\n[dim]{summary}[/dim]")
    if from_snapshot:
        console.print(
            "[dim](from a persisted snapshot — the live transcript was "
            "pruned)[/dim]"
        )


__all__ = ["cmd_session_story", "render_session_story"]
