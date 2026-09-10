"""tj statusline — a ZERO-model-token status line for Claude Code.

Claude Code runs this out-of-band after each turn and pipes the session JSON on
stdin (``{session_id, model, transcript_path, cwd, ...}``). It never enters the
model's context window, so it costs ZERO quota — the deliberate opposite of the
MCP surface, which sits in the request path and taxes every turn.

The whole Claude-Code-usage tooling ecosystem is
out-of-band for exactly this reason. tj's differentiator on that free surface is
the "where did my quota go" decomposition: the **re-read share** of THIS session
— cache-read tokens as a fraction of all tokens the session has spent — plus an
ACTIONABLE ``/compact`` nudge once re-reading starts eating the budget.

Hard requirements (a statusline is on the user's terminal, every turn):
  * fail-safe — any error prints a minimal line or nothing and ALWAYS exits 0;
    a broken statusline must never break the user's terminal with a traceback.
  * fast — pure stdlib, no network, no ``tj serve``, no DB open/write; a single
    linear pass over the transcript JSONL.
  * zero model tokens — nothing here re-enters the model context.
"""
from __future__ import annotations

import glob
import itertools
import json
import os

import click

from tokenjam.cli.tj_status import TjCommand
from tokenjam.core.usage import last_turn_context_tokens, session_usage
from tokenjam.utils.humanize import format_tokens

# Re-read share (cache-read ÷ total tokens over the session) thresholds. Past
# WARN the context is mostly re-reading itself; past CRIT it is almost entirely
# re-read. These gate the BADGE (severity) only — the remedy the nudge suggests
# is chosen by WHAT is driving the re-reads, not by the share alone. Tunable in
# one place.
REREAD_WARN = 70.0
REREAD_CRIT = 88.0

_BADGE_OK = "✓"     # ✓  healthy
_BADGE_WARN = "⚠"   # ⚠  re-read climbing
_BADGE_CRIT = "\U0001f573️"  # 🕳️ re-read dominating

# Inclusion types (mirrors ``core.context_diagnostic``'s INCLUSION_* tags,
# carried through the attribution cache). STATIC drivers are re-injected /
# re-read every turn REGARDLESS of compaction (CLAUDE.md, @file reads, re-run
# searches) — ``/compact`` shrinks conversation history only, so it cannot reduce
# them; the fix is structural. Everything else (repeated prompts, large repeated
# tool outputs = "history bloat") DOES accumulate in conversation history, but a
# natural-boundary fresh session preserves session memory (the resume-brief
# carries the thread) where ``/compact`` is lossy — so that leads, ``/compact``
# second.
_STATIC_DRIVERS = frozenset({"file_read", "search"})
# The history-bloat types, enumerated rather than left as "everything that is
# not static". A type this build doesn't recognise (a future INCLUSION_* tag, a
# cache written by a newer tj, an empty string) is something we do NOT know how
# to remedy, so it routes to the unknown nudge instead of inheriting an offer of
# ``/compact`` we can't justify.
_HISTORY_DRIVERS = frozenset({"prompt", "tool_output"})

# Driver-conditional remedies. Terse (it's a statusline), no em dashes (tokenjam
# copy rule); the ``→`` arrow and punctuation match the shipped nudge style.
_NUDGE_NEAR_LIMIT = "→ /compact now (window near full)"
_NUDGE_STATIC = "→ trim re-reads: see tj context"
# Near-full window AND a static driver: both remedies, COMPOSED from the bare
# near-limit string so the two can't drift. ``/compact`` is still the right
# immediate move (a user-chosen compaction beats a forced auto-compact) but it
# cannot touch statically re-injected content, so the structural remedy has to
# ride along instead of being displaced by it.
_NUDGE_NEAR_LIMIT_STATIC = f"{_NUDGE_NEAR_LIMIT}, then tj context"
# Memory-preserving default: a fresh session (the resume-brief carries the
# thread) beats a lossy /compact. Used for history-bloat drivers, where we know
# compaction can reach the driver.
_NUDGE_HISTORY = "→ fresh session (resume-brief) or /compact"
# Driver UNKNOWN (no backfill yet / capture off / a cache written before the
# type was recorded). We can't offer /compact here: the driver may well be a
# static one, and recommending a remedy that might not be able to work is the
# same fault as recommending it when we know it can't. So: the always-safe
# memory-preserving option, plus the diagnostic that names the actual driver.
_NUDGE_UNKNOWN = "→ fresh session (resume-brief) or see tj context"

# Context-window limits used to decide "genuinely near the limit". Claude Code
# runs a 200K window by default; the 1M-context beta stamps "[1m]" onto the
# model's display name, which we detect to avoid falsely flagging a 1M session as
# near-full at 200K-scale occupancy. When we can't tell, we assume the common
# 200K window (a false "near limit" is capped by requiring genuinely high
# occupancy, and it only ever ADDS a /compact option the CC user already has).
CONTEXT_LIMIT_DEFAULT = 200_000
CONTEXT_LIMIT_1M = 1_000_000
NEAR_LIMIT_FRACTION = 0.85


def find_transcript(data: dict) -> str | None:
    """Locate the session transcript JSONL.

    Prefer the explicit ``transcript_path`` Claude Code hands us; otherwise glob
    ``~/.claude/projects/**/<session_id>.jsonl``. Returns ``None`` when neither
    resolves to a real file — the caller degrades to a model-only line.
    """
    tp = data.get("transcript_path")
    if tp and os.path.isfile(tp):
        return tp
    sid = data.get("session_id") or data.get("sessionId")
    if not sid:
        return None
    pattern = os.path.expanduser(f"~/.claude/projects/**/{sid}.jsonl")
    hits = glob.glob(pattern, recursive=True)
    return hits[0] if hits else None


def _session_figures(path: str) -> tuple[int, float, int]:
    """Return ``(total_tokens, reread_pct, window_tokens)`` in ONE transcript read.

    A single file open, split into two lazy streams via ``itertools.tee``, feeds
    both ``core.usage`` computations so the statusline keeps its "one linear pass
    over the transcript" contract without loading the whole file into memory:
    ``session_usage`` for the session-total re-read %, and
    ``last_turn_context_tokens`` for the live window occupancy that case (c) of
    the nudge needs. Both delegate the four-bucket parse + last-wins dedup to the
    same ``core.usage`` source of truth shared with the backfill ingest path, so
    the numbers can't drift from the Cost tab for the same session.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines_a, lines_b = itertools.tee(fh)
        usage = session_usage(lines_a)
        total = usage.total
        reread_pct = (100.0 * usage.cache_read_tokens / total) if total else 0.0
        window_tokens = last_turn_context_tokens(lines_b)
    return total, reread_pct, window_tokens


def _model_name(data: dict) -> str:
    """Extract a display model name from the statusline payload.

    Claude Code sends ``model`` either as a plain string or as a dict with a
    ``display_name`` (and other) fields.
    """
    model = data.get("model")
    if isinstance(model, dict):
        return str(model.get("display_name") or model.get("id") or "?")
    return str(model) if model else "?"


def _nudge_for(driver_type: str | None, near_limit: bool) -> str:
    """The remedy to suggest, chosen by the top driver (and window fullness).

    ``/compact`` shrinks conversation history only, so suggesting it for a
    statically re-injected driver (CLAUDE.md, @file, re-run searches) is
    factually wrong — it cannot reduce those re-reads. Near-limit therefore
    COMPOSES with the driver rather than short-circuiting it: the window being
    near full makes ``/compact`` urgent, it does not make it sufficient.
      * near limit + static driver -> both, ``/compact`` now and ``tj context``
        after (the re-reads survive the compaction);
      * near limit, driver not static or not known -> a direct ``/compact``
        (a user-chosen compaction beats a forced auto-compact). This is the ONE
        place an unknown driver is still offered ``/compact``: the window being
        about to force an auto-compact is true whatever the driver, and the
        user-chosen one is strictly better than the forced one;
      * static driver, window not near full -> a STRUCTURAL remedy only, never
        ``/compact``;
      * history-bloat driver (repeated prompts / large tool outputs), window not
        near full -> a memory-preserving fresh session first, ``/compact`` as
        the blunt second;
      * driver unknown or unrecognised (no backfill yet / capture off / a cache
        from a newer tj), window not near full -> the memory-preserving fresh
        session, pointing at ``tj context`` to diagnose, and no ``/compact``:
        the driver may be one compaction cannot reach.
    """
    if near_limit:
        return (
            _NUDGE_NEAR_LIMIT_STATIC if driver_type in _STATIC_DRIVERS
            else _NUDGE_NEAR_LIMIT
        )
    if driver_type in _STATIC_DRIVERS:
        return _NUDGE_STATIC
    if driver_type in _HISTORY_DRIVERS:
        # Compaction CAN reach this one, but a fresh session preserves memory,
        # so that leads and /compact is the blunt second.
        return _NUDGE_HISTORY
    # None, or a type this build doesn't recognise. Both are "we don't know",
    # and the /compact we can't justify for an unknown driver is exactly as
    # unjustified for an unrecognised one.
    return _NUDGE_UNKNOWN


def _badge_and_nudge(
    reread_pct: float,
    driver_type: str | None = None,
    near_limit: bool = False,
) -> tuple[str, str | None]:
    """Map re-read %, the top driver's inclusion type, and window fullness to a
    (badge, nudge) pair.

    The BADGE stays purely threshold-based (severity), except that a genuinely
    near-full window always earns at least a WARN badge — a session about to be
    force-auto-compacted is urgent regardless of its re-read share. The NUDGE is
    driver-conditional (see :func:`_nudge_for`) so the statusline never offers
    ``/compact`` as the ONLY remedy for a re-read class compaction can't touch.
    Below WARN, with the
    window not near its limit, there's no nudge at all.
    """
    if reread_pct >= REREAD_CRIT:
        return _BADGE_CRIT, _nudge_for(driver_type, near_limit)
    if reread_pct >= REREAD_WARN or near_limit:
        return _BADGE_WARN, _nudge_for(driver_type, near_limit)
    return _BADGE_OK, None


def format_status_line(
    model_name: str,
    total: int | None,
    reread_pct: float | None,
    top_driver: str | None = None,
    *,
    driver_type: str | None = None,
    near_limit: bool = False,
) -> str:
    """Build the statusline string from already-resolved figures.

    Shared by the live line (``render_line``, fed the hook's per-turn payload)
    and ``tj quickstart``'s "what you'd see live" preview section (fed a
    point-in-time figure from a past transcript) — so the two surfaces render
    IDENTICAL text for identical inputs and the preview can never drift from
    the real product. ``total``/``reread_pct`` of ``None`` degrades to the
    model-only segment (mirrors "no transcript" / "unreadable transcript").

    ``top_driver`` is an optional pre-formatted "<label> ×<count>" suffix
    (e.g. ``"CLAUDE.md ×14"``) naming the top recurring-inclusion source, not
    just its raw percentage. Defaults to ``None`` — every existing caller that
    doesn't pass it renders byte-for-byte unchanged. ``driver_type`` and
    ``near_limit`` steer the driver-conditional nudge (see :func:`_badge_and_nudge`);
    both default to the driver-agnostic remedy, so the ``tj quickstart`` preview
    (which has neither) renders a sensible memory-preserving default.
    """
    parts = [f"◆ {model_name}"]  # ◆ model
    if total is not None and reread_pct is not None:
        parts.append(f"{format_tokens(total)} tok")
        badge, nudge = _badge_and_nudge(reread_pct, driver_type, near_limit)
        reread_part = f"{badge} re-read {reread_pct:.0f}%"
        if top_driver:
            reread_part += f" ({top_driver})"
        parts.append(reread_part)
        if nudge:
            parts.append(nudge)
    return "  ".join(parts)


def _top_driver() -> tuple[str | None, str | None]:
    """The cached top driver as ``(display_label, inclusion_type)``.

    Thin wrapper over ``attribution_cache.format_driver`` — the single reader of
    the cache's JSON schema, shared with the resume-brief — so this zero-token
    surface never issues a DuckDB query. The label feeds the "<label> ×<count>"
    suffix; the type makes the nudge driver-conditional. Fail-safe: any error or
    a missing/stale cache degrades to ``(None, None)``.
    """
    from tokenjam.core.attribution_cache import format_driver
    return format_driver()


def _context_limit(model_name: str) -> int:
    """The model's context-window limit for the window-fullness check.

    Claude Code stamps ``[1m]`` onto the display name of a 1M-context session;
    everything else is the default 200K window. Detecting it avoids falsely
    flagging a 1M session as near-full at 200K-scale occupancy. Anchored to the
    bracketed ``[1m]`` form (not a bare "1m" substring match) so a future model
    name containing e.g. "3.1m" can't falsely match.
    """
    return CONTEXT_LIMIT_1M if "[1m]" in model_name.lower() else CONTEXT_LIMIT_DEFAULT


def _window_near_limit(window_tokens: int, model_name: str) -> bool:
    """Whether the live context window is genuinely near its limit.

    ``window_tokens`` of 0 (no billable turns / unreadable) is "can't tell" ->
    False, so the driver-conditional remedy applies instead of a spurious
    ``/compact``.
    """
    if window_tokens <= 0:
        return False
    return window_tokens >= _context_limit(model_name) * NEAR_LIMIT_FRACTION


def render_line(data: dict) -> str:
    """Build the one-line statusline string from a parsed payload.

    Always returns at least the model segment; appends token count, the re-read
    badge, and the driver-conditional nudge only when a transcript is found and
    readable. Whenever the line carries a nudge at all, it also names the top
    cached re-read driver (e.g. ``"CLAUDE.md ×14"``) when one is available, and
    uses its classified type — plus the live window fullness — to pick a remedy
    that ``/compact`` can actually deliver. Never raises — any failure degrades
    to the model-only line.

    The driver lookup is gated on the SAME condition as the badge escalation in
    :func:`_badge_and_nudge` (past WARN **or** near the window limit), not on
    the re-read share alone. Gating it on the share alone left the composed
    near-limit remedy unreachable for the case it exists for: a short session
    with a large CLAUDE.md can sit at 170K of a 200K window with a cumulative
    re-read share under WARN, and it would then be handed a bare ``/compact``
    the module documents as unable to touch its re-reads.
    """
    model_name = _model_name(data)
    path = find_transcript(data)
    if not path:
        return format_status_line(model_name, None, None)
    try:
        total, reread_pct, window_tokens = _session_figures(path)
    except Exception:
        return format_status_line(model_name, None, None)
    near_limit = _window_near_limit(window_tokens, model_name)
    top_driver: str | None = None
    driver_type: str | None = None
    if reread_pct >= REREAD_WARN or near_limit:
        top_driver, driver_type = _top_driver()
    return format_status_line(
        model_name, total, reread_pct, top_driver,
        driver_type=driver_type, near_limit=near_limit,
    )


#: Structural opt-out: Claude Code invokes this after EVERY turn and reads
#: its stdout directly into the terminal status line — it must stay
#: near-instant and byte-exact, never carry a spinner.
@click.command("statusline", cls=TjCommand, no_status=True)
def cmd_statusline() -> None:
    """Print the zero-token Claude Code statusline.

    Wire it into ``~/.claude/settings.json`` as::

        "statusLine": {"type": "command", "command": "tj statusline"}

    Claude Code invokes it out-of-band each turn; it costs no model quota.
    """
    import sys

    try:
        raw = sys.stdin.read()
    except Exception:
        raw = ""
    try:
        data = json.loads(raw) if raw.strip() else {}
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    try:
        line = render_line(data)
    except Exception:
        # Absolute last-resort fail-safe: never emit a traceback to the terminal.
        line = f"◆ {_model_name(data)}" if isinstance(data, dict) else ""
    if line:
        click.echo(line)
    # Always exit 0 — a non-zero status from a statusline command surfaces as an
    # error in Claude Code's UI.
