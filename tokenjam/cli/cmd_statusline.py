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
import json
import os

import click

from tokenjam.core.usage import session_usage
from tokenjam.utils.humanize import format_tokens

# Re-read share (cache-read ÷ total tokens over the session) thresholds. Past
# WARN the context is mostly re-reading itself; past CRIT it is almost entirely
# re-read and a /compact reclaims real quota. Tunable in one place.
REREAD_WARN = 70.0
REREAD_CRIT = 88.0

_BADGE_OK = "✓"     # ✓  healthy
_BADGE_WARN = "⚠"   # ⚠  re-read climbing
_BADGE_CRIT = "\U0001f573️"  # 🕳️ re-read dominating


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


def session_shares(path: str) -> tuple[int, float]:
    """Return ``(total_tokens, reread_pct)`` over a session's assistant turns.

    Delegates the four-bucket parse + last-wins message-id dedup to
    ``core.usage`` — the single source of truth shared with the backfill ingest
    path, so the statusline's re-read % can't drift from the Cost tab for the
    same session. ``reread_pct`` is cache-read tokens as a percent of the total.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        usage = session_usage(fh)
    total = usage.total
    reread_pct = (100.0 * usage.cache_read_tokens / total) if total else 0.0
    return total, reread_pct


def _model_name(data: dict) -> str:
    """Extract a display model name from the statusline payload.

    Claude Code sends ``model`` either as a plain string or as a dict with a
    ``display_name`` (and other) fields.
    """
    model = data.get("model")
    if isinstance(model, dict):
        return str(model.get("display_name") or model.get("id") or "?")
    return str(model) if model else "?"


def _badge_and_nudge(reread_pct: float) -> tuple[str, str | None]:
    """Map a re-read percentage to a (badge, nudge) pair."""
    if reread_pct >= REREAD_CRIT:
        return _BADGE_CRIT, "→ /compact to reclaim quota"
    if reread_pct >= REREAD_WARN:
        return _BADGE_WARN, "→ consider /compact"
    return _BADGE_OK, None


def format_status_line(
    model_name: str, total: int | None, reread_pct: float | None
) -> str:
    """Build the statusline string from already-resolved figures.

    Shared by the live line (``render_line``, fed the hook's per-turn payload)
    and ``tj quickstart``'s "what you'd see live" preview section (fed a
    point-in-time figure from a past transcript) — so the two surfaces render
    IDENTICAL text for identical inputs and the preview can never drift from
    the real product. ``total``/``reread_pct`` of ``None`` degrades to the
    model-only segment (mirrors "no transcript" / "unreadable transcript").
    """
    parts = [f"◆ {model_name}"]  # ◆ model
    if total is not None and reread_pct is not None:
        parts.append(f"{format_tokens(total)} tok")
        badge, nudge = _badge_and_nudge(reread_pct)
        parts.append(f"{badge} re-read {reread_pct:.0f}%")
        if nudge:
            parts.append(nudge)
    return "  ".join(parts)


def render_line(data: dict) -> str:
    """Build the one-line statusline string from a parsed payload.

    Always returns at least the model segment; appends token count, the re-read
    badge, and the compaction nudge only when a transcript is found and readable.
    Never raises — a transcript read failure degrades to the model-only line.
    """
    model_name = _model_name(data)
    path = find_transcript(data)
    if not path:
        return format_status_line(model_name, None, None)
    try:
        total, reread_pct = session_shares(path)
    except Exception:
        return format_status_line(model_name, None, None)
    return format_status_line(model_name, total, reread_pct)


@click.command("statusline")
def cmd_statusline() -> None:
    """Print a zero-token Claude Code status line (reads payload JSON on stdin).

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
