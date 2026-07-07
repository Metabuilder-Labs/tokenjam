"""Pure-stdlib display helpers (no Rich, no I/O).

Kept dependency-free on purpose: the zero-token ``tj statusline`` runs every
turn and must not pull Rich into its import path. ``utils.formatting`` re-exports
these for the many Rich-context callers, so there is one source of truth for the
formatting rules regardless of where they're imported.
"""
from __future__ import annotations

from pathlib import Path


def format_tokens(n: int) -> str:
    """Human-size a token count: ``M`` at/above a million, ``k`` at/above a
    thousand, otherwise the raw integer."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def display_path(p: object) -> str:
    """Abbreviate a filesystem path for terminal display: collapse the home
    directory to ``~``.

    Long absolute paths (deep project nesting, long usernames) wrap mid-filename
    in fixed-width output, which can't be copy-pasted correctly — and the paths
    onboarding / doctor print are exactly the ones we tell the user to go edit.
    Collapsing ``$HOME`` to ``~`` is the single biggest shortener. Never raises:
    an unresolvable or non-path value is returned stringified unchanged.
    """
    text = str(p)
    try:
        home = str(Path.home())
    except Exception:
        return text
    if not home:
        return text
    if text == home:
        return "~"
    if text.startswith(home + "/"):
        return "~" + text[len(home):]
    return text
