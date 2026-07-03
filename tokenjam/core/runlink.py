"""Best-effort linkage from a launcher session to the *run* it spawned.

A fan-out harness (a governor / launcher) kicks off many worker sessions and
groups them under a shared ``tokenjam.run_id`` (see ``api/routes/runs.py`` and
``docs/harness-integration.md``). There are two ways to discover that a session
is connected to such a run:

  * **tagged** — the session itself carries ``tokenjam.run_id`` (a worker, or a
    launcher the harness also tagged). The session record already has it; no
    scanning needed.
  * **inferred** — the launcher only *announces* the run id in its output (e.g.
    a governor printing ``TokenJam run id: gov-20260623T093359Z-11694``) but is
    not itself tagged with it. We scan the launcher's transcript text for such
    an id, then the caller looks it up in the runs table to confirm.

This module does only the **pure transcript scan** — it never touches the DB.
The run rollup (a DB query) is assembled in the API layer, which already owns
the runs query; ``core`` must not import ``api``. Everything here is
best-effort and returns empty on any failure; an inferred id is a *candidate*
until the caller validates it against real run data.
"""
from __future__ import annotations

import re
from pathlib import Path

from tokenjam.core.transcript import session_transcript_path

#: Cap on transcript bytes scanned for a run-id announcement. The announcement
#: is printed once when the loop starts (near the top), so a generous head slice
#: catches it without reading a multi-MB cache-heavy transcript end to end.
MAX_SCAN_BYTES = 2_000_000

#: Cap on how many distinct candidate ids we return (defensive; there is
#: realistically one run id per launcher).
MAX_RUN_IDS = 8

#: ``tokenjam.run_id = <id>`` / ``tokenjam run id: <id>`` / ``run id: <id>`` —
#: the explicit announcement forms. Tolerates ``.``/``_``/space separators and
#: an optional surrounding quote/backtick around the id.
_ANNOUNCE_RE = re.compile(
    r"(?:tokenjam[._ ]run[._ ]id|run[ ]id)\s*[:=]\s*['\"`]?"
    r"([A-Za-z0-9][A-Za-z0-9._-]{2,}[A-Za-z0-9])",
    re.IGNORECASE,
)

#: A governor-style run id token: ``gov-YYYYMMDDTHHMMSSZ-<n>``. Matched directly
#: so we still find the id even if the announcing prose is phrased unusually.
_GOV_RE = re.compile(r"\bgov-\d{8}T\d{6}Z-\d+\b")


def scan_transcript_run_ids(
    session_id: str, projects_root: Path | str | None = None
) -> list[str]:
    """Scan a session's transcript for announced run ids (ordered, de-duped).

    Returns the candidate run ids a launcher printed, in first-seen order, or an
    empty list when there's no transcript or nothing matches. These are
    *candidates* — the caller validates each against the runs table before
    surfacing it. Never raises.
    """
    path = session_transcript_path(session_id, projects_root)
    if path is None:
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(MAX_SCAN_BYTES)
    except OSError:
        return []

    seen: set[str] = set()
    ordered: list[str] = []
    for match in _ANNOUNCE_RE.finditer(text):
        _add(match.group(1), seen, ordered)
    for match in _GOV_RE.finditer(text):
        _add(match.group(0), seen, ordered)
    return ordered[:MAX_RUN_IDS]


def _add(candidate: str, seen: set[str], ordered: list[str]) -> None:
    """Append a stripped candidate id if non-empty and not already seen."""
    cid = candidate.strip().strip("'\"`")
    # Reject the literal ellipsis placeholders a doc-style announcement may use
    # (e.g. "tokenjam.run_id=gov-…") — a real id has no unicode ellipsis.
    if not cid or "…" in cid:
        return
    if cid not in seen:
        seen.add(cid)
        ordered.append(cid)


__all__ = ["scan_transcript_run_ids", "MAX_SCAN_BYTES", "MAX_RUN_IDS"]
