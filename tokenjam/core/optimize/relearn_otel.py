"""The self-improve loop's OTel lane: relearns mined from stored spans.

The transcript detector (``analyzers.relearn``) can only see agents that leave a
session transcript on disk, which in practice means the workspace agents (Claude
Code, and Claude Agent SDK apps pointed at a transcript path). Every other agent
reaches tokenjam as OpenTelemetry spans in DuckDB, and until now the detector
skipped them entirely: ``extract_failures_for_session`` returns ``[]`` the moment
a session has no transcript.

This module is the second extraction path. It reads FAILING spans straight from
the ``spans`` table and turns them into the same ``FailureEpisode`` the
transcript path produces, so clustering, the novelty filter and proposal
building are reused verbatim rather than forked.

Two deliberate limits, both honest:

  1. **Coarser signatures.** A transcript gives the raw tool error text plus the
     surrounding method-spine move. A span gives ``status_message`` (often a
     one-line exception) and a tool/span name. Clustering on that is coarser and
     will merge failures a transcript would have separated. Coarse-but-real
     beats invisible, and the recurrence threshold (>=3 distinct sessions) still
     gates what surfaces.
  2. **No apply path.** A workspace-less agent has no ``.claude/`` to write into,
     so its clusters are marked ``advise_only`` and carry no suggested target.
     The loop detects, advises and verifies for them; it never applies. See
     ``build_proposals``. tokenjam never touches a live request stream either
     way: this reads stored spans, after the fact.

Only NON-coding agents are read here. Coding agents (``is_interactive_coding_agent``)
already come in through the transcript path, and folding their spans in too would
double-count the same failure.

Never raises: an unreadable/absent ``spans`` table degrades to no failures, not a
crash, because this runs unattended on the detector's schedule.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from tokenjam.core.alerts import is_interactive_coding_agent
from tokenjam.core.optimize.analyzers.relearn import (
    FailureEpisode,
    HONESTY_CAVEAT,
    RelearnCluster,
    is_user_decline,
)

#: Cap on how much span error text feeds a signature. Mirrors the transcript
#: path, where ``transcript.py`` has already length-capped the raw error.
MAX_SPAN_ERROR_CHARS = 2000

#: The status_code value the ingest layer writes for a failed span
#: (``SpanStatus.ERROR``). Compared as the stored string, not the enum, because
#: this reads the raw table.
ERROR_STATUS = "error"


def _repo_label(agent_id: str | None) -> str:
    """Repo/service label for a span-sourced failure.

    The transcript path strips the ``claude-code-`` prefix off ``agent_id`` to
    get a repo name. A workspace-less agent has no repo, so its ``agent_id`` (the
    service name) IS the label; that is what scope and the advise-only check key
    on.
    """
    return str(agent_id or "unknown")


def non_coding_agent_ids(conn: Any | None) -> set[str]:
    """Every distinct non-coding ``agent_id`` present in ``spans``.

    Passed to ``build_proposals`` as ``advise_only_repos``: a cluster whose
    contributing repos are ALL in this set has no workspace to apply into, so it
    is advise-only. Best-effort; an empty set just means nothing is marked
    advise-only and the apply path stays as-is.
    """
    if conn is None:
        return set()
    try:
        rows = conn.execute(
            "SELECT DISTINCT agent_id FROM spans WHERE agent_id IS NOT NULL"
        ).fetchall()
    except Exception:
        return set()
    return {
        _repo_label(r[0]) for r in rows if not is_interactive_coding_agent(r[0])
    }


def _coding_repo_label(agent_id: str | None) -> str:
    """Repo label for a CODING agent, matching the transcript lane exactly.

    ``analyzers.relearn._repo_map_from_db`` strips the ``claude-code-`` prefix
    off ``agent_id`` to recover the repo name. The archive lane has to strip it
    the same way or the same repo would carry two different labels depending on
    which lane surfaced the failure, and ``_scope_for`` would then read one
    project as two.
    """
    repo = str(agent_id or "unknown")
    prefix = "claude-code-"
    return repo[len(prefix):] if repo.startswith(prefix) else repo


def extract_archived_coding_failures(
    conn: Any | None,
    session_ids: set[str],
    since: datetime | None = None,
) -> list[FailureEpisode]:
    """Failing spans from CODING sessions whose transcript is already gone.

    The archive lane (see ``analyzers.relearn.compute_relearn_finding``). The
    transcript lane owns every session Claude Code still has a ``.jsonl`` for;
    this one covers the sessions it has rotated away but tokenjam still holds
    telemetry for, which is the entire reason the archive exists. ``session_ids``
    is that transcript-less set, computed by the caller, so the two lanes are
    disjoint by construction and no failure is counted twice.

    Deliberately the same coarse-signature trade the OTel lane already makes: a
    span carries ``status_message`` rather than the raw tool error, so an
    archived cluster is coarser than a transcript-sourced one. Coarse-but-real
    beats structurally-invisible, which is what these sessions are today.

    Never raises: an unreadable ``spans`` table degrades to no failures.
    """
    if conn is None or not session_ids:
        return []

    sql = (
        "SELECT session_id, agent_id, tool_name, name, status_message, start_time "
        "FROM spans WHERE status_code = $1"
    )
    params: list[Any] = [ERROR_STATUS]
    if since is not None:
        sql += " AND start_time >= $2"
        params.append(since)

    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        return []

    failures: list[FailureEpisode] = []
    for session_id, agent_id, tool_name, name, status_message, start_time in rows:
        if not is_interactive_coding_agent(agent_id):
            continue  # the OTel lane's territory, not the archive's
        if str(session_id or "") not in session_ids:
            continue  # its transcript is still on disk; the transcript lane has it

        error_text = (status_message or "").strip()[:MAX_SPAN_ERROR_CHARS]
        if not error_text:
            error_text = (name or "").strip()
        if not error_text:
            continue
        if is_user_decline(error_text):
            continue

        failures.append(FailureEpisode(
            session_id=str(session_id or ""),
            repo=_coding_repo_label(agent_id),
            ts=start_time.isoformat() if hasattr(start_time, "isoformat") else (
                str(start_time) if start_time else None
            ),
            tool_name=str(tool_name or name or "unknown"),
            label="",
            error_text=error_text,
            kind="act",
            is_retry=False,
            depth=0,
        ))
    return failures


def extract_span_failures(
    conn: Any | None, since: datetime | None = None,
) -> list[FailureEpisode]:
    """Every failing span from a non-coding agent, as ``FailureEpisode``s.

    ``since`` optionally restricts to spans at or after a timestamp (the
    incremental-scan case). Returns ``[]`` on any query failure: a missing or
    malformed ``spans`` table must not sink the whole detector pass.
    """
    if conn is None:
        return []

    sql = (
        "SELECT session_id, agent_id, tool_name, name, status_message, start_time "
        "FROM spans WHERE status_code = $1"
    )
    params: list[Any] = [ERROR_STATUS]
    if since is not None:
        sql += " AND start_time >= $2"
        params.append(since)

    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        return []

    failures: list[FailureEpisode] = []
    for session_id, agent_id, tool_name, name, status_message, start_time in rows:
        if is_interactive_coding_agent(agent_id):
            continue  # already covered by the transcript path; never double-count

        # status_message is the real error text. Fall back to the span name so a
        # failure with no message still carries a stable signature; skip only
        # when neither says anything at all.
        error_text = (status_message or "").strip()[:MAX_SPAN_ERROR_CHARS]
        if not error_text:
            error_text = (name or "").strip()
        if not error_text:
            continue
        if is_user_decline(error_text):
            continue

        failures.append(FailureEpisode(
            session_id=str(session_id or ""),
            repo=_repo_label(agent_id),
            ts=start_time.isoformat() if hasattr(start_time, "isoformat") else (
                str(start_time) if start_time else None
            ),
            tool_name=str(tool_name or name or "unknown"),
            label="",          # spans carry no arg label the way a transcript does
            error_text=error_text,
            kind="act",        # no method spine off a span; every failure is an act
            is_retry=False,
            depth=0,
        ))
    return failures


# --- Eval-case artifact (the advise lane's hand-off) -------------------------

def to_eval_case(cluster: RelearnCluster) -> dict:
    """A JSON-serializable eval case for one clustered failure.

    The advise lane's deliverable for a workspace-less agent: tokenjam cannot
    apply a fix into an agent it has no workspace for, so it hands back the
    clustered evidence in a shape the user can feed their own eval tooling
    (regression case, assertion, or a prompt/config change to A/B themselves).

    Deliberately plain data. No verdict, no grade: the recommendation is a
    suggestion built from recurrence, and the caveat travels with it.
    """
    return {
        "signature": cluster.signature,
        "family_key": cluster.family_key,
        "title": cluster.title,
        "failure_examples": [
            {
                "session_id": ex.session_id,
                "agent": ex.repo,
                "ts": ex.ts,
                "error": ex.snippet,
            }
            for ex in cluster.examples
        ],
        "sessions": cluster.sessions,
        "occurrences": cluster.occurrences,
        "agents": list(cluster.repos),
        "proposed_fix": cluster.proposed_fix,
        "suggested_recommendation": cluster.proposed_fix,
        "advise_only": cluster.advise_only,
        "past_overspend_tokens": cluster.past_overspend_tokens,
        "note": HONESTY_CAVEAT,
    }
