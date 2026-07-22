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
        "estimated_recoverable_tokens": cluster.estimated_recoverable_tokens,
        "note": HONESTY_CAVEAT,
    }
