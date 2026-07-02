"""Session timeline over Claude Code sessions (the zero-install first-run view).

A small, pure-logic summary of the most recent Claude Code sessions: when each
ran, its token spend, and a token-share bar. It is the "session timeline" half
of the zero-install ``tj quickstart`` first-run (issue #6) — the quota-composition
half is :mod:`tokenjam.core.context_diagnostic`.

This reads only aggregate columns already on the ``sessions`` table (no captured
content, no config, no daemon), so it runs against a transient in-memory DB that
``tj quickstart`` backfills on the fly. Dollars are computed but framed as a
*secondary* calibration signal — the headline is always token share, matching
the subscription-majority framing the rest of tj uses.

Honesty discipline (CLAUDE.md Rule 14): every figure is a *measured* token count
re-derived from the on-disk JSONL, never a projected saving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Most-recent sessions shown in the timeline. Enough to feel like "your week"
# without scrolling a terminal off-screen.
DEFAULT_TIMELINE_LIMIT = 12


@dataclass
class TimelineSession:
    """One session's roll-up for the timeline."""

    session_id: str
    agent_id: str
    started_at: datetime | None
    ended_at: datetime | None
    input_tokens: int
    output_tokens: int
    cache_tokens: int  # cache reads (re-read context)
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_tokens
        )

    @property
    def reread_tokens(self) -> int:
        return self.cache_tokens

    @property
    def reread_share(self) -> float:
        total = self.total_tokens
        return (self.cache_tokens / total) if total else 0.0

    @property
    def project(self) -> str:
        """Best-effort project label from the ``claude-code-<name>`` agent_id."""
        prefix = "claude-code-"
        if self.agent_id.startswith(prefix):
            return self.agent_id[len(prefix):] or self.agent_id
        return self.agent_id


@dataclass
class SessionTimeline:
    """Result of the session-timeline summary over a window."""

    sessions: list[TimelineSession] = field(default_factory=list)
    total_sessions: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    earliest: datetime | None = None
    latest: datetime | None = None
    project_count: int = 0

    @property
    def has_data(self) -> bool:
        return self.total_sessions > 0


def compute_session_timeline(
    conn: Any,
    *,
    agent_id: str | None = None,
    limit: int = DEFAULT_TIMELINE_LIMIT,
) -> SessionTimeline:
    """Summarize the most recent sessions for the timeline view.

    ``conn`` is a direct DuckDB connection. Aggregates (totals, project count,
    window bounds) are over ALL matching sessions; only the per-session rows are
    capped at ``limit`` (most-recent first).
    """
    result = SessionTimeline()

    where = "started_at IS NOT NULL"
    params: list[Any] = []
    if agent_id:
        where += " AND agent_id = $1"
        params.append(agent_id)

    # Aggregate over the full window (not just the capped rows).
    agg = conn.execute(
        "SELECT COUNT(*), "
        "COALESCE(SUM(COALESCE(input_tokens,0) + COALESCE(output_tokens,0) + "
        "COALESCE(cache_tokens,0)), 0), "
        "COALESCE(SUM(COALESCE(total_cost_usd,0.0)), 0.0), "
        "MIN(started_at), MAX(COALESCE(ended_at, started_at)), "
        "COUNT(DISTINCT agent_id) "
        "FROM sessions WHERE " + where,
        params,
    ).fetchone()
    if not agg or not agg[0]:
        return result

    result.total_sessions = int(agg[0] or 0)
    result.total_tokens = int(agg[1] or 0)
    result.total_cost_usd = round(float(agg[2] or 0.0), 6)
    result.earliest = agg[3]
    result.latest = agg[4]
    result.project_count = int(agg[5] or 0)

    rows = conn.execute(
        "SELECT session_id, agent_id, started_at, ended_at, "
        "COALESCE(input_tokens,0), COALESCE(output_tokens,0), "
        "COALESCE(cache_tokens,0), "
        "COALESCE(total_cost_usd,0.0) "
        "FROM sessions WHERE " + where
        + " ORDER BY started_at DESC LIMIT " + str(int(limit)),
        params,
    ).fetchall()

    for (sid, aid, started, ended, in_tok, out_tok, cache_tok, cost) in rows:
        result.sessions.append(
            TimelineSession(
                session_id=str(sid) if sid is not None else "unknown",
                agent_id=str(aid) if aid is not None else "claude-code-unknown",
                started_at=started,
                ended_at=ended,
                input_tokens=int(in_tok or 0),
                output_tokens=int(out_tok or 0),
                cache_tokens=int(cache_tok or 0),
                cost_usd=float(cost or 0.0),
            )
        )
    return result


def timeline_to_dict(timeline: SessionTimeline) -> dict[str, Any]:
    """JSON-serialisable view for ``--json`` output."""
    return {
        "total_sessions": timeline.total_sessions,
        "total_tokens": timeline.total_tokens,
        "total_cost_usd": round(timeline.total_cost_usd, 6),
        "project_count": timeline.project_count,
        "earliest": timeline.earliest.isoformat() if timeline.earliest else None,
        "latest": timeline.latest.isoformat() if timeline.latest else None,
        "sessions": [
            {
                "session_id": s.session_id,
                "agent_id": s.agent_id,
                "project": s.project,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "input_tokens": s.input_tokens,
                "output_tokens": s.output_tokens,
                "cache_tokens": s.cache_tokens,
                "total_tokens": s.total_tokens,
                "reread_share": round(s.reread_share, 4),
                "cost_usd": round(s.cost_usd, 6),
            }
            for s in timeline.sessions
        ],
    }


__all__ = [
    "DEFAULT_TIMELINE_LIMIT",
    "TimelineSession",
    "SessionTimeline",
    "compute_session_timeline",
    "timeline_to_dict",
]
