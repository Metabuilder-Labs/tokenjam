"""Batch API placement candidates (a check inside the downsize lane).

Some workloads are not interactive at all: a scheduled job wakes up on a
near-fixed cadence, runs without a person in the loop, and nobody is waiting on
the answer. Those are the only workloads for which the Batch API's flat 50%
discount is even discussable, because adopting it means submitting work and
polling for it later.

Detection is deliberately conservative and structural. A workload group
qualifies only when BOTH hold over the window:

  * its session start times are cadence-regular (coefficient of variation of the
    inter-start gaps below ``MAX_START_GAP_CV``), and
  * no session in the group has a human turn after its first model call.

Either condition failing means not a candidate. This module never proposes the
switch as a config flip: the card states plainly that batch adoption is an
architectural change (asynchronous submit and poll), and the check stays
advise-only.

Home: this lives in the downsize lane rather than the proposal adapter layer
because detection needs the span table, and the adapter layer is pure by
contract (it reads an already-built report and never touches the DB). It is a
sibling module of ``model_downgrade`` rather than more code inside it, and is
run by that analyzer's registry entry point, so no new analyzer is registered.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tokenjam.core.optimize.accounting import four_type_token_sum_sql

#: The Batch API bills a flat half of standard prices.
BATCH_DISCOUNT = 0.50

#: Below this many sessions there are fewer than three inter-start gaps, which
#: is not enough spacing to call a cadence regular at all.
MIN_SESSIONS_FOR_CADENCE = 5

#: Coefficient of variation (stdev / mean) of the inter-start gaps. A cron-like
#: workload sits near zero; anything a person triggers by hand scatters well
#: above this.
MAX_START_GAP_CV = 0.15

#: A workload below this window spend is not worth an architectural change.
MIN_GROUP_COST_USD = 1.0

#: Construction footnote for the card's dollar figure.
BATCH_ESTIMATE_BASIS = (
    "Window spend of the workloads whose session starts fit a regular cadence "
    "and that ran with no human turn after their first model call, halved at "
    "the Batch API's flat 50% rate. An estimate of the price difference on the "
    "same tokens; it assumes the same work runs unchanged on the batch lane."
)

#: The friction the card must state next to the number.
BATCH_FRICTION_NOTE = (
    "Batch adoption is an architectural change, not a configuration flip: work "
    "is submitted and polled for later, so only workloads nobody is waiting on "
    "can move. Most batches finish within an hour, but none are interactive."
)


@dataclass
class BatchCandidate:
    """One workload group that fits the batch-placement shape."""
    agent_id:            str
    sessions:            int
    first_start:         str
    last_start:          str
    median_gap_seconds:  float
    gap_cv:              float
    cost_usd:            float
    tokens:              int
    estimated_batch_saving_usd: float


@dataclass
class BatchPlacementFinding:
    """Batch-placement candidates over the window."""
    candidates:          list[BatchCandidate] = field(default_factory=list)
    window_cost_usd:     float = 0.0
    candidate_cost_usd:  float = 0.0
    percent_of_window_cost: float = 0.0
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis:      str = BATCH_ESTIMATE_BASIS
    estimate_confidence: str = "estimated"
    friction:            str = BATCH_FRICTION_NOTE
    # Effective thresholds this run applied (config-overridable, see
    # core.config.OptimizeConfig) — carried on the finding so a renderer never
    # hardcodes a number that could be stale against the user's own config.
    min_sessions_for_cadence: int   = MIN_SESSIONS_FOR_CADENCE
    min_group_cost_usd:       float = MIN_GROUP_COST_USD


def gap_coefficient_of_variation(starts: list[datetime]) -> float | None:
    """CV of the inter-start gaps, or ``None`` when it cannot be computed.

    ``None`` for fewer than three gaps (nothing to call regular) and for a
    degenerate zero mean gap (identical start times, which is an ingest artifact
    rather than a cadence).
    """
    ordered = sorted(starts)
    gaps = [
        (b - a).total_seconds()
        for a, b in zip(ordered, ordered[1:], strict=False)
    ]
    if len(gaps) < 3:
        return None
    mean_gap = statistics.fmean(gaps)
    if mean_gap <= 0:
        return None
    return statistics.stdev(gaps) / mean_gap


def _session_rows(
    conn: Any, since: datetime, until: datetime, agent_id: str | None,
) -> list[tuple]:
    """Per-session start, agent, spend and all four billed token types."""
    clauses = ["start_time >= $1", "start_time < $2", "session_id IS NOT NULL"]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    return conn.execute(
        f"SELECT session_id, "
        f"FIRST(agent_id) AS agent_id, "
        f"MIN(start_time) FILTER (WHERE model IS NOT NULL) AS first_call, "
        f"COALESCE(SUM(cost_usd), 0.0) AS cost_usd, "
        f"{four_type_token_sum_sql(alias='tokens')} "
        f"FROM spans WHERE {where} "
        f"GROUP BY session_id",
        params,
    ).fetchall()


def _human_turn_starts(
    conn: Any, since: datetime, until: datetime, agent_id: str | None,
) -> dict[str, list[datetime]]:
    """Human-turn timestamps per session.

    A human turn arrives as an ``invoke_agent`` span (the Claude Code and Codex
    user-prompt events both land as one). The caller compares them against each
    session's first model call: the opening prompt that starts a run is not
    disqualifying, a turn after the run began is.
    """
    clauses = ["start_time >= $1", "start_time < $2", "session_id IS NOT NULL"]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT session_id, start_time FROM spans "
        f"WHERE {where} AND name = 'invoke_agent'",
        params,
    ).fetchall()
    starts: dict[str, list[datetime]] = {}
    for session_id, start in rows:
        if session_id and start is not None:
            starts.setdefault(str(session_id), []).append(start)
    return starts


def analyze_batch_placement(
    conn: Any,
    since: datetime,
    until: datetime,
    agent_id: str | None,
    window_cost_usd: float,
    *,
    min_sessions_for_cadence: int = MIN_SESSIONS_FOR_CADENCE,
    min_group_cost_usd: float = MIN_GROUP_COST_USD,
) -> BatchPlacementFinding | None:
    """Workload groups whose shape allows a batch-lane discussion.

    Returns ``None`` when nothing qualifies, so the card is simply absent rather
    than rendering an empty candidate list. ``min_sessions_for_cadence`` and
    ``min_group_cost_usd`` override the module constants of the same name
    (config-overridable via ``core.config.OptimizeConfig``); the defaults
    reproduce today's behaviour unchanged.
    """
    rows = _session_rows(conn, since, until, agent_id)
    if not rows:
        return None
    human_turns = _human_turn_starts(conn, since, until, agent_id)

    by_agent: dict[str, list[dict[str, Any]]] = {}
    interactive: set[str] = set()
    for session_id, agent, first_call, cost, tokens in rows:
        if first_call is None:
            continue   # no model call in this session: nothing to place
        sid = str(session_id)
        if any(ts > first_call for ts in human_turns.get(sid, [])):
            interactive.add(sid)
        by_agent.setdefault(str(agent or "unknown"), []).append({
            "session_id": sid,
            "start": first_call,
            "cost_usd": float(cost or 0.0),
            "tokens": int(tokens or 0),
        })

    candidates: list[BatchCandidate] = []
    for agent, sessions in sorted(by_agent.items()):
        if len(sessions) < min_sessions_for_cadence:
            continue
        if any(s["session_id"] in interactive for s in sessions):
            continue
        starts = sorted(s["start"] for s in sessions)
        cv = gap_coefficient_of_variation(starts)
        if cv is None or cv >= MAX_START_GAP_CV:
            continue
        cost = sum(s["cost_usd"] for s in sessions)
        if cost < min_group_cost_usd:
            continue
        gaps = [
            (b - a).total_seconds()
            for a, b in zip(starts, starts[1:], strict=False)
        ]
        candidates.append(BatchCandidate(
            agent_id=agent,
            sessions=len(sessions),
            first_start=starts[0].isoformat(),
            last_start=starts[-1].isoformat(),
            median_gap_seconds=round(statistics.median(gaps), 1),
            gap_cv=round(cv, 4),
            cost_usd=round(cost, 6),
            tokens=sum(s["tokens"] for s in sessions),
            estimated_batch_saving_usd=round(cost * BATCH_DISCOUNT, 6),
        ))

    if not candidates:
        return None
    candidate_cost = sum(c.cost_usd for c in candidates)
    return BatchPlacementFinding(
        candidates=sorted(candidates, key=lambda c: c.cost_usd, reverse=True),
        window_cost_usd=round(window_cost_usd, 6),
        candidate_cost_usd=round(candidate_cost, 6),
        percent_of_window_cost=(
            round(100.0 * candidate_cost / window_cost_usd, 1)
            if window_cost_usd > 0 else 0.0
        ),
        estimated_recoverable_usd=round(candidate_cost * BATCH_DISCOUNT, 6),
        estimated_recoverable_tokens=sum(c.tokens for c in candidates),
        min_sessions_for_cadence=min_sessions_for_cadence,
        min_group_cost_usd=min_group_cost_usd,
    )
