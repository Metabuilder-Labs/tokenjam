"""
Drift detection: baseline computation and Z-score evaluation.
Fires DRIFT_DETECTED alert when a session deviates significantly from baseline.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tokenjam.core.config import AgentConfig
from tokenjam.core.models import (
    AlertType,
    DriftBaseline,
    DriftResult,
    DriftViolation,
    Severity,
)
from tokenjam.utils.time_parse import utcnow

if TYPE_CHECKING:
    from tokenjam.core.alerts import AlertEngine
    from tokenjam.core.config import TjConfig
    from tokenjam.core.db import StorageBackend
    from tokenjam.core.models import SessionRecord


def z_score(value: float, mean: float, stddev: float) -> float:
    """Standard Z-score. Returns inf if stddev is 0 and value != mean (maximum anomaly);
    returns 0.0 if stddev is 0 and value == mean (no deviation)."""
    if stddev == 0:
        return 0.0 if value == mean else float('inf')
    return (value - mean) / stddev


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Jaccard similarity between two sets.
    Returns 1.0 if both sets are empty (identical).
    """
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stddev(values: list[float], mean_val: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def build_baseline(
    agent_id: str,
    sessions: list[SessionRecord],
    db: StorageBackend,
) -> DriftBaseline:
    """
    Compute a DriftBaseline from completed sessions.
    Calculates mean/stddev for tokens, duration, tool calls,
    and collects tool sequences per session.
    """
    input_tokens = [float(s.input_tokens) for s in sessions]
    output_tokens = [float(s.output_tokens) for s in sessions]
    durations = [
        s.duration_seconds for s in sessions if s.duration_seconds is not None
    ]
    tool_counts = [float(s.tool_call_count) for s in sessions]

    # Collect tool sequences: for each session, get ordered tool names from spans
    tool_sequences: list[list[str]] = []
    for s in sessions:
        spans = db.get_recent_spans(s.session_id, limit=500)
        seq = [sp.tool_name for sp in reversed(spans) if sp.tool_name]
        tool_sequences.append(seq)

    avg_in = _mean(input_tokens)
    avg_out = _mean(output_tokens)
    avg_dur = _mean(durations) if durations else None
    avg_tc = _mean(tool_counts)

    return DriftBaseline(
        agent_id=agent_id,
        sessions_sampled=len(sessions),
        computed_at=utcnow(),
        avg_input_tokens=avg_in,
        stddev_input_tokens=_stddev(input_tokens, avg_in),
        avg_output_tokens=avg_out,
        stddev_output_tokens=_stddev(output_tokens, avg_out),
        avg_session_duration_s=avg_dur,
        stddev_session_duration=_stddev(durations, avg_dur) if avg_dur is not None else None,
        avg_tool_call_count=avg_tc,
        stddev_tool_call_count=_stddev(tool_counts, avg_tc),
        common_tool_sequences=tool_sequences,
    )


def evaluate_drift(
    session: SessionRecord,
    baseline: DriftBaseline,
    config_threshold: float = 2.0,
    sequence_diff_threshold: float = 0.4,
    db: StorageBackend | None = None,
) -> DriftResult:
    """
    Compare a completed session against the baseline.
    Returns DriftResult with violations and whether drift was detected.
    """
    violations: list[DriftViolation] = []

    # 1. Input tokens
    if baseline.avg_input_tokens is not None and baseline.stddev_input_tokens is not None:
        z = z_score(
            float(session.input_tokens),
            baseline.avg_input_tokens,
            baseline.stddev_input_tokens,
        )
        if abs(z) > config_threshold:
            violations.append(DriftViolation(
                dimension="input_tokens",
                z_score=z,
                expected=f"{baseline.avg_input_tokens:.0f}",
                observed=str(session.input_tokens),
            ))

    # 2. Output tokens
    if baseline.avg_output_tokens is not None and baseline.stddev_output_tokens is not None:
        z = z_score(
            float(session.output_tokens),
            baseline.avg_output_tokens,
            baseline.stddev_output_tokens,
        )
        if abs(z) > config_threshold:
            violations.append(DriftViolation(
                dimension="output_tokens",
                z_score=z,
                expected=f"{baseline.avg_output_tokens:.0f}",
                observed=str(session.output_tokens),
            ))

    # 3. Session duration
    dur = session.duration_seconds
    if (
        dur is not None
        and baseline.avg_session_duration_s is not None
        and baseline.stddev_session_duration is not None
    ):
        z = z_score(dur, baseline.avg_session_duration_s, baseline.stddev_session_duration)
        if abs(z) > config_threshold:
            violations.append(DriftViolation(
                dimension="session_duration",
                z_score=z,
                expected=f"{baseline.avg_session_duration_s:.1f}s",
                observed=f"{dur:.1f}s",
            ))

    # 4. Tool call count
    if baseline.avg_tool_call_count is not None and baseline.stddev_tool_call_count is not None:
        z = z_score(
            float(session.tool_call_count),
            baseline.avg_tool_call_count,
            baseline.stddev_tool_call_count,
        )
        if abs(z) > config_threshold:
            violations.append(DriftViolation(
                dimension="tool_call_count",
                z_score=z,
                expected=f"{baseline.avg_tool_call_count:.0f}",
                observed=str(session.tool_call_count),
            ))

    # 5. Tool sequence similarity (Jaccard)
    if db is not None and baseline.common_tool_sequences:
        spans = db.get_recent_spans(session.session_id, limit=500)
        session_tools = {sp.tool_name for sp in spans if sp.tool_name}
        baseline_tools: set[str] = set()
        for seq in baseline.common_tool_sequences:
            baseline_tools.update(seq)

        similarity = jaccard_similarity(session_tools, baseline_tools)
        min_similarity = 1.0 - sequence_diff_threshold
        if similarity < min_similarity:
            violations.append(DriftViolation(
                dimension="tool_sequence",
                z_score=None,
                expected=f"similarity >= {min_similarity:.2f}",
                observed=f"similarity = {similarity:.2f}",
                detail=f"Jaccard similarity {similarity:.2f} below threshold {min_similarity:.2f}",
            ))

    return DriftResult(
        violations=violations,
        drifted=len(violations) > 0,
    )


class DriftDetector:
    """
    Manages baseline lifecycle and evaluates drift at session end.
    Called by IngestPipeline (or a session-end hook) when a session completes.
    """

    def __init__(self, db: StorageBackend, alert_engine: AlertEngine, config: TjConfig) -> None:
        self.db = db
        self.alert_engine = alert_engine
        self.config = config

    def on_session_end(self, agent_id: str, session: SessionRecord) -> None:
        """
        Called when a session reaches status='completed'.
        Builds baseline if enough sessions exist, or evaluates drift against existing baseline.
        Falls back to default AgentConfig (drift.enabled=True) when agent isn't explicitly
        configured, so drift detection works out of the box for any observed agent.
        """
        agent_config = self.config.agents.get(agent_id) or AgentConfig()
        if not agent_config.drift.enabled:
            return

        baseline = self.db.get_baseline(agent_id)

        if baseline is None:
            count = self.db.get_completed_session_count(agent_id)
            if count >= agent_config.drift.baseline_sessions:
                sessions = self.db.get_completed_sessions(
                    agent_id, limit=agent_config.drift.baseline_sessions,
                )
                baseline = build_baseline(agent_id, sessions, self.db)
                self.db.upsert_baseline(baseline)
            return

        result = evaluate_drift(
            session=session,
            baseline=baseline,
            config_threshold=agent_config.drift.token_threshold,
            sequence_diff_threshold=agent_config.drift.tool_sequence_diff,
            db=self.db,
        )

        if result.drifted:
            self.alert_engine.fire(
                alert_type=AlertType.DRIFT_DETECTED,
                span_or_session=session,
                detail={
                    "violations": [vars(v) for v in result.violations],
                    "sessions_sampled": baseline.sessions_sampled,
                },
                severity=Severity.WARNING,
            )
