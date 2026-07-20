"""Verify, OTel lane: did a failure signature actually recur less after a fix?

The transcript verify (``relearn_verify``) measures recurrence by walking on-disk
sessions newer than an applied fix's ``applied_at``. That works because tokenjam
itself applied the fix and recorded when. For a workspace-less agent there IS no
apply step (see ``relearn_otel``), so there is no ``applied_at`` to key on: the
user deployed their own fix, on their own schedule.

The fix marker therefore comes from the loop primitive that already exists for
exactly this ("I changed something, watch whether it holds"): an ``Expectation``
from ``core/loop.py``, promoted off the failure cluster. Its ``created_at`` IS
"fix deployed at T". The outcome is written back through the same loop ledger via
``record_expectation_run``, so an OTel fix accumulates the same pass/regress
history a Claude Code fix does, in the same tables.

Honesty carries over unchanged from ``relearn_verify``:

  * **Normalize by exposure.** Rates are occurrences-per-session on both sides of
    T, never raw counts, so a long post-window can't manufacture a regression.
  * **Admit what can't be measured.** A distilled (LLM-merged) family has no
    deterministic re-matcher, so it reports ``insufficient_data`` rather than
    risk a false "improved". Same ``_matcher_for`` the transcript lane uses.
  * **Correlational, never causal.** The user's deploy is one of many things that
    changed at T. This measures co-occurrence, and says so.

Never raises: every public function degrades to a conservative
"can't measure this" result.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tokenjam.core.optimize.analyzers.relearn import (
    FailureEpisode,
    GROUNDED_TOKENS_PER_OCCURRENCE,
)
from tokenjam.core.optimize.relearn_otel import ERROR_STATUS, _repo_label
from tokenjam.core.optimize.relearn_verify import (
    VERDICT_IMPROVED,
    VERDICT_REGRESSED,
    _matcher_for,
    compute_verdict,
)

ESTIMATE_BASIS_OTEL_VERIFY = (
    "occurrences-per-session for this failure signature, before vs after the "
    "recorded fix marker, measured over stored spans — a correlation with your "
    "deploy, not proof tokenjam's advice caused the change"
)


def _as_aware(value: Any) -> datetime | None:
    """A tz-aware datetime, or None. Naive input is read as UTC (Rule 9)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def measure_span_recurrence(
    conn: Any | None,
    *,
    signature: str | None = None,
    family_key: str | None = None,
    agent_id: str | None = None,
    at: datetime,
) -> dict[str, Any]:
    """Count matching-signature occurrences and session exposure either side of
    ``at``, over failing spans.

    ``agent_id`` scopes to one service when given (the usual case: an expectation
    is promoted for a specific agent). Returns ``measurable=False`` with a reason
    when the signature has no deterministic re-matcher, mirroring the transcript
    lane. Never raises.
    """
    empty = {
        "pre_sessions": 0, "pre_occurrences": 0,
        "post_sessions": 0, "post_occurrences": 0,
        "measurable": False, "reason": None,
    }

    marker = _as_aware(at)
    if marker is None:
        return {**empty, "reason": "no usable fix marker timestamp"}

    matcher, measurable, reason = _matcher_for(
        {"family_key": family_key, "signature": signature}
    )
    if not measurable:
        return {**empty, "reason": reason}

    if conn is None:
        return {**empty, "reason": "no database connection"}

    sql = (
        "SELECT session_id, agent_id, tool_name, name, status_message, start_time "
        "FROM spans WHERE status_code = $1 AND start_time IS NOT NULL"
    )
    params: list[Any] = [ERROR_STATUS]
    if agent_id:
        sql += " AND agent_id = $2"
        params.append(agent_id)

    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        return {**empty, "reason": "couldn't read spans for this signature"}

    pre_sessions: set[str] = set()
    post_sessions: set[str] = set()
    pre_occurrences = 0
    post_occurrences = 0

    for session_id, row_agent, tool_name, name, status_message, start_time in rows:
        ts = _as_aware(start_time)
        if ts is None:
            continue
        is_post = ts >= marker
        sid = str(session_id or "")
        (post_sessions if is_post else pre_sessions).add(sid)

        error_text = (status_message or "").strip() or (name or "").strip()
        if not error_text:
            continue
        episode = FailureEpisode(
            session_id=sid,
            repo=_repo_label(row_agent),
            ts=ts.isoformat(),
            tool_name=str(tool_name or name or "unknown"),
            label="",
            error_text=error_text,
            kind="act",
            is_retry=False,
            depth=0,
        )
        try:
            if not matcher(episode):
                continue
        except Exception:
            continue
        if is_post:
            post_occurrences += 1
        else:
            pre_occurrences += 1

    return {
        "pre_sessions": len(pre_sessions),
        "pre_occurrences": pre_occurrences,
        "post_sessions": len(post_sessions),
        "post_occurrences": post_occurrences,
        "measurable": True,
        "reason": None,
    }


#: Loop-ledger outcome per verify verdict. Only a measured drop is a "pass" and
#: only a measured non-drop is a "regress"; everything else is honestly unknown.
_OUTCOME_FOR_VERDICT = {
    VERDICT_IMPROVED: "pass",
    VERDICT_REGRESSED: "regress",
}


def verify_otel_expectation(
    db_or_conn: Any,
    expectation: Any,
    *,
    signature: str | None = None,
    family_key: str | None = None,
    tokens_per_occurrence: int = GROUNDED_TOKENS_PER_OCCURRENCE,
    record: bool = True,
) -> dict[str, Any]:
    """Verify one OTel-lane fix against its expectation marker.

    ``expectation`` is a ``core.loop.Expectation`` promoted off the cluster; its
    ``created_at`` is the "fix deployed at T" marker and its ``agent_id`` scopes
    the measurement. When ``record`` is True and the verdict is decisive, the
    outcome is appended to the expectation's fix-history through
    ``core.loop.record_expectation_run`` (``pass``/``regress``), so the OTel lane
    writes into the same ledger the rest of the loop reads.

    Returns the verdict dict plus ``estimate_basis`` and, when written,
    ``run_ledger_id``. Never raises.
    """
    from tokenjam.core.loop import _resolve_conn, record_expectation_run

    try:
        conn = _resolve_conn(db_or_conn)
    except Exception:
        conn = None

    marker = _as_aware(getattr(expectation, "created_at", None))
    agent_id = getattr(expectation, "agent_id", None)

    if marker is None:
        result = compute_verdict(
            rung=None, enforcement=None,
            baseline_occurrences=None, baseline_total_sessions=None,
            baseline_sessions=None, post_occurrences=0, post_sessions=0,
            measurable=False,
            unmeasurable_reason="the expectation carries no usable created_at marker",
            tokens_per_occurrence=tokens_per_occurrence,
        )
        result["estimate_basis"] = ESTIMATE_BASIS_OTEL_VERIFY
        return result

    measurement = measure_span_recurrence(
        conn, signature=signature, family_key=family_key,
        agent_id=agent_id, at=marker,
    )

    # rung=None keeps the enforcement gate out of it: an OTel fix has no
    # tokenjam-installed hook whose enabled/disabled state we could check.
    result = compute_verdict(
        rung=None,
        enforcement=None,
        baseline_occurrences=measurement["pre_occurrences"],
        baseline_total_sessions=measurement["pre_sessions"],
        baseline_sessions=measurement["pre_sessions"],
        post_occurrences=measurement["post_occurrences"],
        post_sessions=measurement["post_sessions"],
        measurable=measurement["measurable"],
        unmeasurable_reason=measurement.get("reason"),
        tokens_per_occurrence=tokens_per_occurrence,
    )
    result["estimate_basis"] = ESTIMATE_BASIS_OTEL_VERIFY
    result["pre_occurrences"] = measurement["pre_occurrences"]
    result["pre_sessions"] = measurement["pre_sessions"]
    result["fix_marker_at"] = marker.isoformat()

    outcome = _OUTCOME_FOR_VERDICT.get(result.get("verdict", ""))
    if record and outcome is not None:
        note = (
            f"OTel verify: {result.get('reason', '')}. "
            f"{measurement['pre_occurrences']} occurrence(s) over "
            f"{measurement['pre_sessions']} session(s) before the fix marker; "
            f"{measurement['post_occurrences']} over "
            f"{measurement['post_sessions']} after."
        )
        try:
            entry = record_expectation_run(
                db_or_conn,
                getattr(expectation, "expectation_id", ""),
                outcome=outcome,
                note=note,
            )
            result["run_ledger_id"] = entry.run_ledger_id
        except Exception:
            pass   # a ledger write failure must not sink the measurement itself

    return result
