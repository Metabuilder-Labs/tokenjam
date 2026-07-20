"""Delta-verify for applied cost proposals — the receipts.

A cost proposal is advise-only, so there is no transcript recurrence to
re-count (that is ``relearn_verify``'s job). What we CAN measure is whether the
cost signal the analyzer flagged actually moved after the user marked their
change: fewer dollars on the oversized model, a higher cache hit ratio, fewer
input tokens per call on the bloated step.

The marker is the applied record's ``applied_at`` (an ``Expectation.created_at``
— see ``cost_apply``). For each analyzer we measure a metric over spans in the
POST window ``[marker, now)`` against a comparable PRE window
``[marker - W, marker)`` of the same duration ``W = now - marker``, scoped to
the proposal's ``agent_id`` when it has one. All dollar figures are priced by
the SAME ``core.cost.calculate_cost`` the rest of tokenjam uses (per-model,
per-token-type, BOTH ``cache_tokens`` and ``cache_write_tokens`` included — the
recurring omission this workspace guards against).

Honesty (identical discipline to ``relearn_verify``):

  * **Correlational, never causal.** The user's change is one of many things
    that shifted at the marker. This measures co-occurrence and says so.
  * **A non-improvement is marked, not hidden.** A realized delta that is
    ~zero or negative is a ``regressed`` verdict, never silently kept as if the
    advice paid off.
  * **Admit thin data.** Too little post-window exposure -> ``insufficient_data``
    ("check back later"), not a confident zero.

Never raises: every public function degrades to a conservative
"can't measure this" result.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from tokenjam.core.cost import calculate_cost
from tokenjam.core.optimize.relearn_verify import (
    VERDICT_IMPROVED,
    VERDICT_INSUFFICIENT_DATA,
    VERDICT_REGRESSED,
)
from tokenjam.core.pricing import get_rates

#: Minimum post-window LLM calls before a verdict is trusted (else
#: "insufficient_data"). Cost windows are call-dense, so this gates on calls,
#: not sessions.
MIN_POST_CALLS_FOR_VERDICT = 20
#: A realized dollar delta at or below this (in absolute USD) is treated as
#: no real movement -> regressed, so noise near zero never reads as a win.
MIN_MEANINGFUL_USD = 1e-4

ESTIMATE_BASIS_COST_VERIFY = (
    "realized delta = the analyzer's own cost metric measured over spans after "
    "your change vs a comparable window before it, priced per-model per-token-"
    "type; estimated / correlational with your change, never proof tokenjam's "
    "advice caused it"
)

#: Loop-ledger outcome per verdict — only a measured improvement passes; a
#: measured non-improvement regresses. Insufficient data records nothing.
_OUTCOME_FOR_VERDICT = {
    VERDICT_IMPROVED: "pass",
    VERDICT_REGRESSED: "regress",
}


def _as_aware(value: Any) -> datetime | None:
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


def _window_bounds(marker: datetime, now: datetime) -> tuple[datetime, datetime, datetime, datetime]:
    """(pre_start, pre_end, post_start, post_end) — two equal-duration windows
    either side of the marker, so the two sides are comparable exposure."""
    span = now - marker
    if span < timedelta(0):
        span = timedelta(0)
    return marker - span, marker, marker, now


def _rows_for(
    conn: Any, since: datetime, until: datetime, agent_id: str | None,
    *, model: str | None = None, provider: str | None = None,
    subagent_only: bool = False,
) -> list[tuple]:
    """LLM spans in ``[since, until)`` with optional agent/model/provider
    filters. Returns (session_id, provider, model, input, output, cache,
    cache_write). ``subagent_only`` scopes to Task-dispatched subagent spans
    (``sub_agent_id IS NOT NULL``) — the fan-out the subagent analyzer flags.
    Never raises — a bad query yields ``[]``."""
    clauses = ["start_time >= $1", "start_time < $2", "model IS NOT NULL"]
    params: list[Any] = [since, until]
    if subagent_only:
        clauses.append("sub_agent_id IS NOT NULL")
    if agent_id:
        params.append(agent_id)
        clauses.append(f"agent_id = ${len(params)}")
    if model:
        params.append(model)
        clauses.append(f"model = ${len(params)}")
    if provider:
        params.append(provider)
        clauses.append(f"provider = ${len(params)}")
    where = " AND ".join(clauses)
    try:
        return conn.execute(
            "SELECT session_id, provider, model, "
            "COALESCE(input_tokens,0), COALESCE(output_tokens,0), "
            "COALESCE(cache_tokens,0), COALESCE(cache_write_tokens,0) "
            f"FROM spans WHERE {where}",
            params,
        ).fetchall()
    except Exception:
        return []


def _priced_usd(rows: list[tuple]) -> float:
    """Total USD across rows, priced per-model per-token-type (both cache token
    types included — the workspace's recurring omission guarded here)."""
    total = 0.0
    for _sid, provider, model, in_tok, out_tok, cache_tok, cache_write in rows:
        total += calculate_cost(
            str(provider or "unknown"), str(model or ""),
            int(in_tok or 0), int(out_tok or 0),
            cache_read_tokens=int(cache_tok or 0),
            cache_write_tokens=int(cache_write or 0),
        )
    return total


# --------------------------------------------------------------------------- #
# Per-analyzer metrics. Each returns a dict with a comparable ``value`` for one
# window plus the raw counts the realized-delta math needs.
# --------------------------------------------------------------------------- #

def _downsize_metric(rows: list[tuple], models: set[str]) -> dict[str, Any]:
    """Per-session dollars spent on the oversized model(s)."""
    sessions = {str(r[0] or "") for r in rows}
    oversized = [r for r in rows if str(r[2] or "") in models]
    oversized_usd = _priced_usd(oversized)
    n_sessions = len(sessions) or 0
    value = (oversized_usd / n_sessions) if n_sessions else 0.0
    return {"value": value, "sessions": n_sessions, "calls": len(rows),
            "oversized_usd": oversized_usd}


def _placement_metric(rows: list[tuple]) -> dict[str, Any]:
    """Per-session dollars on the flagged workload group.

    Usage records carry no batch/service-tier marker today, so the receipt is a
    spend drop on the same workload after the user marks the change applied,
    with no claim about what caused it.
    """
    sessions = {str(r[0] or "") for r in rows}
    n_sessions = len(sessions)
    total_usd = _priced_usd(rows)
    return {"value": (total_usd / n_sessions) if n_sessions else 0.0,
            "sessions": n_sessions, "calls": len(rows), "total_usd": total_usd}


def _cache_metric(rows: list[tuple]) -> dict[str, Any]:
    """Cache-read efficacy = cache_tokens / (input + cache) for the flagged
    (provider, model), plus the input-vs-cache priced value of the gap."""
    input_tok = sum(int(r[3] or 0) for r in rows)
    cache_tok = sum(int(r[5] or 0) for r in rows)
    total = input_tok + cache_tok
    efficacy = (cache_tok / total) if total else 0.0
    return {"value": efficacy, "sessions": len({str(r[0] or "") for r in rows}),
            "calls": len(rows), "input_tokens": input_tok, "cache_tokens": cache_tok}


def _trim_metric(rows: list[tuple]) -> dict[str, Any]:
    """Average input tokens per call on the flagged step."""
    input_tok = sum(int(r[3] or 0) for r in rows)
    calls = len(rows)
    value = (input_tok / calls) if calls else 0.0
    return {"value": value, "sessions": len({str(r[0] or "") for r in rows}),
            "calls": calls, "input_tokens": input_tok}


def _dominant_model(rows: list[tuple]) -> tuple[str, str]:
    """(provider, model) of the most-called model in the rows — used to price a
    token delta at the right input rate. ("unknown","") when rows are empty."""
    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (str(r[1] or "unknown"), str(r[2] or ""))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "unknown", ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _verdict(realized_usd_delta: float, post_calls: int) -> tuple[str, str]:
    """(verdict, reason). Insufficient below the exposure gate; else improved
    iff the realized dollar delta cleared the meaningful-movement floor."""
    if post_calls < MIN_POST_CALLS_FOR_VERDICT:
        return (
            VERDICT_INSUFFICIENT_DATA,
            f"only {post_calls} call(s) observed since you marked this applied; "
            f"check back after at least {MIN_POST_CALLS_FOR_VERDICT}",
        )
    if realized_usd_delta > MIN_MEANINGFUL_USD:
        return (
            VERDICT_IMPROVED,
            f"measured ${realized_usd_delta:.4f} lower on the flagged cost "
            f"dimension across {post_calls} call(s) after your change",
        )
    return (
        VERDICT_REGRESSED,
        f"no measured improvement on the flagged cost dimension (realized delta "
        f"${realized_usd_delta:.4f}) across {post_calls} call(s) after your change",
    )


def measure_cost_delta(
    conn: Any | None,
    record: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The realized-delta receipt for one applied cost record. Never raises.

    Returns a dict merged onto the record's ``verify`` block: ``verdict``,
    ``reason``, ``realized_usd_delta``, ``realized_tokens_delta``, the pre/post
    metric values + call counts, and ``estimate_basis``.
    """
    from tokenjam.utils.time_parse import utcnow

    empty = {
        "verdict": VERDICT_INSUFFICIENT_DATA, "reason": None,
        "realized_usd_delta": None, "realized_tokens_delta": None,
        "pre_value": None, "post_value": None,
        "pre_sessions": None, "post_sessions": None,
        "estimate_basis": ESTIMATE_BASIS_COST_VERIFY,
        "last_checked_at": (now or utcnow()).isoformat(),
    }

    marker = _as_aware(record.get("applied_at"))
    if marker is None:
        return {**empty, "reason": "no usable applied-at marker on this record"}
    if conn is None:
        return {**empty, "reason": "no database connection"}

    now = now or utcnow()
    pre_start, pre_end, post_start, post_end = _window_bounds(marker, now)
    analyzer = str(record.get("analyzer") or "")
    agent_id = str(record.get("agent_id") or "") or None
    target = record.get("target_key") or {}

    realized_usd = 0.0
    realized_tokens: int | None = None

    if analyzer == "downsize":
        models = {str(m) for m in (target.get("models") or [])}
        pre = _downsize_metric(_rows_for(conn, pre_start, pre_end, agent_id), models)
        post_rows = _rows_for(conn, post_start, post_end, agent_id)
        post = _downsize_metric(post_rows, models)
        # Per-session oversized dollars that fell, projected across post sessions.
        realized_usd = max(0.0, pre["value"] - post["value"]) * post["sessions"]
        post_calls = post["calls"]

    elif analyzer == "cache":
        provider = str(target.get("provider") or "") or None
        model = str(target.get("model") or "") or None
        pre = _cache_metric(_rows_for(conn, pre_start, pre_end, agent_id,
                                      model=model, provider=provider))
        post_rows = _rows_for(conn, post_start, post_end, agent_id,
                              model=model, provider=provider)
        post = _cache_metric(post_rows)
        # Tokens shifted from fresh input to cache, priced at the input-vs-cache
        # rate delta for this model.
        efficacy_gain = max(0.0, post["value"] - pre["value"])
        shifted = efficacy_gain * post["input_tokens"]
        realized_tokens = int(shifted)
        rates = get_rates(provider or "unknown", model or "")
        rate_delta = 0.0
        if rates is not None:
            rate_delta = max(0.0, rates.input_per_mtok - rates.cache_read_per_mtok)
        realized_usd = (shifted / 1_000_000) * rate_delta
        post_calls = post["calls"]

    elif analyzer == "trim":
        pre = _trim_metric(_rows_for(conn, pre_start, pre_end, agent_id))
        post_rows = _rows_for(conn, post_start, post_end, agent_id)
        post = _trim_metric(post_rows)
        # Fewer input tokens/call, across post calls, priced at the dominant
        # model's input rate.
        per_call_drop = max(0.0, pre["value"] - post["value"])
        saved_tokens = per_call_drop * post["calls"]
        realized_tokens = int(saved_tokens)
        prov, model = _dominant_model(post_rows)
        rates = get_rates(prov, model)
        input_rate = rates.input_per_mtok if rates is not None else 0.0
        realized_usd = (saved_tokens / 1_000_000) * input_rate
        post_calls = post["calls"]

    elif analyzer == "subagent":
        # Fan-out model-mix cost delta: per-session dollars spent on the
        # oversized model(s) across SUBAGENT spans only (sub_agent_id NOT NULL),
        # before vs after the marker. Same shape as downsize, scoped to the
        # Task-dispatched fan-out the subagent analyzer flags.
        models = {str(m) for m in (target.get("models") or [])}
        pre = _downsize_metric(
            _rows_for(conn, pre_start, pre_end, agent_id, subagent_only=True), models)
        post_rows = _rows_for(conn, post_start, post_end, agent_id, subagent_only=True)
        post = _downsize_metric(post_rows, models)
        realized_usd = max(0.0, pre["value"] - post["value"]) * post["sessions"]
        post_calls = post["calls"]

    elif analyzer == "placement":
        # Spend on the flagged workload group, per session, before versus after
        # the marker. Scoped by agent where the card names a single workload;
        # a multi-workload card measures the whole set it listed.
        agents = [str(a) for a in (target.get("agents") or []) if a]
        scope_agent = agent_id or (agents[0] if len(agents) == 1 else None)
        pre = _placement_metric(_rows_for(conn, pre_start, pre_end, scope_agent))
        post_rows = _rows_for(conn, post_start, post_end, scope_agent)
        post = _placement_metric(post_rows)
        realized_usd = max(0.0, pre["value"] - post["value"]) * post["sessions"]
        post_calls = post["calls"]

    else:
        return {**empty, "reason": f"unknown cost analyzer {analyzer!r}"}

    realized_usd = round(realized_usd, 6)
    verdict, reason = _verdict(realized_usd, post_calls)
    # Below the exposure gate we don't claim a delta.
    if verdict == VERDICT_INSUFFICIENT_DATA:
        realized_usd = None  # type: ignore[assignment]
        realized_tokens = None

    return {
        "verdict": verdict,
        "reason": reason,
        "realized_usd_delta": realized_usd,
        "realized_tokens_delta": realized_tokens,
        "pre_value": round(pre["value"], 6),
        "post_value": round(post["value"], 6),
        "pre_sessions": pre["sessions"],
        "post_sessions": post["sessions"],
        "estimate_basis": ESTIMATE_BASIS_COST_VERIFY,
        "last_checked_at": now.isoformat(),
    }


def verify_record(
    db_or_conn: Any,
    config: Any,
    record: dict[str, Any],
    *,
    record_outcome: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Measure one record's realized delta, persist it into the record's
    ``verify`` block, and (when decisive) write the pass/regress outcome to the
    loop's fix-history ledger via ``record_expectation_run``. Never raises."""
    from tokenjam.core.loop import _resolve_conn, record_expectation_run
    from tokenjam.core.optimize import cost_apply

    try:
        conn = _resolve_conn(db_or_conn)
    except Exception:
        conn = None

    verify = measure_cost_delta(conn, record, now=now)
    try:
        cost_apply.set_verify(config, record["id"], verify)
    except Exception:
        pass

    outcome = _OUTCOME_FOR_VERDICT.get(verify.get("verdict", ""))
    expectation_id = record.get("expectation_id")
    if record_outcome and outcome is not None and expectation_id:
        try:
            record_expectation_run(
                db_or_conn, expectation_id,
                outcome=outcome, note=f"cost verify: {verify.get('reason', '')}",
            )
        except Exception:
            pass  # a ledger write failure must not sink the measurement
    return verify


def rescan_all(
    db_or_conn: Any,
    config: Any,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    """Recompute the realized delta for every applied (non-reverted) cost
    record — the entry point the Review-inbox refresh rides on the SAME cadence
    as the proposal recompute. Never raises; one bad record is skipped."""
    from tokenjam.core.optimize import cost_apply

    checked = updated = 0
    for rec in cost_apply.list_applied(config):
        if rec.get("state") == "reverted":
            continue
        checked += 1
        try:
            verify_record(db_or_conn, config, rec, now=now)
            updated += 1
        except Exception:
            continue
    return {"checked": checked, "updated": updated}


def cost_compound_ledger(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The Applied section's cost summary: total realized dollars across every
    VERIFIED, improved (non-reverted) cost fix, plus a verdict breakdown. The
    dollar-denominated sibling of ``relearn_verify.compound_ledger``."""
    total_usd = 0.0
    total_tokens = 0
    verified = improved = regressed = insufficient = 0
    for rec in records:
        if rec.get("state") == "reverted":
            continue
        verify = rec.get("verify") or {}
        verdict = verify.get("verdict")
        if verdict is None:
            continue
        verified += 1
        if verdict == VERDICT_IMPROVED:
            improved += 1
            total_usd += verify.get("realized_usd_delta") or 0.0
            total_tokens += verify.get("realized_tokens_delta") or 0
        elif verdict == VERDICT_REGRESSED:
            regressed += 1
        elif verdict == VERDICT_INSUFFICIENT_DATA:
            insufficient += 1
    return {
        "total_realized_usd": round(total_usd, 6),
        "total_realized_tokens": total_tokens,
        "verified_count": verified,
        "improved_count": improved,
        "regressed_count": regressed,
        "insufficient_data_count": insufficient,
        "estimate_basis": ESTIMATE_BASIS_COST_VERIFY,
    }
