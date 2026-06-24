"""Audit log + savings meter persistence for proxy decisions (#221).

The observer (`observer.py`) holds recent decisions in memory; this module is
the durable sink. It converts each recorded :class:`ProxyObservation` into an
append-only ``policy_decisions`` row and, for eligible POLICY-path decisions, a
``savings_ledger`` row.

**Honesty (Critical Rule 14).** Suggest mode enforces NOTHING. The savings meter
therefore records ESTIMATED-RECOVERABLE / "would-have-saved" amounts — what each
policy WOULD have recovered *if it had been enforced* — never realized savings.
``realized`` is always False; dollar figures are api-only; the ``unvalidated``
label rides through from the envelope. :func:`reconcile_savings` reports the
estimate against actual spend from the SAME source ``tj cost`` reads
(:meth:`get_cost_summary`), and its summary never uses the word "saved".

This module lives in the proxy package (it may import core); core never imports
it (package-dependency rule).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from tokenjam.core.models import (
    CostFilters,
    NormalizedSpan,
    PolicyDecisionFilters,
    PolicyDecisionRecord,
    SavingsLedgerEntry,
    SpanKind,
    SpanStatus,
)
from tokenjam.otel.semconv import TjAttributes
from tokenjam.utils.ids import new_span_id, new_trace_id, new_uuid
from tokenjam.utils.time_parse import utcnow

# Span name for the proxy's self-observation spans (#223).
POLICY_DECISION_SPAN_NAME = "tokenjam.policy.decision"

logger = logging.getLogger("tokenjam.proxy")

UNVALIDATED_LABEL = "unvalidated"

# The load-bearing honesty string. Suggest mode enforces nothing, so this is a
# counterfactual estimate, never a realized number. Never the word "saved".
SAVINGS_DISCLAIMER = (
    "Estimated recoverable — suggest mode enforces nothing, so this is what these "
    "policies WOULD have recovered if enforced, not realized savings. Unvalidated."
)


def _billing_period(ts: datetime) -> str:
    return ts.strftime("%Y-%m")


def _extract_savings(envelope: dict | None) -> tuple[float, int, str]:
    """Sum the would-have-recovered estimate the evaluators put in `details`.

    A `kind` (e.g. a future budget_cap, #222) advertises its counterfactual via
    `details.estimated_recoverable_usd` / `estimated_recoverable_tokens` /
    `estimate_basis`. The shipped `noop` kind advertises nothing → 0 (honest:
    a no-op would recover nothing).
    """
    if not envelope:
        return 0.0, 0, ""
    usd = 0.0
    tokens = 0
    bases: list[str] = []
    for ev in envelope.get("evaluations", []):
        details = ev.get("details") or {}
        usd += float(details.get("estimated_recoverable_usd", 0) or 0)
        tokens += int(details.get("estimated_recoverable_tokens", 0) or 0)
        basis = details.get("estimate_basis")
        if basis:
            bases.append(str(basis))
    return usd, tokens, "; ".join(bases)


class AuditSink:
    """Persists each recorded observation to the audit log + savings ledger.

    Wired into :class:`ProxyObserver` as its ``sink``. Best-effort: a persistence
    failure is logged and swallowed so it never breaks proxy pass-through.
    """

    def __init__(self, db: Any, pipeline: Any = None) -> None:
        self.db = db
        # Optional IngestPipeline (#223): when provided, the self-observation
        # span flows through it (running the cost/etc. hooks) exactly like any
        # other span. Falls back to a direct db.insert_span otherwise.
        self.pipeline = pipeline

    def __call__(self, obs: Any) -> None:
        try:
            self._persist(obs)
        except Exception:  # noqa: BLE001 — audit must never break the proxy
            logger.exception("policy-decision audit persistence failed (ignored)")

    def _persist(self, obs: Any) -> None:
        env = getattr(obs, "policy", None)
        is_policy = obs.decision == "policy"
        ts = utcnow()

        evals = (env or {}).get("evaluations", []) if env else []
        primary = evals[0] if evals else {}
        # Observe-only due to provider TOS (subscription) = "not permitted to act",
        # distinct from a policy-path noop = "we chose not to act".
        passthrough_tos = (not is_policy) and obs.pricing_mode == "subscription"

        usd, tokens, basis = _extract_savings(env) if is_policy else (0.0, 0, "")

        decision_id = new_uuid()
        self.db.insert_policy_decision(PolicyDecisionRecord(
            decision_id=decision_id,
            ts=ts,
            provider=obs.provider,
            pricing_mode=obs.pricing_mode,
            gate_decision=obs.decision,
            path=obs.path,
            would_action=(env or {}).get("overall_action", "noop") if env else "noop",
            policy_name=primary.get("policy_name"),
            policy_kind=primary.get("kind"),
            passthrough_tos=passthrough_tos,
            label=(env or {}).get("label", UNVALIDATED_LABEL),
            suggest_only=bool(getattr(obs, "suggest_only", True)),
            envelope=env,
        ))

        # Savings accrue ONLY on the POLICY path (api/usage-billed). Observe-only
        # traffic is never acted on, so it can never "would-have-saved".
        if is_policy:
            self.db.insert_savings_entry(SavingsLedgerEntry(
                ledger_id=new_uuid(),
                decision_id=decision_id,
                ts=ts,
                provider=obs.provider,
                pricing_mode=obs.pricing_mode,
                policy_name=primary.get("policy_name"),
                would_action=(env or {}).get("overall_action", "noop"),
                estimated_recoverable_usd=usd,
                estimated_recoverable_tokens=tokens,
                estimate_basis=basis,
                billing_period=_billing_period(ts),
                label=(env or {}).get("label", UNVALIDATED_LABEL),
                realized=False,  # suggest mode: NEVER realized
            ))

        # Self-observation span (#223): TokenJam observing its own policy decision,
        # under the tokenjam.policy.* attribute namespace, so the web UI + drift
        # see enforcement activity. Best-effort — never break the proxy.
        try:
            span = policy_decision_span(obs, ts=ts, estimated_recoverable_usd=usd)
            if self.pipeline is not None:
                self.pipeline.process(span)
            else:
                self.db.insert_span(span)
        except Exception:  # noqa: BLE001
            logger.exception("policy self-observation span emit failed (ignored)")


def policy_decision_span(obs: Any, *, ts: Any = None,
                         estimated_recoverable_usd: float = 0.0) -> NormalizedSpan:
    """Build the `tokenjam.policy.decision` self-observation span for a decision.

    Pure: carries the gate/policy outcome under the `tokenjam.policy.*` attribute
    namespace (semconv constants — never hardcoded strings). Suggest mode only,
    so REALIZED is always False and ESTIMATED_RECOVERABLE_USD is would-have-saved.
    """
    ts = ts or utcnow()
    env = getattr(obs, "policy", None)
    evals = (env or {}).get("evaluations", []) if env else []
    primary = evals[0] if evals else {}
    is_policy = obs.decision == "policy"
    attrs = {
        TjAttributes.POLICY_DECISION: obs.decision,
        TjAttributes.POLICY_ACTION: (env or {}).get("overall_action", "noop") if env else "noop",
        TjAttributes.POLICY_MODE: "suggest",
        TjAttributes.POLICY_LABEL: (env or {}).get("label", UNVALIDATED_LABEL),
        TjAttributes.POLICY_PRICING_MODE: obs.pricing_mode,
        TjAttributes.POLICY_PASSTHROUGH_TOS: (not is_policy) and obs.pricing_mode == "subscription",
        TjAttributes.POLICY_REALIZED: False,  # suggest mode: never realized
    }
    if primary.get("policy_name"):
        attrs[TjAttributes.POLICY_NAME] = primary["policy_name"]
    if primary.get("kind"):
        attrs[TjAttributes.POLICY_KIND] = primary["kind"]
    if is_policy:
        attrs[TjAttributes.POLICY_ESTIMATED_RECOVERABLE_USD] = estimated_recoverable_usd
    return NormalizedSpan(
        span_id=new_span_id(),
        trace_id=new_trace_id(),
        name=POLICY_DECISION_SPAN_NAME,
        kind=SpanKind.INTERNAL,
        status_code=SpanStatus.OK,
        start_time=ts,
        end_time=ts,
        duration_ms=0.0,
        attributes=attrs,
        provider=obs.provider,
        billing_account=obs.provider,
    )


@dataclass(frozen=True)
class SavingsSummary:
    """The savings meter — estimated-recoverable reconciled vs actual spend.

    Every field is an ESTIMATE / would-have figure (suggest mode enforces
    nothing). ``realized`` is always False and there is no "saved" field.
    """
    since:                        datetime | None
    until:                        datetime | None
    decisions:                    int
    estimated_recoverable_usd:    float
    estimated_recoverable_tokens: int
    actual_spend_usd:             float       # from get_cost_summary — the tj cost source
    estimated_recoverable_pct:    float | None  # est_recoverable / actual_spend * 100
    realized:                     bool = False
    label:                        str = UNVALIDATED_LABEL
    disclaimer:                   str = SAVINGS_DISCLAIMER

    def to_dict(self) -> dict:
        return {
            "decisions": self.decisions,
            "estimated_recoverable_usd": round(self.estimated_recoverable_usd, 6),
            "estimated_recoverable_tokens": self.estimated_recoverable_tokens,
            "actual_spend_usd": round(self.actual_spend_usd, 6),
            "estimated_recoverable_pct": (
                round(self.estimated_recoverable_pct, 2)
                if self.estimated_recoverable_pct is not None else None
            ),
            "realized": self.realized,
            "label": self.label,
            "disclaimer": self.disclaimer,
        }


def reconcile_savings(
    db: Any, *, since: datetime | None = None, until: datetime | None = None,
    provider: str | None = None,
) -> SavingsSummary:
    """Reconcile the savings ledger's estimate against actual spend.

    Actual spend comes from :meth:`get_cost_summary` — the SAME source ``tj
    cost`` reads — so the meter always reconciles to it. The returned figure is
    "estimated recoverable" (what enforcing these policies WOULD have recovered),
    never a realized saving.
    """
    entries = db.get_savings_entries(
        PolicyDecisionFilters(since=since, until=until, provider=provider, limit=100_000)
    )
    est_usd = sum(e.estimated_recoverable_usd for e in entries)
    est_tokens = sum(e.estimated_recoverable_tokens for e in entries)

    cost_rows = db.get_cost_summary(CostFilters(since=since, until=until, group_by="day"))
    actual_spend = sum(r.cost_usd for r in cost_rows)

    pct = (est_usd / actual_spend * 100.0) if actual_spend > 0 else None
    return SavingsSummary(
        since=since, until=until, decisions=len(entries),
        estimated_recoverable_usd=est_usd, estimated_recoverable_tokens=est_tokens,
        actual_spend_usd=actual_spend, estimated_recoverable_pct=pct,
    )


def decision_to_display_dict(d: PolicyDecisionRecord) -> dict:
    """A row-friendly dict for `tj policy decisions` (DB-backed view)."""
    return {
        "ts": d.ts.isoformat() if hasattr(d.ts, "isoformat") else str(d.ts),
        "provider": d.provider,
        "path": d.path,
        "gate_decision": d.gate_decision,
        "would_action": d.would_action,
        "policy_name": d.policy_name,
        "passthrough_tos": d.passthrough_tos,
        "label": d.label,
    }
