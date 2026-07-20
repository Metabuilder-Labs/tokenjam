"""Component G1: the cumulative "verified saved" receipts — the measured
twin of Component E's estimated-recoverable rollup
(``cost_proposals.estimated_recoverable_rollup``).

Combines two independently-verified ledgers:

  * the relearn ledger (``relearn_verify.compound_ledger``) — realized
    TOKENS saved across workspace-write fixes (notes/skills/hooks/wrappers/
    config), keyed by recurrence rate.
  * the cost-verify ledger (``cost_verify.cost_compound_ledger``) — realized
    TOKENS AND DOLLARS across advise-only cost proposals, priced per-model
    per-token-type by ``core.cost.calculate_cost``.

The two ledgers are summed WITHIN their own unit (tokens with tokens,
dollars with dollars) but never cross-converted. A relearn fix has no model
attached — it isn't scoped to one provider/model, so there is no real
per-token $ rate to price its savings at; inventing one would fabricate
precision this house style forbids. ``verified_saved_usd`` is therefore the
cost-verify ledger's own real, per-model-priced dollars only;
``verified_saved_tokens`` adds the relearn ledger's tokens on top of the
cost ledger's tokens (both are genuine token counts — safely additive),
broken out individually below so neither contribution is silently dropped.

Regressed / no-change / insufficient-data counts from BOTH ledgers are
carried into the combined breakdown, never hidden — the honesty is the
feature (spec §2b Component G1). Never raises: a malformed record list just
contributes zeros (the two source ``compound_ledger`` functions are
themselves never-raise).
"""
from __future__ import annotations

from typing import Any

from tokenjam.core.optimize import cost_verify, relearn_verify

#: The confidence tag every G1 figure carries — the counterpart to
#: ``cost_proposals.COST_ESTIMATE_CONFIDENCE`` ("estimated"). Kept as a
#: distinct string so nothing can accidentally render the two ledgers'
#: numbers under the same label.
ESTIMATE_CONFIDENCE_MEASURED = "measured"

RECEIPTS_BASIS = (
    "verified_saved_usd = the cost-verify ledger's realized dollars only "
    "(real, per-model per-token-type pricing, only counted once a fix's "
    "post-apply exposure window clears the verify gate). "
    "verified_saved_tokens = relearn-ledger tokens + cost-verify-ledger "
    "tokens (both real token counts, safely additive). Relearn's token "
    "savings are never converted to dollars: a relearn fix has no single "
    "model to price at. Measured, correlational with your change, never a "
    "causal claim; regressed / no-change / insufficient-data fixes are "
    "counted here, not hidden."
)


def verified_saved_summary(
    relearn_records: list[dict[str, Any]],
    cost_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """The dashboard's single "verified saved to date: $Y" figure (G1) — the
    measured twin of Component E's estimated-recoverable rollup.

    Pure: takes the two ledgers' raw applied-fix record lists
    (``relearn_apply.list_applied`` / ``cost_apply.list_applied``) and
    returns the combined receipt. Never raises — a bad record list degrades
    to a zeroed, ``verified_count == 0`` summary rather than an exception.
    """
    relearn_ledger = relearn_verify.compound_ledger(relearn_records or [])
    cost_ledger = cost_verify.cost_compound_ledger(cost_records or [])

    relearn_tokens = int(relearn_ledger.get("total_realized_tokens_saved") or 0)
    cost_tokens = int(cost_ledger.get("total_realized_tokens") or 0)
    cost_usd = round(float(cost_ledger.get("total_realized_usd") or 0.0), 6)

    return {
        "verified_saved_usd": cost_usd,
        "verified_saved_tokens": relearn_tokens + cost_tokens,
        "relearn_tokens_saved": relearn_tokens,
        "cost_tokens_saved": cost_tokens,
        "cost_usd_saved": cost_usd,
        "verified_count": (
            int(relearn_ledger.get("verified_count") or 0)
            + int(cost_ledger.get("verified_count") or 0)
        ),
        "improved_count": (
            int(relearn_ledger.get("improved_count") or 0)
            + int(cost_ledger.get("improved_count") or 0)
        ),
        "regressed_count": (
            int(relearn_ledger.get("regressed_count") or 0)
            + int(cost_ledger.get("regressed_count") or 0)
        ),
        # cost_verify has no no_change/enforcement_disabled verdicts (its
        # `_verdict` only ever returns improved/regressed/insufficient_data)
        # — these two counts are relearn-only, carried through unchanged.
        "no_change_count": int(relearn_ledger.get("no_change_count") or 0),
        "enforcement_disabled_count": int(relearn_ledger.get("enforcement_disabled_count") or 0),
        "insufficient_data_count": (
            int(relearn_ledger.get("insufficient_data_count") or 0)
            + int(cost_ledger.get("insufficient_data_count") or 0)
        ),
        "estimate_confidence": ESTIMATE_CONFIDENCE_MEASURED,
        "estimate_basis": RECEIPTS_BASIS,
        # The two source ledgers, untouched — a caller that wants the
        # relearn-only or cost-only view (e.g. the existing Applied
        # sections) doesn't need to recompute them separately.
        "relearn_ledger": relearn_ledger,
        "cost_ledger": cost_ledger,
    }
