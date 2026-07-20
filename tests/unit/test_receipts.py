"""Unit tests for Component G1's combined "verified saved" receipts
(`core.optimize.receipts.verified_saved_summary`) — the measured twin of
Component E's estimated-recoverable rollup.

Pure: both source ledgers (`relearn_verify.compound_ledger` /
`cost_verify.cost_compound_ledger`) already operate on plain dicts, so these
tests build minimal applied-fix records by hand rather than round-tripping
through a real apply flow (mirrors `test_cost_compound_ledger_sums_realized_
dollars` in `test_cost_proposals.py`).
"""
from __future__ import annotations

from tokenjam.core.optimize import cost_proposals, receipts


def _relearn_rec(verdict, tokens=0, state="applied"):
    return {"state": state, "rung": 1, "verify": {"verdict": verdict, "realized_tokens_saved": tokens}}


def _cost_rec(verdict, usd=0.0, tokens=0, state="applied"):
    return {"state": state, "verify": {
        "verdict": verdict, "realized_usd_delta": usd, "realized_tokens_delta": tokens,
    }}


def test_summary_combines_both_ledgers_tokens_and_dollars():
    relearn_records = [_relearn_rec("improved", tokens=1500)]
    cost_records = [_cost_rec("improved", usd=0.5, tokens=1000)]

    summary = receipts.verified_saved_summary(relearn_records, cost_records)

    # Dollars come ONLY from the cost-verify ledger — relearn tokens are
    # never force-converted to a $ figure (no per-model rate to price them at).
    assert summary["verified_saved_usd"] == 0.5
    assert summary["cost_usd_saved"] == 0.5
    # Tokens are additive across both ledgers — both are real token counts.
    assert summary["verified_saved_tokens"] == 2500
    assert summary["relearn_tokens_saved"] == 1500
    assert summary["cost_tokens_saved"] == 1000
    assert summary["improved_count"] == 2
    assert summary["verified_count"] == 2


def test_summary_empty_state_when_nothing_verified_yet():
    summary = receipts.verified_saved_summary([], [])
    assert summary["verified_saved_usd"] == 0.0
    assert summary["verified_saved_tokens"] == 0
    assert summary["verified_count"] == 0
    assert summary["improved_count"] == 0


def test_summary_shows_regressed_and_no_change_not_hidden():
    # A regressed relearn fix, a regressed cost fix, and a no_change relearn
    # fix must all be COUNTED in the breakdown — never dropped from the
    # figure just because they didn't pay off. "The honesty is the feature."
    relearn_records = [
        _relearn_rec("improved", tokens=500),
        _relearn_rec("regressed"),
        _relearn_rec("no_change"),
        _relearn_rec("insufficient_data"),
    ]
    cost_records = [
        _cost_rec("improved", usd=1.0, tokens=200),
        _cost_rec("regressed"),
    ]
    summary = receipts.verified_saved_summary(relearn_records, cost_records)

    assert summary["improved_count"] == 2
    assert summary["regressed_count"] == 2          # one from each ledger
    assert summary["no_change_count"] == 1           # relearn-only concept
    assert summary["insufficient_data_count"] == 1
    # The realized total only reflects the IMPROVED fixes — regressed/no-change
    # never inflate the headline figure, they just aren't silently dropped
    # from the verified_count/regressed_count breakdown above.
    assert summary["verified_saved_usd"] == 1.0
    assert summary["verified_saved_tokens"] == 700


def test_summary_ignores_reverted_records():
    relearn_records = [_relearn_rec("improved", tokens=999, state="reverted")]
    cost_records = [_cost_rec("improved", usd=9.9, state="reverted")]
    summary = receipts.verified_saved_summary(relearn_records, cost_records)
    assert summary["verified_count"] == 0
    assert summary["verified_saved_usd"] == 0.0
    assert summary["verified_saved_tokens"] == 0


def test_summary_is_tagged_measured():
    summary = receipts.verified_saved_summary([_relearn_rec("improved", tokens=1)], [])
    assert summary["estimate_confidence"] == "measured"


# --- Estimated vs measured: the two never merge into one figure -------------

def test_estimated_rollup_and_measured_receipts_stay_structurally_separate():
    """Regression guard for the spec's hard rule (Component E's docstring /
    G1's spec text): an estimate and a measurement must never be summed into
    one number. We can't inspect the UI from here, so this asserts the two
    Python payloads themselves keep disjoint vocabularies — no shared numeric
    key that a future change could accidentally add together — and carry
    distinct confidence tags.
    """
    rollup = cost_proposals.estimated_recoverable_rollup(
        [{"signature": "cost:downsize", "analyzer": "downsize", "title": "t",
          "estimated_recoverable_usd": 3.0}],
    )
    summary = receipts.verified_saved_summary(
        [_relearn_rec("improved", tokens=500)],
        [_cost_rec("improved", usd=1.0, tokens=200)],
    )

    assert rollup["estimate_confidence"] == "estimated"
    assert summary["estimate_confidence"] == "measured"
    assert rollup["estimate_confidence"] != summary["estimate_confidence"]

    # No dollar-figure key overlaps between the two payloads — nothing here
    # could be naively `+`-ed together into one blended number.
    dollar_keys_rollup = {k for k in rollup if "usd" in k}
    dollar_keys_summary = {k for k in summary if "usd" in k}
    assert dollar_keys_rollup.isdisjoint(dollar_keys_summary)
