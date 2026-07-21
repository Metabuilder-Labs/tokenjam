"""Unit tests for the recoverable-total presentation contract in
`api/routes/cost.py` (A1, analyzer-audit #482).

`/cost/components` used to expose a flat `sum()` of every analyzer's
`estimated_recoverable_usd` with nothing signaling that the analyzers price
waste from overlapping angles over the same sessions. These tests pin the
fix: individual analyzer estimates are NEVER touched (the operator does not
want the headline magnitude deflated), but the response now says, in the
data itself, that the sum is a ceiling rather than an achievable total, and
carries a standalone "largest single line" figure that IS honest on its own.
"""
from __future__ import annotations

from dataclasses import dataclass

from tokenjam.api.routes.cost import _collect_recoverable, _recoverable_overlap_note


@dataclass
class _FakeFinding:
    estimated_recoverable_usd: float | None
    estimated_recoverable_tokens: int | None
    estimate_basis: str = ""
    caveat: str = ""


@dataclass
class _FakeReport:
    downgrade: object | None
    findings: dict


def test_collect_recoverable_sorted_biggest_first():
    """Order must be deterministic and USD-descending so 'largest opportunity
    + N more' can render directly off list order, with no client-side sort."""
    report = _FakeReport(
        downgrade=None,
        findings={
            "cache": _FakeFinding(estimated_recoverable_usd=5.0, estimated_recoverable_tokens=100),
            "trim": _FakeFinding(estimated_recoverable_usd=50.0, estimated_recoverable_tokens=200),
            "reuse": _FakeFinding(estimated_recoverable_usd=20.0, estimated_recoverable_tokens=50),
        },
    )
    out = _collect_recoverable(report)
    assert [r["analyzer"] for r in out] == ["trim", "reuse", "cache"]
    assert [r["estimated_recoverable_usd"] for r in out] == [50.0, 20.0, 5.0]


def test_collect_recoverable_ties_broken_by_tokens():
    report = _FakeReport(
        downgrade=None,
        findings={
            "cache": _FakeFinding(estimated_recoverable_usd=10.0, estimated_recoverable_tokens=100),
            "trim": _FakeFinding(estimated_recoverable_usd=10.0, estimated_recoverable_tokens=500),
        },
    )
    out = _collect_recoverable(report)
    assert [r["analyzer"] for r in out] == ["trim", "cache"]


def test_overlap_note_empty_for_zero_or_one_finding():
    # Zero findings: nothing to disclose, no false "these overlap" claim.
    assert _recoverable_overlap_note([]) == ""
    # A single finding cannot double-count anything by construction.
    assert _recoverable_overlap_note([{"estimated_recoverable_usd": 10.0}]) == ""


def test_overlap_note_present_for_two_or_more_findings():
    note = _recoverable_overlap_note([
        {"estimated_recoverable_usd": 10.0},
        {"estimated_recoverable_usd": 5.0},
    ])
    assert note != ""
    assert "2" in note  # names how many estimates it's disclaiming
    # House style: never claim the analyzers' figures were changed here.
    assert "reduce" not in note.lower()
    # No em dashes in user-facing copy (house rule).
    assert "—" not in note


def test_overlap_note_scales_the_count_with_the_list():
    note = _recoverable_overlap_note([
        {"estimated_recoverable_usd": 1.0},
        {"estimated_recoverable_usd": 1.0},
        {"estimated_recoverable_usd": 1.0},
    ])
    assert "3" in note
