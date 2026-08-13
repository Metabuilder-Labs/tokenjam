"""Cross-table guard: every DOWNGRADE_CANDIDATES entry must be priceable.

``model_downgrade.DOWNGRADE_CANDIDATES`` and ``pricing/models.toml`` are two
independently hand-maintained lists (see the DOWNGRADE_CANDIDATES comment in
``core/optimize/analyzers/model_downgrade.py``). Nothing forces them to agree:
if either side of a downgrade pair has no pricing row, `lookup_downgrade`
still returns the pair, but `build_agent_price_rows`
(`core/optimize/analyzers/downsize_agents.py`) silently drops the whole group
— no log, no counter, no disclosure field, the candidate simply never
produces a card. That drop is deliberate and correct (tokenjam refuses to
invent a rate — see `test_group_without_pricing_for_both_sides_is_dropped` in
`tests/unit/test_downsize_agent_pricing.py`); what this guards is the INPUT,
not that behaviour.

This is a two-lists-agreement invariant, not a symbol-reachability one — the
defect is a missing TABLE ROW, not a second call site — so it is registered
in `core/optimize/single_derivation.py::BESPOKE_SEAMS` rather than
`SEAMS`, alongside the module's other non-mechanizable pins.
"""
from __future__ import annotations

from tokenjam.core.optimize.analyzers.model_downgrade import DOWNGRADE_CANDIDATES
from tokenjam.core.optimize.span_pricing import rates_at
from tokenjam.utils.time_parse import utcnow


def _unpriced_entries() -> list[str]:
    """Every (provider, model) named by DOWNGRADE_CANDIDATES — as a key or as
    an alternative — that resolves to no rate right now.

    "Resolves to a price" is checked at the current instant (`utcnow()`),
    not a historical one: `DOWNGRADE_CANDIDATES` entries carry no date of
    their own (unlike a span), so there is no "when this ran" to price
    against. The question this guard asks is the same one a NEW pair
    added today has to answer before it can ever produce a card: is this
    swap priceable as of right now. A model priced only in the past (a
    withdrawn row) or only in the future is exactly the drift this guard
    exists to catch — both leave the pair silently unpriceable despite
    still appearing in the map.
    """
    now = utcnow()
    unpriced: list[str] = []
    for provider, mapping in DOWNGRADE_CANDIDATES.items():
        for model, alt_model in mapping.items():
            for name in (model, alt_model):
                if rates_at(provider, name, now) is None:
                    label = f"{provider}/{name}"
                    if label not in unpriced:
                        unpriced.append(label)
    return unpriced


def test_every_downgrade_candidate_is_currently_priceable() -> None:
    """A downgrade pair with either side unpriced is invisible to
    `build_agent_price_rows` with no disclosure — see the module docstring.

    Failing here names the exact (provider, model) missing a row in
    `pricing/models.toml`, so the fix is either adding that row or removing
    the pair from `DOWNGRADE_CANDIDATES` — never relaxing this assertion.
    """
    unpriced = _unpriced_entries()
    assert not unpriced, (
        "DOWNGRADE_CANDIDATES names model(s) with no current pricing row, so "
        "every pair referencing them is silently dropped (no card, no log) "
        "by build_agent_price_rows: " + ", ".join(sorted(unpriced))
        + ". Add a row to pricing/models.toml for each, or remove the pair "
        "from DOWNGRADE_CANDIDATES in "
        "core/optimize/analyzers/model_downgrade.py."
    )


def test_the_guard_has_a_corpus_to_guard() -> None:
    """A change that leaves DOWNGRADE_CANDIDATES empty would make the guard
    above vacuously green forever."""
    assert sum(len(mapping) for mapping in DOWNGRADE_CANDIDATES.values()) >= 5
