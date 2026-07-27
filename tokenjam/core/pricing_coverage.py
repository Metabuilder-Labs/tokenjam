"""How much of a cost figure was priced at the flat default rate.

`calculate_cost` falls back to `DEFAULT_INPUT_PER_MTOK` / `DEFAULT_OUTPUT_PER_MTOK`
when `get_rates` finds no row, and logs one warning per process. The resulting
dollar figure looks exactly like a real one on every surface, so a model missing
from the table is silently wrong rather than visibly unknown — measured once on a
real corpus, that silence hid errors of 5-30x across most of the models in it.

`CostEngine` already stamps each span's provenance onto `spans.pricing_source`
(see `pricing.classify_pricing_source`). This module reads that column back and
turns it into one sentence a cost view can print, so the two surfaces that show
it cannot disagree about what "unpriced" means.

Per Critical Rule 22 (`tokenjam/CLAUDE.md`) the note never renders a zero and
never restates the estimate as a quoted price — it names the models and says the
figure is a default-rate estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

#: The `pricing_source` value `classify_pricing_source` returns when nothing in
#: the table matched. Kept as a constant so a rename there fails here loudly
#: rather than silently reporting every window as fully priced.
DEFAULT_FALLBACK_SOURCE = "default_fallback"

#: Models named individually in the note before it summarises the rest. A note
#: that lists thirty model ids is not a note anyone reads.
_MAX_NAMED_MODELS = 3


@dataclass(frozen=True)
class PricingCoverage:
    """The default-rate share of a cost window.

    `measured` is False when the store could not be asked at all (no direct
    connection). That is deliberately distinct from "asked, and nothing was
    unpriced" — reporting an unmeasured window as clean is the failure this
    whole module exists to stop.
    """

    measured: bool
    unpriced_models: tuple[tuple[str, str, int], ...]
    unpriced_call_count: int
    unpriced_cost_usd: float


def summarize_pricing_coverage(
    conn,
    agent_id: str | None,
    since: datetime | None,
    until: datetime | None,
) -> PricingCoverage:
    """Which models in this window were priced at the default rate.

    Returns an unmeasured `PricingCoverage` when `conn` is None (the API-shim
    CLI path and any caller without a direct DuckDB handle), so a caller can
    tell "nothing unpriced" from "never looked".
    """
    if conn is None:
        return PricingCoverage(
            measured=False, unpriced_models=(), unpriced_call_count=0,
            unpriced_cost_usd=0.0,
        )

    clauses = ["pricing_source = $1"]
    params: list = [DEFAULT_FALLBACK_SOURCE]
    if agent_id:
        params.append(agent_id)
        clauses.append(f"agent_id = ${len(params)}")
    if since is not None:
        params.append(since)
        clauses.append(f"start_time >= ${len(params)}")
    if until is not None:
        params.append(until)
        clauses.append(f"start_time <= ${len(params)}")

    sql = (
        "SELECT provider, model, COUNT(*) AS calls, "
        "COALESCE(SUM(cost_usd), 0) AS cost "
        f"FROM spans WHERE {' AND '.join(clauses)} "
        "GROUP BY provider, model ORDER BY calls DESC"
    )
    rows = conn.execute(sql, params).fetchall()

    models = tuple(
        (str(r[0] or "unknown"), str(r[1] or "unknown"), int(r[2] or 0))
        for r in rows
    )
    return PricingCoverage(
        measured=True,
        unpriced_models=models,
        unpriced_call_count=sum(m[2] for m in models),
        unpriced_cost_usd=round(sum(float(r[3] or 0.0) for r in rows), 8),
    )


def coverage_note(coverage: PricingCoverage) -> str | None:
    """One sentence naming the unpriced models, or None when there is nothing
    to say (an unmeasured window, or a window with no default-rate spans)."""
    if not coverage.measured or not coverage.unpriced_models:
        return None

    named = coverage.unpriced_models[:_MAX_NAMED_MODELS]
    remainder = len(coverage.unpriced_models) - len(named)
    listed = ", ".join(f"{provider}/{model}" for provider, model, _ in named)
    if remainder > 0:
        listed += f" and {remainder} more"

    calls = coverage.unpriced_call_count
    return (
        f"{listed} {'is' if len(coverage.unpriced_models) == 1 else 'are'} not in "
        f"the pricing table: {calls:,} call{'' if calls == 1 else 's'} "
        f"here are estimated at tokenjam's default rate, not a rate published for "
        f"{'that model' if len(coverage.unpriced_models) == 1 else 'those models'}. "
        "Add one with a [pricing] override, or upgrade tokenjam; see `tj pricing list`."
    )
