from __future__ import annotations
import logging
from tj.core.models import NormalizedSpan
from tj.core.pricing import get_rates, ModelRates, DEFAULT_INPUT_PER_MTOK, DEFAULT_OUTPUT_PER_MTOK

logger = logging.getLogger(__name__)


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """
    Calculate USD cost for a single LLM call.

    Returns cost rounded to 8 decimal places.
    Falls back to default rates if the provider/model is not in the pricing table.
    Logs a warning on fallback so developers know to add the model.
    Zero tokens -> zero cost (no warning).
    """
    if input_tokens == 0 and output_tokens == 0:
        return 0.0

    rates = get_rates(provider, model)
    if rates is None:
        logger.warning(
            "No pricing data for %s/%s — using default rates. "
            "Add to pricing/models.toml to get accurate costs.",
            provider, model,
        )
        rates = ModelRates(
            input_per_mtok=DEFAULT_INPUT_PER_MTOK,
            output_per_mtok=DEFAULT_OUTPUT_PER_MTOK,
        )

    cost = (
        (input_tokens / 1_000_000) * rates.input_per_mtok
        + (output_tokens / 1_000_000) * rates.output_per_mtok
        + (cache_read_tokens / 1_000_000) * rates.cache_read_per_mtok
        + (cache_write_tokens / 1_000_000) * rates.cache_write_per_mtok
    )
    return round(cost, 8)


class CostEngine:
    """
    Post-ingest hook. Called by IngestPipeline after each span is written.
    Calculates cost and updates span.cost_usd + session.total_cost_usd in DB.
    """

    def __init__(self, db: object) -> None:
        self.db = db

    def process_span(self, span: NormalizedSpan) -> None:
        """
        If the span has token counts and a provider/model, calculate cost,
        update span.cost_usd in DB, update session.total_cost_usd in DB.
        No-op if tokens are missing or zero.
        """
        if not span.provider or not span.model:
            return
        input_tokens = span.input_tokens or 0
        output_tokens = span.output_tokens or 0
        if input_tokens == 0 and output_tokens == 0:
            return

        cache_read_tokens = span.cache_tokens or 0

        # Record whether the span was already pre-priced before we compute.
        # Pre-priced spans have their session cost handled by _build_or_update_session
        # in ingest.py; updating the session again here would double-count.
        was_pre_priced = span.cost_usd is not None

        cost = calculate_cost(
            provider=span.provider,
            model=span.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
        )

        span.cost_usd = cost

        # Update span cost in DB
        if hasattr(self.db, 'conn'):
            self.db.conn.execute(
                "UPDATE spans SET cost_usd = $1 WHERE span_id = $2",
                [cost, span.span_id],
            )

            # Only accumulate into session total when we computed the cost here.
            # Skip the session update for pre-priced spans to avoid double-counting.
            if span.session_id and not was_pre_priced:
                self.db.conn.execute(
                    "UPDATE sessions SET total_cost_usd = COALESCE(total_cost_usd, 0) + $1 "
                    "WHERE session_id = $2",
                    [cost, span.session_id],
                )
