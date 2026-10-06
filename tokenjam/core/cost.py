from __future__ import annotations
import contextvars
import logging
import re
from contextlib import contextmanager
from rich.markup import escape
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from tokenjam.core.models import NormalizedSpan
from tokenjam.core.pricing import (
    DEFAULT_CACHE_READ_PER_MTOK,
    DEFAULT_CACHE_WRITE_PER_MTOK,
    DEFAULT_INPUT_PER_MTOK,
    DEFAULT_OUTPUT_PER_MTOK,
    STANDARD_VARIANT,
    ModelRates,
    classify_pricing_source,
    get_rates,
)

logger = logging.getLogger(__name__)

# Dedupe the "No pricing data" warning to one log line per (provider, model)
# pair per process. Backfilling a 247-session Claude Code project used to
# emit the same warning hundreds of times in a row (issue #98). Now it
# emits exactly once and stays out of the way.
_UNKNOWN_MODEL_WARNED: set[tuple[str, str]] = set()

# When set, unknown-model pricing warnings are collected here instead of being
# logged immediately. CLI surfaces that ingest inline before rendering (e.g.
# `tj quickstart`, `tj backfill`) enter this mode so a pricing note lands after
# the report body rather than ahead of it (issue #585).
_DEFERRED_PRICING_WARNINGS: contextvars.ContextVar[list[str] | None] = (
    contextvars.ContextVar("_deferred_pricing_warnings", default=None)
)


def _unknown_model_warning_message(provider: str, model: str) -> str:
    return (
        "No pricing data for %s/%s — using default rates, including "
        "guessed cache rates (cost figures may be inaccurate, "
        "especially for cache-heavy traffic). Upgrade tokenjam for "
        "current pricing, or add an override to "
        "~/.config/tj/pricing.toml — see `tj pricing list`."
        % (provider, model)
    )


@contextmanager
def defer_pricing_warnings():
    """Collect unknown-model pricing warnings until the caller renders them."""
    messages: list[str] = []
    token = _DEFERRED_PRICING_WARNINGS.set(messages)
    try:
        yield messages
    finally:
        # Fail-safe: any warning collected but not rendered (early return,
        # exception, Ctrl-C) must still reach the user. The dedup set is
        # marked at collection time, so a swallowed warning is gone forever.
        unconsumed = list(messages)
        _DEFERRED_PRICING_WARNINGS.reset(token)
        for message in unconsumed:
            logger.warning("%s", message)


def drain_deferred_pricing_warnings() -> list[str]:
    """Return and clear any pricing warnings collected under defer mode."""
    bucket = _DEFERRED_PRICING_WARNINGS.get()
    if not bucket:
        return []
    messages = list(bucket)
    bucket.clear()
    return messages


def print_deferred_pricing_warnings(*, console, messages: list[str] | None = None) -> None:
    """Render deferred unknown-model pricing warnings, if any."""
    if messages is not None:
        pending = list(messages)
        messages.clear()
    else:
        pending = drain_deferred_pricing_warnings()
    for message in pending:
        console.print(f"[warn]{escape(message)}[/warn]")


def billing_rates(
    provider: str,
    model: str,
    *,
    at: datetime | None = None,
    variant: str = STANDARD_VARIANT,
) -> ModelRates:
    """The rates this call is actually BILLED at — never ``None``.

    The stored-cost convention, in one place. :func:`calculate_cost` prices
    every span through this, so ``spans.cost_usd`` is by definition
    ``tokens x these rates``; anything that has to reconcile against that
    stored figure (``/cost/components`` splitting the same window into its four
    token components) must resolve its rates here too, or it is pricing a
    different quantity and calling it the same total.

    That is not a hypothetical. ``_component_costs`` used to call
    ``span_pricing.rates_at`` and ``continue`` past a ``None``, so every model
    absent from the pricing table contributed real fallback dollars to
    ``/cost``'s total and exactly ``$0`` to ``/cost/components``' — two
    endpoints publishing the same window under two conventions.

    The fallback is deliberately NOT the analyzers' convention. ``span_pricing``
    returns ``None`` for an unpriced model because an analyzer that cannot price
    a figure must not claim one. A MEASURED spend total has the opposite duty:
    the traffic happened and cost real money, so estimating it at the default
    rate and disclosing that through ``pricing_coverage`` beats reporting zero.
    """
    rates = get_rates(provider, model, at=at, variant=variant)
    if rates is not None:
        return rates
    # Warn once per (provider, model) per process — see _UNKNOWN_MODEL_WARNED.
    key = (provider, model)
    if key not in _UNKNOWN_MODEL_WARNED:
        _UNKNOWN_MODEL_WARNED.add(key)
        message = _unknown_model_warning_message(provider, model)
        bucket = _DEFERRED_PRICING_WARNINGS.get()
        if bucket is not None:
            bucket.append(message)
        else:
            logger.warning("%s", message)
    # cache_read/write_per_mtok are non-zero guesses, not 0.0: a model with
    # NO pricing entry that is mostly cache traffic would otherwise have
    # ~all of its real cost priced at zero, silently, rather than merely
    # estimated (see DEFAULT_CACHE_READ_PER_MTOK). specified=False marks
    # both as guesses, not a quoted rate for this model.
    return ModelRates(
        input_per_mtok=DEFAULT_INPUT_PER_MTOK,
        output_per_mtok=DEFAULT_OUTPUT_PER_MTOK,
        cache_read_per_mtok=DEFAULT_CACHE_READ_PER_MTOK,
        cache_write_per_mtok=DEFAULT_CACHE_WRITE_PER_MTOK,
        cache_read_specified=False,
        cache_write_specified=False,
    )


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    *,
    at: datetime | None = None,
    variant: str = STANDARD_VARIANT,
) -> float:
    """
    Calculate USD cost for a single LLM call.

    `at` is when the call happened: pricing has a time axis, so a call made
    before a rate change prices at the rate that actually billed it (defaults to
    now, which is what a live call wants). `variant` is how it was bought —
    `fast` mode and the Batch API bill the same model id at different rates.

    **That `at=None` default is for LIVE callers only.** Ingest is the reference
    use: `CostEngine.process_span` below passes `at=span.start_time` and
    `variant=rate_variant_for_span(span)`, so a cost computed once at write time
    never moves under a later rate change. Anything reading BACKWARDS — every
    optimize analyzer — must pass the instant explicitly, and goes through
    :mod:`tokenjam.core.optimize.span_pricing`, which makes it a required
    argument and states the aggregation convention (price each span at its own
    timestamp, then sum; never price an aggregate at a single date). Do not add
    a new backward-looking caller here that relies on the default.

    Returns cost rounded to 8 decimal places.
    Falls back to default rates if the provider/model is not in the pricing table.
    Logs a warning on fallback so developers know to add the model.
    Zero tokens -> zero cost (no warning).
    """
    if (
        input_tokens == 0
        and output_tokens == 0
        and cache_read_tokens == 0
        and cache_write_tokens == 0
    ):
        return 0.0

    rates = billing_rates(provider, model, at=at, variant=variant)

    cost = (
        (input_tokens / 1_000_000) * rates.input_per_mtok
        + (output_tokens / 1_000_000) * rates.output_per_mtok
        + (cache_read_tokens / 1_000_000) * rates.cache_read_per_mtok
        + (cache_write_tokens / 1_000_000) * rates.cache_write_per_mtok
    )
    return round(cost, 8)


#: Request-param values that name a non-standard way of buying the same model.
#: `speed` is Anthropic's fast mode (`speed="fast"`); anything else observed
#: there is passed through as a variant name so a future variant prices itself
#: as soon as the table carries a row for it, with `get_rates` warning once when
#: it doesn't.
_SPEED_PARAM = "speed"


def rate_variant_for_span(span: NormalizedSpan) -> str:
    """Which pricing variant a span was billed under (`standard` when unknown).

    Read from the captured request params rather than inferred: `speed="fast"`
    is a request parameter the caller sent, so it is the only honest evidence
    that the premium rate applied. A span whose capture never saw it prices at
    the standard rate — understating a fast call, which is the safe direction
    for a figure checked against a bill, and `get_rates` says so in a warning.
    """
    params = span.request_params or {}
    speed = params.get(_SPEED_PARAM)
    if isinstance(speed, str) and speed and speed.lower() != STANDARD_VARIANT:
        return speed.lower()
    return STANDARD_VARIANT


#: :func:`rate_variant_for_span`, re-expressed over the STORED ``request_params``
#: JSON column, for an aggregate query that has to price the way ingest priced.
#:
#: Kept here, beside the function it mirrors, rather than in the one route that
#: needs it: a second spelling of "which variant billed this" living in an API
#: module is exactly how the component split ended up pricing every span at the
#: standard rate while ``spans.cost_usd`` carried the fast one. Group by this
#: alongside (provider, model) and the UTC day, and a rolled-up query prices at
#: the variant that actually billed each bucket.
#:
#: A NULL ``request_params`` (capture off, the common case) coalesces to
#: ``standard``, and so does a literal ``"standard"`` — the same two cases the
#: function folds together. ``lower()`` matches its normalisation.
SPAN_VARIANT_SQL = (
    "lower(COALESCE(NULLIF("
    f"json_extract_string(request_params, '$.{_SPEED_PARAM}'), ''), "
    f"'{STANDARD_VARIANT}'))"
)


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
        cache_read_tokens = span.cache_tokens or 0
        cache_write_tokens = span.cache_write_tokens or 0
        if (
            input_tokens == 0
            and output_tokens == 0
            and cache_read_tokens == 0
            and cache_write_tokens == 0
        ):
            return

        # Whatever cost the span arrived carrying has ALREADY been added to the
        # session total by `_build_or_update_session` in ingest.py (it does
        # `existing.total_cost_usd += span.cost_usd` for any non-None value, and
        # seeds a new session's total from it). An unpriced span contributed
        # nothing, so its prior contribution is 0.
        #
        # This used to be a boolean skip: pre-priced spans got no session update
        # at all, on the reasoning that ingest had already handled them. But we
        # then OVERWRITE `spans.cost_usd` with tj's own figure below, so the
        # session kept the upstream number while the span row carried ours.
        # Incrementing by the DELTA keeps the live session total aligned with
        # the post-hook span cost under every combination (unpriced, pre-priced,
        # or re-priced) with no special case and no extra read. Repair and late
        # attribution paths recompute from canonical logical observations.
        prior_cost = span.cost_usd or 0.0

        cost = calculate_cost(
            provider=span.provider,
            model=span.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            # Price the call at the rate in effect WHEN IT HAPPENED, and at the
            # rate for how it was bought. Cost is computed once at ingest and
            # stored, so a later rate change never moves this figure.
            at=span.start_time,
            variant=rate_variant_for_span(span),
        )

        span.cost_usd = cost
        # Provenance: HOW that cost resolved (real rate vs. a guessed
        # fallback), stamped on the span so it survives past ingest — see
        # classify_pricing_source and the `pricing_source` column (spans
        # migration adding it). Without this a fallback-priced span and a
        # correctly-priced one are indistinguishable once only the dollar
        # figure remains, which is exactly how the unpriced-cache-at-zero bug
        # went unnoticed.
        span.pricing_source = classify_pricing_source(span.provider, span.model)

        # Persist through the StorageBackend protocol (issue #309 — this used to
        # reach into self.db.conn directly). Backends that can't persist (e.g.
        # the read-only API backend) simply don't expose these methods, mirroring
        # the previous hasattr(self.db, 'conn') guard.
        update = getattr(self.db, "update_span_cost", None)
        if update is None:
            return
        update(span.span_id, cost, span.pricing_source)

        # Move the session total by exactly what this span's stored cost moved.
        # A zero delta (we agreed with the incoming figure, or the span is being
        # reprocessed with the same rates) writes nothing. Repair and late
        # attribution paths use canonical logical observations instead of a raw
        # span sum.
        delta = cost - prior_cost
        if span.session_id and delta:
            self.db.increment_session_cost(span.session_id, delta)


# ---------------------------------------------------------------------------
# Period comparison (tj cost --compare / tj optimize --compare)
# ---------------------------------------------------------------------------

@dataclass
class WindowTotals:
    """Aggregate spend + tokens for a single time window."""
    since:              datetime
    until:              datetime
    sessions:           int  = 0
    input_tokens:       int  = 0
    output_tokens:      int  = 0
    cache_tokens:       int  = 0
    cache_write_tokens: int  = 0
    total_cost_usd:     float = 0.0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens + self.output_tokens
            + self.cache_tokens + self.cache_write_tokens
        )

@dataclass
class CostDiff:
    """Diff between two equal-or-arbitrary-length windows of cost data."""
    current:   WindowTotals
    previous:  WindowTotals
    # Top contributors that shifted (positive delta = increased spend).
    by_agent:  list[dict] = field(default_factory=list)
    by_model:  list[dict] = field(default_factory=list)

    @property
    def cost_delta_usd(self) -> float:
        return self.current.total_cost_usd - self.previous.total_cost_usd

    @property
    def cost_delta_pct(self) -> float | None:
        if self.previous.total_cost_usd <= 0:
            return None
        return (self.cost_delta_usd / self.previous.total_cost_usd) * 100.0

    @property
    def tokens_delta(self) -> int:
        return self.current.total_tokens - self.previous.total_tokens

    @property
    def tokens_delta_pct(self) -> float | None:
        if self.previous.total_tokens <= 0:
            return None
        return (self.tokens_delta / self.previous.total_tokens) * 100.0


# Recognised --compare keywords. Each maps to a window-resolution rule.
_COMPARE_KEYWORDS = {"previous", "last-week", "last-month", "last-7d", "last-30d"}


def parse_compare_window(
    compare: str,
    current_since: datetime,
    current_until: datetime,
) -> tuple[datetime, datetime]:
    """
    Resolve the --compare value to an absolute (since, until) tuple.

    Keywords (`previous`, `last-week`, etc.) resolve to the equal-length
    window immediately preceding the current window. Explicit date ranges
    (`2026-04-01:2026-04-30`) are used verbatim — they don't have to match
    the current window's length.

    Examples (current = 2026-05-08 → 2026-05-15, length 7d):
      previous        → 2026-05-01 → 2026-05-08
      last-week       → 2026-05-01 → 2026-05-08 (same as previous)
      last-7d         → 2026-05-01 → 2026-05-08
      last-month      → 2026-04-15 → 2026-05-08 (30d before until)
      2026-04-01:2026-04-30 → that exact range
    """
    compare = compare.strip()

    # Explicit date range "YYYY-MM-DD:YYYY-MM-DD"
    m = re.fullmatch(
        r"(\d{4}-\d{2}-\d{2}):(\d{4}-\d{2}-\d{2})", compare
    )
    if m:
        start = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(m.group(2)).replace(tzinfo=timezone.utc)
        if end <= start:
            raise ValueError("Compare range end must be after start.")
        return start, end

    if compare not in _COMPARE_KEYWORDS:
        raise ValueError(
            f"Unknown --compare value '{compare}'. Use one of "
            f"{sorted(_COMPARE_KEYWORDS)} or 'YYYY-MM-DD:YYYY-MM-DD'."
        )

    if compare == "last-month":
        # 30 days immediately before the current until (not the current since).
        # This is independent of the current window length so monthly trends
        # stay readable when the user runs `tj cost --since 7d --compare last-month`.
        prev_until = current_since
        prev_since = current_until - timedelta(days=30) - (current_until - current_since)
        return prev_since, prev_until

    # All other keywords: equal-length window immediately before `since`.
    length = current_until - current_since
    prev_until = current_since
    prev_since = current_since - length
    return prev_since, prev_until


def override_since_for_compare(
    compare: str, default_since: datetime, current_until: datetime,
) -> datetime:
    """
    Resolve `--compare` keywords that imply a *specific* current-window
    length (`last-7d`, `last-30d`, `last-week`) to a `since` datetime that
    makes the comparison symmetric.

    Without this, `tj optimize --compare last-7d` would render a 30d-vs-30d
    comparison (because `--since` defaults to 30d) while
    `tj cost --compare last-7d` would render a 7d-vs-7d comparison (because
    `--since` defaults to 7d) — the same flag producing different shapes
    across commands (#71 finding 5). Forcing `last-Nd` to N days everywhere
    gives the user the comparison they asked for.

    Returns `default_since` unchanged for keywords without an implied window
    length (`previous`, `last-month`) or explicit date ranges.
    """
    c = compare.strip().lower()
    if c == "last-7d" or c == "last-week":
        return current_until - timedelta(days=7)
    if c == "last-30d":
        return current_until - timedelta(days=30)
    return default_since


def compute_window_totals(
    db, since: datetime, until: datetime, agent_id: str | None = None,
    persona: str | None = None,
) -> WindowTotals:
    """Aggregate sessions/tokens/cost across the spans table for a window.

    Reads through the StorageBackend protocol (`get_window_cost_totals`) rather
    than touching `db.conn` directly (issue #309).
    """
    sessions, in_tok, out_tok, cache_tok, cache_write_tok, cost = db.get_window_cost_totals(
        since, until, agent_id, persona,
    )
    return WindowTotals(
        since=since, until=until,
        sessions=sessions,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_tokens=cache_tok,
        cache_write_tokens=cache_write_tok,
        total_cost_usd=cost,
    )


def compute_cost_diff(
    db,
    current_since: datetime,
    current_until: datetime,
    compare: str,
    agent_id: str | None = None,
    top_n: int = 5,
    persona: str | None = None,
) -> CostDiff:
    """
    Build a CostDiff between the current window and the resolved compare window.

    Reports per-agent and per-model cost deltas (top N each) so the renderer
    can surface which agents/models drove the change.

    Reads through the StorageBackend protocol (issue #309) rather than reaching
    into `db.conn`.
    """
    prev_since, prev_until = parse_compare_window(
        compare, current_since, current_until,
    )

    # ONE persona across all four reads. The two window totals and the two
    # delta breakdowns are rendered as one comparison, so a scope applied to
    # some of them would put different populations on either side of a delta.
    current = compute_window_totals(db, current_since, current_until, agent_id, persona)
    previous = compute_window_totals(db, prev_since, prev_until, agent_id, persona)

    return CostDiff(
        current=current,
        previous=previous,
        by_agent=db.get_cost_delta_by_group(
            "agent_id", current_since, current_until, prev_since, prev_until, top_n,
            persona,
        ),
        by_model=db.get_cost_delta_by_group(
            "model", current_since, current_until, prev_since, prev_until, top_n,
            persona,
        ),
    )
