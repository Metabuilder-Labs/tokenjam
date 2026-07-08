from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from tokenjam.core.models import NormalizedSpan
from tokenjam.core.pricing import get_rates, ModelRates, DEFAULT_INPUT_PER_MTOK, DEFAULT_OUTPUT_PER_MTOK

logger = logging.getLogger(__name__)

# Dedupe the "No pricing data" warning to one log line per (provider, model)
# pair per process. Backfilling a 247-session Claude Code project used to
# emit the same warning hundreds of times in a row (issue #98). Now it
# emits exactly once and stays out of the way.
_UNKNOWN_MODEL_WARNED: set[tuple[str, str]] = set()


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
    if (
        input_tokens == 0
        and output_tokens == 0
        and cache_read_tokens == 0
        and cache_write_tokens == 0
    ):
        return 0.0

    rates = get_rates(provider, model)
    if rates is None:
        # Warn once per (provider, model) per process — see _UNKNOWN_MODEL_WARNED.
        key = (provider, model)
        if key not in _UNKNOWN_MODEL_WARNED:
            _UNKNOWN_MODEL_WARNED.add(key)
            logger.warning(
                "No pricing data for %s/%s — using default rates (cost figures "
                "may be inaccurate). Upgrade tokenjam for current pricing, or "
                "add an override to ~/.config/tj/pricing.toml — see `tj pricing list`.",
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
        cache_read_tokens = span.cache_tokens or 0
        cache_write_tokens = span.cache_write_tokens or 0
        if (
            input_tokens == 0
            and output_tokens == 0
            and cache_read_tokens == 0
            and cache_write_tokens == 0
        ):
            return

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
            cache_write_tokens=cache_write_tokens,
        )

        span.cost_usd = cost

        # Persist through the StorageBackend protocol (issue #309 — this used to
        # reach into self.db.conn directly). Backends that can't persist (e.g.
        # the read-only API backend) simply don't expose these methods, mirroring
        # the previous hasattr(self.db, 'conn') guard.
        update = getattr(self.db, "update_span_cost", None)
        if update is None:
            return
        update(span.span_id, cost)

        # Only accumulate into the session total when we computed the cost here.
        # Skip the session update for pre-priced spans to avoid double-counting
        # (their session cost is handled by ingest's _build_or_update_session).
        if span.session_id and not was_pre_priced:
            self.db.increment_session_cost(span.session_id, cost)


# ---------------------------------------------------------------------------
# Period comparison (tj cost --compare / tj optimize --compare)
# ---------------------------------------------------------------------------

@dataclass
class WindowTotals:
    """Aggregate spend + tokens for a single time window."""
    since:          datetime
    until:          datetime
    sessions:       int  = 0
    input_tokens:   int  = 0
    output_tokens:  int  = 0
    cache_tokens:   int  = 0
    total_cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_tokens


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
) -> WindowTotals:
    """Aggregate sessions/tokens/cost across the spans table for a window.

    Reads through the StorageBackend protocol (`get_window_cost_totals`) rather
    than touching `db.conn` directly (issue #309).
    """
    sessions, in_tok, out_tok, cache_tok, cost = db.get_window_cost_totals(
        since, until, agent_id,
    )
    return WindowTotals(
        since=since, until=until,
        sessions=sessions,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_tokens=cache_tok,
        total_cost_usd=cost,
    )


def compute_cost_diff(
    db,
    current_since: datetime,
    current_until: datetime,
    compare: str,
    agent_id: str | None = None,
    top_n: int = 5,
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

    current = compute_window_totals(db, current_since, current_until, agent_id)
    previous = compute_window_totals(db, prev_since, prev_until, agent_id)

    return CostDiff(
        current=current,
        previous=previous,
        by_agent=db.get_cost_delta_by_group(
            "agent_id", current_since, current_until, prev_since, prev_until, top_n,
        ),
        by_model=db.get_cost_delta_by_group(
            "model", current_since, current_until, prev_since, prev_until, top_n,
        ),
    )
