from __future__ import annotations
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from tokenjam.core.models import NormalizedSpan
from tokenjam.core.pricing import get_rates, ModelRates, DEFAULT_INPUT_PER_MTOK, DEFAULT_OUTPUT_PER_MTOK

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
    conn, since: datetime, until: datetime, agent_id: str | None = None,
) -> WindowTotals:
    """Aggregate sessions/tokens/cost across the spans table for a window."""
    clauses = ["start_time >= $1", "start_time < $2"]
    params: list = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    row = conn.execute(
        f"SELECT COUNT(DISTINCT session_id) AS sessions, "
        f"COALESCE(SUM(input_tokens), 0)   AS in_tok, "
        f"COALESCE(SUM(output_tokens), 0)  AS out_tok, "
        f"COALESCE(SUM(cache_tokens), 0)   AS cache_tok, "
        f"COALESCE(SUM(cost_usd), 0.0)     AS cost "
        f"FROM spans WHERE {where}",
        params,
    ).fetchone()
    return WindowTotals(
        since=since, until=until,
        sessions=int(row[0] or 0),
        input_tokens=int(row[1] or 0),
        output_tokens=int(row[2] or 0),
        cache_tokens=int(row[3] or 0),
        total_cost_usd=float(row[4] or 0.0),
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
    """
    conn = getattr(db, "conn", None)
    if conn is None:
        raise RuntimeError("compare requires a direct DuckDB connection")

    prev_since, prev_until = parse_compare_window(
        compare, current_since, current_until,
    )

    current = compute_window_totals(conn, current_since, current_until, agent_id)
    previous = compute_window_totals(conn, prev_since, prev_until, agent_id)

    # Per-agent and per-model cost deltas. Joins are window-scoped because we
    # only care about agents/models that appear in either window.
    def _grouped_delta(group_col: str) -> list[dict]:
        sql = f"""
            SELECT {group_col} AS grp,
                   COALESCE(SUM(CASE WHEN start_time >= $1 AND start_time < $2
                                     THEN cost_usd ELSE 0 END), 0.0) AS cur_cost,
                   COALESCE(SUM(CASE WHEN start_time >= $3 AND start_time < $4
                                     THEN cost_usd ELSE 0 END), 0.0) AS prev_cost
            FROM spans
            WHERE (start_time >= $3 AND start_time < $2)
              AND {group_col} IS NOT NULL
            GROUP BY {group_col}
            HAVING ABS(cur_cost - prev_cost) > 0.0001
            ORDER BY ABS(cur_cost - prev_cost) DESC
            LIMIT $5
        """
        rows = conn.execute(
            sql, [current_since, current_until, prev_since, prev_until, top_n],
        ).fetchall()
        return [
            {"group": r[0], "current_cost": float(r[1]), "previous_cost": float(r[2]),
             "delta": float(r[1]) - float(r[2])}
            for r in rows
        ]

    return CostDiff(
        current=current,
        previous=previous,
        by_agent=_grouped_delta("agent_id"),
        by_model=_grouped_delta("model"),
    )
