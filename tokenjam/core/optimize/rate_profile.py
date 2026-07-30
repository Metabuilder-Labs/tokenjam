"""Blended input rate + cache-read ratio, measured off observed spans.

Two analyzers need to price the SAME shape of thing: a block of tokens that is
sent once at the input rate and then re-read on later calls at the cache-read
rate. ``relearn`` prices a failure's re-read tail that way; ``summarize``
prices an always-on prompt file's per-session re-reads that way. Both need the
two rates blended over whichever models the user actually ran, and neither may
invent one when the data cannot supply it.

Deliberately NOT derived from observed ``cost_usd``. An all-in blended $/token
cannot tell an input token from a cache read, and the whole point here is to
price those two classes differently. The rates come from
``pricing/models.toml`` via :func:`tokenjam.core.pricing.get_rates`, weighted
by the token volume each model actually carried.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tokenjam.core.optimize.span_pricing import SPAN_UTC_DAY_SQL


@dataclass(frozen=True)
class RateProfile:
    """The two rates a re-read block is billed at, blended over observed models.

    ``input_rate_per_token`` prices the first send; ``cache_read_ratio``
    (cache-read rate / input rate — exactly 0.100 for every Anthropic model in
    ``pricing/models.toml``) prices every re-read of it. ``basis`` names the
    models the blend came from so the derivation is never a black box.
    """

    input_rate_per_token: float
    cache_read_ratio: float
    basis: str

    def cost_of(self, tokens: float, rereads: int) -> float:
        """What ``tokens`` cost when sent once and re-read ``rereads`` times."""
        return (
            tokens
            * self.input_rate_per_token
            * (1.0 + self.cache_read_ratio * max(rereads, 0))
        )


def blended_rate_profile(
    conn: Any,
    *,
    session_ids: set[str] | None = None,
    since: Any = None,
    until: Any = None,
    agent_id: str | None = None,
) -> RateProfile | None:
    """Blend input + cache-read rates over the spans the filters select.

    Pass ``session_ids`` to scope to an explicit set of sessions, or
    ``since``/``until`` (plus an optional ``agent_id``) to scope to a window.
    Returns ``None`` — never a default rate — when there is no connection, no
    matching spans, or no model with pricing data; the caller then reports a
    token figure with no dollars rather than a number borrowed from a model
    the user never ran (CLAUDE.md anti-pattern #22).
    """
    if conn is None:
        return None
    clauses = ["model IS NOT NULL"]
    params: list[Any] = []
    if session_ids is not None:
        if not session_ids:
            return None
        ids = sorted(session_ids)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(ids)))
        clauses.append(f"session_id IN ({placeholders})")
        params.extend(ids)
    if since is not None:
        clauses.append(f"start_time >= ${len(params) + 1}")
        params.append(since)
    if until is not None:
        clauses.append(f"start_time < ${len(params) + 1}")
        params.append(until)
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)

    try:
        rows = conn.execute(
            "SELECT provider, model, "
            "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(cache_tokens), 0), "
            "MIN(start_time) "
            "FROM spans WHERE " + " AND ".join(clauses)
            + f" GROUP BY provider, model, {SPAN_UTC_DAY_SQL}",
            params,
        ).fetchall()
    except Exception:
        return None

    from tokenjam.core.optimize.span_pricing import rates_at

    # Grouped by (provider, model, UTC day) rather than (provider, model): each
    # bucket then prices at the rate that actually billed it, so a window
    # straddling a rate change blends the real rates instead of repricing all
    # of it at today's. A UTC day never straddles a boundary — every
    # `valid_from` in the pricing table is a date. See
    # `tokenjam.core.optimize.span_pricing` for the convention.
    #
    # Each rate is weighted by ITS OWN matching token class, not a shared
    # combined total -- input tokens for the input rate, cache-read tokens for
    # the cache-read ratio. Weighting either by output/cache-write volume (a
    # token class the priced rate has nothing to do with) biases the blend
    # whenever a corpus's model mix has different output/cache-write
    # proportions than input/cache-read proportions.
    weighted_input = 0.0
    input_tokens_total = 0
    weighted_cache_read = 0.0
    cache_read_tokens_total = 0
    models: list[str] = []
    for provider, model, input_tokens, cache_read_tokens, day_start in rows:
        input_tokens = int(input_tokens or 0)
        cache_read_tokens = int(cache_read_tokens or 0)
        if input_tokens <= 0 and cache_read_tokens <= 0:
            continue
        rates = rates_at(str(provider or "unknown"), str(model), day_start)
        if rates is None or rates.input_per_mtok <= 0:
            continue
        if input_tokens > 0:
            weighted_input += rates.input_per_mtok * input_tokens
            input_tokens_total += input_tokens
        if cache_read_tokens > 0:
            weighted_cache_read += rates.cache_read_per_mtok * cache_read_tokens
            cache_read_tokens_total += cache_read_tokens
        models.append(f"{provider}/{model}")
    if input_tokens_total <= 0:
        return None
    input_per_mtok = weighted_input / input_tokens_total
    if input_per_mtok <= 0:
        return None
    # No cache reads observed at all: no cache-read volume to weight the ratio
    # by, so fall back to the same models' rates weighted by INPUT volume
    # instead -- still derived from pricing/models.toml for the models
    # actually observed, never an invented number.
    if cache_read_tokens_total > 0:
        cache_read_per_mtok = weighted_cache_read / cache_read_tokens_total
    else:
        weighted_cache_read_by_input = 0.0
        for provider, model, input_tokens, _cache_read_tokens, day_start in rows:
            input_tokens = int(input_tokens or 0)
            if input_tokens <= 0:
                continue
            rates = rates_at(str(provider or "unknown"), str(model), day_start)
            if rates is None or rates.input_per_mtok <= 0:
                continue
            weighted_cache_read_by_input += rates.cache_read_per_mtok * input_tokens
        cache_read_per_mtok = weighted_cache_read_by_input / input_tokens_total
    cache_read_ratio = min(cache_read_per_mtok / input_per_mtok, 1.0)
    return RateProfile(
        input_rate_per_token=input_per_mtok / 1_000_000,
        cache_read_ratio=cache_read_ratio,
        basis=(
            f"${input_per_mtok:.2f}/MTok input, re-reads at "
            f"{cache_read_ratio:.3f}x that, blended over the models actually "
            f"observed: {', '.join(sorted(set(models)))}"
        ),
    )
