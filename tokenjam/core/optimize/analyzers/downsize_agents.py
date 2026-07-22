"""Per-agent price arithmetic for the downsize card.

The downsize analyzer flags sessions whose *shape* matches a class of work a
cheaper same-family model is worth reviewing for. That finding is a window-wide
aggregate; a user reading the card still has to answer "which of my agents, and
what does the swap actually cost me?".

This module answers that from the same candidate sessions, per agent:

  * every token type the agent was billed for (input, output, cache read AND
    cache write) priced at its CURRENT model's rates, and the identical token
    mix repriced at the PROPOSED model's rates, both through
    ``core.pricing``/``core.cost`` at runtime (rates are never hardcoded here);
  * the difference over the analyzed window and a 30-day projection of it;
  * an informational thinking-token share of output, when the runtime reports
    one. It is a number on the card and nothing else: no effort proposal is
    derived from it.

The price delta is measured arithmetic GIVEN the switch. Whether the cheaper
model produces an equivalent answer is not measured anywhere in this file, and
the card says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tokenjam.core.cost import calculate_cost
from tokenjam.core.pricing import get_rates

#: Runtimes report a thinking/reasoning token count under these span-attribute
#: keys (Codex writes ``reasoning_token_count``). Anthropic bills thinking as
#: output without breaking it out per call, so most rows carry no share at all
#: and the card must omit the number rather than infer one.
THINKING_ATTRIBUTE_KEYS = ("reasoning_token_count", "thinking_tokens")

#: Days in the projection basis. The window figure is what was measured; the
#: projection is an "up to" restatement of it at the same daily rate.
PROJECTION_DAYS = 30.0

#: How the per-agent numbers were constructed. Rendered verbatim as the card's
#: construction footnote so no channel can print the dollar delta bare.
AGENT_PRICE_BASIS = (
    "Each agent's own input, output, cache read and cache write tokens over the "
    "candidate sessions, priced at its current model's published rates and "
    "repriced at the proposed model's rates. The difference is arithmetic on "
    "measured tokens, given the switch; it is not a claim that the cheaper "
    "model answers as well."
)


@dataclass
class AgentPriceRow:
    """One agent's exact price arithmetic for the proposed model swap."""
    agent_id:           str
    provider:           str
    model:              str
    alt_model:          str
    sessions:           int
    input_tokens:       int
    output_tokens:      int
    cache_tokens:       int
    cache_write_tokens: int
    current_cost_usd:     float
    alternative_cost_usd: float
    delta_usd:            float
    projected_30d_delta_usd: float
    window_days:          float
    # Informational only (no proposal is derived from it). ``None`` when the
    # runtime does not report a thinking/reasoning token count.
    thinking_tokens:          int | None   = None
    thinking_share_of_output: float | None = None
    estimate_basis: str = AGENT_PRICE_BASIS

    @property
    def total_tokens(self) -> int:
        """Every token type the agent was billed for. All four types, always:
        omitting a cache type here silently understates the swap."""
        return (
            self.input_tokens + self.output_tokens
            + self.cache_tokens + self.cache_write_tokens
        )


@dataclass
class _Acc:
    """Mutable accumulator for one (agent, provider, model, alt) group."""
    sessions: set[str] = field(default_factory=set)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    cache_write_tokens: int = 0


def price_tokens(
    provider: str,
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_tokens: int,
    cache_write_tokens: int,
) -> float | None:
    """Cost of an exact token mix at ``model``'s rates, or ``None`` when the
    model has no pricing data (we refuse to invent a rate).

    Prices all four billed token types. A cheaper model is still charged for
    cache reads and cache writes, so dropping either type from one side of the
    comparison would manufacture a saving that does not exist.
    """
    if get_rates(provider, model) is None:
        return None
    return calculate_cost(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def thinking_share(thinking_tokens: int | None, output_tokens: int) -> float | None:
    """Thinking tokens as a share of output tokens, or ``None`` when unreported.

    Informational: thinking tokens bill as output, so this says how much of the
    output spend was internal reasoning. No recommendation follows from it.
    """
    if thinking_tokens is None or output_tokens <= 0:
        return None
    return round(thinking_tokens / output_tokens, 4)


def thinking_tokens_by_session(
    conn: Any, since: datetime, until: datetime, agent_id: str | None,
) -> dict[str, int]:
    """Reported thinking/reasoning tokens per session, from span attributes.

    Empty when no runtime in the window reports them, which is the common case:
    the rows then carry ``thinking_tokens=None`` and the card omits the number
    instead of printing a zero that reads like a measurement.
    """
    clauses = ["start_time >= $1", "start_time < $2", "session_id IS NOT NULL"]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    coalesce = " + ".join(
        f"COALESCE(TRY_CAST(json_extract_string(attributes, '$.{key}') AS BIGINT), 0)"
        for key in THINKING_ATTRIBUTE_KEYS
    )
    try:
        rows = conn.execute(
            f"SELECT session_id, COALESCE(SUM({coalesce}), 0) "
            f"FROM spans WHERE {where} AND attributes IS NOT NULL "
            f"GROUP BY session_id",
            params,
        ).fetchall()
    except Exception:
        # Attribute shape varies by runtime; a reporting gap must never sink
        # the price arithmetic, which does not depend on it.
        return {}
    return {str(r[0]): int(r[1] or 0) for r in rows if r[0] and int(r[1] or 0) > 0}


def build_agent_price_rows(
    candidates: list[dict[str, Any]],
    window_days: float,
    thinking_by_session: dict[str, int] | None = None,
) -> list[AgentPriceRow]:
    """Group candidate sessions by (agent, provider, model) and price the swap.

    ``candidates`` carries one dict per candidate session with keys
    ``session_id``, ``agent_id``, ``provider``, ``model``, ``alt_model`` and the
    four token counts. A group whose current or proposed model has no pricing
    data is dropped rather than priced at a default rate. Rows come back
    largest-delta first.
    """
    thinking_by_session = thinking_by_session or {}
    groups: dict[tuple[str, str, str, str], _Acc] = {}
    thinking: dict[tuple[str, str, str, str], int] = {}
    saw_thinking: set[tuple[str, str, str, str]] = set()

    for row in candidates:
        key = (
            str(row.get("agent_id") or "unknown"),
            str(row["provider"]),
            str(row["model"]),
            str(row["alt_model"]),
        )
        acc = groups.setdefault(key, _Acc())
        session_id = str(row.get("session_id") or "")
        if session_id:
            acc.sessions.add(session_id)
        acc.input_tokens += int(row.get("input_tokens") or 0)
        acc.output_tokens += int(row.get("output_tokens") or 0)
        acc.cache_tokens += int(row.get("cache_tokens") or 0)
        acc.cache_write_tokens += int(row.get("cache_write_tokens") or 0)
        if session_id in thinking_by_session:
            thinking[key] = thinking.get(key, 0) + thinking_by_session[session_id]
            saw_thinking.add(key)

    rows: list[AgentPriceRow] = []
    for (agent, provider, model, alt_model), acc in groups.items():
        tokens = {
            "input_tokens": acc.input_tokens,
            "output_tokens": acc.output_tokens,
            "cache_tokens": acc.cache_tokens,
            "cache_write_tokens": acc.cache_write_tokens,
        }
        current = price_tokens(provider, model, **tokens)
        alternative = price_tokens(provider, alt_model, **tokens)
        if current is None or alternative is None:
            continue
        delta = current - alternative
        projected = (delta / window_days * PROJECTION_DAYS) if window_days > 0 else 0.0
        key = (agent, provider, model, alt_model)
        thinking_tokens = thinking.get(key) if key in saw_thinking else None
        rows.append(AgentPriceRow(
            agent_id=agent,
            provider=provider,
            model=model,
            alt_model=alt_model,
            sessions=len(acc.sessions),
            input_tokens=acc.input_tokens,
            output_tokens=acc.output_tokens,
            cache_tokens=acc.cache_tokens,
            cache_write_tokens=acc.cache_write_tokens,
            current_cost_usd=round(current, 6),
            alternative_cost_usd=round(alternative, 6),
            delta_usd=round(delta, 6),
            projected_30d_delta_usd=round(projected, 2),
            window_days=window_days,
            thinking_tokens=thinking_tokens,
            thinking_share_of_output=thinking_share(thinking_tokens, acc.output_tokens),
        ))
    rows.sort(key=lambda r: r.delta_usd, reverse=True)
    return rows
