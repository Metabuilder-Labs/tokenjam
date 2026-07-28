"""
Model-downgrade analyzer — "was this model the right ROLE for this session?"

Two cases, in order of how much they are worth:

**Primary — the driver-role case.** A premium-tier model acting as the DRIVER
of a long session, doing the reads and edits inline instead of routing them to
cheap workers. Every turn it runs inline re-reads the whole accumulated context
at the premium rate, so the material it introduces is billed again and again
until the next compaction. See :func:`analyze_driver_role` and
``analyzers/resend_tail.py`` for the tail arithmetic and the disjointness
partition against ``resend``.

**Secondary — the tiny-session case.** Sessions whose structural shape (short
input, short output, few tool calls) matches a class of work where a cheaper
model in the same provider family is worth reviewing. This was the analyzer's
whole definition until it was measured against a real coding-agent corpus: over
a 30-day window of 2,258 Claude Code sessions, 231 of them (10.2%) cleared the
``SMALL_*`` gate — but those sessions carried $6.74 of $8,222.53 in main-thread
spend, 0.082%, and only $4.81 of that had a cheaper same-family target at all.
The gate is not unreachable; it selects sessions that are worth nothing. So it
is kept (it is correct, just small) and demoted to a secondary contribution
beside the driver-role case, which is where the money for this persona is.

Neither case claims quality equivalence — only that the *shape* matches a class
worth a closer look.
"""
from __future__ import annotations

import re as _re
from datetime import datetime
from typing import Any

from tokenjam.core.context_diagnostic import (
    TurnComposition,
    load_turn_compositions,
)
from tokenjam.core.model_tiers import is_premium_tier
from tokenjam.core.optimize.analyzers.resend_tail import (
    DRIVER_TOOL_FANOUT_FLOOR,
    MIN_SESSION_CONTEXT_TOKENS,
    main_thread_turns,
    premium_driver_role,
    tail_lengths,
    tool_driven_stretch_mask,
)
from tokenjam.core.optimize.analyzers.batch_placement import (
    MIN_GROUP_COST_USD,
    MIN_SESSIONS_FOR_CADENCE,
    analyze_batch_placement,
)
from tokenjam.core.optimize.analyzers.downsize_agents import (
    thinking_tokens_by_session,
    build_agent_price_rows,
)
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.span_pricing import price_span, rates_at, span_instant
from tokenjam.core.optimize.stats import bootstrap_ci
from tokenjam.core.optimize.types import (
    AnalyzerContext,
    DowngradeExample,
    DriverRoleExample,
    DowngradeFinding,
    OpusAuditExample,
    OpusQuotaAudit,
)

# Structural heuristic thresholds for the SECONDARY tiny-session case. Sessions
# are flagged only when ALL three hold; the analyzer never claims the cheaper
# model would have produced the same answer — it claims the *shape* matches a
# class of work worth reviewing.
#
# Deliberately NOT loosened. Measured on a real coding-agent corpus these select
# 10.2% of sessions carrying 0.082% of spend (see the module docstring): the
# problem was never that the gate is too tight, it is that small sessions are
# cheap. Widening it produces MORE cards rather than bigger ones, against the
# standing don't-fill-the-inbox constraint; the driver-role case below is where
# the money is.
SMALL_INPUT_TOKENS = 5_000
SMALL_OUTPUT_TOKENS = 500
SMALL_TOOL_CALLS = 5

# Cap on the number of spot-check example sessions carried on the driver-role
# case, mirroring OPUS_AUDIT_MAX_EXAMPLES for the quota audit.
DRIVER_MAX_EXAMPLES = 5

# Per-TURN cheap-shape thresholds for the segment-level premium quota audit. A
# turn is a fraction of a session, so these sit well below the session-level
# constants above. Measured on NEW work (uncached input + output + this turn's
# tool fan-out), deliberately NOT on re-read/cache tokens: a mechanical turn can
# still re-read a huge context — that cache-read quota is exactly what a cheaper
# model would have burned more cheaply, so it belongs in the reclaimable metric,
# not the shape test. Starting calibration (non-blocking follow-up); left as
# named constants so a real-DB sweep can retune them in one place.
TURN_SMALL_INPUT = 2_000
TURN_SMALL_OUTPUT = 300
TURN_SMALL_TOOL_CALLS = 2

# A cheap segment is a maximal contiguous run of cheap-shaped turns. A run of at
# least this many is a flagged stretch; a shorter run is only flagged when it
# spans the session's ENTIRE set of turns (the whole-session floor — a session
# that was Sonnet-shaped end to end, which the old audit already flagged). This
# stops a lone mechanical turn wedged between two hard turns from counting.
MIN_STRETCH_TURNS = 2

# Cap on the number of spot-check example sessions carried in the audit.
OPUS_AUDIT_MAX_EXAMPLES = 5

# Premium → cheaper alternative in the same provider family. Pricing for both
# sides is resolved at runtime from pricing/models.toml; if either is missing
# the candidate is silently skipped (we won't invent a savings number).
# Premium-tier ladder (fable/mythos > opus > sonnet > haiku): Fable/Mythos and
# Opus (4.x) each drop TWO tiers (to Sonnet and Haiku respectively), the
# aggressive step justified because the quota audit only proposes these for
# structurally tiny (Sonnet-shaped) sessions. `claude-opus-5` is the one
# exception, mapped only ONE tier down to `claude-sonnet-5`: it is also the
# swap target subagent_rightsizing.py prices its own over_powered candidates
# against (a full agent loop, not a tiny session, so it earns the narrower
# step) — keeping both consumers of "what does opus-5 downgrade to" in
# agreement matters more than matching the opus-4.x precedent. `claude-sonnet-5`
# itself downgrades to haiku, same as sonnet-4.x. Keep new premium families in
# sync with tokenjam.core.model_tiers.PREMIUM_TIERS so every flagged session
# has a real routing target. `claude-mythos-5` used to be a cautionary example
# of the two lists drifting: it had a row here and a rate in models.toml but no
# "mythos" entry in model_tiers.TIER_SUBSTRINGS, so it was invisible to every
# premium-gated flag and only the tiny-session case below (gated on membership
# HERE, not on is_premium_tier) ever reached it. Both lists now carry it, and
# `test_every_priced_model_resolves_to_a_tier` fails the next half-added family.
DOWNGRADE_CANDIDATES: dict[str, dict[str, str]] = {
    "anthropic": {
        "claude-fable-5":    "claude-sonnet-4-6",
        "claude-mythos-5":   "claude-sonnet-5",
        "claude-opus-5":     "claude-sonnet-5",
        "claude-opus-4-8":   "claude-haiku-4-5",
        "claude-opus-4-7":   "claude-haiku-4-5",
        "claude-opus-4-6":   "claude-haiku-4-5",
        "claude-sonnet-5":   "claude-haiku-4-5",
        "claude-sonnet-4-6": "claude-haiku-4-5",
        "claude-sonnet-4-5": "claude-haiku-4-5",
    },
    "openai": {
        "gpt-4o":      "gpt-4o-mini",
        "o3":          "o4-mini",
    },
    "google": {
        "gemini-2-5-pro": "gemini-2-5-flash",
    },
}


def _model_name_candidates(model: str) -> list[str]:
    """Ordered DOWNGRADE_CANDIDATES keys to try for a raw model id.

    Tolerates the same id shapes ``is_premium_tier`` accepts so the downgrade
    lookup never disagrees with tier membership: bracketed context tags
    (``claude-opus-4-8[1m]``), trailing ``-YYYYMMDD`` dates, and Bedrock
    region/provider prefixes (``us-anthropic-claude-opus-4-8-20260115-v1`` →
    ``claude-opus-4-8``). The exact name is tried first; more-aggressive
    normalisations follow.
    """
    seen: list[str] = []

    def _add(name: str | None) -> None:
        if name and name not in seen:
            seen.append(name)

    _add(model)
    # Strip a bracketed context tag ([1m]) and/or a trailing YYYYMMDD date, in
    # both orders — mirrors pricing's own _lookup_candidates fallback.
    no_tag = _re.match(r"^(.*?)\[[^\]]*\]$", model)
    base_no_tag = no_tag.group(1) if (no_tag and no_tag.group(1)) else None
    _add(base_no_tag)
    for candidate in (model, base_no_tag):
        if not candidate:
            continue
        dated = _re.match(r"^(.*)-(\d{8})$", candidate)
        if dated:
            _add(dated.group(1))
    # Bedrock-hosted Claude ids embed the base Anthropic name amid a region/
    # provider prefix and a -YYYYMMDD-vN suffix; pull it out so a Bedrock opus
    # downgrades via the same base model as its first-party twin.
    embedded = _re.search(r"(claude-[a-z]+(?:-\d+)+)", model.lower())
    if embedded:
        name = embedded.group(1)
        redated = _re.match(r"^(.*)-(\d{8})$", name)
        _add(redated.group(1) if redated else name)
    return seen


def lookup_downgrade(provider: str, model: str) -> str | None:
    """The cheaper same-family alternative for ``(provider, model)``, or ``None``.

    Resolves the id shapes ``is_premium_tier`` accepts (bracketed context tags,
    trailing ``-YYYYMMDD`` dates, Bedrock region/provider prefixes) so the quota
    audit's premium recognition matches the subagent analyzer's — otherwise a
    ``claude-fable-5[1m]`` or Bedrock-hosted premium session is flagged by one
    and silently dropped by the other. Public (the premium quota audit and its
    routing export both need it); the underscore alias is kept so existing
    internal call sites don't churn.
    """
    candidates = _model_name_candidates(model)
    # The raw provider first; fall back to "anthropic" for Bedrock-hosted Claude
    # ids whose provider arrives as an aws/bedrock alias but whose downgrade
    # target is the same first-party Anthropic model.
    providers = [provider]
    if provider != "anthropic" and any(c.startswith("claude-") for c in candidates):
        providers.append("anthropic")
    for prov in providers:
        mapping = DOWNGRADE_CANDIDATES.get(prov, {})
        for name in candidates:
            if name in mapping:
                return mapping[name]
    return None


# Backward-compatible private alias — internal call sites predate the promotion.
_lookup_downgrade = lookup_downgrade


def _alt_unit_cost(provider: str, original_model: str, alt_model: str,
                   input_tokens: int, output_tokens: int, cache_tokens: int,
                   cache_write_tokens: int = 0, *,
                   at: datetime) -> float | None:
    """Cost of the given token mix priced at ``alt_model``, or ``None`` when the
    alternative has no pricing data at ``at``.

    Routes through :func:`tokenjam.core.optimize.span_pricing.price_span` — and
    so through :func:`tokenjam.core.cost.calculate_cost`, the ONE place that
    prices all four token classes (input, output, cache-read, cache-write) —
    rather than hand-rolling the arithmetic here a second time. The hand-rolled
    version used to price only input/output/cache-read, silently dropping
    cache-write from the alternative side while the actual side (the
    ``cost_usd`` column) already includes it; that asymmetry inflated every
    savings figure derived from this function by the full cache-write cost of
    the candidate. ``cache_write_tokens`` defaults to 0 for the one call site
    (the per-turn quota-audit counterfactual) whose upstream aggregation does
    not carry a cache-write figure at all — that is unchanged behaviour, not a
    new omission.

    ``at`` is the instant the ORIGINAL traffic ran, and is required. A
    counterfactual priced at a different date than the traffic it replaces
    would make the delta between them part rate-change and part model choice,
    with nothing in the number to say which — and the old default (whatever
    today's rate happened to be) made every historical window that way.

    ``original_model`` is unused by the arithmetic (kept for call-site
    readability / potential future per-model-pair overrides).
    """
    del original_model
    return price_span(
        provider, alt_model, at=at,
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_read_tokens=cache_tokens, cache_write_tokens=cache_write_tokens,
    )


# ---------------------------------------------------------------------------
# Primary case: a premium model in the DRIVER role
# ---------------------------------------------------------------------------


def _turns_by_session(
    conn, since: datetime, until: datetime, agent_id: str | None,
) -> dict[str, list[TurnComposition]]:
    """Window turns grouped by session, ordered, with tool activity attached.

    ``with_tool_activity=True`` is not optional here: ``tool_fanout`` and
    ``delegates`` default to inert values without it, which would make every
    premium session look like an undelegated tool-heavy one.
    """
    by_session: dict[str, list[TurnComposition]] = {}
    for turn in load_turn_compositions(
        conn, since, until, agent_id, ordered=True, with_tool_activity=True,
    ):
        by_session.setdefault(turn.session_id, []).append(turn)
    return by_session


DRIVER_ROLE_BASIS_TEMPLATE = (
    "Sessions where a premium-tier model drove the whole session inline: it "
    "never dispatched a subagent, ran {floor}+ main-thread tool calls, and "
    "accumulated more than {ctx:,} tokens of context. Two exact arithmetic "
    "halves over the turns inside a tool-driven stretch (a contiguous run of "
    "turns that each ran at least one tool — the reads/searches/edits a worker "
    "would have carried). (1) OFFLOAD: the uncached material those turns pulled "
    "in — tool results, file contents — is re-read by every later main-thread "
    "turn until the next compaction, billed at the premium cache-read rate; a "
    "worker's tool output never enters the caller's context, so that tail is "
    "removed. Only the pulled-in material is counted, never the assistant's own "
    "output, because a worker's short conclusion does come back to the caller. "
    "(2) TIER: those same turns' own input and output repriced at the "
    "substitute worker tier ({substitutes}) instead of the premium driver's. "
    "The two compound — where the work runs and what it runs on are "
    "independent. The premium driver itself is left in place and is NOT "
    "repriced: an interactive session must stay smart, so nothing here assumes "
    "you downgrade your own thread. Disjoint by construction from the resend "
    "card (which skips exactly these sessions) and from the subagent card "
    "(these sessions dispatch no subagent, so they carry no subagent spans)."
)


class _DriverAgg:
    """One flagged driver-role session's arithmetic, kept per session so the
    example list and the confidence interval both describe real sessions."""

    __slots__ = ("session_id", "agent_id", "provider", "model", "alt_model",
                 "offload_usd", "tier_usd", "tokens", "turns", "stretch_turns",
                 "tool_calls", "tail_tokens")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.agent_id = "unknown"
        self.provider = ""
        self.model = ""
        self.alt_model = ""
        self.offload_usd = 0.0
        self.tier_usd = 0.0
        self.tokens = 0
        self.turns = 0
        self.stretch_turns = 0
        self.tool_calls = 0
        self.tail_tokens = 0

    @property
    def recoverable_usd(self) -> float:
        return self.offload_usd + self.tier_usd


def _driver_session_arithmetic(
    session_id: str,
    main_turns: list[TurnComposition],
    provider: str,
    driver_model: str,
    alt_model: str,
    *,
    window_start: datetime,
) -> _DriverAgg | None:
    """Price one flagged session's inline-vs-routed counterfactual.

    ``None`` when the session carries no tool-driven stretch, or when no turn in
    one had priced rates on both sides — we refuse to invent a savings number.

    Both sides are resolved per turn at that turn's own timestamp, so the tier
    gap is a gap between two models on one date rather than a gap that has
    absorbed a price change. ``window_start`` is the fallback instant for a turn
    with no recorded time; see :func:`span_pricing.span_instant`.
    """
    # No `offloadable_share`-style factor is applied here, and that is a
    # deliberate difference in MECHANISM, not an omission — this case scopes by
    # SELECTION rather than by a fraction.
    #
    # `context_resend` multiplies every main-thread turn's tail by a measured
    # delegable-share proxy. This case instead prices only the turns `mask`
    # admits: contiguous runs where each turn actually ran a tool, inside
    # sessions `premium_driver_role` has already restricted to premium drivers
    # that never delegated. That selection IS the delegability scope — the
    # material claimed is the exploration/edit work a subagent would have
    # carried, taken at full weight, with everything outside a stretch priced
    # at nothing.
    #
    # So both analyzers scope the same quantity by different means: a fraction
    # of a broad population there, all of a narrow one here. The difference is
    # which structure each population offers to select on — resend's figure
    # deliberately covers the sessions this gate REJECTS, where no contiguous
    # tool-run signal is available to select with. Neither carries a
    # realization or compliance discount, and adding one here would be a
    # second, different claim rather than a consistency fix.
    mask = tool_driven_stretch_mask(main_turns)
    if not any(mask):
        return None
    lengths = tail_lengths(main_turns)

    agg = _DriverAgg(session_id)
    agg.provider = provider
    agg.model = driver_model
    agg.alt_model = alt_model
    agg.turns = len(main_turns)
    agg.tool_calls = sum(t.tool_fanout for t in main_turns)
    priced_any = False

    for i, turn in enumerate(main_turns):
        if not mask[i]:
            continue
        at = span_instant(turn.start_time, window_start=window_start)
        rates = rates_at(turn.provider or provider, turn.model, at)
        alt_rates = rates_at(provider, alt_model, at)
        if rates is None or alt_rates is None:
            continue
        priced_any = True
        agg.stretch_turns += 1

        # (1) OFFLOAD. Only `new_input_tokens` — the material this turn pulled
        # INTO the thread (tool results, file contents, pasted context) — is
        # counted as removed. The turn's own output is deliberately excluded:
        # a subagent's short conclusion does return to the caller, so claiming
        # its tail as well would be claiming a saving the fix does not deliver.
        # That is the one place this case is strictly more conservative than
        # `resend`'s own tail, which prices `new_input + output`.
        tail_tokens = turn.new_input_tokens * lengths[i]
        if rates.cache_read_per_mtok > 0:
            agg.offload_usd += tail_tokens / 1_000_000 * rates.cache_read_per_mtok
        agg.tail_tokens += tail_tokens

        # (2) TIER. The same turns' own work, repriced at the worker tier. Both
        # sides of the turn move: the worker reads the material at its own input
        # rate and produces the output at its own output rate. Clamped at zero
        # per rate so a pricing table where the "cheaper" model is dearer on one
        # axis can never turn into a negative contribution.
        agg.tier_usd += (
            turn.new_input_tokens / 1_000_000
            * max(0.0, rates.input_per_mtok - alt_rates.input_per_mtok)
            + turn.output_tokens / 1_000_000
            * max(0.0, rates.output_per_mtok - alt_rates.output_per_mtok)
        )
        agg.tokens += tail_tokens + turn.new_input_tokens + turn.output_tokens

    if not priced_any or agg.recoverable_usd <= 0:
        return None
    return agg


def analyze_driver_role(
    turns_by_session: dict[str, list[TurnComposition]],
    agent_by_session: dict[str, str],
    *,
    window_start: datetime,
    tool_fanout_floor: int = DRIVER_TOOL_FANOUT_FLOOR,
) -> list[_DriverAgg]:
    """Flagged driver-role sessions, priced.

    ``turns_by_session`` must come from ``load_turn_compositions(...,
    ordered=True, with_tool_activity=True)`` — the ordering is what makes a
    contiguous stretch meaningful, and ``tool_fanout`` / ``delegates`` are inert
    without the tool-activity join.

    The flag test itself is ``resend_tail.premium_driver_role``, shared with
    ``context_resend`` so the two analyzers partition the same population
    instead of each carrying its own copy of the thresholds.
    """
    flagged: list[_DriverAgg] = []
    for session_id, session_turns in turns_by_session.items():
        driver = premium_driver_role(
            session_turns, tool_fanout_floor=tool_fanout_floor,
        )
        if driver is None:
            continue
        provider, driver_model = driver
        if not provider:
            continue
        alt = lookup_downgrade(provider, driver_model)
        if not alt:
            continue
        agg = _driver_session_arithmetic(
            session_id, main_thread_turns(session_turns),
            provider, driver_model, alt, window_start=window_start,
        )
        if agg is None:
            continue
        agg.agent_id = agent_by_session.get(session_id, "unknown")
        flagged.append(agg)
    return flagged


def analyze_model_downgrade(
    conn,
    since: datetime,
    until: datetime,
    agent_id: str | None,
    window_days: float,
) -> DowngradeFinding | None:
    """
    Two cases, one finding (see the module docstring for why the split exists).

    PRIMARY — the driver-role case. A per-turn walk that flags premium-tier
    models driving long, undelegated, tool-heavy sessions and prices the
    inline-vs-routed counterfactual. This is the case that carries the money for
    a coding-agent corpus.

    SECONDARY — the tiny-session case. Walk sessions in the window. For each:
      - aggregate input/output/cache tokens, tool count, cost, dominant model
      - if model is in DOWNGRADE_CANDIDATES and shape matches the heuristic,
        treat as a candidate and compute alternative cost

    A session flagged by the driver-role case is EXCLUDED from the tiny-session
    case. The two gates can both admit the same session — the ``SMALL_*`` gate
    reads uncached ``input_tokens``, so a session with 3K of uncached input and
    500K of cache reads clears it while still being a context-heavy driver
    session — and both contribute to the one ``past_overspend_usd``, so
    without the exclusion the finding would double-count against itself.

    The candidate aggregation is scoped to the MAIN THREAD (``sub_agent_id IS
    NULL``). Task-dispatched subagent turns live under the parent session id,
    and ``subagent_rightsizing`` already prices the identical model swap over
    them (``sub_agent_id IS NOT NULL``, per (session, subagent)). Without this
    filter the two analyzers claim the same tokens under two structurally
    different signatures (``cost:downsize:<agent>`` and
    ``cost:subagent[:<name>]``), which ``past_overspend_rollup``'s
    dedup-by-signature cannot catch — so the Review-inbox headline came out
    inflated by the overlap. Same class of guard as
    ``cost_proposals._per_agent_cache_recoverable_by_model`` provides for the
    cache family; made disjoint at the source here because the subagent spans
    also skewed this query's ``MODE(model)``, which could name a subagent's
    model as the session's dominant one. The window-wide DENOMINATORS below
    stay unfiltered: only what we CLAIM narrows, never what we compare it to.
    """
    clauses = ["start_time >= $1", "start_time < $2", "session_id IS NOT NULL"]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)

    # First pass: main-thread LLM spans grouped by session.
    llm_rows = conn.execute(
        f"SELECT session_id, "
        f"FIRST(trace_id) AS trace_id, "
        f"FIRST(agent_id) AS agent_id, "
        f"MIN(start_time) AS start_time, MAX(end_time) AS end_time, "
        f"MIN(provider) AS provider, "
        f"MODE(model) AS model, "
        f"COALESCE(SUM(input_tokens),0)  AS input_tokens, "
        f"COALESCE(SUM(output_tokens),0) AS output_tokens, "
        f"COALESCE(SUM(cache_tokens),0)  AS cache_tokens, "
        f"COALESCE(SUM(cache_write_tokens),0) AS cache_write_tokens, "
        f"COALESCE(SUM(cost_usd),0.0)    AS cost_usd "
        f"FROM spans WHERE {where} AND model IS NOT NULL "
        f"AND sub_agent_id IS NULL "
        f"GROUP BY session_id",
        params,
    ).fetchall()

    if not llm_rows:
        return None

    # PRIMARY case first: its flagged sessions are excluded from the
    # tiny-session loop below, so it has to run before that loop, not after.
    agent_by_session = {
        str(r[0]): (str(r[2]) if r[2] else "unknown") for r in llm_rows if r[0]
    }
    driver_aggs = analyze_driver_role(
        _turns_by_session(conn, since, until, agent_id), agent_by_session,
        window_start=since,
    )
    driver_sessions = {a.session_id for a in driver_aggs}

    # Tool span counts per session (separate query — tool spans have model=NULL).
    # Main thread only, matching the aggregation above: a subagent's own tool
    # fan-out is not the parent thread's shape.
    tool_rows = conn.execute(
        f"SELECT session_id, COUNT(*) FROM spans "
        f"WHERE {where} AND tool_name IS NOT NULL AND sub_agent_id IS NULL "
        f"GROUP BY session_id",
        params,
    ).fetchall()
    tool_counts: dict[str, int] = {r[0]: int(r[1] or 0) for r in tool_rows if r[0]}

    # Window-wide denominators, over EVERY LLM span in the window — subagent
    # turns included. `percent_of_sessions` and `percent_of_tokens` are shares
    # of what the user actually spent; computing them over the main-thread
    # subset would shrink the denominator and overstate both.
    total_sessions, window_total_tokens = conn.execute(
        f"SELECT COUNT(DISTINCT session_id), "
        f"COALESCE(SUM(input_tokens),0) + COALESCE(SUM(output_tokens),0) "
        f"+ COALESCE(SUM(cache_tokens),0) + COALESCE(SUM(cache_write_tokens),0) "
        f"FROM spans WHERE {where} AND model IS NOT NULL",
        params,
    ).fetchone()
    total_sessions = int(total_sessions or 0)
    window_total_tokens = int(window_total_tokens or 0)

    candidate_sessions = 0
    actual_cost = 0.0
    alt_cost = 0.0
    candidate_tokens = 0
    examples: list[DowngradeExample] = []
    suggestions: dict[str, str] = {}
    # Per-candidate-session rows for the per-agent price arithmetic on the card
    # (exact per-type tokens at both models' rates). Collected in this same pass
    # so the card's agent breakdown and the finding's headline always describe
    # the identical set of sessions.
    price_candidates: list[dict[str, Any]] = []
    swaps: list[tuple[str, str, str]] = []
    # Per-candidate-session window savings, for the sampling-confidence interval
    # (#308). One value per candidate session = (actual cost − cheaper-model cost).
    per_session_savings: list[float] = []

    for row in llm_rows:
        session_id, trace_id, agent, start_time, end_time, provider, model, \
            in_tok, out_tok, cache_tok, cache_write_tok, cost = row
        if not provider or not model:
            continue
        if session_id in driver_sessions:
            # Already claimed, in full, by the driver-role case above. Both
            # cases feed one `past_overspend_usd`, so a session admitted
            # by both gates would be counted twice against itself.
            continue
        alt = _lookup_downgrade(provider, model)
        if not alt:
            continue
        tool_calls = tool_counts.get(session_id, 0)
        if not (
            in_tok < SMALL_INPUT_TOKENS
            and out_tok < SMALL_OUTPUT_TOKENS
            and tool_calls <= SMALL_TOOL_CALLS
        ):
            continue

        # Priced at this session's own start, not today: the row is a
        # per-session aggregate, and a session is short enough that its start is
        # inside whatever rate era billed all of it.
        alt_unit = _alt_unit_cost(
            provider, model, alt, int(in_tok), int(out_tok), int(cache_tok),
            int(cache_write_tok or 0),
            at=span_instant(start_time, window_start=since),
        )
        if alt_unit is None:
            # No pricing data for the alternative — refuse to invent a savings number.
            continue

        candidate_sessions += 1
        actual_cost += float(cost or 0.0)
        alt_cost += alt_unit
        candidate_tokens += (
            int(in_tok or 0) + int(out_tok or 0) + int(cache_tok or 0)
            + int(cache_write_tok or 0)
        )
        # This session's recoverable saving (clamped at 0 — a cheaper model
        # never costs more in our candidate set, but guard against pricing noise).
        per_session_savings.append(max(float(cost or 0.0) - alt_unit, 0.0))
        price_candidates.append({
            "session_id": str(session_id) if session_id else "",
            "agent_id": str(agent) if agent else "unknown",
            "provider": str(provider),
            "model": str(model),
            "alt_model": alt,
            "input_tokens": int(in_tok or 0),
            "output_tokens": int(out_tok or 0),
            "cache_tokens": int(cache_tok or 0),
            "cache_write_tokens": int(cache_write_tok or 0),
            "started_at": start_time,
        })
        suggestions[model] = alt
        if (provider, model, alt) not in swaps:
            swaps.append((provider, model, alt))

        if len(examples) < 3:
            duration = None
            try:
                if start_time and end_time:
                    duration = (end_time - start_time).total_seconds()
            except Exception:
                duration = None
            examples.append(DowngradeExample(
                trace_id=str(trace_id) if trace_id else "",
                session_id=str(session_id) if session_id else None,
                model=str(model),
                tool_calls=tool_calls,
                duration_seconds=duration,
                cost_usd=float(cost or 0.0),
            ))

    if candidate_sessions == 0 and not driver_aggs:
        return None

    savings_window = max(actual_cost - alt_cost, 0.0)
    driver_savings = sum(a.recoverable_usd for a in driver_aggs)
    driver_tokens = sum(a.tokens for a in driver_aggs)
    total_savings = savings_window + driver_savings
    monthly_savings = (total_savings / window_days * 30.0) if window_days > 0 else 0.0
    percent = (candidate_sessions / total_sessions * 100.0) if total_sessions else 0.0
    # The confidence interval brackets what the card actually shows, so it has
    # to resample BOTH cases' per-session values, not just the tiny-session one.
    per_session_savings.extend(a.recoverable_usd for a in driver_aggs)

    # Sampling confidence on the monthly projection (#308). Resample the
    # candidate sessions with replacement and recompute the projected monthly
    # savings, so the interval widens when the estimate rests on few sessions.
    # Same `scale` as monthly_savings so the CI brackets that exact figure.
    monthly_scale = (30.0 / window_days) if window_days > 0 else 1.0
    ci = bootstrap_ci(per_session_savings, scale=monthly_scale)
    ci_low = round(ci[0], 2) if ci else None
    ci_high = round(ci[1], 2) if ci else None
    percent_tokens = (
        candidate_tokens / window_total_tokens * 100.0
        if window_total_tokens > 0 else 0.0
    )
    monthly_tokens_in_candidates = (
        int(candidate_tokens / window_days * 30.0) if window_days > 0 else 0
    )

    # Per-agent price arithmetic (one row per agent/model group). The thinking
    # lookup is a separate, best-effort query: most runtimes never report a
    # thinking token count, and a row without one omits the share rather than
    # printing a zero. Scoped to the main thread like the candidate aggregation
    # above, so the share's numerator and denominator describe the same turns.
    per_agent = build_agent_price_rows(
        price_candidates, window_days,
        thinking_tokens_by_session(
            conn, since, until, agent_id, main_thread_only=True,
        ),
        window_start=since,
    )

    commands = [f"tjb run --original {p}:{orig} --candidate {p}:{alt}" for p, orig, alt in swaps]
    bench_command = "\n".join(commands) if commands else None

    # Driver-role rollup. `driver_substitutes` names the worker tier every
    # flagged driver would have routed to, which is what the basis string has to
    # state: a counterfactual whose substitute is unnamed is not inspectable.
    driver_substitutes: dict[str, str] = {}
    for a in driver_aggs:
        driver_substitutes[a.model] = a.alt_model
    driver_ordered = sorted(
        driver_aggs, key=lambda a: a.recoverable_usd, reverse=True,
    )
    driver_examples = [
        DriverRoleExample(
            session_id=a.session_id,
            agent_id=a.agent_id,
            model=a.model,
            alt_model=a.alt_model,
            turns=a.turns,
            stretch_turns=a.stretch_turns,
            tool_calls=a.tool_calls,
            tail_tokens=a.tail_tokens,
            offload_usd=round(a.offload_usd, 6),
            tier_usd=round(a.tier_usd, 6),
            recoverable_usd=round(a.recoverable_usd, 6),
        )
        for a in driver_ordered[:DRIVER_MAX_EXAMPLES]
    ]
    driver_basis = (
        DRIVER_ROLE_BASIS_TEMPLATE.format(
            floor=DRIVER_TOOL_FANOUT_FLOOR,
            ctx=MIN_SESSION_CONTEXT_TOKENS,
            substitutes=", ".join(
                f"{m} -> {alt}" for m, alt in sorted(driver_substitutes.items())
            ) or "none priced",
        )
        if driver_aggs else ""
    )

    return DowngradeFinding(
        candidate_sessions=candidate_sessions,
        total_sessions=total_sessions,
        actual_cost_usd=round(actual_cost, 6),
        alternative_cost_usd=round(alt_cost, 6),
        monthly_savings_usd=round(monthly_savings, 2),
        percent_of_sessions=round(percent, 1),
        examples=examples,
        suggestions=suggestions,
        bench_command=bench_command,
        candidate_tokens=candidate_tokens,
        window_total_tokens=window_total_tokens,
        percent_of_tokens=round(percent_tokens, 1),
        monthly_tokens_in_candidates=monthly_tokens_in_candidates,
        # THE canonical field (see the field contract in the repo CLAUDE.md).
        # The WINDOW savings, never the 30-day projection, so this analyzer's
        # figure sits on the same time basis as every other analyzer's and a
        # sum across them means something. This module's OWN
        # `monthly_savings_usd` / `monthly_tokens_in_candidates` survive for
        # the CLI's `tj optimize` line only; they reach NO cost card, NO
        # payload, and NO aggregate. Routing a paced figure onto a card is the
        # exact defect the field collapse removed — do not re-open it.
        #
        # Deliberately session-scoped, not turn-scoped: `audit_opus_quota`
        # below also computes a cheap-segment / mechanical-stretch dollar
        # counterfactual (`audit.actual_cost_usd`/`alternative_cost_usd`),
        # but that is a DIFFERENT metric on a DIFFERENT dataclass
        # (`OpusQuotaAudit`, not `DowngradeFinding`) for a DIFFERENT,
        # deliberately separate surface (`tj quota-audit` / GET
        # /api/v1/quota-audit) — verified by tracing every call site of
        # `audit_opus_quota` (cli/data_access.py, cli/cmd_quota_audit.py,
        # api/routes/quota_audit.py); `run()` above never calls it, so it
        # never reaches `ctx.report.downgrade` today. Three reasons this
        # candidate figure was NOT folded in here: (1) `audit_opus_quota`'s
        # own docstring and its "Secondary API-only implied-dollar
        # counterfactual" comment both say its dollar pair is explicitly
        # NEVER a headline — the audit's headline is
        # `percent_quota_misallocated`, a quota-weighted PERCENT, because the
        # subscription majority is on flat-rate plans a dollar figure
        # mis-targets; (2) it is scoped to PREMIUM-tier turns only
        # (`is_premium_tier`), never the full `DOWNGRADE_CANDIDATES` ladder
        # this function covers (sonnet->haiku, openai, google); folding it in
        # would silently narrow what "downsize recoverable" means on windows
        # with no premium-tier usage at all. (3) merging two independently
        # -scoped estimate methodologies onto one contract field is a
        # product decision beyond a bug fix — see CLAUDE.md anti-pattern #18
        # (a ticket's Where: line is the scope boundary, not an "approved
        # design" narrative to redesign a whole subsystem from).
        #
        # BOTH cases sum into this one field. They are disjoint by
        # construction: a driver-flagged session is skipped by the
        # tiny-session loop above, and the two never price the same span.
        past_overspend_usd=round(total_savings, 6),
        past_overspend_tokens=candidate_tokens + driver_tokens,
        estimate_basis=(
            (
                "PRIMARY (premium model in the driver role): " + driver_basis
                + " "
            ) if driver_basis else ""
        ) + (
            (
                "SECONDARY (structurally tiny sessions): candidate sessions "
                "routed to a cheaper model over the window — structural fit "
                "only, no quality validation."
            ) if candidate_sessions else ""
        ),
        n_sessions=candidate_sessions + len(driver_aggs),
        ci_low=ci_low,
        ci_high=ci_high,
        per_agent=per_agent,
        driver_sessions=len(driver_aggs),
        driver_recoverable_usd=round(driver_savings, 6),
        driver_offload_usd=round(sum(a.offload_usd for a in driver_aggs), 6),
        driver_tier_usd=round(sum(a.tier_usd for a in driver_aggs), 6),
        driver_tokens=driver_tokens,
        driver_tail_tokens=sum(a.tail_tokens for a in driver_aggs),
        driver_substitutes=driver_substitutes,
        driver_examples=driver_examples,
        driver_estimate_basis=driver_basis,
        # Per-session breakdown of `driver_tokens` above (they sum to it),
        # carried so the rule this card proposes can be placed in the projects
        # that actually incurred it — see `DowngradeFinding.driver_session_tokens`.
        driver_session_tokens={
            str(a.session_id): int(a.tokens) for a in driver_aggs if a.session_id
        },
    )


@register("downsize")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Mutates ctx.report.downgrade.

    Also runs the batch-placement check, a sibling module in this same lane
    (both answer "is this work on the right lane for its price?"), and attaches
    it under ``findings['placement']``. It stays a check rather than its own
    registered analyzer so the placement card ships without a new analyzer name
    on the CLI.

    ``analyze_batch_placement`` always returns a finding (never ``None``), so
    it is attached unconditionally — including a non-qualifying window, where
    it carries an empty candidate list plus the effective thresholds applied.
    That keeps ``findings['placement']`` reachable by ``_rank_findings`` /
    ``_render_placement`` in every case, so the CLI's empty-state message
    (with live threshold values) actually renders instead of being dead code.

    The one exception is the persona skip gate: because ``placement`` is a
    sub-check rather than a registry name, ``build_report``'s gate cannot drop
    it, so this function consults the same
    ``disabled_analyzers_for_persona`` map. When it's disabled the check is
    never run at all (no query, no empty finding attached), which is what
    keeps it out of the ranking, the payload and the Review inbox instead of
    surfacing an unactionable empty state.
    """
    # Deferred import: `runner` imports the analyzers package to trigger the
    # `@register` side effects, so a module-level import here would cycle.
    from tokenjam.core.optimize.runner import disabled_analyzers_for_persona

    ctx.report.downgrade = analyze_model_downgrade(
        ctx.conn, ctx.since, ctx.until, ctx.agent_id, ctx.window_days,
    )
    if "placement" in disabled_analyzers_for_persona(ctx.persona):
        return
    optimize_cfg = getattr(ctx.config, "optimize", None)
    ctx.report.findings["placement"] = analyze_batch_placement(
        ctx.conn, ctx.since, ctx.until, ctx.agent_id,
        ctx.summary.total_cost_usd or 0.0,
        min_sessions_for_cadence=getattr(
            optimize_cfg, "min_sessions_for_cadence", MIN_SESSIONS_FOR_CADENCE,
        ),
        min_group_cost_usd=getattr(
            optimize_cfg, "min_group_cost_usd", MIN_GROUP_COST_USD,
        ),
    )


# ---------------------------------------------------------------------------
# Premium-tier quota audit (retroactive Downsize, reframed as accountability)
# ---------------------------------------------------------------------------
# This reframes the structural downsize heuristic as an accountability "quota
# audit" scoped to premium-tier work (Fable + Opus, via the shared model_tiers
# predicate; issue #5, research/evidence/feature-downsize.md). The headline is
# "% of your premium (Opus/Fable) quota that went to Sonnet-shaped work" — a
# retrospective behaviour mirror in premium token share, not dollars and not
# "reclaimable" (the tokens are already spent) — because the subscription
# majority is on flat-rate plans where dollar framing mis-targets them
# (subscription-vs-cost-framing.md). It complements opusplan / `/model`
# (forward-looking) by answering the backward-looking question those tools
# can't: "which stretches of my PAST premium work were Sonnet-shaped?" Honest
# framing throughout — candidates to spot-check, never "safe to downgrade".
#
# Grain: a PER-TURN walk over the `TurnComposition` rows `tj context` already
# produces, NOT the old whole-session `MODE(model)` + SUM aggregate. Two
# structural fixes fall out:
#   (a) a mechanical STRETCH inside an otherwise-hard session is now detectable
#       (the whole-session heuristic structurally missed it — the hard part blew
#       the sums past the thresholds); and
#   (b) each turn's tokens count under ITS OWN model, so a Sonnet turn inside an
#       Opus session no longer inflates the premium numerator or denominator.
#
# The `audit_opus_quota` name and the `opus_*` fields on OpusQuotaAudit are kept
# for API/serialization stability; they now denote the whole premium tier, not
# Opus alone (the audit counts and flags Fable turns too), and are quota-weighted
# (the #119 weighting) rather than raw token sums.


def _is_cheap_shaped(turn: TurnComposition) -> bool:
    """True when a turn's *reasoning* footprint is Sonnet-shaped (design §2.1).

    Measured on NEW work — uncached input, output, this turn's tool fan-out — and
    on the absence of a Task delegation. Deliberately NOT gated on re-read /
    cache-write tokens: a mechanical turn can still re-read a huge context, and
    that cache-read quota is exactly what a cheaper model would have burned more
    cheaply — it belongs in the reclaimable metric, not the shape test.
    """
    return (
        turn.new_input_tokens < TURN_SMALL_INPUT
        and turn.output_tokens < TURN_SMALL_OUTPUT
        and turn.tool_fanout <= TURN_SMALL_TOOL_CALLS
        and not turn.delegates
    )


def _cheap_segments(
    session_turns: list[TurnComposition],
) -> list[list[TurnComposition]]:
    """Maximal contiguous runs of cheap-shaped turns (design §2.2).

    A non-cheap turn (or a delegation, which makes a turn non-cheap) breaks the
    run. Turns must already be ordered by ``start_time``.
    """
    segments: list[list[TurnComposition]] = []
    run: list[TurnComposition] = []
    for turn in session_turns:
        if _is_cheap_shaped(turn):
            run.append(turn)
        elif run:
            segments.append(run)
            run = []
    if run:
        segments.append(run)
    return segments


def _segment_counts(
    segment: list[TurnComposition], session_turn_count: int,
    *, min_stretch_turns: int = MIN_STRETCH_TURNS,
) -> bool:
    """Whether a cheap segment is flagged (design §2.2 + the whole-session floor).

    A stretch of at least ``min_stretch_turns`` counts; a shorter run counts only
    when it spans the session's ENTIRE turn set (a session that was Sonnet-shaped
    end to end — the whole-session floor the old audit already flagged, and the
    reason a single-turn cheap session is still counted). This is what keeps a
    lone mechanical turn wedged between two hard turns from being flagged.
    """
    return (
        len(segment) >= min_stretch_turns
        or len(segment) == session_turn_count
    )


class _SessionAgg:
    """Per-session aggregate of the flagged cheap-segment PREMIUM turns, for the
    spot-check example list (largest-quota first)."""

    __slots__ = ("session_id", "quota_by_model", "quota", "new_input",
                 "output", "reread", "tool_calls", "cost", "first_turn_at")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        # Earliest flagged turn in this session — the instant the counterfactual
        # below is priced at. Carried on the aggregate because the aggregate is
        # what gets priced: without it the dollar figure would fall back to
        # today's rate for traffic that ran weeks ago.
        self.first_turn_at: datetime | None = None
        # (provider, model) -> flagged premium quota, so the example reports the
        # model that actually carried the most misallocated quota in this session
        # rather than freezing on whichever premium turn happened to appear first
        # (a mixed-model session could otherwise mislabel the routing suggestion).
        self.quota_by_model: dict[tuple[str | None, str], float] = {}
        self.quota = 0.0
        self.new_input = 0
        self.output = 0
        self.reread = 0
        self.tool_calls = 0
        self.cost = 0.0

    def dominant_model(self) -> tuple[str | None, str]:
        """The (provider, model) carrying the most flagged premium quota in this
        session — the honest label for the spot-check example + routing hint."""
        provider, model = max(
            self.quota_by_model.items(), key=lambda kv: kv[1],
        )[0]
        return provider, model


def audit_opus_quota(
    conn,
    since: datetime,
    until: datetime,
    agent_id: str | None,
    window_days: float,
    config: Any | None = None,
) -> OpusQuotaAudit:
    """Retroactive segment-level premium-tier quota audit (issue #5).

    ``config`` (a ``TjConfig``, optional) supplies ``config.optimize.
    min_stretch_turns`` to override ``MIN_STRETCH_TURNS`` (config-overridable
    via ``core.config.OptimizeConfig``); ``None`` (the default — every current
    caller passes none) reproduces today's behaviour unchanged.

    Walks the window's assistant turns per session in ``start_time`` order and
    reports ONE honest figure (founder decision D1): the share of premium
    (Opus/Fable) quota that went to Sonnet-shaped *work* — whole Sonnet-shaped
    sessions PLUS mechanical stretches inside otherwise-hard sessions — on
    exact per-turn model attribution (D2). Quota-weighted (the #119 weighting),
    never a dollar headline; the `opus_*` field names are kept for serialization
    stability but denote the whole premium tier.

    The figure is a labelled ESTIMATE (D3): `segment_estimate_confidence` +
    `estimate_basis` mark it heuristic, and a WIDE bootstrap CI over resampled
    SEGMENTS brackets it (`segment_ci_low/high`) so the band widens honestly when
    few segments carry it.

    Computes purely from already-backfilled token/model metadata — no captured
    content (#3). Always returns an :class:`OpusQuotaAudit` (never ``None``) so
    the renderer can show an honest empty state.
    """
    optimize_cfg = getattr(config, "optimize", None)
    min_stretch_turns = getattr(
        optimize_cfg, "min_stretch_turns", MIN_STRETCH_TURNS,
    )

    turns = load_turn_compositions(
        conn, since, until, agent_id,
        ordered=True, with_tool_activity=True,
    )
    audit = OpusQuotaAudit(window_days=window_days)
    if not turns:
        return audit

    # Group into sessions, preserving the (already-applied) start_time ordering.
    by_session: dict[str, list[TurnComposition]] = {}
    for turn in turns:
        by_session.setdefault(turn.session_id, []).append(turn)

    premium_quota = 0.0            # denominator: quota-weighted premium turns
    misallocated_quota = 0.0       # numerator: premium turns in flagged segments
    segment_values: list[float] = []   # one premium-quota value per flagged segment
    suggestions: dict[str, str] = {}
    aggs: dict[str, _SessionAgg] = {}
    actual_cost = 0.0
    alt_cost = 0.0

    for session_id, session_turns in by_session.items():
        premium_turns = [t for t in session_turns if is_premium_tier(t.model)]
        if not premium_turns:
            continue
        audit.opus_sessions += 1
        premium_quota += sum(t.quota_weighted_tokens for t in premium_turns)

        session_flagged = False
        for segment in _cheap_segments(session_turns):
            if not _segment_counts(
                segment, len(session_turns), min_stretch_turns=min_stretch_turns,
            ):
                continue
            seg_premium = [t for t in segment if is_premium_tier(t.model)]
            if not seg_premium:
                continue
            seg_quota = sum(t.quota_weighted_tokens for t in seg_premium)
            misallocated_quota += seg_quota
            segment_values.append(seg_quota)
            audit.segment_count += 1
            session_flagged = True

            agg = aggs.setdefault(session_id, _SessionAgg(session_id))
            for t in seg_premium:
                agg.quota += t.quota_weighted_tokens
                agg.new_input += t.new_input_tokens
                agg.output += t.output_tokens
                agg.reread += t.reread_tokens
                agg.tool_calls += t.tool_fanout
                agg.cost += t.cost_usd
                if t.start_time is not None and (
                    agg.first_turn_at is None or t.start_time < agg.first_turn_at
                ):
                    agg.first_turn_at = t.start_time
                key = (t.provider, t.model)
                agg.quota_by_model[key] = (
                    agg.quota_by_model.get(key, 0.0) + t.quota_weighted_tokens
                )
                alt = (
                    lookup_downgrade(t.provider, t.model)
                    if t.provider else None
                )
                if alt:
                    suggestions[t.model] = alt

        if session_flagged:
            audit.candidate_sessions += 1

    # Secondary API-only implied-dollar counterfactual, aggregated over the
    # flagged premium turns and priced at each session's dominant premium model's
    # cheaper alternative (never the headline). Deliberately never folded into
    # `analyze_model_downgrade`'s `DowngradeFinding.past_overspend_usd`
    # (the `tj optimize` downsize card) — see the comment on that assignment
    # for why the two stay separate.
    for agg in aggs.values():
        provider, model = agg.dominant_model()
        alt = lookup_downgrade(provider, model) if provider else None
        if not alt:
            continue
        # ``alt`` is only set when ``provider`` was truthy above.
        assert provider is not None
        alt_unit = _alt_unit_cost(
            provider, model, alt, agg.new_input, agg.output, agg.reread,
            at=span_instant(agg.first_turn_at, window_start=since),
        )
        if alt_unit is not None:
            actual_cost += agg.cost
            alt_cost += alt_unit

    audit.opus_tokens = int(round(premium_quota))
    audit.candidate_tokens = int(round(misallocated_quota))
    audit.suggestions = suggestions
    audit.percent_quota_misallocated = (
        round(100.0 * misallocated_quota / premium_quota, 1)
        if premium_quota > 0 else 0.0
    )
    audit.percent_sessions = (
        round(100.0 * audit.candidate_sessions / audit.opus_sessions, 1)
        if audit.opus_sessions > 0 else 0.0
    )
    audit.actual_cost_usd = round(actual_cost, 6)
    audit.alternative_cost_usd = round(alt_cost, 6)

    # Confidence interval on the HEADLINE PERCENT (design §6, D3). Each
    # flagged segment contributes its premium-quota value; scaling the resampled
    # SUM by 100/premium_quota turns the bootstrap into an interval on the share.
    # Resampling segments (not sessions) widens the band honestly when the
    # estimate rests on few stretches; below 2 segments there is no spread to
    # bracket, so the bounds stay None (the estimate is inherently wide).
    if premium_quota > 0 and len(segment_values) >= 2:
        ci = bootstrap_ci(segment_values, scale=100.0 / premium_quota)
        if ci is not None:
            audit.segment_ci_low = round(ci[0], 1)
            audit.segment_ci_high = round(ci[1], 1)

    # Largest-quota candidate sessions first — the most worthwhile spot-checks.
    ordered_aggs = sorted(aggs.values(), key=lambda a: a.quota, reverse=True)
    audit.examples = [
        _example_for(agg) for agg in ordered_aggs[:OPUS_AUDIT_MAX_EXAMPLES]
    ]
    return audit


def _example_for(agg: _SessionAgg) -> OpusAuditExample:
    """Build a spot-check example labelled with the session's DOMINANT premium
    model (and its cheaper alternative), not whichever premium turn came first."""
    provider, model = agg.dominant_model()
    alt = (lookup_downgrade(provider, model) if provider else None) or ""
    return OpusAuditExample(
        trace_id="",
        session_id=agg.session_id,
        model=model,
        alt_model=alt,
        input_tokens=agg.new_input,
        output_tokens=agg.output,
        cache_tokens=agg.reread,
        tool_calls=agg.tool_calls,
        duration_seconds=None,
        cost_usd=round(agg.cost, 6),
    )
