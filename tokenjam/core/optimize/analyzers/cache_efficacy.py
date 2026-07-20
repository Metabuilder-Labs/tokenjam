"""
Cache-efficacy analyzer.

Measures the user's *current* prompt-caching usage by comparing cache_tokens
to input_tokens per (provider, model). Surfaces models where a large input
volume sees little caching — strong signal that enabling or expanding
`cache_control` placements would save tokens.

Per-provider scope:
  - Anthropic: fully supported. JSONL backfill and provider patches populate
    cache_tokens accurately, so the ratio is numerically meaningful.
  - OpenAI: best-effort. OpenAI's caching is implicit and per-call cache-hit
    data isn't exposed via the SDK in a consistent way. Output is rendered
    with an explicit caveat.
  - Google Gemini: best-effort, model-dependent. Some models expose cache
    metrics, some don't; the renderer surfaces a per-model caveat.
  - All others (Bedrock, LiteLLM, Cohere, etc.): unsupported in v1.

No content capture is required for this analyzer — it reads aggregate token
counts only.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.pricing import get_rates

# Minimum input volume to surface a recommendation. Below this, the
# absolute savings are negligible regardless of efficacy.
MIN_INPUT_TOKENS = 100_000

# Efficacy threshold below which we flag the (provider, model). At 30%,
# the model is still leaving substantial caching savings on the table.
EFFICACY_THRESHOLD = 0.30

# Per-provider support level. Used by the renderer to qualify the finding.
PROVIDER_SUPPORT: dict[str, str] = {
    "anthropic":     "full",
    "openai":        "best_effort",
    "google":        "best_effort",
    "bedrock":       "unsupported",
    "local.ollama":  "unsupported",
}


@dataclass
class CacheEfficacyRow:
    """One (provider, model) row of current caching usage."""
    provider:      str
    model:         str
    input_tokens:  int
    cache_tokens:  int
    efficacy:      float           # cache_tokens / (input_tokens + cache_tokens)
    support:       str             # full | best_effort | unsupported
    flagged:       bool            # surfaced as a recommendation candidate


# Realistic cache-read efficacy ceiling. The recoverable estimate measures the
# gap between current efficacy and this ceiling — not 100%, which is never
# achievable (system + first-call input is always uncached).
EFFICACY_CEILING = 0.80


@dataclass
class CacheEfficacyFinding:
    """All (provider, model) rows in the window, plus the flagged subset."""
    rows:        list[CacheEfficacyRow] = field(default_factory=list)
    flagged:     list[CacheEfficacyRow] = field(default_factory=list)
    confidence:  str = "structural"
    # The realistic efficacy ceiling, exposed so the UI can classify an
    # "already optimized" state without hardcoding the threshold (#135).
    efficacy_ceiling: float = EFFICACY_CEILING
    # Recoverable-savings contract (#111). See types.DowngradeFinding for the
    # field semantics. None when no row has a caching dimension to recover.
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis:               str          = ""
    estimate_confidence:          str          = "heuristic"
    # Root-caused per-agent proposals (A1/A2/A3, see the section below). Empty
    # when no agent group cleared a check's thresholds. Mutually exclusive per
    # agent — an agent appears in at most one of these three lists (uncached
    # beats thrash beats lookback).
    uncached_agents:      list["UncachedAgentCandidate"] = field(default_factory=list)
    thrash_agents:        list["ThrashAgentCandidate"]   = field(default_factory=list)
    lookback_miss_agents: list["LookbackMissCandidate"]  = field(default_factory=list)


def estimate_cache_recoverable(
    rows: list[CacheEfficacyRow],
) -> tuple[float | None, int | None]:
    """Estimate recoverable spend from closing the cache-efficacy gap.

    For each (provider, model) row with a known cache-read rate, take the gap
    between current efficacy and the 80% ceiling, apply it to the row's input
    tokens, and price the shifted tokens at the input-vs-cache rate delta.
    Returns (usd, tokens) summed across rows, or (None, None) when no row has a
    caching dimension to recover against.
    """
    from tokenjam.core.pricing import get_rates

    total_usd = 0.0
    total_tokens = 0
    any_priced = False
    for r in rows:
        rates = get_rates(r.provider, r.model)
        if rates is None or rates.cache_read_per_mtok <= 0:
            continue
        rate_delta = rates.input_per_mtok - rates.cache_read_per_mtok
        if rate_delta <= 0:
            continue
        gap = max(0.0, EFFICACY_CEILING - r.efficacy)
        if gap <= 0:
            continue
        recoverable_tokens = gap * r.input_tokens
        if recoverable_tokens <= 0:
            continue
        any_priced = True
        total_usd += (recoverable_tokens / 1_000_000) * rate_delta
        total_tokens += int(recoverable_tokens)

    if not any_priced:
        return None, None
    return round(total_usd, 6), total_tokens


def _compute_rows(conn, since, until, agent_id: str | None) -> list[CacheEfficacyRow]:
    """Aggregate input_tokens and cache_tokens per (provider, model) in window."""
    clauses = [
        "start_time >= $1", "start_time < $2",
        "provider IS NOT NULL", "model IS NOT NULL",
    ]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT provider, model, "
        f"COALESCE(SUM(input_tokens), 0) AS in_tok, "
        f"COALESCE(SUM(cache_tokens), 0) AS cache_tok "
        f"FROM spans WHERE {where} "
        f"GROUP BY provider, model "
        f"ORDER BY in_tok + cache_tok DESC",
        params,
    ).fetchall()

    result: list[CacheEfficacyRow] = []
    for provider, model, in_tok, cache_tok in rows:
        in_tok = int(in_tok or 0)
        cache_tok = int(cache_tok or 0)
        total = in_tok + cache_tok
        # Efficacy is the share of total input bytes served from cache. 0
        # means no caching; 1 means every byte was a cache read (impossible
        # in practice — there's always some uncached system+user input).
        efficacy = (cache_tok / total) if total > 0 else 0.0
        support = PROVIDER_SUPPORT.get(str(provider).lower(), "unsupported")
        flagged = (
            support in {"full", "best_effort"}
            and in_tok >= MIN_INPUT_TOKENS
            and efficacy < EFFICACY_THRESHOLD
        )
        result.append(CacheEfficacyRow(
            provider=str(provider),
            model=str(model),
            input_tokens=in_tok,
            cache_tokens=cache_tok,
            efficacy=round(efficacy, 4),
            support=support,
            flagged=flagged,
        ))
    return result


# --------------------------------------------------------------------------- #
# Root-caused per-agent cache proposals (A1 uncached / A2 thrash / A3 lookback
# miss). One shared data pass over per-agent, per-call usage records, then
# three classifiers applied in priority order so one underlying waste source
# never produces two cards (uncached beats thrash beats lookback).
# --------------------------------------------------------------------------- #

# A1 — minimum call volume before the absolute waste is worth a card, and the
# minimum median per-call input size below which even a perfectly cached
# prefix wouldn't save much.
MIN_CALLS_FOR_ROOT_CAUSE = 20
MIN_UNCACHED_MEDIAN_INPUT_TOKENS = 2048

# A2 — below this read:write ratio, more was spent WRITING the prefix than was
# ever recovered reading it back. Starting point; tune against founder data.
THRASH_READ_WRITE_RATIO_THRESHOLD = 1.0
# Inter-call gap (minutes) above which the thrash root cause is classified as
# TTL expiry rather than prefix instability.
TTL_CAUSE_GAP_MINUTES = 5.0
# Reference fact (spec sanity table, Anthropic pricing): a 1-hour cache write
# costs 2x the input rate, vs 1.25x for the default 5-minute write. The live
# pricing table only tracks one cache-write rate per model, so this multiplier
# is applied to the model's live `input_per_mtok` rather than a hardcoded $
# figure — never a raw dollar rate baked into analyzer logic.
ONE_HOUR_TTL_WRITE_MULTIPLIER = 2.0

# A3 — Anthropic's cache breakpoint search looks back at most this many
# content blocks (spec sanity table).
LOOKBACK_BLOCK_LIMIT = 20
# Construction constant for the block-count PROXY: tokenjam doesn't capture
# per-block content shape, so this converts an OBSERVABLE count (tool-call
# spans between two LLM calls in the same session) into an estimated block
# count — a tool call typically contributes a tool_use block (assistant turn)
# + a tool_result block (user turn). Documented on the card footnote; never
# presented as a byte-exact count.
BLOCKS_PER_TOOL_CALL = 2
MIN_LOOKBACK_MISS_RECURRENCE = 3

# The A2 "instability" card checklist — a diagnostic the user runs on their
# own prompt-assembly code, not a tokenjam-side detection.
SILENT_INVALIDATOR_CHECKLIST = (
    "Likely silent cache-invalidators to check in your prompt-assembly code: "
    "a timestamp or UUID placed early in the prompt; non-deterministic JSON "
    "key ordering; a tool set that varies per request; switching models "
    "mid-conversation."
)


@dataclass
class UncachedAgentCandidate:
    """A1 — an agent group making cacheable calls with prompt caching never
    attempted (zero cache_read AND zero cache_write on every call)."""
    agent_id:      str
    provider:      str
    model:         str
    calls:         int
    sessions:      int
    assumed_prefix_tokens: int      # p25 of per-call input_tokens (conservative)
    cache_control_snippet: str = ""  # the one-paste fix, this agent's own values
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis: str = ""


@dataclass
class ThrashAgentCandidate:
    """A2 — an agent attempting caching (regular cache_write) but paying more
    to write the prefix than it recovers reading it back."""
    agent_id:      str
    provider:      str
    model:         str
    calls:         int
    cache_write_tokens: int
    cache_read_tokens:  int
    read_write_ratio:   float
    cause:              str        # "ttl" | "instability"
    inter_call_gap_p50_minutes: float
    # Only set when cause == "ttl": whether the honest 1-hour-TTL break-even
    # arithmetic actually clears (can be negative -> False).
    ttl_worth_it:    bool | None  = None
    ttl_breakeven_usd: float | None = None
    cache_control_snippet: str = ""
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis: str = ""


@dataclass
class LookbackMissCandidate:
    """A3 — recurring cache misses that directly follow a long, tool-heavy
    turn: the shape of the 20-block breakpoint lookback limit. Weakest-
    confidence check of the three; only classified when A1/A2 don't already
    explain the agent's waste."""
    agent_id:      str
    provider:      str
    model:         str
    miss_count:    int
    avg_prior_turn_blocks: float
    cache_control_snippet: str = ""
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis: str = ""


@dataclass
class _AgentCallRow:
    """One LLM call, the shared per-agent unit A1/A2/A3 all classify over."""
    session_id:         str
    start_time:         Any            # datetime; Any avoids a hard import here
    provider:           str
    model:              str
    input_tokens:       int
    cache_tokens:        int           # cache-READ tokens
    cache_write_tokens: int


def _fetch_agent_calls(
    conn, since, until, agent_id: str | None,
) -> dict[str, list[_AgentCallRow]]:
    """The shared data pass: LLM spans in the window, grouped by agent_id,
    ordered by session then start_time. A1/A2/A3 all classify off this."""
    clauses = [
        "start_time >= $1", "start_time < $2",
        "agent_id IS NOT NULL", "model IS NOT NULL", "provider IS NOT NULL",
    ]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT agent_id, session_id, start_time, provider, model, "
        f"COALESCE(input_tokens, 0), COALESCE(cache_tokens, 0), "
        f"COALESCE(cache_write_tokens, 0) "
        f"FROM spans WHERE {where} ORDER BY agent_id, session_id, start_time",
        params,
    ).fetchall()
    by_agent: dict[str, list[_AgentCallRow]] = {}
    for aid, sid, start_time, provider, model, in_tok, cache_tok, cache_write in rows:
        by_agent.setdefault(str(aid), []).append(_AgentCallRow(
            session_id=str(sid or ""), start_time=start_time,
            provider=str(provider), model=str(model),
            input_tokens=int(in_tok or 0), cache_tokens=int(cache_tok or 0),
            cache_write_tokens=int(cache_write or 0),
        ))
    return by_agent


def _fetch_agent_tool_starts(
    conn, since, until, agent_id: str | None,
) -> dict[tuple[str, str], list[Any]]:
    """Tool-call span start times per (agent_id, session_id) — the A3 block-
    count proxy input."""
    clauses = [
        "start_time >= $1", "start_time < $2",
        "agent_id IS NOT NULL", "tool_name IS NOT NULL",
    ]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT agent_id, session_id, start_time FROM spans WHERE {where}",
        params,
    ).fetchall()
    by_key: dict[tuple[str, str], list[Any]] = {}
    for aid, sid, start_time in rows:
        by_key.setdefault((str(aid), str(sid or "")), []).append(start_time)
    return by_key


def _percentile(values: list[int], pct: float) -> float:
    """Linear-interpolated percentile (0.0-1.0) of a non-empty list. No numpy
    dependency."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = pct * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _dominant_provider_model(calls: list[_AgentCallRow]) -> tuple[str, str]:
    """(provider, model) of the most-called pair — used to price the finding
    and to name the model in the one-paste snippet."""
    counts: dict[tuple[str, str], int] = {}
    for c in calls:
        key = (c.provider, c.model)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return "unknown", ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _inter_call_gap_minutes(calls: list[_AgentCallRow]) -> list[float]:
    """Gaps (minutes) between consecutive calls within the same session."""
    by_session: dict[str, list[Any]] = {}
    for c in calls:
        by_session.setdefault(c.session_id, []).append(c.start_time)
    gaps: list[float] = []
    for times in by_session.values():
        times.sort()
        for a, b in zip(times, times[1:]):
            gaps.append((b - a).total_seconds() / 60.0)
    return gaps


def _uncached_snippet(model: str, prefix_tokens: int) -> str:
    return (
        f"# {model}: no prompt caching attempted; ~{prefix_tokens:,} tokens "
        "assumed stable prefix (this agent's own p25 input size)\n"
        + json.dumps({
            "type": "text",
            "text": "<the stable system / tool-definition prefix you send every call>",
            "cache_control": {"type": "ephemeral"},
        }, indent=2)
    )


def _ttl_snippet(model: str) -> str:
    return (
        f"# {model}: calls land more than {TTL_CAUSE_GAP_MINUTES:.0f} min "
        "apart; try the 1-hour cache TTL\n"
        + json.dumps({
            "type": "text",
            "text": "<the stable prefix>",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }, indent=2)
    )


def _lookback_snippet(model: str) -> str:
    return (
        f"# {model}: long tool-heavy turns are pushing the prior breakpoint "
        f"past the {LOOKBACK_BLOCK_LIMIT}-block lookback; add an intermediate "
        "breakpoint every ~15 content blocks in long tool-use turns\n"
        + json.dumps({
            "type": "tool_result",
            "tool_use_id": "<id>",
            "content": "...",
            "cache_control": {"type": "ephemeral"},
        }, indent=2)
    )


def _classify_a1(agent_id: str, calls: list[_AgentCallRow]) -> UncachedAgentCandidate | None:
    """Uncached agent: >=20 calls, never caches, and the prefix is large
    enough that caching it would actually matter."""
    if len(calls) < MIN_CALLS_FOR_ROOT_CAUSE:
        return None
    if any(c.cache_tokens > 0 or c.cache_write_tokens > 0 for c in calls):
        return None
    input_tokens = [c.input_tokens for c in calls]
    if statistics.median(input_tokens) < MIN_UNCACHED_MEDIAN_INPUT_TOKENS:
        return None

    provider, model = _dominant_provider_model(calls)
    prefix = int(_percentile(input_tokens, 0.25))
    sessions = len({c.session_id for c in calls})
    rates = get_rates(provider, model)
    usd: float | None = None
    tokens: int | None = None
    if rates is not None and prefix > 0:
        read_savings = (
            len(calls) * prefix
            * max(0.0, rates.input_per_mtok - rates.cache_read_per_mtok)
            / 1_000_000
        )
        # One write per session (conservative single-write-per-burst
        # assumption — a real prefix may re-write more than once per session).
        write_cost = sessions * prefix * rates.cache_write_per_mtok / 1_000_000
        usd = round(max(0.0, read_savings - write_cost), 6)
        tokens = max(0, len(calls) - sessions) * prefix

    return UncachedAgentCandidate(
        agent_id=agent_id, provider=provider, model=model, calls=len(calls),
        sessions=sessions, assumed_prefix_tokens=prefix,
        cache_control_snippet=_uncached_snippet(model, prefix),
        estimated_recoverable_usd=usd, estimated_recoverable_tokens=tokens,
        estimate_basis=(
            "assumed stable prefix = this agent's own p25 per-call input "
            "tokens; recoverable = calls x prefix x (input rate - cache-read "
            "rate), minus one 5-minute-TTL cache write per session "
            "(conservative single-write-per-burst assumption)"
        ),
    )


def _classify_a2(agent_id: str, calls: list[_AgentCallRow]) -> ThrashAgentCandidate | None:
    """Cache thrash: caching attempted regularly but the read:write ratio
    shows more was spent writing the prefix than was ever read back."""
    write_events = [c for c in calls if c.cache_write_tokens > 0]
    if not write_events or len(write_events) < max(1, len(calls) // 2):
        return None  # not "attempted regularly"
    total_write = sum(c.cache_write_tokens for c in calls)
    total_read = sum(c.cache_tokens for c in calls)
    if total_write <= 0:
        return None
    ratio = total_read / total_write
    if ratio >= THRASH_READ_WRITE_RATIO_THRESHOLD:
        return None

    gaps = _inter_call_gap_minutes(calls)
    gap_p50 = statistics.median(gaps) if gaps else 0.0
    cause = "ttl" if gap_p50 > TTL_CAUSE_GAP_MINUTES else "instability"

    provider, model = _dominant_provider_model(calls)
    rates = get_rates(provider, model)
    wasted_usd: float | None = None
    if rates is not None:
        wasted_usd = round(
            total_write / 1_000_000
            * max(0.0, rates.cache_write_per_mtok - rates.cache_read_per_mtok),
            6,
        )

    ttl_worth_it: bool | None = None
    ttl_breakeven_usd: float | None = None
    snippet: str
    if cause == "ttl":
        snippet = _ttl_snippet(model)
        if rates is not None:
            read_events = sum(1 for c in calls if c.cache_tokens > 0)
            bursts = len({c.session_id for c in calls}) or 1
            avg_write_tokens = total_write / len(write_events)
            cost_now = (
                total_write / 1_000_000 * rates.cache_write_per_mtok
                + total_read / 1_000_000 * rates.cache_read_per_mtok
            )
            # Under a 1-hour TTL: one write per session-burst (the prefix
            # survives the whole session instead of expiring every 5 min),
            # every other write/read event becomes a cache read.
            remaining_as_reads = max(0, len(write_events) + read_events - bursts)
            cost_1hr = (
                bursts * avg_write_tokens / 1_000_000
                * (ONE_HOUR_TTL_WRITE_MULTIPLIER * rates.input_per_mtok)
                + remaining_as_reads * avg_write_tokens / 1_000_000
                * rates.cache_read_per_mtok
            )
            ttl_breakeven_usd = round(cost_now - cost_1hr, 6)
            ttl_worth_it = ttl_breakeven_usd > 0
    else:
        snippet = SILENT_INVALIDATOR_CHECKLIST

    return ThrashAgentCandidate(
        agent_id=agent_id, provider=provider, model=model, calls=len(calls),
        cache_write_tokens=total_write, cache_read_tokens=total_read,
        read_write_ratio=round(ratio, 4), cause=cause,
        inter_call_gap_p50_minutes=round(gap_p50, 2),
        ttl_worth_it=ttl_worth_it, ttl_breakeven_usd=ttl_breakeven_usd,
        cache_control_snippet=snippet,
        estimated_recoverable_usd=wasted_usd,
        estimate_basis=(
            "wasted = cache-write tokens x (write rate - cache-read rate); "
            "what was paid to write the prefix versus what the same tokens "
            "would have cost read from a stable cache. The TTL variant's "
            "break-even (ttl_breakeven_usd) is a separate, honest projection "
            "that can come out negative — see cause=='ttl' ? ttl_worth_it."
        ),
    )


def _classify_a3(
    agent_id: str, calls: list[_AgentCallRow],
    tool_starts: dict[tuple[str, str], list[Any]],
) -> LookbackMissCandidate | None:
    """20-block lookback miss: a cache_read collapse directly following a
    turn whose (proxy) content-block count exceeds the lookback limit,
    recurring at least MIN_LOOKBACK_MISS_RECURRENCE times for the agent."""
    by_session: dict[str, list[_AgentCallRow]] = {}
    for c in calls:
        by_session.setdefault(c.session_id, []).append(c)

    miss_tokens: list[int] = []
    miss_blocks: list[int] = []
    for sid, session_calls in by_session.items():
        tool_times = sorted(tool_starts.get((agent_id, sid), []))
        for prev, cur in zip(session_calls, session_calls[1:]):
            if cur.cache_tokens > 0:
                continue  # not a miss
            prior_tool_calls = sum(
                1 for t in tool_times if prev.start_time < t <= cur.start_time
            )
            est_blocks = prior_tool_calls * BLOCKS_PER_TOOL_CALL
            if est_blocks > LOOKBACK_BLOCK_LIMIT:
                miss_tokens.append(cur.cache_write_tokens or cur.input_tokens)
                miss_blocks.append(est_blocks)

    if len(miss_tokens) < MIN_LOOKBACK_MISS_RECURRENCE:
        return None

    provider, model = _dominant_provider_model(calls)
    rates = get_rates(provider, model)
    usd: float | None = None
    tokens: int | None = None
    if rates is not None:
        usd = round(
            sum(miss_tokens) / 1_000_000
            * max(0.0, rates.cache_write_per_mtok - rates.cache_read_per_mtok),
            6,
        )
        tokens = sum(miss_tokens)

    return LookbackMissCandidate(
        agent_id=agent_id, provider=provider, model=model,
        miss_count=len(miss_tokens),
        avg_prior_turn_blocks=round(sum(miss_blocks) / len(miss_blocks), 1),
        cache_control_snippet=_lookback_snippet(model),
        estimated_recoverable_usd=usd, estimated_recoverable_tokens=tokens,
        estimate_basis=(
            f"cache breakpoints look back at most {LOOKBACK_BLOCK_LIMIT} "
            "content blocks; a proxy block count (tool-call spans between two "
            f"LLM calls x {BLOCKS_PER_TOOL_CALL}, for the tool_use + "
            "tool_result blocks each contributes) flags turns that likely "
            "pushed the prior breakpoint out of range. Recoverable = "
            "rewritten prefix tokens x (write rate - cache-read rate) per miss"
        ),
    )


def _compute_root_cause_candidates(
    conn, since, until, agent_id: str | None,
) -> tuple[
    list[UncachedAgentCandidate], list[ThrashAgentCandidate], list[LookbackMissCandidate],
]:
    """One shared pass, classified in priority order per agent: A1 (uncached)
    beats A2 (thrash) beats A3 (lookback miss) — one underlying waste source
    never produces two cards."""
    by_agent = _fetch_agent_calls(conn, since, until, agent_id)
    tool_starts = _fetch_agent_tool_starts(conn, since, until, agent_id)

    uncached: list[UncachedAgentCandidate] = []
    thrash: list[ThrashAgentCandidate] = []
    lookback: list[LookbackMissCandidate] = []
    for aid, calls in by_agent.items():
        a1 = _classify_a1(aid, calls)
        if a1 is not None:
            uncached.append(a1)
            continue
        a2 = _classify_a2(aid, calls)
        if a2 is not None:
            thrash.append(a2)
            continue
        a3 = _classify_a3(aid, calls, tool_starts)
        if a3 is not None:
            lookback.append(a3)
    return uncached, thrash, lookback


@register("cache")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches the finding to ctx.report.findings."""
    rows = _compute_rows(ctx.conn, ctx.since, ctx.until, ctx.agent_id)
    uncached, thrash, lookback = _compute_root_cause_candidates(
        ctx.conn, ctx.since, ctx.until, ctx.agent_id,
    )
    if not rows and not uncached and not thrash and not lookback:
        return
    rec_usd, rec_tokens = estimate_cache_recoverable(rows) if rows else (None, None)
    ctx.report.findings["cache"] = CacheEfficacyFinding(
        rows=rows,
        flagged=[r for r in rows if r.flagged],
        estimated_recoverable_usd=rec_usd,
        estimated_recoverable_tokens=rec_tokens,
        estimate_basis=(
            "gap between current cache-read efficacy and 80% ceiling at the "
            "input-vs-cache rate delta"
        ),
        uncached_agents=uncached,
        thrash_agents=thrash,
        lookback_miss_agents=lookback,
    )
