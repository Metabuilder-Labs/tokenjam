"""
Context re-send analyzer ("resend"): the product's headline waste category,
previously unmeasured.

This corpus's own benchmark (Princeton HAL, 9 runs, 21,562 calls) found that
**93.8% of prompt tokens sent to real agents were context they already sent**
(benchmarks/RESULTS.md, "2. Repeat-context detection"). No existing analyzer
measures this. `cache_efficacy` computes a caching-ADOPTION rate
(cache_tokens / (input_tokens + cache_tokens)); it reads 0.0 whenever a
scaffold never turned `cache_control` on, even if identical content is
re-sent every single turn. `core/context_diagnostic.py`'s `reread_share` is
adjacent but cache-READ based (a billing signal, nonzero only if caching
happened to be enabled) and is never imported by this package. This analyzer
is the structural gap: it measures repeat context independent of whether
caching was ever turned on, so it flags exactly what `cache_efficacy` misses.

Metric (benchmarks/RESULTS.md:223-231, preserved verbatim; do not invent a
variant):

    prompt_size(turn) = input_tokens + cache_tokens
    repeat_share = 1 - (max(prompt_size) / sum(prompt_size))

aggregated **token-weighted** across sessions:

    repeat_share = 1 - (sum of each session's max / sum of every prompt
                         token across all sessions)

This is an explicitly CONSERVATIVE LOWER BOUND (per the benchmark): if a
session's prompt size only ever grows turn over turn, `sum - max` is exactly
the repeated portion and the bound is tight; a session whose prompt size
sometimes shrinks (e.g. a mid-session `/compact`) only makes this an
UNDERESTIMATE of the true repeat share, never an overestimate.

Honesty discipline (CLAUDE.md Rule 14 / anti-pattern #22): `repeat_share`
itself is a measured token-share, not a savings claim; it is shown
regardless of pricing or caching state. The full repeat share is NOT claimed
as recoverable (the benchmark explicitly warns against this: 93.8% re-sent is
a different, larger number than the 68.3% of Anthropic spend the SAME corpus
found actually avoidable once caching was added). `estimated_recoverable_*`
below is discounted by AVOIDABLE_FRACTION_OF_REPEAT, and the dollar figure is
additionally scoped to only the currently-uncached share of the repeated
volume; see that constant's docstring and RESEND_ESTIMATE_BASIS for why.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from tokenjam.core.context_diagnostic import (
    RecurringInclusion,
    TurnComposition,
    compute_context_diagnostic,
    load_turn_compositions,
)
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.pricing import get_rates

# A window needs at least this many sessions and this many total LLM turns
# before the aggregate repeat-share means anything: a 1-2 session sample is
# noise, not a headline. Named separately from MIN_TURNS below because a
# window can clear one and not the other (e.g. 3 single-turn sessions clear
# the session count but carry zero possible signal).
MIN_SESSIONS_FOR_SIGNAL = 3
MIN_TURNS_FOR_SIGNAL = 6

# Cross-corpus calibration: the only real-world validated "how much of
# cache-blind context actually converts into savings" figure this codebase
# has produced. Measured on this repo's own HAL benchmark corpus (9 Princeton
# HAL runs, 21,562 calls) when prompt caching was added to previously
# cache-blind Anthropic-provider calls: spend fell from $778.16 to $246.57,
# a 68.3% reduction cross-checked against one real ground-truth case (see
# benchmarks/RESULTS.md, "1. Caching recommendations"). That 68.3% is a
# DIFFERENT metric from the 93.8% repeat-share above (dollars vs tokens,
# Anthropic-only vs all-providers); it is not a nested fraction of it. It is
# used here only as the best available proxy for "not every re-sent token
# converts to a recoverable saving" (cache lookback limits, TTL expiry, and
# prefix instability all eat into the theoretical maximum; see
# cache_efficacy.py's A2/A3 root causes for the mechanisms), so both
# recoverable claims below are discounted by it rather than claiming the
# full structural repeat share as recoverable.
AVOIDABLE_FRACTION_OF_REPEAT = 0.683

RESEND_HONESTY_CAVEAT = (
    "Structural token-share, not a savings claim: a conservative lower bound "
    "(benchmarks/RESULTS.md, HAL corpus: 93.8% of prompt tokens re-sent). "
    "Measured independent of whether caching is enabled: this can read high "
    "even when every re-sent byte was already a cheap cache read. Review "
    "sessions before restructuring."
)

RESEND_ESTIMATE_BASIS = (
    "repeat_tokens = sum(prompt_size) - max(prompt_size) per session "
    "(prompt_size = input_tokens + cache_tokens per turn), aggregated "
    "token-weighted across sessions. TOKENS claim (compaction lever): "
    "repeat_tokens x 68.3% avoidable-fraction (see AVOIDABLE_FRACTION_OF_REPEAT "
    "docstring); cache-agnostic, since compaction cuts gross token volume "
    "regardless of caching state. USD claim (cache_control-adoption lever): "
    "the CURRENTLY-UNCACHED share of that repeat volume only "
    "(repeat_tokens x new_input_tokens/prompt_tokens for the session), priced "
    "at the dominant model's (input - cache-read) rate delta x 68.3%; "
    "already-cached repeat volume already has its caching benefit, so "
    "re-claiming it here would double-count cache_efficacy's own recoverable "
    "estimate. Both are heuristic: reviewed against this repo's own benchmark "
    "corpus, not measured on this user's own data."
)

COMPACTION_FIX = (
    "Run /compact (or start a fresh session) once accumulated context crosses "
    "your working set. The repeated volume this finding measures is the same "
    "content being re-sent turn over turn: trimming it directly cuts future "
    "prompt size, regardless of whether caching is on."
)

# Cap on evidence rows carried in the finding payload; aggregates are over ALL
# sessions with measurable prompt volume, not just the capped examples.
TOP_N_EXAMPLES = 10


@dataclass
class ResendSessionExample:
    """One session's repeat-share breakdown: an evidence row, not the
    aggregate. Ranked by `repeat_tokens` descending (heaviest re-send first).
    """
    session_id: str
    turns: int
    prompt_tokens_sum: int
    prompt_tokens_max: int
    repeat_share: float
    repeat_tokens: int
    provider: str
    model: str


@dataclass
class ResendFinding:
    """Structural context-resend finding. See module docstring for the
    metric and the honesty discipline behind the recoverable estimates."""
    sessions_examined:   int = 0   # all sessions with an LLM turn in window
    multi_turn_sessions: int = 0   # subset with >= 2 turns (can structurally repeat)
    turns_examined:      int = 0
    # The headline: token-weighted aggregate repeat share across every
    # session with measurable prompt volume. None below the data threshold.
    repeat_share:        float | None = None
    repeat_share_median: float | None = None   # per-session median (benchmark parity)
    repeat_share_p90:    float | None = None   # per-session p90 (benchmark parity)
    repeat_tokens:       int = 0    # sum(session sum - session max), the raw resend volume
    prompt_tokens_total: int = 0    # denominator (sum of prompt_size over every turn)
    examples: list[ResendSessionExample] = field(default_factory=list)
    # The "why": recurring inclusions (re-read files, re-run searches,
    # re-pasted prompts/outputs) reused from context_diagnostic rather than
    # reimplemented (capture-gated; empty + a note when no capture toggle is on).
    recurring_examples: list[RecurringInclusion] = field(default_factory=list)
    # Both fixes are always carried: the lever differs by persona (agent
    # harness user: compaction; SDK user: cache_control), and the renderer
    # picks which to lead with. `fix_cache_control` is "" when no example
    # session had a model to name in the snippet.
    fix_compaction:    str = COMPACTION_FIX
    fix_cache_control: str = ""
    caveat:            str = RESEND_HONESTY_CAVEAT
    estimate_basis:    str = ""
    estimate_confidence: str = "heuristic"
    estimated_recoverable_tokens: int | None = None
    estimated_recoverable_usd:    float | None = None
    notes: list[str] = field(default_factory=list)


def _dominant_provider_model(turns: list[TurnComposition]) -> tuple[str, str]:
    """(provider, model) of the most-called pair in a session's turns."""
    counts = Counter((t.provider or "unknown", t.model) for t in turns)
    if not counts:
        return "unknown", ""
    return counts.most_common(1)[0][0]


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (0.0-1.0) of a non-empty list. No numpy
    dependency; mirrors cache_efficacy.py's own local helper."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = pct * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _cache_control_snippet(model: str, tokens: int) -> str:
    """The one-paste fix for the SDK-adoption lever, this session's own
    numbers (mirrors cache_efficacy.py's per-agent snippet style)."""
    return (
        f"# {model}: ~{tokens:,} tokens of this session's context are resent "
        "unchanged turn over turn and are not yet benefiting from caching\n"
        + json.dumps({
            "type": "text",
            "text": "<the stable prefix you resend every turn>",
            "cache_control": {"type": "ephemeral"},
        }, indent=2)
    )


def _capture_flags(config) -> tuple[bool, bool, bool]:
    capture = getattr(config, "capture", None)
    return (
        bool(capture and getattr(capture, "tool_inputs", False)),
        bool(capture and getattr(capture, "prompts", False)),
        bool(capture and getattr(capture, "tool_outputs", False)),
    )


@register("resend")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a ResendFinding to ctx.report.findings."""
    finding = ResendFinding()

    turns = load_turn_compositions(ctx.conn, ctx.since, ctx.until, ctx.agent_id, ordered=True)
    if not turns:
        finding.notes.append("No LLM turns in the window.")
        ctx.report.findings["resend"] = finding
        return

    by_session: dict[str, list[TurnComposition]] = defaultdict(list)
    for t in turns:
        by_session[t.session_id].append(t)

    finding.sessions_examined = len(by_session)
    finding.turns_examined = len(turns)
    finding.multi_turn_sessions = sum(1 for ts in by_session.values() if len(ts) >= 2)

    if len(by_session) < MIN_SESSIONS_FOR_SIGNAL:
        finding.notes.append(
            f"Only {len(by_session)} session(s) in the window (need >= "
            f"{MIN_SESSIONS_FOR_SIGNAL}): too few sessions to measure a "
            "stable repeat-share."
        )
        ctx.report.findings["resend"] = finding
        return
    if len(turns) < MIN_TURNS_FOR_SIGNAL:
        finding.notes.append(
            f"Only {len(turns)} LLM turn(s) in the window (need >= "
            f"{MIN_TURNS_FOR_SIGNAL}): too few turns to measure repeat-share."
        )
        ctx.report.findings["resend"] = finding
        return

    total_sum = 0
    total_max = 0
    examples: list[ResendSessionExample] = []
    priced_usd_total = 0.0
    any_priced = False

    for sid, session_turns in by_session.items():
        prompt_sizes = [t.new_input_tokens + t.reread_tokens for t in session_turns]
        s_sum = sum(prompt_sizes)
        if s_sum <= 0:
            # No measurable prompt volume at all: excluded from the share
            # distribution, same treatment RESULTS.md gives the one
            # zero-volume HAL trajectory in its corpus.
            continue
        s_max = max(prompt_sizes)
        total_sum += s_sum
        total_max += s_max

        repeat_share = 1.0 - (s_max / s_sum)
        repeat_tokens = s_sum - s_max
        provider, model = _dominant_provider_model(session_turns)
        examples.append(ResendSessionExample(
            session_id=sid, turns=len(session_turns),
            prompt_tokens_sum=s_sum, prompt_tokens_max=s_max,
            repeat_share=round(repeat_share, 4), repeat_tokens=repeat_tokens,
            provider=provider, model=model,
        ))

        if repeat_tokens <= 0:
            continue
        # Dollar opportunity, cache_control-adoption lever: scoped to the
        # CURRENTLY-UNCACHED share of this session's prompt volume only. See
        # RESEND_ESTIMATE_BASIS for why already-cached repeat volume is
        # excluded (double-counting cache_efficacy's own recoverable figure).
        total_new_input = sum(t.new_input_tokens for t in session_turns)
        uncached_fraction = (total_new_input / s_sum) if s_sum else 0.0
        uncached_repeat_tokens = repeat_tokens * uncached_fraction
        if uncached_repeat_tokens <= 0:
            continue
        rates = get_rates(provider, model)
        if rates is None or rates.cache_read_per_mtok <= 0:
            continue  # unpriced / no caching dimension for this model
        rate_delta = max(0.0, rates.input_per_mtok - rates.cache_read_per_mtok)
        if rate_delta <= 0:
            continue
        priced_usd_total += (
            uncached_repeat_tokens / 1_000_000 * rate_delta
            * AVOIDABLE_FRACTION_OF_REPEAT
        )
        any_priced = True

    if total_sum <= 0:
        finding.notes.append(
            "No session in the window carried measurable prompt-token volume."
        )
        ctx.report.findings["resend"] = finding
        return

    finding.prompt_tokens_total = total_sum
    finding.repeat_tokens = total_sum - total_max
    finding.repeat_share = round(1.0 - (total_max / total_sum), 4)

    shares = [e.repeat_share for e in examples]
    finding.repeat_share_median = round(statistics.median(shares), 4)
    finding.repeat_share_p90 = round(_percentile(shares, 0.90), 4)

    examples.sort(key=lambda e: e.repeat_tokens, reverse=True)
    finding.examples = examples[:TOP_N_EXAMPLES]

    finding.estimated_recoverable_tokens = round(
        AVOIDABLE_FRACTION_OF_REPEAT * finding.repeat_tokens
    )
    finding.estimated_recoverable_usd = round(priced_usd_total, 6) if any_priced else None
    finding.estimate_basis = RESEND_ESTIMATE_BASIS

    heaviest = finding.examples[0] if finding.examples else None
    if heaviest is not None and heaviest.model and heaviest.repeat_tokens > 0:
        finding.fix_cache_control = _cache_control_snippet(
            heaviest.model, heaviest.repeat_tokens
        )

    tool_inputs_captured, prompts_captured, tool_outputs_captured = _capture_flags(ctx.config)
    if tool_inputs_captured or prompts_captured or tool_outputs_captured:
        diag = compute_context_diagnostic(
            ctx.conn, ctx.since, ctx.until, agent_id=ctx.agent_id,
            tool_inputs_captured=tool_inputs_captured,
            prompts_captured=prompts_captured,
            tool_outputs_captured=tool_outputs_captured,
        )
        finding.recurring_examples = diag.recurring
    else:
        finding.notes.append(
            "Enable `[capture] tool_inputs = true` / `prompts = true` / "
            "`tool_outputs = true` in tj.toml, then `tj backfill claude-code "
            "--reingest`, to see WHICH re-read files, re-run searches, "
            "re-pasted prompts, or re-pasted outputs are driving this number."
        )

    ctx.report.findings["resend"] = finding
