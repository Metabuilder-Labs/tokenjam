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
regardless of pricing or caching state.

**Two dollar figures live on this finding and they are NEVER summed.**

`cost_of_waste_usd` is an OBSERVATION, not a saving: what the re-sent volume
actually cost over the window, priced per token class at the rates it really
billed at (cache reads at the cache-read rate, uncached repeat at the input
rate). Nothing is projected and nothing is discounted, because nothing is
being claimed — this answers "what did re-sending context cost me", which is
a question the data answers exactly. It is deliberately much larger than the
recoverable figure, and publishing it AS a saving would fail the product's
"if I apply the fix, do I actually save this?" bar outright: multi-turn
conversation inherently re-sends context (the floor is not zero), subagent
offload MOVES context rather than deleting it, and compaction only helps
forward and costs a summarization call of its own.

`estimated_recoverable_usd` is what the fix actually returns, and is derived
from THIS user's corpus, not from a cross-corpus constant. The lever is a
compound one — offload context-heavy in-thread work to a subagent AND
right-size that subagent — and both halves are measured here; see
`_offload_recoverable` and RESEND_ESTIMATE_BASIS.
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
from tokenjam.core.optimize.analyzers.model_downgrade import lookup_downgrade
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
# Anthropic-only vs all-providers); it is not a nested fraction of it.
#
# It is a calibration constant from ANOTHER corpus, so it prices the TOKENS
# claim only — the cache_control / compaction lever, whose mechanism is
# cross-corpus by nature. The DOLLAR claim no longer inherits it: the offload
# lever's avoidable fraction is measured from the user's own `sub_agent_id`
# telemetry instead (see `_measure_offloadable_share`), because that lever's
# realisable share is a property of how THIS user already works, not of a
# benchmark suite.
AVOIDABLE_FRACTION_OF_REPEAT = 0.683

# --- The offload lever, measured -------------------------------------------
# Claude Code's durable fix for repeated context is not caching and not
# `/compact`: it is keeping context-heavy work OFF the long-lived parent
# thread. A subagent's tool outputs live in the subagent's own context, so
# they are never re-billed on any later parent call. The saving is the tail
# that never happens:
#
#     saving_offload  ~ introduced_tokens x tail_calls x cache_read_rate
#     saving_rightsize ~ offloaded_tokens x (premium_rate - right_sized_rate)
#
# and the two COMPOUND — moving the work off the parent thread does not stop
# the subagent that now does it from running on a cheaper model.
#
# Scope, and why it is disjoint from every other analyzer's claim (Critical
# Rule 27 — two analyzers claiming `estimated_recoverable_*` must draw from
# disjoint spans). This claim is computed over MAIN-THREAD turns only
# (`sub_agent_id IS NULL`), so it cannot overlap `subagent`, which filters to
# `sub_agent_id IS NOT NULL`. And it only counts sessions whose accumulated
# main-thread context exceeds MIN_SESSION_CONTEXT_TOKENS, which no `downsize`
# candidate can be: that analyzer's structural gate is input < 5K tokens for
# the whole session.

#: A session only has an offload lever once its main-thread context is big
#: enough for re-reading it to cost real money. Same threshold
#: `subagent_rightsizing.CONTEXT_HEAVY_TOKENS` uses for the same idea, and
#: an order of magnitude above `downsize`'s 5K structural ceiling, which is
#: what keeps the two claims disjoint by construction.
MIN_SESSION_CONTEXT_TOKENS = 50_000

#: A prompt whose size falls to at most this share of the previous turn's has
#: had its context reset (a `/compact`, a resume, a fresh window). Material
#: introduced before that point is not re-read after it, so the tail stops
#: there rather than running to end-of-session.
COMPACTION_PROMPT_DROP_RATIO = 0.5

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
    "regardless of caching state, and cross-corpus calibrated rather than "
    "measured here. USD claim (subagent-offload + right-sizing lever, measured "
    "on YOUR data): for each main-thread turn of a context-heavy session, the "
    "material it introduces (uncached input + output) is re-read by every later "
    "main-thread turn until the next compaction, billed at the cache-read rate "
    "— that tail is what offloading the work to a subagent removes, because a "
    "subagent's tool output never enters the parent context. Only the share of "
    "that volume you demonstrably CAN offload is claimed, and that share is "
    "measured from your own sub_agent_id telemetry (how much of the "
    "context-introducing volume already runs in subagents, in the sessions "
    "where you delegate at all) rather than inherited from a benchmark. "
    "Right-sizing stacks on top: the same offloaded volume priced at the "
    "cheaper same-family model's input rate instead of the premium one. "
    "Computed over main-thread spans only, so it never overlaps the subagent "
    "analyzer's own claim."
)

RESEND_COST_OF_WASTE_BASIS = (
    "OBSERVED, not recoverable: what re-sent context actually cost over the "
    "window, priced per token class at the rates it really billed at — cache "
    "reads at the cache-read rate, the still-uncached share of the repeat "
    "volume at the input rate. Nothing here is projected or discounted because "
    "nothing is being claimed. Do NOT read this as a saving: multi-turn work "
    "inherently re-sends context (the floor is not zero), offloading to a "
    "subagent moves context rather than deleting it, and compaction only helps "
    "forward and costs a summarization call of its own. The figure the fix "
    "actually returns is estimated_recoverable_usd, which is much smaller and "
    "derived separately."
)

COMPACTION_FIX = (
    "Run /compact (or start a fresh session) once accumulated context crosses "
    "your working set. The repeated volume this finding measures is the same "
    "content being re-sent turn over turn: trimming it directly cuts future "
    "prompt size, regardless of whether caching is on. This is a manual, "
    "per-session action, so it never fixes the pattern going forward — treat "
    "it as immediate relief for an already-full session, not the durable fix."
)

# The durable claude-code lever: a rung-1 CLAUDE.md rule (same write machinery
# `script`/`reuse`/`verbosity` use via `cost_proposals._persona_gated_write_fields`)
# so the context that would otherwise get re-sent every turn never accumulates
# on the main thread in the first place. Unlike `/compact`, this persists
# across sessions and is on the CC action surface (a workspace file an
# orchestrating agent reads), which is why it leads for a claude-code window
# instead of `/compact` (founder critique, 2026-07-25: a real CC user abandons
# an over-full session and starts fresh rather than compacting it, so telling
# them to compact isn't a useful recommendation).
SUBAGENT_OFFLOAD_FIX = (
    "Offload context-heavy sub-tasks (broad file reads, multi-file search, "
    "long tool-output loops, exploratory investigation) to a subagent instead "
    "of running them inline in the main thread. A subagent's own tool logs "
    "and intermediate output stay in its own context; only its short "
    "conclusion returns to the caller, so the material that keeps getting "
    "re-sent turn over turn never accumulates on the main thread to begin "
    "with. Where available, pair this with a hook that warns once context "
    "crosses a size threshold, as a second, automated nudge toward the same "
    "behavior."
)

#: The second half of the compound lever. Offloading decides WHERE the work
#: runs; this decides what it runs ON. Both are settable in the same agent
#: file's frontmatter, so the two land as one artifact rather than two cards.
RIGHTSIZE_FIX_TEMPLATE = (
    "Then right-size what you offload to. A subagent doing broad reads and "
    "returning a short conclusion rarely needs the premium tier: pin both its "
    "model and its reasoning effort in its own definition file so every future "
    "dispatch inherits them instead of defaulting to whatever the parent runs "
    "on."
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
    # All three fixes are always carried: the lever differs by persona
    # (agent harness user: subagent-offload, with compaction as a secondary
    # immediate-relief note; SDK user: cache_control), and the renderer
    # picks which to lead with. `fix_cache_control` is "" when no example
    # session had a model to name in the snippet.
    fix_compaction:        str = COMPACTION_FIX
    fix_subagent_offload:  str = SUBAGENT_OFFLOAD_FIX
    fix_rightsize:         str = RIGHTSIZE_FIX_TEMPLATE
    fix_cache_control:     str = ""
    caveat:            str = RESEND_HONESTY_CAVEAT
    estimate_basis:    str = ""
    estimate_confidence: str = "heuristic"
    estimated_recoverable_tokens: int | None = None
    estimated_recoverable_usd:    float | None = None
    # COST OF WASTE — an observation, never a saving, and NEVER summed with
    # `estimated_recoverable_usd` anywhere. See the module docstring and
    # `RESEND_COST_OF_WASTE_BASIS`. `None` when no turn in the window carried a
    # priced model (a zero would read as "re-sending context is free").
    cost_of_waste_usd:      float | None = None
    cost_of_waste_tokens:   int = 0
    cost_of_waste_basis:    str = RESEND_COST_OF_WASTE_BASIS
    # The two halves of the compound recoverable claim, kept visible so the
    # headline is never a black box. They ARE summed into
    # `estimated_recoverable_usd` — unlike cost-of-waste, these price the same
    # fix and are independent of one another (moving work off the parent thread
    # does not stop it from also running on a cheaper model).
    offload_recoverable_usd:   float | None = None
    rightsize_recoverable_usd: float | None = None
    #: Share of context-introducing volume this user already routes through
    #: subagents, in the sessions where they delegate at all — the measured
    #: replacement for the inherited 68.3% constant on the dollar claim.
    #: `None` when no session in the window delegates, in which case no
    #: offload dollar figure is claimed.
    offloadable_share:         float | None = None
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


def _introduced_tokens(turn: TurnComposition) -> int:
    """Material this turn ADDS to the conversation, and that every later turn
    in the same context window therefore re-reads: uncached input (a tool
    result, a pasted file, a user message) plus the assistant's own output.
    Cache reads are excluded — those are the re-reading, not the material."""
    return turn.new_input_tokens + turn.output_tokens


def _measure_offloadable_share(by_session: dict[str, list[TurnComposition]]) -> float | None:
    """Share of context-introducing volume this user already routes through
    subagents, measured across the sessions where they delegate at all.

    This is the corpus-measured replacement for inheriting a cross-corpus
    constant. Telemetry carries ``sub_agent_id``, so in-thread and offloaded
    work are directly comparable: sessions that already lean on subagents show
    what fraction of the material is offloadable IN PRACTICE for this user's
    kind of work. Sessions that never delegate are excluded from the
    measurement (they have nothing to measure) but are exactly where the
    saving is then claimed.

    ``None`` when no session in the window delegates — nothing to measure, so
    nothing is claimed rather than a fraction being invented.
    """
    delegated = 0
    total = 0
    for turns in by_session.values():
        if not any(t.sub_agent_id for t in turns):
            continue
        for turn in turns:
            introduced = _introduced_tokens(turn)
            total += introduced
            if turn.sub_agent_id:
                delegated += introduced
    if total <= 0 or delegated <= 0:
        return None
    return min(delegated / total, 1.0)


def _resend_tail_tokens_per_turn(main_turns: list[TurnComposition]) -> list[int]:
    """Per-turn tail-token contribution — same computation as
    ``_resend_tail_tokens``, kept ungrouped so each turn's own contribution can
    be priced at that turn's own model's rate rather than one rate applied to
    the session-wide sum.

    For each turn, the material it introduces is re-read by every later
    main-thread turn until the next compaction — a turn whose prompt collapses
    to at most ``COMPACTION_PROMPT_DROP_RATIO`` of the previous one's, which is
    what a ``/compact`` or a context reset looks like in the data. Counting
    past that boundary would claim a cost the user never paid.

    Returns, per turn, ``introduced x tail_length``, i.e. tokens billed as
    cache reads purely because the work stayed on the parent thread.
    """
    n = len(main_turns)
    prompt_sizes = [t.new_input_tokens + t.reread_tokens for t in main_turns]
    # boundary_after[i] = index of the first compaction boundary strictly after
    # turn i, or n when the context survives to the end of the session.
    boundary_after = [n] * n
    next_boundary = n
    for i in range(n - 1, -1, -1):
        boundary_after[i] = next_boundary
        collapsed = (
            i >= 1
            and prompt_sizes[i - 1] > 0
            and prompt_sizes[i] <= prompt_sizes[i - 1] * COMPACTION_PROMPT_DROP_RATIO
        )
        if collapsed:
            next_boundary = i

    return [
        _introduced_tokens(turn) * max(boundary_after[i] - i - 1, 0)
        for i, turn in enumerate(main_turns)
    ]


def _resend_tail_tokens(main_turns: list[TurnComposition]) -> int:
    """Window-scoped total of ``_resend_tail_tokens_per_turn`` — kept for
    callers (and tests) that only need the session's aggregate tail."""
    return sum(_resend_tail_tokens_per_turn(main_turns))


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

    # Measured on this user's own telemetry, not inherited: how much of the
    # context-introducing volume already runs in subagents where they delegate
    # at all. `None` means the corpus can't answer it, so no dollar claim.
    offloadable_share = _measure_offloadable_share(by_session)

    total_sum = 0
    total_max = 0
    examples: list[ResendSessionExample] = []
    waste_usd_total = 0.0
    waste_tokens_total = 0
    any_waste_priced = False
    offload_usd_total = 0.0
    rightsize_usd_total = 0.0
    offload_tokens_total = 0

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

        # COST OF WASTE (observed), priced per TURN at that turn's OWN model's
        # rate — a session that mixes models (e.g. opus for some turns, haiku
        # for others) must not have every turn priced at whichever model
        # happened to dominate the turn count. `repeat_tokens` is inherently a
        # session-level quantity (it comes from the sum-vs-max prompt-size
        # comparison, not a per-turn one), so its uncached share is allocated
        # across turns in proportion to each turn's own `new_input_tokens` —
        # the same proportion the old single blended fraction expressed in
        # aggregate, just applied per turn instead of once. Every cache read
        # IS re-sent context by definition, billed at that turn's cache-read
        # rate; the still-uncached share billed at that turn's input rate.
        # Nothing discounted, nothing projected — this is what it cost, not
        # what a fix returns. NEVER summed with the recoverable figures below.
        for t in session_turns:
            turn_rates = get_rates(t.provider or "unknown", t.model)
            if turn_rates is None or turn_rates.input_per_mtok <= 0:
                continue  # this turn's model unpriced: contributes no dollar figure
            uncached_repeat = repeat_tokens * (t.new_input_tokens / s_sum) if s_sum else 0.0
            waste_usd_total += (
                t.reread_tokens / 1_000_000 * turn_rates.cache_read_per_mtok
                + uncached_repeat / 1_000_000 * turn_rates.input_per_mtok
            )
            waste_tokens_total += t.reread_tokens + round(uncached_repeat)
            any_waste_priced = True

        # RECOVERABLE (the compound offload + right-size lever). Main-thread
        # turns only, and only in sessions whose context is heavy enough for
        # the lever to exist — see MIN_SESSION_CONTEXT_TOKENS for why that also
        # keeps this disjoint from `downsize`. Priced per turn at that turn's
        # own model's rate, same reasoning as cost-of-waste above.
        if offloadable_share is None:
            continue
        main_turns = [t for t in session_turns if not t.sub_agent_id]
        if len(main_turns) < 2:
            continue
        if max((t.new_input_tokens + t.reread_tokens for t in main_turns), default=0) < MIN_SESSION_CONTEXT_TOKENS:
            continue

        # The tail that offloading removes: material re-read by later
        # main-thread turns purely because the work stayed in the thread.
        for t, tail_tokens in zip(main_turns, _resend_tail_tokens_per_turn(main_turns)):
            turn_rates = get_rates(t.provider or "unknown", t.model)
            if turn_rates is None or turn_rates.cache_read_per_mtok <= 0:
                continue
            offloadable_tail = tail_tokens * offloadable_share
            offload_usd_total += offloadable_tail / 1_000_000 * turn_rates.cache_read_per_mtok

            # Right-sizing stacks independently: the same offloaded material
            # still has to be read once by whatever runs it, so pricing it at
            # the cheaper same-family model's input rate instead of this
            # turn's is a second, non-overlapping cut. Skipped when no
            # cheaper alternative is priced for this turn's own model.
            offloaded_material = _introduced_tokens(t) * offloadable_share
            offload_tokens_total += round(offloadable_tail + offloaded_material)
            alt = lookup_downgrade(t.provider or "unknown", t.model)
            alt_rates = get_rates(t.provider or "unknown", alt) if alt else None
            if alt_rates is not None:
                rate_gap = max(0.0, turn_rates.input_per_mtok - alt_rates.input_per_mtok)
                rightsize_usd_total += offloaded_material / 1_000_000 * rate_gap

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
    finding.cost_of_waste_usd = round(waste_usd_total, 6) if any_waste_priced else None
    finding.cost_of_waste_tokens = waste_tokens_total
    finding.offloadable_share = (
        round(offloadable_share, 4) if offloadable_share is not None else None
    )
    if offloadable_share is not None and offload_tokens_total > 0:
        finding.offload_recoverable_usd = round(offload_usd_total, 6)
        finding.rightsize_recoverable_usd = round(rightsize_usd_total, 6)
        # The two halves compound: offloading decides where the work runs,
        # right-sizing decides what it runs on. Neither cancels the other, so
        # they sum — unlike cost-of-waste, which never enters this figure.
        finding.estimated_recoverable_usd = round(
            offload_usd_total + rightsize_usd_total, 6
        )
    else:
        finding.notes.append(
            "No dollar figure for the offload lever: this window has no "
            "session that both delegates to a subagent (nothing to measure "
            "your offloadable share from) and carries enough main-thread "
            "context for offloading to pay. The token figure above still "
            "stands."
        )
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
