"""Shared context-tail math, and the one definition of a "premium driver"
session that partitions ``resend``'s and ``downsize``'s dollar claims.

Two analyzers price the same mechanism — material that stays on the long-lived
main thread gets re-read by every later turn until the next compaction — so the
arithmetic lives here once rather than being reimplemented on each side:

* ``context_resend`` claims the offload lever over context-heavy sessions.
* ``model_downgrade`` claims the driver-role lever: a PREMIUM-tier model acting
  as the driver of a long session, doing the reads and edits inline instead of
  routing them to cheap workers.

Those two claims describe the identical tokens, so Critical Rule 27 (two
analyzers claiming ``estimated_recoverable_*`` must draw from DISJOINT spans)
would be violated if both ran over the same sessions. :func:`premium_driver_role`
is the partition: a session it flags belongs to ``downsize``, every other
context-heavy session belongs to ``resend``. Both analyzers call THIS function —
not a local copy — so the two populations cannot drift apart.

This module deliberately holds no query and no pricing: it takes
``TurnComposition`` rows and returns token counts and predicates, so it is
importable from either analyzer without a cycle (``context_resend`` already
imports ``lookup_downgrade`` from ``model_downgrade``, so the shared code cannot
live in either of them).
"""
from __future__ import annotations

from tokenjam.core.context_diagnostic import TurnComposition
from tokenjam.core.model_tiers import is_premium_tier

#: THE `relearn` / `resend` BOUNDARY, stated once so neither side can restate
#: it differently. Both analyzers are imported from here, and both quote this
#: constant in their basis strings rather than paraphrasing it.
#:
#: The two look overlapping because on a coding corpus almost every token in
#: almost every call is re-sent context — measured on the local corpus, the
#: MEDIAN turn in a failure-bearing session is 96,064 prompt tokens of which
#: 95,919 (99.8%) are cache reads and 2 are fresh input. So "the cost of a
#: retry turn" and "re-sent context" name nearly the same physical tokens, and
#: an unstated boundary would let both cards price them.
#:
#: The line is the COUNTERFACTUAL, not the token class:
#:
#:   * `relearn` owns whether a call EXISTS. Its fix removes the failure, so
#:     the retry turn never happens and its ENTIRE cost goes with it —
#:     including the context that turn re-read. You do not pay a call's cache
#:     reads for a call you never make. That is why relearn prices the whole
#:     turn and not just the ~2 fresh tokens it introduced; pricing only the
#:     new tokens would value eliminating a 96k-token call at a fraction of a
#:     cent, which is false.
#:   * `resend` owns how BIG a call is. Its levers (offload to a subagent,
#:     compaction) shrink calls that still have to happen. It never claims a
#:     call's existence, only its size.
#:
#: Two consequences that keep the claims disjoint in practice:
#:
#:   1. relearn does NOT claim the error text's own downstream tail — the
#:      ~1,500 tokens of error text re-read by every LATER call. Those later
#:      calls did have to happen, so their size is resend's population.
#:      relearn reports that share as an observation (`past_reread_*`) and
#:      excludes it from its claim.
#:   2. The residual overlap runs the other way: a retry turn is itself one of
#:      the later calls resend counts in its tails, so resend's claim includes
#:      re-reads performed by calls relearn says should not exist. Measured
#:      2026-07-26 this is 5,735 retry turns against 275,537 LLM calls
#:      (2.08%), and after resend's own in-scope and offloadable-share gates
#:      it is under 0.1% of resend's claim — bounded, disclosed, and pinned by
#:      `tests/unit/test_relearn_resend_boundary.py` rather than left to be
#:      rediscovered.
RELEARN_RESEND_BOUNDARY = (
    "relearn prices calls that should not have happened; resend prices the "
    "size of calls that had to. A failure's fix removes the retry turn "
    "entirely, so relearn claims that turn whole — including the context it "
    "re-read, which is not billed at all once the call is gone. The error "
    "text's own re-read tail on LATER calls is excluded from relearn's claim "
    "and reported as an observation, because those calls happen regardless "
    "and their size is resend's to claim."
)

#: A prompt whose size falls to at most this share of the previous turn's has
#: had its context reset (a ``/compact``, a resume, a fresh window). Material
#: introduced before that point is not re-read after it, so a tail stops there
#: rather than running to end-of-session.
COMPACTION_PROMPT_DROP_RATIO = 0.5

#: A session only has an offload lever once its main-thread context is big
#: enough for re-reading it to cost real money. Same threshold
#: ``subagent_rightsizing.CONTEXT_HEAVY_TOKENS`` uses for the same idea.
MIN_SESSION_CONTEXT_TOKENS = 50_000

#: Evidence that a driver session actually had routable work in it: total
#: main-thread tool calls. A premium session that ran two greps is not a
#: "should have delegated" session, it is a short conversation. Measured on the
#: corpus this was calibrated against (30d, 2,356 sessions): the median
#: context-heavy non-delegating premium session runs 29 main-thread tool calls,
#: p25 is 18, so a floor of 10 drops only the genuinely tool-light tail
#: (782 sessions qualify structurally, 748 clear this floor).
DRIVER_TOOL_FANOUT_FLOOR = 10

#: A tool-driven stretch is a maximal contiguous run of at least this many
#: main-thread turns that each ran at least one tool. One isolated tool call
#: between two conversational turns is not the "broad file reads / multi-file
#: search / long tool-output loop" shape a subagent replaces; a run is.
MIN_TOOL_STRETCH_TURNS = 2


def introduced_tokens(turn: TurnComposition) -> int:
    """Material this turn ADDS to the conversation, and that every later turn
    in the same context window therefore re-reads: uncached input (a tool
    result, a pasted file, a user message) plus the assistant's own output.
    Cache reads are excluded — those are the re-reading, not the material."""
    return turn.new_input_tokens + turn.output_tokens


def main_thread_turns(session_turns: list[TurnComposition]) -> list[TurnComposition]:
    """The session's own turns, excluding anything a Task subagent ran."""
    return [t for t in session_turns if not t.sub_agent_id]


def session_context_tokens(main_turns: list[TurnComposition]) -> int:
    """Peak accumulated main-thread prompt size — the session's context high
    water mark, which is what decides whether an offload lever exists at all."""
    return max(
        (t.new_input_tokens + t.reread_tokens for t in main_turns), default=0,
    )


def tail_lengths(main_turns: list[TurnComposition]) -> list[int]:
    """Per turn, how many LATER main-thread turns re-read its material.

    The count stops at the next compaction boundary — a turn whose prompt
    collapses to at most :data:`COMPACTION_PROMPT_DROP_RATIO` of the previous
    one's, which is what a ``/compact`` or a context reset looks like in the
    data. Counting past that boundary would claim a cost the user never paid.
    Turns must already be ordered by ``start_time``.
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
    return [max(boundary_after[i] - i - 1, 0) for i in range(n)]


def resend_tail_tokens_per_turn(main_turns: list[TurnComposition]) -> list[int]:
    """Per-turn tail-token contribution, ungrouped so each turn's own
    contribution can be priced at that turn's own model's rate rather than one
    rate applied to the session-wide sum.

    Returns, per turn, ``introduced x tail_length``, i.e. tokens billed as
    cache reads purely because the work stayed on the parent thread.
    """
    lengths = tail_lengths(main_turns)
    return [
        introduced_tokens(turn) * lengths[i]
        for i, turn in enumerate(main_turns)
    ]


def tool_driven_stretch_mask(
    main_turns: list[TurnComposition],
    *,
    min_run: int = MIN_TOOL_STRETCH_TURNS,
) -> list[bool]:
    """Which main-thread turns sit inside a tool-driven stretch.

    Same contiguous-run idiom ``model_downgrade._cheap_segments`` uses for its
    cheap stretches, applied to the opposite shape: a maximal run of turns that
    each ran at least one tool. That run is the concrete "exploration/edit work
    done inline" a subagent would otherwise have carried, which is why only
    those turns' material is claimed as routable.
    """
    mask = [False] * len(main_turns)
    run: list[int] = []

    def _flush() -> None:
        if len(run) >= min_run:
            for i in run:
                mask[i] = True

    for i, turn in enumerate(main_turns):
        if turn.tool_fanout >= 1:
            run.append(i)
        else:
            _flush()
            run = []
    _flush()
    return mask


def dominant_main_thread_model(
    main_turns: list[TurnComposition],
) -> tuple[str | None, str] | None:
    """The ``(provider, model)`` carrying the most quota-weighted tokens on the
    main thread — the session's DRIVER, not merely its most frequent model.

    Quota-weighted rather than turn-counted so a session whose expensive model
    did the heavy turns is labelled by that model even if a cheaper one answered
    more short turns.
    """
    quota: dict[tuple[str | None, str], float] = {}
    for turn in main_turns:
        key = (turn.provider, turn.model)
        quota[key] = quota.get(key, 0.0) + turn.quota_weighted_tokens
    if not quota:
        return None
    return max(quota.items(), key=lambda kv: kv[1])[0]


def premium_driver_role(
    session_turns: list[TurnComposition],
    *,
    tool_fanout_floor: int = DRIVER_TOOL_FANOUT_FLOOR,
    min_context_tokens: int = MIN_SESSION_CONTEXT_TOKENS,
) -> tuple[str | None, str] | None:
    """``(provider, model)`` when this session is a premium model acting as the
    DRIVER of undelegated work, else ``None``.

    **This is the partition key.** ``model_downgrade`` claims the sessions this
    returns non-``None`` for; ``context_resend`` skips exactly those sessions
    when computing its own recoverable figure. Both call this function, so the
    two claims are disjoint by construction rather than by two thresholds
    happening to agree (Critical Rule 27).

    All four conditions are required, and each rules out a session that looks
    similar but has no driver-role lever:

    1. **at least two main-thread turns** — a single-turn session has no tail;
    2. **never delegates** (no ``sub_agent_id`` row, no turn with
       ``delegates``) — a session that already dispatches Task subagents is not
       making this mistake, and its subagent spans belong to the ``subagent``
       analyzer;
    3. **the main-thread driver is premium-tier** — the whole point is the
       price of the model doing the inline work, so a Sonnet- or Haiku-driven
       session is not this finding;
    4. **context-heavy and tool-heavy** — accumulated main-thread context above
       ``min_context_tokens`` (or there is nothing to re-read) and at least
       ``tool_fanout_floor`` main-thread tool calls (or there was no routable
       work to begin with).

    Requires turns loaded with ``with_tool_activity=True``: ``tool_fanout`` and
    ``delegates`` are inert defaults otherwise, which would silently flag every
    premium session.
    """
    main_turns = main_thread_turns(session_turns)
    if len(main_turns) < 2:
        return None
    if any(t.sub_agent_id for t in session_turns):
        return None
    if any(t.delegates for t in main_turns):
        return None
    driver = dominant_main_thread_model(main_turns)
    if driver is None or not is_premium_tier(driver[1]):
        return None
    if session_context_tokens(main_turns) < min_context_tokens:
        return None
    if sum(t.tool_fanout for t in main_turns) < tool_fanout_floor:
        return None
    return driver
