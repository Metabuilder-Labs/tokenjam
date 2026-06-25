"""Sampling-confidence helpers for savings estimates (#308).

A savings projection drawn from 5 sessions is far less reliable than one drawn
from 500, yet they render identically. These helpers attach a *sampling*
confidence interval to a projection so the figure carries its own reliability
signal.

IMPORTANT — this is SAMPLING confidence on the projection (how much usage the
estimate rests on), NOT a claim that the optimization preserves quality. The
existing "estimated recoverable / review before switching" caveats stay; this
only quantifies the spread you'd expect if you'd sampled a different slice of
the same usage. The CI label must always read as sampling confidence.

Pure Python only — no scipy / numpy (CLAUDE.md: no heavy deps).
"""
from __future__ import annotations

import math
import random

# Default bootstrap parameters. Seed is FIXED so the same window produces the
# same interval every run (a report shouldn't jitter between invocations).
DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95
_BOOTSTRAP_SEED = 1308  # arbitrary, stable


def bootstrap_ci(
    per_session: list[float],
    *,
    scale: float = 1.0,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
) -> tuple[float, float] | None:
    """Bootstrap CI for the SUM of a per-session quantity, optionally scaled.

    ``per_session`` holds one savings value per sampled session. The point
    estimate is ``sum(per_session) * scale`` (e.g. window savings projected to a
    month via ``scale = 30 / window_days``). We resample the sessions with
    replacement, recompute the scaled sum each time, and take the central
    ``confidence`` quantiles — capturing how much the total would swing if the
    same number of sessions had landed differently.

    Returns ``(ci_low, ci_high)`` or ``None`` when there isn't enough data to say
    anything (0 or 1 session — a single point has no spread to estimate).
    """
    n = len(per_session)
    if n < 2:
        return None

    rng = random.Random(_BOOTSTRAP_SEED)
    totals: list[float] = []
    for _ in range(resamples):
        s = 0.0
        for _ in range(n):
            s += per_session[rng.randrange(n)]
        totals.append(s * scale)
    totals.sort()

    alpha = (1.0 - confidence) / 2.0
    low = _quantile(totals, alpha)
    high = _quantile(totals, 1.0 - alpha)
    # Savings can't go negative — clamp the lower bound at 0 (you never *gain*
    # money by routing to a cheaper model; the floor is "no savings").
    return (max(0.0, low), max(0.0, high))


def _quantile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac
