"""The ONE span the user chose, and the retention that has to back it.

There were two numbers here and they were allowed to disagree. ``retention_days``
decided how much history the store KEPT; a separate constant decided how much
history the analyzers LOOKED AT; and nothing tied either to what the product
told the user it was analyzing. On a real store the two drifted apart by weeks —
the oldest data the analyzers were sizing their window against had already been
deleted underneath them — and the only way to notice was to measure the store
twice, days apart, and diff the answers.

So there is one choice, made once, at onboarding: **how far back should tj
analyze?** Everything else is derived from it.

  * ``analysis_span`` — the choice, as the user made it: ``"30d"``, ``"90d"``,
    or ``"all"``.
  * retention is DERIVED from it, and ``"all"`` disables retention outright.
    Deletion can therefore never remove history the product is still offering to
    analyze, because the thing that decides what to delete is downstream of the
    thing that decides what to promise.
  * a config that sets ``retention_days`` and nothing else — every config
    written before this existed — has its value read AS the span. That is the
    honest reading of it: what such a user kept is the most the product could
    ever have analyzed, so nothing about their setup changes.

The invariant is one-directional and enforced here rather than documented:
**retention is never shorter than the span.** Lowering ``retention_days`` under
the span does not quietly shrink the span — a promise already made is not
retracted by a storage setting — it raises retention back to the span and says
so. The reverse (retention longer than the span) is fine and needs no
correction: keeping more than you analyze costs disk and misleads nobody.

Not to be confused with ``core/data_span.py``, which measures what the store
ACTUALLY holds. This module is what the user ASKED FOR. The two meet in
``resolved_window_days``, since a 90-day promise over a store that has been
running a week is answerable only for that week.
"""
from __future__ import annotations

from typing import Any

#: The choices offered at onboarding. Deliberately three: the two rungs a user
#: can reason about, plus the opt-out that turns deletion off entirely.
ANALYSIS_SPAN_CHOICES: tuple[str, ...] = ("30d", "90d", "all")

#: What a config with no opinion gets. The wider of the two bounded rungs: the
#: analyzers that matter most here (recurring-failure and re-read detection)
#: measure recurrence, and a month of history is thin evidence of a habit.
DEFAULT_ANALYSIS_SPAN = "90d"

#: The value ``analysis_span`` takes to mean "keep and analyze everything".
UNBOUNDED_SPAN = "all"

#: LAST RESORT ONLY — the lookback to use when neither bound can be established
#: (an unreadable store AND an unbounded choice). Not a default in the sense of
#: "what a normal run uses": ``window_days_for`` is that. A window has to be a
#: number of days to subtract from ``utcnow()``, so some answer is required even
#: when there is no honest one, and this is deliberately the smallest claim.
FALLBACK_WINDOW_DAYS = 30


def parse_analysis_span(raw: Any) -> int | None:
    """``"30d"``/``"90d"``/``"all"`` (or a bare day count) to days, ``None`` for
    unbounded. Raises ``ValueError`` on anything else — a span nobody can parse
    must not silently become a default, because the default is a different
    promise from the one written down."""
    if raw is None:
        raise ValueError("analysis span is unset")
    text = str(raw).strip().lower()
    if text in (UNBOUNDED_SPAN, "all-available", "unbounded"):
        return None
    if text.endswith("d"):
        text = text[:-1]
    try:
        days = int(text)
    except ValueError:
        raise ValueError(
            f"unrecognised analysis span {raw!r} — expected one of "
            f"{', '.join(ANALYSIS_SPAN_CHOICES)}"
        ) from None
    if days <= 0:
        raise ValueError(f"analysis span must be positive, got {raw!r}")
    return days


def analysis_span_days(storage: Any) -> int | None:
    """How far back the product may claim to analyze. ``None`` = all available.

    Reads the explicit choice when there is one. When there is not, an explicit
    ``retention_days`` IS the choice (see the module docstring) — that is the
    forward-migration path for every config written before the coupling existed,
    and it changes nothing about what such a user already had.
    """
    chosen = getattr(storage, "analysis_span", None)
    if chosen is not None:
        return parse_analysis_span(chosen)
    kept = getattr(storage, "retention_days", None)
    if kept is not None:
        return int(kept)
    return parse_analysis_span(DEFAULT_ANALYSIS_SPAN)


def retention_days_for(storage: Any) -> int | None:
    """The retention the span requires. ``None`` = deletion disabled.

    The clamp is the whole point: an explicit ``retention_days`` shorter than
    the span is raised to the span rather than honoured, because honouring it
    would delete the history a claim the product is already making depends on.
    """
    span = analysis_span_days(storage)
    if span is None:
        return None
    explicit = getattr(storage, "retention_days", None)
    if explicit is None:
        return span
    return max(int(explicit), span)


def retention_was_raised_to_span(storage: Any) -> bool:
    """True when an explicit ``retention_days`` was overridden by the clamp.

    Callers that can talk to the user (onboarding, ``tj doctor``) say so; the
    retention job itself only needs the number.
    """
    span = analysis_span_days(storage)
    explicit = getattr(storage, "retention_days", None)
    if explicit is None:
        return False
    if span is None:
        return True
    return int(explicit) < span


def resolved_window_days(storage: Any, available_days: int | None) -> int | None:
    """The span an analyzer may actually accumulate over.

    The chosen span bounded by what the store holds. ``available_days`` is
    ``core/data_span.available_data_span``'s measure, and ``None`` there means
    "unknown" rather than zero — an unknown span cannot narrow anything, so the
    choice stands unchanged (``core/data_span`` makes the same distinction for
    the same reason).

    Returns ``None`` only when the span is unbounded AND the store cannot say
    how much history it has, i.e. genuinely "everything, extent unknown".
    """
    span = analysis_span_days(storage)
    if available_days is None:
        return span
    if span is None:
        return available_days
    return min(span, available_days)


def span_label(storage: Any) -> str:
    """How the CHOSEN span is spelled on a surface. Never a bare number."""
    span = analysis_span_days(storage)
    return "all available" if span is None else f"{span}d"


def days_of_history(conn: Any) -> int | None:
    """How far back from TODAY the store's oldest dated row sits.

    Measured from now rather than from the newest row, because a window is
    subtracted from ``utcnow()``: a store whose last activity was two days ago
    still needs a lookback reaching past those idle days to see anything at all.

    Reads ``DataSpan.oldest`` rather than ``available_days`` — see that field's
    note in ``core/data_span.py`` for why the recent unbroken run is the wrong
    bound for a lookback.
    """
    from datetime import date

    from tokenjam.core.data_span import available_data_span
    from tokenjam.utils.time_parse import utcnow

    if conn is None:
        return None
    oldest = available_data_span(conn).oldest
    if not oldest:
        return None
    return (utcnow().date() - date.fromisoformat(oldest)).days + 1


def window_days_for(
    storage: Any, conn: Any, *, fallback: int = FALLBACK_WINDOW_DAYS,
) -> int:
    """The lookback in days that the chosen span resolves to on THIS store.

    One seam, so every surface that needs a number of days — the cost-proposal
    recompute, relearn's precomputed window vocabulary — derives the same one
    and cannot publish two windows for one span.

    ``fallback`` is reached only when neither bound can be established: an
    unreadable store AND an unbounded choice, i.e. genuinely "everything, extent
    unknown". A window has to be a number to subtract from ``utcnow()``, so
    there is no third answer available here.
    """
    if storage is None:
        return fallback
    days = resolved_window_days(storage, days_of_history(conn))
    return max(1, int(days)) if days else fallback


def window_label_for(
    storage: Any, conn: Any, *, fallback: int = FALLBACK_WINDOW_DAYS,
) -> str:
    """``window_days_for`` as the label surfaces publish it."""
    return f"{window_days_for(storage, conn, fallback=fallback)}d"
