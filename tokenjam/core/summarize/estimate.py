"""Per-candidate savings estimate for the summarize surface.

Honesty discipline: this estimates the **per-call token reduction** of
summarizing a static prompt's prose. It needs no telemetry — tokens are derived
from the file. The saving amortizes across every reuse of the (cached) prompt.

Dollar figures are deliberately NOT fabricated here: a per-call dollar amount at
default rates is noise, and the meaningful *amortized* figure needs a real
call-count (telemetry), which the optional usage-ranked path can add later. The
token figure is the one we can stand behind from the file alone.

**The ratio is an ASK, not a prediction — and the two must not be confused.**
`DEFAULT_TARGET_RATIO` is what `session.prepare` puts in the rewriter's prompt
("summarize this to N words"). There is no retry loop and no gate on hitting it:
whatever the model returns is accepted so long as the structure markers survive,
and rewrites observed while building this delivered materially less than the
target asks for. Using the ask as the estimate therefore overstates, so this
module separates the two:

* :data:`DEFAULT_TARGET_RATIO` — what the rewriter is asked for.
* :func:`observed_prose_ratio` — what rewrites on THIS machine have actually
  delivered, derived from staged results (`session.CheckVerdict`), or ``None``
  when there is no credible sample.

A caller that wants a defensible number asks for the observed ratio first and
falls back to the target only while saying so (see the summarize analyzer's
`estimate_basis`). Nothing here invents a middle number: a ratio that is neither
the documented ask nor a measured outcome would be the least defensible of the
three.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tokenjam.core.summarize.detect import CHARS_PER_TOKEN, StructureBreakdown

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig

logger = logging.getLogger(__name__)

#: What the rewriter is INSTRUCTED to hit — `session.prepare` turns this into
#: the word count in the model's prompt. It is an ask, not a prediction, and
#: **nothing enforces it**: there is no retry and no gate on the achieved word
#: count, since `check` gates on structure surviving and nothing else. So do not
#: tune this as though it were a measurement of what rewrites deliver — changing
#: it changes what the model is ASKED for. The measured counterpart is
#: :func:`observed_prose_ratio`, which supersedes this the moment real staged
#: results exist.
DEFAULT_TARGET_RATIO = 0.5

#: Minimum evidence before an observed ratio may replace the target. Both gates
#: must pass: too few files, or too little prose, and one unusual rewrite would
#: set the number for every file the user owns.
MIN_OBSERVED_SAMPLES = 3
MIN_OBSERVED_PROSE_WORDS = 500


def tokens_saved(breakdown: StructureBreakdown, ratio: float = DEFAULT_TARGET_RATIO) -> int:
    """Estimated tokens removed per call by summarizing the prose to ``ratio``.

    Protected structure (fenced code/JSON, tables, tags, inline code, templates) is preserved
    verbatim and never counted as savings — only prose shrinks. Returns 0 when ``ratio >= 1``.
    """
    if ratio >= 1.0:
        return 0
    prose_tokens = breakdown.prose_chars / CHARS_PER_TOKEN
    return max(0, int(prose_tokens * (1.0 - ratio)))


def observed_prose_ratio(config: "TjConfig | None") -> tuple[float | None, int]:
    """The prose ratio rewrites have ACTUALLY delivered here, and the sample size.

    Returns ``(None, n)`` when the sample is too small to be credible — the
    caller must then fall back to the target ratio and disclose that it is an
    ask rather than a measurement. Never guesses: no staged results means no
    observed ratio, not a zero and not a nudged constant.

    Derivation. `session.check` restores the rewrite with its structure put back
    verbatim, so every word the file lost was a prose word: ``words_before -
    words_after`` is prose removed, and ``prose_words_before`` is what it started
    with. Summing both across the sample before dividing weights the ratio by
    volume, so a handful of tiny files cannot outvote a large one. Only staged
    (structure-passing) results count — a rewrite that failed the structure gate
    was never a usable outcome, and letting it into the sample would measure the
    rewriter's failures as if the user had accepted them.

    Never raises: an unreadable or half-written staged result is skipped.
    """
    if config is None:
        return None, 0
    from tokenjam.core.summarize.session import list_staged

    try:
        staged = list_staged(config)
    except Exception:
        logger.debug("summarize estimate: could not read staged results", exc_info=True)
        return None, 0

    samples = 0
    prose_before_total = 0
    removed_total = 0
    for record in staged:
        if not isinstance(record, dict) or not record.get("staged"):
            continue
        prose_before = int(record.get("prose_words_before") or 0)
        before = int(record.get("words_before") or 0)
        after = int(record.get("words_after") or 0)
        # A result staged before `prose_words_before` existed carries no
        # denominator; it is not evidence, so it is skipped rather than
        # counted as a zero-prose file.
        if prose_before <= 0 or before <= 0:
            continue
        samples += 1
        prose_before_total += prose_before
        removed_total += max(0, before - after)

    if samples < MIN_OBSERVED_SAMPLES or prose_before_total < MIN_OBSERVED_PROSE_WORDS:
        return None, samples
    # Clamped to [0, 1]: a rewrite that somehow grew the file is a 1.0 (no
    # reduction), never a negative saving.
    ratio = 1.0 - (removed_total / prose_before_total)
    return min(1.0, max(0.0, ratio)), samples
