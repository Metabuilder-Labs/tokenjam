"""Per-candidate savings estimate for the summarize surface.

Honesty discipline: this estimates the **per-call token reduction** of
summarizing a static prompt's prose. It needs no telemetry — tokens are derived
from the file. The saving amortizes across every reuse of the (cached) prompt.

Dollar figures are deliberately NOT fabricated here: a per-call dollar amount at
default rates is noise, and the meaningful *amortized* figure needs a real
call-count (telemetry), which the optional usage-ranked path can add later. The
token figure is the one we can stand behind from the file alone.
"""
from __future__ import annotations

from tokenjam.core.summarize.detect import CHARS_PER_TOKEN, StructureBreakdown

# Summarize prose to ~half its words by default; only prose shrinks.
DEFAULT_TARGET_RATIO = 0.5


def tokens_saved(breakdown: StructureBreakdown, ratio: float = DEFAULT_TARGET_RATIO) -> int:
    """Estimated tokens removed per call by summarizing the prose to ``ratio``.

    Protected structure (fenced code/JSON, tables, tags, inline code, templates) is preserved
    verbatim and never counted as savings — only prose shrinks. Returns 0 when ``ratio >= 1``.
    """
    if ratio >= 1.0:
        return 0
    prose_tokens = breakdown.prose_chars / CHARS_PER_TOKEN
    return max(0, int(prose_tokens * (1.0 - ratio)))
