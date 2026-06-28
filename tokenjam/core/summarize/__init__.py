"""TokenJam Summarize — structure-aware prompt summarization.

v1 (PR1) ships the **advisory** surface only: detect prompt files worth
summarizing and estimate the per-call token saving, without writing anything.
The mechanism (protect → summarize → restore → verify) and apply/backup land in
later PRs. Pure `core` logic — no CLI or HTTP imports.
"""
from tokenjam.core.summarize.detect import (
    MIN_PROSE_WORDS,
    StructureBreakdown,
    analyze,
    is_candidate,
)
from tokenjam.core.summarize.estimate import DEFAULT_TARGET_RATIO, tokens_saved
from tokenjam.core.summarize.candidates import Candidate, ScanResult, list_candidates

__all__ = [
    "MIN_PROSE_WORDS",
    "StructureBreakdown",
    "analyze",
    "is_candidate",
    "DEFAULT_TARGET_RATIO",
    "tokens_saved",
    "Candidate",
    "ScanResult",
    "list_candidates",
]
