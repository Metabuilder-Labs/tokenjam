"""TokenJam Summarize — structure-aware prompt summarization.

Detects prompt files worth summarizing and estimates the per-call token saving;
the mechanism (protect → summarize → restore → verify) lives in `wrap`/`session`,
and apply/undo/backup in `apply`/`backup`. Only the advisory scan API is
re-exported here — the CLI and MCP import the rest from the submodules directly.
Pure `core` logic — no CLI or HTTP imports.
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
