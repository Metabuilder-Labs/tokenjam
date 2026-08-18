"""Shared finding ranking for CLI text view and the /optimize payload.

Lives in core so the API does not import `tokenjam.cli`. The name set is the
card-bearing findings (not the full analyzer registry — budget-projection has
no ranked card).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tokenjam.core.optimize.types import OptimizeReport

# Insertion order matches `cmd_optimize._FINDING_RENDERERS`. A drift test
# pins the two together so a new card cannot rank in one surface only.
CARD_FINDING_NAMES: tuple[str, ...] = (
    "cache",
    "cache-recommend",
    "resend",
    "script",
    "reuse",
    "trim",
    "subagent",
    "relearn",
    "verbosity",
    "deadweight",
    "placement",
    "summarize",
    "stream-usage",
)

# Findings that must never collapse into the "Minor findings" pointer by
# token share. `relearn` is a cluster finding; its token figure is not a
# real fraction of the window.
ALWAYS_FULL_FINDINGS: frozenset[str] = frozenset({"relearn"})


def reclaimable_share(finding: Any, window_total_tokens: int) -> float | None:
    """Estimated-recoverable-tokens share of the window, for ranking.

    Returns ``None`` — not 0.0 — when the finding has no quantified estimate
    at all. Those findings still render in full (unranked), they are not
    de-minimis.
    """
    tokens = getattr(finding, "past_overspend_tokens", None)
    if tokens is None or window_total_tokens <= 0:
        return None
    return max(float(tokens), 0.0) / window_total_tokens


def rank_findings(
    report: OptimizeReport,
    requested: list[str] | None,
    *,
    known_names: Mapping[str, Any] | Sequence[str] = CARD_FINDING_NAMES,
    always_full: Sequence[str] | set[str] | frozenset[str] = ALWAYS_FULL_FINDINGS,
) -> list[tuple[str, float | None]]:
    """Rank findings with something to show by reclaimable token share.

    Largest first; unranked findings (no quantified estimate) sort last.
    Ties fall back to ``known_names`` insertion order, with ``downsize``
    first — matching the CLI renderer table.
    """
    window_tokens = report.window.total_tokens
    names = tuple(known_names)
    known = set(names)
    full = set(always_full)
    order = ["downsize", *names]
    order_index = {name: i for i, name in enumerate(order)}

    items: list[tuple[str, float | None]] = []
    # Render an explicit "no candidates" empty state when the downsize
    # analyzer ran but found nothing. Skip the section entirely when the
    # user asked for a different positional subset (`tj optimize cache`
    # should not mention downsize). `tj optimize placement` resolves to
    # running the downsize analyzer, so `report.downgrade` is populated
    # even though the user never typed "downsize" — without this guard
    # its card would leak into that report.
    downsize_was_requested = (not requested) or ("downsize" in requested)
    if downsize_was_requested:
        if report.downgrade is not None:
            items.append(("downsize", reclaimable_share(report.downgrade, window_tokens)))
        else:
            items.append(("downsize", None))

    for name, finding in (report.findings or {}).items():
        if name not in known:
            continue
        share = (
            None if name in full
            else reclaimable_share(finding, window_tokens)
        )
        items.append((name, share))

    items.sort(key=lambda item: (
        item[1] is None,
        -(item[1] or 0.0),
        order_index.get(item[0], len(order)),
    ))
    return items
