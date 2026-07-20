"""Shared clustering primitives for the pattern-mining analyzers.

Three analyzers mine recurring cross-session patterns and shared the same
control flow, each with its own hand-rolled copy:

  * ``analyzers.pothole``            — clusters failure signatures.
  * ``analyzers.plan_reuse``         — clusters planning skeletons.
  * ``analyzers.workflow_restructure`` — clusters (tool, arg-shape) sequences.

The genuinely shared mechanism is small and lives here: the setdefault-append
grouping, the recurrence-threshold filter, and (for the two text-normalizing
miners) the variable-masking apply loop. What stays analyzer-specific is the
domain logic those primitives are pointed at — WHICH regexes a normalizer masks,
HOW a domain object is built from a group, WHAT counts as one recurrence. Those
differ on purpose and are not merged here; only the plumbing is.

This module has no tokenjam imports — it's pure, dependency-free, and easy to
test in isolation.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Hashable, Iterable, Pattern


def group_by_key(
    items: Iterable[Any], key_fn: Callable[[Any], Hashable],
) -> dict[Any, list[Any]]:
    """Group ``items`` into ``{key: [items...]}`` in first-seen order.

    The setdefault-append every miner used to write inline. Insertion order is
    preserved (Python dicts), so a caller that reads the first member of each
    group gets the same "first item to land in this bucket" it did before.
    """
    buckets: dict[Any, list[Any]] = {}
    for item in items:
        buckets.setdefault(key_fn(item), []).append(item)
    return buckets


def recurring(
    buckets: dict[Any, Any], *, min_members: int,
    size_fn: Callable[[Any], int] = len,
) -> dict[Any, Any]:
    """The sub-mapping of ``buckets`` whose ``size_fn(bucket) >= min_members``.

    Keys are preserved (each miner still needs its group key to build the
    surfaced finding), and insertion order is kept so downstream ranking is
    unchanged. ``size_fn`` defaults to ``len`` (a bucket is a member list); pass
    a custom one when the recurrence measure isn't the raw count — e.g. the
    pothole lane counts DISTINCT sessions, not raw occurrences.
    """
    return {
        key: bucket for key, bucket in buckets.items()
        if size_fn(bucket) >= min_members
    }


_WS_RE = re.compile(r"\s+")


def mask_variables(
    text: str,
    substitutions: Iterable[tuple[Pattern[str], str]],
    *,
    collapse_ws: bool = False,
    lowercase: bool = False,
) -> str:
    """Apply an ordered list of ``(compiled_regex, replacement)`` substitutions,
    then optionally collapse whitespace and lowercase.

    The two text-normalizing miners share THIS apply loop; each supplies its own
    ordered substitution list. The patterns are deliberately analyzer-specific —
    a failure-signature normalizer masks timestamps/uuids/hex-ids that a
    prompt-prefix hasher doesn't, and they use different placeholder tokens — so
    only the loop is shared, never the pattern set.
    """
    for pattern, repl in substitutions:
        text = pattern.sub(repl, text)
    if collapse_ws:
        text = _WS_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text
