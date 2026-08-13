"""A cheap, cannot-lie signal for "has new telemetry landed since X".

THE PROBLEM THIS EXISTS FOR. The daemon's scheduled analyzer pass
(`scan_cycle.py`) used to be a pure wall-clock `interval` job — it re-ran the
full analyzer sweep, including the relearn distill pass (full-corpus, several
LLM subprocess calls), every `scan_interval_hours` regardless of whether a
single new span had been ingested. On an idle machine that recomputes an
answer that cannot have changed.

THE SIGNAL. A single process-local counter, bumped at the two DuckDB write
paths that both the live ingest pipeline and every backfill/catch-up pass
funnel through (`DuckDBBackend.insert_span`, `.bulk_insert_spans`) — the
lowest choke point shared by every ingestion source, so nothing needs its own
bump call. `scan_cycle` reads it once per scheduled tick and compares against
the value it saw at the start of the last pass it actually ran; a delta below
`[optimize] scan_watermark_min_new_spans` means the tick is a no-op.

DELIBERATELY IN-MEMORY, NOT PERSISTED. A daemon restart resets it to zero,
which is fine: `scan_cycle`'s own "never run in this process" state is
`None` at that point too, so the very next scheduled tick (and the startup
kick) runs the pass unconditionally rather than trusting a watermark it has
no history for. The cost of losing the counter across a restart is at most
one extra pass — a correctness-safe direction to be wrong in, and cheaper
than a durable counter that could itself be stale, corrupt, or read before a
migration lands it.

NOT A DEDUPE OR ACCOUNTING SIGNAL. This is a coarse "did anything happen"
flag, not a count anyone should compute past overspend from — no query, no
config, no analyzer may ever import this for anything other than the gate
above.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_count = 0


def bump(n: int = 1) -> None:
    """Record that *n* new spans were just written to the store.

    Called from `DuckDBBackend.insert_span` (the live path) and
    `.bulk_insert_spans` (backfill/catch-up) — the two, and only two, places a
    span actually lands in DuckDB. `n <= 0` is a no-op so a caller need not
    special-case an empty batch.
    """
    global _count
    if n <= 0:
        return
    with _lock:
        _count += n


def current() -> int:
    """Monotonically increasing count of spans written since process start."""
    with _lock:
        return _count
