"""Shared lock-free sink-location helper.

Originally home to the `tj hook cap-output` output-trim feature, which was
measured negative and removed (see CLAUDE.md's `output_cap` note and
`.specs/2026-07-02-tj-output-trim-hook-{design,AB-result}.md`). All that
survives is `hooks_dir`: consumers that need to write an append-only JSONL
sink without taking a DuckDB lock (the `tj serve` daemon may hold the write
lock) derive their sink path from it — e.g. `core/recommendations.py`'s
outcome ledger. The path derives from `config.storage.path`'s parent
(mirroring `summarize.session.summary_root`) so `--db` / `TJ_CONFIG`
overrides isolate it too.
"""
from __future__ import annotations

from pathlib import Path


def hooks_dir(config) -> Path:
    """`<storage-parent>/hooks/` — sink dir, next to the DB, honoring overrides."""
    sp = getattr(getattr(config, "storage", None), "path", "") or ""
    base = Path.home() / ".tj" if sp in ("", ":memory:") else Path(sp).expanduser().parent
    return base / "hooks"
