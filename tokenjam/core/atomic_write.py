"""Atomic file write — the shared write primitive for every apply rail.

``atomic_write`` writes text to an existing file via a temp file in the same
dir → ``os.replace`` (so a reader never sees a partial write, and a crash
mid-write leaves the original untouched), preserving the file's mode. It
refuses a symlinked target outright — every caller gets that guard for free,
including one that forgets its own pre-check.

This is deliberately feature-agnostic: ``core.summarize.apply`` (the prompt
summarizer's apply/undo path) and ``core.optimize.relearn_apply`` (the
self-improve loop's apply/revert path) both reuse it. Neither owns it; each
maps ``AtomicWriteRefused`` to its own domain exception at the call site.
"""
from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


class AtomicWriteRefused(Exception):
    """Refusing an atomic write — the target is a symlink, so writing through it
    could land outside where the caller expects. Callers map this to their own
    domain exception (e.g. summarize's ``SummarizeRefused``, relearn's
    ``RelearnApplyRefused``)."""


def atomic_write(p: Path, text: str) -> None:
    """Write ``text`` to ``p`` atomically (temp in the same dir → ``os.replace``), preserving mode.

    Raises ``AtomicWriteRefused`` if ``p`` is a symlink.
    """
    if p.is_symlink():
        raise AtomicWriteRefused(f"{p} is a symlink — refusing to write through it.")
    mode = stat.S_IMODE(p.stat().st_mode)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tj-tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
