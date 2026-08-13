"""What gets stored about a reused system prefix, and what reads it back.

The prefix (a project's ``CLAUDE.md``, resent verbatim on every API call) is
what ``cache-recommend`` looks for. It used to be stored on the span as the
whole file, which is the most expensive possible way to hold it: the text is
identical across every span of a project, so the cost is (size x span count).
Measured before the fix — 92,514 spans holding ~43 KB each, **4.06 GB** on disk
to carry **1.84 MB** of distinct text.

The analyzer never needed the text. It needs exactly three answers, and all
three are derivable at capture time:

===================  =========================================================
``hash``             identity — which prefix this is, for grouping and counting
``sample``           the first :data:`SAMPLE_CHARS`, for display only
``length``           the character count, for the analyzer's own size gate
===================  =========================================================

Storing those three instead of the text takes ~230 bytes per span rather than
~43,000, and none of the three loses information the analyzer was using.

**This module is the single source of both halves.** The producer
(``core.backfill``) and the consumer (``cache_recommend``) previously agreed
only by coincidence — the analyzer hashed ``text[:PREFIX_HASH_BYTES]`` while
the parser stored whatever it read off disk. Now both call
:func:`prefix_hash`, so a change to the window cannot silently reclassify old
spans into new groups. ``test_hash_window_is_single_sourced`` fails if the
analyzer stops routing through here.
"""
from __future__ import annotations

import hashlib
from typing import Any

# The window that DEFINES prefix identity. Two prompts sharing their first
# HASH_CHARS characters are one cacheable prefix as far as this product is
# concerned — the number is a claim about how much of a prefix has to match to
# be worth a cache breakpoint, not an implementation detail.
HASH_CHARS = 2000

# Kept for display only: enough to recognise which prefix a candidate is,
# never enough to reconstruct the file.
SAMPLE_CHARS = 120

# Below this a prompt is too small for caching to pay for itself, so the
# analyzer skips it. Stored length is compared against this.
MIN_CACHEABLE_CHARS = 200


def prefix_hash(text: str) -> str:
    """Stable identity for a reused prefix: SHA-256 of its first HASH_CHARS."""
    head = text[:HASH_CHARS].encode("utf-8", errors="replace")
    return hashlib.sha256(head).hexdigest()[:16]


def summarize(text: str) -> dict[str, Any]:
    """The compact form stored on a span, or ``{}`` when there is nothing to store.

    Returns bare values rather than attribute keys so the caller decides
    namespacing; see ``TjAttributes.SYSTEM_PREFIX_*``.
    """
    if not text:
        return {}
    return {
        "hash": prefix_hash(text),
        "sample": text[:SAMPLE_CHARS],
        "length": len(text),
    }
