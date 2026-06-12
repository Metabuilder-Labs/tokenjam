"""Stable, privacy-safe signatures derived from span content.

Used to detect *identical* repeated tool calls (a genuine retry loop) without
retaining the raw, possibly-sensitive tool input: we keep only a one-way hash.
Pure module — no internal imports.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def tool_arg_signature(raw: Any) -> str | None:
    """Short hash of a tool call's arguments, or ``None`` when there are none.

    ``raw`` is the ``gen_ai.tool.input`` value — a JSON string or an object.
    Returns a 16-char hex digest (stable across runs for identical input), or
    ``None`` for missing/empty input so callers can tell "no argument data" apart
    from a real signature (telemetry without tool args never trips retry-loop).
    """
    if raw is None:
        return None
    text = raw if isinstance(raw, str) else json.dumps(raw, sort_keys=True)
    text = text.strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


__all__ = ["tool_arg_signature"]
