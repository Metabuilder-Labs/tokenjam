"""
Cache-recommend analyzer (Anthropic-only in v1).

Walks captured Anthropic prompts in the window, identifies stable
prefix substrings shared across many calls, and recommends placing
`cache_control` breakpoints to convert those prefixes into cache reads.

Requires `[capture] prompts = true` in the config. Without captured
content the analyzer can't see the actual prompt text and exits early
with a clear message.

Per-provider scope: Anthropic only. Other providers' caching mechanics
(or lack thereof) differ enough that a v1 recommendation engine would
mislead. Multi-provider support is a future research project.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.otel.semconv import GenAIAttributes

# The first N characters of the prompt are hashed to identify a "prefix
# signature." Long enough to discriminate, short enough to avoid hashing
# every call's full payload.
PREFIX_HASH_BYTES = 2000

# Minimum occurrences of the same prefix before it's worth recommending
# a cache_control placement. Three calls share a prefix => likely a stable
# template; below that, it could be coincidence.
MIN_PREFIX_OCCURRENCES = 3


@dataclass
class CachePrefixCandidate:
    """One stable prefix shared across multiple calls."""
    prefix_hash:    str
    sample_chars:   str            # first ~120 chars for preview
    occurrences:    int
    avg_input_tokens: float        # average input_tokens on calls that share this prefix
    estimated_cacheable_tokens: int


@dataclass
class CacheRecommendFinding:
    """Cache-control breakpoint recommendations for Anthropic prompts."""
    enabled:          bool                          # false when capture.prompts disabled
    candidates:       list[CachePrefixCandidate] = field(default_factory=list)
    skipped_provider_count: int = 0                 # non-Anthropic spans we ignored
    confidence:       str = "structural"
    hint:             str | None = None             # surfaced when enabled is false


def _stringify_prompt(value: Any) -> str:
    """
    Turn the prompt-content attribute into a single string. Anthropic spans
    may carry it as a string, a list of message dicts, or a dict — we
    canonicalise so prefix hashing is consistent regardless of shape.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Concatenate role+content per message, separated by null bytes
        # to avoid accidental cross-message hash collisions.
        parts: list[str] = []
        for msg in value:
            if isinstance(msg, dict):
                role = str(msg.get("role", "")).strip()
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Anthropic message-content blocks
                    inner: list[str] = []
                    for block in content:
                        if isinstance(block, dict):
                            inner.append(str(block.get("text", "")))
                        else:
                            inner.append(str(block))
                    content = "".join(inner)
                parts.append(f"{role}\x00{content}")
            else:
                parts.append(str(msg))
        return "\x00".join(parts)
    if isinstance(value, dict):
        return _stringify_prompt(value.get("content"))
    return str(value)


def _prefix_hash(text: str) -> str:
    """SHA-256 of the first PREFIX_HASH_BYTES chars (UTF-8) of the prompt."""
    head = text[:PREFIX_HASH_BYTES].encode("utf-8", errors="replace")
    return hashlib.sha256(head).hexdigest()[:16]


@register("cache-recommend")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a CacheRecommendFinding to ctx.report.findings."""
    capture = getattr(ctx.config, "capture", None)
    if capture is None or not getattr(capture, "prompts", False):
        ctx.report.findings["cache-recommend"] = CacheRecommendFinding(
            enabled=False,
            hint=(
                "Enable `[capture] prompts = true` in tj.toml and let the "
                "daemon collect a window of data before re-running this "
                "analyzer. Without captured prompt content there's no way "
                "to identify stable prefixes."
            ),
        )
        return

    clauses = [
        "start_time >= $1", "start_time < $2",
        "model IS NOT NULL", "provider IS NOT NULL",
    ]
    params: list[Any] = [ctx.since, ctx.until]
    if ctx.agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(ctx.agent_id)
    where = " AND ".join(clauses)
    rows = ctx.conn.execute(
        f"SELECT provider, attributes, input_tokens FROM spans WHERE {where}",
        params,
    ).fetchall()

    # Aggregate by prefix hash. We track only Anthropic spans; everything
    # else gets counted so the renderer can disclose what was skipped.
    prefix_counts: dict[str, dict[str, Any]] = {}
    skipped_provider = 0
    for provider, attrs_raw, input_tokens in rows:
        if str(provider).lower() != "anthropic":
            skipped_provider += 1
            continue
        attrs = attrs_raw
        if isinstance(attrs, str):
            import json as _json
            try:
                attrs = _json.loads(attrs)
            except Exception:
                continue
        if not isinstance(attrs, dict):
            continue
        content = attrs.get(GenAIAttributes.PROMPT_CONTENT)
        if not content:
            continue
        text = _stringify_prompt(content)
        if len(text) < 200:
            # Skip tiny prompts — no caching opportunity.
            continue
        h = _prefix_hash(text)
        entry = prefix_counts.setdefault(h, {
            "sample": text[:120],
            "count": 0,
            "tokens_sum": 0,
        })
        entry["count"] += 1
        entry["tokens_sum"] += int(input_tokens or 0)

    candidates: list[CachePrefixCandidate] = []
    for h, entry in prefix_counts.items():
        if entry["count"] < MIN_PREFIX_OCCURRENCES:
            continue
        avg_in = entry["tokens_sum"] / entry["count"] if entry["count"] else 0.0
        # Rough estimate of cacheable tokens per call. A 2000-char prefix
        # is roughly 500 tokens (4 chars/token heuristic). The renderer
        # can refine via the user's actual tokeniser, but for v1 this
        # gives an order-of-magnitude signal.
        estimated_cacheable = min(int(avg_in * 0.5), PREFIX_HASH_BYTES // 4)
        candidates.append(CachePrefixCandidate(
            prefix_hash=h,
            sample_chars=str(entry["sample"]),
            occurrences=int(entry["count"]),
            avg_input_tokens=round(avg_in, 1),
            estimated_cacheable_tokens=estimated_cacheable,
        ))

    candidates.sort(key=lambda c: c.occurrences * c.estimated_cacheable_tokens,
                    reverse=True)

    ctx.report.findings["cache-recommend"] = CacheRecommendFinding(
        enabled=True,
        candidates=candidates[:10],
        skipped_provider_count=skipped_provider,
    )
