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
import json
from dataclasses import dataclass, field
from typing import Any

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.pricing import get_rates
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
    model:          str = ""       # this prefix's dominant model, for pricing
    # The one-paste fix (#128): a ready cache_control placement for this
    # exact prefix, built by `_prefix_cache_control_snippet` below. This
    # analyzer's whole job is WHERE to place a breakpoint, so leaving this
    # empty and only printing prose stats (as v1 did) meant the one analyzer
    # whose purpose is placement advice was the one that gave no placement
    # code, while `cache`'s own A3 lookback-miss path already built one.
    cache_control_snippet: str = ""
    # Recoverable-savings contract (#111), same field names as every other
    # analyzer. None when no priced rate was observed for `model` — never a
    # zero or a borrowed rate (CLAUDE.md anti-pattern #22).
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None


@dataclass
class CacheRecommendFinding:
    """Cache-control breakpoint recommendations for Anthropic prompts."""
    enabled:          bool                          # false when capture.prompts disabled
    candidates:       list[CachePrefixCandidate] = field(default_factory=list)
    skipped_provider_count: int = 0                 # non-Anthropic spans we ignored
    confidence:       str = "structural"
    hint:             str | None = None             # surfaced when enabled is false
    # Sum across candidates with a priced rate. None when none of them had one
    # (never a $0.00 standing in for "no rate observed").
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis:               str          = ""
    # The effective occurrence bar this run applied (config-overridable, see
    # core.config.OptimizeConfig.min_prefix_occurrences) — carried on the
    # finding so a renderer's empty-state message never hardcodes a number
    # that could be stale against the user's own config.
    min_prefix_occurrences:       int          = MIN_PREFIX_OCCURRENCES


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


def _prefix_cache_control_snippet(
    model: str, sample: str, occurrences: int, cacheable_tokens: int,
) -> str:
    """The one-paste fix for a recurring prefix. Modelled on
    `cache_efficacy._uncached_snippet`: a comment identifying which prefix
    (from the truncated preview already captured, never the full text) plus
    a placeholder `text` block carrying the `cache_control` breakpoint.

    A placeholder, not the real captured content: `sample` is only the first
    ~120 characters kept for the UI preview, not the full prefix, and
    pasting a partial prefix into a "snippet" would silently truncate the
    user's actual boundary if copied as-is. The preview is enough to let the
    user recognise WHICH of their own prefixes this is.
    """
    preview = sample[:80].replace("\n", " ").replace("\r", " ").strip()
    model_label = model or "this model"
    return (
        f"# {model_label}: prefix seen in {occurrences} calls, starting "
        f'"{preview}..."; ~{cacheable_tokens:,} tokens estimated cacheable\n'
        + json.dumps({
            "type": "text",
            "text": "<the stable prefix that starts this way, same content every call>",
            "cache_control": {"type": "ephemeral"},
        }, indent=2)
    )


def _estimate_candidate_recoverable(
    model: str, cacheable_tokens_per_call: int, occurrences: int,
) -> tuple[float | None, int | None]:
    """Price one prefix candidate — same rate lookup and rate-delta math as
    `cache_efficacy.estimate_cache_recoverable()` (get_rates, then
    `input_per_mtok - cache_read_per_mtok`), applied to the tokens THIS
    candidate would actually move onto the cache-read rate: the first
    occurrence still pays full price plus one cache write to establish the
    breakpoint, and every occurrence after that reads it back. Returns
    (None, None) when fewer than 2 occurrences (nothing left to read back) or
    no priced Anthropic rate was observed for `model` — never a zero, never a
    rate borrowed from a different model (CLAUDE.md anti-pattern #22).
    """
    if cacheable_tokens_per_call <= 0 or occurrences <= 1 or not model:
        return None, None
    rates = get_rates("anthropic", model)
    if rates is None or rates.cache_read_per_mtok <= 0:
        return None, None
    rate_delta = rates.input_per_mtok - rates.cache_read_per_mtok
    if rate_delta <= 0:
        return None, None
    reads = occurrences - 1
    read_savings = reads * cacheable_tokens_per_call * rate_delta / 1_000_000
    write_cost = cacheable_tokens_per_call * rates.cache_write_per_mtok / 1_000_000
    usd = round(max(0.0, read_savings - write_cost), 6)
    tokens = reads * cacheable_tokens_per_call
    return usd, tokens


@register("cache-recommend")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a CacheRecommendFinding to ctx.report.findings."""
    optimize_cfg = getattr(ctx.config, "optimize", None)
    min_prefix_occurrences = getattr(
        optimize_cfg, "min_prefix_occurrences", MIN_PREFIX_OCCURRENCES,
    )

    capture = getattr(ctx.config, "capture", None)
    if capture is None or not getattr(capture, "prompts", False):
        ctx.report.findings["cache-recommend"] = CacheRecommendFinding(
            enabled=False,
            min_prefix_occurrences=min_prefix_occurrences,
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
        f"SELECT provider, model, attributes, input_tokens FROM spans WHERE {where}",
        params,
    ).fetchall()

    # Aggregate by prefix hash. We track only Anthropic spans; everything
    # else gets counted so the renderer can disclose what was skipped.
    prefix_counts: dict[str, dict[str, Any]] = {}
    skipped_provider = 0
    for provider, model, attrs_raw, input_tokens in rows:
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
            "model_counts": {},
        })
        entry["count"] += 1
        entry["tokens_sum"] += int(input_tokens or 0)
        m = str(model or "")
        entry["model_counts"][m] = entry["model_counts"].get(m, 0) + 1

    candidates: list[CachePrefixCandidate] = []
    for h, entry in prefix_counts.items():
        if entry["count"] < min_prefix_occurrences:
            continue
        avg_in = entry["tokens_sum"] / entry["count"] if entry["count"] else 0.0
        # Rough estimate of cacheable tokens per call. A 2000-char prefix
        # is roughly 500 tokens (4 chars/token heuristic). The renderer
        # can refine via the user's actual tokeniser, but for v1 this
        # gives an order-of-magnitude signal.
        estimated_cacheable = min(int(avg_in * 0.5), PREFIX_HASH_BYTES // 4)
        # This prefix's most-called model prices the candidate — a prefix is
        # occasionally shared across a model switch mid-window, but pricing
        # off the dominant model is the honest single number to show.
        model_counts: dict[str, int] = entry["model_counts"]
        dominant_model = (
            max(model_counts.items(), key=lambda kv: kv[1])[0] if model_counts else ""
        )
        usd, tokens = _estimate_candidate_recoverable(
            dominant_model, estimated_cacheable, int(entry["count"]),
        )
        candidates.append(CachePrefixCandidate(
            prefix_hash=h,
            sample_chars=str(entry["sample"]),
            occurrences=int(entry["count"]),
            avg_input_tokens=round(avg_in, 1),
            estimated_cacheable_tokens=estimated_cacheable,
            model=dominant_model,
            cache_control_snippet=_prefix_cache_control_snippet(
                dominant_model, str(entry["sample"]), int(entry["count"]), estimated_cacheable,
            ),
            estimated_recoverable_usd=usd,
            estimated_recoverable_tokens=tokens,
        ))

    candidates.sort(key=lambda c: c.occurrences * c.estimated_cacheable_tokens,
                    reverse=True)
    candidates = candidates[:10]

    priced_usd = [c.estimated_recoverable_usd for c in candidates
                  if c.estimated_recoverable_usd is not None]
    priced_tokens = [c.estimated_recoverable_tokens for c in candidates
                      if c.estimated_recoverable_tokens is not None]
    finding_usd = round(sum(priced_usd), 6) if priced_usd else None
    finding_tokens = sum(priced_tokens) if priced_tokens else None
    basis = (
        "reads after the first occurrence at (input - cache-read) rate, minus "
        "one cache write per prefix at the write rate, priced off each "
        "prefix's most-called model; candidates with no priced rate observed "
        "for their model contribute no dollar figure"
    ) if priced_usd else ""

    ctx.report.findings["cache-recommend"] = CacheRecommendFinding(
        enabled=True,
        candidates=candidates,
        skipped_provider_count=skipped_provider,
        estimated_recoverable_usd=finding_usd,
        estimated_recoverable_tokens=finding_tokens,
        estimate_basis=basis,
        min_prefix_occurrences=min_prefix_occurrences,
    )
