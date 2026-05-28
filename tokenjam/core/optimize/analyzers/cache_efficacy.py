"""
Cache-efficacy analyzer.

Measures the user's *current* prompt-caching usage by comparing cache_tokens
to input_tokens per (provider, model). Surfaces models where a large input
volume sees little caching — strong signal that enabling or expanding
`cache_control` placements would save tokens.

Per-provider scope:
  - Anthropic: fully supported. JSONL backfill and provider patches populate
    cache_tokens accurately, so the ratio is numerically meaningful.
  - OpenAI: best-effort. OpenAI's caching is implicit and per-call cache-hit
    data isn't exposed via the SDK in a consistent way. Output is rendered
    with an explicit caveat.
  - Google Gemini: best-effort, model-dependent. Some models expose cache
    metrics, some don't; the renderer surfaces a per-model caveat.
  - All others (Bedrock, LiteLLM, Cohere, etc.): unsupported in v1.

No content capture is required for this analyzer — it reads aggregate token
counts only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext

# Minimum input volume to surface a recommendation. Below this, the
# absolute savings are negligible regardless of efficacy.
MIN_INPUT_TOKENS = 100_000

# Efficacy threshold below which we flag the (provider, model). At 30%,
# the model is still leaving substantial caching savings on the table.
EFFICACY_THRESHOLD = 0.30

# Per-provider support level. Used by the renderer to qualify the finding.
PROVIDER_SUPPORT: dict[str, str] = {
    "anthropic":     "full",
    "openai":        "best_effort",
    "google":        "best_effort",
    "bedrock":       "unsupported",
    "local.ollama":  "unsupported",
}


@dataclass
class CacheEfficacyRow:
    """One (provider, model) row of current caching usage."""
    provider:      str
    model:         str
    input_tokens:  int
    cache_tokens:  int
    efficacy:      float           # cache_tokens / (input_tokens + cache_tokens)
    support:       str             # full | best_effort | unsupported
    flagged:       bool            # surfaced as a recommendation candidate


@dataclass
class CacheEfficacyFinding:
    """All (provider, model) rows in the window, plus the flagged subset."""
    rows:        list[CacheEfficacyRow] = field(default_factory=list)
    flagged:     list[CacheEfficacyRow] = field(default_factory=list)
    confidence:  str = "structural"


def _compute_rows(conn, since, until, agent_id: str | None) -> list[CacheEfficacyRow]:
    """Aggregate input_tokens and cache_tokens per (provider, model) in window."""
    clauses = [
        "start_time >= $1", "start_time < $2",
        "provider IS NOT NULL", "model IS NOT NULL",
    ]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT provider, model, "
        f"COALESCE(SUM(input_tokens), 0) AS in_tok, "
        f"COALESCE(SUM(cache_tokens), 0) AS cache_tok "
        f"FROM spans WHERE {where} "
        f"GROUP BY provider, model "
        f"ORDER BY in_tok + cache_tok DESC",
        params,
    ).fetchall()

    result: list[CacheEfficacyRow] = []
    for provider, model, in_tok, cache_tok in rows:
        in_tok = int(in_tok or 0)
        cache_tok = int(cache_tok or 0)
        total = in_tok + cache_tok
        # Efficacy is the share of total input bytes served from cache. 0
        # means no caching; 1 means every byte was a cache read (impossible
        # in practice — there's always some uncached system+user input).
        efficacy = (cache_tok / total) if total > 0 else 0.0
        support = PROVIDER_SUPPORT.get(str(provider).lower(), "unsupported")
        flagged = (
            support in {"full", "best_effort"}
            and in_tok >= MIN_INPUT_TOKENS
            and efficacy < EFFICACY_THRESHOLD
        )
        result.append(CacheEfficacyRow(
            provider=str(provider),
            model=str(model),
            input_tokens=in_tok,
            cache_tokens=cache_tok,
            efficacy=round(efficacy, 4),
            support=support,
            flagged=flagged,
        ))
    return result


@register("cache-efficacy")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches the finding to ctx.report.findings."""
    rows = _compute_rows(ctx.conn, ctx.since, ctx.until, ctx.agent_id)
    if not rows:
        return
    ctx.report.findings["cache-efficacy"] = CacheEfficacyFinding(
        rows=rows,
        flagged=[r for r in rows if r.flagged],
    )
