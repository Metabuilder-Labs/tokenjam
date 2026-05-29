"""
Trim analyzer (internal: trim).

Scores token-by-token significance in captured prompts using LLMLingua-2
(BERT-class token classifier, MIT-licensed, runs on CPU). Identifies
"bloat" regions — long, low-significance spans of repeated text the
model probably ignores. The user reviews the report and edits their
prompt template (e.g. CLAUDE.md, system prompt) to trim those regions.

This analyzer never auto-rewrites prompts. The honesty constraint is
strict: subtle compression breaks tasks in surprising ways. The renderer
output is recommendation-only; the apply step is manual.

Dependency handling:
  - `llmlingua` is an optional extra: `pip install "tokenjam[bloat]"`.
    The transitive footprint is ~2GB (PyTorch + transformers), so we
    don't pull it into the base install.
  - The import is deferred to the analysis function body so the analyzer
    self-registers and shows up in positional analyzer name choices regardless of
    whether the extra is installed.
  - Missing extra → analysis returns a finding with a clear message
    pointing the user at the install command.

Capture dependency:
  Requires `[capture] prompts = true`. Without captured content there's
  nothing to score.

Model handling:
  LLMLingua-2's BERT classifier (~110MB) downloads on first use and
  caches under `~/.cache/tokenjam/models/`. The cache directory is
  reused across runs; offline use after the first download works.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.otel.semconv import GenAIAttributes

# Tokens with a predicted significance score below this threshold are
# considered "bloat" — they contribute little to the model's output.
# 0.40 is LLMLingua-2's default; lower values are more aggressive.
SIGNIFICANCE_THRESHOLD = 0.40

# Minimum number of consecutive low-significance tokens before a region
# counts as a "bloat region." Single-token noise isn't actionable.
MIN_REGION_LENGTH = 20

# How many prompts to sample per (agent, prompt-template) cluster. Each
# scored prompt is ~100ms of CPU compute; capping keeps runs snappy.
MAX_PROMPTS_PER_RUN = 50


def _model_cache_dir() -> str:
    base = os.environ.get(
        "TOKENJAM_MODEL_CACHE",
        os.path.expanduser("~/.cache/tokenjam/models"),
    )
    os.makedirs(base, exist_ok=True)
    return base


@dataclass
class BloatRegion:
    """One contiguous low-significance region inside a prompt."""
    start_char:    int
    end_char:      int
    char_length:   int
    avg_score:     float
    sample_chars:  str        # first 80 chars of the region for preview


@dataclass
class BloatPrompt:
    """A single prompt's bloat analysis."""
    agent_id:        str
    sample_chars:    str         # first 120 chars of the prompt for identification
    prompt_chars:    int
    significant_chars: int       # chars above SIGNIFICANCE_THRESHOLD
    bloat_chars:     int         # chars in flagged regions
    regions:         list[BloatRegion] = field(default_factory=list)
    estimated_token_reduction: int = 0


@dataclass
class PromptBloatFinding:
    """Aggregate findings + per-prompt details."""
    enabled:           bool        # false when capture.prompts off or extra not installed
    prompts_scored:    int = 0
    prompts_skipped:   int = 0
    total_bloat_chars: int = 0
    total_chars:       int = 0
    per_prompt:        list[BloatPrompt] = field(default_factory=list)
    confidence:        str = "structural"
    hint:              str | None = None


def _try_import_llmlingua():
    """
    Deferred import. Returns the PromptCompressor class or raises a
    typed ImportError-with-hint that the analyzer surfaces to the user.
    """
    try:
        from llmlingua import PromptCompressor  # type: ignore[import-not-found]
        return PromptCompressor
    except ImportError as exc:
        raise ImportError(
            "Trim analyzer requires extra dependencies.\n\n"
            "Install with: pip install \"tokenjam[bloat]\"\n\n"
            "(This pulls in PyTorch and transformers, ~2GB. Optional because "
            "most users don't need this analyzer.)"
        ) from exc


def _score_prompt(compressor, text: str) -> list[tuple[str, float]]:
    """
    Run LLMLingua-2 on a single prompt text and return [(token_str, score), ...].

    LLMLingua-2's public API is `compress_prompt`, which both classifies
    and rewrites. For analysis-only we want the per-token scores. The
    underlying model is exposed as `compressor.model` — we use it
    directly to get raw scores without altering the prompt.

    This function is split out so tests can mock it without instantiating
    a real model.
    """
    # The compressor exposes `.compress_prompt(text, ratio=0.5)` and also
    # `.get_distillation_token_scores(text)` (or similar — API varies by
    # llmlingua version). We try the most-common entry points in order.
    if hasattr(compressor, "get_distillation_token_scores"):
        scores = compressor.get_distillation_token_scores(text)
        # Expected shape: list of (token, score) tuples
        return [(str(t), float(s)) for t, s in scores]
    # Fallback: call compress_prompt(rate=1.0) so nothing is removed and
    # parse the `compressed_prompt` plus `kept_tokens` metadata.
    result = compressor.compress_prompt(text, rate=1.0)
    tokens = result.get("kept_tokens") or result.get("tokens") or []
    scores = result.get("token_scores") or [1.0] * len(tokens)
    return [(str(t), float(s)) for t, s in zip(tokens, scores)]


def _regions_from_scores(text: str, token_scores: list[tuple[str, float]]) -> list[BloatRegion]:
    """
    Convert token-level scores into contiguous bloat regions of the
    underlying text. Walks the prompt linearly, advancing a character
    cursor by each token's length to map back to original positions.
    """
    regions: list[BloatRegion] = []
    cursor = 0
    current_start: int | None = None
    current_scores: list[float] = []

    for token_str, score in token_scores:
        # Find this token in the remaining text (LLMLingua-2 tokens may
        # have leading whitespace stripped; do a small forward scan).
        idx = text.find(token_str, cursor)
        if idx == -1:
            # Token doesn't appear in the source (could be a tokenizer
            # special char). Advance cursor minimally and continue.
            cursor += max(len(token_str), 1)
            continue
        token_start = idx
        token_end = idx + len(token_str)
        cursor = token_end

        if score < SIGNIFICANCE_THRESHOLD:
            if current_start is None:
                current_start = token_start
            current_scores.append(score)
        else:
            if current_start is not None:
                length = token_end - current_start - len(token_str)
                if length >= MIN_REGION_LENGTH:
                    end = token_end - len(token_str)
                    avg = sum(current_scores) / len(current_scores)
                    regions.append(BloatRegion(
                        start_char=current_start,
                        end_char=end,
                        char_length=end - current_start,
                        avg_score=round(avg, 3),
                        sample_chars=text[current_start:current_start + 80],
                    ))
                current_start = None
                current_scores = []

    # Flush trailing region
    if current_start is not None:
        end = cursor
        if end - current_start >= MIN_REGION_LENGTH:
            avg = sum(current_scores) / len(current_scores) if current_scores else 0.0
            regions.append(BloatRegion(
                start_char=current_start,
                end_char=end,
                char_length=end - current_start,
                avg_score=round(avg, 3),
                sample_chars=text[current_start:current_start + 80],
            ))
    return regions


def _stringify_prompt(value: Any) -> str:
    """Mirror cache_recommend's stringifier — kept local to avoid cross-analyzer imports."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for msg in value:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, list):
                    inner: list[str] = []
                    for block in content:
                        if isinstance(block, dict):
                            inner.append(str(block.get("text", "")))
                        else:
                            inner.append(str(block))
                    content = "".join(inner)
                parts.append(str(content))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    if isinstance(value, dict):
        return _stringify_prompt(value.get("content"))
    return str(value)


@register("trim")
def run(ctx: AnalyzerContext) -> None:
    """Registry entry point. Attaches a PromptBloatFinding to ctx.report.findings."""
    capture = getattr(ctx.config, "capture", None)
    if capture is None or not getattr(capture, "prompts", False):
        ctx.report.findings["trim"] = PromptBloatFinding(
            enabled=False,
            hint=(
                "Enable `[capture] prompts = true` in tj.toml and let the "
                "daemon collect a window of data before re-running this "
                "analyzer. Trim needs captured prompt text to score."
            ),
        )
        return

    # Defer LLMLingua-2 import to runtime so the analyzer self-registers
    # without forcing torch into the base install.
    try:
        PromptCompressor = _try_import_llmlingua()
    except ImportError as exc:
        ctx.report.findings["trim"] = PromptBloatFinding(
            enabled=False,
            hint=str(exc),
        )
        return

    # Fetch captured prompts from the window. Cap the sample size so a
    # single run doesn't score 1000s of prompts at ~100ms each.
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
        f"SELECT agent_id, attributes FROM spans WHERE {where} "
        f"LIMIT {MAX_PROMPTS_PER_RUN + 100}",  # over-fetch; skip empty-content rows below
        params,
    ).fetchall()

    if not rows:
        ctx.report.findings["trim"] = PromptBloatFinding(enabled=True)
        return

    # Lazy-instantiate the compressor with cached model storage.
    compressor = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
        model_config={"cache_dir": _model_cache_dir()},
    )

    per_prompt: list[BloatPrompt] = []
    prompts_scored = 0
    prompts_skipped = 0
    total_bloat = 0
    total_chars = 0

    for agent_id, attrs in rows:
        if prompts_scored >= MAX_PROMPTS_PER_RUN:
            break
        if isinstance(attrs, str):
            import json as _json
            try:
                attrs = _json.loads(attrs)
            except Exception:
                prompts_skipped += 1
                continue
        if not isinstance(attrs, dict):
            prompts_skipped += 1
            continue
        content = attrs.get(GenAIAttributes.PROMPT_CONTENT)
        if not content:
            prompts_skipped += 1
            continue
        text = _stringify_prompt(content)
        if len(text) < 200:
            prompts_skipped += 1
            continue

        try:
            scores = _score_prompt(compressor, text)
        except Exception:
            prompts_skipped += 1
            continue

        regions = _regions_from_scores(text, scores)
        bloat_chars = sum(r.char_length for r in regions)
        significant_chars = len(text) - bloat_chars
        # Rough token-reduction estimate: 4 chars/token, weighted by the
        # average low-significance score (lower score = more confidently bloat).
        est_tokens = int(bloat_chars / 4)
        total_bloat += bloat_chars
        total_chars += len(text)

        per_prompt.append(BloatPrompt(
            agent_id=str(agent_id),
            sample_chars=text[:120],
            prompt_chars=len(text),
            significant_chars=significant_chars,
            bloat_chars=bloat_chars,
            regions=regions,
            estimated_token_reduction=est_tokens,
        ))
        prompts_scored += 1

    # Sort by bloat absolute volume — biggest opportunities first.
    per_prompt.sort(key=lambda p: p.bloat_chars, reverse=True)

    ctx.report.findings["trim"] = PromptBloatFinding(
        enabled=True,
        prompts_scored=prompts_scored,
        prompts_skipped=prompts_skipped,
        total_bloat_chars=total_bloat,
        total_chars=total_chars,
        per_prompt=per_prompt[:10],
    )
