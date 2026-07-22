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

Provenance (source-file attribution, read-only):
  A bloat prompt's raw text carries no pointer back to disk — the DB span
  only has `agent_id` and the captured prompt string. But some bloated
  prompts are, verbatim, the content of a file the user owns and could edit
  (CLAUDE.md, a subagent definition, a slash command) — `core/summarize/`'s
  catalog (`agent_files.toml`) already knows the fixed/global locations and
  the project-relative names/globs for that class of file. This module reads
  that catalog (never writes it — see `core/summarize/` for the editor) and
  attempts to attribute each scored prompt to the ONE catalog file, if any,
  whose full content it verbatim-contains.

  Matching rule (conservative by construction, not by tuning): a catalog
  file is attributed to a prompt ONLY when the file's ENTIRE content — after
  whitespace-only normalization (collapsed inline runs, stripped trailing
  space per line; never reordering or dropping words) — appears as a
  contiguous substring of the prompt's normalized text. This is verbatim
  containment, not a similarity/fuzzy score: a prompt that merely resembles
  a catalog file, or contains an edited/stale copy of one, does not match.
  Files under MIN_PROVENANCE_CHARS (after normalization) are never used as
  candidates, so a near-empty or boilerplate catalog file can't "match"
  everything. The overwhelming majority of prompts are expected to end with
  NO attribution — "provenance unknown" is the conservative, correct answer
  for prompt text that isn't a verbatim file copy (a summarized excerpt, a
  tool-result echo, hand-typed chat), not a shortfall of this pass.

  Scope of what's honestly checkable: global catalog paths (absolute, no cwd
  needed) are always candidates. Project-scoped catalog files (CLAUDE.md at
  a repo root, `.claude/agents/*.md`, etc.) are only checked for the ONE
  repo `tj optimize` is being run from (mirrors `summarize.candidates`'
  own cwd-rooted default) — and only when the prompt's own `agent_id`
  plausibly names that same repo (`claude-code-<basename>`, the convention
  `backfill.py` uses for Claude Code sessions). A prompt captured from a
  DIFFERENT repo than the one this run is scanning has no honestly-checkable
  local file and is correctly left unattributed, never guessed at.

  This pass establishes provenance ONLY. It has no apply path and writes
  nothing — whether tokenjam should ever auto-edit an attributed source file
  is a separate decision this module does not make.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.summarize.candidates import find_repo_root
from tokenjam.core.summarize.catalog import load_catalog
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

# A catalog file's (whitespace-normalized) content must be at least this many
# chars before it's used as a provenance candidate. Guards against a tiny or
# boilerplate catalog file (a near-empty CLAUDE.md, say) trivially "matching"
# as a substring of nearly any long-enough prompt.
MIN_PROVENANCE_CHARS = 200

# Claude Code-sourced spans carry agent_id = "claude-code-<repo-basename>"
# (backfill.py's `_agent_id_from_cwd`) — the only cwd hint a span carries.
_CLAUDE_CODE_AGENT_PREFIX = "claude-code-"


def _model_cache_dir() -> str:
    base = os.environ.get(
        "TOKENJAM_MODEL_CACHE",
        os.path.expanduser("~/.cache/tokenjam/models"),
    )
    os.makedirs(base, exist_ok=True)
    return base


# --- Provenance: catalog-file attribution (read-only) ----------------------
#
# See the module docstring's "Provenance" section for the full rule and its
# scope. Everything below only ever READS files (the catalog itself, and the
# catalog-named files it points at) — never writes one.

_WS_RUN_RE = re.compile(r"[ \t]+")


def _normalize_for_match(text: str) -> str:
    """Whitespace-only normalization for verbatim-containment matching:
    collapse runs of spaces/tabs to one space and drop trailing space on each
    line. Never reorders, drops, or fuzzes word content — a prompt that
    differs from a catalog file by more than incidental whitespace (a stale
    copy, a paraphrase, an excerpt) will NOT normalize to the same string and
    so will not match."""
    return "\n".join(_WS_RUN_RE.sub(" ", line).rstrip() for line in text.splitlines())


def _agent_repo_basename(agent_id: str) -> str | None:
    """The repo basename implied by a Claude Code-sourced `agent_id`
    (`claude-code-<basename>`), or None for any other agent_id shape — SDK/
    litellm-captured spans use a different convention and carry no cwd hint
    at all, so they never qualify for project-scoped provenance candidates
    (global-catalog candidates are still checked for them)."""
    if not agent_id.startswith(_CLAUDE_CODE_AGENT_PREFIX):
        return None
    basename = agent_id[len(_CLAUDE_CODE_AGENT_PREFIX):]
    return basename or None


@dataclass
class _ProvenanceIndex:
    """Precomputed, whitespace-normalized catalog-file candidates for one
    analyzer run — built once so per-prompt matching is a substring check,
    not a re-read-and-renormalize of every catalog file per prompt.
    ``global_candidates``/``project_candidates`` are ``(path, normalized
    content)`` pairs, pre-filtered to ``>= MIN_PROVENANCE_CHARS``.
    """
    global_candidates:  list[tuple[str, str]] = field(default_factory=list)
    project_candidates: list[tuple[str, str]] = field(default_factory=list)
    project_root:        Path | None          = None


def _catalog_global_files() -> list[Path]:
    """Catalog global/system paths that exist on this machine ("~" expanded,
    globs expanded) — always honestly checkable, no cwd needed."""
    out: list[Path] = []
    for raw in load_catalog().global_paths:
        p = Path(os.path.expanduser(raw))
        if p.is_file():
            out.append(p)
    return out


def _catalog_project_files(project_root: Path) -> list[Path]:
    """Catalog-known bare filenames + globs present at ``project_root``."""
    cat = load_catalog()
    out: list[Path] = []
    for name in sorted(cat.project_files):
        p = project_root / name
        if p.is_file():
            out.append(p)
    for pattern in cat.project_globs:
        out.extend(sorted(p for p in project_root.glob(pattern) if p.is_file()))
    return out


def _normalized_candidates(paths: list[Path]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        normalized = _normalize_for_match(content)
        if len(normalized) >= MIN_PROVENANCE_CHARS:
            out.append((str(path), normalized))
    return out


def build_provenance_index(project_root: Path | None) -> _ProvenanceIndex:
    """Build the candidate set for one run. ``project_root`` is the repo
    ``tj optimize`` is being run from (or None when it can't be resolved,
    e.g. not inside a git repo) — see the module docstring for why
    project-scoped candidates are limited to that ONE repo."""
    return _ProvenanceIndex(
        global_candidates=_normalized_candidates(_catalog_global_files()),
        project_candidates=(
            _normalized_candidates(_catalog_project_files(project_root))
            if project_root is not None else []
        ),
        project_root=project_root,
    )


def attribute_provenance(
    text: str, agent_id: str, index: _ProvenanceIndex,
) -> tuple[str | None, str]:
    """Attribute ``text`` to a catalog file, or (None, "") when nothing
    clears the verbatim-containment bar (the common, and conservatively
    correct, outcome — see the module docstring). Project-scoped candidates
    are only considered when ``agent_id``'s implied repo basename matches
    ``index.project_root`` — a prompt from a different repo has no
    honestly-checkable local file here.
    """
    candidates = list(index.global_candidates)
    if (
        index.project_root is not None
        and _agent_repo_basename(agent_id) == index.project_root.name.lower()
    ):
        candidates += index.project_candidates
    if not candidates:
        return None, ""
    normalized_text = _normalize_for_match(text)
    for path, normalized_file in candidates:
        if normalized_file in normalized_text:
            return path, (
                f"verbatim match: prompt contains {path}'s full content "
                f"unchanged (whitespace-normalized)."
            )
    return None, ""


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
    # Provenance (read-only, see module docstring): the catalog file this
    # prompt's text verbatim-contains, or None when no catalog file cleared
    # the bar — the expected outcome for most prompts, not a failure.
    source_path:  str | None = None
    #: One-line explanation of the match; "" when source_path is None.
    source_basis: str = ""


# Rough characters-per-token ratio for English prose. Used to convert the
# char-denominated bloat measurement into a token-denominated estimate.
CHARS_PER_TOKEN = 4


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
    # Provenance coverage (read-only pass, no apply path — see module
    # docstring): of `prompts_scored` prompts, how many were attributed to a
    # catalog file. Counted over EVERY scored prompt, not just the top-10
    # kept in `per_prompt`, so this fraction stays honest under truncation.
    prompts_with_provenance: int = 0
    # Recoverable-savings contract (#111). See types.DowngradeFinding for field
    # semantics. None when the analyzer is not ready (capture off, extra
    # missing) or no bloat was found.
    estimated_recoverable_usd:    float | None = None
    estimated_recoverable_tokens: int | None   = None
    estimate_basis:               str          = ""
    estimate_confidence:          str          = "heuristic"
    # The effective significance bar this run applied (config-overridable,
    # see core.config.OptimizeConfig.trim_significance_threshold) — carried on
    # the finding so a renderer never hardcodes a number that could be stale
    # against the user's own config.
    significance_threshold:       float        = SIGNIFICANCE_THRESHOLD


def estimate_trim_recoverable(
    low_sig_tokens: int, avg_input_rate_per_mtok: float
) -> float:
    """Price low-significance tokens at the window-average input rate."""
    return round((low_sig_tokens / 1_000_000) * avg_input_rate_per_mtok, 6)


def _window_avg_input_rate(conn, since, until, agent_id: str | None) -> float:
    """Input-token-weighted average input rate ($/MTok) across the window's
    model mix, used to price trimmable tokens."""
    from tokenjam.core.pricing import get_rates

    clauses = ["start_time >= $1", "start_time < $2",
               "provider IS NOT NULL", "model IS NOT NULL"]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT provider, model, COALESCE(SUM(input_tokens), 0) "
        f"FROM spans WHERE {where} GROUP BY provider, model",
        params,
    ).fetchall()
    weighted = 0.0
    total = 0
    for provider, model, in_tok in rows:
        in_tok = int(in_tok or 0)
        if in_tok <= 0:
            continue
        rates = get_rates(str(provider), str(model))
        if rates is None:
            continue
        weighted += rates.input_per_mtok * in_tok
        total += in_tok
    return (weighted / total) if total > 0 else 0.0


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


def _regions_from_scores(
    text: str, token_scores: list[tuple[str, float]],
    *, significance_threshold: float = SIGNIFICANCE_THRESHOLD,
) -> list[BloatRegion]:
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

        if score < significance_threshold:
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
    optimize_cfg = getattr(ctx.config, "optimize", None)
    significance_threshold = getattr(
        optimize_cfg, "trim_significance_threshold", SIGNIFICANCE_THRESHOLD,
    )

    capture = getattr(ctx.config, "capture", None)
    if capture is None or not getattr(capture, "prompts", False):
        ctx.report.findings["trim"] = PromptBloatFinding(
            enabled=False,
            significance_threshold=significance_threshold,
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
            significance_threshold=significance_threshold,
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
        ctx.report.findings["trim"] = PromptBloatFinding(
            enabled=True, significance_threshold=significance_threshold,
        )
        return

    # Lazy-instantiate the compressor with cached model storage.
    compressor = PromptCompressor(
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        use_llmlingua2=True,
        model_config={"cache_dir": _model_cache_dir()},
    )

    # Provenance candidates, built once for the run (see module docstring):
    # global catalog files always; project-scoped ones only for the repo
    # this run is being invoked from.
    provenance_index = build_provenance_index(find_repo_root(Path.cwd()))

    per_prompt: list[BloatPrompt] = []
    prompts_scored = 0
    prompts_skipped = 0
    prompts_with_provenance = 0
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

        regions = _regions_from_scores(
            text, scores, significance_threshold=significance_threshold,
        )
        bloat_chars = sum(r.char_length for r in regions)
        significant_chars = len(text) - bloat_chars
        # Rough token-reduction estimate: 4 chars/token, weighted by the
        # average low-significance score (lower score = more confidently bloat).
        est_tokens = int(bloat_chars / 4)
        total_bloat += bloat_chars
        total_chars += len(text)

        source_path, source_basis = attribute_provenance(
            text, str(agent_id or ""), provenance_index,
        )
        if source_path is not None:
            prompts_with_provenance += 1

        per_prompt.append(BloatPrompt(
            agent_id=str(agent_id),
            sample_chars=text[:120],
            prompt_chars=len(text),
            significant_chars=significant_chars,
            bloat_chars=bloat_chars,
            regions=regions,
            estimated_token_reduction=est_tokens,
            source_path=source_path,
            source_basis=source_basis,
        ))
        prompts_scored += 1

    # Sort by bloat absolute volume — biggest opportunities first.
    per_prompt.sort(key=lambda p: p.bloat_chars, reverse=True)

    # Recoverable-savings estimate (#111): low-significance tokens across all
    # scored prompts (not just the top-10 retained for display), priced at the
    # window-average input rate. None when nothing was flagged.
    low_sig_tokens = int(total_bloat / CHARS_PER_TOKEN) if total_bloat > 0 else 0
    rec_usd: float | None = None
    rec_tokens: int | None = None
    if low_sig_tokens > 0:
        avg_rate = _window_avg_input_rate(
            ctx.conn, ctx.since, ctx.until, ctx.agent_id
        )
        rec_usd = estimate_trim_recoverable(low_sig_tokens, avg_rate)
        rec_tokens = low_sig_tokens

    ctx.report.findings["trim"] = PromptBloatFinding(
        enabled=True,
        prompts_scored=prompts_scored,
        prompts_skipped=prompts_skipped,
        total_bloat_chars=total_bloat,
        total_chars=total_chars,
        per_prompt=per_prompt[:10],
        prompts_with_provenance=prompts_with_provenance,
        estimated_recoverable_usd=rec_usd,
        estimated_recoverable_tokens=rec_tokens,
        significance_threshold=significance_threshold,
        estimate_basis=(
            "low-significance tokens (≈4 chars/token) predicted by LLMLingua-2 "
            "× window input rate — review before editing prompts"
        ),
    )
