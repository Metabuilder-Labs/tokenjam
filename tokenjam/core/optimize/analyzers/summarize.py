"""
Summarize analyzer — surfaces static prompt files worth summarizing (Track A).

Unlike the other analyzers, this one reasons over the **filesystem**, not
telemetry: it runs the read-only, catalog-default summarize scan
(`core/summarize/candidates.list_candidates`) and reports the per-call
prompt-token reduction available by summarizing those files' prose. It carries
the #111 recoverable-savings contract so the Overview waste band and the
`/cost/components` overlay pick it up with no UI change (registry-driven).

Window-guarded: like every other recoverable finding, it contributes nothing on
a dead telemetry window (`ctx.summary.total_tokens == 0`). A window with no calls
has no per-call saving to attach a recoverable figure to, and surfacing one would
break the empty-window overlay invariant (#211) — a dead window must show no
recoverable waste. The filesystem scan is skipped entirely until the window shows
activity.

Honesty discipline (Critical Rule 14 + `core/summarize/estimate.py`): the figure
we can stand behind from a file alone is **tokens**. A per-call dollar amount at
default rates is noise, and the meaningful *amortized* dollar needs a real
per-file call count (telemetry) we don't have — so `estimated_recoverable_usd`
is left `None` (the contract's "no estimate available for this state"), and the
usage-ranked/amortized path is deferred. Every user-visible string says
"estimated" / "review before applying" — never "saves you"; the mandatory
`caveat` names summary's one risk (meaning may change, structure won't).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.summarize.detect import CHARS_PER_TOKEN

logger = logging.getLogger(__name__)

# Surfaced verbatim next to the recoverable figure (contract requires an explicit
# basis; states the filesystem/per-call basis so it isn't mistaken for a
# telemetry-window figure like the other analyzers).
SUMMARIZE_ESTIMATE_BASIS = (
    "Per-call prompt-token reduction from a read-only filesystem scan of catalog "
    "prompt files (CLAUDE.md / AGENTS.md / globals); prose is summarized, structure "
    "kept verbatim. Realized on every call that sends the cached prompt — NOT "
    "multiplied by this window's calls (that amortized figure needs per-file usage "
    "telemetry). Advisory; review each rewrite before applying."
)

# Mandatory caveat (Rule 14) — carried as the dataclass default like the other
# recoverable findings' caveats (MODEL_DOWNGRADE_CAVEAT etc.) so no surface can
# drop it. Names summary's ONE risk: structure is guaranteed (restore-by-id),
# meaning is not.
SUMMARIZE_HONESTY_CAVEAT = (
    "Structure is guaranteed; meaning may change — review each rewrite before applying."
)


@dataclass
class SummarizeCandidate:
    """One summarizable prompt file (mirrors core/summarize Candidate, trimmed)."""

    path: str
    kind: str          # "prompt" | "other"
    scope: str         # global | project | repo | path
    est_tokens_saved: int
    total_chars: int = 0     # source size (feeds the aggregate reduction %)
    reduction_pct: int = 0   # per-file prose reduction %, computed server-side (no JS chars/4)


@dataclass
class SummarizeFinding:
    """Filesystem-derived summarize opportunity, on the #111 recoverable contract.

    Tokens-only by design (see module docstring): `estimated_recoverable_usd`
    stays None until a usage-ranked (telemetry) path lands.
    """

    candidates: list[SummarizeCandidate] = field(default_factory=list)
    files: int = 0
    estimated_recoverable_usd: float | None = None
    estimated_recoverable_tokens: int | None = None
    estimate_basis: str = ""
    estimate_confidence: str = "heuristic"
    caveat: str = SUMMARIZE_HONESTY_CAVEAT
    # Prose-reduction %s computed server-side (single source of truth — the Lens
    # screen renders these instead of re-deriving chars/CHARS_PER_TOKEN in JS):
    #   reduction_pct     = aggregate saved ÷ source tokens across all candidates
    #   avg_reduction_pct = mean of the per-file reduction %s
    reduction_pct: int | None = None
    avg_reduction_pct: int | None = None


def _src_tokens(total_chars: int) -> int:
    """Source-token estimate for a file's raw size, on the shared chars→tokens
    constant (not a magic /4) so the % matches the rest of the pipeline."""
    return round(total_chars / CHARS_PER_TOKEN)


def _reduction_pct(est_tokens_saved: int, total_chars: int) -> int:
    """Per-file prose reduction % (saved ÷ source tokens), on the shared basis."""
    src = _src_tokens(total_chars)
    return round(est_tokens_saved / src * 100) if src > 0 else 0


@register("summarize")
def run(ctx: AnalyzerContext) -> None:
    """Attach a SummarizeFinding: catalog-default candidates + per-call token saving.

    Reasons over the filesystem (config-driven scan), not `ctx.conn`. The scan is
    catalog-default (a handful of known prompt files) so it's cheap enough for the
    polling Overview; a filesystem hiccup never breaks the optimize report.
    """
    finding = SummarizeFinding(estimate_basis=SUMMARIZE_ESTIMATE_BASIS)

    # Window-guard: a dead telemetry window has no calls to realize a per-call
    # saving against, so — like every recoverable finding — contribute nothing
    # rather than leak a filesystem figure into the empty-window overlay (#211).
    # Also skips the scan entirely on an idle window.
    if ctx.summary.total_tokens == 0:
        ctx.report.findings["summarize"] = finding
        return

    from tokenjam.core.summarize.candidates import list_candidates

    try:
        scan = list_candidates(config=ctx.config)  # read-only, never writes
    except Exception:
        # Empty finding on any scan failure so a filesystem hiccup never breaks the
        # optimize report — but log it: a silent broad-swallow would hide a real
        # code/config regression in list_candidates as if it were a benign hiccup.
        logger.debug(
            "summarize analyzer: candidate scan failed; returning empty finding",
            exc_info=True,
        )
        ctx.report.findings["summarize"] = finding
        return

    finding.candidates = [
        SummarizeCandidate(
            path=c.path,
            kind="prompt" if c.is_prompt else "other",
            scope=c.scope,
            est_tokens_saved=c.est_tokens_saved,
            total_chars=c.total_chars,
            reduction_pct=_reduction_pct(c.est_tokens_saved, c.total_chars),
        )
        for c in scan.candidates
        if c.est_tokens_saved > 0
    ]
    finding.files = len(finding.candidates)
    if finding.candidates:
        finding.estimated_recoverable_tokens = sum(
            c.est_tokens_saved for c in finding.candidates
        )
        # estimated_recoverable_usd intentionally left None — see module docstring.
        # Prose-reduction %s, computed here so the UI has a single compute path:
        #   reduction_pct     = token-weighted aggregate (saved ÷ source tokens)
        #   avg_reduction_pct = mean of the per-file reduction %s
        total_src = sum(_src_tokens(c.total_chars) for c in finding.candidates)
        if total_src > 0:
            finding.reduction_pct = round(
                finding.estimated_recoverable_tokens / total_src * 100
            )
        finding.avg_reduction_pct = round(
            sum(c.reduction_pct for c in finding.candidates) / len(finding.candidates)
        )
    ctx.report.findings["summarize"] = finding
