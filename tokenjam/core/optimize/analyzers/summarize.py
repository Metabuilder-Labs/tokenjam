"""
Summarize analyzer — surfaces static prompt files worth summarizing (Track A).

Unlike the other analyzers, this one reasons over the **filesystem**, not
telemetry: it runs the read-only, catalog-default summarize scan
(`core/summarize/candidates.list_candidates`) and reports the per-call
prompt-token reduction available by summarizing those files' prose. It carries
the #111 recoverable-savings contract so the Overview waste band and the
`/cost/components` overlay pick it up with no UI change (registry-driven).

Honesty discipline (Critical Rule 14 + `core/summarize/estimate.py`): the figure
we can stand behind from a file alone is **tokens**. A per-call dollar amount at
default rates is noise, and the meaningful *amortized* dollar needs a real
per-file call count (telemetry) we don't have — so `estimated_recoverable_usd`
is left `None` (the contract's "no estimate available for this state"), and the
usage-ranked/amortized path is deferred. Every user-visible string says
"estimated" / "review before applying" — never "saves you".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext

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


@dataclass
class SummarizeCandidate:
    """One summarizable prompt file (mirrors core/summarize Candidate, trimmed)."""

    path: str
    kind: str          # "prompt" | "other"
    scope: str         # global | project | repo | path
    est_tokens_saved: int
    total_chars: int = 0   # source size, so the UI can render an avg-reduction % (DEC-032 tile)


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


@register("summarize")
def run(ctx: AnalyzerContext) -> None:
    """Attach a SummarizeFinding: catalog-default candidates + per-call token saving.

    Reasons over the filesystem (config-driven scan), not `ctx.conn`. The scan is
    catalog-default (a handful of known prompt files) so it's cheap enough for the
    polling Overview; a filesystem hiccup never breaks the optimize report.
    """
    from tokenjam.core.summarize.candidates import list_candidates

    finding = SummarizeFinding(estimate_basis=SUMMARIZE_ESTIMATE_BASIS)
    try:
        scan = list_candidates(config=ctx.config)  # read-only, never writes
    except Exception:
        ctx.report.findings["summarize"] = finding
        return

    finding.candidates = [
        SummarizeCandidate(
            path=c.path,
            kind="prompt" if c.is_prompt else "other",
            scope=c.scope,
            est_tokens_saved=c.est_tokens_saved,
            total_chars=c.total_chars,
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
    ctx.report.findings["summarize"] = finding
