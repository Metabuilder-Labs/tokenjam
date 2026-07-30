"""Produce the rewrite evidence the estimate has been waiting for, on demand.

`estimate.observed_prose_ratio` supersedes the target ratio the moment enough
verified rewrites exist. Before this module, those rewrites only appeared if a
user happened to rewrite files for their own reasons, so on a fresh machine the
headline figure was permanently derived from the ASK. This runs the real rewrite
over a bounded sample so the measurement can exist because someone asked for it.

Three properties are load-bearing.

**Explicit and bounded.** A rewrite is an LLM call: it spends the user's tokens
and needs their auth. So this is a command the user types, it defaults to a
dry-run that spends nothing, it never runs inside an analyzer or on a request
path, and the sample size is capped (:data:`MAX_SAMPLES`) no matter what is
asked for.

**Deliberately sampled.** The largest prose candidates are taken first: the
ratio is weighted by prose volume, so the biggest files buy the most evidence
per call spent, and both credibility gates (`MIN_OBSERVED_SAMPLES` files,
`MIN_OBSERVED_PROSE_WORDS` words) are reached with the fewest rewrites.

**One pipeline.** Every rewrite goes through `delivery.summarize_via`, which is
prep → deliver → `session.check` → stage. There is no second rewrite path and no
second store: what this writes is exactly what `observed_prose_ratio` reads. A
rewrite that fails the structure gate is recorded as an attempt rather than
dropped, because "this file cannot be safely compressed" is itself a finding.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from tokenjam.core.config import TjConfig
from tokenjam.core.summarize import session
from tokenjam.core.summarize.candidates import list_candidates
from tokenjam.core.summarize.delivery import DeliveryError, summarize_via
from tokenjam.core.summarize.estimate import (
    MIN_OBSERVED_PROSE_WORDS,
    MIN_OBSERVED_SAMPLES,
    observed_prose_ratio,
)

logger = logging.getLogger(__name__)

#: Hard ceiling on rewrites per invocation, whatever ``limit`` asks for. Each one
#: is a billed model call on the user's account; a typo in ``--limit`` must not
#: be able to turn into an unbounded spend.
MAX_SAMPLES = 10
#: Default sample size: the file-count gate, which the volume gate almost always
#: clears too once the largest candidates are the ones chosen.
DEFAULT_SAMPLES = MIN_OBSERVED_SAMPLES

CALIBRATE_CONSENT_NOTE = (
    "Each sample is a real rewrite, which is a model call billed to you: "
    "`--via claude-p` spends your Claude Code quota, `--via api` bills your "
    "TJ_ANTHROPIC_API_KEY. Nothing is written to your files; the rewrites are "
    "staged for review exactly as `tj summarize prep --via` stages them."
)


@dataclass(frozen=True)
class CalibrationTarget:
    """One file selected for sampling, before any model call is made."""

    path: str
    prose_words: int
    est_tokens_saved: int

    def to_dict(self) -> dict:
        return {"path": self.path, "prose_words": self.prose_words,
                "est_tokens_saved": self.est_tokens_saved}


@dataclass(frozen=True)
class CalibrationSample:
    """The outcome of one sampled rewrite.

    ``achieved_ratio`` is ``None`` for anything that did not produce a usable
    outcome — a failed structure gate, a delivery error, a file below the prose
    gate. It is never a zero: "we did not learn a ratio here" and "this file
    compressed to nothing" are different statements.
    """

    path: str
    structure_ok: bool
    staged: bool
    prose_words_before: int = 0
    words_before: int = 0
    words_after: int = 0
    achieved_ratio: float | None = None
    rewrite_usd: float | None = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path, "structure_ok": self.structure_ok, "staged": self.staged,
            "prose_words_before": self.prose_words_before,
            "words_before": self.words_before, "words_after": self.words_after,
            "achieved_ratio": self.achieved_ratio, "rewrite_usd": self.rewrite_usd,
            "error": self.error,
        }


@dataclass(frozen=True)
class CalibrationReport:
    """What a calibration run learned, and what it cost to learn it.

    ``ratio_before`` / ``ratio_after`` are the value `observed_prose_ratio`
    returns either side of the run, so the caller can state plainly whether the
    estimate has moved off the target ratio. Both are ``None`` while the sample
    is too small — that is the honest reading, not a zero.

    ``rewrite_usd`` is ``None`` when no sampled call reported priced usage
    (``claude -p`` reports none; it spends quota, not per-token billing), which
    is distinct from a run that cost nothing.
    """

    via: str
    dry_run: bool
    planned: list[CalibrationTarget] = field(default_factory=list)
    samples: list[CalibrationSample] = field(default_factory=list)
    rewrite_usd: float | None = None
    ratio_before: float | None = None
    samples_before: int = 0
    ratio_after: float | None = None
    samples_after: int = 0
    gate_failures: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "via": self.via, "dry_run": self.dry_run,
            "planned": [t.to_dict() for t in self.planned],
            "samples": [s.to_dict() for s in self.samples],
            "rewrite_usd": self.rewrite_usd,
            "ratio_before": self.ratio_before, "samples_before": self.samples_before,
            "ratio_after": self.ratio_after, "samples_after": self.samples_after,
            "gate_failures": self.gate_failures,
            "note": self.note,
        }


def _already_sampled(config: TjConfig, path: str) -> bool:
    """True if this file already carries a staged result or a failed attempt.

    Re-rewriting a file that has already been sampled buys no new evidence and
    costs another call, so calibration moves on to the next-largest candidate.
    """
    if session.read_staged(config, path) is not None:
        return True
    return session.attempt_file(config, path).exists()


def plan_calibration(
    config: TjConfig, *, limit: int = DEFAULT_SAMPLES, path: str | None = None,
) -> list[CalibrationTarget]:
    """The files a run would sample: the largest un-sampled prose candidates.

    Reads the real filesystem through the same catalog scan `tj summarize list`
    uses, so the population is the user's actual prompt files.
    """
    n = max(1, min(int(limit), MAX_SAMPLES))
    scan = list_candidates(path, config=config)
    ranked = sorted(scan.candidates, key=lambda c: (-c.prose_words, c.path))
    out: list[CalibrationTarget] = []
    for c in ranked:
        if len(out) >= n:
            break
        p = Path(c.path).expanduser()
        if p.is_symlink() or not p.is_file():
            continue          # summarize refuses to rewrite through a link
        if _already_sampled(config, c.path):
            continue
        out.append(CalibrationTarget(
            path=c.path, prose_words=c.prose_words, est_tokens_saved=c.est_tokens_saved,
        ))
    return out


def _achieved_ratio(verdict: session.CheckVerdict) -> float | None:
    """Prose kept by this rewrite, on `observed_prose_ratio`'s own derivation.

    Structure is restored verbatim, so every word the file lost was a prose
    word. Clamped to [0, 1] for the same reason the aggregate is: a rewrite that
    grew the file kept all of its prose, it did not save a negative amount.
    """
    if verdict.prose_words_before <= 0 or verdict.words_before <= 0:
        return None
    removed = max(0, verdict.words_before - verdict.words_after)
    return min(1.0, max(0.0, 1.0 - removed / verdict.prose_words_before))


def _note(report_samples: list[CalibrationSample], ratio_after: float | None,
          n_after: int, prose_words: int) -> str:
    """One sentence saying whether the estimate can now use measured evidence."""
    if ratio_after is not None:
        return (
            f"The estimate now uses the measured ratio: prose to "
            f"{ratio_after * 100:.0f}% of its words across {n_after:,} "
            f"structure-checked rewrite(s)."
        )
    usable = sum(1 for s in report_samples if s.achieved_ratio is not None)
    missing_files = max(0, MIN_OBSERVED_SAMPLES - n_after)
    missing_words = max(0, MIN_OBSERVED_PROSE_WORDS - prose_words)
    need = []
    if missing_files:
        need.append(f"{missing_files:,} more structure-checked rewrite(s)")
    if missing_words:
        need.append(f"{missing_words:,} more prose word(s) of sample")
    tail = " and ".join(need) if need else "more evidence"
    return (
        f"Not enough evidence yet ({usable:,} usable sample(s) this run): "
        f"{tail} needed before the measured ratio replaces the target. The "
        f"estimate keeps using the target ratio and keeps saying so."
    )


def run_calibration(
    config: TjConfig,
    *,
    via: str,
    limit: int = DEFAULT_SAMPLES,
    go: bool = False,
    path: str | None = None,
    on_progress: "object | None" = None,
) -> CalibrationReport:
    """Sample real rewrites so the estimate can stop assuming what one delivers.

    Default is a DRY RUN: it reports which files would be sampled and spends
    nothing. ``go=True`` issues the model calls. ``on_progress``, when callable,
    receives a short status string per file so a slow run isn't silent.
    """
    def _p(msg: str) -> None:
        if callable(on_progress):
            on_progress(msg)

    ratio_before, n_before = observed_prose_ratio(config)
    planned = plan_calibration(config, limit=limit, path=path)
    if not go:
        return CalibrationReport(
            via=via, dry_run=True, planned=planned,
            ratio_before=ratio_before, samples_before=n_before,
            ratio_after=ratio_before, samples_after=n_before,
            note=(
                CALIBRATE_CONSENT_NOTE + " Dry run: nothing was called and "
                "nothing was spent. Re-run with --go to sample."
                if planned else
                "No un-sampled candidate to calibrate against. Every prompt "
                "file the scan found already carries a staged result or a "
                "recorded attempt."
            ),
        )

    samples: list[CalibrationSample] = []
    priced: list[float] = []
    for target in planned:
        _p(f"Rewriting {target.path} via {via}")
        try:
            outcome = summarize_via(config, target.path, via)
        except (DeliveryError, session.SummarizeRefused) as exc:
            # One unrewritable file must not abandon the sample: the next
            # candidate may well succeed, and the run has already been paid for.
            samples.append(CalibrationSample(
                path=target.path, structure_ok=False, staged=False, error=str(exc),
            ))
            continue
        usd = outcome.amortization.rewrite_usd if outcome.amortization else None
        if usd is not None:
            priced.append(usd)
        if outcome.verdict is None:                    # below the prose gate
            samples.append(CalibrationSample(
                path=target.path, structure_ok=False, staged=False,
                rewrite_usd=usd, error=outcome.skipped_note or "",
            ))
            continue
        v = outcome.verdict
        samples.append(CalibrationSample(
            path=target.path, structure_ok=v.structure_ok, staged=v.staged,
            prose_words_before=v.prose_words_before,
            words_before=v.words_before, words_after=v.words_after,
            achieved_ratio=_achieved_ratio(v) if v.structure_ok else None,
            rewrite_usd=usd,
            error="" if v.structure_ok else v.reason,
        ))

    ratio_after, n_after = observed_prose_ratio(config)
    sampled_prose = sum(
        int(r.get("prose_words_before") or 0)
        for r in session.list_staged(config) if r.get("staged")
    )
    return CalibrationReport(
        via=via, dry_run=False, planned=planned, samples=samples,
        rewrite_usd=round(sum(priced), 6) if priced else None,
        ratio_before=ratio_before, samples_before=n_before,
        ratio_after=ratio_after, samples_after=n_after,
        gate_failures=sum(1 for s in samples if not s.structure_ok),
        note=_note(samples, ratio_after, n_after, sampled_prose),
    )
