"""
Summarize analyzer — surfaces static prompt files worth summarizing (Track A).

Unlike the other analyzers, this one reasons over the **filesystem**, not
telemetry: it runs the read-only, catalog-default summarize scan
(`core/summarize/candidates.list_candidates`) and reports the prompt-token
reduction available by summarizing those files' prose, priced over the
analyzed window (see below) so `estimated_recoverable_tokens` and
`estimated_recoverable_usd` are on the same basis. It carries the #111
recoverable-savings contract so the Overview waste band and the
`/cost/components` overlay pick it up with no UI change (registry-driven).

Window-guarded: like every other recoverable finding, it contributes nothing on
a dead telemetry window (`ctx.summary.total_tokens == 0`). A window with no calls
has no per-call saving to attach a recoverable figure to, and surfacing one would
break the empty-window overlay invariant (#211) — a dead window must show no
recoverable waste. The filesystem scan is skipped entirely until the window shows
activity.

**The saving RECURS; it is not a one-time figure.** Every file in the catalog
is always-on context: re-sent at the head of every session that loads it, and
re-read on every call within that session. Compressing it therefore pays on
each of those, so the recoverable figure is priced the same way each of the
other cost analyzers' is — over the analyzed window, at the rates the tokens
actually bill at (first call of a session at the input rate, later calls at the
cache-read rate, which is 0.100x the input rate for every Anthropic model in
`pricing/models.toml`). See `_price_reduction`.

This is also the ONE analyzer whose fix has a NEGATIVE standing cost: it
shrinks the always-loaded footprint that the rule-writing analyzers (`relearn`,
`script`, `reuse`, `resend`) grow. `write_budget.measured_agent_file_tokens`
reads this finding as the denominator of their write budget, which is why
`summarize` is deliberately not a `COST_ANALYZERS` member — see the note there.

Honesty discipline (Critical Rule 14 + `core/summarize/estimate.py`): a window
figure — tokens or dollars — is only attached where the load count is
OBSERVED. A global-scope file (`~/.claude/...`) is loaded by every session in
the window, which telemetry counts directly; a project-scope file is loaded
only by its own repo's sessions, matched through the same `agent_id` -> repo
derivation `analyzers/relearn.py` uses. A file whose loading sessions cannot be
identified contributes NEITHER window figure — never a zero, never a rate
borrowed from a file that did resolve (anti-pattern #22) — but its one-time
per-call reduction still surfaces via `file_reduction_tokens` and each
candidate's own `est_tokens_saved`. Every user-visible string says
"estimated" / "review before applying" — never "saves you"; the mandatory
`caveat` names summary's one risk (meaning may change, structure won't).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenjam.core.optimize.rate_profile import RateProfile, blended_rate_profile
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.summarize.detect import CHARS_PER_TOKEN

logger = logging.getLogger(__name__)

#: Prefix Claude Code backfill stamps on a session's ``agent_id``
#: (``core/backfill.py`` ``_agent_id_from_cwd``). Stripping it yields the repo
#: directory name, which is what a project-scope candidate path is matched on.
_CC_AGENT_PREFIX = "claude-code-"

# Surfaced verbatim next to the recoverable figure (contract requires an explicit
# basis). States the basis of BOTH aggregate fields — tokens as well as dollars —
# so no consumer has to guess that they describe the same quantity, and names
# where the one-time per-call reduction lives instead.
SUMMARIZE_ESTIMATE_BASIS = (
    "Read-only filesystem scan of catalog prompt files (CLAUDE.md / AGENTS.md / "
    "globals); prose is summarized, structure kept verbatim. These files are "
    "ALWAYS-ON context, so the reduction is realized repeatedly, not once: "
    "`estimated_recoverable_tokens` and `estimated_recoverable_usd` are on the "
    "SAME basis — both price reduction x (sessions that load the file) x (first "
    "call at the input rate + each later call in that session at the cache-read "
    "rate), on the same window basis as the other cost analyzers; the tokens "
    "field reports that event count in tokens, the dollars field prices it. "
    "Load counts are observed from telemetry — a global-scope file against "
    "every session in the window, a project-scope file against its own repo's "
    "sessions only. A file whose loading sessions cannot be identified carries "
    "neither figure here; its one-time per-call reduction still appears in "
    "`file_reduction_tokens` and each candidate's own `est_tokens_saved`. "
    "Advisory; review each rewrite before applying."
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
    #: How many of the window's sessions actually load this file, and what the
    #: reduction is worth across them. ``est_usd_saved`` is ``None`` when the
    #: loading sessions could not be identified or no model was priced — a
    #: zero would read as "compressing this file is worth nothing".
    sessions_loading: int = 0
    est_usd_saved: float | None = None
    #: The actual token volume ``est_tokens_saved`` is worth over the analyzed
    #: window: removed on every read (first send + every re-read) across every
    #: loading session — the SAME event count ``est_usd_saved`` prices in
    #: dollars. ``None`` on the same "no loading session observed"
    #: condition ``est_usd_saved`` uses; the one-time per-call reduction stays
    #: available as ``est_tokens_saved``.
    est_tokens_saved_window: int | None = None


@dataclass
class SummarizeFinding:
    """Filesystem-derived summarize opportunity, on the #111 recoverable contract.

    ``estimated_recoverable_tokens`` and ``estimated_recoverable_usd`` are on
    the SAME basis: both price the reduction over the analyzed window
    (removed on every read, across every loading session), so a rollup that
    sums tokens across analyzers and one that sums dollars across analyzers
    describe the same underlying quantity. The one-time, per-call file
    reduction the curate/diff UI cares about lives separately in
    ``file_reduction_tokens`` and each candidate's own ``est_tokens_saved`` —
    it is never the aggregate field's basis.
    """

    candidates: list[SummarizeCandidate] = field(default_factory=list)
    files: int = 0
    estimated_recoverable_usd: float | None = None
    estimated_recoverable_tokens: int | None = None
    #: One-time sum of every candidate's ``est_tokens_saved`` — the aggregate
    #: per-call prose reduction, independent of how many sessions/calls
    #: actually reread it. Feeds the curate/diff UI and ``reduction_pct``;
    #: NOT summed into cross-analyzer token rollups (``estimated_recoverable_
    #: tokens`` is the window-priced figure that belongs there).
    file_reduction_tokens: int | None = None
    estimate_basis: str = ""
    estimate_confidence: str = "heuristic"
    caveat: str = SUMMARIZE_HONESTY_CAVEAT
    # Prose-reduction %s computed server-side (single source of truth — the Lens
    # screen renders these instead of re-deriving chars/CHARS_PER_TOKEN in JS):
    #   reduction_pct     = aggregate saved ÷ source tokens across all candidates
    #   avg_reduction_pct = mean of the per-file reduction %s
    reduction_pct: int | None = None
    avg_reduction_pct: int | None = None
    # The observed inputs behind `estimated_recoverable_usd`, carried so the
    # figure is inspectable rather than a black box: how many sessions the
    # window held, how many calls each made on average (every one of which
    # re-sends these files), and which models the rate blend came from.
    sessions_examined: int = 0
    calls_per_session: float | None = None
    rate_basis: str = ""


def _src_tokens(total_chars: int) -> int:
    """Source-token estimate for a file's raw size, on the shared chars→tokens
    constant (not a magic /4) so the % matches the rest of the pipeline."""
    return round(total_chars / CHARS_PER_TOKEN)


def _reduction_pct(est_tokens_saved: int, total_chars: int) -> int:
    """Per-file prose reduction % (saved ÷ source tokens), on the shared basis."""
    src = _src_tokens(total_chars)
    return round(est_tokens_saved / src * 100) if src > 0 else 0


@dataclass(frozen=True)
class _LoadProfile:
    """How often the window's sessions would re-send an always-on prompt file.

    ``sessions_total`` is every session in the window (what a global-scope file
    is loaded by); ``sessions_by_repo``/``calls_by_repo`` narrow that for a
    project-scope file. ``calls_per_session`` is the window-WIDE average number
    of LLM calls a session makes — the first sends the file at the input rate,
    each later one re-reads it at the cache-read rate. A project-scope file
    must price against its OWN repo's average, not this blend across every
    other repo/agent in the window (see ``_repo_calls_per_session``); this
    field stays window-wide because it's also what a global-scope file
    legitimately prices against.
    """

    sessions_total: int
    sessions_by_repo: dict[str, int]
    calls_by_repo: dict[str, int]
    calls_per_session: float
    rates: RateProfile


def _load_profile(ctx: AnalyzerContext) -> _LoadProfile | None:
    """Observed session and call counts for the window, per repo.

    ``None`` when the window carries no LLM call or no priced model — the
    finding then reports tokens with no dollars rather than inventing a load
    count. Never raises: a DB hiccup degrades to the tokens-only shape.
    """
    rates = blended_rate_profile(
        ctx.conn, since=ctx.since, until=ctx.until, agent_id=ctx.agent_id,
    )
    if rates is None:
        return None
    clauses = ["name = 'gen_ai.llm.call'", "start_time >= $1", "start_time < $2"]
    params: list[Any] = [ctx.since, ctx.until]
    if ctx.agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(ctx.agent_id)
    try:
        rows = ctx.conn.execute(
            "SELECT agent_id, COUNT(DISTINCT session_id), COUNT(*) "
            "FROM spans WHERE " + " AND ".join(clauses) + " GROUP BY agent_id",
            params,
        ).fetchall()
    except Exception:
        logger.debug("summarize analyzer: load-profile query failed", exc_info=True)
        return None

    sessions_by_repo: dict[str, int] = {}
    calls_by_repo: dict[str, int] = {}
    sessions_total = 0
    calls_total = 0
    for agent_id, sessions, calls in rows:
        sessions = int(sessions or 0)
        calls = int(calls or 0)
        sessions_total += sessions
        calls_total += calls
        label = str(agent_id or "")
        if label.startswith(_CC_AGENT_PREFIX):
            label = label[len(_CC_AGENT_PREFIX):]
        if label:
            sessions_by_repo[label] = sessions_by_repo.get(label, 0) + sessions
            calls_by_repo[label] = calls_by_repo.get(label, 0) + calls
    if sessions_total <= 0:
        return None
    return _LoadProfile(
        sessions_total=sessions_total,
        sessions_by_repo=sessions_by_repo,
        calls_by_repo=calls_by_repo,
        calls_per_session=calls_total / sessions_total,
        rates=rates,
    )


def _sessions_loading(path: str, scope: str, profile: _LoadProfile) -> int:
    """How many of the window's sessions send this file at the head of every call.

    A global-scope file lives under ``~/.claude`` and is loaded by every
    session. A project/repo/path-scoped one is loaded only by sessions in its
    own repo, matched by walking the file's ancestor directory names against
    the repo labels telemetry recorded. An unmatched path returns 0, which
    leaves the file tokens-only rather than charging it to every session.
    """
    if scope == "global":
        return profile.sessions_total
    ancestors = {parent.name for parent in Path(path).parents if parent.name}
    return sum(
        count for repo, count in profile.sessions_by_repo.items() if repo in ancestors
    )


def _repo_calls_per_session(path: str, scope: str, profile: _LoadProfile) -> float:
    """This candidate's own repo's average calls-per-session, on the SAME
    ancestor-matching basis ``_sessions_loading`` uses.

    A global-scope file legitimately spans every session in the window, so it
    keeps the window-wide average. A project-scope file uses only its own
    repo's observed call rate — blending in every other repo/agent in the
    window would over- or under-state the recoverable figure whenever that
    repo's actual session behavior differs from the window average. Falls
    back to the window-wide average only when the repo match carries no
    session evidence (in practice unreachable from ``_price_reduction``, which
    already returns ``None`` for a zero-session candidate before this is
    consulted).
    """
    if scope == "global":
        return profile.calls_per_session
    ancestors = {parent.name for parent in Path(path).parents if parent.name}
    matched_sessions = sum(
        count for repo, count in profile.sessions_by_repo.items() if repo in ancestors
    )
    if matched_sessions <= 0:
        return profile.calls_per_session
    matched_calls = sum(
        count for repo, count in profile.calls_by_repo.items() if repo in ancestors
    )
    return matched_calls / matched_sessions


def _reads_per_session(calls_per_session: float) -> int:
    """Total times one session reads an always-on file: the first send plus
    every later re-read in that session — at least 1. Shared by
    ``_price_reduction`` and ``_tokens_saved_over_window`` so the dollar and
    token figures price the exact same event count."""
    return max(round(calls_per_session), 1)


def _price_reduction(
    tokens_saved: int, sessions: int, calls_per_session: float, rates: RateProfile,
) -> float | None:
    """What removing ``tokens_saved`` from an always-on file is worth over the
    window: the reduction, on each loading session, sent once at the input rate
    and re-read on that session's every later call at the cache-read rate.

    ``calls_per_session`` is the candidate's OWN repo's average (see
    ``_repo_calls_per_session``) for a project-scope file, or the window-wide
    average for a global-scope one — never a blend of the two.

    ``None`` when no session loads the file — the saving is real but this
    window carries no evidence of its size, and a zero would misreport that as
    "worth nothing".
    """
    if sessions <= 0 or tokens_saved <= 0:
        return None
    rereads = _reads_per_session(calls_per_session) - 1
    return rates.cost_of(float(tokens_saved), rereads) * sessions


def _tokens_saved_over_window(
    tokens_saved: int, sessions: int, calls_per_session: float,
) -> int | None:
    """Actual token volume ``tokens_saved`` is worth across the window: removed
    once per read (first send + every re-read), on every one of ``sessions``
    loading sessions — the exact event count ``_price_reduction`` prices in
    dollars, so the two fields stay on the same basis.

    ``None`` on the same "no evidence" condition ``_price_reduction`` returns
    ``None`` for: no session in the window was observed loading this file.
    """
    if sessions <= 0 or tokens_saved <= 0:
        return None
    return round(tokens_saved * _reads_per_session(calls_per_session) * sessions)


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

    # Observed load counts + blended rates for the window. `None` (dead or
    # unpriced window) leaves every candidate tokens-only, exactly as before.
    profile = _load_profile(ctx)

    candidates: list[SummarizeCandidate] = []
    for c in scan.candidates:
        if c.est_tokens_saved <= 0:
            continue
        sessions = _sessions_loading(c.path, c.scope, profile) if profile else 0
        calls_per_session = (
            _repo_calls_per_session(c.path, c.scope, profile) if profile else 0.0
        )
        candidates.append(SummarizeCandidate(
            path=c.path,
            kind="prompt" if c.is_prompt else "other",
            scope=c.scope,
            est_tokens_saved=c.est_tokens_saved,
            total_chars=c.total_chars,
            reduction_pct=_reduction_pct(c.est_tokens_saved, c.total_chars),
            sessions_loading=sessions,
            est_usd_saved=(
                _price_reduction(
                    c.est_tokens_saved, sessions, calls_per_session, profile.rates,
                )
                if profile else None
            ),
            est_tokens_saved_window=(
                _tokens_saved_over_window(c.est_tokens_saved, sessions, calls_per_session)
                if profile else None
            ),
        ))
    finding.candidates = candidates
    finding.files = len(finding.candidates)
    if finding.candidates:
        # One-time aggregate (curate/diff basis) — always available regardless
        # of whether any loading session was observed.
        finding.file_reduction_tokens = sum(
            c.est_tokens_saved for c in finding.candidates
        )
        window_tokens = [
            c.est_tokens_saved_window for c in finding.candidates
            if c.est_tokens_saved_window is not None
        ]
        priced = [c.est_usd_saved for c in finding.candidates if c.est_usd_saved is not None]
        # None, not 0, when nothing resolved: "no session in this window was
        # observed loading these files" is a different statement from
        # "compressing them is worth nothing" (anti-pattern #22). Applies
        # symmetrically to tokens and dollars now that both are window-priced
        # — a candidate contributes to either both sums or neither.
        finding.estimated_recoverable_tokens = sum(window_tokens) if window_tokens else None
        finding.estimated_recoverable_usd = round(sum(priced), 6) if priced else None
        if profile is not None:
            finding.sessions_examined = profile.sessions_total
            finding.calls_per_session = round(profile.calls_per_session, 2)
            finding.rate_basis = profile.rates.basis
        # Prose-reduction %s, computed here so the UI has a single compute path,
        # on the one-time file_reduction_tokens basis (a window-priced numerator
        # here would read as >100% reduction once sessions/calls multiply in):
        #   reduction_pct     = token-weighted aggregate (saved ÷ source tokens)
        #   avg_reduction_pct = mean of the per-file reduction %s
        total_src = sum(_src_tokens(c.total_chars) for c in finding.candidates)
        if total_src > 0:
            finding.reduction_pct = round(
                finding.file_reduction_tokens / total_src * 100
            )
        finding.avg_reduction_pct = round(
            sum(c.reduction_pct for c in finding.candidates) / len(finding.candidates)
        )
    ctx.report.findings["summarize"] = finding
