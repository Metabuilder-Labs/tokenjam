"""
Summarize analyzer — surfaces static prompt files worth summarizing (Track A).

Unlike the other analyzers, this one reasons over the **filesystem**, not
telemetry: it runs the read-only, catalog-default summarize scan
(`core/summarize/candidates.list_candidates`) and reports the prompt-token
reduction available by summarizing those files' prose, priced over the
analyzed window (see below) so `past_overspend_tokens` and
`past_overspend_usd` are on the same basis. It carries the #111
recoverable-savings contract so the Overview waste band and the
`/cost/components` overlay pick it up with no UI change (registry-driven).

Window-guarded: like every other recoverable finding, it contributes nothing on
a dead telemetry window (`ctx.summary.total_tokens == 0`). A window with no calls
has no per-call saving to attach a recoverable figure to, and surfacing one would
break the empty-window overlay invariant (#211) — a dead window must show no
recoverable waste. The filesystem scan is skipped entirely until the window shows
activity.

**The file POPULATION is the window's, not the process's.** The scan used to
enumerate project-scope files from a single `Path.cwd()`, while the window it
prices spans every repo the user worked in — so every repo except the one the
process happened to sit in contributed nothing at all, hiding most of the true
figure. The roots are now derived from the working directories the analysed
sessions themselves recorded
(`core/summarize/repo_roots`), read off the same transcript pass that counts
invocations. A recorded root that no longer exists on disk is counted and
skipped, never reconstructed. Because a git worktree carries its repo's
directory name, every checkout of one file matches the same sessions; those
copies are charged once (see `_copy_key`), while a genuinely stacked pair — a
meta-repo's `CLAUDE.md` and a sub-repo's, both of which really are loaded — is
kept, distinguished by whether their roots nest.

That population composes with the filesystem scope (`core/optimize/scope`)
in one direction, and the direction matters: **the scope is the boundary, the
recorded working directories are the selection within it.** The scope decides
which disk this run may read; the recorded cwds decide which directories on it
are worth reading. So the transcript corpus is read under the scope's projects
root, and the roots derived from it must sit inside the scope's home before
they are scanned (`_scan_boundary`) — a transcript stored under a scoped root
can still record a working directory anywhere, so confining the corpus does not
by itself confine the population. The inverse composition is the trap to avoid:
filtering an UNSCOPED run against the default projects root would collapse the
population back to a single directory, because that root holds no source
project, restoring the very bug this enumeration exists to fix. A boundary is
therefore applied only when one was actually drawn, and whatever it excludes is
disclosed in the basis rather than silently dropped.

**The saving RECURS, but only for the part of a file that is actually
resident.** The catalog lumps five different things together and they are not
loaded the same way: `CLAUDE.md` and `.claude/rules/*.md` are re-sent whole at
the head of every session that loads them, whereas a `.claude/skills/*/SKILL.md`,
`.claude/commands/*.md` or `.claude/agents/*.md` surfaces only its frontmatter
that way and delivers its BODY when it is invoked. Pricing every file's whole
body as always-resident made a skill library that had not been invoked once in
the window read as the most expensive prompt file a user owns. So the figure is
the sum of two separately-observed terms:

    always-resident reduction x (sessions that load it) x (reads per session)
  + on-demand reduction       x (times it was actually invoked)

The first term bills as before — first send at the input rate, each later call
in that session at the cache-read rate (0.100x the input rate for every
Anthropic model in `pricing/models.toml`). The second bills at the input rate,
once per invocation. The split itself lives in `core/summarize/load_semantics`
(shared with `core/rulewrite/delivery`, which applies the same rule when
pricing a WRITE); the invocation counts are observed from Claude Code
transcripts by `core/summarize/invocations`. See `_price_reduction`.

That second term is a FLOOR: an invoked body stays in that session's context
for the calls that follow, and those re-reads are not counted here because the
transcript does not say how many followed. Understating is the safe direction
(Critical Rule 22).

This is also the ONE analyzer whose fix shrinks the always-loaded footprint
that the rule-writing analyzers (`relearn`, `script`, `reuse`, `resend`) grow,
which is why `summarize` is deliberately not a `COST_ANALYZERS` member — see
the note there.

Honesty discipline (Critical Rule 14 + `core/summarize/estimate.py`): a window
figure — tokens or dollars — is only attached where the load count is
OBSERVED. A global-scope file (`~/.claude/...`) is loaded by every session in
the window, which telemetry counts directly; a project-scope file is loaded
only by its own repo's sessions, matched through the same `agent_id` -> repo
derivation `analyzers/relearn.py` uses. A file whose loading sessions cannot be
identified contributes NEITHER window figure — never a zero, never a rate
borrowed from a file that did resolve (Critical Rule 22) — but its one-time
per-call reduction still surfaces via `file_reduction_tokens` and each
candidate's own `est_tokens_saved`. The same symmetry governs the on-demand
term: an on-demand file whose invocations could not be observed at all (no
transcript corpus) contributes neither figure, while one observed to have been
invoked ZERO times contributes exactly zero — a measurement, not a gap. Every
user-visible string says "estimated" / "review before applying" — never "saves
you"; the mandatory `caveat` names summary's one risk (meaning may change,
structure won't).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenjam.core.optimize.rate_profile import RateProfile, blended_rate_profile
from tokenjam.core.optimize.registry import register
from tokenjam.core.optimize.types import AnalyzerContext
from tokenjam.core.summarize import load_semantics
from tokenjam.core.summarize.detect import CHARS_PER_TOKEN
from tokenjam.core.summarize.estimate import (
    DEFAULT_TARGET_RATIO,
    MIN_OBSERVED_SAMPLES,
    PUBLISHED_LINE_TARGET,
    PUBLISHED_LINE_TARGET_QUOTE,
    PUBLISHED_LINE_TARGET_SOURCE,
    UNMEASURED_PRIOR_RANGE,
    UNMEASURED_PRIOR_RATIO,
    UNMEASURED_PRIOR_SAMPLES,
    gate_failed_attempts,
    observed_prose_ratio,
)
from tokenjam.core.summarize.invocations import InvocationCounts
from tokenjam.core.summarize.route import BEST_PRACTICES_SOURCE, PRUNE_TEST_QUOTE
from tokenjam.core.summarize.repo_roots import ResolvedRoots, resolve_roots
from tokenjam.core.persona_scope import add_persona_clause

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
    "rules / skills / commands / agents / globals). Prose is summarized; "
    "structure is kept verbatim. `past_overspend_tokens` and `past_overspend_usd` "
    "are on the SAME basis: the same event count, one counted and one priced. "
    "Each file is charged by how it is actually LOADED, not as if all of it "
    "were always in context.\n\n"
    "Always-resident text is a whole CLAUDE.md / rules file, and the frontmatter "
    "of a skill / command / agent (which is what the harness lists them by). It "
    "is charged reduction x sessions that load the file x reads per session. The "
    "first send is priced at the input rate. Each later call in that session is "
    "priced at the cache-read rate. On-demand text is a skill / command / agent "
    "BODY, which reaches the model only when invoked. It is charged reduction x "
    "OBSERVED invocations, at the input rate, once each.\n\n"
    "Session load counts come from telemetry. A global-scope file counts "
    "against every distinct session in the window; a project-scope file counts "
    "against its own repo's sessions only. Invocation counts are observed, "
    "never assumed ({invocation_source}). Zero observed invocations is a "
    "measurement and prices that body at zero.\n\n"
    "An absent transcript corpus carries NEITHER figure for that file. The "
    "on-demand term is a floor: an invoked body stays in that session's context "
    "afterwards, and those re-reads are not counted.\n\n"
    "The two terms are reported separately per file "
    "(`always_resident_tokens_saved` / `on_demand_tokens_saved`, alongside "
    "`load_class` and `invocations`). They must not be collapsed back into one: "
    "on this corpus the always-resident term dominates, because a small "
    "frontmatter re-read on every call of every session outweighs a large body "
    "delivered a handful of times. That is exactly why charging only the "
    "frontmatter, or only the body, would both be wrong.\n\n"
    "A file whose loading sessions cannot be identified carries neither figure "
    "here; its one-time per-call reduction still appears in "
    "`file_reduction_tokens` and each candidate's own `est_tokens_saved`. A "
    "file MEASURED to be resident in no session and invoked zero times is not "
    "listed at all, rather than listed at zero.\n\n"
    "{population}\n\n"
    "{ratio}\n\n"
    "Advisory. Review each rewrite before applying."
)

#: Reduction half of the basis (Critical Rule 14): how much of each file the
#: figure assumes goes away. Kept separate because the honest answer depends on
#: whether any rewrite has actually been verified on this machine.
_RATIO_TARGET = (
    "Symlinked files are excluded. The fix refuses to rewrite through a link, "
    "so a saving offered on one could never be realized.\n\n"
    "No verified rewrite exists on THIS machine yet ({samples:,} usable so "
    "far, {needed:,} needed). So the per-file reduction assumes prose "
    "compresses to {pct:.0f}% of its words. That figure is tokenjam's own "
    "measurement across {prior_n:,} rewrites of real instruction files on "
    "other machines, with achieved ratios spanning {lo:.0%} to {hi:.0%}. That "
    "spread is wide and those were not your files, so treat it as a prior "
    "rather than a property of your corpus.\n\n"
    "It is deliberately NOT the {target:.0%} target the rewriter is asked "
    "for. Nothing enforces that target (there is no retry and no gate on "
    "hitting it), and measured rewrites came nowhere near it. So estimating "
    "at the ask overstated this figure by roughly an order of magnitude. Run "
    "`tj summarize calibrate --via claude-p --go` to replace the prior with "
    "what rewrites actually deliver on your own files."
)
_RATIO_OBSERVED = (
    "Symlinked files are excluded. The fix refuses to rewrite through a link, "
    "so a saving offered on one could never be realized.\n\n"
    "The per-file reduction uses the ratio rewrites have ACTUALLY delivered "
    "on this machine: prose to {pct:.0f}% of its words, measured across "
    "{samples:,} structure-checked rewrite(s), weighted by prose volume. That "
    "figure is used rather than the target the rewriter is asked for, because "
    "nothing enforces that target and observed rewrites do not reach it."
)
#: Appended to either half whenever some rewrite here failed the structure gate.
#: Those attempts are excluded from the ratio on purpose (a mangled rewrite was
#: never a usable outcome), and a basis that stayed silent about them would let
#: a sample built mostly of failures read as a clean measurement.
_RATIO_GATE_FAILURES = (
    " {failures:,} attempted rewrite(s) here failed the structure check and "
    "are excluded from the ratio. Those files could not be safely compressed "
    "as summarized, which is a finding about them rather than a figure."
)
#: The reduction target for the files this guidance is published for. Only the
#: token delta is claimed as a saving (Critical Rule 14) — the adherence benefit
#: of a shorter file is real, is Anthropic's stated reason for the target, and is
#: NOT something tokenjam measures, so it appears here as rationale and never as
#: a dollar or token figure. The target also cannot inflate anything: it changes
#: what the rewriter is ASKED for, while the figure above stays bounded by the
#: measured (or target) prose ratio.
_LINE_TARGET_NOTE = (
    "\n\nFor an always-resident instruction file, the fix aims at a size "
    "rather than only a ratio: getting the file under {lines} lines. That "
    'target is Anthropic\'s published guidance ("{quote}" {source}), not '
    "tokenjam's. A large file compressed by a ratio can still be far over "
    "that size. The adherence half of that guidance is their rationale for "
    "the target, not a saving claimed here; only the token reduction is."
)
#: The offer must not present compression as the only or default route for an
#: instruction file. Such a file is usually long because rules ACCUMULATED, so
#: compressing it shortens each surviving rule rather than removing any — which
#: trades adherence for tokens and fails Critical Rule 26's third gate. The
#: other three routes do not. Every candidate carries a `reduction_route` so a
#: rule-heavy file is flagged as a PRUNE candidate rather than being offered
#: compression as though it were the obvious move.
_ROUTE_NOTE = (
    "\n\nCompression is only ONE of four routes to that size, and the only "
    "one that costs specificity. The others are pruning rules that do not "
    "earn their place (\"{prune_test}\" {best_practices}), moving "
    "area-specific rules behind `paths:` frontmatter so they load only for "
    "matching files, and escalating a must-always-run instruction to a hook. "
    "Each removes tokens without making any surviving instruction vaguer, and "
    "summarize performs NONE of them; it names them.\n\n"
    "Each candidate carries a `reduction_route` diagnosing which it wants. "
    "That is read from whether its prose is written as discrete directives "
    "(rules accumulated: prune or scope it) or as running paragraphs "
    "(padding: a genuine compression candidate). That is a measurement of "
    "the SHAPE of the prose, never of which rules earn their place. It is "
    "withheld rather than guessed where a file is too small to read a shape "
    "from.\n\n"
    "Applying stays a dry-run until `--go`, because a human reading the diff "
    "is the only check on meaning. The structure gate verifies that code "
    "blocks, tables and tags came back verbatim; it does not read the "
    "prose.\n\n"
    "COVERAGE: the figure above prices ONLY what compression recovers. It "
    "is not the size of the opportunity. A SECOND operation is now performed "
    "and priced separately in `relocation_past_overspend_usd` / "
    "`relocation_past_overspend_tokens`. It relocates a whole REFERENCE "
    "section (a module inventory, an API surface, a directory layout) out of "
    "an always-loaded file into a linked one, leaving a followable pointer "
    "behind.\n\n"
    "That is a pure MOVE with no semantic loss at all, which is why it is "
    "the safer of the two. Compression rewrites prose and can erode a "
    "modifier that no structural check would catch; nothing here is "
    "rewritten or deleted.\n\n"
    "**The two must never be added together.** A section relocated out of a "
    "file is no longer there to be compressed, so summing them would price "
    "the same text twice (Critical Rule 27). They are alternatives to pick "
    "between, per file.\n\n"
    "The relocation figure covers only sections a validated classifier is "
    "confident about, and it is deliberately conservative. The cost of "
    "moving an instruction out of the file that carries it is a silent "
    "correctness bug. The cost of leaving a reference section in place is a "
    "missed saving. So ambiguity always resolves to leaving it alone.\n\n"
    "Measured against a hand-labelled set of real sections, it moved "
    "nothing it should not have and declined a majority of what it could "
    "have moved. The remaining two routes (pruning rules that do not earn "
    "their place, and path-scoping area-specific ones behind `paths:` "
    "frontmatter) recover real tokens that are still unmeasured, not zero. "
    "They are the part this analyzer does not perform. Pricing an operation "
    "the product cannot carry out would be a ceiling the user could "
    "disprove."
)


def _ratio_basis(ratio: float, observed: bool, samples: int, gate_failures: int = 0) -> str:
    """How much of each file the figure assumes goes away, stated truthfully."""
    if observed:
        head = _RATIO_OBSERVED.format(pct=ratio * 100, samples=samples)
    else:
        head = _RATIO_TARGET.format(
            pct=ratio * 100, samples=samples, needed=MIN_OBSERVED_SAMPLES,
            prior_n=UNMEASURED_PRIOR_SAMPLES, lo=UNMEASURED_PRIOR_RANGE[0],
            hi=UNMEASURED_PRIOR_RANGE[1], target=DEFAULT_TARGET_RATIO,
        )
    if gate_failures > 0:
        head += _RATIO_GATE_FAILURES.format(failures=gate_failures)
    return head + _LINE_TARGET_NOTE.format(
        lines=PUBLISHED_LINE_TARGET, quote=PUBLISHED_LINE_TARGET_QUOTE,
        source=PUBLISHED_LINE_TARGET_SOURCE,
    ) + _ROUTE_NOTE.format(
        prune_test=PRUNE_TEST_QUOTE, best_practices=BEST_PRACTICES_SOURCE,
    )

#: Population half of the basis (Critical Rule 14): which files were even looked
#: at. Stated separately because the honest answer changes with the corpus — how
#: many repos the window worked in, and how many of them are gone from disk.
_POPULATION_CORPUS = (
    "Project-scope files are scanned across the {roots:,} project root(s) "
    "this window's sessions actually recorded working in, not just the "
    "directory this process runs from.\n\n"
    "Sometimes several of those roots hold the same file in the same place, "
    "and telemetry attributes them all to the same repo. A git worktree "
    "carries its repo's name, so every checkout of one `CLAUDE.md` matches "
    "the same sessions. When that happens, the file is charged ONCE, at the "
    "copy with the most observed loading sessions, so checkouts cannot "
    "multiply the figure.\n\n"
    "{vanished}"
)
_POPULATION_CWD = (
    "No session in this window recorded a working directory, so "
    "project-scope files were scanned only in the directory this process "
    "runs from. Every other repo's always-resident files are missing from "
    "these totals rather than counted as zero."
)
_VANISHED_NONE = "Every recorded root still exists on disk."
_VANISHED_SOME = (
    "{vanished:,} further recorded root(s) no longer exist on disk and were "
    "scanned as nothing. Their files cannot be read, so no figure is quoted "
    "for them, and these totals understate the corpus by that much."
)
#: Appended whenever an explicitly drawn filesystem scope excluded recorded
#: roots. "Confined on purpose" is a different statement from "not observed",
#: and a basis that cannot tell them apart understates the corpus silently.
_CONFINED_SOME = (
    "\n\nA further {out_of_scope:,} recorded root(s) fall outside the "
    "filesystem scope this run was given and were deliberately not scanned. "
    "These totals describe the scoped corpus only."
)
#: The population statement when a scope was drawn and NOTHING the window
#: recorded survives it. Distinct from `_POPULATION_CWD`: sessions did record
#: working directories, the scope simply excluded all of them, and saying "no
#: session recorded one" would be a plain misstatement of why the scan is
#: narrow (root anti-pattern 22).
_POPULATION_CONFINED = (
    "Every one of the {out_of_scope:,} project root(s) this window's "
    "sessions recorded falls outside the filesystem scope this run was "
    "given. So project-scope files were scanned only within that scope. The "
    "recorded repos' always-resident files are excluded from these totals "
    "by the scope, not counted as zero."
)


def _population_basis(roots: "ResolvedRoots | None", scanned: int) -> str:
    """How the scanned file population was arrived at, stated truthfully."""
    if roots is None or scanned <= 0:
        # Two different reasons the corpus scan produced nothing to scan, and
        # they must not share a sentence: an explicit scope excluded every
        # recorded root, versus no root was ever recorded.
        if roots is not None and roots.out_of_scope:
            return _POPULATION_CONFINED.format(out_of_scope=roots.out_of_scope)
        return _POPULATION_CWD
    vanished = (
        _VANISHED_SOME.format(vanished=roots.vanished) if roots.vanished
        else _VANISHED_NONE
    )
    confined = (
        _CONFINED_SOME.format(out_of_scope=roots.out_of_scope)
        if roots.out_of_scope else ""
    )
    return _POPULATION_CORPUS.format(roots=scanned, vanished=vanished) + confined


def _estimate_basis(
    invocations: "InvocationCounts | None",
    roots: "ResolvedRoots | None" = None,
    roots_scanned: int = 0,
    ratio: float = UNMEASURED_PRIOR_RATIO,
    ratio_observed: bool = False,
    ratio_samples: int = 0,
    ratio_gate_failures: int = 0,
) -> str:
    """The basis string with the evidence actually used spelled out.

    Critical Rule 14: the basis must state the arithmetic truthfully, so it
    names the observed invocation total rather than the mechanism alone, and
    the file population rather than implying the whole machine was scanned.
    """
    from tokenjam.core.summarize.invocations import INVOCATION_SOURCE

    if invocations is None or not invocations.observed:
        source = (
            f"{INVOCATION_SOURCE}: none available in this environment, so no "
            "on-demand file carries a window figure"
        )
    else:
        source = (
            f"{INVOCATION_SOURCE}; {invocations.total_invocations:,} invocation(s) "
            f"observed across {invocations.sessions_scanned:,} transcript(s)"
        )
    return SUMMARIZE_ESTIMATE_BASIS.format(
        invocation_source=source,
        population=_population_basis(roots, roots_scanned),
        ratio=_ratio_basis(ratio, ratio_observed, ratio_samples, ratio_gate_failures),
    )

# Mandatory caveat (Rule 14) — carried as the dataclass default like the other
# recoverable findings' caveats (MODEL_DOWNGRADE_CAVEAT etc.) so no surface can
# drop it. Names summary's ONE risk: structure is guaranteed (restore-by-id),
# meaning is not.
SUMMARIZE_HONESTY_CAVEAT = (
    "Structure is guaranteed; meaning may change, and nothing automatic "
    "checks it. The structure gate verifies that code blocks, tables and "
    "tags came back verbatim; it does not read the prose.\n\n"
    "Review each rewrite's diff before applying, and look specifically at "
    "MODIFIERS. Across verified rewrites of real instruction files, no "
    "instruction was dropped and nothing was invented. But qualifiers "
    "eroded: an 'only' deleted from a constraint, a 'can go stale' hardened "
    "into 'goes stale', a 'the rule exists to prevent' flattened to 'the "
    "rule prevents'. Each is small, changes what the rule actually says, "
    "and is invisible to a structural check."
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
    #: How this file reaches the model: ``always`` (whole body re-sent every
    #: session) vs ``skill`` / ``command`` / ``agent`` (frontmatter always,
    #: body only on invocation). See ``core/summarize/load_semantics``.
    load_class: str = "always"
    #: ``est_tokens_saved`` split the same way, so a consumer can see WHICH
    #: half of the file the window figure came from.
    always_resident_tokens_saved: int = 0
    on_demand_tokens_saved: int = 0
    #: Source size of the always-resident portion — the whole file for an
    #: ALWAYS-class one, the measured frontmatter for an on-demand one.
    always_resident_chars: int = 0
    #: How many times this file was OBSERVED being invoked in the window.
    #: ``None`` means "not measured" (no transcript corpus) — never 0, which
    #: is the real, priceable answer "it was never invoked".
    invocations: int | None = None
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
    #: Which route to a smaller file this candidate wants — see
    #: ``core/summarize/route``. ``prune`` means the file is long because rules
    #: accumulated, so compressing it would shorten each surviving rule instead
    #: of removing any; ``compress`` means the prose is genuinely padded;
    #: ``undiagnosed`` means the file was too small to read a shape from and no
    #: route is guessed. The saving stays the same either way — this changes
    #: what the product OFFERS, never what it claims.
    reduction_route: str = ""
    #: Share of prose words in discrete directives — the evidence for the route.
    directive_share: float = 0.0
    #: One-time always-resident token reduction available by RELOCATING this
    #: file's reference sections into a non-loaded document, and what that is
    #: worth over the window on the SAME session/call multiplier the
    #: compression figures use. An ALTERNATIVE operation on overlapping text,
    #: never an addition: relocating a section and then compressing it prices
    #: the same tokens twice (Critical Rule 27).
    relocatable_tokens: int = 0
    relocation_tokens_window: int | None = None
    relocation_usd: float | None = None


@dataclass
class SummarizeFinding:
    """Filesystem-derived summarize opportunity, on the #111 recoverable contract.

    ``past_overspend_tokens`` and ``past_overspend_usd`` are on
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
    past_overspend_usd: float | None = None
    past_overspend_tokens: int | None = None
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
    # The observed inputs behind `past_overspend_usd`, carried so the
    # figure is inspectable rather than a black box: how many sessions the
    # window held, how many calls each made on average (every one of which
    # re-sends these files), and which models the rate blend came from.
    sessions_examined: int = 0
    calls_per_session: float | None = None
    rate_basis: str = ""
    #: Whether the on-demand half of the model had any evidence at all. False
    #: means no Claude Code transcript corpus was readable, so every
    #: skill/command/agent candidate degrades to no window figure rather than
    #: being priced as if it were never invoked.
    invocations_observed: bool = False
    #: Total invocation events observed and how many transcripts were read for
    #: them — the inputs behind the on-demand term, kept inspectable for the
    #: same reason ``sessions_examined``/``calls_per_session`` are.
    invocations_total: int = 0
    transcripts_examined: int = 0
    #: How many project roots the filesystem scan covered — the observed
    #: working directories of the window's own sessions, not the one directory
    #: the process runs from. Carried for the same reason
    #: ``sessions_examined`` is: the population behind the figure must be
    #: inspectable, not implied.
    project_roots_scanned: int = 0
    #: Recorded working directories that no longer exist on disk. They are
    #: scanned as nothing (there is no file to read and no fix to offer), so
    #: this is the size of the finding's known blind spot, not a figure.
    project_roots_vanished: int = 0
    #: How many candidate files were dropped as byte-identical COPIES of a file
    #: already counted (a git worktree duplicates its repo's `CLAUDE.md` once
    #: per worktree). Each distinct content is charged once, at the copy with
    #: the most observed loading sessions.
    duplicate_copies_collapsed: int = 0
    #: The prose ratio behind every per-file reduction here, and whether it was
    #: MEASURED from verified rewrites on this machine or is the target the
    #: rewriter is merely asked for. The distinction is load-bearing: nothing
    #: enforces the target, and rewrites observed while building this delivered
    #: materially less than it, so an unobserved figure is an upper bound rather
    #: than an expectation. That framing stays internal — the user-facing basis
    #: says the target is unverified without asserting anything about rewrites
    #: on THEIR machine, where by construction none have happened yet.
    prose_ratio: float = UNMEASURED_PRIOR_RATIO
    prose_ratio_observed: bool = False
    #: Structure-checked rewrites the ratio was derived from (0 when assumed).
    prose_ratio_samples: int = 0
    #: Rewrites attempted here that FAILED the structure gate. Excluded from the
    #: ratio (never a usable outcome) but carried, because a sample made mostly
    #: of failures must not read as a clean measurement — and because a file
    #: that cannot be safely compressed at all is a real finding about it.
    prose_ratio_gate_failures: int = 0
    #: The published size target the fix aims at for an always-resident
    #: instruction file, and where it comes from. Carried so no surface has to
    #: restate the number itself; it is Anthropic's guidance, not tokenjam's,
    #: and it governs what the rewriter is ASKED for, never what is claimed.
    line_target: int = PUBLISHED_LINE_TARGET
    line_target_source: str = PUBLISHED_LINE_TARGET_SOURCE
    #: How many candidates want each route (``route`` -> count), so a surface
    #: can say "N of these want pruning, not compression" without walking the
    #: list. Counts only; it changes what is OFFERED, never what is claimed —
    #: a prune-route candidate keeps its full token and dollar figure, because
    #: the tokens are recoverable by whichever route the user picks.
    candidates_by_route: dict[str, int] = field(default_factory=dict)
    #: The RELOCATION figures, carried beside the compression ones and
    #: deliberately NOT summed into ``past_overspend_usd`` /
    #: ``past_overspend_tokens``. The two operations act on overlapping text —
    #: a section relocated out of a file is no longer there to be compressed —
    #: so adding them would double-count exactly the way Critical Rule 27
    #: forbids. They are alternatives the user picks between, and relocation is
    #: the safer of the two by construction: it moves text and rewrites none, so
    #: it cannot erode a modifier the way a rewrite can.
    #:
    #: Same basis as their compression counterparts (Critical Rule 28): the same
    #: sessions x reads-per-session event count, one counted and one priced, and
    #: ``None`` on the same "no loading session observed" condition rather than a
    #: zero.
    relocation_past_overspend_usd: float | None = None
    relocation_past_overspend_tokens: int | None = None
    #: One-time sum of the per-file relocatable reduction — the relocation
    #: counterpart of ``file_reduction_tokens``, on the same one-time basis.
    relocation_file_reduction_tokens: int | None = None
    #: How many candidate files have any relocatable reference section at all.
    #: Deliberately reported: on a corpus where this is small the ceiling above
    #: is concentrated in a handful of files, which is a materially different
    #: claim from a broad one and a reader cannot tell them apart from a total.
    relocation_files: int = 0


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

    ``sessions_total`` is every DISTINCT session in the window (what a
    global-scope file is loaded by) — counted once across the whole window, not
    summed from the per-``agent_id`` groups below. Summing them was the defect:
    a session that touched two repos appears in both groups, so every
    multi-repo session was counted once per repo and inflated the window total
    that every global-scope candidate is priced against.
    ``sessions_by_repo``/``calls_by_repo`` narrow that for a project-scope
    file; those per-repo counts stay per-``agent_id`` on purpose — a session
    that touched two repos really does load both repos' `CLAUDE.md`.
    ``calls_per_session`` is the window-WIDE average number
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
        persona_scope=ctx.persona_scope,
    )
    if rates is None:
        return None
    clauses = ["name = 'gen_ai.llm.call'", "start_time >= $1", "start_time < $2"]
    params: list[Any] = [ctx.since, ctx.until]
    if ctx.agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(ctx.agent_id)
    # The persona POPULATION scope: this load profile is the multiplier the
    # finding's dollars are built on, so it has to count the same sessions the
    # rest of the pass does. See `core/persona_scope.py`.
    add_persona_clause(clauses, ctx.persona_scope)
    where = " AND ".join(clauses)
    try:
        rows = ctx.conn.execute(
            "SELECT agent_id, COUNT(DISTINCT session_id), COUNT(*) "
            "FROM spans WHERE " + where + " GROUP BY agent_id",
            params,
        ).fetchall()
        # Counted separately, NOT summed from `rows`: a session that touched
        # two agent_ids is one session in the window but two rows above.
        totals = ctx.conn.execute(
            "SELECT COUNT(DISTINCT session_id), COUNT(*) FROM spans WHERE " + where,
            params,
        ).fetchone()
    except Exception:
        logger.debug("summarize analyzer: load-profile query failed", exc_info=True)
        return None

    sessions_by_repo: dict[str, int] = {}
    calls_by_repo: dict[str, int] = {}
    sessions_total = int((totals or (0, 0))[0] or 0)
    calls_total = int((totals or (0, 0))[1] or 0)
    # DO NOT "fix" the per-repo counts below to match `sessions_total`. They
    # are deliberately NOT distinct across repos, and that is correct: a
    # session that touched two repos really does load BOTH repos' `CLAUDE.md`,
    # so it belongs to both per-repo counts. Only the WINDOW total had to be
    # counted once — summing these groups to get it was the defect (it made a
    # multi-repo session inflate every global candidate). The two numbers
    # answer different questions and are supposed to disagree.
    for agent_id, sessions, calls in rows:
        sessions = int(sessions or 0)
        calls = int(calls or 0)
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


def _matched_repos(path: str, profile: _LoadProfile) -> frozenset[str]:
    """The telemetry repo labels this project-scope path belongs to.

    Matched by walking the file's ancestor directory names against the repo
    labels telemetry recorded — the single derivation behind BOTH how many
    sessions load the file and how many calls each of those sessions makes, so
    the two can never drift apart. Note what it CANNOT distinguish: a git
    worktree's directory carries the same name as the repo it was cut from, so
    every copy matches the same labels. That is why two files matching the same
    labels in the same slot are treated as one chargeable file (see
    ``_copy_key``) rather than charged twice.
    """
    ancestors = {parent.name for parent in Path(path).parents if parent.name}
    return frozenset(repo for repo in profile.sessions_by_repo if repo in ancestors)


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
    return sum(
        profile.sessions_by_repo[repo] for repo in _matched_repos(path, profile)
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
    matched = _matched_repos(path, profile)
    matched_sessions = sum(profile.sessions_by_repo[repo] for repo in matched)
    if matched_sessions <= 0:
        return profile.calls_per_session
    matched_calls = sum(profile.calls_by_repo.get(repo, 0) for repo in matched)
    return matched_calls / matched_sessions


def _reads_per_session(calls_per_session: float) -> int:
    """Total times one session reads an always-on file: the first send plus
    every later re-read in that session — at least 1. Shared by
    ``_price_reduction`` and ``_tokens_saved_over_window`` so the dollar and
    token figures price the exact same event count."""
    return max(round(calls_per_session), 1)


def _price_reduction(
    resident_tokens: int,
    on_demand_tokens: int,
    sessions: int,
    calls_per_session: float,
    invocations: int,
    rates: RateProfile,
) -> float | None:
    """What removing this file's prose is worth over the window, by load class.

    The always-resident part (a whole `CLAUDE.md`, or a skill/command/agent's
    frontmatter) is worth the reduction on each loading session, sent once at
    the input rate and re-read on that session's every later call at the
    cache-read rate. The on-demand part (a skill/command/agent BODY) is worth
    the reduction once per OBSERVED invocation, at the input rate — it is not
    in context at all until then.

    ``calls_per_session`` is the candidate's OWN repo's average (see
    ``_repo_calls_per_session``) for a project-scope file, or the window-wide
    average for a global-scope one — never a blend of the two.

    ``None`` when no session loads the file — the saving is real but this
    window carries no evidence of its size, and a zero would misreport that as
    "worth nothing". A file that IS loaded but was never invoked returns a real
    figure (its frontmatter term), which is the honest answer.
    """
    if sessions <= 0:
        return None
    total = 0.0
    if resident_tokens > 0:
        rereads = _reads_per_session(calls_per_session) - 1
        total += rates.cost_of(float(resident_tokens), rereads) * sessions
    if on_demand_tokens > 0 and invocations > 0:
        total += rates.cost_of(float(on_demand_tokens), 0) * invocations
    return total


def _tokens_saved_over_window(
    resident_tokens: int,
    on_demand_tokens: int,
    sessions: int,
    calls_per_session: float,
    invocations: int,
) -> int | None:
    """Actual token volume this file's reduction is worth across the window —
    the EXACT event count ``_price_reduction`` prices in dollars, so the two
    fields stay on the same basis (Critical Rule 28).

    ``None`` on the same "no evidence" condition ``_price_reduction`` returns
    ``None`` for: no session in the window was observed loading this file.
    """
    if sessions <= 0:
        return None
    total = 0
    if resident_tokens > 0:
        total += resident_tokens * _reads_per_session(calls_per_session) * sessions
    if on_demand_tokens > 0 and invocations > 0:
        total += on_demand_tokens * invocations
    return round(total)


def _load_split(candidate: Any) -> tuple[int, int]:
    """A scan candidate's reduction split into (always-resident, on-demand).

    Reads the split ``core/summarize/candidates`` measured on the real text.
    A candidate object that predates those fields — a stub from another caller,
    or a hand-built fixture — carries neither half, and its whole reduction
    falls back to always-resident (the pre-split behaviour) rather than
    silently dropping to zero. The two halves of a real candidate can never
    BOTH be zero while the whole is not: each is floored independently from a
    slice of the same prose, and a candidate needs 100+ prose words to exist.
    """
    total = int(getattr(candidate, "est_tokens_saved", 0) or 0)
    resident = int(getattr(candidate, "always_resident_tokens_saved", 0) or 0)
    on_demand = int(getattr(candidate, "on_demand_tokens_saved", 0) or 0)
    if resident <= 0 and on_demand <= 0:
        return total, 0
    return resident, on_demand


def _is_measured_zero(candidate: SummarizeCandidate) -> bool:
    """True when this file was MEASURED to cost nothing over the window.

    A `.claude/commands/x.md` with no frontmatter that was never invoked is
    resident in no session and delivered on no call: not a candidate at all,
    rather than a candidate worth `$0.00`. Rendering the zero would invite the
    reader to think the analyzer looked for a saving and found none, when the
    truth is there is nothing here to summarize this window (Critical Rule 22 —
    never show a figure the user cannot act on).

    Deliberately keyed on a MEASURED zero, never on a missing one: a candidate
    whose window figure degraded to ``None`` (no loading session observed, or
    no transcript corpus) is kept, because "not measured" is not "worth
    nothing" and suppressing it would hide a file we simply failed to price.
    """
    return (
        candidate.est_tokens_saved_window == 0
        and candidate.est_usd_saved == 0
    )


def _slot(path: str, scan_root: str) -> str:
    """A file's SLOT — its path relative to the root it was found under.

    `CLAUDE.md`, `.claude/commands/ship.md`. Two files in the same slot under
    two different roots are the same project file seen twice. Falls back to the
    bare filename when the root is unknown or is not actually a prefix.
    """
    name = Path(path).name
    if not scan_root:
        return name
    try:
        return Path(path).relative_to(scan_root).as_posix()
    except ValueError:
        return name


def _copy_key(
    candidate: SummarizeCandidate, scan_root: str, repos: frozenset[str],
) -> tuple[frozenset[str], str] | None:
    """The identity two candidates must share to be one chargeable file, or
    ``None`` for a candidate that must never be collapsed.

    Keyed on **the session population that loads the file, plus the file's
    slot** — deliberately NOT on the file's content, which is not sufficient
    and not necessary:

    * NOT sufficient. A corpus-wide scan visits every repo the window worked
      in, and a git worktree is a copy of its repo, so one repo's `CLAUDE.md`
      appears once per checkout — and those checkouts DRIFT, because each sits
      on its own branch. `_sessions_loading` matches all of them to the same
      repo by ancestor directory name and charges every one against the same
      sessions. Content hashing collapses only the copies that happen to be
      byte-equal right now and leaves every drifted one multiplying a single
      file's cost, which is money the user cannot act on (Critical Rule 22).
    * NOT necessary. Two byte-identical files in repos with DIFFERENT session
      populations (a `CLAUDE.md` copied between two projects) genuinely are
      loaded twice and genuinely cost twice; content hashing would wrongly
      merge them and understate.

    Two genuinely different files can only collapse when they occupy the same
    slot AND are priced against the identical session population — in which
    case this analyzer cannot tell them apart in the first place, and charging
    both would be counting one repo's `CLAUDE.md` twice. Collapsing is then the
    conservative direction, and it is reported (`duplicate_copies_collapsed`).

    A global-scope file is never collapsed: those paths are already unique and
    every one of them is a distinct file under `~/.claude`.
    """
    if candidate.scope == "global" or not repos:
        return None
    return (repos, _slot(candidate.path, scan_root))


def _is_nested(a: str, b: str) -> bool:
    """Whether one root path contains the other (or they are the same path).

    This is what separates a genuinely STACKED pair from a duplicate. A
    meta-repo's `CLAUDE.md` and the `CLAUDE.md` of a sub-repo checked out
    inside it are both loaded — the harness reads the working directory's file
    and its ancestors' — and their roots are nested. A worktree cut from that
    sub-repo lives in a disjoint tree; it is the SAME file seen twice.
    """
    pa, pb = Path(a), Path(b)
    return pa == pb or pa in pb.parents or pb in pa.parents


def _collapse_copies(
    scored: list[
        tuple["tuple[frozenset[str], str] | None", str, SummarizeCandidate]
    ],
) -> tuple[list[SummarizeCandidate], int]:
    """Charge each distinct file once. Returns ``(kept, collapsed_count)``.

    Within one group — same loading population, same slot — roots are visited
    shallowest first and a candidate is kept only while it stays nested with
    every candidate already kept. So an ancestor/descendant stack survives
    intact, and the first root that sits in a parallel tree (a worktree, a
    second clone) is dropped as a copy along with everything below it.

    Shallowest-first also decides WHICH copy is reported: the outermost
    checkout rather than a worktree cut from it, which is the file the user can
    actually act on (Critical Rule 22). Ordering by size instead would let an
    incidental checkout both inflate the figure and name itself as the fix.
    Every candidate in a group shares one session population, so none of them
    can be better or worse evidenced than the others; the path is the final
    tiebreak so the choice is deterministic run to run.

    A ``None`` key means "never collapse this one" — an unknown identity is not
    evidence of sameness.
    """
    groups: dict[tuple[frozenset[str], str], list[tuple[str, SummarizeCandidate]]] = {}
    kept: list[SummarizeCandidate] = []
    collapsed = 0
    for key, scan_root, candidate in scored:
        if key is None:
            kept.append(candidate)
            continue
        groups.setdefault(key, []).append((scan_root, candidate))

    for members in groups.values():
        members.sort(key=lambda m: (len(Path(m[0]).parts), m[0], m[1].path))
        roots_kept: list[str] = []
        for scan_root, candidate in members:
            if all(_is_nested(scan_root, other) for other in roots_kept):
                roots_kept.append(scan_root)
                kept.append(candidate)
            else:
                collapsed += 1
    return kept, collapsed


def _invocation_counts(ctx: AnalyzerContext, scope) -> InvocationCounts:
    """Observed skill/command/agent invocations for the window.

    Reads the transcript corpus under the RESOLVED scope's projects root — the
    same root `deadweight` and `relearn` are handed — so the sessions counted
    here, and the working directories derived from them, describe the corpus
    the run was scoped to. Left unscoped this walked `~/.claude/projects`
    regardless, which is the escape the scope contract exists to close.

    Never raises: any failure degrades to ``observed=False``, which makes every
    on-demand candidate report no window figure rather than one priced as if
    nothing had ever been invoked.
    """
    from tokenjam.core.summarize.invocations import count_invocations
    from tokenjam.core.transcript_cache import default_cache_dir

    try:
        return count_invocations(
            ctx.since, ctx.until,
            projects_root=scope.projects_root,
            cache_dir=default_cache_dir(ctx.config),
        )
    except Exception:
        logger.debug("summarize analyzer: invocation scan failed", exc_info=True)
        return InvocationCounts()


def _scan_boundary(scope) -> "Path | None":
    """The outermost directory a session-derived project root may sit in.

    The two scoping mechanisms here answer different questions and compose in
    one direction only: the scope is the BOUNDARY, the recorded cwds are the
    SELECTION within it. A transcript living under a scoped projects root can
    still record a working directory anywhere on disk, so confining the
    transcript corpus does not by itself confine the roots derived from it.

    `None` — no boundary — when no root was drawn explicitly. The default
    projects root is not a boundary anyone asked for, and treating it as one
    would filter the observed population down to whatever sits under
    `~/.claude/projects`, which holds no source project at all: the scan would
    collapse back to the single-cwd population it exists to replace. Once
    `--projects-root` or the env var HAS drawn a root, the scope's home is the
    boundary, the same unit `_project_scan_root` measures the cwd against.
    """
    if scope.source not in ("flag", "env"):
        return None
    return scope.home.expanduser()


def _project_scan_root(scope) -> "Path | None":
    """Where the candidate scan's PROJECT half may start looking.

    `None` means "the process's cwd", the behavior every run had before the
    scan was scoped at all — and the behavior an unscoped run must keep
    byte-for-byte, since the default projects root (`~/.claude/projects`)
    contains no source project and rooting there would empty the scan.

    Once a root has been drawn explicitly (`--projects-root`, or the env var),
    the cwd is only an acceptable starting point while it is INSIDE that
    scope; from anywhere else it reaches straight past the boundary the caller
    asked for, so the scan starts at the scope's home instead.
    """
    if scope.source not in ("flag", "env"):
        return None
    home = scope.home.expanduser()
    cwd = Path.cwd()
    try:
        cwd.relative_to(home)
    except ValueError:
        return home
    return cwd


@register("summarize")
def run(ctx: AnalyzerContext) -> None:
    """Attach a SummarizeFinding: catalog-default candidates + per-call token saving.

    Reasons over the filesystem (config-driven scan), not `ctx.conn`. The scan is
    catalog-default (a handful of known prompt files) so it's cheap enough for the
    polling Overview; a filesystem hiccup never breaks the optimize report.
    """
    finding = SummarizeFinding(estimate_basis=_estimate_basis(None))

    # Window-guard: a dead telemetry window has no calls to realize a per-call
    # saving against, so — like every recoverable finding — contribute nothing
    # rather than leak a filesystem figure into the empty-window overlay (#211).
    # Also skips the scan entirely on an idle window.
    if ctx.summary.total_tokens == 0:
        ctx.report.findings["summarize"] = finding
        return

    from tokenjam.core.optimize.scope import resolve_analyzer_scope
    from tokenjam.core.agent_config import store_for
    from tokenjam.core.summarize.candidates import list_candidates

    scope = ctx.scope if ctx.scope is not None else resolve_analyzer_scope(ctx.config)
    if not scope.enabled:
        # Same reason the sibling filesystem analyzers bail: the catalog scan
        # reads always-loaded prompt files off the operator's real home, which
        # an explicit `--db` was asked to isolate away from.
        ctx.report.filesystem_scan_skipped_reason = scope.reason
        ctx.report.findings["summarize"] = finding
        return

    # Observed invocation counts for the on-demand half of the model, plus the
    # working directories the window's sessions recorded. Scanned once for the
    # whole finding, off the same corpus + persistent parse cache
    # `deadweight`/`relearn` already use — and off the SAME resolved scope, so
    # the sessions this finding reasons about are the ones the run was scoped
    # to rather than whatever `~/.claude/projects` happens to hold. Runs BEFORE
    # the filesystem scan because the scan's project population is derived from
    # those cwds.
    invocations = _invocation_counts(ctx, scope)
    roots = resolve_roots(invocations.session_cwds, within=_scan_boundary(scope))

    # How much of each file the reduction assumes goes away. The target ratio is
    # what the rewriter is ASKED for and nothing enforces it, so a measured
    # ratio from verified rewrites on this machine supersedes it whenever one
    # exists; the basis says which is in force either way (Critical Rule 14).
    # The FALLBACK is the measured prior, not the target: the target is what the
    # rewriter is ASKED for, and estimating at an ask nothing enforces is the
    # defect this analyzer's basis exists to disclose. Two independent runs on
    # real instruction files put the delivered ratio ~10x (in savings terms)
    # away from the ask, so the ask is not a conservative default, it is a
    # flattering one.
    measured_ratio, ratio_samples = observed_prose_ratio(ctx.config)
    ratio = measured_ratio if measured_ratio is not None else UNMEASURED_PRIOR_RATIO
    ratio_gate_failures = gate_failed_attempts(ctx.config)

    try:
        # read-only, never writes. Three scope inputs, one resolved scope:
        #
        # * `home` scopes the catalog's `~`-rooted global paths.
        # * `project_roots` is the project population: every repo the window
        #   actually worked in, not the one directory this process happens to
        #   sit in — which left every other repo in the corpus contributing
        #   nothing. Already confined to the scope boundary by
        #   `_scan_boundary` above, so it can never reach past a root the
        #   caller drew.
        # * `project_root` is the FALLBACK starting point for when no observed
        #   root survives (no corpus, or every recorded cwd outside the
        #   boundary). It carries upstream's rule that a cwd outside an
        #   explicitly drawn scope is replaced by the scope's home, so the
        #   fallback cannot escape the boundary either.
        #
        # Roots are deliberately NOT passed as `path`: that widens the net to
        # every `*.md` in a directory and would price `README.md` /
        # `CHANGELOG.md` as if the harness auto-loaded them (Critical Rule 22).
        # The scan's file population is INGESTED (`core/agent_config`) and read
        # back from the store rather than re-globbed at analysis time — same
        # store the deadweight and trim analyzers populate, so one pass leaves
        # behind one description of the config surface instead of three
        # independent walks leaving behind nothing.
        scan = list_candidates(
            config=ctx.config, home=scope.home,
            project_roots=roots.roots or None,
            project_root=_project_scan_root(scope),
            ratio=ratio,
            store=store_for(getattr(ctx, "conn", None), getattr(ctx, "write_lock", None)),
        )
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
    finding.estimate_basis = _estimate_basis(
        invocations, roots, scan.project_roots_scanned,
        ratio, measured_ratio is not None, ratio_samples, ratio_gate_failures,
    )
    finding.project_roots_scanned = scan.project_roots_scanned
    finding.project_roots_vanished = roots.vanished
    finding.prose_ratio = round(ratio, 4)
    finding.prose_ratio_observed = measured_ratio is not None
    finding.prose_ratio_samples = ratio_samples
    finding.prose_ratio_gate_failures = ratio_gate_failures
    finding.invocations_observed = invocations.observed
    finding.invocations_total = invocations.total_invocations
    finding.transcripts_examined = invocations.sessions_scanned

    scored: list[
        tuple["tuple[frozenset[str], str] | None", str, SummarizeCandidate]
    ] = []
    for c in scan.candidates:
        if c.est_tokens_saved <= 0:
            continue
        load_class = getattr(c, "load_class", load_semantics.ALWAYS)
        on_demand = load_class in load_semantics.ON_DEMAND_CLASSES
        resident_tokens, on_demand_tokens = _load_split(c)
        relocatable = int(getattr(c, "relocatable_tokens", 0) or 0)
        sessions = _sessions_loading(c.path, c.scope, profile) if profile else 0
        calls_per_session = (
            _repo_calls_per_session(c.path, c.scope, profile) if profile else 0.0
        )
        # Rule 28 corollary (a): "never invoked" (a measurement) prices the
        # body at zero; "no corpus to look in" is not a measurement, so the
        # whole candidate degrades rather than being quietly priced as zero.
        measured_invocations: int | None = None
        if not on_demand:
            measured_invocations = 0
        elif invocations.observed:
            measured_invocations = invocations.get(
                getattr(c, "invocation_key", "") or "",
            )
        priceable = profile is not None and measured_invocations is not None
        candidate = SummarizeCandidate(
            path=c.path,
            kind="prompt" if c.is_prompt else "other",
            scope=c.scope,
            est_tokens_saved=c.est_tokens_saved,
            total_chars=c.total_chars,
            reduction_pct=_reduction_pct(c.est_tokens_saved, c.total_chars),
            load_class=load_class,
            always_resident_tokens_saved=resident_tokens,
            on_demand_tokens_saved=on_demand_tokens,
            always_resident_chars=int(
                getattr(c, "always_resident_chars", 0) or 0,
            ) or (c.total_chars if not on_demand else 0),
            invocations=measured_invocations if on_demand else None,
            sessions_loading=sessions,
            est_usd_saved=(
                _price_reduction(
                    resident_tokens, on_demand_tokens, sessions,
                    calls_per_session, measured_invocations or 0, profile.rates,
                )
                if priceable and profile is not None else None
            ),
            est_tokens_saved_window=(
                _tokens_saved_over_window(
                    resident_tokens, on_demand_tokens, sessions,
                    calls_per_session, measured_invocations or 0,
                )
                if priceable else None
            ),
            reduction_route=str(getattr(c, "reduction_route", "") or ""),
            directive_share=float(getattr(c, "directive_share", 0.0) or 0.0),
            relocatable_tokens=relocatable,
            # Routed through the SAME multiplier the compression figures use, so
            # the two operations are directly comparable and the token and
            # dollar fields count the same events (Critical Rule 28). Relocation
            # only ever touches always-resident text, so the on-demand term is
            # structurally zero rather than merely unobserved.
            relocation_tokens_window=(
                _tokens_saved_over_window(
                    relocatable, 0, sessions, calls_per_session, 0,
                )
                if priceable and relocatable > 0 else None
            ),
            relocation_usd=(
                _price_reduction(
                    relocatable, 0, sessions, calls_per_session, 0, profile.rates,
                )
                if priceable and relocatable > 0 and profile is not None else None
            ),
        )
        if _is_measured_zero(candidate):
            continue
        repos = _matched_repos(c.path, profile) if profile else frozenset()
        scan_root = str(getattr(c, "scan_root", "") or "")
        scored.append((_copy_key(candidate, scan_root, repos), scan_root, candidate))
    # One charge per distinct file, however many roots hold a copy of it.
    candidates, collapsed = _collapse_copies(scored)
    candidates.sort(key=lambda c: (-(c.est_usd_saved or 0.0), c.path))
    finding.duplicate_copies_collapsed = collapsed
    finding.candidates = candidates
    finding.files = len(finding.candidates)
    route_counts: dict[str, int] = {}
    for cand in finding.candidates:
        if cand.reduction_route:
            route_counts[cand.reduction_route] = route_counts.get(cand.reduction_route, 0) + 1
    finding.candidates_by_route = route_counts
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
        finding.past_overspend_tokens = sum(window_tokens) if window_tokens else None
        finding.past_overspend_usd = round(sum(priced), 6) if priced else None
        # The relocation aggregates, on the same degrade-symmetrically rule and
        # deliberately kept OUT of the compression sums above (Critical Rule 27
        # — the two operations act on overlapping text and are alternatives,
        # never addends).
        reloc_one_time = sum(c.relocatable_tokens for c in finding.candidates)
        finding.relocation_file_reduction_tokens = reloc_one_time or None
        finding.relocation_files = sum(
            1 for c in finding.candidates if c.relocatable_tokens > 0
        )
        reloc_window = [
            c.relocation_tokens_window for c in finding.candidates
            if c.relocation_tokens_window is not None
        ]
        reloc_priced = [
            c.relocation_usd for c in finding.candidates if c.relocation_usd is not None
        ]
        finding.relocation_past_overspend_tokens = (
            sum(reloc_window) if reloc_window else None
        )
        finding.relocation_past_overspend_usd = (
            round(sum(reloc_priced), 6) if reloc_priced else None
        )
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
