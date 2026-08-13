"""THE registry of values that must have exactly one derivation.

**The defect class.** A published number, window or classification gets
computed independently in two places, nothing forces the two to agree, and
they drift. It has shipped roughly ten times in this product, always fixed by
hand with a bespoke test written after the fact: the rollup population, the
write budget, the report window, the persona gate, the write-apply target.
Each fix was correct and each one left the general shape of the defect
unguarded, because nothing forced the NEXT shared value through the same
discipline.

**The fix is one registry, not one test per value.** :data:`SEAMS` names every
value whose derivation is pinned to a SINGLE module, by the symbol a second
derivation would necessarily touch (a function call, or an attribute read).
``tests/unit/test_single_derivation.py`` walks the whole package once per
entry and fails if that symbol is reached from anywhere outside the module
that owns it. Adding the next shared value is a new :class:`SingleSeam` line
here — never a new AST walk, never a new test file. That is the entire point:
a design that needs a new test per value has rebuilt the problem it exists to
retire.

**Not every seam fits a symbol guard**, and forcing one is worse than not
having one — see the module docstring on :data:`BESPOKE_SEAMS` below for why
persona classification, the write-apply target and the scan-cycle anchor stay
as hand-written tests instead. This registry still names them, so a reviewer
scanning ONE file sees every pinned seam, mechanized or not, and
:func:`check_bespoke_seam` fails loudly if the test guarding one of them is
ever deleted.

**Aggregate-versus-parts is a different shape of the same defect** — see the
module docstring further down, near :data:`KNOWN_GAPS`.
"""
from __future__ import annotations

import ast
import importlib
import pathlib
from dataclasses import dataclass

import tokenjam as _pkg

#: Every module under here is in scope for a seam's offender walk. Test files
#: are exempt everywhere in this module — they legitimately construct or pin
#: the raw guarded symbol's own behaviour (see any seam's own unit test), and
#: this walk only ever inspects the SHIPPED package.
PACKAGE_ROOT = pathlib.Path(_pkg.__file__).parent


@dataclass(frozen=True)
class SingleSeam:
    """One value whose derivation may live in exactly one module.

    ``symbol`` is the bare identifier a second derivation would have to
    touch — a function name for ``kind="call"``, or an attribute name for
    ``kind="attr"`` (matched both as ``obj.symbol`` and as the string literal
    key of ``getattr(obj, "symbol", ...)``, since config reads use both
    forms in this codebase).

    ``allowed_modules`` are paths relative to :data:`PACKAGE_ROOT`, POSIX
    style (``"core/optimize/report_window.py"``). The symbol's OWN defining
    module always needs to be listed if it also USES the symbol internally
    (e.g. a factory function calling its own class constructor).
    """

    name: str
    description: str
    symbol: str
    kind: str  # "call" | "attr"
    allowed_modules: frozenset[str]
    reason: str

    def __post_init__(self) -> None:
        if self.kind not in ("call", "attr"):
            raise ValueError(f"unknown SingleSeam.kind {self.kind!r} for {self.name!r}")


def _calls_symbol(node: ast.AST, symbol: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
    return name == symbol


def _reads_attr(node: ast.AST, symbol: str) -> bool:
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return node.attr == symbol
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr":
        args = node.args
        if len(args) >= 2 and isinstance(args[1], ast.Constant):
            return args[1].value == symbol
    return False


def offenders_for(seam: SingleSeam) -> list[str]:
    """Every ``path:lineno`` outside ``seam.allowed_modules`` that reaches
    ``seam.symbol``. Empty means the seam holds.

    Walks every ``.py`` under the shipped package, in file order, so the
    result is deterministic and reviewable — a caller can paste it straight
    into a failure message.
    """
    check = _calls_symbol if seam.kind == "call" else _reads_attr
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        if rel in seam.allowed_modules:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if check(node, seam.symbol):
                offenders.append(f"{rel}:{node.lineno}")
    return offenders


#: --------------------------------------------------------------------- #
#: THE REGISTRY. Add a shared value here, never as a new test file.
#: --------------------------------------------------------------------- #
SEAMS: tuple[SingleSeam, ...] = (
    SingleSeam(
        name="rollup population",
        description=(
            "the Review inbox headline: every open cost proposal PLUS "
            "relearn's open clusters, summed once."
        ),
        symbol="past_overspend_rollup",
        kind="call",
        allowed_modules=frozenset({"core/optimize/inbox_contribution.py"}),
        reason=(
            "completeness used to be a caller CONVENTION — whoever built the "
            "input list had to remember to concatenate relearn's clusters in. "
            "cmd_quickstart's first-run screen forgot, and its own comment "
            "asserted the two totals could never disagree. "
            "gather_rollup_population() in inbox_contribution.py is now the "
            "only path that assembles both feeds before calling the raw "
            "rollup function."
        ),
    ),
    SingleSeam(
        name="window length",
        description=(
            "the trailing look-back, in days, EVERY past-overspend surface "
            "observes over — the Dashboard tiles and the Review inbox "
            "headline must resolve the same number."
        ),
        symbol="scan_window_days",
        kind="attr",
        allowed_modules=frozenset({
            "core/optimize/report_window.py",
            "core/config.py",
        }),
        reason=(
            "the Dashboard read `[optimize] scan_window_days` directly (a "
            "fixed config int) while the Review inbox read the resolved "
            "analysis span, bounded by measured history — 30 vs 69 days on "
            "a real corpus, so the six Dashboard tiles summed to roughly "
            "half the inbox headline. report_window.report_window_days() "
            "is now the only reader of the raw config field; core/config.py "
            "stays allowed because it OWNS the field's definition and "
            "default."
        ),
    ),
    SingleSeam(
        name="cycle provenance",
        description=(
            "what produced a stored analyzer artifact — the cycle id, the "
            "anchor, the observed window, the persona and the producing "
            "build — minted ONCE per pass and carried by every store that "
            "pass writes."
        ),
        symbol="CycleProvenance",
        kind="call",
        allowed_modules=frozenset({"core/optimize/cycle_provenance.py"}),
        reason=(
            "the three provenance facts were three ad-hoc conventions. "
            "`tj_build()` was called independently at every write site and "
            "again at every read site, and nothing ever COMPARED a stored "
            "stamp against the running build, so an upgrade served the "
            "previous build's cards under a fresh timestamp. The window had "
            "two spellings (`scan_since`/`scan_until` on the report, "
            "`cost_since`/`cost_until` on the cost block) with no shared type "
            "forcing them to describe the same span. And the cycle had no "
            "identity at all, so a report-derived panel could serve cycle N "
            "figures beside inbox figures from cycle N-1 with nothing able to "
            "tell them apart. `cycle_provenance.begin_cycle()` is now the one "
            "place the record is constructed; every store takes one it was "
            "handed."
        ),
    ),
    SingleSeam(
        name="rate profile",
        description=(
            "the blended $/token rate an analyzer prices its findings "
            "against."
        ),
        symbol="RateProfile",
        kind="call",
        allowed_modules=frozenset({"core/optimize/rate_profile.py"}),
        reason=(
            "RateProfile is a plain dataclass — nothing stops a second "
            "constructor call from hand-rolling a rate outside "
            "blended_rate_profile()'s own weighting. Every analyzer that "
            "prices tokens (relearn, summarize, downsize) calls the shared "
            "function; only rate_profile.py itself is allowed to construct "
            "the record it returns."
        ),
    ),
)


@dataclass(frozen=True)
class BespokeSeam:
    """A single-derivation invariant real enough to pin, but NOT expressible
    as a symbol-reachable-from-one-module check.

    Each of these was tried against :class:`SingleSeam` first and rejected
    for a concrete, stated reason — never "harder to write". Listing them
    here (rather than leaving them as orphaned tests nobody indexes) means a
    reviewer scanning this one file sees every pinned single-derivation
    invariant in the product, not just the mechanized ones, and
    :func:`check_bespoke_seam` fails loudly — not silently — if the test
    naming a seam here is ever deleted or renamed out from under it.
    """

    name: str
    description: str
    reason_not_mechanized: str
    test_module: str
    test_name: str


BESPOKE_SEAMS: tuple[BespokeSeam, ...] = (
    BespokeSeam(
        name="persona classification",
        description=(
            "a persona gate (which analyzers run, whether relearn may "
            "write) may never be resolved over ALL history."
        ),
        reason_not_mechanized=(
            "the defect is in the CALL ARGUMENTS (an unwindowed "
            "agent_persona_mix() reaching dominant_persona()), not in which "
            "module makes the call — a symbol-reachability guard can't see "
            "into a call's own arguments, only whether the call exists."
        ),
        test_module="tests.unit.test_report_window",
        test_name="test_no_surface_classifies_persona_over_all_history",
    ),
    BespokeSeam(
        name="write-apply target",
        description=(
            "the path relearn suggests as a write target, and the path the "
            "API's write guard authorizes against, resolve through the "
            "SAME call: resolve_write_scope(scope=scope).suggest_root."
        ),
        reason_not_mechanized=(
            "scope.claude_home has a legitimate SECOND, unrelated purpose "
            "(deadweight's own read-only MCP-config scope) — a symbol guard "
            "on `.claude_home` would false-positive on that call. The pin "
            "has to be scoped to the two write-target call sites "
            "specifically, which only a source-text match on the exact "
            "call shape can do."
        ),
        test_module="tests.unit.test_report_window",
        test_name="test_the_apply_target_and_the_write_guard_share_one_derivation",
    ),
    BespokeSeam(
        name="scan-cycle anchor",
        description=(
            "one provenance record per scan cycle — carrying one anchor, one "
            "window and one producing build — threaded into BOTH the report "
            "pass and the cost-proposal pass, so they measure the same "
            "instant instead of two instants seconds apart."
        ),
        reason_not_mechanized=(
            "the CONSTRUCTION of the record is mechanized (see the "
            "'cycle provenance' SingleSeam above); what a reachability guard "
            "cannot express is that scan_cycle threads ONE instance through "
            "both calls in the SAME cycle. Both recomputes legitimately mint "
            "their own when called standalone outside a cycle (`provenance` "
            "is optional by design), so the invariant is a data-flow "
            "property, not a call-site one."
        ),
        test_module="tests.unit.test_report_window",
        test_name="test_the_report_and_cost_stores_come_from_ONE_analyzer_pass",
    ),
    BespokeSeam(
        name="lone-refresh exclusion",
        description=(
            "while a scan cycle is in flight, no standalone caller may mint a "
            "SECOND measurement into the cost-proposal store — so the report "
            "store and the cost store can never end up carrying two different "
            "cycle ids at two different anchors."
        ),
        reason_not_mechanized=(
            "the seam above pins that ONE record is threaded through both legs "
            "of a cycle; this pins that nothing ELSE writes between them. That "
            "is a concurrency property of two entry points, not a symbol "
            "reachable from one module — `recompute_cost_proposals` is "
            "legitimately called by the cycle, the refresh route and "
            "`tj optimize` alike, and which of them is allowed to proceed "
            "depends on runtime state (`scan_cycle.is_cycle_computing()`), "
            "which no static walk can evaluate."
        ),
        test_module="tests.unit.test_cost_proposals",
        test_name="test_a_lone_cost_refresh_declines_while_a_scan_cycle_is_in_flight",
    ),
    BespokeSeam(
        name="downgrade-map priceability",
        description=(
            "every DOWNGRADE_CANDIDATES entry — the key model AND its "
            "alternative, for every provider — must resolve to a price in "
            "pricing/models.toml."
        ),
        reason_not_mechanized=(
            "the defect is a missing ROW in a second, independently "
            "maintained TABLE (pricing/models.toml), not a second call site "
            "reaching a guarded symbol — DOWNGRADE_CANDIDATES and the "
            "pricing table have no shared symbol a reachability walk could "
            "pin. This is the same two-lists-drift shape the "
            "DOWNGRADE_CANDIDATES comment already documents biting once "
            "before via model_tiers.TIER_SUBSTRINGS, checked here from the "
            "other list's direction."
        ),
        test_module="tests.unit.test_downgrade_candidates_priceable",
        test_name="test_every_downgrade_candidate_is_currently_priceable",
    ),
)


def check_bespoke_seam(seam: BespokeSeam) -> str | None:
    """``None`` if the test pinning ``seam`` still exists and is callable;
    otherwise a string explaining what went missing."""
    try:
        module = importlib.import_module(seam.test_module)
    except ImportError as exc:
        return f"{seam.test_module} failed to import: {exc}"
    fn = getattr(module, seam.test_name, None)
    if fn is None or not callable(fn):
        return f"{seam.test_module}.{seam.test_name} no longer exists"
    return None


#: --------------------------------------------------------------------- #
#: AGGREGATE VERSUS PARTS — the mechanized half.
#:
#: The shape below used to be pinned one assertion at a time: `cache` held it,
#: `downsize` did not, and every OTHER many-from-one adapter was simply
#: unchecked. That is the same failure the SEAMS registry above exists to
#: retire — a design needing a new hand-written test per value has rebuilt the
#: problem it was built to solve. So the fan-out adapters get the same
#: treatment: ONE registry naming every family, ONE property test driven by
#: it, and a mechanical check that the registry still covers every adapter the
#: live dispatcher actually runs.
#: --------------------------------------------------------------------- #

#: The module and function that own the cost-adapter dispatch table. Every
#: finding-to-proposal adapter the product runs is named inside this function's
#: `adapters` tuple, so it is the one place :func:`cost_adapter_symbols` has to
#: read to answer "what fan-out adapters exist today".
COST_ADAPTER_MODULE = "core/optimize/cost_proposals.py"
COST_ADAPTER_DISPATCH = "_adapt_report"

#: What an adapter's name ends with. The dispatch table holds a mix of bare
#: references (``_trim_to_proposals``) and lambdas wrapping a call
#: (``lambda f: _cache_to_proposals(f, persona=persona)``); matching on the
#: NAME rather than the reference shape catches both without caring which form
#: a future adapter is registered in.
_ADAPTER_SUFFIXES = ("_to_proposal", "_to_proposals")


def cost_adapter_symbols() -> frozenset[str]:
    """Every finding-to-proposal adapter reachable from the live dispatcher.

    Read out of the source rather than imported, for the same reason
    :func:`offenders_for` walks the AST: the dispatch table is a local inside
    :data:`COST_ADAPTER_DISPATCH`, built per call from the report's own
    persona and config, so there is no importable object to enumerate. Parsing
    it means a newly-added adapter is visible to the registry check the moment
    it is wired in, with no second list to remember to update.
    """
    path = PACKAGE_ROOT / COST_ADAPTER_MODULE
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == COST_ADAPTER_DISPATCH:
            return frozenset(
                inner.id
                for inner in ast.walk(node)
                if isinstance(inner, ast.Name)
                and inner.id.endswith(_ADAPTER_SUFFIXES)
            )
    raise LookupError(
        f"{COST_ADAPTER_MODULE} no longer defines {COST_ADAPTER_DISPATCH}() — "
        "the cost-adapter dispatch table moved. Point COST_ADAPTER_DISPATCH at "
        "its new home; do not delete this check."
    )


@dataclass(frozen=True)
class AggregateFamily:
    """One finding, and every adapter that fans it out into proposals.

    The invariant: the figure a surface publishes as the family's TOTAL (the
    finding's own ``past_overspend_usd``, which
    ``api/routes/cost.py::_collect_recoverable`` reads generically off every
    finding for the Dashboard overlay) must equal the sum of the ``past_
    overspend_usd`` on the cards another surface publishes for it (the Review
    inbox), or the difference must be disclosed on the cards.

    ``verdict`` is one of:

    ``"conserves"``
        the parts sum to the whole (or disclose the difference) for a
        representative finding. The property test asserts it.
    ``"single-card"``
        the family emits at most ONE card however many rows the finding
        carries, so aggregate-versus-parts cannot arise — the card simply
        carries the finding's own figure. The property test asserts the
        at-most-one part, so an adapter that starts fanning out stops being
        exempt automatically rather than silently.
    ``"gap"``
        it does NOT hold, closing it is a product decision, and a strict
        ``xfail`` pins the current behaviour. ``gap_pins`` names those tests
        and they are checked to still exist, exactly like
        :data:`BESPOKE_SEAMS`.

    ``gap_pins`` may also be non-empty on a ``"conserves"`` family: `cache`
    conserves in the regime its cards cover and fails outside it, and both
    facts are worth pinning.
    """

    name: str
    description: str
    adapters: frozenset[str]
    verdict: str
    reason: str
    gap_pins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.verdict not in ("conserves", "single-card", "gap"):
            raise ValueError(
                f"unknown AggregateFamily.verdict {self.verdict!r} for {self.name!r}"
            )


AGGREGATE_FAMILIES: tuple[AggregateFamily, ...] = (
    AggregateFamily(
        name="downsize",
        description=(
            "the model-over-sizing finding, fanned out into a driver-role card "
            "plus either one card per agent or one window-wide card."
        ),
        adapters=frozenset({"_downsize_to_proposal"}),
        verdict="gap",
        reason=(
            "the per-agent cards come from build_agent_price_rows, a SEPARATE "
            "per-(agent, provider, model) repricing that drops any group it "
            "cannot price, while finding.past_overspend_usd is computed "
            "window-wide over every candidate session, priced or not. Nothing "
            "subtracts the dropped groups and nothing discloses them."
        ),
        gap_pins=(
            "test_the_downsize_per_agent_path_can_undercount_the_findings_own_total",
        ),
    ),
    AggregateFamily(
        name="cache",
        description=(
            "one cache-efficacy finding, fanned out by four adapters: the "
            "generic per-(provider, model) efficacy row plus the A1 uncached / "
            "A2 thrash / A3 lookback-miss per-agent root-cause cards."
        ),
        adapters=frozenset({
            "_cache_to_proposals",
            "_cache_uncached_to_proposals",
            "_cache_thrash_to_proposals",
            "_cache_lookback_to_proposals",
        }),
        verdict="conserves",
        reason=(
            "_cache_to_proposals nets each generic row against whatever the "
            "per-agent root-cause cards already claimed for the same "
            "(provider, model), so the family partitions the finding's total "
            "instead of double-claiming it — WHILE EVERY ROW IS FLAGGED. It "
            "does not hold once a row is not; see gap_pins."
        ),
        gap_pins=(
            "test_the_cache_family_drops_every_unflagged_rows_share",
        ),
    ),
    AggregateFamily(
        name="cache-recommend",
        description=(
            "the repeated-prefix finding, one card per prefix candidate."
        ),
        adapters=frozenset({"_cache_recommend_to_proposals"}),
        verdict="conserves",
        reason=(
            "the finding's total is the sum of its candidates' own figures, "
            "and each card carries its candidate's figure. Where a card is "
            "reduced to avoid double-counting a sibling cache card, the "
            "subtraction is named in that card's own estimate_basis, so the "
            "difference is disclosed rather than silent."
        ),
    ),
    AggregateFamily(
        name="trim",
        description="the prompt-bloat finding, one card per flagged agent.",
        adapters=frozenset({"_trim_to_proposals"}),
        verdict="conserves",
        reason=(
            "each card carries the finding's dollar figure PRORATED by that "
            "agent's share of total bloat characters, so the shares sum to 1 "
            "and the cards sum to the finding by construction."
        ),
    ),
    AggregateFamily(
        name="deadweight",
        description=(
            "the unused-MCP-server-and-plugin finding, one card per unused "
            "server plus one card per unused (every-component-unused) "
            "plugin."
        ),
        adapters=frozenset({"_deadweight_to_proposals", "_deadweight_plugin_to_proposals"}),
        verdict="conserves",
        reason=(
            "the finding's total is the sum of exactly the same two lists "
            "the cards iterate (unused_servers, unused_plugins), skipping "
            "the same unpriced entries each adapter skips, and the count of "
            "skipped ones is stated in the basis. A plugin with SOME "
            "components used never appears in unused_plugins at all "
            "(PluginDeadweight.partial_use_no_fix), so it contributes to "
            "neither side."
        ),
    ),
    AggregateFamily(
        name="script",
        description=(
            "the deterministic-tool-pattern finding, one card per cluster."
        ),
        adapters=frozenset({"_script_to_proposals"}),
        verdict="conserves",
        reason=(
            "the finding's total is accumulated over the same surfaced "
            "cluster list the cards iterate; the only clusters skipped carry "
            "no money to skip."
        ),
    ),
    AggregateFamily(
        name="reuse",
        description=(
            "the repeated-planning-skeleton finding, one card per cluster."
        ),
        adapters=frozenset({"_reuse_to_proposals"}),
        verdict="conserves",
        reason=(
            "the finding's total is the sum of the surfaced clusters' "
            "cache_reuse_recoverable_usd, which is the field each card "
            "carries; the only clusters skipped carry no money to skip."
        ),
    ),
    AggregateFamily(
        name="subagent",
        description="the subagent right-sizing finding.",
        adapters=frozenset({"_subagent_to_proposals"}),
        verdict="single-card",
        reason=(
            "the delta-verify pass measures the fan-out model-mix cost delta "
            "across ALL over-powered models at once, so one card listing them "
            "keeps that finding-level estimate coherent — splitting it per "
            "model would publish parts of an aggregate nothing re-derived "
            "per model."
        ),
    ),
    AggregateFamily(
        name="placement",
        description="the batch-placement finding.",
        adapters=frozenset({"_placement_to_proposals"}),
        verdict="single-card",
        reason=(
            "moving to the batch lane is one architectural change in the "
            "user's own application; the candidates are evidence on that one "
            "card, not separate levers."
        ),
    ),
    AggregateFamily(
        name="verbosity",
        description="the cohort-relative verbosity finding.",
        adapters=frozenset({"_verbosity_to_proposals"}),
        verdict="single-card",
        reason=(
            "a cohort-scoped signal is window-wide by construction — there is "
            "no per-row lever to card up."
        ),
    ),
    AggregateFamily(
        name="resend",
        description="the context-re-send finding.",
        adapters=frozenset({"_resend_to_proposals"}),
        verdict="single-card",
        reason=(
            "the card is deliberately COMPOUND — it consolidates resend's and "
            "subagent's levers into one CLAUDE.md rule rather than growing "
            "the inbox."
        ),
    ),
    AggregateFamily(
        name="summarize",
        description="the oversized-catalog-prompt-file finding.",
        adapters=frozenset({"_summarize_to_proposals"}),
        verdict="single-card",
        reason=(
            "one card however many files the scan flags: the card routes to "
            "the summarize curate/diff surface, which is where per-file work "
            "actually happens."
        ),
    ),
)


def unregistered_cost_adapters() -> frozenset[str]:
    """Adapters the dispatcher runs that no :data:`AGGREGATE_FAMILIES` entry
    claims. Non-empty means a new fan-out shipped with nothing checking that
    its cards sum to the figure the Dashboard publishes for it."""
    claimed = frozenset().union(*(f.adapters for f in AGGREGATE_FAMILIES))
    return cost_adapter_symbols() - claimed


#: --------------------------------------------------------------------- #
#: AGGREGATE VERSUS PARTS — the known gaps.
#:
#: A DIFFERENT shape of the same defect class: not one value derived twice,
#: but one FINDING fanned out into several proposals whose figures a surface
#: publishes as parts, next to (or instead of) an aggregate figure the
#: finding itself carries — with nothing forcing the parts to sum to the
#: whole, or disclosing it when they don't.
#:
#: The `cache` family holds this invariant WHERE ITS CARDS REACH:
#: `_cache_to_proposals` subtracts whatever the per-agent root-cause cards
#: (`_per_agent_cache_recoverable_by_model`) already claimed for the same
#: (provider, model) before it surfaces the generic row, so the family's
#: cards sum EXACTLY to the finding's own `past_overspend_usd` when every row
#: the finding priced is a flagged one — pinned in
#: `tests/unit/test_single_derivation.py::
#: test_the_cache_family_sums_exactly_to_the_findings_own_total`.
#:
#: It does NOT hold once a row is not flagged, and this is the second known
#: gap. `CacheEfficacyFinding.past_overspend_usd` is
#: `estimate_cache_recoverable(rows)` over EVERY (provider, model) row in the
#: window, which charges each row the gap between its own efficacy and the
#: 80% ceiling. The cards, though, are emitted per FLAGGED row only, and a row
#: is flagged on a much narrower test — supported provider, at least
#: `MIN_INPUT_TOKENS` of input, AND efficacy below `EFFICACY_THRESHOLD`. Every
#: row sitting between the flag threshold and the ceiling therefore contributes
#: real money to the aggregate the Dashboard tile publishes
#: (`api/routes/cost.py::_collect_recoverable` reads `past_overspend_usd`
#: straight off each finding) and produces no card at all, with nothing on
#: either surface naming the difference. The same asymmetry runs the other way
#: for the netting: `_per_agent_cache_recoverable_by_model`'s subtraction is
#: applied only while walking flagged rows, so a per-agent root-cause card on
#: an unflagged model's (provider, model) is never netted against anything.
#: Pinned as a strict xfail in
#: `test_the_cache_family_drops_every_unflagged_rows_share`.
#:
#: `downsize` does NOT hold it. When a finding carries `per_agent` rows,
#: `_downsize_to_proposal` drops the window-wide card and emits the
#: driver-role card plus one card per agent — but the per-agent rows come
#: from `build_agent_price_rows`, a SEPARATE per-(agent, provider, model)
#: repricing that silently DROPS any group whose model has no pricing data
#: or no dated candidate. `finding.past_overspend_usd` is unaffected by that
#: drop (it is `savings_window + driver_savings`, computed window-wide over
#: EVERY candidate session, priced or not). So the Dashboard tile
#: (`api/routes/cost.py::_collect_recoverable`, which reads
#: `report.downgrade.past_overspend_usd` straight off the finding) and the
#: Review inbox (the driver card plus the per-agent cards) can name
#: arbitrarily different totals for the SAME analyzer, with no disclosure
#: that the per-agent cards are a partial accounting.
#:
#: Measured directly (see the module docstring on
#: `test_the_downsize_per_agent_path_can_undercount_the_findings_own_total`):
#: an aggregate of $998.00 (one candidate's model carries no pricing data)
#: surfaced as $0.11 across the per-agent cards the inbox actually shows,
#: with nothing on either card naming the other $997.89.
#:
#: THIS IS NOT FIXED HERE. Choosing how to close it — fall back to the
#: window-wide card when the per-agent total falls materially short,
#: disclose the gap on each per-agent card the way the cache family's
#: `estimate_basis` does, or reprice every dropped group at a default rate —
#: is a product decision about what downsize/`build_agent_price_rows` should
#: do, not a mechanical guard. `test_the_downsize_per_agent_path_can_
#: undercount_the_findings_own_total` pins the CURRENT behaviour as an
#: xfail(strict=True): it exists to fail LOUDLY the moment someone closes
#: this gap (an unexpected pass forces the xfail's removal, which is the
#: signal to also delete this docstring's account of it) and to make the gap
#: impossible to silently reintroduce a different way in the meantime.
#: --------------------------------------------------------------------- #
KNOWN_GAPS = (
    "downsize: the per-agent card path can undercount the finding's own "
    "past_overspend_usd with no disclosure — see the module docstring "
    "above this constant, and "
    "test_the_downsize_per_agent_path_can_undercount_the_findings_own_total "
    "in tests/unit/test_single_derivation.py.",
    "cache: the finding's total is priced over EVERY (provider, model) row "
    "while its cards are emitted per FLAGGED row only, so an unflagged row's "
    "share reaches the Dashboard tile and no card — see the module docstring "
    "above this constant, and "
    "test_the_cache_family_drops_every_unflagged_rows_share "
    "in tests/unit/test_single_derivation.py.",
)


__all__ = [
    "PACKAGE_ROOT",
    "AGGREGATE_FAMILIES",
    "COST_ADAPTER_DISPATCH",
    "COST_ADAPTER_MODULE",
    "SEAMS",
    "BESPOKE_SEAMS",
    "KNOWN_GAPS",
    "AggregateFamily",
    "SingleSeam",
    "BespokeSeam",
    "check_bespoke_seam",
    "cost_adapter_symbols",
    "offenders_for",
    "unregistered_cost_adapters",
]
