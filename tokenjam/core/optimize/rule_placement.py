"""WHERE a permanent rule should be written, derived from the sessions it is
written about.

**Named ``rule_placement``, not ``placement``, on purpose.** ``placement`` is
already a registered analyzer name in this tree (``analyzers/batch_placement.py``
→ ``@register("placement")``), which asks an unrelated question: whether work
belongs on the Batch API's lane. Registry strings and module names are decoupled
here (Critical Rule 19), so a bare ``placement.py`` beside that registration
would read as its implementation. This module is about where a CLAUDE.md rule
lands and has nothing to do with batch pricing.

Four analyzers (``downsize``, ``resend``, ``subagent``, ``relearn``) share one
fix shape: append a rule to a ``CLAUDE.md``. Each used to offer exactly one
destination, the workspace/root file. That single-destination assumption is
wrong in both directions. A root-level rule is re-sent in every session of
every project, including the projects that never exhibited the problem, so its
standing cost is paid globally for waste that is usually concentrated in a few
repos — and it dilutes the one file whose dilution makes rules start getting
ignored.

The data to place a rule correctly was already there and unused: **every
session records its working directory**. ``analyzers/deadweight.py`` reads it
to resolve per-project MCP configs, and ``core/summarize/repo_roots`` already
turns a window's recorded cwds into boundary-safe scan roots. This module
reuses that resolver rather than adding a third cwd-resolution path, and asks
one further question of the result: which of those roots would a rule about
THESE sessions have to land in, and what share of the finding belongs to each?

**Why this is arithmetic and not aesthetics.** ``core/optimize/write_budget``
already nets a rule's standing cost against its saving to decide whether the
write is OFFERED at all. It priced exactly one destination. Once placement is
real, ``N project files x their own session counts`` is frequently cheaper
than ``1 root file x every session in the window``, so a rule that reads
net-negative against the root file can legitimately flip to net-positive purely
by landing in the right place. Placement is therefore an INPUT to whether the
fix is worth offering, not a presentation detail.

Two costs, deliberately kept apart, because they answer different questions:

* **standing cost** — what the rule costs to KEEP. Per-session tokens x the
  sessions that actually load the file. Summed across the chosen destinations.
  This is what the saving is netted against.
* **footprint cost** — what the FILES have to carry. Per-session tokens x the
  number of files written. This is what the write budget is spent in, and it
  is why N destinations are not free: N files each grew.

The root/workspace file stays a legitimate destination — the correct answer
when waste really is spread across many projects — and :func:`choose_placement`
picks it by comparing those two costs, never by preference.

**What this module does not do.** It never reconstructs a directory that has
gone: a recorded cwd with nothing behind it is counted, reported in the
``coverage_note``, and left out. The note says the gap is what was NOT
ANALYSED, never what was unavoidable (Critical Rule 30), and the counting
itself follows Critical Rule 31 — an analyzer reading live filesystem state
degrades silently, so the degradation has to be stated rather than inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
from tokenjam.core.rulewrite.kinds import DELIVERY_CLAUDE_MD_RULE
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from tokenjam.core.summarize.repo_roots import ResolvedRoots, resolve_roots

#: Scope strings, shared with ``relearn_apply.default_target_path`` so a
#: destination this module derives and a target that module writes agree.
SCOPE_PROJECT = "project"
SCOPE_USER_GLOBAL = "user-global"

#: A destination has to be worth naming. Below this many attributed sessions a
#: project root is folded into the user-global fallback rather than earning its
#: own block: writing a permanent rule into a repo the window barely touched
#: spends a file's budget on a population too small to have shown anything.
MIN_SESSIONS_PER_DESTINATION = 2


@dataclass(frozen=True)
class SessionShare:
    """One session's contribution to a finding, as placement needs to see it.

    ``weight`` is the session's share of whatever the finding measured, in
    TOKENS — the one attribution weight, used for both the token and the dollar
    split so a destination's implied per-token rate is identical to the
    finding's own (Critical Rule 28). A caller with no per-session breakdown
    passes ``weight=0`` for every session, and the split falls back to session
    count; :attr:`PlacementPlan.attribution_basis` says which happened.
    """

    session_id: str
    weight: int = 0


@dataclass(frozen=True)
class Destination:
    """One CLAUDE.md a rule could be written into, and what belongs to it."""

    #: Absolute path of the file the rule would be appended to.
    path: str
    #: ``project`` or ``user-global``.
    scope: str
    #: The project root the path was derived from. Empty for user-global.
    root: str
    #: The sessions attributed to this destination, sorted.
    session_ids: tuple[str, ...] = ()
    #: This destination's share of the finding's ``past_overspend_tokens``.
    tokens: int = 0
    #: Its share of ``past_overspend_usd``, split by the SAME weights as
    #: ``tokens``. ``None`` when the finding carried no dollar figure — both
    #: fields degrade together, never one to zero (Critical Rule 28 (a)).
    usd: float | None = None

    @property
    def sessions(self) -> int:
        """How many of the window's sessions re-send a rule written here."""
        return len(self.session_ids)


@dataclass(frozen=True)
class PlacementPlan:
    """Every destination a finding's rule could land in, ranked, plus the gap.

    ``destinations`` is ranked by attributed tokens (then session count, then
    path) — highest first. ``user_global`` is the single-file alternative,
    always present, carrying the WHOLE finding: it is the destination every
    session loads, so it is the one that needs no attribution at all.
    """

    destinations: tuple[Destination, ...] = ()
    user_global: Destination | None = None
    #: Sessions the caller handed in.
    total_sessions: int = 0
    #: Sessions whose destination could not be resolved: no recorded cwd, a cwd
    #: that no longer exists, or one the boundary gate refuses. Their share is
    #: reported, never redistributed onto the destinations that did resolve —
    #: redistributing would quietly grow every surviving destination's claim.
    unresolved_sessions: int = 0
    unresolved_tokens: int = 0
    unresolved_usd: float | None = None
    #: Distinct recorded directories, and what happened to them. Straight from
    #: :class:`~tokenjam.core.summarize.repo_roots.ResolvedRoots`.
    recorded_cwds: int = 0
    vanished_cwds: int = 0
    refused_cwds: int = 0
    #: ``"weighted"`` when the caller supplied per-session weights,
    #: ``"by session count"`` when it did not.
    attribution_basis: str = "by session count"
    #: What the split does and does not cover, in words, ending by saying the
    #: difference is what was not analysed (Critical Rule 30). Empty when
    #: nothing was dropped.
    coverage_note: str = ""

    @property
    def resolved_sessions(self) -> int:
        return sum(d.sessions for d in self.destinations)


def _split_usd(total_usd: float | None, share: float) -> float | None:
    """A destination's dollar share, or ``None`` when there is nothing to split.

    Never returns 0.0 for a missing figure: an unpriced finding leaves every
    destination unpriced, which is the symmetric degrade the token/dollar
    contract requires.
    """
    if total_usd is None:
        return None
    return round(total_usd * share, 6)


def _claude_md_for(root: Path) -> str:
    """The CLAUDE.md a session working in ``root`` actually loads.

    Delegates to ``relearn_apply``'s own target resolution so the file this
    module ranks and the file the apply path writes are decided by one
    function. Imported lazily: ``relearn_apply`` reaches back into the optimize
    package, and a module-level import would cycle.
    """
    from tokenjam.core.optimize.relearn_apply import default_target_path

    return default_target_path(DELIVERY_CLAUDE_MD_RULE, SCOPE_PROJECT, str(root), "")


def _user_global_path(claude_home: Path | None) -> str:
    from tokenjam.core.optimize.relearn_apply import default_target_path

    return default_target_path(
        DELIVERY_CLAUDE_MD_RULE, SCOPE_USER_GLOBAL, "", "", claude_home=claude_home,
    )


def _destination_root(cwd: str, within: Path | None) -> tuple[str, ResolvedRoots]:
    """The project root one recorded cwd resolves to, or ``""``.

    ``resolve_roots`` returns BOTH the directory and its enclosing repo root
    (Claude Code loads a CLAUDE.md from the cwd and from its ancestors), which
    is right for a scan that prices every file. A rule is written ONCE, so this
    picks the outermost surviving root — the file a session in a sub-package
    and a session at the repo root both load.
    """
    resolved = resolve_roots([cwd], within=within)
    if not resolved.roots:
        return "", resolved
    # `roots` is sorted, so the shortest path that is a prefix of the rest is
    # the outermost. Compare on parts rather than string length: a sibling
    # directory with a shorter name is not an ancestor.
    outermost = resolved.roots[0]
    for candidate in resolved.roots:
        if len(candidate.parts) < len(outermost.parts):
            outermost = candidate
    return str(outermost), resolved


def _coverage_note(
    *,
    total_sessions: int,
    unresolved_sessions: int,
    vanished_cwds: int,
    refused_cwds: int,
    missing_cwds: int,
    destinations: int,
) -> str:
    """State what the placement split left out, in words.

    Ends by saying the difference is what was NOT ANALYSED. That sentence is
    the whole point (Critical Rule 30): a reader shown "3 of 11 projects" will
    otherwise conclude the other 8 were examined and found clean.
    """
    if unresolved_sessions <= 0:
        return ""
    parts: list[str] = []
    if missing_cwds:
        parts.append(f"{missing_cwds} recorded no working directory")
    if vanished_cwds:
        parts.append(f"{vanished_cwds} named a directory that is no longer on disk")
    if refused_cwds:
        parts.append(
            f"{refused_cwds} named a directory that is not a safe place to write "
            "into (a home directory or a bare top-level path)"
        )
    detail = ("; ".join(parts) + ". ") if parts else ""
    return (
        f"This placement covers {total_sessions - unresolved_sessions} of "
        f"{total_sessions} session(s), across {destinations} project file(s). "
        f"The other {unresolved_sessions} could not be placed: {detail}"
        "Their share is reported here and is not folded into any destination "
        "above. That difference is what was not analysed for placement, not "
        "work that had nowhere better to go."
    )


def build_placement_plan(
    shares: Sequence[SessionShare],
    session_cwds: Mapping[str, str],
    *,
    total_tokens: int = 0,
    total_usd: float | None = None,
    within: Path | None = None,
    claude_home: Path | None = None,
    min_sessions_per_destination: int = MIN_SESSIONS_PER_DESTINATION,
) -> PlacementPlan:
    """Resolve every CLAUDE.md a finding's rule could be written into.

    ``shares`` are the sessions the finding is derived from; ``session_cwds``
    maps a session id to the working directory it recorded (missing or empty is
    fine and is counted, not guessed). ``total_tokens``/``total_usd`` are the
    finding's own ``past_overspend_*``, split across destinations by each
    session's ``weight`` — or evenly by session count when no weights were
    supplied.

    ``within`` is passed straight through to ``repo_roots.resolve_roots`` as
    the outer filesystem boundary; ``claude_home`` scopes the user-global
    fallback the same way ``relearn_apply.default_target_path`` does, so a
    review served from a throwaway ``--db`` cannot name a real path on the
    operator's machine.

    Never raises: an unreadable path contributes nothing but a count.
    """
    unique: dict[str, SessionShare] = {}
    for share in shares:
        sid = str(share.session_id or "")
        if not sid:
            continue
        # A session handed in twice (one row per dispatch) is one session.
        existing = unique.get(sid)
        if existing is None:
            unique[sid] = share
        elif share.weight > existing.weight:
            unique[sid] = share

    total_sessions = len(unique)
    weighted = any(s.weight > 0 for s in unique.values())
    weight_total = (
        sum(max(0, s.weight) for s in unique.values()) if weighted else total_sessions
    )

    by_root: dict[str, list[SessionShare]] = {}
    unresolved: list[SessionShare] = []
    recorded: set[str] = set()
    vanished: set[str] = set()
    refused: set[str] = set()
    missing_cwd = 0
    root_cache: dict[str, str] = {}

    for sid, share in unique.items():
        cwd = str(session_cwds.get(sid, "") or "")
        if not cwd:
            missing_cwd += 1
            unresolved.append(share)
            continue
        recorded.add(cwd)
        if cwd not in root_cache:
            try:
                root, resolved = _destination_root(cwd, within)
            except (OSError, RuntimeError):
                root, resolved = "", ResolvedRoots()
            root_cache[cwd] = root
            if not root:
                if resolved.vanished:
                    vanished.add(cwd)
                else:
                    refused.add(cwd)
        root = root_cache[cwd]
        if not root:
            unresolved.append(share)
            continue
        by_root.setdefault(root, []).append(share)

    def _weight_of(members: Iterable[SessionShare]) -> int:
        members = list(members)
        return (
            sum(max(0, m.weight) for m in members) if weighted else len(members)
        )

    # A root the window barely touched is folded back into the unresolved pool
    # rather than earning a permanent block of its own.
    thin = [
        root for root, members in by_root.items()
        if len(members) < max(1, min_sessions_per_destination)
    ]
    for root in thin:
        unresolved.extend(by_root.pop(root))

    destinations: list[Destination] = []
    for root, members in by_root.items():
        weight = _weight_of(members)
        fraction = (weight / weight_total) if weight_total > 0 else 0.0
        destinations.append(Destination(
            path=_claude_md_for(Path(root)),
            scope=SCOPE_PROJECT,
            root=root,
            session_ids=tuple(sorted(m.session_id for m in members)),
            tokens=round(total_tokens * fraction),
            usd=_split_usd(total_usd, fraction),
        ))
    destinations.sort(key=lambda d: (-d.tokens, -d.sessions, d.path))

    unresolved_weight = _weight_of(unresolved)
    unresolved_fraction = (
        (unresolved_weight / weight_total) if weight_total > 0 else 0.0
    )
    return PlacementPlan(
        destinations=tuple(destinations),
        user_global=Destination(
            path=_user_global_path(claude_home),
            scope=SCOPE_USER_GLOBAL,
            root="",
            session_ids=tuple(sorted(unique)),
            tokens=int(total_tokens),
            usd=total_usd,
        ),
        total_sessions=total_sessions,
        unresolved_sessions=len(unresolved),
        unresolved_tokens=round(total_tokens * unresolved_fraction),
        unresolved_usd=_split_usd(total_usd, unresolved_fraction),
        recorded_cwds=len(recorded),
        vanished_cwds=len(vanished),
        refused_cwds=len(refused),
        attribution_basis="weighted" if weighted else "by session count",
        coverage_note=_coverage_note(
            total_sessions=total_sessions,
            unresolved_sessions=len(unresolved),
            vanished_cwds=len(vanished),
            refused_cwds=len(refused),
            missing_cwds=missing_cwd,
            destinations=len(destinations),
        ),
    )


@dataclass(frozen=True)
class PlacementChoice:
    """Where the rule actually goes, and the comparison that decided it."""

    scope: str
    destinations: tuple[Destination, ...] = ()
    #: Per-session tokens the artifact adds to each file it lands in.
    standing_tokens_per_session: int = 0
    #: What the chosen placement costs to KEEP: per-session x the sessions that
    #: load each chosen file, summed.
    standing_tokens: int = 0
    #: What the chosen placement costs the write BUDGET: per-session x the
    #: number of files written. A project split of 3 files charges 3 blocks of
    #: growth even though no single session pays more than one.
    footprint_tokens: int = 0
    #: The rejected alternative's standing cost, kept so the decision is
    #: inspectable rather than a bare verdict.
    alternative_standing_tokens: int = 0
    #: Sessions that re-send the artifact under this choice — what
    #: ``write_budget`` charges the standing cost against.
    exposure_sessions: int = 0
    basis: str = ""


PLACEMENT_BASIS_PROJECT = (
    "written into the project files whose own sessions incurred this, so it is "
    "re-sent only in those projects rather than in every session of every "
    "project"
)
PLACEMENT_BASIS_GLOBAL = (
    "written into the user-global CLAUDE.md, because the sessions that incurred "
    "this are spread widely enough that one shared rule is cheaper to keep than "
    "one rule per project"
)


def choose_placement(
    plan: PlacementPlan,
    *,
    standing_tokens_per_session: int,
    total_sessions: int,
) -> PlacementChoice:
    """Pick project-scoped or user-global placement by the arithmetic.

    The comparison is between two standing costs on the SAME basis:

    * project — ``per_session x sum(sessions loading each chosen file)``
    * user-global — ``per_session x every session in the window``

    Project placement wins whenever it covers fewer sessions than the whole
    window, which is the common case and is exactly the defect this exists to
    fix. The global file wins when the waste really is spread everywhere (the
    two costs converge) or when nothing resolved to a project at all — and it is
    also the answer when a project split would write more blocks than the write
    budget can carry, which the caller sees via ``footprint_tokens``.

    ``total_sessions`` is the window's own session count, not the finding's:
    a user-global rule is re-sent by every session the user runs, including the
    ones this finding says nothing about. That is the whole cost being avoided.
    """
    per_session = max(0, int(standing_tokens_per_session))
    window_sessions = max(0, int(total_sessions))
    global_dest = plan.user_global
    global_standing = per_session * window_sessions

    if not plan.destinations:
        return PlacementChoice(
            scope=SCOPE_USER_GLOBAL,
            destinations=(global_dest,) if global_dest is not None else (),
            standing_tokens_per_session=per_session,
            standing_tokens=global_standing,
            footprint_tokens=per_session,
            alternative_standing_tokens=global_standing,
            exposure_sessions=window_sessions,
            basis=PLACEMENT_BASIS_GLOBAL,
        )

    project_exposure = sum(d.sessions for d in plan.destinations)
    project_standing = per_session * project_exposure
    if project_standing >= global_standing:
        return PlacementChoice(
            scope=SCOPE_USER_GLOBAL,
            destinations=(global_dest,) if global_dest is not None else (),
            standing_tokens_per_session=per_session,
            standing_tokens=global_standing,
            footprint_tokens=per_session,
            alternative_standing_tokens=project_standing,
            exposure_sessions=window_sessions,
            basis=PLACEMENT_BASIS_GLOBAL,
        )
    return PlacementChoice(
        scope=SCOPE_PROJECT,
        destinations=plan.destinations,
        standing_tokens_per_session=per_session,
        standing_tokens=project_standing,
        footprint_tokens=per_session * len(plan.destinations),
        alternative_standing_tokens=global_standing,
        exposure_sessions=project_exposure,
        basis=PLACEMENT_BASIS_PROJECT,
    )
