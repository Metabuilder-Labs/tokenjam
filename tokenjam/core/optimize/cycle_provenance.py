"""ONE provenance record per scan cycle, carried by every artifact it writes.

WHICH BUILD, WHICH WINDOW, WHICH PASS. The analyzer stores are caches, and
nothing invalidates them on upgrade. Each carries a ``computed_at``, which
answers HOW OLD — and every surface renders it as ``computed <age> ago``, which
readers take to mean WHICH BUILD. Those are different questions, and they come
apart at exactly the moment it matters most: after upgrading tokenjam, every
card on screen was produced by the PREVIOUS build and keeps being served until
a pass completes. Analysis runs on ``[optimize] scan_interval_hours`` while
ingestion runs on ``[ingest] interval_minutes``, so the data under the cards
refreshes on a minute scale while the cards themselves are hours old. That is
exactly the reading which makes an upgrade look like it did nothing.

THREE DEFECTS, ONE RECORD. They are one record because they are one fact — "what
produced this figure" — that had been split into three ad-hoc conventions:

1. **The build was stamped but never compared.** ``tj_build()`` was called
   independently at every write site and again at every read site, and nothing
   anywhere asked whether a STORED stamp matched the RUNNING one. Two copies of
   "what version am I" are free to disagree, and a provenance label that
   disagrees with itself is worse than none.
2. **The window had two spellings.** The report store wrote ``since``/``until``
   and published them as ``scan_since``/``scan_until``; the cost store wrote
   ``cost_since``/``cost_until``. Both were raw dict keys with no shared type,
   so nothing forced them to describe the same span and nothing could compare
   them from the artifacts alone.
3. **The cycle had no identity.** One pass refreshes three stores SEQUENTIALLY
   on one thread (report, then relearn, then the cost proposals), and each store
   owns an independent in-flight flag. So a report-derived panel can serve cycle
   N figures beside inbox figures from cycle N-1, and ``cycle_computing``
   discloses "still scanning" without letting any surface tell the two cycles
   apart.

WHAT THE RECORD IS. :class:`CycleProvenance` is minted ONCE per cycle
(:func:`begin_cycle`), threaded into every store that cycle writes, and stored
verbatim beside each artifact under :data:`RECORD_KEY`. Every field describes
THE CYCLE, not the individual artifact: two artifacts from one pass carry byte
-identical records, which is what lets a surface ask "are these two figures from
the same cycle?" — :attr:`CycleProvenance.cycle_id` — without knowing anything
about the store topology.

WHAT THE WINDOW FIELD MEANS, AND WHAT IT DOES NOT. ``since``/``until`` are the
span THE CYCLE observed the corpus over: one anchor, one length, resolved
through the one window seam (``core/optimize/report_window.py``). It is a LABEL,
never a divisor — nothing rescales a stored figure by it. It is deliberately NOT
a claim that every artifact's own figures are bounded to it: relearn's detector
is unbounded on purpose (its figure is the write budget's pre-net gross), and
the relearn payload publishes its OWN ``window``/``past_overspend_windowed``
block for what its rows are bounded to. The record answers "which corpus window
did the pass that produced this read", which is true of the relearn cache too.

STALENESS IS DISCLOSED, NEVER DISCARDED. :func:`build_provenance` resolves the
stored build against the running one into three states — ``match``, ``stale``,
``unknown`` — and :func:`provenance_block` publishes that verdict on every
analyzer-fed payload. Deliberately tri-state rather than a boolean: "we cannot
tell" is not agreement and must not be able to read as it. Equally deliberately,
a stale artifact is still SERVED: dropping it would turn a populated surface
into an empty one across an upgrade, and an empty state is the strongest claim
a surface can make (``$0.00`` reads as "no waste", i.e. reassurance the data
does not support). A disclosed stale figure beats a confident absence. Resolving
the comparison HERE rather than in each surface is the point — the browser
already derived it once; the CLI, ``--json`` and the MCP server would each have
had to derive it again, and three derivations of one verdict is the defect this
module exists to retire.

DEGRADING ON AN OLD CACHE. An artifact written before this record existed has no
:data:`RECORD_KEY` block. :func:`provenance_block` falls back to the legacy keys
that artifact does carry (``tj_version``/``cost_tj_version`` for the build,
``since``/``until`` or ``cost_since``/``cost_until`` for the window) and reports
``cycle_id: None`` — meaning "this artifact predates cycle identity", which is
NOT "the same cycle as everything else" and must never render as it. Nothing
raises, nothing is discarded, and the next pass replaces it.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any, Mapping

#: What an unknown build reads as on the wire. A real value, never ``None`` —
#: a surface has to be able to distinguish "produced by a build that did not
#: stamp itself" from "the key is missing because the payload is malformed".
UNKNOWN = "unknown"

#: The three answers to "did the build that produced this figure match the one
#: serving it". Tri-state on purpose: a boolean ``stale`` would make the third
#: case read as agreement, which is the exact claim we cannot make.
BUILD_MATCH = "match"
BUILD_STALE = "stale"
BUILD_UNKNOWN = "unknown"

#: The key every store writes the record under. Prefixed stores (the cost
#: proposals share the relearn cache FILE and namespace their keys ``cost_``)
#: use ``<prefix>provenance``; see ``relearn_store.write_cache``, which copies
#: ``cost_``-prefixed keys forward by prefix precisely so a new one like this
#: cannot be dropped by a whitelist nobody updated.
RECORD_KEY = "provenance"


def tj_build() -> str:
    """The running build's version string. Never raises.

    A provenance label must not be able to sink the write it labels, so every
    failure mode collapses to :data:`UNKNOWN` — which a surface renders as an
    unqualified build rather than as agreement.

    This is the ONLY place the running build is resolved for a stored artifact:
    the stores no longer call it, they carry a record that already did.
    Read-side callers (a payload naming the build that is SERVING the figures)
    legitimately call it too — that is the other half of the comparison, not a
    second copy of the same one.
    """
    try:
        from tokenjam import __version__

        return str(__version__) or UNKNOWN
    except Exception:  # noqa: BLE001 - see docstring
        return UNKNOWN


def _iso(value: Any) -> str | None:
    """A timestamp as an ISO string, whatever shape it arrived in.

    Accepts a ``datetime`` (the in-process case) or a string (already stored,
    or handed across a thread boundary). Anything else is discarded rather than
    raised on — a malformed provenance field must not sink a pass that would
    otherwise have succeeded.
    """
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and value:
        return value
    return None


def _dt(value: Any) -> datetime | None:
    """The inverse of :func:`_iso`, equally tolerant.

    ``fromisoformat`` on 3.10 cannot read a trailing ``Z``, and stored payloads
    predating this module were written by ``datetime.now(timezone.utc)
    .isoformat()`` (``+00:00``) in some places and may carry ``Z`` from others,
    so the suffix is normalised before parsing.
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class CycleProvenance:
    """What produced a stored analyzer artifact. One per cycle, not per store.

    Frozen because it is a fact about a pass, not a mutable accumulator: the
    only permitted change is :meth:`with_persona`, which SEALS the label once
    from the report the pass built rather than deriving it a second time.

    * ``cycle_id`` — the identity two artifacts share iff one pass wrote both.
    * ``build`` — the build that produced them (:func:`tj_build`, resolved once).
    * ``anchor`` — the single instant every window in this cycle is subtracted
      from, so two surfaces publishing one metric cannot cover windows offset
      from each other.
    * ``since``/``until`` — the resolved bounds. ``until`` is the anchor; they
      are separate fields because a caller may legitimately pass an explicit
      ``until`` that is not "now" (``tj optimize --since``), and because the
      pair is what the two retired window conventions were each half of.
    * ``window_days`` — the length, resolved through the one window seam.
    * ``persona`` — the window's dominant persona, taken from the report the
      pass built (see :meth:`with_persona`). :data:`UNKNOWN` until sealed.
    * ``computed_at`` — when the CYCLE began. Distinct from each artifact's own
      ``computed_at``, which is when that artifact landed; the two differ by
      however long the pass took, which on a cold corpus is minutes.
    """

    cycle_id: str
    build: str
    anchor: str | None
    since: str | None
    until: str | None
    window_days: float | None
    persona: str
    computed_at: str

    @property
    def since_dt(self) -> datetime | None:
        return _dt(self.since)

    @property
    def until_dt(self) -> datetime | None:
        return _dt(self.until)

    @property
    def anchor_dt(self) -> datetime | None:
        return _dt(self.anchor)

    def with_persona(self, persona: Any) -> CycleProvenance:
        """Seal the persona label from the report this cycle built.

        The persona is NOT re-derived here. ``build_report`` already resolves
        the window's dominant persona exactly once (it is the choke point for
        the analyzer skip gate), so the cycle takes that value rather than
        running ``agent_persona_mix``/``dominant_persona`` a second time over
        the same connection and window — which would be a second derivation of
        precisely the kind this module exists to remove.

        A missing or non-string persona leaves the record untouched: a label we
        could not read is not a reason to overwrite one we already had.
        """
        if not isinstance(persona, str) or not persona:
            return self
        return replace(self, persona=persona)

    def with_window(self, since: Any, until: Any, window_days: Any) -> CycleProvenance:
        """Fill in a window this record was minted without.

        :func:`begin_cycle` degrades to ``None`` window fields rather than
        raising when the window seam cannot be resolved, and a caller that then
        resolves one itself must put it back ON THE RECORD — otherwise the
        artifact's flat window keys and its record describe different things,
        which is the two-conventions defect reappearing inside one payload.
        """
        try:
            days = float(window_days) if window_days is not None else self.window_days
        except (TypeError, ValueError):
            days = self.window_days
        return replace(
            self,
            since=_iso(since) or self.since,
            until=_iso(until) or self.until,
            window_days=days,
        )

    def to_dict(self) -> dict[str, Any]:
        """The stored/wire form. Plain JSON — these payloads round-trip through
        ``json.dumps`` in every store."""
        return {
            "cycle_id": self.cycle_id,
            "build": self.build,
            "anchor": self.anchor,
            "since": self.since,
            "until": self.until,
            "window_days": self.window_days,
            "persona": self.persona,
            "computed_at": self.computed_at,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> CycleProvenance | None:
        """Rehydrate a stored record, or ``None`` when there isn't one.

        ``None`` for anything that is not a mapping carrying a ``cycle_id``:
        a record without an identity cannot answer the one question the record
        exists for, and a half-built one would be worse than falling back to
        the legacy keys the artifact does carry. Never raises — this runs on a
        read path over a file any earlier build may have written.
        """
        if not isinstance(raw, Mapping):
            return None
        cycle_id = raw.get("cycle_id")
        if not isinstance(cycle_id, str) or not cycle_id:
            return None
        window_days = raw.get("window_days")
        try:
            days = float(window_days) if window_days is not None else None
        except (TypeError, ValueError):
            days = None
        build = raw.get("build")
        persona = raw.get("persona")
        return cls(
            cycle_id=cycle_id,
            build=build if isinstance(build, str) and build else UNKNOWN,
            anchor=_iso(raw.get("anchor")),
            since=_iso(raw.get("since")),
            until=_iso(raw.get("until")),
            window_days=days,
            persona=persona if isinstance(persona, str) and persona else UNKNOWN,
            computed_at=_iso(raw.get("computed_at")) or "",
        )


def begin_cycle(
    config: Any = None,
    *,
    conn: Any = None,
    anchor: Any = None,
    since: Any = None,
    until: Any = None,
    window_days: Any = None,
    persona: str = UNKNOWN,
) -> CycleProvenance:
    """Mint THE record for one pass. The only constructor of the record.

    Called once per cycle by ``scan_cycle``, and once per lone refresh by a
    store invoked outside a cycle (a test, ``tj optimize``, a standalone
    relearn recompute) — which is the same "``None`` means you decide" rule the
    shared anchor already used, expressed as a record instead of a bare
    timestamp.

    ``window_days`` is resolved through ``report_window.report_window_days``
    when a ``config`` is given and no explicit length is passed — the one seam
    both the report store and the cost store already resolved their window
    through.

    THE BOUNDS ARE NEVER INVENTED. ``until`` falls back to the anchor, and
    ``since`` to ``until - window_days``, ONLY when this call has real grounds
    for a window: an explicitly supplied bound, or a ``config`` to resolve the
    length from. A caller that passes a bare length and no bounds (a legacy
    direct write) gets ``since``/``until`` of ``None`` rather than a pair
    derived from the day count — deriving them would invent exactly the
    provenance the fields exist to supply, and a cache predating the bounds has
    to keep reporting them absent.

    Never raises: a window that cannot be resolved degrades to ``None`` rather
    than sinking the pass it was supposed to label.
    """
    from tokenjam.utils.time_parse import utcnow

    explicit_anchor = _dt(anchor)
    explicit_until = _dt(until)
    anchor_dt = explicit_anchor or utcnow()

    days: float | None = None
    if window_days is not None:
        try:
            days = float(window_days)
        except (TypeError, ValueError):
            days = None
    elif config is not None:
        try:
            from tokenjam.core.optimize.report_window import report_window_days

            days = float(report_window_days(config, conn))
        except Exception:  # noqa: BLE001 - see docstring
            days = None

    grounded = explicit_anchor is not None or explicit_until is not None or config is not None
    until_dt = explicit_until or (anchor_dt if grounded else None)
    since_dt = _dt(since)
    if since_dt is None and days is not None and until_dt is not None:
        since_dt = until_dt - timedelta(days=days)

    return CycleProvenance(
        cycle_id=uuid.uuid4().hex,
        build=tj_build(),
        anchor=_iso(anchor_dt),
        since=_iso(since_dt),
        until=_iso(until_dt),
        window_days=days,
        persona=persona if isinstance(persona, str) and persona else UNKNOWN,
        computed_at=_iso(utcnow()) or "",
    )


def from_stored(
    stored: Mapping[str, Any] | None, *, prefix: str = "",
) -> CycleProvenance | None:
    """The record an artifact carries, or ``None`` when it predates the record.

    ``prefix`` addresses a store that namespaces its keys — the cost proposals
    live in the relearn cache FILE under ``cost_``-prefixed keys, so their
    record is ``cost_provenance``.
    """
    if not isinstance(stored, Mapping):
        return None
    return CycleProvenance.from_dict(stored.get(f"{prefix}{RECORD_KEY}"))


def build_provenance(computed_build: Any, running: str | None = None) -> str:
    """Resolve a stored build against the running one. THE comparison.

    Three states, and the third is the trap. :data:`BUILD_MATCH` means the
    figures were produced by the build serving them, so the timestamp is the
    whole truth. :data:`BUILD_STALE` means they were not — these stores are
    caches and an upgrade does not invalidate them, so the timestamp is still
    honest about age and no longer sufficient on its own. :data:`BUILD_UNKNOWN`
    means the artifact predates the stamp, which is NOT agreement.
    """
    if not isinstance(computed_build, str) or not computed_build:
        return BUILD_UNKNOWN
    if computed_build == UNKNOWN:
        return BUILD_UNKNOWN
    return BUILD_MATCH if computed_build == (running or tj_build()) else BUILD_STALE


def provenance_block(
    stored: Mapping[str, Any] | None, *, prefix: str = "",
) -> dict[str, Any]:
    """The provenance keys EVERY analyzer-fed payload carries, in one place.

    One ``ScanBar`` reads all three surfaces (the Dashboard/Optimize report, the
    Review inbox's relearn feed, its cost feed), so a key spelled differently
    per feed is a surface that silently loses the qualification — the same
    two-derivations-of-one-truth defect, at the key-name layer. Both the report
    store's envelope and the relearn routes' hand-assembled payloads call this,
    rather than each listing the keys itself.

    * ``cycle_id`` — which pass produced these figures. Two payloads with equal
      non-``None`` ids are from one cycle; ``None`` means the artifact predates
      cycle identity, which is not "the same cycle" and must not read as it.
    * ``cycle_computing`` — the WHOLE pass, not this store. Each store's own
      flag goes false as its leg lands, so a surface reading only its own
      stops saying "Scanning…" while the figures beside it are still the
      previous pass's. A surface wants ``computing or cycle_computing``.
    * ``computed_build`` / ``build`` / ``build_provenance`` — the build that
      PRODUCED the figures, the one SERVING them, and the resolved verdict.
    * ``scan_since`` / ``scan_until`` — the cycle's observed window, under the
      SAME two names on every feed. The cost feed used to publish this pair as
      ``cost_since``/``cost_until`` and the relearn feed not at all.
    * ``provenance`` — the whole record, for a consumer that wants the anchor,
      the length or the persona without another endpoint.

    Degrades on an artifact written before the record existed: the legacy
    per-store keys are read instead, and ``cycle_id`` reports ``None``.
    """
    from tokenjam.core.optimize import scan_cycle

    stored = stored if isinstance(stored, Mapping) else {}
    record = from_stored(stored, prefix=prefix)
    running = tj_build()

    if record is not None:
        computed_build: Any = record.build
        since: Any = record.since
        until: Any = record.until
    else:
        # PRE-RECORD ARTIFACT. Everything it does carry is still published; only
        # the cycle identity is genuinely unavailable.
        computed_build = stored.get(f"{prefix}tj_version")
        since = stored.get(f"{prefix}since")
        until = stored.get(f"{prefix}until")

    return {
        "cycle_id": record.cycle_id if record is not None else None,
        "cycle_computing": scan_cycle.is_cycle_computing(),
        "computed_build": computed_build,
        "build": running,
        "build_provenance": build_provenance(computed_build, running),
        "scan_since": since,
        "scan_until": until,
        "provenance": record.to_dict() if record is not None else None,
    }


__all__ = [
    "BUILD_MATCH",
    "BUILD_STALE",
    "BUILD_UNKNOWN",
    "RECORD_KEY",
    "UNKNOWN",
    "CycleProvenance",
    "begin_cycle",
    "build_provenance",
    "from_stored",
    "provenance_block",
    "tj_build",
]
