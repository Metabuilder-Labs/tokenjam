"""On-disk cache for the relearn aggregator's (expensive, full-corpus) result.

The detector (``core.optimize.analyzers.relearn``) takes tens of seconds over
a real local corpus — far too slow to compute per HTTP request. ``tj serve``
computes it on a background schedule using a FRESH DuckDB connection (mirrors
the retention job's own-connection pattern in ``cli/cmd_serve.py``, so a slow
scan never contends with the live request connection's write lock — see the
DuckDB single-writer relearn this very module exists to help catch more of).
This module is the read/write boundary: a small JSON file at
``~/.tj/relearn_cache.json`` plus an in-process lock so two overlapping
recomputes never race each other's writes.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tokenjam.core.analysis_span import retention_days_for
from tokenjam.core.optimize.cycle_provenance import CycleProvenance, begin_cycle
from tokenjam.core.optimize.analyzers.relearn import RelearnFinding, compute_relearn_finding

if TYPE_CHECKING:
    from tokenjam.core.config import TjConfig

_LOCK = threading.Lock()
_COMPUTING = threading.Event()


def default_cache_path(config: TjConfig | None = None) -> Path:
    """``<storage-parent>/relearn_cache.json`` when ``config`` is given — this
    honors ``--config`` / ``storage.path`` (and falls back to a config-scoped
    TEMP dir, never the real ``~/.tj``, when ``storage.path`` is ``""``/
    ``":memory:"``; see ``relearn_apply._storage_base_dir``). Without a
    ``config`` (legacy callers), the old hardcoded ``~/.tj`` default."""
    if config is not None:
        from tokenjam.core.optimize.relearn_apply import _storage_base_dir

        return _storage_base_dir(config) / "relearn_cache.json"
    return Path.home() / ".tj" / "relearn_cache.json"


def _distill_cache_dir_for(config: TjConfig | None) -> Path | None:
    """The distill cache this run may write to, or None to leave the default.

    Imported lazily and never allowed to raise: a cache-path resolution
    failure must not sink a recompute that would otherwise succeed.
    """
    if config is None:
        return None
    try:
        from tokenjam.core.optimize.analyzers.relearn import _distill_cache_dir

        return _distill_cache_dir(config)
    except Exception:
        return None


def read_cache(
    path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any] | None:
    """The last-written ``{"computed_at", "finding"}`` payload, or ``None`` if
    no recompute has ever completed (fresh install) or the file is corrupt."""
    p = path or default_cache_path(config)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def write_cache(
    finding: RelearnFinding, path: Path | None = None, *, config: TjConfig | None = None,
    provenance: CycleProvenance | None = None,
) -> dict[str, Any]:
    """Atomically write the finding (temp file + rename), never a partial file
    a concurrent reader could observe.

    The cache file is shared with the cost-proposal producer (see
    ``write_cost_proposals``): the two write on different cadences (the relearn
    detector job vs the optimize path). To keep this "the same proposal store"
    without one producer clobbering the other, an existing ``cost_proposals``
    block is read back and preserved here rather than dropped.

    Detection time is also when each cluster gets its stable ``proposal_id``
    (``relearn_proposals.stamp_proposal_ids``): the apply paths accept a stored
    proposal ID and nothing else, so the IDs have to exist on the record the
    detector itself wrote.

    ``provenance`` is the CYCLE's record (``core/optimize/cycle_provenance.py``),
    identical to the one on the report and the cost proposals that same pass
    wrote — which is what lets a surface tell "these rows and that tile are one
    cycle" from "one of them is a cycle behind". Its window fields describe the
    span the CYCLE read the corpus over; they are NOT a claim that these
    clusters are bounded to it, because relearn's detector is deliberately
    unbounded and the relearn payload publishes its own ``window`` block for
    what its rows are bounded to.
    """
    from tokenjam.core.optimize.relearn_proposals import stamp_proposal_ids

    p = path or default_cache_path(config)
    # No cycle: an identity-only record (no ``config``, so no window is
    # resolved). This cache carries no window keys of its own, so a window
    # stamped here would be decoration at best and a claim relearn's unbounded
    # figures cannot support at worst.
    record = provenance if isinstance(provenance, CycleProvenance) else begin_cycle()
    existing = read_cache(p, config=config) or {}
    # Explicit annotation: without it mypy infers the dict-literal's value
    # type from the two initial values (str, dict[str, Any]) and joins them
    # down to `Collection[str]`, rejecting the `cost_computed_at` assignment
    # below even though the payload is really `dict[str, Any]`.
    payload: dict[str, Any] = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "finding": stamp_proposal_ids(asdict(finding)),
        # The build that produced these clusters — see `report_store.write_report`
        # on why a timestamp alone lets a previous build's figures read as merely
        # recent across an upgrade. Taken off the cycle's record rather than
        # resolved here: two stores each calling `tj_build()` for themselves is
        # two copies of "what version am I" that are free to disagree.
        "tj_version": record.build,
        "provenance": record.to_dict(),
    }
    if "cost_proposals" in existing:
        # THE WHITELIST WAS THE TRAP. This used to copy forward a hand-picked
        # list of `cost_*` keys, and every new key a cost write introduced had
        # to be added here or this write silently dropped it — the symptom was
        # never an error, just a field that read as "never stamped". Caught
        # live twice: `cost_tj_version` (added to `write_cost_proposals` and
        # omitted here, so a freshly-booted daemon served a cost payload
        # claiming an unknown producing build) and `cost_proposals_error`/
        # `cost_proposals_error_at` (a recorded recompute failure silently
        # erased by the very next relearn-leg recompute, reading downstream as
        # if nothing had failed). Copy forward by PREFIX instead: this branch
        # (unlike `write_cost_proposals_error`/`clear_cost_proposals_error`,
        # which preserve the WHOLE existing payload) only ever originates
        # `computed_at`/`finding`/`tj_version`/`provenance` above — the cost
        # producer's own record is `cost_provenance`, so it copies forward here
        # like every other `cost_` key rather than needing to be named — so
        # every `cost_`-prefixed
        # key in `existing` belongs to the cost-proposal producer and none of
        # them can collide with a key this branch means to set itself.
        for key, value in existing.items():
            if key.startswith("cost_"):
                payload[key] = value
    _atomic_write(p, payload)
    return payload


def _atomic_write(p: Path, payload: dict[str, Any]) -> None:
    """Temp-file + rename write; a concurrent reader never sees a partial file.
    Best-effort — an OSError (read-only fs, missing parent that can't be made)
    degrades to a no-op, never raises."""
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        pass


def read_cost_proposals(
    path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any] | None:
    """The last-written cost-proposal block, or ``None`` if none has ever been
    computed AND no recompute has ever failed either (a genuinely fresh
    install). Shape: ``{"cost_computed_at": iso, "cost_proposals": [dict,
    ...], "cost_proposals_error": str | None, "cost_proposals_error_at": iso |
    None, "cost_window_days": int, "cost_excluded": dict}``.
    ``cost_proposals_error``
    is the last recompute failure's message (behavioral requirement #5) —
    present alongside a GOOD ``cost_proposals`` list when a later recompute
    failed after an earlier one had already succeeded, so a transient failure
    never hides the last good result. ``cost_window_days`` is the window the
    stored figures were OBSERVED over, so the headline names the window its
    own data came from rather than whatever picker the reader's screen is set
    to. The pace inputs this block used to also carry (``cost_active_days`` /
    ``cost_n_sessions``) are gone with the projection they fed: nothing in the
    cost pipeline paces a figure any more, and a cached pace input is an
    invitation to re-derive one. ``cost_excluded`` is the rollup's cross-reference for
    waste a caller deliberately did NOT sum in — generic infrastructure with
    no current occupant (``summarize`` used to be the one entry here until
    it got a real peer card instead; see ``cost_proposals.
    COST_ANALYZERS``); ``{}`` for a cache written before it existed, and
    ``{}`` going forward until a future analyzer needs it again."""
    raw = read_cache(path, config=config)
    if raw is None:
        return None
    has_proposals = "cost_proposals" in raw
    has_error = "cost_proposals_error" in raw
    if not has_proposals and not has_error:
        return None
    # Project by PREFIX rather than a hand-written dict — this used to be the
    # SECOND whitelist a `cost_*` key had to be named in (the first was the
    # round-trip loop in `write_cache`, now also prefix-based). Both dropped
    # an unnamed key silently, and the symptom was a field that read as
    # "never stamped" rather than an error. Every key the cost-proposal
    # producer writes is prefixed `cost_`, and `write_cache` never originates
    # one of its own (see the comment there), so a plain prefix filter over
    # the raw cache is exactly the cost block.
    block: dict[str, Any] = {k: v for k, v in raw.items() if k.startswith("cost_")}
    # Defaulting behavior callers depend on: absent/falsy reads as the empty
    # value below rather than `None`, so a caller's own `or` chain isn't
    # needed at every call site.
    block["cost_proposals"] = block.get("cost_proposals") or []
    block["cost_window_days"] = block.get("cost_window_days") or 0
    block["cost_excluded"] = block.get("cost_excluded") or {}
    block["cost_proposals_by_persona"] = block.get("cost_proposals_by_persona") or {}
    block["cost_relearn_by_persona"] = block.get("cost_relearn_by_persona") or {}
    # Absent on any cache written before per-persona proposals existed, and
    # `False` is the truthful reading of such a block: it holds one whole-corpus
    # list and cannot answer for a persona.
    block["cost_persona_scoped"] = bool(block.get("cost_persona_scoped"))
    # Stable shape: these read as `None` (never guessed/derived) on a cache
    # written before the field existed, same as before prefix-projection.
    for key in (
        "cost_computed_at", "cost_proposals_error", "cost_proposals_error_at",
        "cost_since", "cost_until", "cost_tj_version", "cost_provenance",
    ):
        block.setdefault(key, None)
    return block


def write_cost_proposals_error(
    message: str, path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any]:
    """Record a cost-proposals recompute failure (behavioral requirement #5),
    preserving whatever ``cost_proposals``/``finding`` block already exists —
    a failed refresh must never wipe the last GOOD result, only annotate that
    the most recent attempt didn't produce a fresher one. Atomic; best-effort
    on I/O error (mirrors every other write in this module)."""
    p = path or default_cache_path(config)
    existing = read_cache(p, config=config) or {}
    payload = dict(existing)
    payload["cost_proposals_error"] = message
    payload["cost_proposals_error_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(p, payload)
    return payload


def clear_cost_proposals_error(
    path: Path | None = None, *, config: TjConfig | None = None,
) -> dict[str, Any]:
    """Clear a previously-recorded cost-proposals error after a SUCCESSFUL
    recompute — called by ``cost_proposals.recompute_cost_proposals`` right
    after it writes a fresh good result, so a one-off transient failure
    doesn't keep flagging the tab as degraded once refreshes recover."""
    p = path or default_cache_path(config)
    existing = read_cache(p, config=config) or {}
    if "cost_proposals_error" not in existing and "cost_proposals_error_at" not in existing:
        return existing
    payload = {
        k: v for k, v in existing.items()
        if k not in ("cost_proposals_error", "cost_proposals_error_at")
    }
    _atomic_write(p, payload)
    return payload


def write_cost_proposals(
    proposals: list[Any], path: Path | None = None, *, config: TjConfig | None = None,
    window_days: int | None = None, excluded: dict[str, Any] | None = None,
    since: str | None = None, until: str | None = None,
    provenance: CycleProvenance | None = None,
    by_persona: dict[str, list[Any]] | None = None,
    relearn_by_persona: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the cost proposals into the SAME cache file the relearn finding
    lives in, under a separate ``cost_proposals`` key, preserving the relearn
    ``finding`` block. ``proposals`` is a list of ``CostProposal`` (or plain
    dicts). Atomic; best-effort on I/O error.

    ``window_days`` is the window this recompute ran over, stored alongside
    the proposals so a later reader labels the figures with the window they
    were actually observed over rather than one it assumes. ``None`` (the
    default) leaves any previously-stored value untouched, so a caller that
    doesn't track it yet (a legacy call site) doesn't silently zero out a real
    prior value. It is a LABEL, never a divisor: nothing rescales a stored
    figure by it.

    ``since``/``until`` are the RESOLVED BOUNDS that recompute actually ran
    over, stored beside the length. A length alone is not provenance: the
    stored analyzer report records ``scan_since``/``scan_until``, and while
    this store recorded only a day count the two surfaces' windows could not
    be compared from the artifacts at all — so a per-analyzer disagreement
    between them was undiagnosable without instrumenting a live daemon, and
    two successive explanations for one were asserted on that missing evidence
    and were wrong. Same "leave untouched when ``None``" rule as
    ``window_days``, for the same legacy-call-site reason.

    THOSE TWO SPELLINGS ARE NOW ONE TYPE. ``cost_since``/``cost_until`` here and
    ``scan_since``/``scan_until`` on the report were two raw-dict conventions
    for one fact, with nothing forcing them to describe the same span. Both are
    now DERIVED from ``provenance`` — the cycle's own
    :class:`~tokenjam.core.optimize.cycle_provenance.CycleProvenance`, minted
    once per pass — and stored under ``cost_provenance`` so a reader can compare
    the two artifacts' windows, builds and cycle ids directly. The two
    string keyword arguments stay for callers with no cycle and win over the
    record when given, exactly as ``window_days`` does.

    ``excluded`` is the rollup's cross-reference block for waste a caller
    deliberately did not fold in as a peer card — generic infrastructure with
    no current occupant (see ``read_cost_proposals``) — always written when
    this call succeeds (unlike ``window_days`` above, this one has no "leave
    untouched" case: a fresh recompute always knows the current excluded
    state, even if it's "nothing").

    ``by_persona`` is the SAME proposals recomputed once per persona over that
    persona's own sessions, keyed by persona name. It exists because a dollar
    figure cannot be narrowed after the fact: ``cost_proposals`` above is one
    list built over the whole corpus, and no read-time filter can extract one
    persona's money from it. A caller with per-persona reports passes them
    here; one without passes ``None``.

    ``None`` writes ``cost_persona_scoped: false`` and an EMPTY per-persona
    map, deliberately clearing any previous one. Keeping a stale per-persona
    block beside a fresher whole-corpus list is the torn artifact
    ``cycle_provenance`` exists to prevent — two measurements taken at two
    different moments presented as one. A reader that asks for a persona this
    block cannot answer for must render NOT-YET-KNOWN, never these figures
    under that persona's label."""
    from dataclasses import is_dataclass

    p = path or default_cache_path(config)
    existing = read_cache(p, config=config) or {}
    # `is_dataclass()` alone narrows to `DataclassInstance | type[DataclassInstance]`
    # (it also accepts a dataclass *class*), but `asdict()` only accepts an
    # instance. Excluding `type` narrows to the instance case for mypy and
    # matches what we actually want here — `proposals` holds instances, never
    # classes.
    serialised = [
        asdict(pr) if is_dataclass(pr) and not isinstance(pr, type) else dict(pr)
        for pr in proposals
    ]
    record = provenance if isinstance(provenance, CycleProvenance) else begin_cycle(
        config, since=since, until=until, window_days=window_days,
    )
    payload = dict(existing)
    payload["cost_proposals"] = serialised
    payload["cost_computed_at"] = datetime.now(timezone.utc).isoformat()
    payload["cost_tj_version"] = record.build
    payload["cost_provenance"] = record.to_dict()
    effective_days = window_days if window_days is not None else record.window_days
    if effective_days is not None:
        payload["cost_window_days"] = effective_days
    effective_since = since if since is not None else record.since
    if effective_since is not None:
        payload["cost_since"] = effective_since
    effective_until = until if until is not None else record.until
    if effective_until is not None:
        payload["cost_until"] = effective_until
    payload["cost_excluded"] = excluded or {}
    payload["cost_proposals_by_persona"] = {
        persona: [
            asdict(pr) if is_dataclass(pr) and not isinstance(pr, type) else dict(pr)
            for pr in rows
        ]
        for persona, rows in (by_persona or {}).items()
    }
    # The FLAG is what a reader checks, not the map's emptiness: a persona
    # whose pass legitimately produced no proposals is an empty list under a
    # scoped ledger, which is a different statement from a ledger that was
    # never scoped at all.
    # Relearn's clusters reach the inbox headline through
    # `inbox_contribution`, not through `cost_proposals`, so a persona-scoped
    # rollup needs that persona's own lane-partitioned relearn finding here or
    # it would fold the whole corpus's back in.
    payload["cost_relearn_by_persona"] = dict(relearn_by_persona or {})
    payload["cost_persona_scoped"] = bool(by_persona)
    _atomic_write(p, payload)
    return payload


def is_computing() -> bool:
    return _COMPUTING.is_set()


def recompute_now(
    conn: Any | None, *, cache_path: Path | None = None, config: Any | None = None,
) -> dict[str, Any] | None:
    """Synchronous compute + cache write on the CALLING thread/connection.

    Returns ``None`` (no-op) if a recompute is already in flight elsewhere —
    never blocks waiting for the other one to finish. Callers that want
    non-blocking HTTP-request behaviour should run this on a background
    thread instead (see ``trigger_background_recompute``).
    """
    if not _LOCK.acquire(blocking=False):
        return None
    _COMPUTING.set()
    try:
        # `[loop].transcript_path` lets a Claude Agent SDK app point the loop at
        # its OWN transcript root instead of ~/.claude/projects. None keeps the
        # historical env/default resolution.
        #
        # The analyzer scope is consulted FIRST: this daemon job is a second
        # entry point into the same scan, so a scope honored only by
        # `analyzers/relearn.run` would leak the machine's global transcript
        # tree back in through the cache the served routes read. See
        # `core/optimize/scope.py`.
        from tokenjam.core.optimize.scope import (
            resolve_analyzer_scope,
            resolve_write_scope,
        )

        scope = resolve_analyzer_scope(config)
        if not scope.enabled:
            return None
        projects_root = scope.projects_root if scope.source == "flag" else None
        if projects_root is None and config is not None:
            try:
                from tokenjam.core.transcript import loop_transcript_root

                projects_root = loop_transcript_root(config)
            except Exception:
                projects_root = None
        # The persistent per-file parse cache (core.transcript_cache): this
        # background job re-scans the FULL corpus on every scheduled tick, so
        # warming/reusing the cache here is where the recurring cost actually
        # gets paid down across ticks, not just within one `tj optimize` run.
        transcript_cache_dir = None
        if config is not None:
            try:
                from tokenjam.core.transcript_cache import default_cache_dir

                transcript_cache_dir = default_cache_dir(config)
            except Exception:
                transcript_cache_dir = None
        # WINDOW-SCOPED persona, matching `runner.build_report`. This used to
        # classify over the full corpus, on the argument that relearn's own
        # evidence is unbounded — a defensible reading in isolation, and a bug
        # across surfaces. Persona gates WHICH ANALYZERS RUN
        # (`PERSONA_DISABLED_ANALYZERS`) and whether relearn may offer a
        # workspace write, so a corpus whose recent window is claude-code
        # dominant but whose full history is mixed (or the reverse) resolved a
        # DIFFERENT gate here than on the report the Dashboard reads. That is
        # two surfaces disagreeing about which findings exist, not about a
        # figure's size. One derivation, over the window every other figure is
        # published on (`core/optimize/report_window.py`).
        #
        # In the daemon's normal path this branch is not even reached: the scan
        # cycle writes this cache from the report pass's own relearn finding,
        # which already carries the report's persona. It stands for a STANDALONE
        # recompute, and it has to agree with the cycle rather than diverge the
        # moment someone calls it directly.
        persona = "unknown"
        if conn is not None:
            try:
                from tokenjam.core.framing import (
                    agent_persona_mix,
                    config_declared_plan,
                    dominant_persona,
                )

                from datetime import timedelta

                from tokenjam.core.optimize.report_window import report_window_days
                from tokenjam.utils.time_parse import utcnow

                until = utcnow()
                since = until - timedelta(days=report_window_days(config, conn))
                persona = dominant_persona(
                    agent_persona_mix(conn, since, until),
                    declared_plan=config_declared_plan(config),
                )
            except Exception:
                persona = "unknown"
        finding = compute_relearn_finding(
            conn, projects_root=projects_root, transcript_cache_dir=transcript_cache_dir,
            persona=persona,
            # The apply target has to agree with the scope the findings came
            # from — a card whose evidence is scoped and whose write target is
            # not describes two different machines. Routed through
            # `resolve_write_scope` rather than reading `scope.claude_home`
            # directly, because the API's write guard authorizes against the
            # OTHER half of that same type; deriving the two independently is
            # what let the suggestion and the guard disagree.
            claude_home=resolve_write_scope(scope=scope).suggest_root,
            # Scoped like every other relearn artifact — the distill cache is a
            # SECOND cache beside this module's own, and it wrote real files
            # under the real ~/.tj even from an isolated config until it was
            # threaded through the same `_storage_base_dir`.
            distill_cache_dir=_distill_cache_dir_for(config),
            # The archive lane's horizon: what tokenjam kept, not what Claude
            # Code left on disk. See `compute_relearn_finding`.
            retention_days=retention_days_for(
                getattr(config, "storage", None)
            ) if getattr(config, "storage", None) is not None else None,
        )
        # cache_path, when omitted, resolves via `config` (honors --config /
        # storage.path, and a :memory:/"" storage.path never falls through to
        # the real ~/.tj — see default_cache_path).
        #
        # A STANDALONE pass is still a cycle of one, so it mints its own record
        # rather than leaving the cache unattributed — carrying the persona this
        # run resolved above, so the record never classifies the window a second
        # time. The daemon's normal path does not reach here: `scan_cycle` hands
        # `write_cache` the record the whole pass shares.
        result = write_cache(
            finding, cache_path, config=config,
            provenance=begin_cycle(config, conn=conn, persona=persona),
        )
        return result
    finally:
        _COMPUTING.clear()
        _LOCK.release()


def trigger_background_recompute(
    backend_factory: Callable[[], Any],
    *,
    cache_path: Path | None = None,
    config: Any | None = None,
) -> bool:
    """Fire-and-forget a recompute on a daemon thread.

    ``backend_factory`` builds a FRESH ``StorageBackend`` (e.g.
    ``lambda: DuckDBBackend(config.storage)``) — never the caller's live
    request connection, so the scan's DuckDB read never contends with a
    concurrent writer. The backend is closed when the job finishes. Returns
    ``False`` (no-op, nothing started) if a recompute is already running.

    ``config`` (optional): passed straight through to ``recompute_now`` so
    its Phase 3 verify pass can locate ``applied_fixes.json``.
    """
    if is_computing():
        return False

    def _job() -> None:
        backend = None
        try:
            backend = backend_factory()
            conn = getattr(backend, "conn", None)
            recompute_now(conn, cache_path=cache_path, config=config)
        except Exception:
            # Best-effort background job — never crash the scheduler/thread pool.
            pass
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    threading.Thread(target=_job, name="relearn-recompute", daemon=True).start()
    return True
