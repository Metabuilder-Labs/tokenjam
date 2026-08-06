"""
Optimize orchestrator. Builds OptimizeReport by running selected analyzers
from the registry in a deterministic order.

Analyzer ordering matters: budget-projection reads ctx.report.downgrade,
so downsize must run first when both are selected.
"""
from __future__ import annotations

import logging

from dataclasses import asdict
from datetime import datetime
from typing import Any

from tokenjam.core.config import TjConfig
from tokenjam.core.framing import agent_persona_mix, config_declared_plan, dominant_persona
from tokenjam.utils.time_parse import utcnow
from tokenjam.core.optimize.registry import ANALYZER_REGISTRY
from tokenjam.core.optimize.scope import resolve_analyzer_scope
from tokenjam.core.optimize.types import (
    AnalyzerContext,
    OptimizeReport,
    WindowSummary,
)

# Ensure analyzers are imported (triggers @register side effects).
# Auto-discovery in analyzers/__init__.py walks the directory.
from tokenjam.core.optimize import analyzers as _analyzers  # noqa: F401

log = logging.getLogger(__name__)

# Deterministic order. Adding a new analyzer? Append to this list. Analyzers
# requested via positional are filtered against ANALYZER_REGISTRY but executed
# in the order defined here, so cross-analyzer dependencies stay stable.
ANALYZER_ORDER: list[str] = [
    "downsize",
    "budget-projection",
    "cache",
    "cache-recommend",
    "resend",
    "script",
    "reuse",
    "trim",
    "subagent",
    "summarize",
    "relearn",
    "verbosity",
    "deadweight",
    "stream-usage",
]

# Analyzers that have NO fix a user of that persona can actually apply.
#
# Keyed by the window's dominant persona (`core.framing.dominant_persona`:
# "claude-code" | "sdk" | "mixed" | "unknown"). A name listed for the dominant
# persona is dropped from `selected` in `build_report` BEFORE the dispatch
# loop, so the analyzer is never invoked: it runs no query, produces no
# finding, and is therefore absent from every downstream surface (the CLI
# report, the /optimize payload, the Review inbox). That is deliberately a
# TRUE skip and not an `enabled: False` finding — a disabled analyzer that
# still queries costs the same time and then renders a row whose only honest
# caption is "there is nothing you can do about this."
#
# The bar for keeping an analyzer is not "a fix could be described" — it is:
#   1. the output ends in a concrete edit to a file or setting this persona
#      actually controls;
#   2. the user is better off NET of the fix's own standing cost (an
#      always-loaded instruction file is re-sent on every future session);
#   3. the saving does not come from making the agent terser or dumber.
# An analyzer that misses any one of those has no business spending query
# time for that persona. Each entry below records which one it misses.
#
# `mixed` / `sdk` / `unknown` disable nothing: the conservative default is to
# run everything, so an unclassified window never silently loses a finding.
PERSONA_DISABLED_ANALYZERS: dict[str, frozenset[str]] = {
    # An interactive coding agent's harness constructs the API request, picks
    # the model for its own main thread, and owns the prompt template. The
    # user only controls their workspace config files. Everything below is a
    # lever that lives on the other side of that line.
    "claude-code": frozenset({
        # Cache efficacy is measured off `cache_control` breakpoint placement,
        # which the harness sets and the user cannot reach — no request field
        # of theirs to change, so the finding is a diagnostic with no fix.
        "cache",
        # Same missing lever as `cache` above: even when the analyzer finds a
        # stable, repeated prefix — it can, for Claude-Code-sourced spans, via
        # the project's CLAUDE.md content backfilled onto them (#272, see
        # cache_recommend.py's module docstring) — a Claude Code user still
        # has no way to set `cache_control` on the request the harness
        # builds, so a candidate would be a diagnostic with no fix. This
        # entry is what keeps the CLAUDE.md-prefix path from ever reaching
        # that dead end: it drops `cache-recommend` from dispatch before this
        # persona's window is analyzed, so the path only actually runs for
        # `sdk`/`mixed` windows (persona is computed per-window, see
        # `dominant_persona`), where it can serve a real fix.
        "cache-recommend",
        # Batch placement recommends re-laning work onto a delayed batch
        # endpoint. Unreachable from an interactive session by definition —
        # the whole point is that a human is waiting on the answer. Not a
        # registry name (the `downsize` analyzer attaches it as a sub-check),
        # so `downsize` reads this same map to skip it — see model_downgrade.
        "placement",
        # Trim predicts low-significance regions of a prompt template so the
        # author can shorten it. That presumes editable prompt-template source
        # code, which an interactive coding-agent user does not have.
        "trim",
        # Verbosity's detection is sound, but its only remedy is a global
        # "be concise, answer in the fewest words" instruction written into an
        # always-loaded file — buying tokens by making the agent terser
        # everywhere, off a finding that was scoped to one cohort of sessions.
        # That is a quality tax, which is not a trade this product makes.
        "verbosity",
        # Script clusters a session by its ENTIRE ordered (tool, arg-shape)
        # tuple, so one extra file read or a reordered call breaks the match.
        # It found zero clusters across ~1.2k real coding sessions, and the
        # cluster threshold is not the bottleneck — the signature is.
        # DECISION: disabled for this persona now rather than silently
        # retired; re-enable if the signature is redesigned to a subsequence
        # or prefix match that tolerates heterogeneous coding work.
        "script",
        # Reuse asserts something no telemetry can establish for an
        # interactive coding agent: that two planning calls were semantically
        # interchangeable. Measured against a real corpus, the clustering that
        # backs the claim has no content signal (every member's prompt-prefix
        # hash is null, so clustering falls back to bare tool-name sequence),
        # no time window, no prior-failure exclusion, and the large majority
        # of its dollar figure comes from a null-tool-signature catch-all
        # bucket ("nothing followed the plan") rather than any actual
        # repeated plan. DECISION: disabled for this persona; the concept
        # stays worth rebuilding behind a real content signal, a recency
        # window, and a prior-failure exclusion. This gate applies to
        # `claude-code` only — the defects measured above are specific to
        # this corpus; the SDK case is a separate, unmeasured question and is
        # deliberately left ungated.
        "reuse",
        # Stream-usage flags streamed calls that closed before the provider
        # emitted its usage payload, so their spend went unrecorded. Its fix
        # is a request-side change — `stream_options={"include_usage": True}`,
        # or draining the stream server-side — to code that constructs the
        # provider call. An interactive coding agent's harness constructs that
        # call, which puts the lever on the other side of the actionable
        # ceiling; and the harness drains its own streams, so the failure mode
        # does not arise here in the first place. This is an SDK-persona
        # finding: it stays enabled for `sdk` / `mixed` / `unknown`.
        "stream-usage",
    }),
    # An SDK/API window has no on-disk Claude Code transcript and never
    # populates `sub_agent_id` (no Task-tool subagent-dispatch concept in
    # generic SDK telemetry) — both analyzers below are gated on a DATA
    # SOURCE that structurally does not exist for this persona, not on a
    # missing lever, so every dispatch would run a real query and still
    # return nothing to act on.
    "sdk": frozenset({
        # Reads project `.mcp.json` / `.claude/settings*.json` / on-disk
        # Claude Code `.jsonl` transcripts (see deadweight.py's module
        # docstring, "Claude Code transcripts lane only") — an SDK window has
        # none of these, so this always renders a permanently-empty card.
        "deadweight",
        # Scopes to `sub_agent_id IS NOT NULL`, which is NULL for every SDK
        # span (see subagent_rightsizing.py's `_compute_rows`) — an SDK
        # window can never have a row here, so this always renders a
        # permanently-empty card.
        "subagent",
    }),
}


def disabled_analyzers_for_persona(persona: str) -> frozenset[str]:
    """Analyzer names with no applicable fix for `persona`.

    Single source of truth for the persona skip gate — `build_report` uses it
    to drop analyzers before dispatch, the `downsize` analyzer uses it for its
    `placement` sub-check, and `cost_proposals` uses it so the Review inbox
    makes the same selection. Unknown/unlisted personas disable nothing.
    """
    return PERSONA_DISABLED_ANALYZERS.get(persona, frozenset())


THIN_DATA_DAYS = 7


def _utcnow() -> datetime:
    # Canonical timezone-aware UTC (CLAUDE.md Rule 9).
    return utcnow()


def summarize_window(
    conn,
    since: datetime,
    until: datetime,
    agent_id: str | None = None,
) -> WindowSummary:
    clauses = ["start_time >= $1", "start_time < $2", "model IS NOT NULL"]
    params: list[Any] = [since, until]
    if agent_id:
        clauses.append(f"agent_id = ${len(params) + 1}")
        params.append(agent_id)
    where = " AND ".join(clauses)
    row = conn.execute(
        f"SELECT COUNT(*) AS spans, "
        f"COUNT(DISTINCT session_id) AS sessions, "
        f"COUNT(DISTINCT CAST(start_time AS DATE)) AS active_days, "
        f"COALESCE(SUM(COALESCE(input_tokens,0) + COALESCE(output_tokens,0) + COALESCE(cache_tokens,0) + COALESCE(cache_write_tokens,0)), 0) AS tokens, "
        f"COALESCE(SUM(cost_usd), 0.0) AS cost "
        f"FROM spans WHERE {where}",
        params,
    ).fetchone()
    spans = int(row[0] or 0)
    sessions = int(row[1] or 0)
    active_days = int(row[2] or 0)
    tokens = int(row[3] or 0)
    cost = float(row[4] or 0.0)
    days = max((until - since).total_seconds() / 86400.0, 0.0)
    return WindowSummary(
        since=since,
        until=until,
        days=days,
        sessions=sessions,
        active_days=active_days,
        spans=spans,
        total_tokens=tokens,
        total_cost_usd=cost,
        thin_data=days < THIN_DATA_DAYS or sessions < 3,
    )


def build_report(
    db,
    config: TjConfig,
    since: datetime,
    until: datetime | None = None,
    agent_id: str | None = None,
    findings: list[str] | None = None,
    budget_provider_filter: str | None = None,
    budget_usd_override: float | None = None,
) -> OptimizeReport:
    """
    Build a complete OptimizeReport.

    `findings`:
      - None  -> run all registered analyzers in ANALYZER_ORDER
      - list  -> run only the named analyzers (must be keys in ANALYZER_REGISTRY)

    Analyzers are executed in ANALYZER_ORDER, never in caller-supplied order,
    so dependent analyzers (e.g. budget-projection reading the downgrade
    finding) work correctly regardless of how the caller lists them.
    """
    until = until or _utcnow()
    if until <= since:
        raise ValueError("until must be after since")

    conn = getattr(db, "conn", None)
    if conn is None:
        raise RuntimeError("optimize requires a direct DuckDB connection")

    summary = summarize_window(conn, since, until, agent_id=agent_id)
    window_days = max(summary.days, 1.0 / 86400.0)

    # Dominant persona for this window, computed exactly once — see
    # `AnalyzerContext.persona` / `OptimizeReport.persona`. Same functions
    # (`agent_persona_mix` / `dominant_persona`) the CLI uses for its own
    # persona-dependent CTA (`cmd_optimize._render_downgrade_cta`); this is
    # not a second classifier, just a second place that needed the answer.
    agent_mix = agent_persona_mix(conn, since, until, agent_id=agent_id)
    persona = dominant_persona(agent_mix, declared_plan=config_declared_plan(config))

    report = OptimizeReport(window=summary, persona=persona)
    if summary.thin_data:
        report.notes.append(
            "Window contains less than ~1 week of activity — projections shown "
            "below should be treated as preliminary."
        )

    ctx = AnalyzerContext(
        conn=conn,
        # The owning backend's write lock, so an analyzer that writes through
        # the raw `conn` can serialize against the backend's own mutating
        # methods (`.claude/rules/core-architecture.md`). `None` when the caller
        # passed something without one; the agent-config store then relies on
        # its retry path instead, which is slower under contention, not wrong.
        write_lock=getattr(db, "write_lock", None),
        config=config,
        since=since,
        until=until,
        agent_id=agent_id,
        window_days=window_days,
        summary=summary,
        report=report,
        budget_provider_filter=budget_provider_filter,
        budget_usd_override=budget_usd_override,
        persona=persona,
        # Resolved exactly once, here, for the same reason the persona is: an
        # analyzer that re-derives its own root from `Path.home()` or the env
        # var escapes whatever scope the caller drew, and `--db` stops meaning
        # anything. See `core/optimize/scope.py`.
        scope=resolve_analyzer_scope(config),
    )

    selected = set(findings) if findings is not None else set(ANALYZER_REGISTRY.keys())
    # Validate against registry; raise on unknown names so typos surface early.
    unknown = selected - set(ANALYZER_REGISTRY.keys())
    if unknown:
        raise ValueError(
            f"Unknown finding(s): {sorted(unknown)}. "
            f"Available: {sorted(ANALYZER_REGISTRY.keys())}"
        )

    # Persona skip gate. Applied AFTER validation (so a typo still raises) and
    # BEFORE dispatch, which is what makes it a true skip: an analyzer removed
    # here is never invoked and runs no query. Unconditional — every caller
    # that selects analyzers (the CLI, /api/v1/optimize, the Review inbox's
    # COST_ANALYZERS recompute, the status teaser) funnels through this one
    # choke point, so none of them can reintroduce a finding this persona
    # cannot act on. See PERSONA_DISABLED_ANALYZERS for the per-name reasons.
    selected -= disabled_analyzers_for_persona(persona)

    def _dispatch(name: str, analyzer: Any) -> None:
        """Run one analyzer, and DISCLOSE it if it fails.

        The loop used to be bare, so one analyzer raising took `build_report`
        down with it and lost every other analyzer's findings — a report of
        thirteen analyzers destroyed by one. That was survivable while every
        analyzer was pure in-memory computation over already-fetched rows; it
        stopped being survivable once one of them could hit a database write
        conflict, which is a failure mode that simply did not exist before.

        Isolation alone would be the WORSE bug, though. An analyzer that
        vanishes silently reads as "this analyzer found nothing", which is a
        positive claim the run has no evidence for — exactly the class of defect
        (root anti-pattern 22) this file's own persona gate is careful about
        when it drops an analyzer deliberately. So a swallowed failure is
        recorded on the report, in a field surfaces render, and never merely
        omitted.
        """
        try:
            analyzer(ctx)
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            from tokenjam.core.db import is_fatal_db_error

            if is_fatal_db_error(exc):
                # NOT survivable, and not this analyzer's failure. A fatal
                # invalidates the whole database instance, so every analyzer
                # after this one would fail too and the "did not complete"
                # note would be attached to all of them for the wrong reason.
                # Recording it as one analyzer's problem and continuing is how
                # this fatal reached the end of a pass without anything
                # recovering the connection.
                report.analyzer_errors[name] = f"{type(exc).__name__}: {exc}"
                report.notes.append(
                    f"The database became unusable while running `{name}`, so "
                    f"this report is incomplete. Nothing here says an analyzer "
                    f"found nothing."
                )
                raise
            log.exception("analyzer %s failed; continuing with the rest", name)
            report.analyzer_errors[name] = f"{type(exc).__name__}: {exc}"
            report.notes.append(
                f"The `{name}` analyzer did not complete ({type(exc).__name__}), "
                f"so this report says nothing about it. That is not the same as "
                f"it finding nothing — re-run to try again."
            )

    for name in ANALYZER_ORDER:
        if name in selected and name in ANALYZER_REGISTRY:
            _dispatch(name, ANALYZER_REGISTRY[name])

    # Analyzers not in ANALYZER_ORDER (future ones, registered but not yet
    # explicitly ordered) run last in arbitrary order. Maintainers should add
    # new analyzers to ANALYZER_ORDER when they land.
    for name, analyzer in ANALYZER_REGISTRY.items():
        if name in selected and name not in ANALYZER_ORDER:
            _dispatch(name, analyzer)

    # THE write allocation, after every analyzer and before anyone reads the
    # report. This is the only place in the tree that decides which permanent
    # writes are OFFERED, and it is here because here is the first moment both
    # producers' candidates exist: relearn's clusters are a finding, and the
    # cost lane's cards are a pure transformation of the findings beside them.
    #
    # It has to be inside `build_report` rather than in any one caller. This is
    # the choke point every surface funnels through (the same argument the
    # persona skip gate above is made on), so a caller cannot forget to
    # allocate and publish a report that offers more writes than one budget
    # allows. Costs one pure adapter pass over findings already computed; it
    # runs no query and re-measures nothing.
    #
    # Never raises: an allocation that fails leaves the decisions empty, and an
    # empty allocation withdraws offers rather than inventing them.
    try:
        from tokenjam.core.optimize import write_allocation

        write_allocation.allocate_report_writes(
            report, config=config, window_days=window_days,
        )
    except Exception:  # noqa: BLE001 - see above
        pass

    return report


def report_to_dict(report: OptimizeReport) -> dict:
    """Convert OptimizeReport to a JSON-serialisable dict."""
    def _serialise(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if hasattr(o, "__dataclass_fields__"):
            return {k: _serialise(v) for k, v in asdict(o).items()}
        if isinstance(o, list):
            return [_serialise(x) for x in o]
        if isinstance(o, dict):
            return {k: _serialise(v) for k, v in o.items()}
        return o
    return _serialise(report)


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO-8601 string back into datetime; tolerate None / already-dt."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        s = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


def hydrate_dataclass(cls: Any, d: Any) -> Any:
    """Rebuild a dataclass from `report_to_dict`'s output, field by field, by
    INTROSPECTION rather than by a hand-written argument list.

    This exists because the hand-written version was not lossless, and the
    asymmetry was invisible: `report_to_dict` is a generic recursive walk that
    serialises every field, while each `report_from_dict` constructor named its
    fields by hand and silently fell back to dataclass defaults for anything it
    forgot. `resend` alone dropped 19 fields on the way back, including
    `cost_of_waste_usd` and `coverage_note`; `relearn` dropped
    `past_reread_usd` and its whole `below_threshold_*` block.

    That was survivable only while the live path handed the renderer a real
    `OptimizeReport` and just the HTTP shim round-tripped. Once analyzer
    results are served from a store, EVERY consumer round-trips, so a dropped
    dollar field comes back as its default and a surface renders a smaller
    number, or zero, with no error anywhere. A wrong figure that looks like a
    successful read is worse than a failed one.

    Coercion is driven by the resolved type hints: nested dataclasses and lists
    of them recurse, `datetime` fields go through `_parse_dt`, everything else
    passes through. A key absent from `d` is left to the dataclass default, so
    an older or foreign payload still loads.

    `tests/unit/test_report_roundtrip.py` fails if any dataclass field the
    serializer writes does not survive the trip back, so the next analyzer
    field cannot vanish silently the way these did.
    """
    import dataclasses
    import typing

    import sys

    if d is None or not dataclasses.is_dataclass(cls):
        return d
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {}

    # The namespace forward references resolve against. `get_type_hints` does
    # NOT reliably resolve a reference written INSIDE a subscript
    # (`list["UncachedAgentCandidate"]`): on Python 3.10 the inner argument
    # comes back as the bare string `'UncachedAgentCandidate'`, while 3.11+
    # returns the class. Without this namespace the string fails every
    # is_dataclass check below and the value passes straight through as a raw
    # dict — the field is present and correctly shaped, but the TYPE is gone,
    # so the failure surfaces far away as `'dict' object has no attribute ...`.
    # See `_resolve`.
    ns = getattr(sys.modules.get(cls.__module__), "__dict__", {})

    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue   # leave the dataclass default in place
        kwargs[f.name] = _coerce(_field_hint(f, hints, d[f.name]), d[f.name], ns)
    # `is_dataclass(cls)` narrows to DataclassInstance for mypy, which it then
    # refuses to call; `cls` here is always the CLASS, never an instance.
    ctor: Any = cls
    return ctor(**kwargs)


def _hydrate_target(f: Any) -> Any:
    """The class named by a field's ``hydrate`` metadata, imported lazily.

    A field whose real element type lives in a module the declaring module
    must not import (an analyzer row inside ``types``, which is deliberately
    analyzer-free) has no annotation the hydrator can dispatch on — it is
    written ``list[Any]``, and `Any` coerces to "leave the dict alone". The
    dict then travels as a correctly-shaped value with the TYPE missing, and
    the failure surfaces far away as ``'dict' object has no attribute ...``
    inside whichever consumer expected the row's attributes. That is the same
    class of silent loss `hydrate_dataclass` was written to end, one layer
    over: not a dropped field, a dropped type.

    So the type is declared as data instead of as an import: the field carries
    ``metadata={"hydrate": "package.module:ClassName"}``, and the class is
    imported HERE, at hydration time, where importing an analyzer is
    harmless. Unresolvable (a renamed class, a module that will not import)
    degrades to the annotation the field already had, which is exactly the
    old behaviour — this can restore a type, never break a load.
    """
    spec = (getattr(f, "metadata", None) or {}).get("hydrate")
    if not spec:
        return None
    module_name, _, qualname = str(spec).partition(":")
    if not module_name or not qualname:
        return None
    try:
        import importlib
        return getattr(importlib.import_module(module_name), qualname, None)
    except Exception:
        return None


def _field_hint(f: Any, hints: dict, value: Any) -> Any:
    """The type hint to coerce one field against, preferring its ``hydrate``
    metadata over a loose annotation. Container-shaped values keep their
    container (``list[Any]`` becomes ``list[Row]``, not ``Row``)."""
    import dataclasses
    import typing

    hint = hints.get(f.name, f.type)
    target = _hydrate_target(f)
    if target is None or not dataclasses.is_dataclass(target):
        return hint
    if typing.get_origin(hint) in (list, tuple) or isinstance(value, list):
        return list[target]      # type: ignore[valid-type]
    return target


def _resolve(hint: Any, ns: dict) -> Any:
    """Turn a forward reference into the class it names, if we can.

    Needed because a quoted reference inside a builtin generic subscript is not
    resolved uniformly across supported Pythons: `list["Candidate"]` yields the
    string on 3.10 and the class on 3.11+. Everything downstream dispatches on
    `is_dataclass`, which a string always fails, so an unresolved reference
    silently degrades a nested dataclass to the raw dict it was serialized
    from. Returning the hint unchanged when it cannot be resolved keeps an
    unknown name behaving exactly as before rather than raising.
    """
    if isinstance(hint, str):
        return ns.get(hint, hint)
    forward_arg = getattr(hint, "__forward_arg__", None)   # typing.ForwardRef
    if forward_arg is not None:
        return ns.get(forward_arg, hint)
    return hint


def _coerce(hint: Any, value: Any, ns: dict | None = None) -> Any:
    """Coerce one serialized value back to what its type hint asks for."""
    import dataclasses
    import typing

    if value is None:
        return None

    ns = ns or {}
    hint = _resolve(hint, ns)
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)

    # Optional[X] / X | None -> coerce against the non-None member.
    if origin is typing.Union or (origin is not None and str(origin) == "|"):
        inner = [a for a in args if a is not type(None)]   # noqa: E721
        return _coerce(inner[0], value, ns) if len(inner) == 1 else value
    try:
        import types as _types
        if isinstance(hint, _types.UnionType):        # PEP 604 `X | None`
            inner = [a for a in args if a is not type(None)]   # noqa: E721
            return _coerce(inner[0], value, ns) if len(inner) == 1 else value
    except Exception:
        pass

    if origin in (list, tuple) and args:
        return [_coerce(args[0], v, ns) for v in value]
    if origin is dict and len(args) == 2:
        return {k: _coerce(args[1], v, ns) for k, v in value.items()}
    if isinstance(hint, type) and issubclass(hint, datetime):
        return _parse_dt(value)
    if dataclasses.is_dataclass(hint) and isinstance(value, dict):
        return hydrate_dataclass(hint, value)
    return value


def report_from_dict(d: dict) -> OptimizeReport:
    """
    Reconstruct an OptimizeReport from the dict produced by `report_to_dict`.

    Symmetric with `report_to_dict`, and LOSSLESS — see `hydrate_dataclass`
    for why that word is load-bearing and what used to go missing.

    Used by the CLI when fetching an optimize report from a running `tj serve`
    via /api/v1/optimize (issue #68 §12), and by `core.optimize.report_store`
    for the handful of consumers that need typed findings rather than the
    stored dict.

    Wave-2 analyzer findings live under `d["findings"]` keyed by analyzer name
    and are hydrated against the dataclass registered for that name. An unknown
    name is dropped silently — that keeps the CLI forward-compatible when a
    newer daemon advertises a finding this install cannot render.
    """
    from tokenjam.core.optimize.types import (
        BudgetProjection,
        DowngradeFinding,
        WindowSummary,
    )

    # Every branch below goes through `hydrate_dataclass`, which reads the
    # dataclass's OWN fields rather than a hand-maintained argument list. That
    # is the whole point: the previous hand-written version silently dropped
    # any field nobody remembered to name, and a field added to an analyzer
    # tomorrow would have been dropped the same way.
    # `WindowSummary` declares no field defaults, so an absent or partial
    # `window` cannot be hydrated field-by-field the way everything else can.
    # A payload without one is legitimate (callers construct minimal report
    # dicts), so seed the required fields first and let the stored values
    # override whichever of them are present.
    now = _utcnow()
    window_seed: dict[str, Any] = {
        "since": now, "until": now, "days": 0.0, "sessions": 0, "spans": 0,
        "total_tokens": 0, "total_cost_usd": 0.0, "thin_data": False,
    }
    window_seed.update(d.get("window") or {})
    window = hydrate_dataclass(WindowSummary, window_seed)
    downgrade = (
        hydrate_dataclass(DowngradeFinding, d["downgrade"]) if d.get("downgrade") else None
    )
    budgets = [hydrate_dataclass(BudgetProjection, b) for b in (d.get("budgets") or [])]

    findings = {}
    for name, payload in (d.get("findings") or {}).items():
        finding_cls = _finding_class_for(name)
        if finding_cls is None:
            # Forward-compatible: ignore unknown findings rather than crash.
            continue
        try:
            findings[name] = hydrate_dataclass(finding_cls, payload)
        except Exception:
            # Don't let one malformed finding break the whole report.
            continue

    return OptimizeReport(
        window=window,
        downgrade=downgrade,
        budgets=budgets,
        notes=list(d.get("notes") or []),
        findings=findings,
        persona=str(d.get("persona", "unknown")),
        # Round-tripped like every other report-level field: a report rebuilt
        # from the daemon's cache must still be able to say its filesystem
        # analyzers never scanned, or a served surface silently reads an
        # unscanned report as a scanned-and-empty one.
        filesystem_scan_skipped_reason=d.get("filesystem_scan_skipped_reason") or None,
        # Round-tripped because the cost half of the write allocation is
        # applied by a LATER pass, on the other side of the report store — a
        # rehydrated report that lost these would re-allocate against a
        # candidate set the first pass has already edited. See
        # `core/optimize/write_allocation.py`.
        write_decisions=dict(d.get("write_decisions") or {}),
    )


# Dispatch table: finding-registration-name -> the dataclass that finding is.
#
# It maps to a TYPE, not to a constructor function. It used to hold ~330 lines
# of hand-written `Cls(field=d.get("field"), ...)` builders, and every one of
# them was a field list somebody had to keep in sync with the dataclass by
# hand. They were not in sync: `resend` dropped 19 fields on the way back,
# `relearn` dropped its whole `below_threshold_*` block, and nothing failed —
# the dropped fields simply came back as defaults. `hydrate_dataclass` reads
# the dataclass's own fields instead, so this table only has to answer "which
# class is this?" and can no longer drift.
#
# Filled lazily so importing runner.py doesn't require every analyzer module to
# be imported (analyzers self-register via auto-discovery; order matters during
# package init).
def _build_finding_classes() -> dict:
    from tokenjam.core.optimize.analyzers.batch_placement import BatchPlacementFinding
    from tokenjam.core.optimize.analyzers.cache_efficacy import CacheEfficacyFinding
    from tokenjam.core.optimize.analyzers.cache_recommend import CacheRecommendFinding
    from tokenjam.core.optimize.analyzers.context_resend import ResendFinding
    from tokenjam.core.optimize.analyzers.deadweight import DeadweightFinding
    from tokenjam.core.optimize.analyzers.output_verbosity import VerbosityFinding
    from tokenjam.core.optimize.analyzers.prompt_bloat import PromptBloatFinding
    from tokenjam.core.optimize.analyzers.relearn import RelearnFinding
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
        SubagentRightsizingFinding,
    )
    from tokenjam.core.optimize.analyzers.stream_usage import StreamUsageFinding
    from tokenjam.core.optimize.analyzers.summarize import SummarizeFinding
    from tokenjam.core.optimize.analyzers.workflow_restructure import (
        WorkflowRestructureFinding,
    )
    from tokenjam.core.optimize.types import ReuseFinding

    return {
        "cache": CacheEfficacyFinding,
        "cache-recommend": CacheRecommendFinding,
        "script": WorkflowRestructureFinding,
        "reuse": ReuseFinding,
        "trim": PromptBloatFinding,
        "subagent": SubagentRightsizingFinding,
        "summarize": SummarizeFinding,
        "relearn": RelearnFinding,
        "deadweight": DeadweightFinding,
        "verbosity": VerbosityFinding,
        "resend": ResendFinding,
        "stream-usage": StreamUsageFinding,
        # Not a registered analyzer name of its own: the downsize analyzer
        # attaches the batch-placement check under this key. It still needs an
        # entry, or the finding is dropped on the daemon path (every consumer
        # deserialises through report_from_dict, which ignores any finding name
        # absent from this table).
        "placement": BatchPlacementFinding,
    }


_FINDING_CLASSES: dict = {}


def _finding_class_for(name: str):
    """The dataclass for a finding name, or None if this install doesn't know it."""
    global _FINDING_CLASSES
    if not _FINDING_CLASSES:
        _FINDING_CLASSES = _build_finding_classes()
    return _FINDING_CLASSES.get(name)


def finding_class_names() -> list[str]:
    """Every finding name the round-trip can rebuild. Used by the round-trip
    losslessness test to enumerate what it must cover."""
    if not _FINDING_CLASSES:
        _finding_class_for("")
    return sorted(_FINDING_CLASSES)
