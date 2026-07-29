"""`report_to_dict` -> `report_from_dict` must not drop a single field.

Why this file exists
--------------------
`report_to_dict` is a generic recursive walk that serialises everything.
`report_from_dict` used to be ~330 lines of hand-written
`Cls(field=d.get("field"), ...)` builders — one argument list per finding, each
of which somebody had to keep in sync with its dataclass by hand. They were not
in sync, and nothing anywhere failed when they drifted: a field nobody named
simply came back as its dataclass default.

Measured casualties at the time this test was written: `resend` lost 19 fields
including `cost_of_waste_usd`, `cost_of_waste_tokens`, `cost_of_waste_basis`,
`rightsize_recoverable_usd`, `coverage_note` and `sessions_in_scope`; `relearn`
lost `past_reread_usd`, `past_reread_tokens`, `corpus_basis`, `window_days`,
`archived_sessions_scanned` and its four `below_threshold_*` fields; `cache`,
`script`, `reuse`, `trim`, `subagent` and `verbosity` each lost some too.

That was survivable only while the live path handed the renderer a real
`OptimizeReport` object and only the HTTP shim round-tripped. It stopped being
survivable the moment analyzer results started being served from a store,
because then EVERY consumer round-trips. A dropped dollar field comes back as
its default, and a surface renders a smaller number — or a zero — with no error
raised anywhere. A wrong figure that looks like a successful read is worse than
a read that visibly failed.

`hydrate_dataclass` now rebuilds by introspecting each dataclass's own fields,
so there is no argument list left to drift. This test is what keeps it that
way: it populates every field of every finding with a non-default sentinel and
fails if the trip back changes any of them. A field added to an analyzer
tomorrow is covered automatically — that is the whole point, since the previous
failure mode was precisely "nobody remembered to add it".
"""
from __future__ import annotations

import dataclasses
import sys
import types
import typing
from datetime import datetime

import pytest

from tokenjam.core.optimize.runner import (
    _finding_class_for,
    finding_class_names,
    hydrate_dataclass,
    report_from_dict,
    report_to_dict,
)
from tokenjam.core.optimize.types import (
    BudgetProjection,
    DowngradeFinding,
    OptimizeReport,
    WindowSummary,
)
from tokenjam.utils.time_parse import utcnow

# Depth cap for recursive/self-referential shapes. Two levels is enough to
# exercise nested dataclasses and lists of them without unbounded recursion.
_MAX_DEPTH = 2


def _sentinel(hint: object, name: str, depth: int, ns: dict | None = None) -> object:
    """A NON-DEFAULT value for `hint`, so a dropped field is always detectable.

    Sentinels matter more than they look: if the value happened to equal the
    dataclass default, a field the round-trip drops would still compare equal
    and the test would pass while the bug shipped.

    `ns` resolves forward references, and it is load-bearing for the same
    reason the production hydrator needs it. Without it this helper had the
    IDENTICAL Python 3.10 blind spot as the code it is meant to guard: a
    `list["Candidate"]` hint stayed a bare string, fell through to `None`, and
    the fixture built no nested dataclass at all — so the type guards below had
    nothing to inspect and passed against a hydrator that was demonstrably
    broken. A fixture that degrades in the same way as the code under test
    cannot catch that code.
    """
    ns = ns or {}
    if isinstance(hint, str):
        hint = ns.get(hint, hint)
    forward_arg = getattr(hint, "__forward_arg__", None)
    if forward_arg is not None:
        hint = ns.get(forward_arg, hint)
    origin, args = typing.get_origin(hint), typing.get_args(hint)

    if isinstance(hint, types.UnionType) or origin is typing.Union:
        inner = [a for a in args if a is not type(None)]   # noqa: E721
        return _sentinel(inner[0], name, depth, ns) if len(inner) == 1 else None
    if origin in (list, tuple) and args:
        return [] if depth >= _MAX_DEPTH else [_sentinel(args[0], name, depth + 1, ns)]
    if origin is dict and len(args) == 2:
        return {} if depth >= _MAX_DEPTH else {"k": _sentinel(args[1], name, depth + 1, ns)}
    if hint is bool:
        return True
    if hint is int:
        return 4242
    if hint is float:
        return 42.42
    if hint is str:
        return f"sentinel-{name}"
    if isinstance(hint, type) and issubclass(hint, datetime):
        return utcnow()
    if dataclasses.is_dataclass(hint):
        return _populate(hint, depth + 1)
    return None


def _populate(cls: object, depth: int = 0) -> object:
    """An instance of `cls` with EVERY field explicitly set."""
    hints = typing.get_type_hints(cls)
    ns = getattr(sys.modules.get(cls.__module__), "__dict__", {})
    kwargs = {
        f.name: _sentinel(hints.get(f.name, f.type), f.name, depth, ns)
        for f in dataclasses.fields(cls)
    }
    return cls(**kwargs)


def _lossy_fields(cls: object) -> list[str]:
    """Field names whose VALUE does not survive to_dict -> from_dict."""
    obj = _populate(cls)
    back = hydrate_dataclass(cls, report_to_dict(obj))
    return [
        f.name for f in dataclasses.fields(cls)
        if report_to_dict(getattr(obj, f.name)) != report_to_dict(getattr(back, f.name))
    ]


def _untyped_fields(value: object, path: str = "") -> list[str]:
    """Paths where a nested dataclass came back as a raw dict.

    Value equality alone CANNOT see this, and that blind spot shipped a red CI:
    `report_to_dict` serializes a dataclass and the plain dict it was built
    from to the identical structure, so a field that lost its TYPE compares
    equal and the value-only guard passed on every interpreter. The break only
    surfaced later, at attribute access, as `'dict' object has no attribute
    'agent_id'` — far from the cause, and only on the Python where the forward
    reference failed to resolve.

    So this walks the rebuilt object and reports anywhere a `dict` sits in a
    slot whose declared type is a dataclass.
    """
    bad: list[str] = []
    if not dataclasses.is_dataclass(value):
        return bad
    hints = typing.get_type_hints(type(value))
    for f in dataclasses.fields(value):
        got = getattr(value, f.name)
        want = hints.get(f.name, f.type)
        here = f"{path}.{f.name}" if path else f.name
        bad.extend(_check_slot(want, got, here, type(value)))
    return bad


def _check_slot(want: object, got: object, here: str, owner: type) -> list[str]:
    """One field: does `got` have the dataclass type `want` asks for?"""
    ns = getattr(sys.modules.get(owner.__module__), "__dict__", {})
    if isinstance(want, str):
        want = ns.get(want, want)
    origin, args = typing.get_origin(want), typing.get_args(want)

    if isinstance(want, types.UnionType) or origin is typing.Union:
        inner = [a for a in args if a is not type(None)]   # noqa: E721
        return _check_slot(inner[0], got, here, owner) if len(inner) == 1 and got is not None else []
    if origin in (list, tuple) and args:
        return [b for i, v in enumerate(got or []) for b in _check_slot(args[0], v, f"{here}[{i}]", owner)]
    if origin is dict and len(args) == 2:
        return [b for k, v in (got or {}).items() for b in _check_slot(args[1], v, f"{here}[{k!r}]", owner)]
    if isinstance(want, str):
        # Still a bare forward reference after resolution: we cannot say what it
        # should be, so we cannot judge it. Reported so it is never silently
        # skipped -- an unjudgeable slot is exactly where the last bug hid.
        return [f"{here} (unresolved forward ref {want!r})"] if isinstance(got, dict) else []
    if dataclasses.is_dataclass(want):
        if isinstance(got, dict):
            return [f"{here} came back as a raw dict, not {want.__name__}"]
        return _untyped_fields(got, here)
    return []


@pytest.mark.parametrize("name", finding_class_names())
def test_no_finding_field_is_dropped_by_the_round_trip(name):
    """Every field of every finding survives. Parametrized over the live class
    table rather than a hand-listed set, so a NEW analyzer is covered the day
    it registers — the previous bug was exactly a field nobody added by hand."""
    cls = _finding_class_for(name)
    assert cls is not None, f"{name} has no class in the round-trip table"
    lost = _lossy_fields(cls)
    assert lost == [], f"{name} loses {lost} on the way back from the store"


@pytest.mark.parametrize("cls", [WindowSummary, DowngradeFinding, BudgetProjection])
def test_no_top_level_dataclass_field_is_dropped(cls):
    """The report's own dataclasses drifted too — `WindowSummary.active_days`
    was dropped by the hand-written version, so a stored report came back
    claiming a different number of active days than the one that was measured."""
    lost = _lossy_fields(cls)
    assert lost == [], f"{cls.__name__} loses {lost} on the way back from the store"


def test_a_whole_report_round_trips_byte_identically():
    report = OptimizeReport(
        window=_populate(WindowSummary),
        downgrade=_populate(DowngradeFinding),
        budgets=[_populate(BudgetProjection)],
        notes=["a note"],
        findings={n: _populate(_finding_class_for(n)) for n in finding_class_names()},
        persona="sdk",
    )
    once = report_to_dict(report)
    twice = report_to_dict(report_from_dict(once))
    assert twice == once


def test_every_registered_analyzer_can_be_rebuilt():
    """A finding name absent from the class table is dropped SILENTLY on the
    way back (forward-compatibility for a newer daemon). That is correct for an
    unknown name and a data-loss bug for a name this install actually produces,
    so every registered analyzer must be rebuildable.

    `downsize` and `budget-projection` are excluded deliberately: they occupy
    typed top-level slots (`downgrade` / `budgets`) rather than the `findings`
    dict, and are covered by the top-level test above.
    """
    from tokenjam.core.optimize import ANALYZER_REGISTRY

    typed_slots = {"downsize", "budget-projection"}
    rebuildable = set(finding_class_names())
    missing = sorted(set(ANALYZER_REGISTRY) - typed_slots - rebuildable)
    assert missing == [], (
        f"analyzers {missing} produce findings the round-trip cannot rebuild; "
        f"add them to _build_finding_classes() in core/optimize/runner.py"
    )


def test_an_unknown_finding_name_is_dropped_rather_than_raising():
    """The forward-compatibility half of the same rule: a daemon newer than
    this install may advertise a finding it cannot render, and that must not
    break the whole report."""
    report = report_from_dict({
        "window": report_to_dict(_populate(WindowSummary)),
        "findings": {"a-finding-from-the-future": {"whatever": 1}},
        "persona": "sdk",
    })
    assert "a-finding-from-the-future" not in report.findings
    assert report.persona == "sdk"


# --- Field TYPE, not just field presence ----------------------------------- #
# The value-only guards above passed on every interpreter while the hydrator
# was silently returning raw dicts for nested dataclasses on Python 3.10, this
# repo's minimum supported version. Cause: `list["UncachedAgentCandidate"]` —
# a quoted forward reference inside a builtin generic subscript — resolves to
# the class on 3.11+ but stays the bare STRING on 3.10, so every is_dataclass
# check downstream failed and the value passed through unconverted.
#
# Value equality cannot see that, because `report_to_dict` serializes a
# dataclass and the dict it came from identically. These assert the type.

@pytest.mark.parametrize("name", finding_class_names())
def test_nested_dataclasses_come_back_as_dataclasses_not_dicts(name):
    cls = _finding_class_for(name)
    back = hydrate_dataclass(cls, report_to_dict(_populate(cls)))
    untyped = _untyped_fields(back)
    assert untyped == [], f"{name} lost the TYPE of: {untyped}"


@pytest.mark.parametrize("cls", [WindowSummary, DowngradeFinding, BudgetProjection])
def test_nested_dataclasses_on_the_top_level_types_keep_their_type(cls):
    back = hydrate_dataclass(cls, report_to_dict(_populate(cls)))
    untyped = _untyped_fields(back)
    assert untyped == [], f"{cls.__name__} lost the TYPE of: {untyped}"


def test_a_whole_report_keeps_every_nested_type():
    report = OptimizeReport(
        window=_populate(WindowSummary),
        downgrade=_populate(DowngradeFinding),
        budgets=[_populate(BudgetProjection)],
        notes=["a note"],
        findings={n: _populate(_finding_class_for(n)) for n in finding_class_names()},
        persona="sdk",
    )
    back = report_from_dict(report_to_dict(report))
    untyped: list[str] = []
    untyped += _untyped_fields(back.window, "window")
    untyped += _untyped_fields(back.downgrade, "downgrade")
    for i, b in enumerate(back.budgets):
        untyped += _untyped_fields(b, f"budgets[{i}]")
    for n, f in back.findings.items():
        assert dataclasses.is_dataclass(f), f"findings[{n!r}] is not a dataclass"
        untyped += _untyped_fields(f, f"findings[{n!r}]")
    assert untyped == [], f"the report lost the TYPE of: {untyped}"


def test_the_specific_shape_that_broke_python_310():
    """A regression pin on the exact construct, not just its symptom.

    `CacheEfficacyFinding.uncached_agents` is declared
    `list["UncachedAgentCandidate"]`. That quoted-forward-ref-inside-a-subscript
    is what `typing.get_type_hints` leaves unresolved on 3.10, and it is a shape
    any analyzer can reintroduce, so it gets its own named test rather than
    relying on someone reading a parametrized failure.
    """
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        UncachedAgentCandidate,
    )

    back = hydrate_dataclass(
        CacheEfficacyFinding, report_to_dict(_populate(CacheEfficacyFinding)),
    )
    assert back.uncached_agents, "the fixture must actually populate the list"
    assert isinstance(back.uncached_agents[0], UncachedAgentCandidate), (
        f"got {type(back.uncached_agents[0]).__name__}; a quoted forward "
        f"reference inside list[...] did not resolve on this interpreter"
    )
    # The attribute access that actually failed in CI.
    assert back.uncached_agents[0].agent_id == "sentinel-agent_id"


def test_a_loosely_typed_row_list_survives_with_its_type():
    """`list[Any]` costs the round trip its TYPE, and the metadata hook is what
    buys it back.

    `DowngradeFinding.per_agent` is declared `list[Any]` on purpose, so that
    `core.optimize.types` stays free of an analyzer import. The value survived
    the trip — the right rows, the right numbers — but came back as plain
    dicts, because `Any` gives the hydrator nothing to dispatch on. Every
    consumer reading `row.delta_usd` then raised `AttributeError`, and the one
    that mattered caught broadly: the cost adapter dropped the ENTIRE downsize
    contribution, so the Review inbox's headline lost that analyzer's whole
    figure while the Dashboard tile — reading the finding directly, never
    round-tripped — went on showing it. Two surfaces, one analyzer, a
    several-hundred-dollar disagreement, and no error anywhere.

    So the element type is declared as data (`metadata={"hydrate": ...}`) and
    resolved at hydration time. This pins that it keeps working; the sibling
    test above pins the quoted-forward-ref shape, which is the same failure
    reached by a different route.
    """
    from tokenjam.core.optimize.analyzers.downsize_agents import AgentPriceRow
    from tokenjam.core.optimize.analyzers.model_downgrade import DowngradeFinding

    finding = dataclasses.replace(
        _populate(DowngradeFinding), per_agent=[_populate(AgentPriceRow)],
    )
    back = hydrate_dataclass(DowngradeFinding, report_to_dict(finding))

    assert back.per_agent, "the fixture must actually populate the list"
    assert isinstance(back.per_agent[0], AgentPriceRow), (
        f"got {type(back.per_agent[0]).__name__}; a `list[Any]` row list lost "
        f"its type on the way back, which is what silently deleted a whole "
        f"analyzer from the Review inbox"
    )
    # The attribute access that actually failed, and the property that only
    # exists on the real class.
    assert back.per_agent[0].delta_usd == finding.per_agent[0].delta_usd
    assert back.per_agent[0].total_tokens == finding.per_agent[0].total_tokens


def test_every_loosely_typed_dataclass_list_declares_how_to_hydrate_it():
    """The rule, not the instance: a `list[Any]` field that is really a list of
    dataclasses must carry `hydrate` metadata, or it will lose its type exactly
    the way `per_agent` did — silently, and far from where it breaks.

    Fields genuinely holding heterogeneous values are unaffected: this only
    fires on a field whose stored rows are dataclass-shaped.
    """
    import tokenjam.core.optimize.types as types_mod

    offenders = []
    for name in dir(types_mod):
        cls = getattr(types_mod, name)
        if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
            continue
        for f in dataclasses.fields(cls):
            annotation = str(f.type)
            if "list[Any]" not in annotation:
                continue
            if (f.metadata or {}).get("hydrate"):
                continue
            offenders.append(f"{name}.{f.name}")
    assert not offenders, (
        "these `list[Any]` fields declare no `hydrate` metadata; if any of them "
        "holds dataclass rows it comes back as raw dicts and the failure "
        f"surfaces somewhere else entirely: {offenders}"
    )
