"""One metric, one window, stated where it is read.

THE DEFECT. ``past_overspend_usd`` was published by two surfaces under two
independently-derived windows, and by one analyzer on two different bases.

  * The Dashboard's recoverable-waste tiles came off the stored analyzer report,
    scoped to ``[optimize] scan_window_days``. The Review inbox headline came
    off the cost-proposal store, scoped to the resolved analysis span
    (``analysis_span.window_days_for``). On a real corpus those were 30 and 69,
    so summing the six tiles gave roughly 3510 dollars against an inbox headline
    of 4927.94 — the same metric, minutes apart, incomparable.
  * relearn's ``past_overspend_usd`` is UNBOUNDED by design (its signal is
    recurrence across all retained history, so ``run(ctx)`` deliberately does
    not forward the report window). The tile row rendered it as a peer beside
    five window-scoped tiles: 386.64 on the Dashboard against 260.21 published
    by the inbox for the same analyzer over the same 30 days, with the
    correctly-bounded figure already sitting unused on the same payload.

WHAT IS PINNED HERE. The window is now ONE seam
(``core/optimize/report_window``) both recompute paths resolve through, and
relearn publishes one figure per window from one code path
(``inbox_contribution``'s netting rule, shared with the inbox row). The
direction invariant the ticket asked for — a sub-window figure can never exceed
the unbounded one for the same analyzer — is asserted at the FINDING level here;
``test_relearn_window`` already pins it per cluster.
"""
from __future__ import annotations

import pathlib
import threading
from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.api.routes.optimize import (
    WINDOW_SCOPED_BASIS,
    WINDOW_SCOPED_TOKENS,
    WINDOW_SCOPED_USD,
    WINDOW_SCOPED_WINDOW,
    _with_window_scoped_relearn,
)
from tokenjam.core.optimize.analyzers.relearn import (
    FailureEpisode,
    analyze_relearns,
)
from tokenjam.core.optimize.cost_proposals import cost_window_days_for
from tokenjam.core.optimize.inbox_contribution import (
    relearn_contribution,
    window_scoped_finding_figure,
)
from tokenjam.core.optimize.rate_profile import RateProfile
from tokenjam.core.optimize.relearn_window import RELEARN_WINDOW_LABELS
from tokenjam.core.optimize.report_store import _window_days as report_store_window_days
from tokenjam.core.optimize.report_window import (
    FALLBACK_WINDOW_DAYS,
    configured_scan_window_days,
    report_window_days,
    report_window_label,
)

ANCHOR = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Config / connection doubles
# --------------------------------------------------------------------------- #
class _Optimize:
    def __init__(self, scan_window_days):
        self.scan_window_days = scan_window_days


class _Storage:
    def __init__(self, analysis_span=None, retention_days=None):
        self.analysis_span = analysis_span
        self.retention_days = retention_days


class _Config:
    def __init__(self, *, scan_window_days=30, analysis_span="90d"):
        self.optimize = _Optimize(scan_window_days)
        self.storage = _Storage(analysis_span=analysis_span)


@pytest.fixture
def held_days(monkeypatch):
    """Pin how much history the store reports, without needing one.

    ``days_of_history`` is the only thing here that touches a connection, and
    it is imported INTO ``report_window``, so the patch has to land on that
    module's binding rather than on ``analysis_span``'s.
    """
    def _set(days):
        monkeypatch.setattr(
            "tokenjam.core.optimize.report_window.days_of_history",
            lambda _conn: days,
        )
    return _set


# --------------------------------------------------------------------------- #
# The seam: both recompute paths resolve ONE window
# --------------------------------------------------------------------------- #
def test_the_two_surfaces_resolve_the_same_window_from_one_seam(held_days):
    # The reported defect in one assertion. Config: a 90-day analysis span over
    # a store holding 69 days, with the analyzer scan knob at its default 30.
    # The cost side used to answer 69 (span bounded by history) while the report
    # side answered 30 (the knob), and each published `past_overspend_usd` under
    # its own number without saying which.
    held_days(69)
    config = _Config(scan_window_days=30, analysis_span="90d")

    assert cost_window_days_for(config, None) == 30
    assert report_store_window_days(config, None) == 30
    assert cost_window_days_for(config, None) == report_store_window_days(config, None)


def test_scan_window_days_is_the_knob_both_sides_follow(held_days):
    # Raising the look-back moves BOTH surfaces, which is the property that
    # makes them one window rather than two that currently agree.
    held_days(365)
    config = _Config(scan_window_days=45, analysis_span="90d")
    assert report_window_days(config, None) == 45
    assert cost_window_days_for(config, None) == 45
    assert report_store_window_days(config, None) == 45


def test_the_chosen_span_bounds_the_window_but_no_longer_widens_it(held_days):
    held_days(365)
    # A span SHORTER than the knob bounds it: retention is derived from the
    # span, so a figure may not claim a window whose data may already be gone.
    assert report_window_days(_Config(scan_window_days=90, analysis_span="30d"), None) == 30
    # A span LONGER than the knob does not widen it. This is the 69 that used
    # to reach the inbox headline: the span's job is to bound what may be
    # claimed, not to decide what is.
    assert report_window_days(_Config(scan_window_days=30, analysis_span="90d"), None) == 30


def test_the_history_actually_held_bounds_the_window(held_days):
    # A 30-day claim over a store that has been running a week is answerable
    # for a week.
    held_days(7)
    assert report_window_days(_Config(scan_window_days=30, analysis_span="90d"), None) == 7


def test_an_unmeasurable_history_narrows_nothing(held_days):
    # None means UNKNOWN, not zero: an unknown span cannot narrow a window, and
    # a zero-day window is not a smaller claim but an empty one.
    held_days(None)
    assert report_window_days(_Config(scan_window_days=30, analysis_span="90d"), None) == 30


@pytest.mark.parametrize("bad", [0, -5, None, "banana"])
def test_an_unusable_scan_window_falls_back_rather_than_covering_nothing(bad):
    # A zero-day window would make every window-scoped figure cover nothing and
    # read as "no waste" — the worst possible degradation for this metric.
    assert configured_scan_window_days(_Config(scan_window_days=bad)) == FALLBACK_WINDOW_DAYS


def test_a_malformed_analysis_span_does_not_sink_the_scan(held_days):
    held_days(None)
    config = _Config(scan_window_days=30, analysis_span="not-a-span")
    assert report_window_days(config, None) == 30


def test_the_label_relearn_precomputes_is_the_same_seam(held_days):
    # relearn precomputes its bounded buckets against a vocabulary, and the
    # inbox matches its headline window against that vocabulary EXACTLY (no
    # nearest-match). A label derived any other way here drops every cluster
    # out of the headline into the excluded channel.
    held_days(69)
    config = _Config(scan_window_days=30, analysis_span="90d")
    assert report_window_label(config, None) == "30d"
    assert report_window_label(config, None) == f"{report_window_days(config, None)}d"


# --------------------------------------------------------------------------- #
# The direction invariant, at the finding level
# --------------------------------------------------------------------------- #
def _iso(days_ago: float) -> str:
    return (ANCHOR - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def _failure(session: str, days_ago: float) -> FailureEpisode:
    return FailureEpisode(
        session_id=session, repo="repo-a", ts=_iso(days_ago),
        tool_name="Bash", label="cd x",
        error_text="(eval):cd:1: no such file or directory: x",
        kind="act", is_retry=False, depth=0, detour_turns=2.0,
    )


#: A flat, known input rate so the analyzer's own pricing runs without a
#: database. Only the RATE is injected — every token count, every window filter
#: and every re-read subtraction below is the real analyzer's arithmetic.
_TEST_PROFILE = RateProfile(input_rate_per_token=3e-6, cache_read_ratio=0.1, basis="test")


@pytest.fixture
def priced(monkeypatch):
    monkeypatch.setattr(
        "tokenjam.core.optimize.analyzers.relearn.blended_rate_profile",
        lambda *_a, **_k: _TEST_PROFILE,
    )


def _finding():
    """A relearn finding over occurrences spread across 60 days."""
    failures = [
        _failure(f"s{i}", days_ago)
        for i, days_ago in enumerate([0.1, 0.5, 2, 5, 9, 20, 35, 50, 60])
    ]
    return analyze_relearns(
        [], min_sessions=3, distill_enabled=False,
        extra_failures=failures,
        window_labels=RELEARN_WINDOW_LABELS, window_anchor=ANCHOR,
    )


def test_no_sub_window_total_exceeds_the_unbounded_one(priced):
    # The invariant the ticket asked for, and the one that would have caught a
    # basis string lying about what it summed. A bounded figure is a FILTER over
    # the same occurrences at the same price, so it can only shrink — a window
    # total larger than the all-history total means two code paths computed the
    # same analyzer's past overspend differently.
    finding = _finding()
    assert finding.past_overspend_windows, "no windowed totals were computed"
    assert finding.past_overspend_usd is not None
    for label, total in finding.past_overspend_windows.items():
        assert total.past_overspend_tokens <= finding.past_overspend_tokens, label
        if total.past_overspend_usd is not None:
            assert total.past_overspend_usd <= finding.past_overspend_usd + 1e-9, label


def test_a_longer_window_is_never_smaller_than_a_shorter_one():
    # Monotonicity across the vocabulary, which is the same statement read
    # sideways: nesting windows nest their sums.
    finding = _finding()
    totals = finding.past_overspend_windows
    assert totals is not None
    ordered = [totals[label] for label in RELEARN_WINDOW_LABELS if label in totals]
    for shorter, longer in zip(ordered, ordered[1:]):
        assert shorter.past_overspend_tokens <= longer.past_overspend_tokens


# --------------------------------------------------------------------------- #
# One relearn figure per window, from one code path
# --------------------------------------------------------------------------- #
def _finding_dict():
    from dataclasses import asdict

    return asdict(_finding())


def test_the_tile_figure_is_the_bounded_bucket_net_of_re_read(priced):
    # The Dashboard tile and the Review inbox row are now the same arithmetic on
    # the same bucket: the window's observed cost minus its measured re-read
    # share (which the context re-send proposal already prices in full, so
    # counting it in both places bills the same tokens twice).
    finding = _finding_dict()
    figure = window_scoped_finding_figure(finding, days=30)
    assert figure is not None
    bucket = finding["past_overspend_windows"]["30d"]
    assert figure["usd"] == pytest.approx(
        max(bucket["past_overspend_usd"] - bucket["past_reread_usd"], 0.0), abs=1e-6,
    )
    assert figure["window"] == "30d"


def test_the_tile_figure_is_not_the_unbounded_one(priced):
    # The defect, stated directly: the tile may not publish all-history money in
    # a row every other member of which is window-scoped.
    finding = _finding_dict()
    figure = window_scoped_finding_figure(finding, days=30)
    assert figure is not None
    assert figure["usd"] < finding["past_overspend_usd"]


def test_the_row_and_the_tile_net_by_the_same_rule():
    # One code path, asserted rather than trusted: a cluster's contribution to
    # the inbox headline and the finding-level tile figure both come from
    # `_net_of_reread`, so a change to one cannot silently spare the other.
    cluster = {
        "signature": "cwd_confusion",
        "past_overspend_windows": {
            "30d": {
                "past_overspend_usd": 10.0, "past_reread_usd": 4.0,
                "past_overspend_tokens": 1000, "past_reread_tokens": 400,
                "window_days": 30.0,
            },
        },
    }
    finding = {"past_overspend_windows": dict(cluster["past_overspend_windows"])}
    row = relearn_contribution(cluster, label="30d")
    tile = window_scoped_finding_figure(finding, days=30)
    assert row is not None and tile is not None
    assert row["usd"] == tile["usd"] == pytest.approx(6.0)
    assert row["tokens"] == tile["tokens"] == 600


def test_no_exact_bucket_is_unknown_rather_than_the_nearest_one(priced):
    # There is no nearest-match fallback: the report names ONE window for every
    # finding on it, so a bucket computed for another span cannot enter it.
    finding = _finding_dict()
    assert window_scoped_finding_figure(finding, days=42) is None
    assert window_scoped_finding_figure(finding, days=None) is None
    assert window_scoped_finding_figure(None, days=30) is None


def test_an_unpriced_bucket_stays_unknown_rather_than_netting_to_zero():
    # Netting an UNKNOWN re-read share as zero publishes a figure that
    # double-counts against resend, so unknown propagates.
    finding = {
        "past_overspend_windows": {
            "30d": {
                "past_overspend_usd": 10.0, "past_reread_usd": None,
                "past_overspend_tokens": 1000, "past_reread_tokens": 0,
                "window_days": 30.0,
            },
        },
    }
    assert window_scoped_finding_figure(finding, days=30) is None


# --------------------------------------------------------------------------- #
# What the /optimize payload publishes
# --------------------------------------------------------------------------- #
def test_the_payload_stamps_relearn_with_a_figure_on_the_reports_own_window(priced):
    findings = {"relearn": _finding_dict(), "resend": {"past_overspend_usd": 1.0}}
    out = _with_window_scoped_relearn(findings, 30)
    relearn = out["relearn"]
    assert relearn[WINDOW_SCOPED_WINDOW] == "30d"
    assert relearn[WINDOW_SCOPED_USD] < relearn["past_overspend_usd"]
    assert relearn[WINDOW_SCOPED_TOKENS] < relearn["past_overspend_tokens"]
    assert relearn[WINDOW_SCOPED_BASIS]
    # Other findings are already window-scoped and are passed through untouched.
    assert out["resend"] == {"past_overspend_usd": 1.0}


def test_the_stamp_is_present_and_null_when_no_bucket_matches(priced):
    # PRESENT, not absent: a renderer keys off presence to know the unbounded
    # figure is off-limits for this tile. Absent would let it fall back, which
    # is the defect. Null is UNKNOWN, and the basis string says so.
    findings = {"relearn": _finding_dict()}
    relearn = _with_window_scoped_relearn(findings, 42)["relearn"]
    assert WINDOW_SCOPED_USD in relearn
    assert relearn[WINDOW_SCOPED_USD] is None
    assert relearn[WINDOW_SCOPED_TOKENS] is None
    assert "unknown, not zero" in relearn[WINDOW_SCOPED_BASIS]


def test_the_stamp_never_writes_into_the_stored_body(priced):
    # The unbounded fields feed the write budget's pre-net gross; shrinking them
    # in place would silently flip clusters between "worth a permanent rule" and
    # net-negative. The derivation is read-side only.
    stored = _finding_dict()
    gross = stored["past_overspend_usd"]
    findings = {"relearn": stored}
    _with_window_scoped_relearn(findings, 30)
    assert stored["past_overspend_usd"] == gross
    assert WINDOW_SCOPED_USD not in stored
    assert findings["relearn"] is stored


def test_a_payload_without_relearn_is_returned_unchanged():
    findings = {"resend": {"past_overspend_usd": 1.0}}
    assert _with_window_scoped_relearn(findings, 30) == findings
    assert _with_window_scoped_relearn(None, 30) is None


# --------------------------------------------------------------------------- #
# One scan cycle, one anchor
# --------------------------------------------------------------------------- #
def test_a_cycle_refreshes_every_analyzer_store(monkeypatch):
    # "Rescan" used to mean a different thing per screen: `/optimize/rescan`
    # refreshed the report, the Review inbox's own Refresh refreshed the other
    # two, and nothing refreshed all three. So the stores feeding two surfaces
    # of one metric could age apart with nothing disclosing it.
    from tokenjam.core.optimize import scan_cycle

    calls = {}
    monkeypatch.setattr(
        scan_cycle, "_trigger_analyzer_pass",
        lambda _f, _c, anchor: calls.setdefault("analyzer_pass", anchor) is None or True,
    )

    started = scan_cycle.trigger_scan_cycle(lambda: None, _Config())
    assert set(started) == {"analyzer_pass"}
    assert all(started.values())
    assert set(calls) == {"analyzer_pass"}


def test_the_report_and_cost_stores_come_from_ONE_analyzer_pass(monkeypatch):
    # THE defect behind the residual delta. A cycle used to run `build_report`
    # TWICE — once for the report store, once inside the cost recompute — so an
    # analyzer was measured twice against a database ingestion keeps writing to,
    # and the two results were published side by side. No window or anchor
    # agreement can reconcile two separate measurements; only one measurement can.
    from tokenjam.core.optimize import cost_proposals, report_store, scan_cycle

    seen = {}
    sentinel = object()

    monkeypatch.setattr(
        report_store, "recompute_now",
        lambda _db, _cfg, **kw: seen.setdefault("anchor_report", kw.get("until")) or {"ok": 1},
    )
    monkeypatch.setattr(report_store, "is_computing", lambda: False)
    monkeypatch.setattr(report_store, "stored_report", lambda _cfg: sentinel)

    def _cost(_db, _cfg, **kw):
        seen["report_arg"] = kw.get("report")
        seen["anchor_cost"] = kw.get("until")
        return []

    done = threading.Event()

    def _cost_then_signal(*a, **kw):
        try:
            return _cost(*a, **kw)
        finally:
            done.set()

    monkeypatch.setattr(cost_proposals, "recompute_cost_proposals", _cost_then_signal)

    scan_cycle.trigger_scan_cycle(lambda: None, _Config())
    assert done.wait(timeout=5), "the cycle never reached the cost half"

    # The cost side was handed the report the pass just built — it did not
    # build its own.
    assert seen["report_arg"] is sentinel
    # And both halves used the same anchor.
    assert seen["anchor_report"] is not None
    assert seen["anchor_report"] == seen["anchor_cost"]


def test_a_pass_that_cannot_start_is_reported_not_raised(monkeypatch):
    from tokenjam.core.optimize import scan_cycle

    def _boom(*_a, **_k):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(scan_cycle, "_trigger_analyzer_pass", _boom)

    started = scan_cycle.trigger_scan_cycle(lambda: None, _Config())
    assert started["analyzer_pass"] is False


def test_the_kill_switch_gates_every_store_not_only_the_report():
    # `scan_enabled` is documented as "keeps the daemon from ever scanning on
    # its own". It used to gate only the report job while relearn and the cost
    # proposals kept scanning on their own 6h schedules.
    from tokenjam.core.optimize import scan_cycle

    config = _Config()
    config.optimize.scan_enabled = False
    assert scan_cycle.scan_enabled(config) is False
    config.optimize.scan_enabled = True
    assert scan_cycle.scan_enabled(config) is True
    # Absent (an older config object) means enabled, not disabled.
    assert scan_cycle.scan_enabled(object()) is True


def test_the_cost_store_records_the_bounds_it_ran_over(tmp_path):
    # A day count alone is not provenance: while this store recorded only a
    # length, its window could not be compared against the report's own
    # scan_since/scan_until, so a per-analyzer disagreement between the two
    # surfaces was undiagnosable from the artifacts.
    from tokenjam.core.optimize import relearn_store

    path = tmp_path / "relearn_cache.json"
    relearn_store.write_cost_proposals(
        [], path,
        window_days=30,
        since="2026-06-29T09:40:21+00:00",
        until="2026-07-29T09:40:21+00:00",
    )
    stored = relearn_store.read_cost_proposals(path)
    assert stored is not None
    assert stored["cost_window_days"] == 30
    assert stored["cost_since"] == "2026-06-29T09:40:21+00:00"
    assert stored["cost_until"] == "2026-07-29T09:40:21+00:00"


def test_a_cache_predating_the_bounds_reports_them_absent_never_derived(tmp_path):
    # Absent, never guessed from the day count: deriving them would invent the
    # very provenance the field exists to supply.
    from tokenjam.core.optimize import relearn_store

    path = tmp_path / "relearn_cache.json"
    relearn_store.write_cost_proposals([], path, window_days=30)
    stored = relearn_store.read_cost_proposals(path)
    assert stored is not None
    assert stored["cost_window_days"] == 30
    assert stored["cost_since"] is None
    assert stored["cost_until"] is None


def test_the_relearn_cache_is_written_from_the_passs_own_finding(monkeypatch):
    # `compute_relearn_finding` ran TWICE per cycle: once inside build_report as
    # the registered analyzer, once more for this cache — the most expensive
    # analyzer in the product, duplicated. And not even the same computation:
    # the two calls passed different min_sessions, different window labels and
    # the write budget in only one, so the two surfaces could disagree about
    # WHICH CLUSTERS to offer.
    from tokenjam.core.optimize import relearn_store, scan_cycle

    finding = object()
    written = {}
    monkeypatch.setattr(
        relearn_store, "write_cache",
        lambda f, *a, **kw: written.setdefault("finding", f),
    )
    recomputed = []
    monkeypatch.setattr(
        relearn_store, "trigger_background_recompute",
        lambda *a, **kw: recomputed.append(1) or True,
    )

    class _Report:
        findings = {"relearn": finding}

    scan_cycle._write_relearn_from(_Report(), _Config(), lambda: None)
    assert written["finding"] is finding
    # And nothing recomputed it.
    assert recomputed == []


def test_a_pass_with_no_relearn_finding_falls_back_rather_than_leaving_it_stale(
    monkeypatch,
):
    from tokenjam.core.optimize import relearn_store, scan_cycle

    recomputed = []
    monkeypatch.setattr(
        relearn_store, "trigger_background_recompute",
        lambda *a, **kw: recomputed.append(1) or True,
    )
    monkeypatch.setattr(
        relearn_store, "write_cache", lambda *a, **kw: pytest.fail("must not write"),
    )

    scan_cycle._write_relearn_from(None, _Config(), lambda: None)
    assert recomputed == [1]


def _mix_call_is_windowed(call: "ast.Call") -> bool:
    """Was this ``agent_persona_mix(...)`` call given an explicit window?

    Windowed means since/until were supplied as something other than ``None`` —
    positionally (``mix(conn, since, until)``) or by keyword. Passing literal
    ``None`` is how a caller says "all history on purpose", so it counts as
    unwindowed and must not reach a classifier.
    """
    import ast as _ast

    supplied = list(call.args[1:3])
    supplied += [kw.value for kw in call.keywords if kw.arg in ("since", "until")]
    if len(supplied) < 2:
        return False
    return all(not (isinstance(a, _ast.Constant) and a.value is None) for a in supplied)


def test_no_surface_classifies_persona_over_all_history():
    """A persona GATE may never be resolved over all history.

    Persona gates which analyzers run (``PERSONA_DISABLED_ANALYZERS``) and
    whether relearn may offer a workspace write. ``agent_persona_mix``'s
    since/until are optional, and callers omitted them — so a corpus whose
    recent window is claude-code dominant but whose full history is mixed
    resolved a different gate on the Dashboard than in the Review inbox. That is
    two surfaces disagreeing about which findings EXIST.

    The property is about CLASSIFICATION, not about the query: an unwindowed
    ``agent_persona_mix`` is legitimate for a question that is genuinely about
    the whole corpus (``/persona``'s ``counts`` answers "has this machine ever
    ingested anything for this persona", which the UI renders as "Nothing has
    been ingested for this persona" — windowing that would make the claim
    false). So this walks the AST and fails only when an unwindowed mix reaches
    ``dominant_persona``, directly or through a local variable. A substring grep
    for ``agent_persona_mix(conn)`` cannot draw that line.
    """
    import ast

    import tokenjam as _pkg

    root = pathlib.Path(_pkg.__file__).parent
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Names bound to an UNWINDOWED mix, anywhere in the module.
        unwindowed_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "agent_persona_mix"
            and not _mix_call_is_windowed(node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != "dominant_persona":
                continue
            for arg in node.args:
                bad = (
                    isinstance(arg, ast.Call)
                    and getattr(arg.func, "id", None) == "agent_persona_mix"
                    and not _mix_call_is_windowed(arg)
                ) or (isinstance(arg, ast.Name) and arg.id in unwindowed_names)
                if bad:
                    offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, (
        "persona classified over all history at:\n  " + "\n  ".join(offenders)
    )


def test_the_apply_target_and_the_write_guard_share_one_derivation():
    # `relearn_store` resolved the suggested write target through
    # `resolve_write_scope(...).suggest_root` and carried a comment recording why:
    # the API's write guard authorizes against the other half of that same type,
    # so deriving them independently let the suggestion and the guard disagree.
    # The registered analyzer passed `scope.claude_home` directly — the exact
    # path that comment warns against.
    from tokenjam.core.optimize.analyzers import relearn as _relearn

    src = pathlib.Path(_relearn.__file__).read_text(encoding="utf-8")
    assert "claude_home=resolve_write_scope(scope=scope).suggest_root" in src
    assert "claude_home=scope.claude_home" not in src
