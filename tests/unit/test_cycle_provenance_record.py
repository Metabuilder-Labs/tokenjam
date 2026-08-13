"""ONE provenance record per cycle, carried by every artifact the cycle writes.

The three defects this pins, all of which shipped together because they are one
fact ("what produced this figure") split into three ad-hoc conventions:

1. **A build stamped but never compared.** ``tj_build()`` was called
   independently at every write site and again at every read site, and NOTHING
   anywhere asked whether a stored stamp matched the running one. The stores are
   caches with no invalidation on upgrade, so after upgrading tokenjam every card
   on screen was produced by the previous build and kept being served as a result.
2. **Two spellings of one window.** ``scan_since``/``scan_until`` on the report,
   ``cost_since``/``cost_until`` on the cost block — raw dict keys, no shared
   type, nothing forcing them to describe the same span.
3. **A cycle with no identity.** One pass writes three stores sequentially on one
   thread, so a report-derived panel can serve cycle N figures beside inbox
   figures from cycle N-1, and no consumer could tell.

Every test below fails without the record: the first two by KeyError on a key
that does not exist, the staleness ones because nothing performed the comparison
at all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import cycle_provenance, relearn_store, report_store, scan_cycle


@pytest.fixture(autouse=True)
def _no_leaked_cycle_state():
    """Same backstop `test_scan_cycle_provenance` carries: both the in-flight
    flag and the ingestion watermark are module-global, and leaking either makes
    a LATER file fail for reasons invisible in that file."""
    scan_cycle._CYCLE_COMPUTING.clear()
    scan_cycle._last_pass_watermark = None
    scan_cycle._last_pass_at = None
    yield
    scan_cycle._CYCLE_COMPUTING.clear()
    scan_cycle._last_pass_watermark = None
    scan_cycle._last_pass_at = None


@pytest.fixture
def cfg(tmp_path) -> TjConfig:
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _thread_inline(monkeypatch):
    """Run the cycle's job body on the calling thread.

    The pass is fire-and-forget by design; a test that has to observe what it
    wrote cannot race it. Returns the list the dispatched targets land in.
    """
    targets: list = []

    def _fake_thread(target=None, name=None, daemon=None):
        class _T:
            def start(_self):
                targets.append(target)
        return _T()

    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread)
    return targets


@dataclass
class _Finding:
    """The shape `relearn_store.write_cache` needs: a dataclass `asdict` accepts.
    A real `RelearnFinding` would drag the whole detector in for no coverage."""

    clusters: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 1. ONE record per cycle, identical on every artifact that cycle writes
# --------------------------------------------------------------------------- #
def test_one_cycle_writes_one_identical_record_to_every_store(monkeypatch, cfg):
    """THE DEFECT. Each store resolved its own build, its own window keys and no
    identity at all, so "these three artifacts came from one pass" was not a fact
    any consumer could check — it was a timing coincidence.
    """
    records: dict[str, dict] = {}

    def _recompute_now(backend, config, provenance=None, **kw):
        records["report"] = provenance.to_dict()
        # What the store really writes, so the cycle's read-back of the sealed
        # record exercises the same path production does.
        return {"computed_at": "now", "provenance": provenance.to_dict()}

    monkeypatch.setattr(report_store, "is_computing", lambda: False)
    monkeypatch.setattr(report_store, "recompute_now", _recompute_now)
    monkeypatch.setattr(report_store, "stored_report", lambda config: _StubReport())
    monkeypatch.setattr(
        relearn_store, "write_cache",
        lambda finding, path=None, *, config=None, provenance=None:
            records.__setitem__("relearn", provenance.to_dict()),
    )

    import tokenjam.core.optimize.cost_proposals as cp

    monkeypatch.setattr(
        cp, "recompute_cost_proposals",
        lambda backend, config, report=None, provenance=None, **kw:
            records.__setitem__("cost", provenance.to_dict()),
    )
    monkeypatch.setattr(scan_cycle, "_refresh_rule_presence", lambda config: None)

    targets = _thread_inline(monkeypatch)
    assert scan_cycle._trigger_analyzer_pass(lambda: None, cfg, None) is True
    targets[0]()

    assert set(records) == {"report", "relearn", "cost"}
    assert records["report"] == records["relearn"] == records["cost"], (
        "the three stores of one pass carry different provenance"
    )
    assert records["report"]["cycle_id"]


class _StubReport:
    """A report carrying only what the cycle reads off it."""

    persona = "claude-code"
    findings = {"relearn": _Finding()}


def test_the_record_is_sealed_with_the_reports_own_persona(monkeypatch, cfg):
    """The persona is NOT classified a second time for the record. `build_report`
    resolves the window's dominant persona once (it is the analyzer skip gate's
    choke point) and the cycle takes THAT value onto the record it hands to the
    later legs — otherwise the label on the artifact and the gate that decided
    which findings exist could disagree.
    """
    seen: dict[str, str] = {}

    def _recompute_now(backend, config, provenance=None, **kw):
        sealed = provenance.with_persona("claude-code")
        return {"computed_at": "now", "provenance": sealed.to_dict()}

    monkeypatch.setattr(report_store, "is_computing", lambda: False)
    monkeypatch.setattr(report_store, "recompute_now", _recompute_now)
    monkeypatch.setattr(report_store, "stored_report", lambda config: _StubReport())
    monkeypatch.setattr(
        relearn_store, "write_cache",
        lambda finding, path=None, *, config=None, provenance=None:
            seen.__setitem__("relearn_persona", provenance.persona),
    )

    import tokenjam.core.optimize.cost_proposals as cp

    monkeypatch.setattr(
        cp, "recompute_cost_proposals",
        lambda backend, config, report=None, provenance=None, **kw:
            seen.__setitem__("cost_persona", provenance.persona),
    )
    monkeypatch.setattr(scan_cycle, "_refresh_rule_presence", lambda config: None)

    targets = _thread_inline(monkeypatch)
    scan_cycle._trigger_analyzer_pass(lambda: None, cfg, None)
    targets[0]()

    assert seen["relearn_persona"] == "claude-code"
    assert seen["cost_persona"] == "claude-code"


# --------------------------------------------------------------------------- #
# 2. A different cycle is DISTINGUISHABLE from the one before it
# --------------------------------------------------------------------------- #
def test_two_cycles_are_distinguishable_by_cycle_id(cfg):
    """Without an id, "is the tile beside these rows from the same pass?" has no
    answer at all — a report-derived panel serving cycle N figures beside inbox
    figures from cycle N-1 is indistinguishable from one where both are current.
    """
    first = cycle_provenance.begin_cycle(cfg)
    second = cycle_provenance.begin_cycle(cfg)
    assert first.cycle_id != second.cycle_id
    assert first.cycle_id and second.cycle_id


def test_two_payloads_from_one_cycle_publish_the_same_id(cfg):
    """The consumer-facing half: a surface compares two payloads' `cycle_id`
    without knowing anything about which store fed which."""
    record = cycle_provenance.begin_cycle(cfg)
    report_store.write_report({"findings": {}}, config=cfg, provenance=record)
    relearn_store.write_cost_proposals([], config=cfg, provenance=record)

    report_block = report_store.stored_report_block(cfg)
    cost_block = cycle_provenance.provenance_block(
        relearn_store.read_cost_proposals(config=cfg), prefix="cost_",
    )
    assert report_block["cycle_id"] == cost_block["cycle_id"] == record.cycle_id


def test_a_later_cycle_replaces_the_id_on_the_store_it_rewrites(cfg):
    """The state the defect actually produces: one store refreshed by a newer
    pass while its sibling still holds the previous one. Both ids are present and
    they differ, which is exactly what makes the split visible."""
    first = cycle_provenance.begin_cycle(cfg)
    report_store.write_report({"findings": {}}, config=cfg, provenance=first)
    relearn_store.write_cost_proposals([], config=cfg, provenance=first)

    second = cycle_provenance.begin_cycle(cfg)
    report_store.write_report({"findings": {}}, config=cfg, provenance=second)

    report_block = report_store.stored_report_block(cfg)
    cost_block = cycle_provenance.provenance_block(
        relearn_store.read_cost_proposals(config=cfg), prefix="cost_",
    )
    assert report_block["cycle_id"] == second.cycle_id
    assert cost_block["cycle_id"] == first.cycle_id
    assert report_block["cycle_id"] != cost_block["cycle_id"]


# --------------------------------------------------------------------------- #
# 3. A stored artifact from another BUILD is detectable as stale ON READ
# --------------------------------------------------------------------------- #
def test_an_artifact_from_another_build_reads_as_stale_not_as_a_result(cfg):
    """THE DEFECT, and the one nothing checked: the build was written but never
    COMPARED. There was no `computed_build != build` anywhere in the product, so
    after an upgrade every card was the previous build's and was served as an
    ordinary result under a fresh-looking timestamp.
    """
    record = cycle_provenance.begin_cycle(cfg)
    report_store.write_report({"findings": {}}, config=cfg, provenance=record)
    p = report_store.default_report_path(cfg)
    stored = json.loads(p.read_text(encoding="utf-8"))
    stored["provenance"]["build"] = "0.0.1-previous"
    stored["tj_version"] = "0.0.1-previous"
    p.write_text(json.dumps(stored), encoding="utf-8")

    block = report_store.stored_report_block(cfg)
    assert block["build_provenance"] == cycle_provenance.BUILD_STALE
    assert block["computed_build"] == "0.0.1-previous"
    assert block["build"] == cycle_provenance.tj_build()
    # DISCLOSED, NEVER DISCARDED. Dropping the figures would turn a populated
    # surface into an empty one across an upgrade, and an empty state is the
    # strongest claim a surface can make. The result still travels.
    assert block["status"] == report_store.STATUS_READY
    assert report_store.stored_report_dict(cfg) == {"findings": {}}


def test_an_artifact_from_this_build_reads_as_a_match(cfg):
    record = cycle_provenance.begin_cycle(cfg)
    report_store.write_report({"findings": {}}, config=cfg, provenance=record)
    block = report_store.stored_report_block(cfg)
    assert block["build_provenance"] == cycle_provenance.BUILD_MATCH


def test_an_unknown_producing_build_is_not_reported_as_agreement(cfg):
    """The trap the tri-state exists for. "We cannot tell" and "they match" are
    different answers, and a boolean `stale: false` would collapse them."""
    assert cycle_provenance.build_provenance(None) == cycle_provenance.BUILD_UNKNOWN
    assert cycle_provenance.build_provenance("") == cycle_provenance.BUILD_UNKNOWN
    assert (
        cycle_provenance.build_provenance(cycle_provenance.UNKNOWN)
        == cycle_provenance.BUILD_UNKNOWN
    )
    assert cycle_provenance.build_provenance("9.9.9", "9.9.9") == cycle_provenance.BUILD_MATCH
    assert cycle_provenance.build_provenance("9.9.8", "9.9.9") == cycle_provenance.BUILD_STALE


def test_every_feed_resolves_staleness_through_the_same_function(cfg):
    """One `ScanBar` reads three surfaces. The verdict is resolved SERVER-side,
    once, so the CLI, `--json` and the MCP server cannot disagree with the page —
    and so a second copy of the comparison cannot drift from the first."""
    record = cycle_provenance.begin_cycle(cfg)
    relearn_store.write_cost_proposals([], config=cfg, provenance=record)
    relearn_store.write_cache(_Finding(), config=cfg, provenance=record)

    cost_block = cycle_provenance.provenance_block(
        relearn_store.read_cost_proposals(config=cfg), prefix="cost_",
    )
    relearn_block = cycle_provenance.provenance_block(relearn_store.read_cache(config=cfg))
    report_store.write_report({"findings": {}}, config=cfg, provenance=record)
    report_block = report_store.stored_report_block(cfg)

    keys = {"cycle_id", "cycle_computing", "computed_build", "build",
            "build_provenance", "scan_since", "scan_until", "provenance"}
    for name, block in (
        ("report", report_block), ("cost", cost_block), ("relearn", relearn_block),
    ):
        assert keys <= set(block), f"{name} is missing {keys - set(block)}"
        assert block["build_provenance"] == cycle_provenance.BUILD_MATCH


# --------------------------------------------------------------------------- #
# 4. The two window conventions resolve through ONE type
# --------------------------------------------------------------------------- #
def test_the_report_and_cost_windows_come_off_one_record(cfg):
    """`scan_since`/`scan_until` and `cost_since`/`cost_until` were two raw-dict
    spellings of one fact, and nothing forced them to describe the same span. Now
    both are projections of the same `CycleProvenance`.
    """
    record = cycle_provenance.begin_cycle(cfg)
    report_store.write_report({"findings": {}}, config=cfg, provenance=record)
    relearn_store.write_cost_proposals([], config=cfg, provenance=record)

    report_block = report_store.stored_report_block(cfg)
    cost_block = cycle_provenance.provenance_block(
        relearn_store.read_cost_proposals(config=cfg), prefix="cost_",
    )
    assert record.since and record.until
    assert report_block["scan_since"] == cost_block["scan_since"] == record.since
    assert report_block["scan_until"] == cost_block["scan_until"] == record.until
    # The legacy per-store keys are still written, and are DERIVED from the same
    # record rather than resolved beside it — an older reader keeps working and
    # cannot see a different span than the record reports.
    stored_cost = relearn_store.read_cost_proposals(config=cfg) or {}
    assert stored_cost["cost_since"] == record.since
    assert stored_cost["cost_until"] == record.until


def test_the_record_never_invents_bounds_it_was_not_given(cfg):
    """A bare day count is not provenance. Deriving `since`/`until` from it would
    manufacture the very evidence the fields exist to supply, so a caller with a
    length and no bounds gets absent bounds."""
    record = cycle_provenance.begin_cycle(window_days=30)
    assert record.window_days == 30.0
    assert record.since is None
    assert record.until is None
    assert record.build == cycle_provenance.tj_build()
    assert record.cycle_id


# --------------------------------------------------------------------------- #
# 5. An OLD cache — written before the record existed — degrades, never crashes
# --------------------------------------------------------------------------- #
def test_a_report_cache_predating_the_record_degrades_to_its_legacy_keys(cfg):
    """The upgrade path as it really arrives: a cache on disk with the flat keys
    and no record. Everything it does carry is still published; only the cycle
    identity is genuinely unavailable, and it reports `None` rather than
    borrowing the running build's."""
    p = report_store.default_report_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "computed_at": "2026-07-20T09:00:00+00:00",
        "report": {"findings": {}},
        "window_days": 30,
        "since": "2026-06-20T09:00:00+00:00",
        "until": "2026-07-20T09:00:00+00:00",
        "tj_version": "0.0.1-previous",
    }), encoding="utf-8")

    block = report_store.stored_report_block(cfg)
    assert block["cycle_id"] is None
    assert block["provenance"] is None
    assert block["computed_build"] == "0.0.1-previous"
    assert block["build_provenance"] == cycle_provenance.BUILD_STALE
    assert block["scan_since"] == "2026-06-20T09:00:00+00:00"
    assert block["scan_until"] == "2026-07-20T09:00:00+00:00"
    assert block["status"] == report_store.STATUS_READY


def test_a_cost_cache_predating_the_record_degrades_to_its_legacy_keys(cfg):
    p = relearn_store.default_cache_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "computed_at": "2026-07-20T09:00:00+00:00",
        "finding": {"clusters": []},
        "cost_proposals": [],
        "cost_computed_at": "2026-07-20T09:00:00+00:00",
        "cost_since": "2026-06-20T09:00:00+00:00",
        "cost_until": "2026-07-20T09:00:00+00:00",
        "cost_tj_version": "0.0.1-previous",
    }), encoding="utf-8")

    block = cycle_provenance.provenance_block(
        relearn_store.read_cost_proposals(config=cfg), prefix="cost_",
    )
    assert block["cycle_id"] is None
    assert block["computed_build"] == "0.0.1-previous"
    assert block["build_provenance"] == cycle_provenance.BUILD_STALE
    assert block["scan_since"] == "2026-06-20T09:00:00+00:00"
    assert block["scan_until"] == "2026-07-20T09:00:00+00:00"


def test_a_corrupt_or_half_written_record_falls_back_instead_of_raising(cfg):
    """A record without an identity cannot answer the question the record exists
    for, so it is treated as absent — never raised on, and never half-trusted."""
    for junk in (None, "not-a-mapping", 17, {}, {"build": "9.9.9"}, {"cycle_id": ""}):
        assert cycle_provenance.CycleProvenance.from_dict(junk) is None

    p = report_store.default_report_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "computed_at": "2026-07-20T09:00:00+00:00",
        "report": {"findings": {}},
        "provenance": {"build": "9.9.9"},          # no cycle_id
        "tj_version": "0.0.1-previous",
    }), encoding="utf-8")
    block = report_store.stored_report_block(cfg)
    assert block["cycle_id"] is None
    assert block["computed_build"] == "0.0.1-previous"


def test_the_cost_record_survives_the_relearn_leg_writing_over_it(cfg):
    """The two producers SHARE one cache file and the relearn leg rewrites it.
    The record is a new `cost_`-prefixed key, which is exactly the shape that
    used to be dropped silently by a whitelist nobody updated."""
    record = cycle_provenance.begin_cycle(cfg)
    relearn_store.write_cost_proposals([], config=cfg, provenance=record)
    relearn_store.write_cache(_Finding(), config=cfg, provenance=record)

    after = relearn_store.read_cost_proposals(config=cfg) or {}
    assert after["cost_provenance"]["cycle_id"] == record.cycle_id
