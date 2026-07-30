"""The server half of "the refresh control must not assert more than it knows".

Three properties, all of them things a surface cannot derive on its own:

1. **The cycle carries its own in-flight flag.** One pass writes three stores
   sequentially on one thread, and each store's own guard goes false as its leg
   lands — so a report-only flag went quiet while relearn (the most expensive
   analyzer in the product) and the cost proposals were still being built.
2. **A refusal always carries a reason.** ``POST /optimize/rescan`` answered 200
   with a bare ``started: false`` on the cycle-guard path, giving the client
   nothing to distinguish a refusal from a start.
3. **The producing build travels with the result.** These stores are caches and
   an upgrade does not invalidate them, so ``computed_at`` alone lets a previous
   build's figures read as merely recent.

Plus the payload half of the apply surface: an applied cost proposal must be
neither ``apply_capable`` nor missing an applied state, verifiable WITHOUT
consulting a second endpoint.
"""
from __future__ import annotations

import json

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.optimize import cost_apply, report_store, scan_cycle
from tokenjam.core.optimize.build_stamp import UNKNOWN, tj_build


@pytest.fixture(autouse=True)
def _no_leaked_cycle_flag():
    """The cycle's in-flight flag is a MODULE-GLOBAL Event, so a test that leaves
    it set makes every later test in the process — in any file — believe a scan is
    running. Cleared on both sides: before, so a leak from elsewhere cannot make a
    test here pass or fail for the wrong reason; after, so nothing this module does
    escapes it. The individual tests still clear it in their own `finally`; this is
    the backstop for the ones that fail before reaching it."""
    scan_cycle._CYCLE_COMPUTING.clear()
    yield
    scan_cycle._CYCLE_COMPUTING.clear()


@pytest.fixture
def cfg(tmp_path) -> TjConfig:
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


# --------------------------------------------------------------------------- #
# 1. The cycle's in-flight flag outlives the report's
# --------------------------------------------------------------------------- #
def test_cycle_flag_is_set_before_the_trigger_returns(monkeypatch, cfg):
    """A GET racing the POST's return must already see the cycle.

    Set before the dispatch rather than at the top of the job, or the first poll
    after a press reads "not scanning" and the control flickers.
    """
    released = []

    def _fake_thread(target=None, name=None, daemon=None):
        class _T:
            def start(_self):
                released.append(target)   # captured, never run
        return _T()

    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread)
    monkeypatch.setattr(report_store, "is_computing", lambda: False)
    assert scan_cycle.is_cycle_computing() is False
    try:
        assert scan_cycle._trigger_analyzer_pass(lambda: None, cfg, None) is True
        assert scan_cycle.is_cycle_computing() is True, (
            "the cycle flag must be visible the moment the trigger returns"
        )
    finally:
        scan_cycle._CYCLE_COMPUTING.clear()
    assert released, "the job was never dispatched"


def test_cycle_flag_outlives_the_report_leg(monkeypatch, cfg):
    """THE DEFECT. The report has landed and its own flag is false; the pass is
    still building the two stores the Review inbox reads. The cycle must still
    report as computing at that instant."""
    seen: dict[str, bool] = {}

    def _recompute_now(backend, config, until=None):
        return {"computed_at": "now"}

    def _write_relearn(report, config, factory):
        # Mid-pass: the report's leg is done, relearn's is not.
        seen["report_flag"] = report_store.is_computing()
        seen["cycle_flag"] = scan_cycle.is_cycle_computing()

    monkeypatch.setattr(report_store, "is_computing", lambda: False)
    monkeypatch.setattr(report_store, "recompute_now", _recompute_now)
    monkeypatch.setattr(report_store, "stored_report", lambda config: object())
    monkeypatch.setattr(scan_cycle, "_write_relearn_from", _write_relearn)

    import tokenjam.core.optimize.cost_proposals as cp

    monkeypatch.setattr(
        cp, "recompute_cost_proposals",
        lambda backend, config, until=None, report=None: None,
    )
    # Run the job body inline instead of on a thread, so the mid-pass instant is
    # observable at all.
    threads = []

    def _fake_thread(target=None, name=None, daemon=None):
        class _T:
            def start(_self):
                threads.append(target)
        return _T()

    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread)
    scan_cycle._trigger_analyzer_pass(lambda: None, cfg, None)
    threads[0]()

    assert seen["report_flag"] is False, "fixture no longer models the defect"
    assert seen["cycle_flag"] is True, (
        "the cycle went quiet while relearn and the cost proposals were still building"
    )
    assert scan_cycle.is_cycle_computing() is False, "the flag leaked past the pass"


def test_cycle_flag_is_cleared_when_the_pass_raises(monkeypatch, cfg):
    """A flag left set with no pass behind it shows "Scanning…" forever AND
    disables the button that would recover it."""
    monkeypatch.setattr(report_store, "is_computing", lambda: False)
    monkeypatch.setattr(
        report_store, "recompute_now",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    threads = []

    def _fake_thread(target=None, name=None, daemon=None):
        class _T:
            def start(_self):
                threads.append(target)
        return _T()

    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread)
    scan_cycle._trigger_analyzer_pass(lambda: None, cfg, None)
    threads[0]()
    assert scan_cycle.is_cycle_computing() is False


def test_a_second_trigger_declines_while_a_cycle_is_in_flight(monkeypatch, cfg):
    """The overlap guard the UI's "already running" state depends on. The report
    store's own flag is false mid-pass, so without the cycle guard a second press
    would start a duplicate pass over the same corpus."""
    monkeypatch.setattr(report_store, "is_computing", lambda: False)

    def _fake_thread(target=None, name=None, daemon=None):
        class _T:
            def start(_self):
                pass
        return _T()

    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread)
    try:
        assert scan_cycle._trigger_analyzer_pass(lambda: None, cfg, None) is True
        assert scan_cycle._trigger_analyzer_pass(lambda: None, cfg, None) is False
    finally:
        scan_cycle._CYCLE_COMPUTING.clear()


# --------------------------------------------------------------------------- #
# 2 + 3. The envelope every analyzer route returns
# --------------------------------------------------------------------------- #
def test_envelope_publishes_the_cycle_and_both_builds(cfg):
    block = report_store.stored_report_block(cfg)
    assert block["cycle_computing"] is False
    assert block["build"] == tj_build()
    # Cold store: no result, so nothing produced it. `None`, never the serving
    # build — that would assert agreement about figures that do not exist.
    assert block["computed_build"] is None


def test_a_written_report_records_the_build_that_produced_it(cfg):
    report_store.write_report({"findings": {}}, config=cfg, window_days=30)
    block = report_store.stored_report_block(cfg)
    assert block["computed_build"] == tj_build()
    assert block["computed_build"] == block["build"]


def test_a_report_written_by_another_build_is_visibly_not_this_one(cfg):
    """The upgrade case, as it actually arrives: a cache on disk stamped by the
    replaced binary. Nothing invalidates it, so the payload has to disclose it."""
    report_store.write_report({"findings": {}}, config=cfg, window_days=30)
    p = report_store.default_report_path(cfg)
    stored = json.loads(p.read_text(encoding="utf-8"))
    stored["tj_version"] = "0.0.1-previous"
    p.write_text(json.dumps(stored), encoding="utf-8")

    block = report_store.stored_report_block(cfg)
    assert block["computed_build"] == "0.0.1-previous"
    assert block["build"] != "0.0.1-previous"


def test_a_report_predating_the_stamp_reports_unknown_not_agreement(cfg):
    report_store.write_report({"findings": {}}, config=cfg, window_days=30)
    p = report_store.default_report_path(cfg)
    stored = json.loads(p.read_text(encoding="utf-8"))
    del stored["tj_version"]
    p.write_text(json.dumps(stored), encoding="utf-8")

    block = report_store.stored_report_block(cfg)
    assert block["computed_build"] is None
    assert block["build"] == tj_build()


def test_the_build_stamp_never_raises_and_never_returns_empty(monkeypatch):
    assert tj_build()
    import builtins

    real_import = builtins.__import__

    def _boom(name, *a, **k):
        if name == "tokenjam":
            raise ImportError("no package metadata")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert tj_build() == UNKNOWN


# --------------------------------------------------------------------------- #
# 4. An applied cost proposal loses its OFFER and keeps its FIGURE
# --------------------------------------------------------------------------- #
def _ledger(cfg: TjConfig, records: list[dict]) -> None:
    p = cost_apply.cost_applied_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records), encoding="utf-8")


def test_an_applied_proposal_is_not_apply_capable_on_the_payload(cfg):
    """THE DEFECT. `cost:deadweight:posthog` came back `apply_capable: true`
    with no applied field of any kind, because the route filtered a separate
    `open_proposals` for its rollup and returned the unfiltered list. Only the
    browser was safe, and only by cross-referencing a second endpoint."""
    _ledger(cfg, [{
        "signature": "cost:deadweight:posthog", "state": "applied",
        "applied_at": "2026-07-20T09:00:00Z",
    }])
    rows = cost_apply.stamp_applied_state([{
        "signature": "cost:deadweight:posthog",
        "apply_capable": True,
        "past_overspend_usd": 358.50,
        "past_overspend_tokens": 1234,
    }], config=cfg)
    assert rows[0]["apply_capable"] is False
    assert rows[0]["applied"] is True
    assert rows[0]["applied_at"] == "2026-07-20T09:00:00Z"
    # Critical Rule 32: the OFFER is withdrawn, the FIGURE is untouched. The
    # waste happened; applying the fix afterwards does not un-spend it.
    assert rows[0]["past_overspend_usd"] == 358.50
    assert rows[0]["past_overspend_tokens"] == 1234


def test_an_open_proposal_carries_the_field_explicitly(cfg):
    """Absent is not false. A reader must not have to distinguish "the ledger
    says open" from "this payload never consulted a ledger"."""
    _ledger(cfg, [])
    rows = cost_apply.stamp_applied_state([{
        "signature": "cost:deadweight:other", "apply_capable": True,
    }], config=cfg)
    assert rows[0]["applied"] is False
    assert rows[0]["applied_at"] is None
    assert rows[0]["apply_capable"] is True, "an open offer must survive intact"


def test_a_reverted_mark_reopens_the_offer(cfg):
    """A revert is the user saying the fix is no longer in place."""
    _ledger(cfg, [{
        "signature": "cost:deadweight:posthog", "state": "reverted",
        "applied_at": "2026-07-20T09:00:00Z",
    }])
    rows = cost_apply.stamp_applied_state([{
        "signature": "cost:deadweight:posthog", "apply_capable": True,
    }], config=cfg)
    assert rows[0]["applied"] is False
    assert rows[0]["apply_capable"] is True


def test_the_legacy_downsize_signature_resolves_through_one_matcher(cfg):
    """A mark recorded under the old agent-only form still covers the later
    model-qualified signatures. Stamping must honour the SAME matcher the
    rollup's filter uses, or a row reads open here and applied there."""
    _ledger(cfg, [{
        "signature": "cost:downsize:agent-a", "state": "applied",
        "applied_at": "2026-07-01T00:00:00Z",
    }])
    sig = "cost:downsize:agent-a:anthropic:opus:sonnet"
    rows = cost_apply.stamp_applied_state(
        [{"signature": sig, "apply_capable": True}], config=cfg,
    )
    assert rows[0]["applied"] is True
    assert cost_apply.signature_is_applied(
        sig, cost_apply.applied_signatures(cfg),
    ) is True


def test_an_unreadable_ledger_leaves_the_offers_standing(cfg):
    """The safe direction: wasting attention beats hiding a fix the user never
    made."""
    p = cost_apply.cost_applied_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    rows = cost_apply.stamp_applied_state([{
        "signature": "cost:deadweight:posthog", "apply_capable": True,
    }], config=cfg)
    assert rows[0]["applied"] is False
    assert rows[0]["apply_capable"] is True


def test_stamping_does_not_mutate_the_input_rows(cfg):
    """Immutability: the stored proposals are read by other consumers in the
    same request."""
    _ledger(cfg, [{
        "signature": "s", "state": "applied", "applied_at": "2026-07-20T09:00:00Z",
    }])
    original = {"signature": "s", "apply_capable": True}
    cost_apply.stamp_applied_state([original], config=cfg)
    assert original == {"signature": "s", "apply_capable": True}


# --------------------------------------------------------------------------- #
# 5. On the WIRE — verified without consulting a second endpoint
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(tmp_path):
    """A real app over a real request, because every property above is a claim
    about what a CONSUMER sees. The CLI, `--json` and any export read these
    payloads with no browser to cross-reference a second endpoint for them."""
    from fastapi.testclient import TestClient

    from tokenjam.api.app import create_app
    from tokenjam.core.config import ApiAuthConfig, ApiConfig, OptimizeConfig
    from tokenjam.core.db import InMemoryBackend
    from tokenjam.core.ingest import build_default_pipeline

    # `scan_enabled=False` IS LOAD-BEARING, not tidiness. App startup otherwise
    # kicks a real analyzer cycle on a background thread, and the in-flight flags
    # that thread sets are MODULE-GLOBAL — so a scan still running when this
    # fixture tears down leaks "computing" into whatever test runs next, in a
    # different file, on the same process. That is exactly how this module made
    # `test_relearn_proposals_carries_persona_when_never_run` fail on 3.10 while
    # passing on 3.11/3.12: a race, surfaced by test ordering rather than by any
    # logic difference. A test that only needs to read a payload shape must not
    # start a corpus pass.
    config = TjConfig(
        version="1",
        storage=StorageConfig(path=str(tmp_path / "t.duckdb")),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
        optimize=OptimizeConfig(scan_enabled=False),
    )
    db = InMemoryBackend()
    app = create_app(
        config=config, db=db, ingest_pipeline=build_default_pipeline(db, config),
    )
    with TestClient(app) as c:
        c.tj_config = config
        c.write_headers = {"X-TJ-Local-Token": app.state.relearn_write_token}
        yield c


def test_optimize_payload_carries_the_cycle_and_the_builds(client):
    body = client.get("/api/v1/optimize").json()
    assert "cycle_computing" in body
    assert body["build"] == tj_build()


def test_cost_proposals_payload_carries_the_cycle_and_the_builds(client):
    body = client.get("/api/v1/relearn/cost-proposals").json()
    for key in ("cycle_computing", "computed_build", "build"):
        assert key in body, f"{key} missing from the cost-proposal payload"
    assert body["build"] == tj_build()


def test_relearn_proposals_payload_carries_the_cycle_and_the_builds(client):
    body = client.get("/api/v1/relearn/proposals").json()
    for key in ("cycle_computing", "computed_build", "build"):
        assert key in body, f"{key} missing from the relearn payload"


def test_every_provenance_key_is_spelled_the_same_on_all_three_feeds(client):
    """One ScanBar reads all three surfaces. A key spelled differently per feed
    is a surface that silently loses the qualification — the same
    two-derivations-of-one-truth defect, at the key-name layer."""
    keys = {"cycle_computing", "computed_build", "build"}
    paths = ["/api/v1/optimize", "/api/v1/relearn/cost-proposals",
             "/api/v1/relearn/proposals"]
    for path in paths:
        body = client.get(path).json()
        assert keys <= set(body), f"{path} is missing {keys - set(body)}"


def test_a_declined_rescan_says_so_on_the_wire(client, monkeypatch):
    """`started: false` must never arrive without a reason to render. The two
    early-return paths always carried one; the cycle-guard path answered 200 with
    a bare false, which is indistinguishable from a start at the client."""
    from tokenjam.core.optimize import scan_cycle as sc

    monkeypatch.setattr(sc, "trigger_scan_cycle", lambda *a, **k: {"analyzer_pass": False})
    monkeypatch.setattr(report_store, "is_computing", lambda: False)
    monkeypatch.setattr(report_store, "rescan_throttled", lambda config: False)

    body = client.post(
        "/api/v1/optimize/rescan", headers=client.write_headers,
    ).json()
    if body.get("status") == "unavailable":
        pytest.skip("no direct database connection in this fixture")
    assert body["started"] is False
    assert body.get("reason"), "a refusal with nothing to render is a silent refusal"


def test_the_shared_cache_does_not_drop_the_cost_builds_stamp(cfg):
    """The relearn detector and the cost proposals SHARE one cache file, and the
    relearn write rebuilds the payload from a whitelist of `cost_*` keys. A key
    missing from that list is dropped silently, and the symptom is not an error —
    it is a payload reporting an unknown producing build.

    Caught on a live daemon: the pass writes the cost proposals and then the
    relearn cache over the top of them, so a freshly-booted, correctly-stamped
    cost write came back with `computed_build: null`.
    """
    from dataclasses import dataclass, field

    from tokenjam.core.optimize import relearn_store

    relearn_store.write_cost_proposals([], config=cfg, window_days=30)
    stamped = relearn_store.read_cost_proposals(config=cfg) or {}
    assert stamped["cost_tj_version"] == tj_build(), "the cost write did not stamp"

    @dataclass
    class _Finding:
        clusters: list = field(default_factory=list)

    relearn_store.write_cache(_Finding(), config=cfg)
    after = relearn_store.read_cost_proposals(config=cfg) or {}
    assert after["cost_tj_version"] == tj_build(), (
        "the relearn write dropped the cost build stamp"
    )
    # The keys this whitelist already carried, pinned in the same place so the
    # next addition is caught by the same test rather than by a live daemon.
    assert after["cost_window_days"] == 30
