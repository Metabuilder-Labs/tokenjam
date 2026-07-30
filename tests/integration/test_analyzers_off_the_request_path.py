"""The architectural guarantee: no HTTP request path runs an analyzer.

Three claims, each of which has been violated in production code before:

1. **No route reaches `build_report`'s analyzer dispatch.** `GET /optimize` had
   no cache read at all — it called `build_report` on the request thread, which
   dispatched every registered analyzer including `relearn`, the same
   full-corpus scan `relearn_store` exists to cache. `/reuse/clusters`,
   `/cost/components` and `/cost/cache` did the same. This is the guard that
   fails if any of them (or a new route) reintroduces it.

2. **A cold store renders as not-yet-computed, never as a zero or an absence
   claim.** Zero is the worst possible placeholder in this product: "no
   recoverable waste" reads as reassurance, and an un-run scan cannot support
   it.

3. **Overlapping rescans no-op rather than stacking.** A user pressing rescan
   twice, or pressing it while the scheduled job runs, must cost one
   full-corpus pass, not two.
"""
from __future__ import annotations

import threading
from datetime import timedelta

import httpx
import pytest

from tokenjam.api.app import create_app
from tokenjam.core.config import TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.optimize import report_store
from tokenjam.core.optimize.registry import ANALYZER_REGISTRY
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span

# Every analyzer-consuming GET. A new one belongs here the day it ships: the
# point of the list is that the guarantee is checked route-by-route rather
# than for whichever route someone remembered.
ANALYZER_CONSUMING_ROUTES = [
    "/api/v1/optimize?since=30d",
    "/api/v1/optimize?since=30d&fast=true",
    "/api/v1/reuse/clusters?since=30d",
    "/api/v1/cost/components?since=30d",
    "/api/v1/cost/cache?since=30d",
]


def _app(db, config):
    return create_app(config=config, db=db, ingest_pipeline=IngestPipeline(db=db, config=config))


def _seed(db):
    """Enough for several analyzers to have something to say, so a route that
    DID dispatch would visibly dispatch."""
    now = utcnow()
    for i in range(6):
        db.upsert_session(make_session(session_id=f"s{i}", agent_id="cc", plan_tier="api"))
        llm = make_llm_span(
            session_id=f"s{i}", agent_id="cc", model="claude-opus-4-7",
            provider="anthropic", input_tokens=1200, output_tokens=200,
            cache_tokens=3000, cache_write_tokens=800, cost_usd=0.05,
            start_time=now - timedelta(days=i),
        )
        db.insert_span(llm)
        for _ in range(2):
            t = make_tool_span(agent_id="cc", tool_name="Read", trace_id=llm.trace_id)
            t.session_id = f"s{i}"
            t.start_time = now - timedelta(days=i)
            db.insert_span(t)


def _trap_analyzers(monkeypatch) -> list[str]:
    """Wrap every registered analyzer so any dispatch is recorded by name."""
    called: list[str] = []
    for name, fn in list(ANALYZER_REGISTRY.items()):
        def _tracking(ctx, _name=name, _fn=fn):
            called.append(_name)
            return _fn(ctx)
        monkeypatch.setitem(ANALYZER_REGISTRY, name, _tracking)
    return called


# --------------------------------------------------------------------------- #
# 1. No route handler reaches the analyzer dispatch.
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
@pytest.mark.parametrize("route", ANALYZER_CONSUMING_ROUTES)
async def test_no_route_dispatches_an_analyzer_with_a_warm_store(route, monkeypatch):
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    report_store.recompute_now(db, cfg)

    called = _trap_analyzers(monkeypatch)
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get(route)

    assert resp.status_code == 200, resp.text
    assert called == [], f"{route} dispatched analyzers on the request thread: {called}"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ANALYZER_CONSUMING_ROUTES)
async def test_no_route_dispatches_an_analyzer_with_a_cold_store(route, monkeypatch):
    """The cold path is the one that would be tempted to compute-on-demand —
    "nothing stored, so just run it this once" is exactly how the inline
    dispatch would come back."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)

    called = _trap_analyzers(monkeypatch)
    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get(route)

    assert resp.status_code == 200, resp.text
    assert called == [], f"{route} computed on a cold store: {called}"


@pytest.mark.asyncio
async def test_build_report_is_unreachable_from_every_route(monkeypatch):
    """Belt-and-braces on the orchestrator itself: no route may call
    `build_report`, whatever analyzer set it would have asked for."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)
    report_store.recompute_now(db, cfg)

    import tokenjam.core.optimize.runner as runner_mod

    def _forbidden(*a, **kw):
        raise AssertionError("a route handler called build_report")

    monkeypatch.setattr(runner_mod, "build_report", _forbidden)
    monkeypatch.setattr("tokenjam.core.optimize.build_report", _forbidden)

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        for route in ANALYZER_CONSUMING_ROUTES:
            assert (await c.get(route)).status_code == 200, route


# --------------------------------------------------------------------------- #
# 2. Cold is not empty, and never a zero.
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_cold_store_reports_never_run_and_withholds_the_report():
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/optimize?since=30d")).json()

    assert d["status"] == "never_run"
    assert d["report_available"] is False
    assert d["computed_at"] is None
    # No report body at all — not an empty one. An empty `findings` dict would
    # render as "every analyzer ran and found nothing", which is false.
    assert "findings" not in d
    assert "downgrade" not in d
    assert "finding_rank" not in d


@pytest.mark.asyncio
async def test_cold_store_never_publishes_a_zero_recoverable_figure():
    """A `$0.00` / `0` recoverable figure off an un-run scan is the single
    worst failure mode here: zero reads as reassurance. `None` — "not
    measured" — is the only honest value."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        comp = (await c.get("/api/v1/cost/components?since=30d")).json()
        cache = (await c.get("/api/v1/cost/cache?since=30d")).json()

    assert comp["recoverable_status"] == "never_run"
    assert comp["recoverable_available"] is False
    assert comp["total_recoverable_usd"] is None
    assert comp["total_recoverable_tokens"] is None
    assert comp["largest_recoverable_usd"] is None
    # The MEASURED component bars are unaffected — ingestion is untouched, so
    # the live spend split still renders. Only the analyzer overlay is cold.
    assert comp["total_cost_usd"] > 0

    assert cache["recoverable_available"] is False
    assert cache["past_overspend_usd"] is None
    assert cache["past_overspend_tokens"] is None


@pytest.mark.asyncio
async def test_a_computed_and_empty_result_is_a_distinct_state_from_cold():
    """The whole point of the distinction: an EMPTY corpus that was actually
    scanned reports `ready`, and only then may a surface say "found nothing"."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    report_store.recompute_now(db, cfg)   # scan an empty corpus

    transport = httpx.ASGITransport(app=_app(db, cfg))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.get("/api/v1/optimize?since=30d")).json()
        comp = (await c.get("/api/v1/cost/components?since=30d")).json()

    assert d["status"] == "ready"
    assert d["report_available"] is True
    assert d["computed_at"] is not None
    # Ready-and-empty CAN legitimately be zero: a scan really did run.
    assert comp["recoverable_status"] == "ready"
    assert comp["recoverable_available"] is True
    assert comp["total_recoverable_usd"] == 0


def test_report_status_separates_cold_from_computed_empty():
    assert report_store.report_status(None, computing=False) == report_store.STATUS_NEVER_RUN
    assert report_store.report_status({}, computing=False) == report_store.STATUS_NEVER_RUN
    # Only-ever-failed is its own state: still nothing measured, but for a
    # reason worth telling the user.
    assert report_store.report_status(
        {"error": "boom"}, computing=False,
    ) == report_store.STATUS_ERROR
    # A completed scan with NO findings is ready — the one state where an
    # empty-state string is honest.
    assert report_store.report_status(
        {"computed_at": "2026-01-01T00:00:00+00:00", "report": {}}, computing=False,
    ) == report_store.STATUS_READY
    # A scan in flight over a previous good result keeps that result and is
    # reported as computing, never as cold.
    block = {"computed_at": "2026-01-01T00:00:00+00:00", "report": {}}
    assert report_store.report_status(block, computing=True) == report_store.STATUS_COMPUTING


def test_a_failed_rescan_never_discards_the_last_good_report(tmp_path):
    """A transient failure must not turn a populated surface into an empty
    one — the last good result stays and the failure is disclosed beside it."""
    path = tmp_path / "report.json"
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    report_store.write_report({"findings": {"cache": {}}}, path, window_days=30)
    report_store.write_report_error("scan blew up", path)

    stored = report_store.read_report(path)
    assert stored["report"] == {"findings": {"cache": {}}}
    assert stored["error"] == "scan blew up"
    block = report_store.stored_report_block(cfg, path=path)
    assert block["status"] == report_store.STATUS_READY
    assert block["degraded"] is True
    assert block["last_error"] == "scan blew up"
    assert db is not None   # keeps the backend alive for the duration


# --------------------------------------------------------------------------- #
# 3. Overlapping rescans no-op instead of stacking.
# --------------------------------------------------------------------------- #

def test_overlapping_recomputes_no_op_rather_than_stacking(tmp_path, monkeypatch):
    """Two concurrent recomputes must produce ONE full-corpus pass. The second
    returns `None` immediately; it never blocks waiting on the first, which
    would just move the cost rather than avoid it."""
    path = tmp_path / "report.json"
    db = InMemoryBackend()
    cfg = TjConfig(version="1")

    passes: list[int] = []
    first_inside = threading.Event()
    release_first = threading.Event()

    def _slow_build_report(**kwargs):
        passes.append(1)
        first_inside.set()
        release_first.wait(timeout=5)
        from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
        now = utcnow()
        return OptimizeReport(window=WindowSummary(
            since=now - timedelta(days=30), until=now, days=30.0,
            sessions=0, spans=0, total_tokens=0, total_cost_usd=0.0,
        ))

    monkeypatch.setattr("tokenjam.core.optimize.build_report", _slow_build_report)

    result: dict[str, object] = {}

    def _first() -> None:
        result["first"] = report_store.recompute_now(db, cfg, path=path)

    t = threading.Thread(target=_first)
    t.start()
    assert first_inside.wait(timeout=5), "the first recompute never started"

    # While the first holds the lock: a second synchronous call no-ops, and the
    # background trigger declines to start a thread at all.
    assert report_store.is_computing() is True
    assert report_store.recompute_now(db, cfg, path=path) is None
    assert report_store.trigger_background_recompute(
        lambda: db, cfg, path=path,
    ) is False

    release_first.set()
    t.join(timeout=5)
    assert result["first"] is not None
    assert passes == [1], f"expected exactly one corpus pass, got {len(passes)}"
    assert report_store.is_computing() is False


@pytest.mark.asyncio
async def test_rescan_endpoint_declines_while_a_scan_is_in_flight(monkeypatch):
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)

    monkeypatch.setattr(report_store, "is_computing", lambda: True)
    app = _app(db, cfg)
    transport = httpx.ASGITransport(app=app)
    hdr = {"X-TJ-Local-Token": app.state.relearn_write_token}
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.post("/api/v1/optimize/rescan", json={}, headers=hdr)).json()

    assert d["started"] is False
    assert "already running" in d["reason"]


@pytest.mark.asyncio
async def test_rescan_endpoint_is_rate_limited(monkeypatch):
    """A user cannot hammer rescan into stacking full-corpus passes: a request
    inside the configured floor is answered with the stored result."""
    db = InMemoryBackend()
    cfg = TjConfig(version="1")
    _seed(db)

    monkeypatch.setattr(report_store, "rescan_throttled", lambda config=None: True)
    started: list[bool] = []
    monkeypatch.setattr(
        report_store, "trigger_background_recompute",
        lambda *a, **kw: started.append(True) or True,
    )

    app = _app(db, cfg)
    transport = httpx.ASGITransport(app=app)
    hdr = {"X-TJ-Local-Token": app.state.relearn_write_token}
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        d = (await c.post("/api/v1/optimize/rescan", json={}, headers=hdr)).json()

    assert d["throttled"] is True
    assert d["started"] is False
    assert started == [], "a throttled rescan must not start a scan"


def test_rescan_throttle_respects_the_configured_floor(monkeypatch):
    cfg = TjConfig(version="1")
    cfg.optimize.scan_min_rescan_seconds = 60
    monkeypatch.setattr(report_store, "seconds_since_last_run", lambda: 5.0)
    assert report_store.rescan_throttled(cfg) is True
    monkeypatch.setattr(report_store, "seconds_since_last_run", lambda: 120.0)
    assert report_store.rescan_throttled(cfg) is False
    # Never-run is never throttled — a fresh daemon must be rescannable at once.
    monkeypatch.setattr(report_store, "seconds_since_last_run", lambda: None)
    assert report_store.rescan_throttled(cfg) is False
    # Zero disables the floor entirely (the documented escape hatch).
    cfg.optimize.scan_min_rescan_seconds = 0
    monkeypatch.setattr(report_store, "seconds_since_last_run", lambda: 1.0)
    assert report_store.rescan_throttled(cfg) is False
