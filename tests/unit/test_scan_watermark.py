"""The ingestion-watermark gate on the SCHEDULED analyzer-pass trigger.

THE DEFECT THIS GUARDS AGAINST. The daemon's scan-cycle job used to be a pure
wall-clock ``interval`` — it re-ran the full analyzer pass, including the
relearn distill pass's LLM subprocess calls, every ``scan_interval_hours``
regardless of whether any new telemetry had landed. On an idle machine that
recomputes an answer that cannot have changed.

Four properties:

1. An idle tick (no new spans since the last pass) is a no-op.
2. A tick after enough new spans have landed DOES trigger.
3. The staleness ceiling forces a pass on a quiet machine even with zero new
   spans, so a machine that never sees traffic still refreshes eventually.
4. An explicit rescan (``force=True``) always attempts a pass, bypassing the
   gate entirely — a human asking is never deferred to the next tick.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.config import OptimizeConfig, StorageConfig, TjConfig
from tokenjam.core.optimize import ingest_watermark, report_store, scan_cycle
from tokenjam.utils.time_parse import utcnow


@pytest.fixture(autouse=True)
def _reset_watermark_state():
    """Both halves are MODULE-GLOBAL — a leaked value from one test would
    silently gate (or un-gate) every later test in the process, in any file.
    Cleared on both sides, same pattern as `_CYCLE_COMPUTING` in
    `test_scan_cycle_provenance.py`."""
    scan_cycle._CYCLE_COMPUTING.clear()
    scan_cycle._last_pass_watermark = None
    scan_cycle._last_pass_at = None
    ingest_watermark._count = 0
    yield
    scan_cycle._CYCLE_COMPUTING.clear()
    scan_cycle._last_pass_watermark = None
    scan_cycle._last_pass_at = None
    ingest_watermark._count = 0


@pytest.fixture
def cfg(tmp_path) -> TjConfig:
    return TjConfig(
        version="1",
        storage=StorageConfig(path=str(tmp_path / "t.duckdb")),
        optimize=OptimizeConfig(
            scan_watermark_min_new_spans=5,
            scan_watermark_max_staleness_hours=12.0,
        ),
    )


def _fake_thread_capturing(sink: list):
    def _fake_thread(target=None, name=None, daemon=None):
        class _T:
            def start(_self):
                sink.append(target)
        return _T()
    return _fake_thread


# --------------------------------------------------------------------------- #
# 0. Nothing has run yet in this process -> the gate must not block the first
#    tick (the startup kick relies on exactly this).
# --------------------------------------------------------------------------- #
def test_first_tick_in_a_fresh_process_always_runs(cfg):
    assert scan_cycle._should_run_scheduled_pass(cfg) is True


# --------------------------------------------------------------------------- #
# 1. Idle tick: no new spans since the last completed pass -> no-op
# --------------------------------------------------------------------------- #
def test_idle_tick_with_no_new_spans_does_not_trigger(monkeypatch, cfg):
    scan_cycle._record_pass_watermark(ingest_watermark.current())

    dispatched: list = []
    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread_capturing(dispatched))
    monkeypatch.setattr(report_store, "is_computing", lambda: False)

    started = scan_cycle.trigger_scan_cycle(lambda: None, cfg)

    assert started == {"analyzer_pass": False}
    assert not dispatched, "an idle tick must never dispatch the pass thread"


# --------------------------------------------------------------------------- #
# 2. A tick after enough new spans DOES trigger
# --------------------------------------------------------------------------- #
def test_tick_after_enough_new_sessions_triggers(monkeypatch, cfg):
    scan_cycle._record_pass_watermark(ingest_watermark.current())
    ingest_watermark.bump(cfg.optimize.scan_watermark_min_new_spans)

    dispatched: list = []
    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread_capturing(dispatched))
    monkeypatch.setattr(report_store, "is_computing", lambda: False)

    started = scan_cycle.trigger_scan_cycle(lambda: None, cfg)

    assert started == {"analyzer_pass": True}
    assert dispatched, "enough new spans must dispatch the pass thread"


def test_a_trickle_below_the_threshold_does_not_trigger(monkeypatch, cfg):
    scan_cycle._record_pass_watermark(ingest_watermark.current())
    ingest_watermark.bump(cfg.optimize.scan_watermark_min_new_spans - 1)

    dispatched: list = []
    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread_capturing(dispatched))
    monkeypatch.setattr(report_store, "is_computing", lambda: False)

    started = scan_cycle.trigger_scan_cycle(lambda: None, cfg)

    assert started == {"analyzer_pass": False}
    assert not dispatched


# --------------------------------------------------------------------------- #
# 3. The staleness ceiling forces a pass on a quiet machine
# --------------------------------------------------------------------------- #
def test_the_ceiling_forces_a_pass_on_a_quiet_machine(monkeypatch, cfg):
    # No new spans at all — only the elapsed time crosses the ceiling.
    scan_cycle._record_pass_watermark(ingest_watermark.current())
    scan_cycle._last_pass_at = utcnow() - timedelta(
        hours=cfg.optimize.scan_watermark_max_staleness_hours + 1,
    )

    dispatched: list = []
    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread_capturing(dispatched))
    monkeypatch.setattr(report_store, "is_computing", lambda: False)

    started = scan_cycle.trigger_scan_cycle(lambda: None, cfg)

    assert started == {"analyzer_pass": True}
    assert dispatched, "a stale-enough machine must refresh even with zero new spans"


def test_within_the_ceiling_and_below_threshold_stays_quiet(cfg):
    scan_cycle._record_pass_watermark(ingest_watermark.current())
    scan_cycle._last_pass_at = utcnow() - timedelta(
        hours=cfg.optimize.scan_watermark_max_staleness_hours - 1,
    )
    assert scan_cycle._should_run_scheduled_pass(cfg) is False


# --------------------------------------------------------------------------- #
# 4. An explicit user rescan always runs, watermark or not
# --------------------------------------------------------------------------- #
def test_explicit_rescan_always_runs_regardless_of_the_watermark(monkeypatch, cfg):
    # Set up the exact state that gates a scheduled tick...
    scan_cycle._record_pass_watermark(ingest_watermark.current())

    dispatched: list = []
    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread_capturing(dispatched))
    monkeypatch.setattr(report_store, "is_computing", lambda: False)

    # ...and confirm the SAME state would have declined a scheduled tick.
    assert scan_cycle._should_run_scheduled_pass(cfg) is False

    started = scan_cycle.trigger_scan_cycle(lambda: None, cfg, force=True)

    assert started == {"analyzer_pass": True}
    assert dispatched, "force=True must never be deferred by the watermark gate"


# --------------------------------------------------------------------------- #
# 5. A completed pass advances the watermark baseline
# --------------------------------------------------------------------------- #
def test_a_completed_pass_advances_the_watermark_baseline(monkeypatch, cfg):
    ingest_watermark.bump(cfg.optimize.scan_watermark_min_new_spans)

    monkeypatch.setattr(report_store, "is_computing", lambda: False)
    monkeypatch.setattr(
        report_store, "recompute_now",
        lambda backend, config, until=None, provenance=None: {"computed_at": "now"},
    )
    monkeypatch.setattr(report_store, "stored_report", lambda config: object())
    monkeypatch.setattr(scan_cycle, "_write_relearn_from", lambda *a, **k: None)

    import tokenjam.core.optimize.cost_proposals as cp
    monkeypatch.setattr(
        cp, "recompute_cost_proposals",
        lambda backend, config, until=None, report=None, provenance=None: None,
    )
    monkeypatch.setattr(scan_cycle, "_refresh_rule_presence", lambda config: None)

    threads: list = []
    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread_capturing(threads))

    started = scan_cycle.trigger_scan_cycle(lambda: None, cfg)
    assert started == {"analyzer_pass": True}
    threads[0]()  # run the job body inline

    assert scan_cycle._last_pass_watermark == ingest_watermark.current()
    assert scan_cycle._last_pass_at is not None

    # A second tick right after, with no further ingestion, is now idle again.
    dispatched: list = []
    monkeypatch.setattr(scan_cycle.threading, "Thread", _fake_thread_capturing(dispatched))
    started_again = scan_cycle.trigger_scan_cycle(lambda: None, cfg)
    assert started_again == {"analyzer_pass": False}
    assert not dispatched
