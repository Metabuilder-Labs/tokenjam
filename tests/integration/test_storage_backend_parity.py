"""Cross-backend StorageBackend parity / contract suite (#51).

The bug-hunt sweep kept surfacing the same *class* of defect: a
``StorageBackend`` method returns something different through the ``tj serve``
HTTP shim (``ApiBackend``) than it does through the direct DuckDB backend —
``get_daily_cost`` returning a cumulative total in shim mode, ``_dict_to_span``
dropping ``cache_write_tokens`` on daemon-fetched spans, cache columns silently
zeroed, and so on. Each was filed and fixed individually, but nothing *enforced*
behavioral parity, so every new or changed method could diverge unnoticed —
and those are exactly the discrepancies users hit depending on whether
``tj serve`` is running.

This module is the structural guard:

1. ``test_shim_matches_db`` runs each faithfully-mirrored read method against
   BOTH the direct DuckDB backend and the ``ApiBackend`` shim (talking to a real
   in-process uvicorn server over the same DB) and fails on any divergence.
2. ``test_duckdb_and_in_memory_agree`` runs the same reads against the DuckDB
   file backend and the ``InMemoryBackend`` used by unit tests, so the three
   backends the codebase ships (DuckDB, in-memory, serve shim) stay in lockstep.
3. ``test_every_protocol_method_is_classified`` /
   ``test_unimplemented_methods_have_no_silent_shim`` are the enforcement layer:
   every ``StorageBackend`` protocol method must be explicitly classified as
   parity-covered, a documented known gap, a deliberately-unimplemented shim
   method, or lifecycle. Adding a protocol method, or teaching the shim a new
   method, fails CI until it is classified — which for a new read method means
   wiring it into the parity set here.

See ``tokenjam/CLAUDE.md`` → "StorageBackend parity" for the one-line rule.
"""
from __future__ import annotations

import dataclasses
import threading
import time
from contextlib import contextmanager
from datetime import timedelta

import pytest
import uvicorn

from tokenjam.api.app import create_app
from tokenjam.core.api_backend import ApiBackend
from tokenjam.core.config import (
    ApiAuthConfig,
    ApiConfig,
    SecurityConfig,
    StorageConfig,
    TjConfig,
)
from tokenjam.core.db import DuckDBBackend, InMemoryBackend, StorageBackend
from tokenjam.core.ingest import IngestPipeline
from tokenjam.core.models import (
    Alert,
    AlertFilters,
    AlertType,
    CostFilters,
    DriftBaseline,
    Severity,
    TraceFilters,
)
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span

AGENT = "parity-agent"
SESSION = "parity-session"


# ---------------------------------------------------------------------------
# Classification of every StorageBackend method (enforcement — see below).
# ---------------------------------------------------------------------------

# Read methods the shim is meant to mirror faithfully. Each MUST have a spec in
# PARITY_SPECS below; ``test_shim_matches_db`` asserts DB result == shim result.
SHIM_PARITY_METHODS = {
    "get_traces",
    "get_trace_spans",
    "get_cost_summary",
    "get_alerts",
    "get_tool_calls",
    "get_daily_cost",
    "get_baseline",
}

# Methods the shim implements but that intentionally return a degraded / stub
# result — NOT a faithful mirror of the DB. Documented here so the divergence is
# visible instead of silent. If you make one of these faithful, move it into
# SHIM_PARITY_METHODS with a spec (that is the whole point of this file).
SHIM_KNOWN_GAPS = {
    "get_completed_sessions": (
        "shim synthesizes a single latest-session view from /api/v1/status "
        "(for `tj status`), not the historical completed-sessions query"
    ),
    "get_completed_session_count": "shim stub returns 0 (no /api endpoint)",
    "get_session_cost": "shim stub returns 0.0 (no /api endpoint)",
    "get_recent_spans": "shim stub returns [] (no /api endpoint)",
}

# Protocol methods the read-only shim deliberately does NOT implement (writes,
# and reads whose CLI callers never run in serve/shim mode). If the shim starts
# implementing one, ``test_unimplemented_methods_have_no_silent_shim`` fails so
# the new method cannot land without parity coverage or an explicit gap entry.
SHIM_NOT_IMPLEMENTED = {
    "close_session_by_id",
    "close_sessions_by_instance",
    "count_traces",
    "count_unknown_plan_tier_sessions",
    "delete_spans_before",
    "get_active_session",
    "get_cost_delta_by_group",
    "get_distinct_agent_ids",
    "get_policy_decisions",
    "get_savings_entries",
    "get_session",
    "get_session_active_seconds",
    "get_session_by_conversation",
    "get_window_cost_totals",
    "increment_session_cost",
    "insert_alert",
    "insert_policy_decision",
    "insert_savings_entry",
    "insert_span",
    "insert_validation",
    "update_span_cost",
    "upsert_agent",
    "upsert_baseline",
    "upsert_session",
}

# Lifecycle, not a data method — parity is not meaningful.
SHIM_LIFECYCLE = {"close"}


def _protocol_methods() -> set[str]:
    return {
        name
        for name, val in vars(StorageBackend).items()
        if callable(val) and not name.startswith("_")
    }


# ---------------------------------------------------------------------------
# Comparable projections — reconstructed-from-JSON objects won't be dataclass
# equal, so each method projects to the fields whose parity actually matters.
# ---------------------------------------------------------------------------

def _proj_traces(traces) -> list:
    return sorted(
        (t.trace_id, t.agent_id, t.span_count, round(t.cost_usd or 0.0, 6), t.status_code)
        for t in traces
    )


def _proj_spans(spans) -> list:
    # Includes cache_tokens / cache_write_tokens — the exact fields the shim
    # used to drop on daemon-fetched spans (#_dict_to_span regression).
    return sorted(
        (
            s.span_id,
            s.input_tokens,
            s.output_tokens,
            s.cache_tokens,
            s.cache_write_tokens,
            round(s.cost_usd or 0.0, 6),
            s.model,
            s.provider,
        )
        for s in spans
    )


def _proj_cost(rows) -> list:
    return sorted(
        (
            r.group,
            r.agent_id,
            r.model,
            r.input_tokens,
            r.output_tokens,
            r.cache_tokens,
            r.cache_write_tokens,
            round(r.cost_usd, 6),
        )
        for r in rows
    )


def _proj_alerts(alerts) -> list:
    return sorted(
        (a.alert_id, a.type.value, a.severity.value, a.title, a.acknowledged,
         a.suppressed, a.agent_id)
        for a in alerts
    )


def _proj_tool_calls(calls) -> int:
    return len(calls)


def _proj_daily_cost(value) -> float:
    return round(value, 6)


def _proj_baseline(baseline) -> tuple | None:
    if baseline is None:
        return None
    return (
        baseline.agent_id,
        baseline.sessions_sampled,
        baseline.avg_input_tokens,
        baseline.avg_output_tokens,
        baseline.avg_tool_call_count,
    )


# (invoke, project) per parity method. ``now``/``trace_id`` are bound at seed
# time so get_daily_cost targets the seeded day and get_trace_spans a real trace.
def _parity_specs(now, trace_id):
    return {
        "get_traces": (lambda b: b.get_traces(TraceFilters(limit=100)), _proj_traces),
        "get_trace_spans": (lambda b: b.get_trace_spans(trace_id), _proj_spans),
        "get_cost_summary": (lambda b: b.get_cost_summary(CostFilters()), _proj_cost),
        "get_alerts": (lambda b: b.get_alerts(AlertFilters()), _proj_alerts),
        "get_tool_calls": (lambda b: b.get_tool_calls(AGENT, None, None), _proj_tool_calls),
        "get_daily_cost": (lambda b: b.get_daily_cost(AGENT, now.date()), _proj_daily_cost),
        "get_baseline": (lambda b: b.get_baseline(AGENT), _proj_baseline),
    }


# ---------------------------------------------------------------------------
# Seeding — one fixed dataset built ONCE (fixed ids) and inserted into every
# backend, so cross-backend id comparisons are meaningful.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _Dataset:
    session: object
    spans: list
    alert: Alert
    baseline: DriftBaseline
    trace_id: str


def _build_dataset(now) -> _Dataset:
    session = make_session(
        session_id=SESSION, agent_id=AGENT, status="completed",
        input_tokens=3000, output_tokens=600, total_cost_usd=6.0, tool_call_count=2,
    )

    spans = []
    trace_id = None
    for i in range(3):
        span = make_llm_span(
            agent_id=AGENT, session_id=SESSION,
            input_tokens=1000, output_tokens=200,
            cache_tokens=50, cache_write_tokens=25, cost_usd=2.0,
            start_time=now - timedelta(minutes=5 + i),
        )
        if trace_id is None:
            trace_id = span.trace_id
        else:
            span = dataclasses.replace(span, trace_id=trace_id)
        spans.append(span)

    # A span three days back — get_daily_cost(today) must exclude it (the old
    # shim bug summed cumulatively across days).
    spans.append(make_llm_span(
        agent_id=AGENT, session_id=SESSION, input_tokens=500, output_tokens=100,
        cost_usd=99.0, start_time=now - timedelta(days=3),
    ))
    for _ in range(2):
        spans.append(make_tool_span(agent_id=AGENT, tool_name="grep"))

    alert = Alert(
        alert_id="parity-alert", fired_at=now, type=AlertType("retry_loop"),
        severity=Severity("warning"), title="Retry loop", detail={"n": 3},
        agent_id=AGENT, session_id=SESSION, span_id=None,
        acknowledged=False, suppressed=False,
    )
    baseline = DriftBaseline(
        agent_id=AGENT, sessions_sampled=10, computed_at=now,
        avg_input_tokens=1000.0, stddev_input_tokens=100.0,
        avg_output_tokens=200.0, stddev_output_tokens=20.0,
        avg_session_duration_s=60.0, stddev_session_duration=5.0,
        avg_tool_call_count=2.0, stddev_tool_call_count=0.5,
    )
    return _Dataset(session=session, spans=spans, alert=alert,
                    baseline=baseline, trace_id=trace_id)


def _seed(db: StorageBackend, dataset: _Dataset) -> None:
    """Insert an already-built dataset (fixed ids) into any backend."""
    db.upsert_session(dataset.session)
    for span in dataset.spans:
        db.insert_span(span)
    db.insert_alert(dataset.alert)
    db.upsert_baseline(dataset.baseline)


def _config() -> TjConfig:
    return TjConfig(
        version="1",
        security=SecurityConfig(ingest_secret="parity-secret"),
        api=ApiConfig(auth=ApiAuthConfig(enabled=False)),
    )


@contextmanager
def _live_server(app):
    """Run ``app`` under a real uvicorn server on an ephemeral port so the shim
    is exercised over genuine HTTP (real JSON round-trip — where shim bugs
    live), not an in-process ASGI shortcut the sync ``ApiBackend`` can't use."""
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="error",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("live test server failed to start within 10s")
        time.sleep(0.02)
    try:
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Fixtures — seed once, stand up one live server, share across parity cases.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _parity_env(tmp_path_factory):
    now = utcnow()
    dataset = _build_dataset(now)
    db_path = tmp_path_factory.mktemp("parity") / "parity.duckdb"
    db = DuckDBBackend(StorageConfig(path=str(db_path)))
    _seed(db, dataset)

    config = _config()
    pipeline = IngestPipeline(db=db, config=config)
    app = create_app(config=config, db=db, ingest_pipeline=pipeline)

    with _live_server(app) as base_url:
        shim = ApiBackend(base_url)
        try:
            yield {"db": db, "shim": shim, "now": now, "trace_id": dataset.trace_id}
        finally:
            shim.close()
    db.close()


# ---------------------------------------------------------------------------
# The parity guard.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", sorted(SHIM_PARITY_METHODS))
def test_shim_matches_db(_parity_env, method):
    """Every faithfully-mirrored read method returns the same result through the
    serve-mode HTTP shim as through the direct DuckDB backend."""
    invoke, project = _parity_specs(_parity_env["now"], _parity_env["trace_id"])[method]
    db_result = project(invoke(_parity_env["db"]))
    shim_result = project(invoke(_parity_env["shim"]))
    assert db_result == shim_result, (
        f"{method}: serve-shim diverges from DB backend\n"
        f"  db  = {db_result}\n"
        f"  shim= {shim_result}"
    )


# ---------------------------------------------------------------------------
# DuckDB <-> InMemory contract parity (the other two shipped backends).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", sorted(SHIM_PARITY_METHODS))
def test_duckdb_and_in_memory_agree(tmp_path, method):
    """The DuckDB file backend and the InMemoryBackend used across unit tests
    must return identical results for the same seeded data."""
    now = utcnow()
    dataset = _build_dataset(now)
    file_db = DuckDBBackend(StorageConfig(path=str(tmp_path / "contract.duckdb")))
    mem_db = InMemoryBackend()
    _seed(file_db, dataset)
    _seed(mem_db, dataset)

    invoke, project = _parity_specs(now, dataset.trace_id)[method]
    try:
        assert project(invoke(file_db)) == project(invoke(mem_db)), (
            f"{method}: DuckDB file backend diverges from InMemoryBackend"
        )
    finally:
        file_db.close()
        mem_db.close()


# ---------------------------------------------------------------------------
# Enforcement — a new/changed method cannot dodge parity classification.
# ---------------------------------------------------------------------------

def test_every_protocol_method_is_classified():
    """Adding a StorageBackend method fails here until it is classified as
    parity-covered, a known gap, deliberately-unimplemented, or lifecycle —
    which for a read method means wiring it into the parity set above."""
    classified = (
        SHIM_PARITY_METHODS
        | set(SHIM_KNOWN_GAPS)
        | SHIM_NOT_IMPLEMENTED
        | SHIM_LIFECYCLE
    )
    protocol = _protocol_methods()
    missing = protocol - classified
    stale = classified - protocol
    assert not missing, (
        f"unclassified StorageBackend method(s): {sorted(missing)} — add a parity "
        f"spec (SHIM_PARITY_METHODS + _parity_specs) or classify as a known gap / "
        f"unimplemented / lifecycle in tests/integration/test_storage_backend_parity.py"
    )
    assert not stale, f"classification names non-existent method(s): {sorted(stale)}"


def test_parity_methods_have_specs():
    """Every parity-covered method has an invoke/project spec."""
    specs = set(_parity_specs(utcnow(), "trace-id"))
    assert specs == SHIM_PARITY_METHODS, (
        f"spec set {sorted(specs)} != SHIM_PARITY_METHODS {sorted(SHIM_PARITY_METHODS)}"
    )


def test_parity_and_gap_methods_are_actually_implemented():
    """Parity + known-gap methods must exist on ApiBackend; if one is removed,
    reclassify it rather than leave a dangling spec."""
    for method in SHIM_PARITY_METHODS | set(SHIM_KNOWN_GAPS):
        assert method in vars(ApiBackend), (
            f"{method} is classified as shim-implemented but ApiBackend no longer "
            f"defines it — update the classification"
        )


def test_unimplemented_methods_have_no_silent_shim():
    """If the shim starts implementing a method we listed as unimplemented, this
    fails so the new method must gain parity coverage (SHIM_PARITY_METHODS) or an
    explicit known-gap entry before it can land."""
    leaked = {m for m in SHIM_NOT_IMPLEMENTED if m in vars(ApiBackend)}
    assert not leaked, (
        f"ApiBackend now implements {sorted(leaked)} but they are listed as "
        f"unimplemented — move them to SHIM_PARITY_METHODS (with a spec) or "
        f"SHIM_KNOWN_GAPS in tests/integration/test_storage_backend_parity.py"
    )
