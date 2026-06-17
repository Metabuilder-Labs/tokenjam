"""Per-analyzer estimated-recoverable-USD tests (issue #111)."""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import analyze_model_downgrade, build_report
from tokenjam.core.optimize.analyzers.cache_efficacy import (
    CacheEfficacyRow,
    estimate_cache_recoverable,
)
from tokenjam.core.optimize.analyzers.prompt_bloat import estimate_trim_recoverable
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session, make_tool_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _window():
    return utcnow() - timedelta(days=30), utcnow() + timedelta(hours=1)


# --------------------------------------------------------------------------- #
# downsize — aliases the existing monthly projection
# --------------------------------------------------------------------------- #
def _insert_small_opus_session(db, session_id="a"):
    start = utcnow() - timedelta(days=2)
    llm = make_llm_span(
        agent_id="claude-code-x", model="claude-opus-4-7", provider="anthropic",
        input_tokens=1000, output_tokens=200, cost_usd=0.030,
        session_id=session_id, start_time=start,
    )
    db.insert_span(llm)
    for _ in range(2):
        tool = make_tool_span(agent_id="claude-code-x", tool_name="Read",
                              trace_id=llm.trace_id)
        tool.session_id = session_id
        tool.start_time = start
        db.insert_span(tool)


def test_downsize_recoverable_aliases_monthly_savings(db):
    _insert_small_opus_session(db)
    since, until = _window()
    finding = analyze_model_downgrade(db.conn, since, until, None, 30.0)
    assert finding is not None
    assert finding.estimated_recoverable_usd == finding.monthly_savings_usd
    assert finding.estimated_recoverable_usd > 0
    assert finding.estimated_recoverable_tokens == finding.candidate_tokens
    assert finding.estimate_basis
    assert finding.estimate_confidence == "heuristic"


# --------------------------------------------------------------------------- #
# cache — efficacy-gap heuristic (pure helper)
# --------------------------------------------------------------------------- #
def test_cache_recoverable_matches_hand_calculation():
    # opus-4-7: input 5.00/MTok, cache_read 0.50/MTok → delta 4.50.
    # 1M input, 0 cache → efficacy 0 → gap 0.80 → 800K recoverable tokens.
    # usd = 800_000/1e6 * 4.50 = 3.60
    row = CacheEfficacyRow(
        provider="anthropic", model="claude-opus-4-7",
        input_tokens=1_000_000, cache_tokens=0, efficacy=0.0,
        support="full", flagged=True,
    )
    usd, tokens = estimate_cache_recoverable([row])
    assert tokens == 800_000
    assert usd == pytest.approx(3.60, abs=0.01)


def test_cache_recoverable_none_when_no_caching_dimension():
    # An unknown model falls back to default rates with cache_read = 0 → no
    # caching dimension → no estimate.
    row = CacheEfficacyRow(
        provider="madeup", model="no-such-model",
        input_tokens=1_000_000, cache_tokens=0, efficacy=0.0,
        support="unsupported", flagged=False,
    )
    usd, tokens = estimate_cache_recoverable([row])
    assert usd is None
    assert tokens is None


def test_cache_recoverable_none_when_already_at_ceiling():
    row = CacheEfficacyRow(
        provider="anthropic", model="claude-opus-4-7",
        input_tokens=1_000_000, cache_tokens=4_000_000, efficacy=0.95,
        support="full", flagged=False,
    )
    usd, tokens = estimate_cache_recoverable([row])
    assert usd is None and tokens is None


# --------------------------------------------------------------------------- #
# script — sum of clustered session cost
# --------------------------------------------------------------------------- #
def test_script_recoverable_sums_cluster_session_cost(db):
    start = utcnow() - timedelta(days=2)
    # 20 identical single-Read sessions clear MIN_CLUSTER_INSTANCES.
    for i in range(20):
        sid = f"sess-{i}"
        db.upsert_session(make_session(
            agent_id="agent-x", session_id=sid,
            input_tokens=500, output_tokens=100, total_cost_usd=0.10,
        ))
        tool = make_tool_span(agent_id="agent-x", tool_name="Read")
        tool.session_id = sid
        tool.start_time = start
        db.insert_span(tool)

    cfg = TjConfig(version="1")  # capture.tool_inputs default off → name-only clusters
    since, until = _window()
    report = build_report(db=db, config=cfg, since=since, until=until,
                          findings=["script"])
    finding = report.findings["script"]
    assert finding.clusters, "expected one cluster of 20 identical sessions"
    # 20 sessions × $0.10 = $2.00
    assert finding.estimated_recoverable_usd == pytest.approx(2.00, abs=0.001)
    # 20 sessions × (500 + 100) tokens = 12_000
    assert finding.estimated_recoverable_tokens == 12_000
    assert finding.estimate_basis


def test_script_recoverable_none_when_no_clusters(db):
    # A single tool span — no cluster above threshold.
    tool = make_tool_span(agent_id="agent-x", tool_name="Read")
    tool.session_id = "solo"
    tool.start_time = utcnow() - timedelta(days=1)
    db.insert_span(tool)
    db.upsert_session(make_session(agent_id="agent-x", session_id="solo",
                                   total_cost_usd=0.10))
    cfg = TjConfig(version="1")
    since, until = _window()
    report = build_report(db=db, config=cfg, since=since, until=until,
                          findings=["script"])
    finding = report.findings["script"]
    assert not finding.clusters
    assert finding.estimated_recoverable_usd is None
    assert finding.estimated_recoverable_tokens is None


# --------------------------------------------------------------------------- #
# trim — low-significance tokens × input rate (helper) + not-ready state
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# serialization — the fields surface in the /api/v1/optimize response shape
# --------------------------------------------------------------------------- #
def test_recoverable_fields_survive_report_to_dict_round_trip(db):
    from tokenjam.core.optimize import report_from_dict, report_to_dict

    _insert_small_opus_session(db)
    cfg = TjConfig(version="1")
    since, until = _window()
    report = build_report(db=db, config=cfg, since=since, until=until,
                          findings=["downsize", "cache", "script", "trim"])
    payload = report_to_dict(report)

    # downsize lives in the typed slot
    assert "estimated_recoverable_usd" in payload["downgrade"]
    assert "estimate_basis" in payload["downgrade"]
    # wave-2 findings carry the fields too
    for name in ("cache", "script", "trim"):
        if name in payload.get("findings", {}):
            assert "estimated_recoverable_usd" in payload["findings"][name]
            assert "estimate_confidence" in payload["findings"][name]

    # and they round-trip back through report_from_dict
    rebuilt = report_from_dict(payload)
    assert (rebuilt.downgrade.estimated_recoverable_usd
            == report.downgrade.estimated_recoverable_usd)


def test_trim_recoverable_helper_prices_tokens():
    # 100K low-significance tokens at $5/MTok = $0.50
    assert estimate_trim_recoverable(100_000, 5.00) == pytest.approx(0.50, abs=1e-6)


def test_trim_recoverable_none_when_capture_off(db):
    _insert_small_opus_session(db)
    cfg = TjConfig(version="1", capture=CaptureConfig(prompts=False))
    since, until = _window()
    report = build_report(db=db, config=cfg, since=since, until=until,
                          findings=["trim"])
    finding = report.findings["trim"]
    assert finding.enabled is False
    assert finding.estimated_recoverable_usd is None
    assert finding.estimate_basis == "" or finding.estimate_basis is not None
