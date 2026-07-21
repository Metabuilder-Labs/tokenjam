"""Unit tests for the reuse analyzer (issue #115)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import CaptureConfig, OptimizeConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers.plan_reuse import (
    MIN_PLANNING_TOKENS,
    MIN_REPETITIONS,
    _SpanRow,
    _cluster_key,
    _identify_planning_call,
    _prompt_prefix_hash,
    _strip_variables,
    _tool_signature,
)
from tokenjam.otel.semconv import GenAIAttributes
from tests.factories import make_llm_span, make_session, make_tool_span

UTC = timezone.utc


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _config(prompts: bool = False) -> TjConfig:
    return TjConfig(version="1", capture=CaptureConfig(prompts=prompts))


# --------------------------------------------------------------------------
# Pure-function tests
# --------------------------------------------------------------------------

def _row(*, model=None, tool_name=None, attributes=None) -> _SpanRow:
    return _SpanRow(
        session_id="s", start_time=0, model=model, tool_name=tool_name,
        input_tokens=300, output_tokens=100, cache_tokens=0,
        cache_write_tokens=0, cost_usd=0.2, attributes=attributes,
    )


def test_strip_variables_normalizes_versions_and_dates():
    a = _strip_variables("release v0.3.4 on 2026-06-15 in /home/me/repo")
    b = _strip_variables("release v0.3.5 on 2026-06-17 in /home/you/work")
    assert a == b


def test_identify_planning_call_llm_before_first_tool():
    rows = [
        _row(model="claude"),        # 0 — the plan
        _row(tool_name="read"),      # 1 — first tool
        _row(model="claude"),        # 2 — later LLM, not the plan
        _row(tool_name="edit"),
    ]
    assert _identify_planning_call(rows) is rows[0]


def test_identify_planning_call_no_tool_calls_uses_first_llm():
    rows = [_row(model="claude"), _row(model="claude")]
    assert _identify_planning_call(rows) is rows[0]


def test_identify_planning_call_no_llm_returns_none():
    rows = [_row(tool_name="read"), _row(tool_name="edit")]
    assert _identify_planning_call(rows) is None


def test_identify_planning_call_multiple_llms_before_first_tool():
    rows = [
        _row(model="claude"),        # 0
        _row(model="claude"),        # 1 — most recent before first tool
        _row(tool_name="read"),      # 2
    ]
    assert _identify_planning_call(rows) is rows[1]


def test_tool_signature_follows_plan():
    rows = [
        _row(model="claude"),
        _row(tool_name="read"),
        _row(tool_name="edit"),
        _row(model="claude"),        # interleaved LLM is not a tool
        _row(tool_name="run"),
    ]
    assert _tool_signature(rows, 0) == ("read", "edit", "run")


def test_prompt_prefix_hash_invariant_to_numbers_and_dates():
    r1 = _row(model="c", attributes={
        GenAIAttributes.PROMPT_CONTENT: "cut release v1.2.3 dated 2026-06-15"})
    r2 = _row(model="c", attributes={
        GenAIAttributes.PROMPT_CONTENT: "cut release v1.2.4 dated 2026-06-18"})
    assert _prompt_prefix_hash(r1) == _prompt_prefix_hash(r2)


def test_prompt_prefix_hash_none_when_no_content():
    assert _prompt_prefix_hash(_row(model="c", attributes={})) is None


# --------------------------------------------------------------------------
# Integration tests via build_report
# --------------------------------------------------------------------------

def _seed_cluster(db, *, count, tool_names, base_session="c",
                  cost=0.20, input_tokens=1000, output_tokens=200,
                  prompt=None, start_base=None):
    """Seed `count` sessions, each = one planning LLM span + a tool sequence."""
    base = start_base or datetime(2026, 5, 10, tzinfo=UTC)
    for i in range(count):
        sid = f"{base_session}-{i}"
        db.upsert_session(make_session(session_id=sid, plan_tier="api"))
        t0 = base + timedelta(minutes=i)
        extra = {GenAIAttributes.PROMPT_CONTENT: prompt} if prompt is not None else None
        plan = make_llm_span(
            session_id=sid, start_time=t0, cost_usd=cost,
            input_tokens=input_tokens, output_tokens=output_tokens,
            extra_attributes=extra,
        )
        db.insert_span(plan)
        for j, tn in enumerate(tool_names):
            ts = make_tool_span(tool_name=tn)
            ts.session_id = sid
            ts.start_time = t0 + timedelta(seconds=j + 1)
            db.insert_span(ts)


def _run(db, config):
    since = datetime(2026, 5, 1, tzinfo=UTC)
    until = datetime(2026, 5, 30, tzinfo=UTC)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["reuse"])
    return report.findings["reuse"]


def test_cluster_surfaced_and_savings_math(db):
    """5 sessions × $0.20 planning → cache-reuse $0.80, script $1.00."""
    _seed_cluster(db, count=5, tool_names=["read", "edit", "run"], cost=0.20)
    finding = _run(db, _config(prompts=False))

    assert len(finding.clusters) == 1
    c = finding.clusters[0]
    assert c.repetitions == 5
    assert c.tool_signature == ("read", "edit", "run")
    assert c.cache_reuse_recoverable_usd == pytest.approx(0.80)
    assert c.script_replacement_recoverable_usd == pytest.approx(1.00)
    # Aggregate uses the conservative cache-reuse number.
    assert finding.estimated_recoverable_usd == pytest.approx(0.80)
    assert finding.estimated_recoverable_tokens == c.cache_reuse_recoverable_tokens


def test_below_repetition_threshold_dropped(db):
    _seed_cluster(db, count=MIN_REPETITIONS - 1, tool_names=["read"])
    finding = _run(db, _config(prompts=False))
    assert finding.clusters == []
    assert finding.estimated_recoverable_usd is None
    assert finding.min_repetitions == MIN_REPETITIONS


def test_config_lowers_repetition_bar_surfaces_previously_hidden_cluster(db):
    """The exact data from test_below_repetition_threshold_dropped clusters
    nothing at the default bar; lowering [optimize] min_reuse_repetitions to
    MIN_REPETITIONS - 1 surfaces it."""
    _seed_cluster(db, count=MIN_REPETITIONS - 1, tool_names=["read"])
    since = datetime(2026, 5, 1, tzinfo=UTC)
    until = datetime(2026, 5, 30, tzinfo=UTC)

    default_report = build_report(db=db, config=_config(prompts=False),
                                  since=since, until=until, findings=["reuse"])
    assert default_report.findings["reuse"].clusters == []

    lowered_config = TjConfig(
        version="1", capture=CaptureConfig(prompts=False),
        optimize=OptimizeConfig(min_reuse_repetitions=MIN_REPETITIONS - 1),
    )
    lowered_report = build_report(db=db, config=lowered_config,
                                 since=since, until=until, findings=["reuse"])
    lowered_finding = lowered_report.findings["reuse"]
    assert len(lowered_finding.clusters) == 1
    assert lowered_finding.clusters[0].repetitions == MIN_REPETITIONS - 1
    assert lowered_finding.min_repetitions == MIN_REPETITIONS - 1


def test_below_token_threshold_dropped(db):
    # 50 + 50 = 100 planning tokens, under MIN_PLANNING_TOKENS.
    assert MIN_PLANNING_TOKENS > 100
    _seed_cluster(db, count=5, tool_names=["read"],
                  input_tokens=50, output_tokens=50)
    finding = _run(db, _config(prompts=False))
    assert finding.clusters == []


def test_session_with_no_tool_calls_still_clusters(db):
    """Empty tool signature is valid — these sessions cluster together."""
    _seed_cluster(db, count=4, tool_names=[])
    finding = _run(db, _config(prompts=False))
    assert len(finding.clusters) == 1
    assert finding.clusters[0].tool_signature == ()
    assert finding.clusters[0].repetitions == 4


def test_tool_only_sessions_are_skipped(db):
    """Sessions with no LLM span have no planner and are omitted."""
    base = datetime(2026, 5, 10, tzinfo=UTC)
    for i in range(4):
        sid = f"tool-only-{i}"
        db.upsert_session(make_session(session_id=sid))
        ts = make_tool_span(tool_name="read")
        ts.session_id = sid
        ts.start_time = base + timedelta(minutes=i)
        db.insert_span(ts)
    finding = _run(db, _config(prompts=False))
    assert finding.clusters == []


def test_mode1_capture_off_has_no_prefix_hash(db):
    _seed_cluster(db, count=4, tool_names=["read"], prompt="ship the release")
    finding = _run(db, _config(prompts=False))
    assert finding.capture_mode == "tool_sequence_only"
    assert finding.hint  # non-empty nudge to enable capture
    assert len(finding.clusters) == 1
    assert finding.clusters[0].prompt_prefix_hash is None


def test_mode2_prompt_prefix_splits_same_tool_signature(db):
    """
    With capture.prompts on, two groups sharing a tool signature but with
    unrelated prompts split into separate clusters (precision gain).
    """
    _seed_cluster(db, count=3, tool_names=["read"], base_session="deploy",
                  prompt="deploy the staging cluster now",
                  start_base=datetime(2026, 5, 10, tzinfo=UTC))
    _seed_cluster(db, count=3, tool_names=["read"], base_session="report",
                  prompt="summarize the quarterly revenue report",
                  start_base=datetime(2026, 5, 15, tzinfo=UTC))
    finding = _run(db, _config(prompts=True))
    assert finding.capture_mode == "with_prompt_prefix"
    assert finding.hint == ""
    assert len(finding.clusters) == 2
    assert all(c.prompt_prefix_hash is not None for c in finding.clusters)


def test_empty_window_returns_clean_finding(db):
    finding = _run(db, _config(prompts=False))
    assert finding.clusters == []
    assert finding.estimated_recoverable_usd is None
    assert finding.estimated_recoverable_tokens is None


def test_estimate_basis_mentions_review(db):
    finding = _run(db, _config(prompts=False))
    assert finding.estimate_basis
    assert "review" in finding.estimate_basis.lower()


def test_cluster_id_is_deterministic_across_runs(db):
    """
    cluster_id must be stable across re-runs over the same data — PR 2's
    Markdown sidecar filenames key off it for idempotent overwrites.
    """
    _seed_cluster(db, count=5, tool_names=["read", "edit", "run"], cost=0.20)
    first = _run(db, _config(prompts=False)).clusters[0].cluster_id
    second = _run(db, _config(prompts=False)).clusters[0].cluster_id
    assert first == second
    # And it equals the pure key derived from the signature (Mode 1: no prefix).
    assert first == _cluster_key(("read", "edit", "run"), None)
