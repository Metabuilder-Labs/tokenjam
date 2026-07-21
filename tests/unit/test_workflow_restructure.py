"""Unit tests for the script analyzer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import CaptureConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers.workflow_restructure import (
    MIN_CLUSTER_INSTANCES,
    _arg_signature,
    _classify_arg,
)
from tokenjam.otel.semconv import GenAIAttributes
from tests.factories import make_session, make_tool_span


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _config(tool_inputs: bool) -> TjConfig:
    return TjConfig(version="1", capture=CaptureConfig(tool_inputs=tool_inputs))


# -- Pure-function tests --

def test_classify_arg_file_path():
    assert _classify_arg("/path/to/file.py") == "file_path"
    assert _classify_arg("~/.bashrc") == "file_path"
    assert _classify_arg("./script.sh") == "file_path"


def test_classify_arg_command_string():
    assert _classify_arg("git status") == "command_string"
    assert _classify_arg("npm install foo") == "command_string"
    assert _classify_arg("pytest tests/") == "command_string"


def test_classify_arg_primitives():
    assert _classify_arg(42) == "number"
    assert _classify_arg(3.14) == "number"
    assert _classify_arg(True) == "boolean"
    assert _classify_arg([1, 2, 3]) == "array"
    assert _classify_arg({"a": 1}) == "json_object"


def test_classify_arg_falls_back_to_string():
    assert _classify_arg("some random text") == "string"
    assert _classify_arg(None) == "string"


def test_arg_signature_is_sorted_by_key():
    """Keys are sorted so dict-ordering doesn't change the signature."""
    sig_a = _arg_signature({"path": "/foo", "verbose": True})
    sig_b = _arg_signature({"verbose": True, "path": "/foo"})
    assert sig_a == sig_b
    assert sig_a == ("file_path", "boolean")


def test_arg_signature_empty_input():
    assert _arg_signature(None) == ()
    assert _arg_signature({}) == ()


# -- Integration tests via build_report --

def _seed_deterministic_cluster(db, *, count: int,
                                  tools: list[tuple[str, dict]],
                                  base_session: str = "det"):
    """
    Seed N sessions, each with the same tool-call sequence. `tools` is a list
    of (tool_name, tool_input_dict). All sessions share the same structural
    signature; argument values can vary across sessions if the test wants.
    """
    base = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(count):
        sid = f"{base_session}-{i}"
        sess = make_session(session_id=sid, plan_tier="api", duration_seconds=30.0)
        db.upsert_session(sess)
        for j, (tool, args) in enumerate(tools):
            span = make_tool_span(tool_name=tool, duration_ms=80.0)
            # Override identifiers and inject the tool_input attribute the
            # analyzer reads.
            span.session_id = sid
            span.start_time = base + timedelta(minutes=i) + timedelta(seconds=j)
            span.attributes = {GenAIAttributes.TOOL_INPUT: args}
            db.insert_span(span)


def test_flags_cluster_at_or_above_threshold(db):
    """A deterministic cluster with ≥MIN_CLUSTER_INSTANCES sessions is surfaced."""
    _seed_deterministic_cluster(
        db, count=MIN_CLUSTER_INSTANCES,
        tools=[
            ("bash", {"command": "git pull origin staging"}),
            ("bash", {"command": "npm install"}),
            ("bash", {"command": "npm run build"}),
        ],
    )
    config = _config(tool_inputs=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["script"])
    finding = report.findings["script"]
    assert finding.degraded is False
    assert len(finding.clusters) == 1
    c = finding.clusters[0]
    assert c.instances == MIN_CLUSTER_INSTANCES
    # Signature surfaced as tool-name + arg-shape labels
    assert c.signature == [
        {"tool": "bash", "args": ["command_string"]},
        {"tool": "bash", "args": ["command_string"]},
        {"tool": "bash", "args": ["command_string"]},
    ]


def test_cluster_carries_avg_tokens_for_per_item_framing(db):
    """Each cluster carries the per-instance mean of input+output tokens so the
    UI can render the cell as TOKENS for subscription/local users (#260)."""
    base = datetime(2026, 5, 10, tzinfo=timezone.utc)
    # 20 sessions, each 1000 in + 200 out → avg_tokens should be 1200.
    for i in range(MIN_CLUSTER_INSTANCES):
        sid = f"tok-{i}"
        sess = make_session(session_id=sid, plan_tier="api", duration_seconds=30.0,
                            input_tokens=1000, output_tokens=200)
        db.upsert_session(sess)
        span = make_tool_span(tool_name="bash")
        span.session_id = sid
        span.start_time = base + timedelta(minutes=i)
        span.attributes = {GenAIAttributes.TOOL_INPUT: {"command": "git pull"}}
        db.insert_span(span)

    config = _config(tool_inputs=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["script"])
    c = report.findings["script"].clusters[0]
    assert c.avg_tokens == 1200


def test_below_threshold_not_flagged(db):
    """Fewer than MIN_CLUSTER_INSTANCES sessions: no recommendation."""
    _seed_deterministic_cluster(
        db, count=MIN_CLUSTER_INSTANCES - 1,
        tools=[("bash", {"command": "echo hi"})],
    )
    config = _config(tool_inputs=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["script"])
    assert report.findings["script"].clusters == []


def test_value_variation_doesnt_split_cluster(db):
    """
    The signature is structural shape, not values. Sessions with different
    file paths but the same tool sequence still cluster together.
    """
    base = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(MIN_CLUSTER_INSTANCES):
        sid = f"var-{i}"
        sess = make_session(session_id=sid, plan_tier="api", duration_seconds=30.0)
        db.upsert_session(sess)
        # Each session opens a *different* file path — different values,
        # same arg_shape (file_path) — should land in the same cluster.
        span = make_tool_span(tool_name="read_file")
        span.session_id = sid
        span.start_time = base + timedelta(minutes=i)
        span.attributes = {GenAIAttributes.TOOL_INPUT: {"path": f"/tmp/file-{i}.txt"}}
        db.insert_span(span)

    config = _config(tool_inputs=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["script"])
    finding = report.findings["script"]
    assert len(finding.clusters) == 1
    assert finding.clusters[0].instances == MIN_CLUSTER_INSTANCES


def test_different_signatures_form_separate_clusters(db):
    """Sessions with different tool sequences cluster separately."""
    _seed_deterministic_cluster(
        db, count=MIN_CLUSTER_INSTANCES,
        tools=[("bash", {"command": "git pull"})],
        base_session="a",
    )
    _seed_deterministic_cluster(
        db, count=MIN_CLUSTER_INSTANCES,
        tools=[("read_file", {"path": "/x"})],
        base_session="b",
    )
    config = _config(tool_inputs=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["script"])
    assert len(report.findings["script"].clusters) == 2


def test_degraded_mode_when_tool_inputs_not_captured(db):
    """
    Without capture.tool_inputs, the analyzer clusters by tool names only
    and marks the finding as degraded.
    """
    # Without arg shape, two clusters with different values but same tool
    # name collapse into one. We still surface the cluster — just flagged
    # as degraded so the user knows the signal is weaker.
    _seed_deterministic_cluster(
        db, count=MIN_CLUSTER_INSTANCES,
        tools=[("bash", {"command": "git pull"})],
    )
    config = _config(tool_inputs=False)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["script"])
    finding = report.findings["script"]
    assert finding.degraded is True
    assert len(finding.clusters) == 1
    # Without arg shape captured, signature carries only tool names.
    assert finding.clusters[0].signature == [{"tool": "bash"}]


def test_no_tool_spans_in_window(db):
    """Empty window returns a finding with no clusters but the right metadata."""
    config = _config(tool_inputs=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["script"])
    finding = report.findings["script"]
    assert finding.clusters == []
    assert finding.sessions_examined == 0


def test_cache_write_tokens_included_in_recoverable_total(db):
    """Regression: cache_write_tokens must be included in estimated_recoverable_tokens.

    This verifies the fix for the bug where cache_write_tokens was omitted
    from the per-cluster token sum, causing cache-write-heavy workloads to
    underreport their recoverable savings.
    """
    base = datetime(2026, 5, 10, tzinfo=timezone.utc)
    # 20 sessions, each with cache writes
    # Session tokens: input=1000, output=200, cache_read=500, cache_write=300
    # Expected per-session total: 2000 tokens
    for i in range(MIN_CLUSTER_INSTANCES):
        sid = f"cache-write-{i}"
        sess = make_session(
            session_id=sid,
            plan_tier="api",
            duration_seconds=30.0,
            input_tokens=1000,
            output_tokens=200,
            cache_tokens=500,  # cache_read
            cache_write_tokens=300,  # cache_write
        )
        db.upsert_session(sess)
        span = make_tool_span(tool_name="bash")
        span.session_id = sid
        span.start_time = base + timedelta(minutes=i)
        span.attributes = {GenAIAttributes.TOOL_INPUT: {"command": "git pull"}}
        db.insert_span(span)

    config = _config(tool_inputs=True)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 30, tzinfo=timezone.utc)
    report = build_report(db=db, config=config, since=since, until=until,
                          findings=["script"])
    finding = report.findings["script"]

    assert len(finding.clusters) == 1
    c = finding.clusters[0]
    assert c.instances == MIN_CLUSTER_INSTANCES

    # Total recoverable tokens = (1000 + 200 + 500 + 300) * 20 = 2000 * 20 = 40000
    # Before the fix, this would be (1000 + 200 + 500) * 20 = 1700 * 20 = 34000
    assert finding.estimated_recoverable_tokens == 40000
    # avg_tokens should be input+output only (per-instance UI framing),
    # not affected by the bug fix
    assert c.avg_tokens == 1200
