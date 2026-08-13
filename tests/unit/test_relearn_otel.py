"""The self-improve loop's OTel lane: relearns mined from stored spans.

Covers the second extraction path (failing spans -> FailureEpisode), the
no-double-count rule against coding agents, that span failures join the SAME
clustering pass as transcript failures, the advise-only seam (workspace-less
agents get no apply path), and the eval-case artifact.

All spans go through tests/factories (Critical Rule 8). Nothing here touches the
real ~/.tj or ~/.claude: the backend is in-memory and no transcript root is read.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.db import InMemoryBackend
from tokenjam.core import distill as distill_mod
from tokenjam.core.optimize.analyzers.relearn import (
    FailureEpisode,
    analyze_relearns,
    build_proposals,
    cluster_failures,
)
from tokenjam.core.optimize.relearn_otel import (
    extract_span_failures,
    non_coding_agent_ids,
    to_eval_case,
)
from tests.factories import make_tool_span

BASE = datetime(2026, 5, 10, tzinfo=timezone.utc)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


def _seed_failure(db, *, agent_id, session_id, message, tool="http_call", i=0,
                  status="error"):
    span = make_tool_span(
        agent_id=agent_id, tool_name=tool, status=status,
        session_id=session_id, start_time=BASE + timedelta(minutes=i),
    )
    span.status_message = message
    db.insert_span(span)
    return span


# -- Extraction --------------------------------------------------------------

def test_extracts_failing_spans_as_failure_episodes(db):
    _seed_failure(db, agent_id="billing-svc", session_id="s1",
                  message="ConnectionResetError: peer closed")

    failures = extract_span_failures(db.conn)

    assert len(failures) == 1
    f = failures[0]
    assert isinstance(f, FailureEpisode)
    assert f.session_id == "s1"
    assert f.repo == "billing-svc"
    assert f.error_text == "ConnectionResetError: peer closed"
    assert f.tool_name == "http_call"


def test_ignores_successful_spans(db):
    _seed_failure(db, agent_id="billing-svc", session_id="s1",
                  message="fine", status="ok")

    assert extract_span_failures(db.conn) == []


def test_skips_coding_agents_so_transcripts_are_not_double_counted(db):
    # Claude Code / Codex agents already come in through the transcript path.
    _seed_failure(db, agent_id="claude-code-myrepo", session_id="s1",
                  message="no such file or directory")
    _seed_failure(db, agent_id="codex-myrepo", session_id="s2",
                  message="no such file or directory")
    _seed_failure(db, agent_id="billing-svc", session_id="s3",
                  message="no such file or directory")

    failures = extract_span_failures(db.conn)

    assert [f.repo for f in failures] == ["billing-svc"]


def test_falls_back_to_span_name_when_no_status_message(db):
    """A span with no ``status_message`` still becomes a FailureEpisode (the
    failure is real, and the money it cost is real — see
    BELOW_THRESHOLD_BASIS), but the fallback text is the span's own NAME, not
    a diagnosis. It must be marked as such: `error_text_is_name_fallback`
    downstream keeps it out of the distill pass and out of any shared
    signature with unrelated failures (see `_failure_signature` /
    `_evidence_too_thin_for_distill` in `analyzers/relearn.py`)."""
    span = make_tool_span(
        agent_id="billing-svc", tool_name="", status="error",
        session_id="s1", start_time=BASE, name="gen_ai.tool.call",
    )
    span.status_message = None
    db.insert_span(span)

    failures = extract_span_failures(db.conn)

    assert len(failures) == 1
    assert failures[0].error_text == "gen_ai.tool.call"
    assert failures[0].error_text_is_name_fallback is True


def test_real_status_message_is_not_flagged_as_name_fallback(db):
    _seed_failure(db, agent_id="billing-svc", session_id="s1",
                  message="ConnectionResetError: peer closed")

    failures = extract_span_failures(db.conn)

    assert len(failures) == 1
    assert failures[0].error_text_is_name_fallback is False


def test_since_filters_older_spans(db):
    _seed_failure(db, agent_id="billing-svc", session_id="old",
                  message="boom", i=0)
    _seed_failure(db, agent_id="billing-svc", session_id="new",
                  message="boom", i=60)

    failures = extract_span_failures(db.conn, since=BASE + timedelta(minutes=30))

    assert [f.session_id for f in failures] == ["new"]


def test_never_raises_without_a_connection():
    assert extract_span_failures(None) == []
    assert non_coding_agent_ids(None) == set()


def test_non_coding_agent_ids_excludes_coding_agents(db):
    _seed_failure(db, agent_id="claude-code-repo", session_id="s1", message="x")
    _seed_failure(db, agent_id="billing-svc", session_id="s2", message="y")

    assert non_coding_agent_ids(db.conn) == {"billing-svc"}


# -- Clustering: span failures use the same pipeline -------------------------

def test_span_failures_cluster_by_normalized_signature(db):
    """Varying paths and standalone numbers are normalized away, so the same
    error SHAPE across sessions collapses into one cluster."""
    for i, sid in enumerate(("s1", "s2", "s3")):
        _seed_failure(db, agent_id="billing-svc", session_id=sid, i=i,
                      message=f"Timeout after {900 + i} ms calling /v1/charge/{i}")

    clusters = cluster_failures(extract_span_failures(db.conn))

    assert len(clusters) == 1
    assert len(next(iter(clusters.values())).session_ids) == 3


def test_recurring_span_failures_become_a_proposal(db):
    for i, sid in enumerate(("s1", "s2", "s3")):
        _seed_failure(db, agent_id="billing-svc", session_id=sid, i=i,
                      message="ConnectionResetError: peer closed")

    finding = analyze_relearns(
        [], extra_failures=extract_span_failures(db.conn),
        advise_only_repos=non_coding_agent_ids(db.conn), distill_enabled=False,
    )

    assert len(finding.clusters) == 1
    assert finding.clusters[0].sessions == 3
    # Span-sourced sessions count as scanned exposure.
    assert finding.sessions_scanned == 3


def test_text_less_span_failures_never_merge_across_sessions(db):
    """Old behavior: every span with no status_message on the same tool fell
    back to the SAME span-name string and clustered as one shared signature
    across every session, however unrelated the underlying calls actually
    were. Now a name-fallback failure is scoped to its own session — see
    `_failure_signature` in analyzers/relearn.py."""
    for i, sid in enumerate(("s1", "s2", "s3", "s4")):
        span = make_tool_span(
            agent_id="billing-svc", tool_name="Bash", status="error",
            session_id=sid, start_time=BASE + timedelta(minutes=i),
            name="gen_ai.tool.call",
        )
        span.status_message = None
        db.insert_span(span)

    failures = extract_span_failures(db.conn)
    assert len(failures) == 4
    assert all(f.error_text_is_name_fallback for f in failures)

    clusters = cluster_failures(failures)

    # Four unrelated no-error-text failures must never collapse into one
    # shared signature — each keeps its own, one per session.
    assert len(clusters) == 4
    assert all(len(c.session_ids) == 1 for c in clusters.values())


def test_text_less_span_failures_never_become_a_recurring_proposal(db):
    for i, sid in enumerate(("s1", "s2", "s3", "s4", "s5")):
        span = make_tool_span(
            agent_id="billing-svc", tool_name="Bash", status="error",
            session_id=sid, start_time=BASE + timedelta(minutes=i),
            name="gen_ai.tool.call",
        )
        span.status_message = None
        db.insert_span(span)

    finding = analyze_relearns(
        [], extra_failures=extract_span_failures(db.conn),
        advise_only_repos=non_coding_agent_ids(db.conn), distill_enabled=False,
    )

    # Never a recurring cluster despite 5 sessions all hitting the same
    # name-fallback shape — the cost is still counted, as below-threshold
    # residue (Critical Rule 32(b): the money stays counted), just never
    # asserted as a named, recurring relearn.
    assert finding.clusters == []
    assert finding.below_threshold_occurrences == 5


def test_text_less_spans_never_reach_the_distill_pass(db, monkeypatch):
    """The primary fix: spending a model call to NAME a failure that carries
    no error text at all must never happen, however many sessions hit it."""
    def _boom(*_args, **_kwargs):
        raise AssertionError("distill must never be invoked for a name-fallback cluster")

    monkeypatch.setattr(distill_mod, "_invoke_claude", _boom)

    for i, sid in enumerate(("s1", "s2", "s3", "s4", "s5")):
        span = make_tool_span(
            agent_id="billing-svc", tool_name="Bash", status="error",
            session_id=sid, start_time=BASE + timedelta(minutes=i),
            name="gen_ai.tool.call",
        )
        span.status_message = None
        db.insert_span(span)

    # distill_enabled=True on purpose — this proves the exclusion, not the
    # opt-out flag.
    finding = analyze_relearns(
        [], extra_failures=extract_span_failures(db.conn),
        advise_only_repos=non_coding_agent_ids(db.conn), distill_enabled=True,
    )

    assert finding.clusters == []


def test_below_recurrence_threshold_does_not_surface(db):
    for i, sid in enumerate(("s1", "s2")):
        _seed_failure(db, agent_id="billing-svc", session_id=sid, i=i,
                      message="ConnectionResetError: peer closed")

    finding = analyze_relearns(
        [], extra_failures=extract_span_failures(db.conn), distill_enabled=False,
    )

    assert finding.clusters == []


# -- The advise-only seam ----------------------------------------------------

def _raw_clusters(failures):
    return list(cluster_failures(failures).values())


def test_workspace_less_cluster_is_advise_only_with_no_target(db):
    for i, sid in enumerate(("s1", "s2", "s3")):
        _seed_failure(db, agent_id="billing-svc", session_id=sid, i=i,
                      message="ConnectionResetError: peer closed")
    failures = extract_span_failures(db.conn)

    proposals, _ = build_proposals(
        _raw_clusters(failures), advise_only_repos={"billing-svc"},
    )

    assert len(proposals) == 1
    p = proposals[0]
    assert p.advise_only is True
    # The enforced seam: no apply path is even suggested.
    assert p.suggested_target == ""
    assert p.repo_cwd == ""


def test_workspace_cluster_is_not_advise_only(db):
    failures = [
        FailureEpisode(
            session_id=sid, repo="myrepo", ts=None, tool_name="Read",
            label="", error_text="no such file or directory",
            kind="act", is_retry=False, depth=0,
        )
        for sid in ("s1", "s2", "s3")
    ]

    # persona="claude-code": isolates the workspace-vs-OTel seam under test
    # from the SEPARATE persona write-gate (see test_relearn.py) — the write
    # is only ever offered for a claude-code/mixed window regardless of
    # workspace, so this test's default "unknown" persona would otherwise
    # mask the very thing it's checking.
    proposals, _ = build_proposals(
        _raw_clusters(failures), advise_only_repos={"billing-svc"},
        persona="claude-code",
    )

    assert len(proposals) == 1
    assert proposals[0].advise_only is False


def test_mixed_cluster_keeps_the_apply_path(db):
    """A signature seen on BOTH a workspace repo and an OTel service is not
    advise-only: there is still a workspace to write the fix into."""
    failures = [
        FailureEpisode(
            session_id="s1", repo="myrepo", ts=None, tool_name="Read", label="",
            error_text="no such file or directory", kind="act",
            is_retry=False, depth=0,
        ),
        FailureEpisode(
            session_id="s2", repo="billing-svc", ts=None, tool_name="Read",
            label="", error_text="no such file or directory", kind="act",
            is_retry=False, depth=0,
        ),
        FailureEpisode(
            session_id="s3", repo="billing-svc", ts=None, tool_name="Read",
            label="", error_text="no such file or directory", kind="act",
            is_retry=False, depth=0,
        ),
    ]

    # persona="claude-code" — see the comment on test_workspace_cluster_is_
    # not_advise_only above.
    proposals, _ = build_proposals(
        _raw_clusters(failures), advise_only_repos={"billing-svc"},
        persona="claude-code",
    )

    assert proposals[0].advise_only is False


# -- Eval-case artifact ------------------------------------------------------

def test_to_eval_case_is_json_serializable_and_carries_evidence(db):
    import json

    for i, sid in enumerate(("s1", "s2", "s3")):
        _seed_failure(db, agent_id="billing-svc", session_id=sid, i=i,
                      message="ConnectionResetError: peer closed")
    proposals, _ = build_proposals(
        _raw_clusters(extract_span_failures(db.conn)),
        advise_only_repos={"billing-svc"},
    )

    case = to_eval_case(proposals[0])

    assert json.loads(json.dumps(case))          # round-trips as plain JSON
    assert case["sessions"] == 3
    assert case["occurrences"] == 3
    assert case["advise_only"] is True
    assert case["agents"] == ["billing-svc"]
    assert case["failure_examples"]
    assert "ConnectionResetError" in case["failure_examples"][0]["error"]
    assert case["note"]                          # the honesty caveat travels with it
