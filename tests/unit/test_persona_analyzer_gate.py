"""The persona skip gate: analyzers with no fix for the window's dominant
persona must never be INVOKED — not run-and-report-disabled.

Two distinct "disabled" mechanisms exist in this codebase and these tests pin
the difference: an analyzer that returns an ``enabled=False`` finding still ran
its queries and still occupies a row, whereas a persona-gated analyzer is
dropped from the selection before dispatch, so it costs nothing and is absent
from every downstream surface.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import (
    ANALYZER_REGISTRY,
    PERSONA_DISABLED_ANALYZERS,
    build_report,
    disabled_analyzers_for_persona,
)
from tokenjam.core.optimize.analyzers.batch_placement import BatchPlacementFinding
from tokenjam.core.optimize.analyzers.cache_efficacy import (
    CacheEfficacyFinding,
    CacheEfficacyRow,
)
from tokenjam.core.optimize.analyzers.prompt_bloat import BloatPrompt, PromptBloatFinding
from tokenjam.core.optimize.cost_proposals import (
    COST_ANALYZERS,
    cost_analyzers_for_persona,
    cost_proposals_from_report,
)
from tokenjam.core.optimize.types import OptimizeReport, WindowSummary
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session

# Derived from the production map, never hand-copied: a name added to the gate
# is automatically covered by every test below, and a name silently DROPPED
# from the gate fails the pin immediately underneath. Hand-maintaining this set
# is how `verbosity` and `script` ended up gated in production but exercised by
# none of these tests.
NO_LEVER = set(PERSONA_DISABLED_ANALYZERS["claude-code"])

# The pin: what the product decision names explicitly. Any change to the gate
# is a product decision and must be made here deliberately.
assert NO_LEVER == {
    "cache", "cache-recommend", "placement", "trim", "verbosity", "script", "reuse",
    "stream-usage",
}

# `placement` is the odd one out: it is not an ANALYZER_REGISTRY name at all
# (the `downsize` analyzer attaches it as a sub-check), so it is never
# "invoked" and never appears in `report.findings` under its own name.
NO_LEVER_REGISTRY_NAMES = NO_LEVER - {"placement"}

# sdk's own gate: `deadweight`/`subagent` are gated on a DATA SOURCE this
# persona structurally never has (an on-disk Claude Code transcript, a
# populated `sub_agent_id`), not on a missing lever — same "must never be
# invoked" bar, different reason. Derived from the production map for the
# same reason NO_LEVER is: a name silently dropped from the gate fails here.
SDK_NO_LEVER = set(PERSONA_DISABLED_ANALYZERS.get("sdk", frozenset()))
assert SDK_NO_LEVER == {"deadweight", "subagent"}

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _seed(db, agent_id: str) -> None:
    """Enough priced spans + sessions for the window to classify a persona.

    Persona is derived from ``sessions.agent_id`` (see
    ``framing.agent_persona_mix``), so the sessions matter as much as the spans.
    """
    for i in range(6):
        started = utcnow() - timedelta(days=i + 1)
        db.upsert_session(make_session(
            agent_id=agent_id,
            session_id=f"{agent_id}-s{i}",
            plan_tier="api",
            started_at=started,
        ))
        db.insert_span(make_llm_span(
            agent_id=agent_id,
            model="claude-opus-4-7",
            provider="anthropic",
            input_tokens=4000,
            output_tokens=800,
            cost_usd=0.12,
            session_id=f"{agent_id}-s{i}",
            start_time=started,
        ))


def _run_recording_invocations(db, cfg, monkeypatch) -> tuple[list[str], OptimizeReport]:
    """Build a report with every registry entry wrapped in a call recorder."""
    invoked: list[str] = []
    for name, fn in list(ANALYZER_REGISTRY.items()):
        def _wrapped(ctx, _name=name, _fn=fn):
            invoked.append(_name)
            return _fn(ctx)
        monkeypatch.setitem(ANALYZER_REGISTRY, name, _wrapped)
    report = build_report(db, cfg, since=utcnow() - timedelta(days=30))
    return invoked, report


def test_claude_code_window_never_invokes_the_no_lever_analyzers(db, cfg, monkeypatch):
    _seed(db, "claude-code-proj")
    invoked, report = _run_recording_invocations(db, cfg, monkeypatch)

    assert report.persona == "claude-code"
    # Never invoked -> no query ran, and nothing to render.
    assert NO_LEVER.isdisjoint(invoked)
    # Absent from the payload entirely, not present-and-disabled.
    assert NO_LEVER.isdisjoint(report.findings)
    # ...including `placement`, which `downsize` attaches as a sub-check and so
    # cannot be dropped by the selection gate alone.
    assert "downsize" in invoked
    assert "placement" not in report.findings


def test_sdk_window_is_unaffected_by_the_claude_code_no_lever_gate(db, cfg, monkeypatch):
    _seed(db, "billing-service")
    invoked, report = _run_recording_invocations(db, cfg, monkeypatch)

    assert report.persona == "sdk"
    assert NO_LEVER_REGISTRY_NAMES <= set(invoked)
    assert "placement" in report.findings
    for name in NO_LEVER_REGISTRY_NAMES:
        assert name in invoked, name


def test_sdk_window_never_invokes_the_data_source_gated_analyzers(db, cfg, monkeypatch):
    """`deadweight` (on-disk Claude Code transcripts) and `subagent`
    (`sub_agent_id IS NOT NULL`) can never produce a candidate for an SDK
    window — true-skip them rather than dispatch a query that structurally
    always returns nothing to act on."""
    _seed(db, "billing-service")
    invoked, report = _run_recording_invocations(db, cfg, monkeypatch)

    assert report.persona == "sdk"
    # Never invoked -> no query ran, and nothing to render.
    assert SDK_NO_LEVER.isdisjoint(invoked)
    assert SDK_NO_LEVER.isdisjoint(report.findings)


def test_claude_code_window_still_invokes_the_sdk_gated_analyzers(db, cfg, monkeypatch):
    """The sdk gate is persona-scoped: a claude-code window (real on-disk
    transcripts, real subagent dispatches) must keep running both."""
    _seed(db, "claude-code-proj")
    invoked, report = _run_recording_invocations(db, cfg, monkeypatch)

    assert report.persona == "claude-code"
    for name in SDK_NO_LEVER:
        assert name in invoked, name


class _CountingConn:
    """Delegating DuckDB connection that records every statement executed."""

    def __init__(self, conn):
        self._conn = conn
        self.calls: list[str] = []

    def execute(self, sql, *args, **kwargs):
        self.calls.append(str(sql))
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


class _CountingDb:
    def __init__(self, backend):
        self.conn = _CountingConn(backend.conn)


def test_gate_saves_real_query_work_on_a_claude_code_window(cfg):
    """Not merely hidden: the skipped analyzers issue no SQL of their own."""
    def _count_queries(agent_id: str) -> int:
        backend = InMemoryBackend()
        try:
            _seed(backend, agent_id)
            counting = _CountingDb(backend)
            build_report(counting, cfg, since=utcnow() - timedelta(days=30))
            return len(counting.conn.calls)
        finally:
            backend.close()

    assert _count_queries("claude-code-proj") < _count_queries("billing-service")


def test_mixed_and_unknown_personas_disable_nothing():
    # Conservative default: an UNCLASSIFIED window loses no analyzer.
    assert disabled_analyzers_for_persona("mixed") == frozenset()
    assert disabled_analyzers_for_persona("unknown") == frozenset()
    # Exact, not a subset: the mirror must cover the whole gate.
    assert NO_LEVER == set(PERSONA_DISABLED_ANALYZERS["claude-code"])


def test_sdk_persona_disables_exactly_the_data_source_gated_pair():
    # sdk is classified, but its gate is on data-source reachability, not the
    # claude-code no-lever set — the two gates are disjoint reasons and must
    # stay disjoint sets (`deadweight`/`subagent` are never gated FOR
    # claude-code, which has real data for both).
    assert disabled_analyzers_for_persona("sdk") == SDK_NO_LEVER
    assert SDK_NO_LEVER.isdisjoint(NO_LEVER)


def test_explicitly_requested_disabled_analyzer_is_still_skipped(db, cfg):
    """A named request doesn't reopen the gate — there is still no fix to show."""
    _seed(db, "claude-code-proj")
    report = build_report(
        db, cfg, since=utcnow() - timedelta(days=30), findings=["cache", "deadweight"],
    )
    assert "cache" not in report.findings
    assert "deadweight" in report.findings


def test_unknown_analyzer_name_still_raises(db, cfg):
    """The gate runs after validation, so a typo is still a hard error."""
    _seed(db, "claude-code-proj")
    with pytest.raises(ValueError, match="Unknown finding"):
        build_report(db, cfg, since=utcnow() - timedelta(days=30), findings=["cahce"])


# --- The second selection surface: COST_ANALYZERS / the Review inbox ---------

def test_cost_analyzers_mirror_the_gate():
    scoped = cost_analyzers_for_persona("claude-code")
    assert NO_LEVER.isdisjoint(scoped)
    assert "deadweight" in scoped and "subagent" in scoped
    # sdk's own gate: the pair with no reachable data source for this
    # persona must be absent from the Review inbox's own selection surface
    # too, or a disabled analyzer's findings would still reach it as
    # apply-able cards (COST_ANALYZERS is an independent second surface).
    sdk_scoped = cost_analyzers_for_persona("sdk")
    assert "deadweight" not in sdk_scoped and "subagent" not in sdk_scoped
    assert set(sdk_scoped) == set(COST_ANALYZERS) - SDK_NO_LEVER
    # Any unclassified persona keeps the full list.
    assert cost_analyzers_for_persona("unknown") == COST_ANALYZERS


def _report_with_no_lever_findings(persona: str) -> OptimizeReport:
    w = WindowSummary(since=NOW - timedelta(days=30), until=NOW, days=30, sessions=10,
                      spans=100, total_tokens=1_000_000, total_cost_usd=50.0,
                      thin_data=False)
    cache = CacheEfficacyFinding(
        flagged=[CacheEfficacyRow("anthropic", "claude-sonnet-5", 100_000, 5_000,
                                  0.05, "full", True)],
        past_overspend_usd=1.2, estimate_basis="cache basis",
    )
    trim = PromptBloatFinding(
        enabled=True,
        per_prompt=[BloatPrompt(agent_id="svc-a", sample_chars="x", prompt_chars=8000,
                                significant_chars=3000, bloat_chars=5000,
                                estimated_token_reduction=1250)],
        past_overspend_usd=0.8, estimate_basis="trim basis",
    )
    return OptimizeReport(
        window=w, persona=persona, findings={"cache": cache, "trim": trim},
    )


def test_review_inbox_shows_no_card_for_a_gated_analyzer():
    """Belt-and-braces: even handed a report that somehow carries the finding,
    the inbox adapter must not build an unactionable card for this persona."""
    cc = {p.analyzer for p in cost_proposals_from_report(_report_with_no_lever_findings("claude-code"))}
    assert not (cc & {"cache", "cache_thrash", "cache-recommend", "trim", "placement"})

    sdk = {p.analyzer for p in cost_proposals_from_report(_report_with_no_lever_findings("sdk"))}
    assert {"cache", "trim"} <= sdk


def test_review_inbox_still_gates_placement_for_claude_code():
    w = WindowSummary(since=NOW - timedelta(days=30), until=NOW, days=30, sessions=10,
                      spans=100, total_tokens=1_000_000, total_cost_usd=50.0,
                      thin_data=False)
    placement = BatchPlacementFinding(
        candidates=[], past_overspend_usd=None, estimate_basis="batch basis",
        min_sessions_for_cadence=5, min_group_cost_usd=1.0,
    )
    rep = OptimizeReport(window=w, persona="claude-code", findings={"placement": placement})
    assert cost_proposals_from_report(rep) == []
