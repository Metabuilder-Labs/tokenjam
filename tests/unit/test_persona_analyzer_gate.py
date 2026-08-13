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
from tokenjam.core.optimize import runner as runner_mod
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

# sdk's own gate: these are gated on an INPUT this persona structurally never
# has (an on-disk Claude Code transcript, a populated `sub_agent_id`, an agent
# instruction file for the summarize catalog to scan), not on a missing lever —
# same "must never be invoked" bar, different reason. Derived from the
# production map for the same reason NO_LEVER is: a name silently dropped from
# the gate fails here.
SDK_NO_LEVER = set(PERSONA_DISABLED_ANALYZERS.get("sdk", frozenset()))
assert SDK_NO_LEVER == {"deadweight", "subagent", "summarize"}

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
    """`deadweight` (on-disk Claude Code transcripts), `subagent`
    (`sub_agent_id IS NOT NULL`) and `summarize` (a filesystem catalog of agent
    instruction files) can never produce a candidate for an SDK window —
    true-skip them rather than dispatch work that structurally always returns
    nothing to act on."""
    _seed(db, "billing-service")
    invoked, report = _run_recording_invocations(db, cfg, monkeypatch)

    assert report.persona == "sdk"
    # Never invoked -> no query ran, and nothing to render.
    assert SDK_NO_LEVER.isdisjoint(invoked)
    assert SDK_NO_LEVER.isdisjoint(report.findings)


def test_summarize_is_gated_for_sdk_and_untouched_for_claude_code(db, cfg, monkeypatch):
    """Named explicitly, not just covered set-wise by SDK_NO_LEVER.

    `summarize` prices what a filesystem scan of AGENT INSTRUCTION FILES found
    (`core/summarize/agent_files.toml`: CLAUDE.md, AGENTS.md, GEMINI.md, the
    rules/skills/commands/agents markdown beside them). It has never scanned
    application source or a prompt template in `.py`/`.ts`, so an SDK window's
    prompt text is outside its population by construction and the scan can only
    return a card with no fix behind it.

    Both halves matter, and the CC half is the one that keeps this test from
    passing vacuously (Critical Rule 40): if `summarize` stopped being
    dispatched at all, the SDK assertion alone would still be green.
    """
    sdk_backend = InMemoryBackend()
    try:
        _seed(sdk_backend, "billing-service")
        sdk_invoked, sdk_report = _run_recording_invocations(sdk_backend, cfg, monkeypatch)
        assert sdk_report.persona == "sdk"
        assert "summarize" not in sdk_invoked
        assert "summarize" not in sdk_report.findings
    finally:
        sdk_backend.close()

    _seed(db, "claude-code-proj")
    cc_invoked, cc_report = _run_recording_invocations(db, cfg, monkeypatch)
    assert cc_report.persona == "claude-code"
    assert "summarize" in cc_invoked


def test_summarize_card_is_dropped_from_the_review_inbox_for_sdk():
    """The second selection surface has to make the same call, or a gated
    analyzer's finding still lands in Review as a card the user cannot act on
    (Critical Rule 26b). A report carrying the finding is a real shape here: a
    cached payload, or a wider selection, can still hold one."""
    from tokenjam.core.optimize.analyzers.summarize import (
        SummarizeCandidate,
        SummarizeFinding,
    )

    w = WindowSummary(since=NOW - timedelta(days=30), until=NOW, days=30, sessions=10,
                      spans=100, total_tokens=1_000_000, total_cost_usd=50.0,
                      thin_data=False)
    finding = SummarizeFinding(
        candidates=[SummarizeCandidate(path="/repo/CLAUDE.md", kind="prompt",
                                       scope="project", est_tokens_saved=900,
                                       total_chars=12_000, reduction_pct=30)],
        files=1, past_overspend_usd=4.0, past_overspend_tokens=90_000,
        avg_reduction_pct=30,
    )

    def _analyzers(persona: str) -> set[str]:
        rep = OptimizeReport(window=w, persona=persona, findings={"summarize": finding})
        return {p.analyzer for p in cost_proposals_from_report(rep)}

    # The vehicle is valid only if the card exists at all for the persona that
    # keeps the analyzer — assert that before asserting its absence.
    assert "summarize" in _analyzers("claude-code")
    assert "summarize" not in _analyzers("sdk")
    assert "summarize" not in cost_analyzers_for_persona("sdk")
    assert "summarize" in cost_analyzers_for_persona("claude-code")


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


def test_gate_saves_real_query_work_for_both_gated_personas(cfg, monkeypatch):
    """Not merely hidden: the skipped analyzers issue no SQL of their own.

    Measured against the SAME persona with the gate emptied, never one persona
    against the other. The cross-persona form this replaced (claude-code issues
    fewer statements than sdk) was a proxy for the claude-code gate being the
    only heavy one, and it inverted the moment the sdk key grew: it then failed
    while both gates were working perfectly. A control that can invert without
    the property under test changing was measuring the wrong thing.
    """
    def _count_queries(agent_id: str) -> int:
        backend = InMemoryBackend()
        try:
            _seed(backend, agent_id)
            counting = _CountingDb(backend)
            build_report(counting, cfg, since=utcnow() - timedelta(days=30))
            return len(counting.conn.calls)
        finally:
            backend.close()

    gated_cc = _count_queries("claude-code-proj")
    gated_sdk = _count_queries("billing-service")
    monkeypatch.setattr(runner_mod, "PERSONA_DISABLED_ANALYZERS", {})
    assert gated_cc < _count_queries("claude-code-proj")
    assert gated_sdk < _count_queries("billing-service")


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
    assert SDK_NO_LEVER.isdisjoint(sdk_scoped)
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


# --------------------------------------------------------------------------- #
# The two gate surfaces must agree — for EVERY persona, not just the two the
# tests above happen to name.
#
# `runner.PERSONA_DISABLED_ANALYZERS` decides what gets DISPATCHED;
# `cost_proposals.COST_ANALYZERS` / `cost_analyzers_for_persona` decides what
# reaches the Review inbox. They are independent selection surfaces over one
# decision, and nothing structural forces them to agree — `cost_analyzers_for_
# persona` merely happens to call the same helper today. `test_cost_analyzers_
# mirror_the_gate` above checks the two personas it names by hand; these check
# the PROPERTY, so a third persona key, or a reimplementation of either side
# that reintroduces a literal list, fails immediately.
# --------------------------------------------------------------------------- #
def test_the_two_gate_surfaces_agree_for_every_persona():
    from tokenjam.core.framing import PERSONAS

    # Every persona the classifier can produce, plus every key the gate map
    # carries, plus a name that is neither (the unlisted-persona path).
    for persona in (*PERSONAS, *PERSONA_DISABLED_ANALYZERS, "not-a-persona"):
        disabled = disabled_analyzers_for_persona(persona)
        scoped = cost_analyzers_for_persona(persona)
        # Exact, both directions: nothing the dispatch gate drops survives into
        # the inbox surface, and nothing it keeps is dropped there.
        assert set(scoped) == set(COST_ANALYZERS) - set(disabled), persona
        # Order is preserved, so the inbox's own ranking cannot be perturbed by
        # the filter.
        assert scoped == tuple(n for n in COST_ANALYZERS if n not in disabled), persona


def test_every_gated_name_is_either_a_registry_entry_or_a_named_sub_check():
    """A gate entry that matches nothing silently protects nothing.

    `placement` is the one legitimate non-registry name (the `downsize`
    analyzer attaches it as a sub-check and reads the same map to skip it). Any
    OTHER unmatched name is a typo that disables an analyzer nobody has — which
    reads, from every surface, exactly like a working gate.
    """
    known = set(ANALYZER_REGISTRY) | {"placement"}
    for persona, names in PERSONA_DISABLED_ANALYZERS.items():
        unknown = set(names) - known
        assert not unknown, f"{persona} gates unknown analyzer(s): {sorted(unknown)}"


# --------------------------------------------------------------------------- #
# Serving one stored report AS a persona other than the corpus's dominant one.
# --------------------------------------------------------------------------- #
def test_a_multi_persona_pass_runs_the_union_and_records_what_it_ran(
    db, cfg, monkeypatch,
):
    """The daemon's pass has to answer for EITHER side of the persona picker
    off ONE artifact, because no request path may run an analyzer.

    So `personas` gates on the INTERSECTION of the named personas' disabled
    sets — a name only stays skipped when nobody can act on it — and the report
    records what it actually dispatched, so a reader can tell "ran and found
    nothing" from "never invoked".
    """
    _seed(db, "claude-code-proj")
    invoked: list[str] = []
    for name, fn in list(ANALYZER_REGISTRY.items()):
        def _wrapped(ctx, _name=name, _fn=fn):
            invoked.append(_name)
            return _fn(ctx)
        monkeypatch.setitem(ANALYZER_REGISTRY, name, _wrapped)

    report = build_report(
        db, cfg, since=utcnow() - timedelta(days=30),
        personas=["claude-code", "sdk"],
    )

    # Still a claude-code corpus — widening the analyzer set must not restate
    # what the window IS.
    assert report.persona == "claude-code"
    # The union: nothing is gated, because the two personas' reasons are
    # disjoint (no-lever vs no-input), so each name is actionable for one side.
    assert set(invoked) == set(ANALYZER_REGISTRY)
    assert set(report.computed_analyzers) == set(ANALYZER_REGISTRY)
    assert report.computed_for_personas == ["claude-code", "sdk"]
    # Including the ones a single-persona claude-code pass would have skipped.
    assert NO_LEVER_REGISTRY_NAMES <= set(invoked)


def test_the_default_pass_is_unchanged_and_still_records_its_selection(db, cfg):
    """`personas=None` is the old behaviour exactly — one persona asked, one
    persona answered — and the CLI depends on that."""
    _seed(db, "claude-code-proj")
    report = build_report(db, cfg, since=utcnow() - timedelta(days=30))

    assert report.persona == "claude-code"
    assert report.computed_for_personas == ["claude-code"]
    assert NO_LEVER.isdisjoint(report.computed_analyzers)
    assert set(report.computed_analyzers) == set(ANALYZER_REGISTRY) - NO_LEVER


def test_findings_for_persona_narrows_a_union_report_back_down():
    """The read-side half of the gate: a report computed for both personas is
    sliced per request, so a surface serving one persona can never render a
    finding for a lever that persona does not have."""
    from tokenjam.core.optimize import findings_for_persona

    findings = {name: object() for name in ANALYZER_REGISTRY}
    for persona in ("claude-code", "sdk"):
        sliced = findings_for_persona(findings, persona)
        disabled = disabled_analyzers_for_persona(persona)
        assert set(sliced) == set(findings) - set(disabled), persona
    # An unclassified persona keeps everything (the conservative default).
    assert set(findings_for_persona(findings, "mixed")) == set(findings)


def test_intersection_gate_is_empty_for_the_two_shipped_personas():
    """Named explicitly because it is the property the whole design rests on:
    the claude-code gate is about a missing LEVER and the sdk gate about a
    missing INPUT, so the two sets are disjoint and their intersection is empty.
    A future gate entry shared by both personas would start being skipped in the
    daemon's pass too — which is correct, and this test is where that shows up.
    """
    from tokenjam.core.optimize import GATED_PERSONAS, disabled_analyzers_for_personas

    assert set(GATED_PERSONAS) == set(PERSONA_DISABLED_ANALYZERS)
    assert disabled_analyzers_for_personas(GATED_PERSONAS) == frozenset()
    # No personas named disables nothing, matching the unlisted-persona default.
    assert disabled_analyzers_for_personas([]) == frozenset()
