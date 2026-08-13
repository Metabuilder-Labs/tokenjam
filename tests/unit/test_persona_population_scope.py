"""The persona POPULATION scope: which ROWS an analyzer pass may read.

The sibling of ``test_persona_analyzer_gate.py``, and deliberately a separate
file because it pins a different question. That one asserts a gated analyzer is
never INVOKED for a persona; this one asserts that the analyzers which DO run
for a persona see only that persona's sessions.

The bug these exist to keep closed: the dispatch gate shipped first and was
mistaken for the whole feature, so every surviving analyzer went on aggregating
the entire mixed corpus and publishing the result under whichever persona the
reader had selected. A gate that decides *whether* an analyzer runs says nothing
about *what it runs over*.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.alerts import (
    is_interactive_coding_agent,
    interactive_coding_agent_sql,
)
from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_persona_reports, build_report
from tokenjam.core.persona_scope import (
    SCOPING_PERSONAS,
    persona_agent_clause,
    persona_scopes_population,
)
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session

CODING_AGENT = "claude-code-proj"
SDK_AGENT = "billing-service"

# Deliberately lopsided, and by an order of magnitude. A scope bug that leaks
# the other population is only visible if the two populations have obviously
# different totals — with balanced fixtures an unscoped figure looks plausible
# for either persona.
CODING_COST = 0.50
SDK_COST = 0.05


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _seed_mixed(db) -> None:
    """A corpus with BOTH personas in it, which is the only shape that can
    catch a scope leak. Six sessions each so neither side is thin-data."""
    for agent_id, cost in ((CODING_AGENT, CODING_COST), (SDK_AGENT, SDK_COST)):
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
                cost_usd=cost,
                session_id=f"{agent_id}-s{i}",
                start_time=started,
            ))


# --------------------------------------------------------------------------
# The predicate itself
# --------------------------------------------------------------------------

# Every margin case `test_alerts.py` pins for the Python classifier, so the two
# implementations are compared on the SAME inputs rather than on whatever each
# author happened to think of.
_MARGIN_CASES = [
    "claude-code", "claude-code-tokenjam", "codex", "codex-cli",
    "billing-service", "tokenjam-bench", "", "claude", "Claude-Code",
]


@pytest.mark.parametrize("agent_id", _MARGIN_CASES)
def test_sql_predicate_matches_the_python_classifier(db, agent_id):
    """The generated SQL and ``is_interactive_coding_agent`` agree, case by case.

    They are two renderings of one prefix tuple, and this is what keeps them
    that way: a hand-written ``LIKE 'claude-code%'`` somewhere in an analyzer
    would pass every test that only exercises the Python side.
    """
    sql = interactive_coding_agent_sql("$1")
    got = db.conn.execute(f"SELECT {sql}", [agent_id]).fetchone()[0]
    assert bool(got) is is_interactive_coding_agent(agent_id), agent_id


def test_the_sql_predicate_is_null_safe_and_total(db):
    """A NULL ``agent_id`` is not a coding agent, so ``NOT (...)`` keeps it.

    Without the COALESCE the SDK-side clause evaluates to NULL for such a row
    and DROPS it from both personas — a session that exists in neither view.
    """
    sql = interactive_coding_agent_sql("$1")
    coding = db.conn.execute(f"SELECT {sql}", [None]).fetchone()[0]
    sdk = db.conn.execute(f"SELECT NOT ({sql})", [None]).fetchone()[0]
    assert bool(coding) is False
    assert bool(sdk) is True


def test_only_the_two_real_personas_narrow_anything():
    """``mixed``/``unknown``/``None`` scope to the whole corpus, by design.

    Same conservative default the dispatch gate takes for an unclassified
    window. A clause for them would have to invent a rule for a window the
    classifier deliberately refused to bucket.
    """
    assert SCOPING_PERSONAS == ("claude-code", "sdk")
    for persona in ("mixed", "unknown", None):
        assert persona_scopes_population(persona) is False
        assert persona_agent_clause(persona) is None
    for persona in SCOPING_PERSONAS:
        assert persona_scopes_population(persona) is True
        assert persona_agent_clause(persona) is not None


# --------------------------------------------------------------------------
# The window summary — the denominator every share is quoted against
# --------------------------------------------------------------------------

def test_the_window_summary_is_scoped_with_the_analyzers(db, cfg):
    """Numerator and denominator must cover one population.

    ``WindowSummary.total_cost_usd`` is what a finding's share-of-window is
    computed against, so an unscoped summary under scoped findings publishes a
    ratio whose two halves describe different corpora.
    """
    _seed_mixed(db)
    since = utcnow() - timedelta(days=30)

    everything = build_report(db, cfg, since=since)
    coding = build_report(db, cfg, since=since, persona_scope="claude-code")
    sdk = build_report(db, cfg, since=since, persona_scope="sdk")

    assert coding.window.sessions == 6
    assert sdk.window.sessions == 6
    assert everything.window.sessions == 12
    # The scoped costs partition the unscoped one exactly — nothing double
    # counted, nothing dropped between the two views.
    assert coding.window.total_cost_usd == pytest.approx(6 * CODING_COST)
    assert sdk.window.total_cost_usd == pytest.approx(6 * SDK_COST)
    assert (
        coding.window.total_cost_usd + sdk.window.total_cost_usd
        == pytest.approx(everything.window.total_cost_usd)
    )


def test_a_scoped_report_records_what_it_looked_at(db, cfg):
    """``persona_scope`` is on the report, so a consumer can check before it
    publishes. ``persona`` (what the corpus IS) does not move with it."""
    _seed_mixed(db)
    since = utcnow() - timedelta(days=30)

    unscoped = build_report(db, cfg, since=since)
    scoped = build_report(db, cfg, since=since, persona_scope="sdk")

    assert unscoped.persona_scope is None
    assert scoped.persona_scope == "sdk"
    # Both classify the same corpus the same way: scoping the rows an analyzer
    # reads must not change what the window is reported to BE.
    assert scoped.persona == unscoped.persona


# --------------------------------------------------------------------------
# The analyzers' own figures
# --------------------------------------------------------------------------

def test_an_sdk_scoped_pass_excludes_interactive_coding_sessions(db, cfg):
    """The headline claim of the whole layer, checked on a real analyzer.

    ``downsize`` is the one that runs for both personas and prices sessions
    directly, so it is where a leak shows up as money.
    """
    _seed_mixed(db)
    since = utcnow() - timedelta(days=30)

    sdk = build_report(db, cfg, since=since, persona_scope="sdk", findings=["downsize"])
    coding = build_report(
        db, cfg, since=since, persona_scope="claude-code", findings=["downsize"],
    )

    for report, expected_agent in ((sdk, SDK_AGENT), (coding, CODING_AGENT)):
        finding = report.downgrade
        if finding is None:
            continue
        for example in finding.examples or []:
            assert example.agent_id == expected_agent, (
                f"{report.persona_scope} pass surfaced a {example.agent_id} "
                f"session — the population leaked"
            )


def test_each_scoped_pass_sees_only_its_own_sessions(db, cfg):
    """Session COUNTS partition, which is the population claim stated directly
    and independently of any one analyzer's arithmetic."""
    _seed_mixed(db)
    since = utcnow() - timedelta(days=30)

    coding = build_report(db, cfg, since=since, persona_scope="claude-code")
    sdk = build_report(db, cfg, since=since, persona_scope="sdk")

    assert coding.window.spans == 6
    assert sdk.window.spans == 6


# --------------------------------------------------------------------------
# The daemon's artifact
# --------------------------------------------------------------------------

def test_the_daemon_artifact_carries_one_scoped_report_per_persona(db, cfg):
    """A population cannot be unioned, so the pass runs once per persona.

    This is the structural difference from the analyzer-set gate, which CAN be
    unioned at compute time and sliced on read.
    """
    _seed_mixed(db)
    since = utcnow() - timedelta(days=30)

    artifact = build_persona_reports(db, cfg, since=since)

    assert set(artifact.persona_reports) == set(SCOPING_PERSONAS)
    # The top-level report stays UNSCOPED: a persona-blind reader gets the
    # corpus, not whichever persona happens to dominate it.
    assert artifact.persona_scope is None
    assert artifact.window.sessions == 12

    for persona, stored in artifact.persona_reports.items():
        assert stored["persona_scope"] == persona
        assert stored["window"]["sessions"] == 6


def test_a_stored_artifact_survives_the_round_trip_with_its_scope(db, cfg):
    """``persona_scope`` and ``persona_reports`` must come back off disk.

    A rehydrated report that lost its scope is indistinguishable from an
    unscoped one, which is precisely the claim a consumer checks before
    publishing a persona-labelled figure.
    """
    from tokenjam.core.optimize import report_from_dict, report_to_dict

    _seed_mixed(db)
    since = utcnow() - timedelta(days=30)
    artifact = build_persona_reports(db, cfg, since=since)

    back = report_from_dict(report_to_dict(artifact))
    assert back.persona_scope is None
    assert set(back.persona_reports) == set(SCOPING_PERSONAS)
    assert back.computed_for_personas == artifact.computed_for_personas

    scoped = report_from_dict(artifact.persona_reports["sdk"])
    assert scoped.persona_scope == "sdk"


def test_a_report_predating_population_scoping_reads_as_unscoped(db, cfg):
    """Back-compat, stated as a property rather than left to chance.

    An artifact written before this field existed has no ``persona_scope`` key.
    It must rehydrate as ``None`` — unscoped, which is the truth about it — and
    never as the persona a caller happened to ask for.
    """
    from tokenjam.core.optimize import report_from_dict, report_to_dict

    _seed_mixed(db)
    legacy = report_to_dict(build_report(db, cfg, since=utcnow() - timedelta(days=30)))
    legacy.pop("persona_scope", None)
    legacy.pop("persona_reports", None)

    back = report_from_dict(legacy)
    assert back.persona_scope is None
    assert back.persona_reports == {}
