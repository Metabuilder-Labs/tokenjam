"""The `stream-usage` analyzer: report the blind spot, never bank it as savings.

The load-bearing constraint here is accounting, not detection. What this
analyzer measures is spend that ALREADY HAPPENED and was never recorded —
capturing it makes the spend figure correct, it does not make it smaller. A
figure derived that way must never reach the recoverable-waste total, whose
whole meaning is "money that could have been avoided". The tests below pin
that separation structurally (no `past_overspend_*` field, absent from
`COST_ANALYZERS`, invisible to the rollup) rather than by inspecting copy,
because copy is what drifted the last three times this class of bug shipped.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from tokenjam.core.config import StorageConfig, TjConfig
from tokenjam.core.db import InMemoryBackend
from tokenjam.core.optimize import build_report
from tokenjam.core.optimize.analyzers.stream_usage import StreamUsageFinding
from tokenjam.core.optimize.cost_proposals import (
    COST_ANALYZERS,
    cost_proposals_from_report,
    past_overspend_rollup,
)
from tokenjam.core.pricing import get_rates
from tokenjam.otel.semconv import TjAttributes
from tokenjam.utils.time_parse import utcnow
from tests.factories import make_llm_span, make_session

WINDOW_DAYS = 30


@pytest.fixture
def db():
    backend = InMemoryBackend()
    yield backend
    backend.close()


@pytest.fixture
def cfg(tmp_path):
    return TjConfig(version="1", storage=StorageConfig(path=str(tmp_path / "t.duckdb")))


def _stream_attrs(*, usage_reported: bool, content_chunks: int = 4) -> dict:
    return {
        TjAttributes.STREAMING: True,
        TjAttributes.STREAM_USAGE_REPORTED: usage_reported,
        TjAttributes.STREAM_CONTENT_CHUNKS: content_chunks,
    }


def _seed_stream(db, *, agent_id, provider, model, usage_reported,
                 input_tokens=0, output_tokens=0, cache_tokens=0,
                 cache_write_tokens=0, content_chunks=4, session_suffix="0",
                 days_ago=1):
    """One streamed call, complete or truncated, plus its session."""
    started = utcnow() - timedelta(days=days_ago)
    session_id = f"{agent_id}-{session_suffix}"
    db.upsert_session(make_session(
        agent_id=agent_id, session_id=session_id,
        plan_tier="api", started_at=started,
    ))
    db.insert_span(make_llm_span(
        agent_id=agent_id, provider=provider, model=model,
        billing_account=provider,
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_tokens=cache_tokens, cache_write_tokens=cache_write_tokens,
        session_id=session_id, start_time=started,
        extra_attributes=_stream_attrs(
            usage_reported=usage_reported, content_chunks=content_chunks,
        ),
    ))


def _seed_openai_app(db, *, truncated=2, complete=4):
    """An SDK app streaming from an OpenAI-compatible API.

    The complete calls are the peer baseline; the truncated ones are the gap.
    """
    for i in range(complete):
        _seed_stream(
            db, agent_id="chat-service", provider="openai", model="gpt-4o-mini",
            usage_reported=True, input_tokens=1000, output_tokens=300,
            session_suffix=f"c{i}", days_ago=i + 1,
        )
    for i in range(truncated):
        _seed_stream(
            db, agent_id="chat-service", provider="openai", model="gpt-4o-mini",
            usage_reported=False, session_suffix=f"t{i}", days_ago=i + 1,
        )


def _seed_anthropic_app(db, *, truncated=1, complete=3):
    for i in range(complete):
        _seed_stream(
            db, agent_id="support-bot", provider="anthropic",
            model="claude-haiku-4-5", usage_reported=True,
            input_tokens=800, output_tokens=200, cache_tokens=500,
            cache_write_tokens=100, session_suffix=f"ac{i}", days_ago=i + 1,
        )
    for i in range(truncated):
        _seed_stream(
            db, agent_id="support-bot", provider="anthropic",
            model="claude-haiku-4-5", usage_reported=False,
            session_suffix=f"at{i}", days_ago=i + 1,
        )


def _finding(db, cfg) -> StreamUsageFinding:
    report = build_report(
        db, cfg, since=utcnow() - timedelta(days=WINDOW_DAYS),
        findings=["stream-usage"],
    )
    return report.findings["stream-usage"]


# ---------------------------------------------------------------------------
# Detection + the fix each provider actually needs
# ---------------------------------------------------------------------------

def test_openai_compatible_stream_without_usage_is_flagged_with_its_own_fix(db, cfg):
    _seed_openai_app(db, truncated=2, complete=4)

    finding = _finding(db, cfg)

    assert finding.streams_observed == 6
    assert finding.streams_missing_usage == 2
    assert len(finding.call_sites) == 1
    site = finding.call_sites[0]
    assert (site.provider, site.model, site.agent_id) == (
        "openai", "gpt-4o-mini", "chat-service",
    )
    assert site.affected_calls == 2
    assert site.complete_calls == 4
    # The remediation must be the OpenAI-compatible one: the API emits no usage
    # payload at all without the opt-in, so naming only "drain the stream"
    # would be advice that cannot work.
    assert 'stream_options={"include_usage": True}' in site.remediation_snippet
    assert "get_final_message" not in site.remediation_snippet


def test_anthropic_stream_without_usage_gets_the_anthropic_fix(db, cfg):
    _seed_anthropic_app(db, truncated=1, complete=3)

    site = _finding(db, cfg).call_sites[0]

    assert site.provider == "anthropic"
    assert site.affected_calls == 1
    # Anthropic has no include_usage flag — offering one would be wrong.
    assert "stream_options" not in site.remediation_snippet
    assert "get_final_message" in site.remediation_snippet
    assert "messages.stream" in site.remediation_snippet


def test_both_providers_are_reported_as_separate_call_sites(db, cfg):
    _seed_openai_app(db)
    _seed_anthropic_app(db)

    finding = _finding(db, cfg)

    providers = {s.provider for s in finding.call_sites}
    assert providers == {"openai", "anthropic"}
    assert all(s.remediation_snippet for s in finding.call_sites)


def test_a_stream_that_produced_no_content_is_not_counted_as_a_gap(db, cfg):
    """An empty stream under-counted nothing; flagging it is noise."""
    _seed_openai_app(db, truncated=0, complete=4)
    _seed_stream(
        db, agent_id="chat-service", provider="openai", model="gpt-4o-mini",
        usage_reported=False, content_chunks=0, session_suffix="empty",
    )

    finding = _finding(db, cfg)

    assert finding.streams_missing_usage == 0
    assert finding.call_sites == []


def test_no_streams_observed_explains_itself_rather_than_claiming_zero(db, cfg):
    db.upsert_session(make_session(
        agent_id="chat-service", session_id="s0", plan_tier="api",
        started_at=utcnow() - timedelta(days=1),
    ))

    finding = _finding(db, cfg)

    assert finding.streams_observed == 0
    assert finding.undercounted_usd is None
    assert "No streamed calls were observed" in finding.hint


# ---------------------------------------------------------------------------
# The estimate and its derivation
# ---------------------------------------------------------------------------

def test_the_estimate_states_its_derivation_and_names_the_peer_baseline(db, cfg):
    _seed_openai_app(db, truncated=2, complete=4)

    finding = _finding(db, cfg)
    site = finding.call_sites[0]

    # Peer medians are the observed complete calls, not a constant.
    assert site.peer_input_tokens == 1000
    assert site.peer_output_tokens == 300
    assert site.undercounted_tokens == 2 * 1300
    assert site.undercounted_usd is not None and site.undercounted_usd > 0
    assert "median" in site.derivation.lower()
    assert "Estimated, not observed" in site.derivation
    assert finding.estimate_basis
    assert finding.estimate_confidence == "heuristic"


def test_cache_token_types_both_enter_the_peer_baseline(db, cfg):
    """Cache-read and cache-write bill at different rates and both count."""
    _seed_anthropic_app(db, truncated=1, complete=3)

    site = _finding(db, cfg).call_sites[0]

    assert site.peer_cache_tokens == 500
    assert site.peer_cache_write_tokens == 100
    assert site.undercounted_tokens == 800 + 200 + 500 + 100


def test_implied_per_token_rate_equals_the_blend_the_basis_advertises(db, cfg):
    """Critical Rule 28: divide the dollar field by the token field and the
    implied rate must be the blend the basis string describes.

    The price-band check alone is necessary but not sufficient — a 2x basis
    mismatch (dollars multiplied per call, tokens left one-time) sits happily
    inside a band that spans cache-read to output rates. Asserting the EXACT
    blend is what makes a drift between the two fields impossible to miss."""
    _seed_anthropic_app(db, truncated=2, complete=3)

    finding = _finding(db, cfg)
    rates = get_rates("anthropic", "claude-haiku-4-5")
    implied_per_mtok = (
        finding.undercounted_usd / finding.undercounted_tokens * 1_000_000
    )

    # The blend the derivation claims: the peer-median mix of the four token
    # types, each at its own rate. Call count cancels out — which is the point:
    # it must cancel out of BOTH fields or not at all.
    site = finding.call_sites[0]
    expected_usd_per_call = (
        site.peer_input_tokens * rates.input_per_mtok
        + site.peer_output_tokens * rates.output_per_mtok
        + site.peer_cache_tokens * rates.cache_read_per_mtok
        + site.peer_cache_write_tokens * rates.cache_write_per_mtok
    ) / 1_000_000
    expected_tokens_per_call = (
        site.peer_input_tokens + site.peer_output_tokens
        + site.peer_cache_tokens + site.peer_cache_write_tokens
    )
    expected_per_mtok = expected_usd_per_call / expected_tokens_per_call * 1_000_000
    assert implied_per_mtok == pytest.approx(expected_per_mtok, rel=1e-6)

    # ...and it is still a rate somebody actually charges.
    floor = min(rates.cache_read_per_mtok, rates.input_per_mtok)
    ceiling = max(rates.output_per_mtok, rates.cache_write_per_mtok)
    assert floor <= implied_per_mtok <= ceiling


def test_a_call_site_with_no_completed_peer_degrades_on_BOTH_fields(db, cfg):
    """Symmetric degrade: a call site contributes to both sums or to neither.
    A zero on one side and a number on the other corrupts the implied rate."""
    _seed_stream(
        db, agent_id="chat-service", provider="openai", model="gpt-4o-mini",
        usage_reported=False, session_suffix="lonely",
    )

    finding = _finding(db, cfg)
    site = finding.call_sites[0]

    assert site.affected_calls == 1
    assert site.undercounted_tokens is None
    assert site.undercounted_usd is None
    assert finding.undercounted_usd is None
    assert finding.undercounted_tokens is None
    assert "no per-call baseline" in site.derivation


def test_an_unpriced_call_site_does_not_silently_shrink_the_total(db, cfg):
    """The priced part is still published; the unpriced part is disclosed in
    words rather than folded in as zero."""
    _seed_openai_app(db, truncated=2, complete=4)
    _seed_stream(
        db, agent_id="other-service", provider="anthropic",
        model="claude-haiku-4-5", usage_reported=False, session_suffix="solo",
    )

    finding = _finding(db, cfg)

    priced = [s for s in finding.call_sites if s.undercounted_usd is not None]
    assert len(priced) == 1
    assert finding.undercounted_usd == priced[0].undercounted_usd
    assert "had no completed stream to size against" in finding.estimate_basis


# ---------------------------------------------------------------------------
# The accounting separation — the reason this analyzer is unusual
# ---------------------------------------------------------------------------

def test_the_finding_carries_no_canonical_recoverable_dollar_field(db, cfg):
    """`past_overspend_*` is THE avoidable-spend field the rollup sums. This
    quantity is not avoidable spend, so it must not wear that name — the
    rollup is generic over analyzers and would pick it up by field presence."""
    _seed_openai_app(db)

    finding = _finding(db, cfg)

    for banned in (
        "past_overspend_usd", "past_overspend_tokens", "past_overspend_basis",
        "estimated_recoverable_usd", "estimated_monthly_usd",
        "gross_recoverable_usd",
    ):
        assert not hasattr(finding, banned), banned
        assert all(not hasattr(s, banned) for s in finding.call_sites), banned


def test_the_serialized_finding_cannot_mint_an_overview_waste_tile(db, cfg):
    """The Overview's recoverable-waste band is registry-driven off the
    PRESENCE of `past_overspend_usd` in the serialized finding dict — it reads
    the payload, not the dataclass — so the key must be absent there too."""
    from tokenjam.core.optimize import report_to_dict

    _seed_openai_app(db)
    report = build_report(db, cfg, since=utcnow() - timedelta(days=WINDOW_DAYS))

    payload = report_to_dict(report)["findings"]["stream-usage"]

    assert "past_overspend_usd" not in payload
    assert payload["undercounted_usd"] is not None  # the figure IS published
    assert payload["accounting_note"]


def test_stream_usage_is_absent_from_the_second_selection_surface():
    """`COST_ANALYZERS` is an independent selector feeding the Review inbox.
    Absence from it is what keeps this finding off the money surfaces."""
    assert "stream-usage" not in COST_ANALYZERS


def test_the_undercount_never_reaches_the_recoverable_waste_rollup(db, cfg):
    """End to end: build the report, adapt it, roll it up. The gap figure is
    real and non-zero, and none of it appears in the recoverable total."""
    _seed_openai_app(db, truncated=3, complete=4)

    report = build_report(db, cfg, since=utcnow() - timedelta(days=WINDOW_DAYS))
    finding = report.findings["stream-usage"]
    proposals = cost_proposals_from_report(report, cfg, window_days=WINDOW_DAYS)
    rollup = past_overspend_rollup(proposals, window_days=WINDOW_DAYS)

    # The gap is real...
    assert finding.undercounted_usd is not None and finding.undercounted_usd > 0
    # ...and no proposal was minted from it, so the rollup cannot see it.
    assert all("stream-usage" not in (p.signature or "") for p in proposals)
    assert all(
        getattr(p, "analyzer", None) != "stream-usage" for p in proposals
    )
    assert rollup["past_overspend_usd"] != pytest.approx(finding.undercounted_usd)
    total = sum(p.past_overspend_usd or 0.0 for p in proposals)
    assert rollup["past_overspend_usd"] == pytest.approx(total)


def test_the_accounting_note_travels_with_the_figure(db, cfg):
    """It is a dataclass default so a renderer cannot drop it."""
    _seed_openai_app(db)

    finding = _finding(db, cfg)

    assert "not recoverable waste" in finding.accounting_note
    assert "does not make it smaller" in finding.accounting_note


def test_the_finding_round_trips_through_the_daemon_path(db, cfg):
    """Every consumer deserialises through report_from_dict; a finding absent
    from that table comes back empty and the card silently renders nothing."""
    from tokenjam.core.optimize import report_to_dict
    from tokenjam.core.optimize.runner import report_from_dict

    _seed_openai_app(db)
    report = build_report(db, cfg, since=utcnow() - timedelta(days=WINDOW_DAYS))

    restored = report_from_dict(report_to_dict(report))

    original = report.findings["stream-usage"]
    came_back = restored.findings["stream-usage"]
    assert isinstance(came_back, StreamUsageFinding)
    assert came_back.undercounted_usd == original.undercounted_usd
    assert came_back.call_sites[0].remediation_snippet == (
        original.call_sites[0].remediation_snippet
    )


# ---------------------------------------------------------------------------
# Token-less observations must not become a baseline of zeros
# ---------------------------------------------------------------------------

def _seed_tokenless_observation(db, *, agent_id, provider, model,
                                session_suffix, days_ago=1):
    """A completed stream whose observer never learned its token counts.

    The proxy's SSE tap watches the wire: it can see a stream run to
    completion and stamp the streaming signature while the provider's usage
    accounting stays entirely out of view. Those spans store NULL token
    columns — not zero — and this seeds exactly that shape.
    """
    _seed_stream(
        db, agent_id=agent_id, provider=provider, model=model,
        usage_reported=True, input_tokens=0, output_tokens=0,
        session_suffix=session_suffix, days_ago=days_ago,
    )
    db.conn.execute(
        "UPDATE spans SET input_tokens = NULL, output_tokens = NULL, "
        "cache_tokens = NULL, cache_write_tokens = NULL "
        "WHERE session_id = $1",
        [f"{agent_id}-{session_suffix}"],
    )


def test_tokenless_observations_never_enter_the_peer_median(db, cfg):
    """A NULL token count is unknown, not zero.

    Coercing it to zero lets a wire-level observation vote in the median, and
    enough of them drive the per-call baseline — and the whole under-count
    estimate — far below the truth while still rendering as a confident
    figure.
    """
    # Arrange: two real peers at 1000/300, swamped by six token-less ones.
    for i in range(2):
        _seed_stream(
            db, agent_id="chat-service", provider="openai", model="gpt-4o-mini",
            usage_reported=True, input_tokens=1000, output_tokens=300,
            session_suffix=f"c{i}", days_ago=i + 1,
        )
    for i in range(6):
        _seed_tokenless_observation(
            db, agent_id="chat-service", provider="openai",
            model="gpt-4o-mini", session_suffix=f"p{i}", days_ago=i + 1,
        )
    _seed_stream(
        db, agent_id="chat-service", provider="openai", model="gpt-4o-mini",
        usage_reported=False, session_suffix="t0", days_ago=1,
    )

    # Act
    finding = _finding(db, cfg)

    # Assert: the median is taken over the two real peers only.
    site = finding.call_sites[0]
    assert site.peer_output_tokens == 300
    assert site.peer_input_tokens == 1000
    assert site.complete_calls == 2
    assert finding.undercounted_usd is not None and finding.undercounted_usd > 0


def test_only_tokenless_peers_estimates_nothing_and_says_why(db, cfg):
    """With no real peer left, both estimate fields degrade together.

    Zero is the worst possible placeholder here: a `$0.00` under-count reads
    as "no blind spot", which is the one claim this analyzer exists to stop
    the product from making.
    """
    for i in range(4):
        _seed_tokenless_observation(
            db, agent_id="chat-service", provider="openai",
            model="gpt-4o-mini", session_suffix=f"p{i}", days_ago=i + 1,
        )
    _seed_stream(
        db, agent_id="chat-service", provider="openai", model="gpt-4o-mini",
        usage_reported=False, session_suffix="t0", days_ago=1,
    )

    finding = _finding(db, cfg)

    site = finding.call_sites[0]
    assert site.undercounted_usd is None
    assert site.undercounted_tokens is None
    assert site.peer_output_tokens is None
    assert finding.undercounted_usd is None
    assert finding.undercounted_tokens is None
    # The basis must distinguish "watched, learned nothing" from "watched
    # nothing" — they point at different remedies.
    assert "none carried token counts" in site.derivation
    assert "4 completed streams" in site.derivation
    assert finding.estimate_basis
