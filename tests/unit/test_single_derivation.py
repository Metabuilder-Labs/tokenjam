"""The generic guard: one test PER SHAPE, driven by the registry, not one
test per value.

See ``core/optimize/single_derivation.py`` for the registry itself and why
this module exists. Three shapes are tested here:

1. :data:`SEAMS` — parametrized: every symbol-reachable-from-one-module
   invariant. Adding a value here is a new line in the registry, never a new
   test function.
2. :data:`BESPOKE_SEAMS` — parametrized: seams real enough to pin but not
   expressible as a reachability check. This does not re-verify the
   invariant (the named test already does that); it verifies the NAMED TEST
   STILL EXISTS, so deleting or renaming it out from under the registry
   fails loudly instead of silently losing coverage.
3. :data:`AGGREGATE_FAMILIES` — parametrized: the aggregate-versus-parts
   shape, one family per finding that fans out into proposals. Driven by the
   registry the same way (1) is, plus a coverage check that the registry
   still names every adapter the live dispatcher runs — so a NEW fan-out
   adapter cannot ship unchecked, it fails at
   ``test_every_cost_adapter_belongs_to_a_registered_family``. Two families
   are pinned as ``xfail(strict=True)`` known gaps instead; see
   ``core/optimize/single_derivation.py``'s docstring on ``KNOWN_GAPS``.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import pytest

from tokenjam.core.optimize.single_derivation import (
    AGGREGATE_FAMILIES,
    BESPOKE_SEAMS,
    SEAMS,
    AggregateFamily,
    BespokeSeam,
    SingleSeam,
    check_bespoke_seam,
    offenders_for,
    unregistered_cost_adapters,
)


@pytest.mark.parametrize("seam", SEAMS, ids=[s.name for s in SEAMS])
def test_no_seam_gains_a_second_derivation(seam: SingleSeam) -> None:
    offenders = offenders_for(seam)
    assert not offenders, (
        f"{seam.name!r} is meant to have exactly one derivation, in "
        f"{sorted(seam.allowed_modules)}. Found {seam.symbol!r} reached "
        f"from outside that module at:\n  " + "\n  ".join(offenders) +
        f"\n\nWhy this must have one seam: {seam.reason}"
    )


@pytest.mark.parametrize("seam", BESPOKE_SEAMS, ids=[s.name for s in BESPOKE_SEAMS])
def test_every_bespoke_seam_still_has_a_live_test(seam: BespokeSeam) -> None:
    problem = check_bespoke_seam(seam)
    assert problem is None, (
        f"the registry names {seam.test_module}.{seam.test_name} as the "
        f"only guard for {seam.name!r}, but {problem}. This seam was not "
        "mechanized because: " + seam.reason_not_mechanized
    )


def test_the_registry_has_no_duplicate_seam_names() -> None:
    names = [s.name for s in SEAMS] + [s.name for s in BESPOKE_SEAMS]
    assert len(names) == len(set(names)), (
        "two registry entries share a name — pick a distinct name for each; "
        f"names were: {names}"
    )


def test_a_seam_reports_its_own_symbol_when_violated() -> None:
    """The walker actually WALKS, rather than trivially passing every entry.

    Only the shipped package is in scope (tests are always exempt — the same
    exemption the rollup and window-anchor tests rely on, since a test
    legitimately constructs the raw guarded symbol to pin its own
    behaviour). So the fixture here has to be a real call site INSIDE
    ``tokenjam/``: pointing at ``RateProfile`` with an empty allow-list
    must catch its own defining module using it.
    """
    fake = SingleSeam(
        name="test-only: RateProfile with nothing allowed",
        description="sanity check on the walker itself",
        symbol="RateProfile",
        kind="call",
        allowed_modules=frozenset(),
        reason="exercises the mechanism, not a real product invariant",
    )
    offenders = offenders_for(fake)
    assert any("rate_profile.py" in o for o in offenders)


# --------------------------------------------------------------------- #
# Aggregate versus parts
# --------------------------------------------------------------------- #
def test_the_cache_family_sums_exactly_to_the_findings_own_total() -> None:
    """The invariant :data:`KNOWN_GAPS` says `downsize` lacks, HOLDS for
    `cache` — proving the shape is enforceable, not just aspirational.

    `_cache_to_proposals` nets each generic row against whatever the more
    specific per-agent cards already claimed for the same (provider, model)
    (`_per_agent_cache_recoverable_by_model`), so the family's cards partition
    the finding's own total rather than double-claiming or dropping any of
    it.
    """
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        CacheEfficacyRow,
        UncachedAgentCandidate,
        estimate_cache_recoverable,
    )
    from tokenjam.core.optimize.cost_proposals import (
        _cache_to_proposals,
        _cache_uncached_to_proposals,
    )

    row = CacheEfficacyRow(
        provider="anthropic", model="claude-opus-4-7",
        input_tokens=100_000, cache_tokens=1_000, efficacy=0.01,
        support="full", flagged=True,
    )
    row_usd, row_tokens = estimate_cache_recoverable([row])
    assert row_usd is not None and row_tokens is not None

    agent = UncachedAgentCandidate(
        agent_id="worker-a", provider="anthropic", model="claude-opus-4-7",
        calls=5, sessions=2, assumed_prefix_tokens=1_000,
        past_overspend_usd=round(row_usd * 0.4, 6),
        past_overspend_tokens=int(row_tokens * 0.4),
        estimate_basis="a1 basis",
    )
    finding = CacheEfficacyFinding(
        rows=[row], flagged=[row],
        past_overspend_usd=row_usd, past_overspend_tokens=row_tokens,
        estimate_basis="cache basis", uncached_agents=[agent],
    )

    proposals = (
        _cache_to_proposals(finding, persona="unknown")
        + _cache_uncached_to_proposals(finding, persona="unknown")
    )
    total = sum(p.past_overspend_usd or 0.0 for p in proposals)
    assert total == pytest.approx(finding.past_overspend_usd, abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known gap, not yet closed — see KNOWN_GAPS in "
        "core/optimize/single_derivation.py. An unexpected pass here means "
        "the gap was closed: delete this xfail AND the matching account in "
        "KNOWN_GAPS together."
    ),
)
def test_the_downsize_per_agent_path_can_undercount_the_findings_own_total() -> None:
    """The gap `KNOWN_GAPS` names, reproduced directly.

    A finding whose aggregate `past_overspend_usd` includes a candidate on a
    model with NO pricing data (so `build_agent_price_rows` legitimately
    drops that group rather than guessing a rate) surfaces almost none of
    that money once the per-agent cards replace the window-wide card — and
    nothing on any card says so.

    Measured here: an aggregate of $998.00 surfaces as roughly $0.11 across
    the cards the Review inbox actually renders.
    """
    from tokenjam.core.optimize.analyzers.downsize_agents import build_agent_price_rows
    from tokenjam.core.optimize.analyzers.model_downgrade import DowngradeFinding
    from tokenjam.core.optimize.cost_proposals import _downsize_to_proposal
    from tokenjam.utils.time_parse import utcnow
    from datetime import timedelta

    priced_candidates = [{
        "session_id": f"s{i}", "agent_id": "worker-a", "provider": "anthropic",
        "model": "claude-opus-4-7", "alt_model": "claude-sonnet-5",
        "input_tokens": 1_000, "output_tokens": 500,
        "cache_tokens": 2_000, "cache_write_tokens": 100,
        "started_at": utcnow() - timedelta(days=1),
    } for i in range(10)]
    # An entire agent's worth of spend on a model `build_agent_price_rows`
    # cannot price, so its group is DROPPED rather than guessed at — that
    # drop is correct in isolation; the gap is that nothing downstream
    # discloses it.
    unpriceable_candidate = [{
        "session_id": "s-unpriced", "agent_id": "worker-b", "provider": "anthropic",
        "model": "totally-unpriced-model-xyz", "alt_model": "claude-sonnet-5",
        "input_tokens": 5_000_000, "output_tokens": 100,
        "cache_tokens": 0, "cache_write_tokens": 0,
        "started_at": utcnow() - timedelta(days=1),
    }]
    rows = build_agent_price_rows(
        priced_candidates + unpriceable_candidate, window_days=30.0,
    )
    assert {"worker-a", "worker-b"} - {r.agent_id for r in rows}, (
        "fixture assumption broken: expected build_agent_price_rows to drop "
        "the unpriceable agent's group"
    )

    finding = DowngradeFinding(
        candidate_sessions=11, total_sessions=20,
        actual_cost_usd=999.0, alternative_cost_usd=1.0,
        monthly_savings_usd=0.0, percent_of_sessions=55.0,
        examples=[], suggestions={"claude-opus-4-7": "claude-sonnet-5"},
        past_overspend_usd=998.0, percent_of_tokens=90.0,
        estimate_basis="downsize basis", per_agent=rows,
    )
    proposals = _downsize_to_proposal(finding, config=None, persona="unknown")
    offered = sum(p.past_overspend_usd or 0.0 for p in proposals)

    # The invariant this file wants: the inbox's cards should sum to (or at
    # least disclose falling short of) the aggregate the Dashboard publishes
    # for the same finding. It currently does neither.
    assert offered == pytest.approx(finding.past_overspend_usd, abs=1.0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known gap, not yet closed — see KNOWN_GAPS in "
        "core/optimize/single_derivation.py. An unexpected pass here means "
        "the gap was closed: delete this xfail AND the matching account in "
        "KNOWN_GAPS together."
    ),
)
def test_the_cache_family_drops_every_unflagged_rows_share() -> None:
    """The second gap `KNOWN_GAPS` names, reproduced directly.

    `CacheEfficacyFinding.past_overspend_usd` is
    `estimate_cache_recoverable(rows)` over EVERY (provider, model) row the
    window produced, charging each row the gap between its own efficacy and
    the 80% ceiling. The cards, though, come from `finding.flagged`, and a row
    is flagged only when its provider supports caching AND it carries at least
    `MIN_INPUT_TOKENS` of input AND its efficacy is under
    `EFFICACY_THRESHOLD` (0.30) — a far narrower test than "below the 0.80
    ceiling".

    So every row sitting between the flag threshold and the ceiling is real
    money on the Dashboard tile (`api/routes/cost.py::_collect_recoverable`
    reads `past_overspend_usd` straight off the finding) and no card at all in
    the Review inbox, with neither surface naming the difference.

    Closing it is a product decision, not a mechanical fix — card the
    unflagged rows (which contradicts the analyzer's own decision not to flag
    them), price the aggregate over flagged rows only (which lowers the
    published figure), or disclose the remainder on the cards. Hence a pin,
    not a patch.
    """
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        CacheEfficacyRow,
        estimate_cache_recoverable,
    )
    from tokenjam.core.optimize.cost_proposals import (
        _cache_lookback_to_proposals,
        _cache_thrash_to_proposals,
        _cache_to_proposals,
        _cache_uncached_to_proposals,
    )

    flagged_row = CacheEfficacyRow(
        provider="anthropic", model="claude-opus-4-7",
        input_tokens=100_000, cache_tokens=1_000, efficacy=0.01,
        support="full", flagged=True,
    )
    # 50% of input served from cache: comfortably ABOVE the 0.30 flag
    # threshold, so no card — and comfortably BELOW the 0.80 ceiling, so the
    # aggregate charges it for the 30-point difference anyway.
    unflagged_row = CacheEfficacyRow(
        provider="anthropic", model="claude-sonnet-5",
        input_tokens=5_000_000, cache_tokens=2_500_000, efficacy=0.50,
        support="full", flagged=False,
    )
    rows = [flagged_row, unflagged_row]
    total_usd, total_tokens = estimate_cache_recoverable(rows)
    assert total_usd is not None
    assert estimate_cache_recoverable([flagged_row])[0] != total_usd, (
        "fixture assumption broken: the unflagged row was expected to carry "
        "its own share of the finding's aggregate"
    )

    finding = CacheEfficacyFinding(
        rows=rows, flagged=[flagged_row],
        past_overspend_usd=total_usd, past_overspend_tokens=total_tokens,
        estimate_basis="cache basis",
    )
    proposals = (
        _cache_to_proposals(finding, persona="unknown")
        + _cache_uncached_to_proposals(finding, persona="unknown")
        + _cache_thrash_to_proposals(finding, persona="unknown")
        + _cache_lookback_to_proposals(finding, persona="unknown")
    )
    offered = sum(p.past_overspend_usd or 0.0 for p in proposals)
    assert offered == pytest.approx(total_usd, abs=1e-6)


# --------------------------------------------------------------------- #
# Aggregate versus parts, MECHANIZED — one property, every family
# --------------------------------------------------------------------- #
#: Any dollar figure appearing in a card's disclosure prose. The invariant
#: admits a shortfall the cards DISCLOSE (`_cache_recommend_to_proposals`
#: states the exact amount it subtracted to avoid double-counting a sibling
#: card), so the check has to be able to read that amount back out rather than
#: just noticing some prose exists.
_DOLLARS_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")


def _disclosed_amounts(proposals: list[Any]) -> set[float]:
    """Every dollar figure the family's cards name in a disclosure field."""
    found: set[float] = set()
    for p in proposals:
        for field in ("estimate_basis", "coverage_note"):
            for match in _DOLLARS_RE.finditer(str(getattr(p, field, "") or "")):
                found.add(float(match.group(1).replace(",", "")))
    return found


def _cache_fixture() -> tuple[Any, list[Any]]:
    """Two flagged rows plus one card of each per-agent root cause, all on a
    flagged row's own (provider, model) so the netting has something to net
    against — the regime the family is built to conserve in."""
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        CacheEfficacyRow,
        LookbackMissCandidate,
        ThrashAgentCandidate,
        UncachedAgentCandidate,
        estimate_cache_recoverable,
    )
    from tokenjam.core.optimize.cost_proposals import (
        _cache_lookback_to_proposals,
        _cache_thrash_to_proposals,
        _cache_to_proposals,
        _cache_uncached_to_proposals,
    )

    rows = [
        CacheEfficacyRow(
            provider="anthropic", model="claude-opus-4-7",
            input_tokens=400_000, cache_tokens=4_000, efficacy=0.01,
            support="full", flagged=True,
        ),
        CacheEfficacyRow(
            provider="anthropic", model="claude-sonnet-5",
            input_tokens=2_000_000, cache_tokens=100_000, efficacy=0.05,
            support="full", flagged=True,
        ),
    ]
    total_usd, total_tokens = estimate_cache_recoverable(rows)
    assert total_usd is not None and total_tokens is not None
    finding = CacheEfficacyFinding(
        rows=rows, flagged=list(rows),
        past_overspend_usd=total_usd, past_overspend_tokens=total_tokens,
        estimate_basis="cache basis",
        uncached_agents=[UncachedAgentCandidate(
            agent_id="worker-a", provider="anthropic", model="claude-opus-4-7",
            calls=40, sessions=8, assumed_prefix_tokens=2_000,
            past_overspend_usd=0.05, past_overspend_tokens=1_000,
            estimate_basis="a1 basis",
        )],
        thrash_agents=[ThrashAgentCandidate(
            agent_id="worker-b", provider="anthropic", model="claude-opus-4-7",
            calls=30, cache_write_tokens=9_000, cache_read_tokens=1_000,
            read_write_ratio=0.11, cause="ttl",
            inter_call_gap_p50_minutes=11.0, ttl_worth_it=True,
            past_overspend_usd=0.04, past_overspend_tokens=800,
            estimate_basis="a2 basis",
        )],
        lookback_miss_agents=[LookbackMissCandidate(
            agent_id="worker-c", provider="anthropic", model="claude-sonnet-5",
            miss_count=12, avg_prior_turn_blocks=34.0,
            past_overspend_usd=0.03, past_overspend_tokens=600,
            estimate_basis="a3 basis",
        )],
    )
    proposals = (
        _cache_to_proposals(finding, persona="unknown")
        + _cache_uncached_to_proposals(finding, persona="unknown")
        + _cache_thrash_to_proposals(finding, persona="unknown")
        + _cache_lookback_to_proposals(finding, persona="unknown")
    )
    return finding, proposals


def _cache_recommend_fixture() -> tuple[Any, list[Any]]:
    """Two prefix candidates, one of which collides with a sibling cache
    card's model — so the adapter's anti-double-count subtraction fires and
    the family's cards fall short of the finding's own total BY A DISCLOSED
    AMOUNT. This is the "or the difference is disclosed" half of the
    invariant, exercised rather than assumed."""
    from tokenjam.core.optimize.analyzers.cache_efficacy import (
        CacheEfficacyFinding,
        UncachedAgentCandidate,
    )
    from tokenjam.core.optimize.analyzers.cache_recommend import (
        CachePrefixCandidate,
        CacheRecommendFinding,
    )
    from tokenjam.core.optimize.cost_proposals import _cache_recommend_to_proposals

    candidates = [
        CachePrefixCandidate(
            prefix_hash="p1", sample_chars="You are a helpful assistant.",
            occurrences=30, avg_input_tokens=12_000.0,
            estimated_cacheable_tokens=8_000, model="claude-opus-4-7",
            past_overspend_usd=0.10, past_overspend_tokens=4_000,
        ),
        CachePrefixCandidate(
            prefix_hash="p2", sample_chars="Repo conventions follow.",
            occurrences=12, avg_input_tokens=9_000.0,
            estimated_cacheable_tokens=5_000, model="claude-sonnet-5",
            past_overspend_usd=0.05, past_overspend_tokens=2_000,
        ),
    ]
    finding = CacheRecommendFinding(
        enabled=True, candidates=candidates,
        past_overspend_usd=round(sum(c.past_overspend_usd or 0.0 for c in candidates), 6),
        past_overspend_tokens=sum(c.past_overspend_tokens or 0 for c in candidates),
        estimate_basis="cache-recommend basis",
    )
    sibling = CacheEfficacyFinding(
        rows=[], flagged=[], estimate_basis="cache basis",
        uncached_agents=[UncachedAgentCandidate(
            agent_id="worker-a", provider="anthropic", model="claude-opus-4-7",
            calls=40, sessions=8, assumed_prefix_tokens=2_000,
            past_overspend_usd=0.02, past_overspend_tokens=500,
            estimate_basis="a1 basis",
        )],
    )
    return finding, _cache_recommend_to_proposals(
        finding, sibling, persona="unknown",
    )


def _trim_fixture() -> tuple[Any, list[Any]]:
    from tokenjam.core.optimize.analyzers.prompt_bloat import (
        BloatPrompt,
        PromptBloatFinding,
    )
    from tokenjam.core.optimize.cost_proposals import _trim_to_proposals

    finding = PromptBloatFinding(
        enabled=True,
        per_prompt=[
            BloatPrompt(agent_id="svc-a", sample_chars="x", prompt_chars=8_000,
                        significant_chars=3_000, bloat_chars=5_000,
                        estimated_token_reduction=1_250),
            BloatPrompt(agent_id="svc-b", sample_chars="y", prompt_chars=6_000,
                        significant_chars=3_000, bloat_chars=3_000,
                        estimated_token_reduction=750),
        ],
        past_overspend_usd=0.8, past_overspend_tokens=2_000,
        estimate_basis="trim basis",
    )
    return finding, _trim_to_proposals(finding)


def _deadweight_fixture() -> tuple[Any, list[Any]]:
    from tokenjam.core.optimize.analyzers.deadweight import (
        DeadweightFinding,
        PluginComponent,
        PluginDeadweight,
        ServerDeadweight,
    )
    from tokenjam.core.optimize.cost_proposals import (
        _deadweight_plugin_to_proposals,
        _deadweight_to_proposals,
    )

    servers = [
        ServerDeadweight(
            name="apollo", scope="project", source="/repo/.mcp.json",
            sessions_present=10, invocations=0, deferred_sessions=0, unused=True,
            estimated_tax_tokens_per_session=25_000,
            estimated_tax_tokens_window=250_000,
            estimated_tax_usd_window=1.25,
            tax_construction="25,000 tok/session, cited estimate.",
            fix="Remove or project-scope apollo.", example_sessions=["s0"],
        ),
        ServerDeadweight(
            name="gdrive", scope="user", source="/home/u/.claude.json",
            sessions_present=6, invocations=0, deferred_sessions=2, unused=True,
            estimated_tax_tokens_per_session=25_000,
            estimated_tax_tokens_window=150_000,
            estimated_tax_usd_window=0.75,
            tax_construction="25,000 tok/session, cited estimate.",
            fix="Remove or project-scope gdrive.", example_sessions=["s1"],
        ),
    ]
    # A single unused plugin, so the family's two adapters both contribute at
    # least one card — the shape that would fan out if `_deadweight_plugin_
    # to_proposals` ever stopped being fanned out alongside the server one.
    plugins = [
        PluginDeadweight(
            name="stale-plugin@mkt", enabled=True, install_scope="user",
            resident=True, not_resident_because="",
            components=[
                PluginComponent(kind="skill", name="s0", resident_tokens=90, used=False),
            ],
            skills=1, agents=0, resident_tokens=90, sessions_present=10,
            unused=True, estimated_tax_tokens_window=900,
            estimated_tax_usd_window=0.01, priced_model="claude-sonnet-4-5",
            tax_construction="90 tok resident per call.",
            fix="Disable it.",
        ),
    ]
    finding = DeadweightFinding(
        sessions_scanned=10, configured_servers=2,
        servers=list(servers), unused_servers=list(servers),
        plugins=list(plugins), unused_plugins=list(plugins), plugins_resident=1,
        past_overspend_tokens=(
            sum(s.estimated_tax_tokens_window for s in servers)
            + sum(p.estimated_tax_tokens_window for p in plugins)
        ),
        past_overspend_usd=round(
            sum(s.estimated_tax_usd_window or 0.0 for s in servers)
            + sum(p.estimated_tax_usd_window or 0.0 for p in plugins), 6,
        ),
        estimate_basis="deadweight basis",
    )
    proposals = _deadweight_to_proposals(finding) + _deadweight_plugin_to_proposals(finding)
    return finding, proposals


def _script_fixture() -> tuple[Any, list[Any]]:
    from tokenjam.core.optimize.analyzers.workflow_restructure import (
        WorkflowCluster,
        WorkflowRestructureFinding,
    )
    from tokenjam.core.optimize.cost_proposals import _script_to_proposals

    clusters = [
        WorkflowCluster(
            signature=[{"tool": "bash", "args": ["command_string"]}], instances=25,
            avg_cost_usd=0.02, avg_duration_seconds=1.5,
            example_session_id="det-0", avg_tokens=500,
            total_cost_usd=0.5, total_tokens=12_500,
            example_session_ids=["det-0", "det-1"],
        ),
        WorkflowCluster(
            signature=[{"tool": "read", "args": ["path"]}], instances=14,
            avg_cost_usd=0.01, avg_duration_seconds=0.9,
            example_session_id="det-2", avg_tokens=300,
            total_cost_usd=0.14, total_tokens=4_200,
            example_session_ids=["det-2"],
        ),
    ]
    finding = WorkflowRestructureFinding(
        clusters=clusters, sessions_examined=39, degraded=False,
        past_overspend_usd=round(sum(c.total_cost_usd for c in clusters), 6),
        past_overspend_tokens=sum(c.total_tokens for c in clusters),
        estimate_basis="script basis",
    )
    return finding, _script_to_proposals(finding, persona="unknown")


def _reuse_fixture() -> tuple[Any, list[Any]]:
    from tokenjam.core.optimize.cost_proposals import _reuse_to_proposals
    from tokenjam.core.optimize.types import ReuseCluster, ReuseFinding

    clusters = [
        ReuseCluster(
            cluster_id="abc123456789", tool_signature=("bash", "read"),
            prompt_prefix_hash=None, repetitions=4, avg_planning_tokens=300,
            avg_planning_cost_usd=0.01, cache_reuse_recoverable_usd=0.03,
            script_replacement_recoverable_usd=0.04,
            cache_reuse_recoverable_tokens=900,
            script_replacement_recoverable_tokens=1_200,
            example_session_ids=["s1", "s2"], skeleton_session_id="s1",
        ),
        ReuseCluster(
            cluster_id="def987654321", tool_signature=("grep",),
            prompt_prefix_hash=None, repetitions=6, avg_planning_tokens=250,
            avg_planning_cost_usd=0.008, cache_reuse_recoverable_usd=0.02,
            script_replacement_recoverable_usd=0.03,
            cache_reuse_recoverable_tokens=600,
            script_replacement_recoverable_tokens=900,
            example_session_ids=["s3"], skeleton_session_id="s3",
        ),
    ]
    finding = ReuseFinding(
        clusters=clusters,
        past_overspend_usd=round(
            sum(c.cache_reuse_recoverable_usd for c in clusters), 6,
        ),
        past_overspend_tokens=sum(c.cache_reuse_recoverable_tokens for c in clusters),
        estimate_basis="reuse basis",
    )
    return finding, _reuse_to_proposals(finding, persona="unknown")


def _subagent_fixture() -> tuple[Any, list[Any]]:
    """TWO over-powered rows on two models — the shape that would fan out if
    this family ever stopped being one card."""
    from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
        SubagentRightsizingFinding,
        SubagentRow,
    )
    from tokenjam.core.optimize.cost_proposals import _subagent_to_proposals

    finding = SubagentRightsizingFinding(
        flagged=[
            SubagentRow(session_id="s1", sub_agent_id="sa0",
                        model="claude-opus-4-8", llm_calls=2, tool_calls=1,
                        input_tokens=60_000, output_tokens=500, cache_tokens=0,
                        cache_write_tokens=0, cost_usd=1.2, provider="anthropic",
                        flags=["over_powered"]),
            SubagentRow(session_id="s2", sub_agent_id="sa1",
                        model="claude-opus-4-7", llm_calls=3, tool_calls=2,
                        input_tokens=40_000, output_tokens=400, cache_tokens=0,
                        cache_write_tokens=0, cost_usd=0.8, provider="anthropic",
                        flags=["over_powered"]),
        ],
        percent_of_cost=0.66, flagged_cost_usd=2.0, subagent_cost_usd=2.5,
        past_overspend_usd=0.4, past_overspend_tokens=100_900,
    )
    return finding, _subagent_to_proposals(finding, None)


def _placement_fixture() -> tuple[Any, list[Any]]:
    from tokenjam.core.optimize.analyzers.batch_placement import (
        BatchCandidate,
        BatchPlacementFinding,
    )
    from tokenjam.core.optimize.cost_proposals import _placement_to_proposals

    candidates = [
        BatchCandidate(agent_id="nightly-a", sessions=12,
                       first_start="2026-07-01", last_start="2026-07-12",
                       median_gap_seconds=86_400.0, gap_cv=0.05,
                       cost_usd=6.0, tokens=900_000,
                       estimated_batch_saving_usd=3.0),
        BatchCandidate(agent_id="nightly-b", sessions=9,
                       first_start="2026-07-01", last_start="2026-07-12",
                       median_gap_seconds=43_200.0, gap_cv=0.08,
                       cost_usd=4.0, tokens=600_000,
                       estimated_batch_saving_usd=2.0),
    ]
    finding = BatchPlacementFinding(
        candidates=candidates, window_cost_usd=40.0, candidate_cost_usd=10.0,
        percent_of_window_cost=25.0,
        past_overspend_usd=5.0, past_overspend_tokens=1_500_000,
    )
    return finding, _placement_to_proposals(
        finding, pricing_mode="api", persona="unknown",
    )


def _verbosity_fixture() -> tuple[Any, list[Any]]:
    from tokenjam.core.optimize.analyzers.output_verbosity import VerbosityFinding
    from tokenjam.core.optimize.cost_proposals import _verbosity_to_proposals

    finding = VerbosityFinding(
        total_candidates=6, sessions_examined=40, cohorts_examined=3,
        past_overspend_usd=0.9, past_overspend_tokens=9_000,
        estimate_basis="verbosity basis", suggested_max_tokens=800,
    )
    return finding, _verbosity_to_proposals(finding, persona="unknown")


def _resend_fixture() -> tuple[Any, list[Any]]:
    from tokenjam.core.optimize.analyzers.context_resend import ResendFinding
    from tokenjam.core.optimize.cost_proposals import _resend_to_proposals

    finding = ResendFinding(
        sessions_examined=40, repeat_share=0.93, repeat_tokens=10_000,
        past_overspend_usd=703.78, past_overspend_tokens=1_400_000_000,
        estimate_basis="resend basis", fix_compaction="Run /compact.",
    )
    return finding, _resend_to_proposals(finding, persona="unknown")


def _summarize_fixture() -> tuple[Any, list[Any]]:
    """TWO candidate files — the shape that would fan out if this family ever
    stopped being one card."""
    from tokenjam.core.optimize.analyzers.summarize import (
        SummarizeCandidate,
        SummarizeFinding,
    )
    from tokenjam.core.optimize.cost_proposals import _summarize_to_proposals

    candidates = [
        SummarizeCandidate(
            path="/repo/CLAUDE.md", kind="prompt", scope="project",
            est_tokens_saved=4_000, total_chars=32_000, reduction_pct=40,
            sessions_loading=120, est_usd_saved=3.5,
            est_tokens_saved_window=480_000,
        ),
        SummarizeCandidate(
            path="/repo/AGENTS.md", kind="prompt", scope="project",
            est_tokens_saved=2_000, total_chars=16_000, reduction_pct=35,
            sessions_loading=120, est_usd_saved=1.5,
            est_tokens_saved_window=240_000,
        ),
    ]
    finding = SummarizeFinding(
        candidates=candidates, files=len(candidates),
        past_overspend_usd=5.0, past_overspend_tokens=720_000,
        estimate_basis="summarize basis", avg_reduction_pct=38,
        sessions_examined=120,
    )
    return finding, _summarize_to_proposals(finding)


#: One representative finding per registered family, already adapted. Keyed by
#: :attr:`AggregateFamily.name`, and checked for completeness below — a family
#: added to the registry with no fixture here fails loudly rather than being
#: quietly skipped.
#:
#: ``"downsize"`` is deliberately absent: it is the one ``verdict="gap"``
#: family, pinned by its own strict xfail above rather than run through the
#: conservation property.
_FAMILY_FIXTURES: dict[str, Callable[[], tuple[Any, list[Any]]]] = {
    "cache": _cache_fixture,
    "cache-recommend": _cache_recommend_fixture,
    "trim": _trim_fixture,
    "deadweight": _deadweight_fixture,
    "script": _script_fixture,
    "reuse": _reuse_fixture,
    "subagent": _subagent_fixture,
    "placement": _placement_fixture,
    "verbosity": _verbosity_fixture,
    "resend": _resend_fixture,
    "summarize": _summarize_fixture,
}

_CHECKED_FAMILIES = tuple(f for f in AGGREGATE_FAMILIES if f.verdict != "gap")


def test_every_cost_adapter_belongs_to_a_registered_family() -> None:
    """THE reason this is a registry and not a pile of assertions.

    ``_adapt_report``'s dispatch table is the live list of finding-to-proposal
    adapters. Wiring a new one in without registering it here means its cards
    are published beside a Dashboard aggregate with nothing checking the two
    agree — which is exactly how the ``downsize`` gap survived. Reading the
    dispatch table out of the source means the new adapter fails HERE, at the
    moment it is wired in, instead of never being noticed.
    """
    missing = unregistered_cost_adapters()
    assert not missing, (
        "these cost adapters are dispatched by _adapt_report but no "
        f"AGGREGATE_FAMILIES entry claims them: {sorted(missing)}. Add each to "
        "an existing family's `adapters`, or register a new family (with a "
        "fixture in _FAMILY_FIXTURES) stating whether its cards sum to the "
        "figure the Dashboard publishes for its finding."
    )


def test_every_registered_family_has_a_fixture() -> None:
    """The other direction: a registered family with no fixture would be
    silently un-exercised by the property test below — the same "invisible to
    the filter, not rejected by it" failure the seam walk avoids by naming its
    own offenders."""
    expected = {f.name for f in _CHECKED_FAMILIES}
    assert set(_FAMILY_FIXTURES) == expected, (
        "the fixture table and the registry disagree. Registered without a "
        f"fixture: {sorted(expected - set(_FAMILY_FIXTURES))}; fixture with no "
        f"registry entry: {sorted(set(_FAMILY_FIXTURES) - expected)}"
    )


@pytest.mark.parametrize(
    "family", _CHECKED_FAMILIES, ids=[f.name for f in _CHECKED_FAMILIES],
)
def test_a_families_cards_sum_to_the_figure_its_finding_publishes(
    family: AggregateFamily,
) -> None:
    """THE aggregate-versus-parts property, once, for every family.

    The Dashboard reads ``past_overspend_usd`` off each finding generically
    (``api/routes/cost.py::_collect_recoverable``); the Review inbox renders
    the cards. Those two surfaces publish the same analyzer's money, so the
    cards must sum to the finding's own figure — or name the difference in a
    disclosure field, which is the escape hatch
    ``_cache_recommend_to_proposals`` legitimately uses when it nets a card
    down to avoid double-counting a sibling.
    """
    finding, proposals = _FAMILY_FIXTURES[family.name]()
    total = float(getattr(finding, "past_overspend_usd", 0.0) or 0.0)
    parts = sum(p.past_overspend_usd or 0.0 for p in proposals)
    shortfall = round(total - parts, 6)
    if abs(shortfall) <= 1e-6:
        return
    assert any(
        abs(amount - abs(shortfall)) <= 1e-4
        for amount in _disclosed_amounts(proposals)
    ), (
        f"{family.name!r} publishes ${total:.6f} as its finding's total but "
        f"its {len(proposals)} card(s) sum to ${parts:.6f}, and no card's "
        f"estimate_basis or coverage_note names the ${abs(shortfall):.6f} "
        "difference. Either partition the total across the cards, disclose "
        "the remainder on them, or — if closing it is a product decision — "
        "move this family to verdict='gap' with a strict xfail pinning the "
        f"current behaviour.\n\nWhy this family was expected to hold: "
        f"{family.reason}"
    )


@pytest.mark.parametrize(
    "family",
    [f for f in _CHECKED_FAMILIES if f.verdict == "conserves"],
    ids=[f.name for f in _CHECKED_FAMILIES if f.verdict == "conserves"],
)
def test_a_conserving_familys_fixture_really_fans_one_finding_out(
    family: AggregateFamily,
) -> None:
    """The conservation property above is vacuous on a fixture that produces
    one card — one card trivially carries the whole figure. So each conserving
    family's fixture has to actually exercise many-from-one, and this is what
    stops the property quietly degrading into a tautology if a fixture is ever
    trimmed."""
    _, proposals = _FAMILY_FIXTURES[family.name]()
    assert len(proposals) > 1, (
        f"{family.name!r} is registered as a fan-out family but its fixture "
        f"produced {len(proposals)} card(s) — the conservation assertion is "
        "vacuous. Give the fixture enough rows/clusters/servers to emit "
        "several cards, or re-register the family as single-card."
    )


@pytest.mark.parametrize(
    "family",
    [f for f in _CHECKED_FAMILIES if f.verdict == "single-card"],
    ids=[f.name for f in _CHECKED_FAMILIES if f.verdict == "single-card"],
)
def test_a_single_card_family_still_emits_at_most_one_card(
    family: AggregateFamily,
) -> None:
    """A family is exempt from the many-from-one shape only while it really
    is one-from-one. Each fixture above carries SEVERAL rows precisely so that
    an adapter which starts fanning them out loses the exemption here rather
    than keeping it by an assumption nobody rechecked."""
    _, proposals = _FAMILY_FIXTURES[family.name]()
    assert len(proposals) <= 1, (
        f"{family.name!r} is registered as a single-card family but emitted "
        f"{len(proposals)} cards. Aggregate-versus-parts now applies to it: "
        "re-register it as verdict='conserves' (and make the parts sum) or "
        "as verdict='gap' with a strict xfail.\n\nWhy it was single-card: "
        f"{family.reason}"
    )


@pytest.mark.parametrize(
    "family",
    [f for f in AGGREGATE_FAMILIES if f.gap_pins],
    ids=[f.name for f in AGGREGATE_FAMILIES if f.gap_pins],
)
def test_every_registered_gap_still_has_a_live_pin(family: AggregateFamily) -> None:
    """Same discipline as :func:`test_every_bespoke_seam_still_has_a_live_test`:
    a gap the registry documents must have the strict xfail it names, so
    deleting the pin fails loudly instead of quietly retiring the record of an
    unfixed defect."""
    for name in family.gap_pins:
        fn = globals().get(name)
        assert callable(fn), (
            f"{family.name!r} names {name} as the pin for a known "
            f"aggregate-versus-parts gap, but this module no longer defines "
            f"it. The gap: {family.reason}"
        )
