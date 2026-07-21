"""CLI rendering for `cache`'s A1/A2/A3 root-caused candidates
(`_render_cache_root_causes` / `_render_cache_efficacy` in cmd_optimize.py),
persona-gated on the `cache_control_snippet`.

This is the same gap `cache-recommend`'s renderer had (see
test_cache_recommend.py's persona tests): `cost_proposals._cache_to_proposals`
/ `_cache_uncached_to_proposals` / `_cache_thrash_to_proposals` /
`_cache_lookback_to_proposals` already gate the same underlying candidates at
the proposal layer via `_persona_gated_cache_fields`, but this CLI renderer
printed every snippet unconditionally regardless of persona. Fixture builders
mirror test_cache_root_cause_proposals.py so the two suites describe the same
candidates.
"""
from __future__ import annotations

from tokenjam.core.optimize.analyzers.cache_efficacy import (
    CacheEfficacyFinding,
    LookbackMissCandidate,
    ThrashAgentCandidate,
    UncachedAgentCandidate,
)
from tokenjam.core.optimize.cost_proposals import CACHE_NO_LEVER_TEXT


def _flat(out: str) -> str:
    """Collapse Rich's terminal-width line wrapping to a single line so a
    long fixed string can be matched by substring regardless of where the
    console happened to wrap it."""
    return " ".join(out.split())


def _uncached_candidate(agent_id="svc-uncached"):
    return UncachedAgentCandidate(
        agent_id=agent_id, provider="anthropic", model="claude-sonnet-5",
        calls=25, sessions=5, assumed_prefix_tokens=4000,
        cache_control_snippet='{"cache_control": {"type": "ephemeral"}}',
        estimated_recoverable_usd=1.5, estimated_recoverable_tokens=90000,
        estimate_basis="p25 prefix basis",
    )


def _thrash_candidate(agent_id="svc-thrash"):
    return ThrashAgentCandidate(
        agent_id=agent_id, provider="anthropic", model="claude-sonnet-5",
        calls=30, cache_write_tokens=50000, cache_read_tokens=10000,
        read_write_ratio=0.2, cause="ttl", inter_call_gap_p50_minutes=12.0,
        ttl_worth_it=True, ttl_breakeven_usd=0.4,
        cache_control_snippet='{"cache_control": {"type": "ephemeral", "ttl": "1h"}}',
        estimated_recoverable_usd=0.6, estimate_basis="thrash basis",
    )


def _lookback_candidate(agent_id="svc-lookback"):
    return LookbackMissCandidate(
        agent_id=agent_id, provider="anthropic", model="claude-sonnet-5",
        miss_count=4, avg_prior_turn_blocks=28.0,
        cache_control_snippet='{"cache_control": {"type": "ephemeral", "note": "intermediate breakpoint"}}',
        estimated_recoverable_usd=0.3, estimated_recoverable_tokens=12000,
        estimate_basis="lookback basis",
    )


# --------------------------------------------------------------------------- #
# Default (no persona threaded) keeps the pre-existing behaviour: actionable.
# --------------------------------------------------------------------------- #

def test_render_cache_root_causes_shows_snippet_by_default(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_root_causes

    finding = CacheEfficacyFinding(uncached_agents=[_uncached_candidate()])
    _render_cache_root_causes(finding, pricing_mode="api")
    out = capsys.readouterr().out

    assert "cache_control:" in out
    assert '"cache_control"' in out


# --------------------------------------------------------------------------- #
# Persona gate, one case per A1/A2/A3 group
# --------------------------------------------------------------------------- #

def test_render_cache_root_causes_claude_code_suppresses_uncached_snippet(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_root_causes

    finding = CacheEfficacyFinding(uncached_agents=[_uncached_candidate()])
    _render_cache_root_causes(finding, pricing_mode="api", persona="claude-code")
    out = _flat(capsys.readouterr().out)

    assert '"cache_control"' not in out
    assert CACHE_NO_LEVER_TEXT in out


def test_render_cache_root_causes_claude_code_suppresses_thrash_snippet(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_root_causes

    finding = CacheEfficacyFinding(thrash_agents=[_thrash_candidate()])
    _render_cache_root_causes(finding, pricing_mode="api", persona="claude-code")
    out = _flat(capsys.readouterr().out)

    assert '"cache_control"' not in out
    assert CACHE_NO_LEVER_TEXT in out
    # The diagnostic (cause) stays visible -- only the fix text/snippet is gated.
    assert "cause: calls land more than 5 min apart" in out


def test_render_cache_root_causes_claude_code_suppresses_lookback_snippet(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_root_causes

    finding = CacheEfficacyFinding(lookback_miss_agents=[_lookback_candidate()])
    _render_cache_root_causes(finding, pricing_mode="api", persona="claude-code")
    out = _flat(capsys.readouterr().out)

    assert '"cache_control"' not in out
    assert CACHE_NO_LEVER_TEXT in out
    # The evidence (miss count, dollar estimate) stays visible.
    assert "4 misses" in out
    assert "$0.30" in out or "0.30" in out


def test_render_cache_root_causes_unknown_persona_stays_actionable(capsys):
    """`unknown` (the CLI default when no persona is threaded) stays on the
    actionable branch -- the opposite grouping from `_persona_gated_write_
    fields`'s writes, and exactly the rule `_persona_gated_cache_fields`
    documents: for cache advice the risky direction is under-offering."""
    from tokenjam.cli.cmd_optimize import _render_cache_root_causes

    finding = CacheEfficacyFinding(
        uncached_agents=[_uncached_candidate()],
        thrash_agents=[_thrash_candidate()],
        lookback_miss_agents=[_lookback_candidate()],
    )
    _render_cache_root_causes(finding, pricing_mode="api", persona="unknown")
    out = capsys.readouterr().out

    assert out.count('"cache_control"') == 3
    assert CACHE_NO_LEVER_TEXT not in out


def test_render_cache_root_causes_mixed_persona_stays_actionable(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_root_causes

    finding = CacheEfficacyFinding(uncached_agents=[_uncached_candidate()])
    _render_cache_root_causes(finding, pricing_mode="api", persona="mixed")
    out = capsys.readouterr().out

    assert '"cache_control"' in out
    assert CACHE_NO_LEVER_TEXT not in out


# --------------------------------------------------------------------------- #
# Everything else about the renderer is preserved: structure, caps/trailers,
# the A2 "TTL not worth it" no-dollar-figure case, pricing-mode suppression.
# --------------------------------------------------------------------------- #

def test_render_cache_root_causes_preserves_group_structure_and_caps(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_root_causes

    uncached = [_uncached_candidate(f"svc-{i}") for i in range(7)]
    finding = CacheEfficacyFinding(uncached_agents=uncached)
    _render_cache_root_causes(finding, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert "7 agents never attempt caching" in out
    assert "svc-0" in out
    assert "svc-4" in out
    assert "svc-5" not in out
    assert "and 2 more" in out


def test_render_cache_root_causes_ttl_not_worth_it_shows_no_dollar_figure(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_root_causes

    c = ThrashAgentCandidate(
        agent_id="svc-thrash-nogo", provider="anthropic", model="claude-sonnet-5",
        calls=10, cache_write_tokens=5000, cache_read_tokens=1000,
        read_write_ratio=0.2, cause="ttl", inter_call_gap_p50_minutes=20.0,
        ttl_worth_it=False, ttl_breakeven_usd=None,
        cache_control_snippet='{"cache_control": {"type": "ephemeral"}}',
        estimated_recoverable_usd=None, estimate_basis="thrash basis",
    )
    finding = CacheEfficacyFinding(thrash_agents=[c])
    _render_cache_root_causes(finding, pricing_mode="api")
    out = _flat(capsys.readouterr().out)

    assert "caching not worth it" in out
    assert "no dollar figure: the recommended fix would not recover it" in out


def test_render_cache_root_causes_suppresses_dollars_off_api(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_root_causes

    finding = CacheEfficacyFinding(uncached_agents=[_uncached_candidate()])
    _render_cache_root_causes(finding, pricing_mode="subscription")
    out = _flat(capsys.readouterr().out)

    assert "$" not in out
    assert "doesn't bill per token" in out
    # The snippet itself still renders -- only the dollar figure is gated by
    # pricing_mode; persona is the only thing that gates the snippet.
    assert '"cache_control"' in out


def test_render_cache_efficacy_threads_persona_to_root_causes(capsys):
    """End-to-end through the top-level `cache` renderer, not just the
    root-cause helper directly."""
    from tokenjam.cli.cmd_optimize import _render_cache_efficacy

    finding = CacheEfficacyFinding(uncached_agents=[_uncached_candidate()])
    _render_cache_efficacy(finding, pricing_mode="api", persona="claude-code")
    out = _flat(capsys.readouterr().out)

    assert '"cache_control"' not in out
    assert CACHE_NO_LEVER_TEXT in out
