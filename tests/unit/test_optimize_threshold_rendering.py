"""`tj optimize` renders EFFECTIVE thresholds, not hardcoded literals.

An `[optimize]` config section (core.config.OptimizeConfig) made every
analyzer's sensitivity bar user-tunable, and each finding dataclass now
carries the threshold it actually applied. Before this fix, cmd_optimize.py's
renderers still baked the historical default numbers into their text as
string literals, so a user who lowered a threshold in tj.toml saw output
still claiming the old value -- a quiet lie, and one that also hid which
config key to turn. These tests construct each finding with a NON-default
threshold and assert the rendered text reflects the finding's own field.
"""
from __future__ import annotations

import pytest

from tokenjam.core.optimize.analyzers.batch_placement import BatchPlacementFinding
from tokenjam.core.optimize.analyzers.cache_efficacy import (
    CacheEfficacyFinding,
    CacheEfficacyRow,
)
from tokenjam.core.optimize.analyzers.cache_recommend import CacheRecommendFinding
from tokenjam.core.optimize.analyzers.output_verbosity import VerbosityFinding
from tokenjam.core.optimize.analyzers.prompt_bloat import PromptBloatFinding
from tokenjam.core.optimize.analyzers.relearn import RelearnFinding
from tokenjam.core.optimize.analyzers.subagent_rightsizing import (
    SubagentRightsizingFinding,
)
from tokenjam.core.optimize.analyzers.workflow_restructure import (
    WorkflowRestructureFinding,
)
from tokenjam.core.optimize.types import ReuseFinding


def _flat(out: str) -> str:
    """Collapse Rich's terminal-width line wrapping to a single line so a
    long fixed string can be matched by substring regardless of where the
    console happened to wrap it."""
    return " ".join(out.split())


def test_render_cache_efficacy_uses_findings_own_threshold(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_efficacy

    row = CacheEfficacyRow(
        provider="anthropic", model="claude-sonnet-4-6",
        input_tokens=60_000, cache_tokens=1_000, efficacy=0.10,
        support="full", flagged=True,
    )
    finding = CacheEfficacyFinding(
        rows=[row], flagged=[row],
        efficacy_threshold=0.55, min_input_tokens=50_000,
    )

    _render_cache_efficacy(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "55% efficacy threshold" in out
    assert "50.0k input tokens" in out
    # The historical hardcoded defaults must not leak through instead.
    assert "30% efficacy threshold" not in out
    assert "100.0k input tokens" not in out


def test_render_cache_recommend_empty_state_names_effective_threshold_and_key(capsys):
    from tokenjam.cli.cmd_optimize import _render_cache_recommend

    finding = CacheRecommendFinding(enabled=True, candidates=[], min_prefix_occurrences=7)

    _render_cache_recommend(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "≥7 Anthropic calls" in out
    assert "≥3 Anthropic calls" not in out
    assert "min_prefix_occurrences" in out


def test_render_workflow_restructure_empty_state_names_effective_threshold_and_key(capsys):
    from tokenjam.cli.cmd_optimize import _render_workflow_restructure

    finding = WorkflowRestructureFinding(
        clusters=[], sessions_examined=5, min_cluster_instances=42,
    )

    _render_workflow_restructure(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "≥42 identical signatures" in out
    assert "≥20 identical signatures" not in out
    assert "min_cluster_instances" in out


def test_render_reuse_empty_state_names_effective_threshold_and_key(capsys):
    from tokenjam.cli.cmd_optimize import _render_reuse

    finding = ReuseFinding(clusters=[], min_repetitions=9)

    _render_reuse(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "≥9 sessions sharing a skeleton" in out
    assert "≥3 sessions sharing a skeleton" not in out
    assert "min_reuse_repetitions" in out


def test_render_relearn_empty_state_names_effective_threshold_and_key(capsys):
    from tokenjam.cli.cmd_optimize import _render_relearn

    finding = RelearnFinding(clusters=[], sessions_scanned=12, min_sessions=11)

    _render_relearn(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "≥11 sessions sharing a signature" in out
    assert "≥3 sessions sharing a signature" not in out
    assert "min_recurring_sessions" in out


def test_render_verbosity_empty_state_names_effective_threshold_and_key(capsys):
    from tokenjam.cli.cmd_optimize import _render_verbosity

    finding = VerbosityFinding(
        candidates=[], sessions_examined=20, cohorts_examined=2,
        min_cohort_sessions=17,
    )

    _render_verbosity(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "≥17 sessions each" in out
    assert "min_cohort_sessions" in out


def test_render_subagent_empty_state_names_effective_threshold_and_key(capsys):
    from tokenjam.cli.cmd_optimize import _render_subagent

    finding = SubagentRightsizingFinding(
        total_subagents=3, sessions_with_subagents=2, flagged=[],
        min_flag_cost_usd=1.23,
    )

    _render_subagent(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "$1.23" in out
    assert "min_flag_cost_usd" in out


def test_render_placement_empty_state_names_effective_thresholds_and_keys(capsys):
    from tokenjam.cli.cmd_optimize import _render_placement

    finding = BatchPlacementFinding(
        candidates=[], min_sessions_for_cadence=8, min_group_cost_usd=2.5,
    )

    _render_placement(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "≥8 sessions" in out
    assert "$2.5" in out or "$2.50" in out
    assert "min_sessions_for_cadence" in out
    assert "min_group_cost_usd" in out


def test_render_prompt_bloat_empty_state_names_effective_threshold_and_key(capsys):
    """The significance check inverts the usual direction: a region counts as
    bloat when its score is BELOW the threshold, so RAISING the threshold (not
    lowering it) is what surfaces more bloat. The copy must say "raise", not
    the generic "lower" phrasing every other threshold uses."""
    from tokenjam.cli.cmd_optimize import _render_prompt_bloat

    finding = PromptBloatFinding(
        enabled=True, per_prompt=[], prompts_scored=4, prompts_skipped=1,
        significance_threshold=0.77,
    )

    _render_prompt_bloat(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "0.77" in out
    assert "trim_significance_threshold" in out
    assert "Raise" in out


# --------------------------------------------------------------------------- #
# Config-key bracket hygiene: `[optimize]` must render literally, not get
# swallowed as unrecognised Rich markup (the exact bug the deadweight-derived
# "Lower [optimize] <key>" phrasing hits if printed unescaped).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("render_fn_name,finding", [
    ("_render_cache_recommend", CacheRecommendFinding(enabled=True, candidates=[])),
    ("_render_workflow_restructure",
     WorkflowRestructureFinding(clusters=[], sessions_examined=1)),
    ("_render_reuse", ReuseFinding(clusters=[])),
    ("_render_relearn", RelearnFinding(clusters=[], sessions_scanned=1)),
    ("_render_verbosity",
     VerbosityFinding(candidates=[], sessions_examined=1, cohorts_examined=1)),
    ("_render_subagent",
     SubagentRightsizingFinding(total_subagents=1, sessions_with_subagents=1, flagged=[])),
    ("_render_placement", BatchPlacementFinding(candidates=[])),
])
def test_optimize_key_bracket_survives_rich_markup(render_fn_name, finding, capsys):
    import tokenjam.cli.cmd_optimize as cmd_optimize_mod

    render_fn = getattr(cmd_optimize_mod, render_fn_name)
    render_fn(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "[optimize]" in out, (
        f"{render_fn_name} swallowed the '[optimize]' config-key tag "
        f"(Rich markup escaping bug) -- got: {out!r}"
    )


def test_render_deadweight_notes_are_escaped(capsys):
    """Regression guard for the analyzer-supplied note text itself: the
    deadweight analyzer's own empty-state note names `[optimize]
    min_sessions_deadweight` in bracket form, which the renderer must escape
    before printing or Rich silently drops it."""
    from tokenjam.cli.cmd_optimize import _render_deadweight
    from tokenjam.core.optimize.analyzers.deadweight import DeadweightFinding

    finding = DeadweightFinding(
        sessions_scanned=5, configured_servers=1, servers=[], dead_servers=[],
        notes=[
            "No configured MCP server cleared the dead-weight bar "
            "(>= 10 sessions present, 0 invocations). Lower "
            "[optimize] min_sessions_deadweight in tj.toml to see servers "
            "present in fewer sessions.",
        ],
    )

    _render_deadweight(finding, pricing_mode="api", marker="①")
    out = _flat(capsys.readouterr().out)

    assert "[optimize] min_sessions_deadweight" in out
