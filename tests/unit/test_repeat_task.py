"""Unit tests for repeat-task clustering and the codification delta gate."""
from __future__ import annotations

import pytest

from tokenjam.core.optimize.repeat_task import (
    MIN_PRECISION_TO_PRICE,
    MIN_SESSIONS_PER_SIDE,
    TASK_STATEMENT_MATCH,
    TOOL_SHAPE_MATCH,
    SimilarityMethod,
    measure_codification_delta,
    model_mix_is_stable,
    normalize_task_statement,
    project_key,
    task_cluster_key,
    tool_profile,
    tool_profile_cosine,
    tool_shape_signature,
)

# --- normalization -----------------------------------------------------------


def test_masks_the_per_invocation_variables_out_of_a_templated_prompt():
    a = "You are a ticket-resolution worker. Item #12. Worktree /Users/x/code/wt/ticket-12."
    b = "You are a ticket-resolution worker. Item #99. Worktree /Users/x/code/wt/ticket-99."

    assert normalize_task_statement(a) == normalize_task_statement(b)


def test_keeps_genuinely_different_templates_apart():
    worker = "You are a ticket-resolution worker spawned by the batch harness."
    supervisor = "SUPERVISOR-REVIEW. You are a short-lived batch supervisor."

    assert normalize_task_statement(worker) != normalize_task_statement(supervisor)


def test_collapses_whitespace_and_case_so_invocation_wrapping_does_not_split_a_cluster():
    assert normalize_task_statement("Run  THE\n task") == normalize_task_statement("run the task")


def test_masks_commit_shas_that_would_otherwise_make_every_run_unique():
    assert normalize_task_statement("head is 761a2654f1f") == normalize_task_statement(
        "head is a1b2c3d4e5f"
    )


def test_normalizing_an_empty_prompt_does_not_raise():
    assert normalize_task_statement("") == ""


# --- project scoping ---------------------------------------------------------


def test_per_ticket_worktrees_collapse_onto_the_parent_project():
    assert project_key("-Users-dev-code-myrepo-wt-ticket-357") == "-users-dev-code-myrepo"
    assert project_key("-Users-dev-code-myrepo.wt-ticket-357") == "-users-dev-code-myrepo"


def test_the_same_template_against_two_projects_is_two_clusters():
    prompt = "You are a ticket-resolution worker."

    assert task_cluster_key(prompt, "repo-a") != task_cluster_key(prompt, "repo-b")


def test_the_same_template_in_two_worktrees_of_one_project_is_one_cluster():
    prompt = "You are a ticket-resolution worker. Item #12."
    other = "You are a ticket-resolution worker. Item #99."

    assert task_cluster_key(prompt, "myrepo-wt-ticket-12") == task_cluster_key(
        other, "myrepo-wt-ticket-99"
    )


# --- span-only shape --------------------------------------------------------


def test_shape_signature_is_the_ordered_opening_of_the_tool_sequence():
    tools = ["Bash", "Read", "Edit", "Bash", "Bash", "Read", "Write", "Bash", "Grep"]

    assert tool_shape_signature(tools, k=3) == ("Bash", "Read", "Edit")
    assert len(tool_shape_signature(tools)) == 8


def test_tool_profile_cosine_is_one_for_identical_mixes_and_zero_for_disjoint_ones():
    a = tool_profile(["Bash", "Bash", "Read"])
    b = tool_profile(["Bash", "Bash", "Read"])
    c = tool_profile(["WebFetch", "WebSearch"])

    assert tool_profile_cosine(a, b) == pytest.approx(1.0)
    assert tool_profile_cosine(a, c) == pytest.approx(0.0)


def test_cosine_of_an_empty_profile_is_zero_rather_than_a_division_error():
    assert tool_profile_cosine({}, tool_profile(["Bash"])) == 0.0


# --- the pricing gate --------------------------------------------------------


def test_the_measured_tool_shape_method_is_not_allowed_to_price():
    assert TOOL_SHAPE_MATCH.precision < MIN_PRECISION_TO_PRICE
    assert not TOOL_SHAPE_MATCH.may_price


def test_the_task_statement_method_is_allowed_to_price():
    assert TASK_STATEMENT_MATCH.may_price


def test_a_low_precision_method_is_refused_even_with_a_huge_sample():
    before = [10.0] * 500
    after = [1.0] * 500

    result = measure_codification_delta(before, after, method=TOOL_SHAPE_MATCH)

    assert result.verdict == "refused"
    assert result.ratio is None
    assert "below the" in result.basis
    assert result.avoidable_usd == 0.0


def test_too_few_sessions_on_one_side_is_refused_not_guessed():
    before = [10.0] * MIN_SESSIONS_PER_SIDE
    after = [1.0] * (MIN_SESSIONS_PER_SIDE - 1)

    result = measure_codification_delta(before, after, method=TASK_STATEMENT_MATCH)

    assert result.verdict == "refused"
    assert "per-side minimum" in result.basis


# --- the verdicts ------------------------------------------------------------


def _spread(center: float, n: int) -> list[float]:
    """A tight, deterministic spread around ``center``."""
    return [center * (0.9 + 0.2 * (i / max(1, n - 1))) for i in range(n)]


def test_a_clean_halving_reads_as_cheaper_and_prices_the_conservative_end():
    before = _spread(10.0, 40)
    after = _spread(5.0, 40)

    result = measure_codification_delta(before, after, method=TASK_STATEMENT_MATCH)

    assert result.verdict == "cheaper"
    assert result.is_priceable
    assert result.ci_high is not None and result.ci_high < 1.0
    # priced off the conservative end of the interval, so never above the
    # naive point-estimate saving
    naive = (result.median_before - result.median_after) * result.n_before
    assert 0.0 < result.avoidable_usd <= naive


def test_a_clean_doubling_reads_as_dearer_and_is_never_priced():
    before = _spread(5.0, 40)
    after = _spread(10.0, 40)

    result = measure_codification_delta(before, after, method=TASK_STATEMENT_MATCH)

    assert result.verdict == "dearer"
    assert not result.is_priceable
    assert result.avoidable_usd == 0.0


def test_a_big_point_estimate_inside_a_wide_interval_reads_as_null():
    # The measured shape of this corpus: within one template x one project the
    # per-session cost CV is ~0.7-1.2, so a 2x point estimate is not a finding.
    before = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 0.7, 1.1, 2.2, 6.0]
    after = [0.4, 0.9, 1.2, 2.5, 4.0, 7.0, 9.0, 0.6, 1.3, 3.1, 5.5, 11.0]

    result = measure_codification_delta(before, after, method=TASK_STATEMENT_MATCH)

    assert result.verdict == "null"
    assert not result.is_priceable
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low <= 1.0 <= result.ci_high


def test_the_interval_is_reproducible_across_runs():
    before = _spread(10.0, 30)
    after = _spread(6.0, 30)

    a = measure_codification_delta(before, after, method=TASK_STATEMENT_MATCH)
    b = measure_codification_delta(before, after, method=TASK_STATEMENT_MATCH)

    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_the_basis_string_names_the_similarity_method_and_the_interval():
    result = measure_codification_delta(
        _spread(10.0, 30), _spread(5.0, 30), method=TASK_STATEMENT_MATCH
    )

    assert "task-statement-exact" in result.basis
    assert "95% bootstrap CI" in result.basis
    assert "precision" in result.basis


def test_a_zero_cost_before_side_yields_null_rather_than_a_division_error():
    result = measure_codification_delta(
        [0.0] * 20, [1.0] * 20, method=TASK_STATEMENT_MATCH
    )

    assert result.verdict == "null"
    assert result.ratio is None


# --- the model-mix confound --------------------------------------------------


def test_a_model_routing_change_under_the_comparison_is_flagged_as_unstable():
    # The measured failure mode: opus-dominant before, mixed after. Every
    # apparently-significant delta on the validation corpus was this.
    assert not model_mix_is_stable([0.99] * 10, [0.35] * 10)


def test_a_comparison_with_one_model_dominant_on_both_sides_is_stable():
    assert model_mix_is_stable([0.98] * 10, [0.96] * 10)


def test_an_empty_side_is_not_stable():
    assert not model_mix_is_stable([], [0.99])


# --- the bar itself ----------------------------------------------------------


def test_the_precision_bar_is_high_because_the_claim_is_causal():
    assert MIN_PRECISION_TO_PRICE >= 0.90


def test_a_custom_method_at_exactly_the_bar_may_price():
    method = SimilarityMethod(
        name="hypothetical", precision=MIN_PRECISION_TO_PRICE, recall=0.5, basis="test"
    )

    assert method.may_price
