"""Which route to a smaller instruction file a candidate actually wants.

Compression is one of four routes to Anthropic's published size target and the
only one that trades adherence for tokens (Critical Rule 26, gate 3). These
tests pin that the offer never presents it as the default for an instruction
file, that the diagnosis measures prose SHAPE rather than rule necessity, and
that it is withheld rather than guessed.
"""
from __future__ import annotations

from tokenjam.core.summarize import detect, load_semantics, route

# A file that is long because RULES ACCUMULATED: many discrete directives.
RULE_HEAVY = "\n".join(
    f"- Never skip step {i} when you touch the {i} subsystem; run its check first."
    for i in range(60)
)
# A file that is long because the PROSE IS PADDED: running explanation.
PROSE_HEAVY = "\n\n".join(
    "This section explains at some length how the system came to work the way "
    "it does, walking through the history and the reasoning and the various "
    "considerations that were weighed along the way before a decision was "
    f"finally reached about approach number {i}." for i in range(20)
)


def test_rule_heavy_file_is_a_prune_candidate_not_a_compression_one():
    """Squeezing words here shortens each surviving rule rather than removing
    any — the quality tax gate 3 forbids."""
    advice = route.recommend_route(text=RULE_HEAVY, load_class=load_semantics.ALWAYS)

    assert advice.route == route.ROUTE_PRUNE
    assert advice.directive_share > 0.6
    assert "rules ACCUMULATED" in advice.advice
    assert "Prune or scope it first" in advice.advice


def test_prose_heavy_file_is_a_genuine_compression_candidate():
    advice = route.recommend_route(text=PROSE_HEAVY, load_class=load_semantics.ALWAYS)

    assert advice.route == route.ROUTE_COMPRESS
    assert advice.directive_share < 0.3
    assert "genuine compression candidate" in advice.advice


def test_a_mixed_file_names_both_routes_rather_than_picking_one():
    advice = route.recommend_route(
        text=RULE_HEAVY + "\n\n" + PROSE_HEAVY, load_class=load_semantics.ALWAYS)

    assert advice.route == route.ROUTE_MIXED
    assert "Compress the explanation, prune or scope the rules" in advice.advice


def test_diagnosis_is_withheld_rather_than_guessed_on_a_tiny_file():
    """A wrong diagnosis is worse than none, so a file with no readable shape
    gets no route recommendation — and says that is why."""
    advice = route.recommend_route(text="- one rule\n", load_class=load_semantics.ALWAYS)

    assert advice.route == route.ROUTE_UNDIAGNOSED
    assert "no route is recommended for it rather than one being guessed" in advice.advice


def test_the_diagnosis_never_claims_to_judge_which_rules_are_needed():
    """It measures FORM. Presenting that as a verdict on necessity would be
    exactly the wrong guess, so every shaped verdict carries the disclaimer."""
    for text in (RULE_HEAVY, PROSE_HEAVY, RULE_HEAVY + "\n\n" + PROSE_HEAVY):
        advice = route.recommend_route(text=text, load_class=load_semantics.ALWAYS)
        assert "not of which rules earn their place" in advice.advice
        assert "only you can answer the removal test" in advice.advice


def test_every_instruction_file_is_told_the_three_cheaper_routes():
    advice = route.recommend_route(text=RULE_HEAVY, load_class=load_semantics.ALWAYS)

    assert route.PRUNE_TEST_QUOTE in advice.advice          # prune
    assert route.PATH_SCOPE_QUOTE in advice.advice          # path-scope
    assert route.HOOK_QUOTE in advice.advice                # escalate to a hook
    assert "does not write them for you" in advice.advice   # ...and we write none


def test_the_quality_tax_is_stated_and_only_tokens_are_claimed():
    """Compression is the only route that costs specificity. That must be said,
    and the adherence benefit must never be dressed up as a saving."""
    advice = route.recommend_route(text=RULE_HEAVY, load_class=load_semantics.ALWAYS)

    assert "only one of these routes that trades adherence for tokens" in advice.advice
    assert route.SPECIFICITY_QUOTE in advice.advice
    assert "Only the token reduction is ever claimed as a saving" in advice.advice
    # And the honest statement of what is and is not checked.
    assert "not checked by anything automatic" in advice.advice
    assert "It does not read the prose" in advice.advice


def test_a_rule_that_already_declares_paths_is_not_told_to_path_scope_itself():
    scoped = '---\npaths:\n  - "src/**/*.ts"\n---\n\n' + RULE_HEAVY

    assert load_semantics.has_paths_scope(scoped) is True
    advice = route.recommend_route(text=scoped, load_class=load_semantics.ALWAYS)

    assert advice.already_path_scoped is True
    assert "already declares `paths:` frontmatter" in advice.advice
    assert route.PATH_SCOPE_QUOTE not in advice.advice       # would be telling
    assert route.PRUNE_TEST_QUOTE in advice.advice           # pruning still applies


def test_paths_named_only_in_the_body_is_not_frontmatter():
    assert load_semantics.has_paths_scope("# Rules\n\npaths: are discussed below\n") is False
    assert load_semantics.has_paths_scope(RULE_HEAVY) is False


def test_an_on_demand_file_keeps_compression_and_says_the_target_is_ours():
    """The published size guidance is about a file that loads every session. A
    skill body is not one, so the instruction-file argument is not restated."""
    advice = route.recommend_route(text=RULE_HEAVY, load_class=load_semantics.SKILL)

    assert advice.route == route.ROUTE_NOT_INSTRUCTION
    assert "does not apply to it" in advice.advice
    assert "tokenjam's own" in advice.advice
    assert route.PRUNE_TEST_QUOTE not in advice.advice


# --------------------------------------------------------------------------- #
# The shape measurement itself.
# --------------------------------------------------------------------------- #

def test_prose_shape_splits_directives_from_running_prose():
    shape = detect.prose_shape("- rule one here\n- rule two here\n\nA paragraph of prose.\n")

    assert shape.directive_units == 2
    assert shape.paragraph_units == 1
    assert shape.units == 3
    assert shape.directive_words + shape.paragraph_words == shape.prose_words


def test_a_wrapped_bullet_stays_one_directive():
    """A continuation line is part of its rule, not a separate paragraph —
    otherwise every wrapped bullet would drag the file toward 'prose-heavy'."""
    shape = detect.prose_shape("- a rule that runs on\n  and wraps to a second line\n")

    assert shape.directive_units == 1
    assert shape.paragraph_units == 0
    assert shape.directive_words == shape.prose_words


def test_numbered_items_and_headings_count_as_directives():
    shape = detect.prose_shape("# Heading here\n\n1. first rule\n\n2) second rule\n")

    assert shape.directive_units == 3
    assert shape.paragraph_units == 0


def test_shape_ignores_protected_structure():
    """Structure is never compressible, so it must not sway the diagnosis."""
    bare = detect.prose_shape(PROSE_HEAVY)
    with_code = detect.prose_shape(PROSE_HEAVY + "\n```\n- not a rule\n- nor this\n```\n")

    assert with_code.directive_units == bare.directive_units
    assert with_code.prose_words == bare.prose_words


def test_an_append_only_dated_log_wants_expiry_not_compression():
    """A `learnings.md` is long because entries accumulated over time. Its own
    stated remedy is expiry — promote what proved durable, delete what went
    stale — and compressing it rewrites history while keeping every stale entry."""
    log = "\n\n".join(
        f"## 2026-0{i % 9 + 1}-14 — something we learned\n\nA paragraph about what "
        f"happened and what it implied for the next session, number {i}."
        for i in range(12))

    advice = route.recommend_route(text=log, load_class=load_semantics.ALWAYS)

    assert advice.route == route.ROUTE_EXPIRE
    assert "append-only LOG" in advice.advice
    assert "promote what proved durable and delete what went stale" in advice.advice \
        or "promote\nwhat proved durable" in advice.advice or "expiry" in advice.advice
    assert "close to useless here" in advice.advice


def test_expiry_is_decided_before_the_directive_share_can_mislabel_it():
    """A dated log is a log whether its entries are bullets or paragraphs, so
    the directive share cannot answer this one."""
    bulleted_log = "\n\n".join(
        f"- 2026-07-{i + 10} learned something specific about the deploy path today"
        for i in range(12))

    shape = detect.prose_shape(bulleted_log)
    assert shape.directive_share > 0.6                 # would read as rule-heavy
    advice = route.recommend_route(text=bulleted_log, load_class=load_semantics.ALWAYS)
    assert advice.route == route.ROUTE_EXPIRE          # ...but it is a log


def test_an_instruction_file_with_no_dates_is_never_called_a_log():
    advice = route.recommend_route(text=RULE_HEAVY, load_class=load_semantics.ALWAYS)
    assert advice.route == route.ROUTE_PRUNE
    assert detect.prose_shape(RULE_HEAVY).dated_units == 0
