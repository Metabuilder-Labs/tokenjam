"""Say the rule in the user's own terms — without saying more of them.

The published test for instruction text is that it be concrete enough to
verify. We were writing "keep files organized" while holding everything needed
to write "API handlers live in `src/api/handlers/`". Grounding closes that by
SUBSTITUTION, and the constraint that makes it hard is that the guidance is
"specific AND concise": a longer instruction file reduces adherence, so a
grounded rule that grew has made the fix worse while looking like an
improvement.
"""
from __future__ import annotations

import pytest

from tokenjam.core.fixes import FIX_CATALOG, fix_for, lint_catalog
from tokenjam.core.fixes.catalog import Substitution
from tokenjam.core.fixes.grounding import (
    MAX_GROUNDED_GROWTH,
    MAX_NAMED_ITEMS,
    Evidence,
    ground,
)

OFFLOAD = "resend.offload_to_subagent"


# --- it REPLACES, it does not append -----------------------------------------#

def test_a_grounded_rule_is_not_materially_longer_than_the_generic_one():
    """THE constraint. "Specific" must not mean "longer": the generic sentence
    plus three lines of observed detail is the failure mode, and it would
    reduce adherence — the very thing grounding exists to improve."""
    for key, record in FIX_CATALOG.items():
        if not record.grounding:
            continue
        grounded = ground(record, Evidence(
            repos=("optimize", "governor"), tools=("Read", "Grep"),
            models=("claude-opus-4-7",), agents=("Explore", "general-purpose"),
        ))
        assert len(grounded) <= len(record.text) * MAX_GROUNDED_GROWTH, key


def test_grounding_the_offload_rule_actually_shortens_it():
    """Not merely "not longer" — the concrete phrasing REPLACES a parenthesised
    enumeration, so it should come out shorter. If it ever does not, a
    substitution has started appending."""
    record = fix_for(OFFLOAD)
    assert record is not None
    grounded = ground(record, Evidence(repos=("optimize",), tools=("Read", "Grep")))
    assert len(grounded) < len(record.text)
    assert "`Read` and `Grep` sweeps you run in `optimize`" in grounded
    # And the vague span it replaced is gone, not sitting beside it.
    assert "broad file reads, log sweeps" not in grounded


def test_a_rendering_that_grew_is_discarded_for_the_generic_text():
    """The guard has to actually fire, or it is decoration."""
    from tokenjam.core.fixes.catalog import FixRecord

    record = FixRecord(
        key="test.bloat", text="Offload the work to a subagent when it is heavy.",
        delivery="claude_md_rule", personas=frozenset({"claude-code"}),
        analyzers=frozenset({"resend"}), answers="x",
        grounding=(Substitution(
            find="the work",
            template=(
                "the work, and note that this was observed across {repos} "
                "over many sessions with a great deal of additional detail "
                "appended here rather than replacing anything at all"
            ),
        ),),
    )
    assert ground(record, Evidence(repos=("a", "b", "c"))) == record.text


# --- graceful degrade ---------------------------------------------------------#

def test_no_evidence_yields_the_generic_text_unchanged():
    record = fix_for(OFFLOAD)
    assert record is not None
    assert ground(record, None) == record.text
    assert ground(record, Evidence()) == record.text


def test_an_unobserved_slot_skips_only_its_own_substitution():
    """Per-substitution, not all-or-nothing: a rule can be grounded in the
    directory it applies to even when the tool mix was not observed."""
    record = fix_for(OFFLOAD)
    assert record is not None
    only_repos = ground(record, Evidence(repos=("optimize",)))
    assert "`optimize`" in only_repos
    # The tools slot was unobserved, so that substitution was skipped and the
    # repo-only fallback carried it instead.
    assert "sweeps you run in" not in only_repos
    assert "context-heavy work you run in `optimize`" in only_repos


def test_specificity_is_never_invented():
    """An empty field means "not observed". Nothing fills it with a plausible
    directory the user does not have — which is also why this needs no model
    call: substitution cannot hallucinate."""
    record = fix_for(OFFLOAD)
    assert record is not None
    grounded = ground(record, Evidence(tools=("Read",)))     # no repos
    assert grounded == record.text


def test_a_long_list_is_capped_rather_than_enumerated():
    """Past a few names a list stops reading as concrete and starts costing
    length for nothing."""
    record = fix_for(OFFLOAD)
    assert record is not None
    many = tuple(f"repo{i}" for i in range(12))
    grounded = ground(record, Evidence(repos=many))
    assert grounded.count("`repo") == MAX_NAMED_ITEMS


# --- one record per fix, and the lints still hold ----------------------------#

def test_grounding_does_not_mint_a_record_per_finding():
    """Grounding is a RENDERING concern. A record per user or per finding would
    make the catalog and its duplicate lint meaningless overnight."""
    before = dict(FIX_CATALOG)
    record = fix_for(OFFLOAD)
    assert record is not None
    for evidence in (
        Evidence(repos=("a",)), Evidence(repos=("b",), tools=("Read",)),
        Evidence(repos=("c",), agents=("Explore",)),
    ):
        ground(record, evidence)
    assert dict(FIX_CATALOG) == before


def test_the_duplicate_lint_still_passes_over_grounded_output():
    """Grounding must not reintroduce redundancy by having two records name the
    same directory in the same words."""
    from tokenjam.core.fixes.lint import NEAR_DUPLICATE_OVERLAP, _overlap

    evidence = Evidence(repos=("optimize",), tools=("Read",), agents=("Explore",))
    grounded = {
        key: ground(record, evidence) for key, record in FIX_CATALOG.items()
    }
    keys = sorted(grounded)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            assert _overlap(grounded[left], grounded[right]) < NEAR_DUPLICATE_OVERLAP, (
                f"{left} vs {right}"
            )
    assert lint_catalog() == {}


def test_the_composed_artifact_stays_clean_when_grounded():
    """The composed check renders, so it has to render GROUNDED output — the
    two halves must not both end up naming the same thing."""
    from tokenjam.core.optimize.cost_proposals import compound_offload_fix

    evidence = Evidence(repos=("optimize",), tools=("Read",), agents=("Explore",))
    where = ground(fix_for(OFFLOAD), evidence)                     # type: ignore[arg-type]
    what_on = ground(fix_for("resend.rightsize_worker"), evidence)  # type: ignore[arg-type]
    composed = compound_offload_fix({}, where, what_on)
    assert composed.lower().count("definition file") == 1
    assert composed.count("`optimize`") <= 1


# --- honesty ------------------------------------------------------------------#

def test_grounding_never_strengthens_a_claim():
    """Naming what was observed is honest; asserting what the fix will achieve
    is not (Critical Rule 14)."""
    evidence = Evidence(
        repos=("optimize",), tools=("Read",), agents=("Explore",),
        models=("claude-opus-4-7",),
    )
    for record in FIX_CATALOG.values():
        lowered = ground(record, evidence).lower()
        for banned in ("will save", "guaranteed", "safe to", "would have worked"):
            assert banned not in lowered, record.key


@pytest.mark.parametrize("key", sorted(FIX_CATALOG))
def test_every_grounded_rendering_still_passes_the_fix_lint(key):
    """A grounded rule is still a rule and still has to hold every property —
    including that it does not re-license the behaviour its analyzer bills
    for, which a careless substitution could reintroduce."""
    from dataclasses import replace as dc_replace

    from tokenjam.core.fixes import lint_fix

    record = FIX_CATALOG[key]
    grounded = ground(record, Evidence(
        repos=("optimize",), tools=("Read", "Grep"),
        agents=("Explore",), models=("claude-opus-4-7",),
    ))
    assert lint_fix(dc_replace(record, text=grounded)) == []
