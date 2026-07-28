"""Reading records written before the delivery mechanism replaced the rung.

The stakes are what make this file worth having: these records are on real
users' disks — cached proposals and the applied-fix ledger — and the value read
out of them decides which artifact gets written to which real path. Guessing
wrong writes the wrong file; guessing wrong about a hook additionally prices a
real user's fix wrongly, because the two hook mechanisms have opposite cost
behaviour.

So the assertions come in two halves, and the second is the one that matters:
what the number CAN establish, and what it cannot.
"""
from __future__ import annotations

import pytest

from tokenjam.core.rulewrite.delivery import DELIVERY_KINDS, resolve_delivery
from tokenjam.core.rulewrite.kinds import (
    DELIVERY_CLAUDE_MD_RULE,
    DELIVERY_EXECUTING_HOOK,
    DELIVERY_INJECTING_HOOK,
    DELIVERY_SKILL,
    UNRESOLVED_DELIVERY,
)
from tokenjam.core.rulewrite.legacy import delivery_from_legacy_record
from tokenjam.core.rulewrite.types import RuleWrite, RuleWriteRefused, StagedRuleWrite


def _wired(event: str) -> dict:
    """A ledger record whose staged settings patch names a hook event."""
    return {"rung": 3, "enforcement": {"patch": {"hooks": {event: []}}}}


# --- What the old record CAN establish ----------------------------------------

@pytest.mark.parametrize("record,expected", [
    ({"rung": 1}, DELIVERY_CLAUDE_MD_RULE),
    ({"rung": 2}, DELIVERY_SKILL),
    # Some records carry the word rather than (or beside) the number.
    ({"kind": "note"}, DELIVERY_CLAUDE_MD_RULE),
    ({"kind": "skill"}, DELIVERY_SKILL),
])
def test_the_unambiguous_half_of_the_ladder_maps_cleanly(record, expected):
    """Rungs 1 and 2 each only ever meant one artifact, so they migrate."""
    assert delivery_from_legacy_record(record) == expected


@pytest.mark.parametrize("event,expected", [
    ("PreToolUse", DELIVERY_EXECUTING_HOOK),
    ("PostToolUseFailure", DELIVERY_INJECTING_HOOK),
])
def test_a_hook_is_resolved_from_the_wiring_that_was_actually_written(event, expected):
    """The strongest evidence a record carries: the settings patch staged
    beside the hook. A `PreToolUse` guard executes and injects nothing; a
    `PostToolUseFailure` hook exists to hand the model `additionalContext`."""
    assert delivery_from_legacy_record(_wired(event)) == expected


@pytest.mark.parametrize("family,expected", [
    ("sleep_chain", DELIVERY_EXECUTING_HOOK),
    ("cwd_confusion", DELIVERY_INJECTING_HOOK),
    ("stale_read_race", DELIVERY_INJECTING_HOOK),
    ("edit_string_not_found", DELIVERY_INJECTING_HOOK),
])
def test_an_unwired_hook_falls_back_to_its_familys_own_matcher(family, expected):
    """Second-strongest evidence, read off the live guard/reactive tables
    rather than a copy — so a family that changes shape cannot leave this
    asserting the old one."""
    assert delivery_from_legacy_record({"rung": 3, "family_key": family}) == expected


# --- What it CANNOT, and must refuse to invent --------------------------------

@pytest.mark.parametrize("record", [
    pytest.param({"rung": 3}, id="hook-with-no-evidence-at-all"),
    pytest.param({"rung": 3, "family_key": "mystery"}, id="hook-of-an-unknown-family"),
    pytest.param({"rung": 4}, id="wrapper-no-build-ever-produced"),
    pytest.param({"rung": 5}, id="config-no-build-ever-produced"),
])
def test_a_record_that_cannot_say_what_it_wrote_is_marked_unresolvable(record):
    """The whole point of the shim.

    Rung 3 covered two mechanisms with OPPOSITE cost behaviour, so mapping
    every legacy 3 onto one of them would misprice a real user's applied fix in
    one direction or the other — silently, and permanently. An honest gap beats
    a confident wrong answer.
    """
    assert delivery_from_legacy_record(record) == UNRESOLVED_DELIVERY


def test_no_artifact_information_at_all_is_not_the_same_as_unresolvable():
    """Empty means "names no mechanism", so the caller's own default applies.
    Collapsing it into UNRESOLVED would refuse ordinary new rules; collapsing
    UNRESOLVED into it would silently render an ambiguous legacy record as a
    CLAUDE.md block into whatever path it happened to carry."""
    assert delivery_from_legacy_record({}) == ""


def test_an_unresolvable_record_refuses_to_render_rather_than_defaulting():
    """The refusal is the safety property. `resolve_delivery` defaults an EMPTY
    name to the markdown rule, so the sentinel has to be a name no registry
    holds — otherwise an ambiguous record would take that default and write a
    real file."""
    assert UNRESOLVED_DELIVERY not in DELIVERY_KINDS
    with pytest.raises(RuleWriteRefused, match="older version"):
        resolve_delivery(UNRESOLVED_DELIVERY)


# --- The read path callers actually use ---------------------------------------

def test_a_cached_proposal_written_by_an_older_build_deserializes():
    """`RuleWrite.from_dict` reads the cache on disk. A record carrying a rung
    and no delivery must come back with a mechanism, not blow up and not
    silently blank."""
    rule = RuleWrite.from_dict({
        "signature": "relearn:x", "analyzer": "relearn", "title": "T",
        "artifact_text": "some fix text", "rung": 2,
    })
    assert rule.delivery == DELIVERY_SKILL
    # And nothing writes the number back.
    assert "rung" not in rule.to_dict()


def test_a_staged_entry_written_by_an_older_build_deserializes():
    staged = StagedRuleWrite.from_dict({
        "signature": "relearn:x", "path": "/tmp/CLAUDE.md", "scope": "project",
        "title": "T", "analyzer": "relearn", "rung": 1,
        "source_sha256": "abc", "rendered": "body", "diff": "",
    })
    assert staged.delivery == DELIVERY_CLAUDE_MD_RULE
    assert "rung" not in staged.to_dict()


def test_a_current_record_names_its_mechanism_and_needs_no_migration():
    """The applied-fix ledger records the artifact under `kind`, and for a
    record written by this build that word already IS a mechanism name."""
    for name in DELIVERY_KINDS:
        assert delivery_from_legacy_record({"kind": name}) == name


# --- The concept stays gone ---------------------------------------------------

def test_no_source_file_reintroduces_the_ladder_rung():
    """A standing guard, not a one-off cleanup.

    Two competing names for one concept is the defect this change removed: the
    rung had three separate constant definitions, and two modules disagreed
    about whether rung 1 meant the FILE an artifact lands in or the artifact's
    SHAPE. A single re-introduced `rung=` is how that starts again.

    Deliberately scoped to `rung` as an IDENTIFIER, not to the word: the
    analysis-window code legitimately calls its 24h/7d/30d steps "rungs" in
    prose, and a guard that cannot tell a live symbol from an English word
    teaches people to add exceptions. `rulewrite/legacy.py` is exempt because
    reading the old field is precisely its job.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "tokenjam"
    exempt = {root / "core" / "rulewrite" / "legacy.py"}
    # An assignment, a keyword argument, a dict key, or an attribute access.
    symbol = re.compile(r"""(\brung\s*[:=]|\.rung\b|["']rung["']|\bRUNG_)""")

    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path in exempt:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if symbol.search(line):
                offenders.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "the intervention-ladder rung is back as a live symbol; use a delivery "
        "mechanism from core/rulewrite/kinds.py instead:\n  "
        + "\n  ".join(offenders)
    )
