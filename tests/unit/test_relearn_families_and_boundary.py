"""Failure-family coverage, the not-a-relearn filter, and the relearn/resend line.

Every error string in here is REAL wording taken from the local Claude Code
corpus (2026-07-26), not invented — a family that matches a plausible-looking
paraphrase but not the harness's actual message is the exact failure mode the
`read_offset_malformed` ordering comment already documents.
"""
from __future__ import annotations

import pytest

from tokenjam.core.rulewrite.delivery import DELIVERY_KINDS

from tokenjam.core.optimize.analyzers.relearn import (
    _KNOWN_FAMILIES,
    classify_known_family,
    is_user_decline,
)
from tokenjam.core.optimize.analyzers.resend_tail import RELEARN_RESEND_BOUNDARY

# (tool_name, real error text, expected family)
_REAL_CORPUS_SAMPLES = [
    ("gen_ai.llm.call", "prompt is too long: 201070 tokens > 200000 maximum",
     "context_overflow"),
    ("Bash", "Exit code 143\nCommand timed out after 2m 0s", "bash_timeout"),
    ("Bash", "Exit code 128\nfatal: a branch named 'ticket-234' already exists",
     "git_branch_exists"),
    ("Bash", "This Bash command contains multiple operations. The following part "
             "requires approval: git push", "bash_chained_approval"),
    ("Bash", "Exit code 1\nuv not found\n/Users/x/.local/bin/uv", "command_not_found"),
    ("Bash", "Exit code 127\n(eval):1: command not found: aws", "command_not_found"),
    ("Read", "EISDIR: illegal operation on a directory, read '/Users/x/skills'",
     "read_directory"),
    ("Read", "File content (34827 tokens) exceeds maximum allowed tokens (25000). "
             "Use offset and limit parameters to read specific portions.",
     "read_too_large"),
    ("Edit", "Found 2 matches of the string to replace, but replace_all is false. "
             "To replace all occurrences, set replace_all to true.",
     "edit_ambiguous_match"),
    ("gen_ai.llm.call", "The following domains are not accessible to our user "
                        "agent: ['reddit.com']. Read more: https://example",
     "webfetch_domain_blocked"),
    ("WebFetch", "Claude Code is unable to fetch from web.archive.org",
     "webfetch_domain_blocked"),
    ("ExitPlanMode", "Error: No such tool available: ExitPlanMode. ExitPlanMode "
                     "exists but is not enabled in this context.",
     "deferred_tool_cold"),
    # Regressions on the families that already existed, so the new entries'
    # ORDERING cannot silently steal their evidence.
    ("Read", "File does not exist. Note: your current working directory is /tmp",
     "cwd_confusion"),
    ("Edit", "File has not been read yet. Read it first before writing to it.",
     "edit_before_read"),
    ("Edit", "String to replace not found in file.", "edit_string_not_found"),
    ("Edit", "File has been modified since read, either by the user or by a linter.",
     "stale_read_race"),
]


@pytest.mark.parametrize("tool,text,expected", _REAL_CORPUS_SAMPLES)
def test_real_corpus_wording_lands_in_the_intended_family(tool, text, expected):
    assert classify_known_family(tool, text, "") == expected


def test_every_family_declares_a_renderable_delivery():
    """Every family declares a mechanism this build can actually render.
    Inverted from a numeric range check that admitted 4 and 5, levels no
    build ever produced — the range passed while asserting nothing about
    whether the artifact could be written at all."""
    for family in _KNOWN_FAMILIES:
        assert family["delivery"] in DELIVERY_KINDS, family["key"]


def test_family_keys_are_unique():
    keys = [f["key"] for f in _KNOWN_FAMILIES]
    assert len(keys) == len(set(keys))


# --- The not-a-relearn filter -------------------------------------------------

def test_a_sibling_cancelled_by_another_calls_failure_is_not_a_relearn():
    """Claude Code marks the SIBLINGS of a parallel tool block as errored when
    one member fails. The sibling never ran, so it taught the agent nothing and
    forced no recovery turn of its own — the one recovery turn belongs to the
    member that actually failed and is already counted. On the local corpus
    this was the 4th-largest cluster (127 sessions) and pure double-count."""
    assert is_user_decline(
        "Cancelled: parallel tool call Bash(cd /Users/x/code && make test) errored"
    )


def test_a_bare_permission_prompt_is_not_a_relearn():
    """The user's own allowlist. The same command succeeds once approved, so no
    rule written into any agent-file surface changes the outcome."""
    assert is_user_decline("This command requires approval")


def test_the_chained_command_variant_IS_a_relearn():
    """The negative lookahead that separates the two: chaining is what forced
    the prompt here, and un-chaining removes it — so it must survive the filter
    and reach its own family."""
    text = ("This Bash command contains multiple operations. The following part "
            "requires approval: git push")
    assert not is_user_decline(text)
    assert classify_known_family("Bash", text, "") == "bash_chained_approval"


def test_a_real_failure_is_never_filtered_out():
    for _tool, text, _fam in _REAL_CORPUS_SAMPLES:
        assert not is_user_decline(text), text[:60]


# --- The relearn / resend boundary --------------------------------------------

def test_the_boundary_is_stated_once_and_quoted_by_both_analyzers():
    """Two analyzers price nearly the same physical tokens (a coding turn is
    ~99.8% re-read context), so an unstated boundary lets both claim them.
    Neither side may paraphrase it — both quote the one constant, so the two
    cards cannot drift into two different accounts of the same line.

    relearn used to quote it from `ESTIMATE_BASIS` (the retired forward-claim
    basis); the one canonical dollar field means the quote now lives on
    `PAST_OVERSPEND_BASIS`, relearn's only basis string."""
    import tokenjam.core.optimize.analyzers.relearn as relearn_mod
    from tokenjam.core.optimize.analyzers.context_resend import RESEND_ESTIMATE_BASIS
    from tokenjam.core.optimize.analyzers.relearn import PAST_OVERSPEND_BASIS

    assert RELEARN_RESEND_BOUNDARY in PAST_OVERSPEND_BASIS
    assert RELEARN_RESEND_BOUNDARY in RESEND_ESTIMATE_BASIS
    assert not hasattr(relearn_mod, "ESTIMATE_BASIS")


def test_the_boundary_names_the_counterfactual_not_the_token_class():
    """The line is 'did this call have to happen', NOT 'is this token re-sent'.
    If it ever gets rewritten as a token-class split, relearn's claim collapses
    to the couple of fresh tokens a retry turn introduces, which would value
    eliminating a 96k-token rejected call at a fraction of a cent."""
    lowered = RELEARN_RESEND_BOUNDARY.lower()
    assert "should not have happened" in lowered
    assert "size of calls that had to" in lowered


# --- The recovery arc ---------------------------------------------------------
# relearn used to price one forced turn per failure. Measured on a real corpus
# the median failure costs 2 turns and some families over 4, so the flat charge
# halved every figure. These pin the measurement and, more importantly, the
# de-duplication that keeps a burst of potholes from billing one turn twice.

def _ep(session="s", tool="Bash"):
    from tokenjam.core.optimize.analyzers.relearn import FailureEpisode

    return FailureEpisode(
        session_id=session, repo="r", ts=None, tool_name=tool, label="",
        error_text="boom", kind="act", is_retry=False, depth=0,
    )


def test_a_failure_costs_the_turns_until_its_tool_works_again():
    from tokenjam.core.optimize.analyzers.relearn import _stamp_detour_turns

    fail = _ep()
    # fail at step 0; Bash succeeds again at step 3 => a 3-turn detour.
    _stamp_detour_turns([(0, "Bash", fail), (3, "Bash", None)], total_steps=10)
    assert fail.detour_turns == 3.0


def test_a_clean_one_retry_recovery_still_costs_one_turn():
    """The floor case, and the value the whole analyzer used to assume."""
    from tokenjam.core.optimize.analyzers.relearn import _stamp_detour_turns

    fail = _ep()
    _stamp_detour_turns([(0, "Bash", fail), (1, "Bash", None)], total_steps=5)
    assert fail.detour_turns == 1.0


def test_overlapping_recovery_arcs_never_bill_the_same_turn_twice():
    """CLAUDE.md rule 27's double-count, inside one analyzer instead of between
    two. Two failures one step apart both recover at step 3, so turns 2 and 3
    are claimed by both and must be split, not charged twice."""
    from tokenjam.core.optimize.analyzers.relearn import _stamp_detour_turns

    first, second = _ep(), _ep()
    _stamp_detour_turns(
        [(0, "Bash", first), (1, "Bash", second), (3, "Bash", None)],
        total_steps=10,
    )
    # Naive sum would be 3 + 2 = 5 turns for a stretch that only contains 3.
    assert first.detour_turns + second.detour_turns == pytest.approx(3.0)
    assert first.detour_turns > second.detour_turns   # the earlier one owns step 1 alone


def test_a_shared_turn_is_split_evenly_and_never_lost():
    from tokenjam.core.optimize.analyzers.relearn import _stamp_detour_turns

    a, b, c = _ep(), _ep(), _ep()
    _stamp_detour_turns(
        [(0, "Bash", a), (0, "Bash", b), (0, "Bash", c), (1, "Bash", None)],
        total_steps=5,
    )
    # One turn of recovery, three failures claiming it: a third each, summing
    # to the one turn that was actually spent (the field is rounded to 4dp, so
    # the split is exact only to that precision — deliberately, since a stored
    # figure that renders is worth more than a repeating decimal).
    for episode in (a, b, c):
        assert episode.detour_turns == pytest.approx(1 / 3, abs=1e-4)
    assert sum(e.detour_turns for e in (a, b, c)) == pytest.approx(1.0, abs=1e-3)


def test_a_different_tool_succeeding_does_not_end_the_arc():
    """Recovery means THIS tool working again. A Read succeeding in the middle
    of a Bash flail is not the Bash pothole being resolved."""
    from tokenjam.core.optimize.analyzers.relearn import _stamp_detour_turns

    fail = _ep(tool="Bash")
    _stamp_detour_turns(
        [(0, "Bash", fail), (1, "Read", None), (2, "Read", None), (4, "Bash", None)],
        total_steps=10,
    )
    assert fail.detour_turns == 4.0


def test_a_failure_that_never_recovers_is_charged_the_sessions_own_median():
    """The agent abandoned the approach, which is not cheaper than retrying it.
    Its length is unmeasurable, so it borrows this session's measured median
    rather than the scan cap (which would let the unknown case dominate) or a
    global constant (which would not be measured from this user's data)."""
    from tokenjam.core.optimize.analyzers.relearn import _stamp_detour_turns

    measured, abandoned = _ep(), _ep()
    _stamp_detour_turns(
        [(0, "Bash", measured), (2, "Bash", None), (50, "Grep", abandoned)],
        total_steps=60,
    )
    assert measured.detour_turns == 2.0
    assert abandoned.detour_turns == 2.0        # the session's own median, not 40


def test_recovery_beyond_the_scan_window_does_not_count_as_recovery():
    from tokenjam.core.optimize.analyzers.relearn import (
        MAX_RECOVERY_SCAN_STEPS,
        _stamp_detour_turns,
    )

    fail = _ep()
    far = MAX_RECOVERY_SCAN_STEPS + 5
    _stamp_detour_turns([(0, "Bash", fail), (far, "Bash", None)], total_steps=far + 5)
    # No median to borrow => the conservative 1.0 floor, never `far`.
    assert fail.detour_turns == 1.0


def test_an_unmeasurable_lane_prices_at_the_old_one_turn_floor():
    """The archive and OTel lanes carry spans, not ordered steps. They must fall
    back to the previous assumption rather than inherit a multiplier measured on
    a different lane."""
    episode = _ep()
    assert episode.detour_turns is None
    assert (episode.detour_turns or 1.0) == 1.0
