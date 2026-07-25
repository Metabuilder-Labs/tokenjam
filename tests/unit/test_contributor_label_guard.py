"""
Unit tests for `.github/scripts/contributor_label_guard.py` — the closing-
keyword parser and the remove/restore decision logic behind the outside-
contribution label guard. Pure logic only, no network calls.

The module under test lives outside the `tokenjam` package (it's a workflow
helper script, following the `archive_traffic.py` convention), so it's
imported by path rather than as a normal package import.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import contributor_label_guard as guard  # noqa: E402

REPO = "Metabuilder-Labs/tokenjam"


# ---------------------------------------------------------------------------
# parse_closing_issue_refs / resolve_pr_closing_issues
# ---------------------------------------------------------------------------


def test_parses_single_closing_keyword():
    assert guard.parse_closing_issue_refs("Closes #560", REPO) == [560]


def test_parses_multiple_distinct_closing_keywords():
    text = "Fixes #1\nAlso resolves #2\nAnd closed #3"
    assert guard.parse_closing_issue_refs(text, REPO) == [1, 2, 3]


def test_parses_multiple_refs_in_one_body():
    text = "This closes #10 and also fixes #20, plus resolves #30."
    assert guard.parse_closing_issue_refs(text, REPO) == [10, 20, 30]


def test_dedupes_repeated_refs():
    text = "Closes #5. Also closes #5 again, and fixes #5."
    assert guard.parse_closing_issue_refs(text, REPO) == [5]


def test_all_closing_keyword_verb_forms():
    for keyword in (
        "close",
        "closes",
        "closed",
        "fix",
        "fixes",
        "fixed",
        "resolve",
        "resolves",
        "resolved",
    ):
        text = f"{keyword} #42"
        assert guard.parse_closing_issue_refs(text, REPO) == [42], keyword


def test_mixed_case_keyword_and_repo():
    assert guard.parse_closing_issue_refs("FIXES #7", REPO) == [7]
    assert (
        guard.parse_closing_issue_refs(f"Closes {REPO.upper()}#9", REPO) == [9]
    )


def test_owner_repo_form_matching_this_repo_is_included():
    text = f"Closes {REPO}#560"
    assert guard.parse_closing_issue_refs(text, REPO) == [560]


def test_owner_repo_form_for_a_different_repo_is_ignored():
    text = "Closes some-other-org/other-repo#560"
    assert guard.parse_closing_issue_refs(text, REPO) == []


def test_owner_repo_and_bare_refs_together():
    text = f"Closes {REPO}#1 and fixes other-org/other-repo#2 and resolves #3"
    assert guard.parse_closing_issue_refs(text, REPO) == [1, 3]


def test_no_reference_pr_body_returns_empty():
    text = "This PR refactors the parser module for clarity, no issue attached."
    assert guard.parse_closing_issue_refs(text, REPO) == []


def test_empty_and_none_like_text_returns_empty():
    assert guard.parse_closing_issue_refs("", REPO) == []


def test_keyword_like_text_without_adjacent_reference_does_not_match():
    # "fixes" appears but is not immediately followed by a '#N' reference —
    # GitHub does not treat this as a closing reference and neither should we.
    text = "This PR fixes the typo in the README. See #123 for background."
    assert guard.parse_closing_issue_refs(text, REPO) == []


def test_keyword_substring_inside_another_word_does_not_match():
    # "enclosed" contains "close" but is not the keyword "close" itself.
    text = "The enclosed #99 diagram explains the flow."
    assert guard.parse_closing_issue_refs(text, REPO) == []


def test_resolve_pr_closing_issues_unions_title_and_body():
    title = "Fix #1: handle empty input"
    body = "Also closes #2 while I'm in here."
    assert guard.resolve_pr_closing_issues(title, body, REPO) == [1, 2]


def test_resolve_pr_closing_issues_dedupes_across_title_and_body():
    title = "Fixes #5"
    body = "This closes #5 as well."
    assert guard.resolve_pr_closing_issues(title, body, REPO) == [5]


def test_resolve_pr_closing_issues_handles_missing_body():
    assert guard.resolve_pr_closing_issues("Fixes #5", "", REPO) == [5]
    assert guard.resolve_pr_closing_issues("No refs here", "", REPO) == []


# ---------------------------------------------------------------------------
# labels_to_remove
# ---------------------------------------------------------------------------


def test_labels_to_remove_returns_present_guarded_labels():
    current = ["good first issue", "help wanted", "bug"]
    assert guard.labels_to_remove(current) == ["good first issue", "help wanted"]


def test_labels_to_remove_only_returns_labels_actually_present():
    current = ["help wanted", "bug"]
    assert guard.labels_to_remove(current) == ["help wanted"]


def test_labels_to_remove_empty_when_no_guarded_labels_present():
    assert guard.labels_to_remove(["bug", "documentation"]) == []


def test_labels_to_remove_empty_current_labels():
    assert guard.labels_to_remove([]) == []


# ---------------------------------------------------------------------------
# labels_to_restore — the exact-restore requirement
# ---------------------------------------------------------------------------


def _unlabeled_event(label: str, actor: str = guard.DEFAULT_ACTOR_LOGIN, at: str = "2026-07-25T00:00:00Z"):
    return {
        "event": "unlabeled",
        "label": {"name": label},
        "actor": {"login": actor},
        "created_at": at,
    }


def _labeled_event(label: str, actor: str = "some-human", at: str = "2026-07-25T00:00:00Z"):
    return {
        "event": "labeled",
        "label": {"name": label},
        "actor": {"login": actor},
        "created_at": at,
    }


def test_restores_only_the_label_the_workflow_removed():
    # Workflow removed "good first issue" but "help wanted" was never on the
    # issue in the first place — restore must not add it.
    events = [_unlabeled_event("good first issue")]
    current = []  # neither label present right now
    assert guard.labels_to_restore(events, current) == ["good first issue"]


def test_does_not_restore_a_label_still_present():
    events = [_unlabeled_event("good first issue")]
    current = ["good first issue"]  # somehow already back (e.g. human re-added)
    assert guard.labels_to_restore(events, current) == []


def test_does_not_restore_when_most_recent_event_is_a_human_relabel():
    events = [
        _unlabeled_event("good first issue", at="2026-07-25T00:00:00Z"),
        _labeled_event("good first issue", actor="a-maintainer", at="2026-07-25T01:00:00Z"),
    ]
    # Label was re-added by a human after our removal, then somehow removed
    # again from `current` externally — the most recent event is a human
    # `labeled`, so this workflow must not claim credit / restore it.
    assert guard.labels_to_restore(events, current_label_names=[]) == []


def test_does_not_restore_when_unlabeled_by_a_different_actor():
    events = [_unlabeled_event("good first issue", actor="a-maintainer")]
    assert guard.labels_to_restore(events, current_label_names=[]) == []


def test_restores_both_labels_when_both_were_removed():
    events = [
        _unlabeled_event("good first issue"),
        _unlabeled_event("help wanted"),
    ]
    assert guard.labels_to_restore(events, current_label_names=[]) == [
        "good first issue",
        "help wanted",
    ]


def test_restore_ignores_malformed_events():
    events = [
        {"event": "unlabeled"},  # missing "label"
        {"event": "unlabeled", "label": {}},  # missing "name"
        {"event": "commented"},  # irrelevant event type
        _unlabeled_event("good first issue"),
    ]
    assert guard.labels_to_restore(events, current_label_names=[]) == ["good first issue"]


def test_restore_no_events_for_label_means_nothing_to_restore():
    assert guard.labels_to_restore([], current_label_names=[]) == []


def test_restore_actor_login_is_configurable():
    events = [_unlabeled_event("good first issue", actor="custom-bot[bot]")]
    assert (
        guard.labels_to_restore(events, current_label_names=[], actor_login="custom-bot[bot]")
        == ["good first issue"]
    )
    assert guard.labels_to_restore(events, current_label_names=[]) == []


# ---------------------------------------------------------------------------
# _is_pull_request
# ---------------------------------------------------------------------------


def test_is_pull_request_true_when_payload_has_pull_request_key():
    assert guard._is_pull_request({"number": 1, "pull_request": {"url": "..."}}) is True


def test_is_pull_request_false_for_plain_issue():
    assert guard._is_pull_request({"number": 1}) is False
