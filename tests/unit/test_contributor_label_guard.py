"""
Unit tests for `.github/scripts/contributor_label_guard.py` — the closing-
keyword parser, remove/restore decision logic, and mocked-HTTP coverage for
the GitHub API I/O layer behind the outside-contribution label guard.

The module under test lives outside the `tokenjam` package (it's a workflow
helper script, following the `archive_traffic.py` convention), so it's
imported by path rather than as a normal package import.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import urllib.error

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


# ---------------------------------------------------------------------------
# Network I/O — mocked HTTP (issue #608)
# ---------------------------------------------------------------------------


def _mock_http_ok(payload: object | None, status: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload).encode() if payload is not None else b""
    return mock_resp


def _mock_http_error(
    url: str, status: int, payload: object | None = None, headers: dict[str, str] | None = None
) -> urllib.error.HTTPError:
    raw = json.dumps(payload).encode() if payload is not None else b""
    return urllib.error.HTTPError(url, status, "error", headers or {}, io.BytesIO(raw))


def test_api_request_success_returns_status_and_payload(monkeypatch):
    issue = {"number": 1, "state": "open", "labels": []}

    def fake_urlopen(req, timeout=30):
        assert req.get_method() == "GET"
        assert req.full_url.endswith("/repos/Metabuilder-Labs/tokenjam/issues/1")
        return _mock_http_ok(issue)

    monkeypatch.setattr(guard.urllib.request, "urlopen", fake_urlopen)
    status, payload = guard._api_request("GET", "/repos/Metabuilder-Labs/tokenjam/issues/1", "token")
    assert status == 200
    assert payload == issue


@pytest.mark.parametrize("status", [401, 403, 429])
def test_api_request_http_auth_and_rate_limit_errors_do_not_raise(monkeypatch, status):
    headers = {"X-RateLimit-Remaining": "0"} if status == 429 else {}

    def fake_urlopen(req, timeout=30):
        raise _mock_http_error(req.full_url, status, {"message": "nope"}, headers)

    monkeypatch.setattr(guard.urllib.request, "urlopen", fake_urlopen)
    returned_status, payload = guard._api_request(
        "GET", "/repos/Metabuilder-Labs/tokenjam/issues/1", "token"
    )
    assert returned_status == status
    assert payload == {"message": "nope"}


def test_api_request_http_error_with_non_json_body_returns_none_payload(monkeypatch):
    def fake_urlopen(req, timeout=30):
        err = urllib.error.HTTPError(req.full_url, 500, "error", {}, io.BytesIO(b"not-json"))
        raise err

    monkeypatch.setattr(guard.urllib.request, "urlopen", fake_urlopen)
    status, payload = guard._api_request("GET", "/repos/Metabuilder-Labs/tokenjam/issues/1", "token")
    assert status == 500
    assert payload is None


def test_fetch_issue_404_does_not_raise(monkeypatch):
    def fake_urlopen(req, timeout=30):
        raise _mock_http_error(req.full_url, 404, {"message": "Not Found"})

    monkeypatch.setattr(guard.urllib.request, "urlopen", fake_urlopen)
    status, payload = guard.fetch_issue(REPO, 999, "token")
    assert status == 404
    assert payload == {"message": "Not Found"}


def test_fetch_issue_timeline_paginates_until_short_page(monkeypatch):
    page_one = [{"event": "labeled", "label": {"name": "bug"}, "created_at": "t1"}] * 100
    page_two = [{"event": "unlabeled", "label": {"name": "bug"}, "created_at": "t2"}]
    responses = iter([_mock_http_ok(page_one), _mock_http_ok(page_two)])
    seen_pages: list[int] = []

    def fake_urlopen(req, timeout=30):
        page = int(req.full_url.rsplit("page=", 1)[-1])
        seen_pages.append(page)
        return next(responses)

    monkeypatch.setattr(guard.urllib.request, "urlopen", fake_urlopen)
    events = guard.fetch_issue_timeline(REPO, 42, "token")
    assert seen_pages == [1, 2]
    assert len(events) == 101


def test_fetch_issue_timeline_non_200_returns_empty(monkeypatch):
    def fake_urlopen(req, timeout=30):
        raise _mock_http_error(req.full_url, 403, {"message": "Forbidden"})

    monkeypatch.setattr(guard.urllib.request, "urlopen", fake_urlopen)
    assert guard.fetch_issue_timeline(REPO, 42, "token") == []


def test_handle_claim_issue_not_found_logs_distinguishable_noop(monkeypatch, capsys):
    monkeypatch.setattr(guard, "fetch_issue", lambda repo, number, token: (404, {"message": "Not Found"}))
    guard._handle_claim(REPO, 999, "token")
    assert "no-op: issue #999 not found (status 404)" in capsys.readouterr().out


def test_handle_claim_auth_failure_logs_distinguishable_noop(monkeypatch, capsys):
    monkeypatch.setattr(guard, "fetch_issue", lambda repo, number, token: (403, None))
    guard._handle_claim(REPO, 12, "token")
    assert "no-op: issue #12 not found (status 403)" in capsys.readouterr().out


def test_handle_claim_payload_missing_labels_key_does_not_crash(monkeypatch, capsys):
    monkeypatch.setattr(
        guard,
        "fetch_issue",
        lambda repo, number, token: (200, {"number": 7, "state": "open"}),
    )
    guard._handle_claim(REPO, 7, "token")
    assert "no-op: issue #7 has none of the guarded labels" in capsys.readouterr().out


def test_handle_claim_remove_failure_logs_warn_to_stderr(monkeypatch, capsys):
    issue = {
        "number": 5,
        "state": "open",
        "labels": [{"name": "good first issue"}],
    }
    monkeypatch.setattr(guard, "fetch_issue", lambda repo, number, token: (200, issue))
    monkeypatch.setattr(guard, "remove_label", lambda repo, number, label, token: 403)
    guard._handle_claim(REPO, 5, "token")
    captured = capsys.readouterr()
    assert "removed label 'good first issue' from issue #5" not in captured.out
    assert "warn: failed to remove label 'good first issue' from issue #5 (status 403)" in captured.err


def test_handle_claim_success_removes_present_guarded_labels(monkeypatch, capsys):
    issue = {
        "number": 8,
        "state": "open",
        "labels": [{"name": "good first issue"}, {"name": "help wanted"}],
    }
    removed: list[str] = []
    monkeypatch.setattr(guard, "fetch_issue", lambda repo, number, token: (200, issue))
    monkeypatch.setattr(
        guard,
        "remove_label",
        lambda repo, number, label, token: removed.append(label) or 204,
    )
    guard._handle_claim(REPO, 8, "token")
    assert removed == ["good first issue", "help wanted"]
    out = capsys.readouterr().out
    assert "removed label 'good first issue' from issue #8" in out
    assert "removed label 'help wanted' from issue #8" in out


def test_handle_release_restore_failure_logs_warn_to_stderr(monkeypatch, capsys):
    issue = {"number": 3, "state": "open", "labels": []}
    events = [_unlabeled_event("help wanted")]
    monkeypatch.setattr(guard, "fetch_issue", lambda repo, number, token: (200, issue))
    monkeypatch.setattr(guard, "fetch_issue_timeline", lambda repo, number, token: events)
    monkeypatch.setattr(guard, "add_labels", lambda repo, number, labels, token: 429)
    guard._handle_release(REPO, 3, "token", guard.DEFAULT_ACTOR_LOGIN)
    captured = capsys.readouterr()
    assert "restored labels ['help wanted'] on issue #3" not in captured.out
    assert "warn: failed to restore labels ['help wanted'] on issue #3 (status 429)" in captured.err


def test_handle_release_success_restores_labels(monkeypatch, capsys):
    issue = {"number": 4, "state": "open", "labels": []}
    events = [_unlabeled_event("good first issue")]
    added: list[list[str]] = []
    monkeypatch.setattr(guard, "fetch_issue", lambda repo, number, token: (200, issue))
    monkeypatch.setattr(guard, "fetch_issue_timeline", lambda repo, number, token: events)
    monkeypatch.setattr(
        guard,
        "add_labels",
        lambda repo, number, labels, token: added.append(list(labels)) or 201,
    )
    guard._handle_release(REPO, 4, "token", guard.DEFAULT_ACTOR_LOGIN)
    assert added == [["good first issue"]]
    assert "restored labels ['good first issue'] on issue #4" in capsys.readouterr().out
