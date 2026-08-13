#!/usr/bin/env python3
"""
Prevent duplicate outside contributions on `good first issue` / `help wanted`
issues.

Background: agentic contribution tools (Cursor background agents and similar)
poll GitHub for `is:open no:assignee label:"good first issue"` and pick up
work from that feed. They do not read claim comments. The lever that
actually keeps a second tool from picking up the same issue is the LABEL,
not a comment — so this script removes the guarded labels the moment a PR
that would close the issue is opened, and puts them back only if that PR is
closed without being merged (i.e. the claim didn't pan out).

Triggered by `.github/workflows/contributor-label-guard.yml` on
`pull_request_target` (`opened` / `reopened` / `closed`). That event type is
required — not `pull_request` — because label writes need a write-scoped
`GITHUB_TOKEN`, which `pull_request` only grants for same-repo PRs. See the
workflow file for the security constraints that come with running
`pull_request_target` (no checkout of PR head, no execution of PR content);
this script treats the PR title/body purely as untrusted text to pattern-match,
never as code.

Exact-restore mechanism (requirement: never re-add a label the issue never
had): this script is STATELESS — it keeps no side file, DB row, or hidden
issue comment. On restore, it reads the issue's own timeline
(`GET /issues/{n}/timeline`) and, for each guarded label, finds that label's
most recent `labeled`/`unlabeled` event. If the most recent event is an
`unlabeled` event whose actor is this workflow's bot identity, and the label
is not currently on the issue, that label is restored. This is exact because
the timeline is already GitHub's own append-only ledger of every label
mutation, ordered by time, without this script needing to write or maintain
any bookkeeping of its own. A hidden HTML-comment marker on the issue was
the other option considered (also stateless from this script's point of
view, since GitHub stores the comment) but was passed over: it requires an
extra write (posting/updating the marker comment) on every removal, and a
second marker-parsing code path to keep in sync with the timeline reality if
a human manually re-labels the issue in between. Reading the timeline needs
no extra write and self-corrects if a human intervenes (their `labeled`
event becomes the most recent event and this script no longer treats the
label as ours to restore).

Every network call is isolated in its own function and reports a normal
(status_code, payload) pair rather than raising for HTTP error responses —
callers decide what a given status means for their case, so a missing
issue, a missing label, a closed issue, or an issue-number-that's-really-a-PR
never fails the workflow run.

Because none of those outcomes raises, the LOG LINE is the only diagnosis an
operator ever gets, so each must be distinguishable from the others: a
permissions failure, a rate limit, a deleted issue, a PR number and a
genuinely-closed issue all end in the same silent no-op, and only the wording
says which one happened. `_unusable_issue_reason` is the single place that
decides, shared by both handlers.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Sequence
from urllib.parse import quote

API_ROOT = "https://api.github.com"

# The two labels the "outside contribution" feeds actually poll on. Order is
# preserved in output (remove/restore lists) purely for stable, readable logs.
DEFAULT_GUARD_LABELS: tuple[str, ...] = ("good first issue", "help wanted")

# The actor GitHub records on timeline events performed by the default
# `GITHUB_TOKEN` inside a workflow run. Overridable via the GUARD_ACTOR_LOGIN
# env var (mainly for testing against a fork/org that renames its bot).
DEFAULT_ACTOR_LOGIN = "github-actions[bot]"

# GitHub's full closing-keyword set (case-insensitive), each optionally
# followed by an owner/repo prefix and then '#<number>'. Keywords must be
# immediately followed (allowing a colon/whitespace) by the reference — text
# like "this fixes the typo" with no adjacent '#N' must NOT match, matching
# GitHub's own closing-reference parsing.
_CLOSING_KEYWORD_PATTERN = re.compile(
    r"(?i)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b\s*:?\s*"
    r"(?:(?P<owner_repo>[\w.-]+/[\w.-]+))?#(?P<number>\d+)"
)


def parse_closing_issue_refs(text: str, repo: str) -> list[int]:
    """Extract issue numbers closed by `text` (a PR title or body) that point
    at `repo` ("owner/name").

    Cross-repo references (`owner/repo#N`) are included only when the
    `owner/repo` part matches `repo` (case-insensitive); a reference to a
    different repo is ignored. Bare `#N` references always count (same
    repo). Order of first appearance is preserved; duplicates are dropped.
    """
    if not text:
        return []
    numbers: list[int] = []
    seen: set[int] = set()
    for match in _CLOSING_KEYWORD_PATTERN.finditer(text):
        owner_repo = match.group("owner_repo")
        if owner_repo is not None and owner_repo.casefold() != repo.casefold():
            continue
        number = int(match.group("number"))
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return numbers


def resolve_pr_closing_issues(title: str, body: str, repo: str) -> list[int]:
    """Union of closing-keyword issue references in a PR's title and body,
    in first-seen order (title scanned first), deduplicated."""
    numbers: list[int] = []
    seen: set[int] = set()
    for text in (title or "", body or ""):
        for number in parse_closing_issue_refs(text, repo):
            if number not in seen:
                seen.add(number)
                numbers.append(number)
    return numbers


def labels_to_remove(
    current_label_names: Sequence[str],
    guard_labels: Sequence[str] = DEFAULT_GUARD_LABELS,
) -> list[str]:
    """Which of `guard_labels` are actually present on the issue right now.

    A label that isn't present is left out — nothing to remove, so nothing
    is ever attempted against the API for it (the no-op-safe "label not
    present" case is handled here, before any network call)."""
    current = set(current_label_names)
    return [label for label in guard_labels if label in current]


def labels_to_restore(
    timeline_events: Sequence[dict[str, Any]],
    current_label_names: Sequence[str],
    guard_labels: Sequence[str] = DEFAULT_GUARD_LABELS,
    actor_login: str = DEFAULT_ACTOR_LOGIN,
) -> list[str]:
    """Which of `guard_labels` this workflow should re-add to the issue.

    For each guarded label, look at its most recent `labeled`/`unlabeled`
    timeline event. The label is restored only if:
      - it is not already on the issue (nothing to restore otherwise), and
      - the most recent event for that label is an `unlabeled` event whose
        actor matches `actor_login`.

    A label whose most recent event is a `labeled` event (e.g. a human
    manually re-added it while the PR was open) is left alone — this
    workflow only ever restores what it, specifically, most recently took
    off. `timeline_events` is expected in GitHub's timeline API shape
    (`{"event": "labeled"|"unlabeled", "label": {"name": ...},
    "actor": {"login": ...}, "created_at": ...}`); entries missing those
    keys are ignored rather than raising.
    """
    current = set(current_label_names)
    restore: list[str] = []
    for label in guard_labels:
        if label in current:
            continue
        relevant = [
            event
            for event in timeline_events
            if event.get("event") in ("labeled", "unlabeled")
            and isinstance(event.get("label"), dict)
            and event["label"].get("name") == label
        ]
        if not relevant:
            continue
        relevant.sort(key=lambda event: event.get("created_at") or "")
        last = relevant[-1]
        actor = last.get("actor") or {}
        last_login = str(actor.get("login", ""))
        if last.get("event") == "unlabeled" and last_login.casefold() == actor_login.casefold():
            restore.append(label)
    return restore


def _api_request(
    method: str, path: str, token: str, body: dict[str, Any] | None = None
) -> tuple[int, Any]:
    """Perform one GitHub API call. Returns (status_code, parsed_json_or_None).

    HTTP error responses (4xx/5xx) are returned as a normal result, not
    raised — every caller in this script treats a particular status (404,
    422, ...) as a legitimate, no-op-safe outcome rather than a crash.
    Only network-level failures (DNS, timeout, TLS) propagate.
    """
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "tokenjam-contributor-label-guard",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{API_ROOT}{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            payload = json.loads(raw) if raw else None
            return resp.status, payload
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = None
        return exc.code, payload


def fetch_issue(repo: str, number: int, token: str) -> tuple[int, Any]:
    return _api_request("GET", f"/repos/{repo}/issues/{number}", token)


def fetch_issue_timeline(repo: str, number: int, token: str) -> list[dict[str, Any]]:
    """All timeline events for an issue, paginated to completion."""
    events: list[dict[str, Any]] = []
    page = 1
    while True:
        status, payload = _api_request(
            "GET", f"/repos/{repo}/issues/{number}/timeline?per_page=100&page={page}", token
        )
        if status != 200 or not payload:
            break
        events.extend(payload)
        if len(payload) < 100:
            break
        page += 1
    return events


def remove_label(repo: str, number: int, label: str, token: str) -> int:
    status, _ = _api_request(
        "DELETE", f"/repos/{repo}/issues/{number}/labels/{quote(label, safe='')}", token
    )
    return status


def add_labels(repo: str, number: int, labels: Sequence[str], token: str) -> int:
    status, _ = _api_request(
        "POST", f"/repos/{repo}/issues/{number}/labels", token, body={"labels": list(labels)}
    )
    return status


def _is_pull_request(issue_payload: dict[str, Any]) -> bool:
    """The issues API returns PRs too (they share the issue number-space);
    a PR payload carries a `pull_request` key an issue payload never has."""
    return "pull_request" in issue_payload


def _unusable_issue_reason(number: int, status: int, issue: Any) -> str | None:
    """The log line explaining why `issue` can't be acted on, or None if it can.

    Shared by both handlers so the two can never diagnose the same response
    differently — the misdiagnosis this exists to prevent was present in
    both, and was noticed in only one.

    Order matters, and is the whole point. The STATUS is checked before the
    payload's shape, because `_api_request` returns `(status, None)` for any
    error response whose body isn't valid JSON (GitHub serves HTML from its
    secondary-rate-limit and abuse walls, as can any proxy in front of the
    API). Testing the shape first collapses "we were forbidden" into "the
    issue doesn't exist" — two different operator actions behind one line.
    """
    if status == 404:
        return f"no-op: issue #{number} not found (status {status})"
    if status != 200:
        return f"no-op: issue #{number} API error (status {status})"
    if not isinstance(issue, dict):
        return f"no-op: issue #{number} returned an unreadable payload (status {status})"
    if _is_pull_request(issue):
        return f"no-op: #{number} is a pull request, not an issue"
    return None


def _handle_claim(repo: str, number: int, token: str) -> None:
    status, issue = fetch_issue(repo, number, token)
    reason = _unusable_issue_reason(number, status, issue)
    if reason is not None:
        print(reason)
        return
    if issue.get("state") != "open":
        print(f"no-op: issue #{number} is already closed")
        return
    current = [lbl["name"] for lbl in issue.get("labels", [])]
    to_remove = labels_to_remove(current)
    if not to_remove:
        print(f"no-op: issue #{number} has none of the guarded labels")
        return
    for label in to_remove:
        result = remove_label(repo, number, label, token)
        if result in (200, 204):
            print(f"removed label '{label}' from issue #{number}")
        else:
            print(
                f"warn: failed to remove label '{label}' from issue #{number} "
                f"(status {result})",
                file=sys.stderr,
            )


def _handle_release(repo: str, number: int, token: str, actor_login: str) -> None:
    status, issue = fetch_issue(repo, number, token)
    reason = _unusable_issue_reason(number, status, issue)
    if reason is not None:
        print(reason)
        return
    if issue.get("state") != "open":
        print(f"no-op: issue #{number} is already closed — leaving labels as-is")
        return
    current = [lbl["name"] for lbl in issue.get("labels", [])]
    events = fetch_issue_timeline(repo, number, token)
    to_restore = labels_to_restore(events, current, actor_login=actor_login)
    if not to_restore:
        print(f"no-op: nothing to restore on issue #{number}")
        return
    result = add_labels(repo, number, to_restore, token)
    if result in (200, 201):
        print(f"restored labels {to_restore} on issue #{number}")
    else:
        print(
            f"warn: failed to restore labels {to_restore} on issue #{number} "
            f"(status {result})",
            file=sys.stderr,
        )


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    actor_login = os.environ.get("GUARD_ACTOR_LOGIN", DEFAULT_ACTOR_LOGIN)
    if not token or not repo or not event_path:
        print(
            "error: GITHUB_TOKEN, GITHUB_REPOSITORY, and GITHUB_EVENT_PATH must all be set",
            file=sys.stderr,
        )
        return 1

    try:
        with open(event_path, encoding="utf-8") as fh:
            event = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read event payload at {event_path}: {exc}", file=sys.stderr)
        return 1

    action = event.get("action")
    pr = event.get("pull_request") or {}
    pr_number = pr.get("number")
    title = pr.get("title") or ""
    body = pr.get("body") or ""
    merged = bool(pr.get("merged"))

    if action not in ("opened", "reopened", "closed"):
        print(f"no-op: unsupported action '{action}'")
        return 0

    issue_numbers = resolve_pr_closing_issues(title, body, repo)
    if not issue_numbers:
        print(f"no-op: PR #{pr_number} has no closing-keyword issue references")
        return 0

    if action in ("opened", "reopened"):
        for number in issue_numbers:
            _handle_claim(repo, number, token)
    elif merged:
        print(f"no-op: PR #{pr_number} merged — issue(s) {issue_numbers} close on merge")
    else:
        for number in issue_numbers:
            _handle_release(repo, number, token, actor_login)

    return 0


if __name__ == "__main__":
    sys.exit(main())
