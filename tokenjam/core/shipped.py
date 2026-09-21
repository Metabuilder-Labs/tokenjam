"""
Session -> commit join and the shipped / unshipped read side (shipped-value
ledger, contracts §1, §2, §4, §5 read side).

Two halves, deliberately split by whether they touch git.

**The matcher** (`match_sessions_to_commits`) runs on the daemon's analyzer
pass and from a direct-DB `tj optimize`, never on a request. It is the ONLY
thing here that shells out, and it does so under W1's discipline
(`core/repo_context.py`): read-only, `timeout=2`, `check=False`, absent on
failure, skipped for a repo root that no longer exists. For every session
that changed since its last scan it reads the commits inside the session's
window and joins each one at the best confidence the evidence supports:

* `tool_span_git_log` (deterministic): a Bash tool call in the session ran
  `git commit` within `MATCH_WINDOW_S` of the commit's own timestamp.
* `trailer_session` (deterministic): the commit body carries
  `TokenJam-Session: <id>` or `Claude-Session: <url>` resolving to an ingested
  session. The URL names the bridge session id, not the local uuid, which is
  why `sessions.bridge_session_id` exists. **Ranked below `tool_span_git_log`
  when the two disagree:** the trailer is written from the shell environment
  the commit inherited (`CLAUDE_CODE_SESSION_ID`), and a subagent spawned by
  a session inherits the parent's, so a commit the subagent's own Bash tool
  ran carries the PARENT's trailer. The session whose tool call ran the
  commit produced it; a trailer naming a different session is inherited
  evidence and is written at `inferred`, so the parent never absorbs its
  subagents' commits at the top confidence (issue #770, fix 6).
* `git_note` (deterministic): a `refs/notes/ai` (Git AI), `refs/notes/exceeds-ink`
  or `refs/notes/tokenjam` (our own, written by the `tj init --notes`
  post-commit hook, `core/commit_hooks.py`) note names a session id we
  ingested. The matcher only ever reads notes.
* `trailer_window` (inferred): an AI co-author trailer, on the session's
  start or end branch, inside the window, with no tool-span match.

Best confidence wins and a stored row is never downgraded
(`DuckDBBackend.upsert_session_commits`). Idempotent: re-running over the same
history writes nothing new.

The same pass keeps three side tables current so that **every read is SQL**:
`repo_commits` (the default branch, indexed incrementally from the last seen
tip, so "on the default branch" and "reverted by" are joins), `commit_file_stats`
(the rework evidence, contracts §1) and `session_commit_scans` (the per-session
watermark). A request handler may never shell out, so the shipped state of a
session is DERIVED at read time from those tables, never from git.

**The read side** (`shipped_summary`, `session_shipped_state`) is pure SQL over
the tables above and is what the `shipped` analyzer, `/api/v1/shipped`, the
session detail route and `tj status` all call, so they cannot disagree.

Honesty (Critical Rule 14 in `tokenjam/CLAUDE.md`, contracts §1): "unshipped"
is a MEASURED cost over sessions with no joined commit; a session can ship
value without a commit, and the caveat says so on every surface. Sessions
with no repo context are not analysed at all and are reported as such, never
folded into either side.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from tokenjam.core.models import SessionCommit
from tokenjam.core.persona_scope import add_persona_clause
from tokenjam.otel.semconv import GenAIAttributes
from tokenjam.utils.time_parse import utcnow

logger = logging.getLogger(__name__)

#: Contracts §2 `match_window_s`: a `git commit` tool call and the commit it
#: produced must sit this close together to be the same event.
MATCH_WINDOW_S = 30.0
#: How far before `started_at` and after `ended_at` a session's commit window
#: reaches (brief §2: `[started_at - 60s, ended_at + 15m]`).
WINDOW_BEFORE = timedelta(seconds=60)
WINDOW_AFTER = timedelta(minutes=15)
#: Contracts §1 `rework_window_days`.
REWORK_WINDOW_DAYS = 14
#: A session's commits count as reworked when at least this share of the
#: lines they added was deleted again inside the rework window.
REWORK_SHARE = 0.5
#: Hard ceiling on every git shell-out (W1's `GIT_TIMEOUT_S`).
GIT_TIMEOUT_S = 2.0
#: Sessions and commits one pass will touch at most, so a first pass over a
#: large history stays bounded; the watermark carries the rest to the next.
MAX_SESSIONS_PER_PASS = 4000
MAX_REWORK_COMMITS_PER_PASS = 400
#: Late evidence (a note attached after the fact, a rebased trailer, a bridge
#: id or tool spans a later backfill filled in) does not move a session's
#: `ended_at`, so a session younger than `RESCAN_HORIZON` is re-scanned once
#: its last scan is older than `RESCAN_AFTER`, until it ages out.
RESCAN_AFTER = timedelta(hours=24)
RESCAN_HORIZON = timedelta(days=30)
#: Rows to carry on the finding.
TOP_N = 5

#: Contracts §4: the caveat every surface renders beside the finding.
SHIPPED_CAVEAT = (
    "Unshipped is measured cost of sessions with no joined commit; a session "
    "can ship value without a commit (research, review, ops). Review before "
    "acting."
)

#: The `shipped_state` values a session can derive to (brief §3).
STATE_SHIPPED = "shipped"                 # a joined commit is on the default branch
STATE_COMMITTED = "committed"             # joined commits exist, none on the default branch
STATE_REVERTED = "reverted"               # every joined commit was reverted
STATE_UNSHIPPED = "unshipped"             # no joined commit at all
STATE_NO_REPO = "no_repo"                 # no repo context, never analysed

_SEP = "\x1f"
_REC = "\x1e"

# `Co-Authored-By: Claude ...`, `Co-authored-by: Copilot`, `Made-with:`,
# `Co-Authored-By: Codex`, `Generated with ...`. Case-insensitive on the key.
_AI_TRAILER = re.compile(
    r"^(?:co-authored-by|made-with|generated-with)\s*:\s*.*"
    r"(?:claude|copilot|codex|anthropic|openai|gemini|cursor|tokenjam)",
    re.IGNORECASE | re.MULTILINE,
)
_TJ_SESSION_TRAILER = re.compile(r"^TokenJam-Session:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)
_CLAUDE_SESSION_TRAILER = re.compile(
    r"^Claude-Session:\s*\S*?/?(?:session_)?([A-Za-z0-9]+)\s*$", re.IGNORECASE | re.MULTILINE,
)
#: Notes refs the matcher reads (contracts §5): Git AI, Exceeds, and ours.
NOTES_REFS_READ: tuple[str, ...] = ("ai", "exceeds-ink", "tokenjam")
_REVERT_BODY = re.compile(r"This reverts commit ([0-9a-f]{7,40})", re.IGNORECASE)
_SHA_IN_SUBJECT = re.compile(r"\b([0-9a-f]{7,40})\b")

# A `git commit` invocation anywhere in a shell line: `git commit`, `git -C x
# commit`, `git -c k=v commit`, chained with `&&`/`;`/`||`. Not `--dry-run`.
_GIT_COMMIT = re.compile(r"(?:^|[;&|(\s])git(?:\s+-[cC]\s+\S+|\s+--\S+)*\s+commit\b")


# --- Pure helpers ------------------------------------------------------------------

def is_git_commit_command(command: Any) -> bool:
    """True when a Bash tool input runs `git commit` (any form the brief lists).

    `--dry-run` is not a commit. `--amend` and `--no-verify` are commits.
    """
    if not isinstance(command, str) or "commit" not in command:
        return False
    if not _GIT_COMMIT.search(command):
        return False
    return "--dry-run" not in command


def bridge_suffix(value: str | None) -> str | None:
    """The part of a bridge id that both spellings share.

    A transcript records `cse_01AbC...`, a `Claude-Session:` trailer names
    `https://claude.ai/code/session_01AbC...`; the id after the prefix is the
    same string, so both resolve on it.
    """
    if not value:
        return None
    v = value.strip()
    if "_" in v:
        v = v.split("_", 1)[1]
    return v or None


def parse_trailers(body: str) -> dict[str, Any]:
    """Session ids and AI signals a commit body carries."""
    tj = [m.group(1) for m in _TJ_SESSION_TRAILER.finditer(body or "")]
    claude = [m.group(1) for m in _CLAUDE_SESSION_TRAILER.finditer(body or "")]
    return {
        "tokenjam_sessions": tj,
        "claude_sessions": claude,
        "ai": bool(_AI_TRAILER.search(body or "")),
    }


def reverted_sha(subject: str | None, body: str | None) -> str | None:
    """The sha a `Revert "..."` commit names, or None.

    The body's `This reverts commit <sha>.` line is authoritative; a subject
    starting with `Revert` that carries a sha is accepted when the body has
    none. A subject starting with `Revert` and naming nothing is not a match.
    """
    if not subject or not subject.lstrip().lower().startswith("revert"):
        return None
    m = _REVERT_BODY.search(body or "")
    if m:
        return m.group(1).lower()
    m = _SHA_IN_SUBJECT.search(subject)
    return m.group(1).lower() if m else None


def content_hash(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            value = json.dumps(value, sort_keys=True)
        except (TypeError, ValueError):
            return None
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16]


def _utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _attrs(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes)):
        try:
            v = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return v if isinstance(v, dict) else {}
    return {}


def _tool_input(raw: Any) -> dict:
    """`gen_ai.tool.input` off a span's attributes, as a dict. The backfill
    stores a dict; the SDK path and the test factory store it JSON-encoded."""
    value = _attrs(raw).get(GenAIAttributes.TOOL_INPUT)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


# --- Git ---------------------------------------------------------------------------

def _git(args: list[str], cwd: str) -> str | None:
    """Read-only git; stdout on exit 0, else None. Never raises."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        proc = subprocess.run(
            [git, *args], cwd=cwd, capture_output=True, text=True,
            timeout=GIT_TIMEOUT_S, check=False, errors="replace",
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


@dataclass
class GitCommit:
    sha: str
    committed_at: datetime
    author_email: str | None
    parents: list[str]
    subject: str
    body: str
    refs: str = ""


def _parse_log(out: str | None) -> list[GitCommit]:
    commits: list[GitCommit] = []
    if not out:
        return commits
    for rec in out.split(_REC):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        parts = rec.split(_SEP)
        if len(parts) < 6:
            continue
        sha, at, email, parents, subject, body = parts[0].strip(), parts[1], parts[2], parts[3], parts[4], parts[5]
        try:
            ts = datetime.fromtimestamp(int(at), tz=timezone.utc)
        except (TypeError, ValueError):
            continue
        commits.append(GitCommit(
            sha=sha.lower(), committed_at=ts, author_email=(email.strip() or None),
            parents=parents.split(), subject=subject.strip(), body=body,
            refs=parts[6] if len(parts) > 6 else "",
        ))
    return commits


_LOG_FORMAT = f"--format=%H{_SEP}%ct{_SEP}%ae{_SEP}%P{_SEP}%s{_SEP}%B{_SEP}%D{_REC}"


def _log_between(root: str, since: datetime, until: datetime) -> list[GitCommit]:
    return _parse_log(_git([
        "log", "--all", _LOG_FORMAT,
        f"--since={since.isoformat()}", f"--until={until.isoformat()}",
    ], root))


def _log_range(root: str, spec: list[str]) -> list[GitCommit]:
    return _parse_log(_git(["log", _LOG_FORMAT, *spec], root))


def _default_branch(root: str) -> str | None:
    """`origin/HEAD`'s branch name, else `main`, else `master`, else the current."""
    out = _git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], root)
    if out and out.strip():
        return out.strip().split("/", 1)[-1] or None
    for name in ("main", "master"):
        if _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{name}"], root) is not None:
            return name
    out = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    if out and out.strip() and out.strip() != "HEAD":
        return out.strip()
    return None


def _default_refs(root: str, branch: str) -> list[str]:
    """The refs that together mean "the default branch": the remote-tracking
    ref (what was merged upstream; tj never fetches, so it is as fresh as the
    user's last fetch) AND the local branch (a merge not pushed yet). Either
    alone is wrong in one direction: a checkout on a feature branch whose
    local `main` is stale would miss upstream merges, and the remote alone
    would miss local ones."""
    return [
        ref for ref in (f"refs/remotes/origin/{branch}", f"refs/heads/{branch}")
        if _has_ref(root, ref)
    ]


def _rev(root: str, ref: str) -> str | None:
    out = _git(["rev-parse", "--verify", "--quiet", ref], root)
    return out.strip().lower() if out and out.strip() else None


def _is_ancestor(root: str, sha: str, ref: str) -> bool:
    return _git(["merge-base", "--is-ancestor", sha, ref], root) is not None


def _has_ref(root: str, ref: str) -> bool:
    return _git(["show-ref", "--verify", "--quiet", ref], root) is not None


def _read_note(root: str, notes_ref: str, sha: str) -> dict | None:
    out = _git(["notes", f"--ref={notes_ref}", "show", sha], root)
    if not out:
        return None
    try:
        v = json.loads(out)
    except ValueError:
        return None
    return v if isinstance(v, dict) else None


def _numstat(root: str, sha: str) -> dict[str, int]:
    """Lines each path gained in `sha` (merges and binaries contribute 0)."""
    out = _git(["show", "--numstat", "--format=", "--no-renames", sha], root)
    added: dict[str, int] = {}
    for line in (out or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        try:
            added[parts[2]] = int(parts[0])
        except ValueError:
            added[parts[2]] = 0
    return added


def _later_deletions(
    root: str, paths: list[str], after: datetime, before: datetime, exclude: set[str],
) -> dict[str, int]:
    """Lines later commits (not in `exclude`) deleted from `paths` in the window."""
    if not paths:
        return {}
    out = _git([
        "log", "--all", "--numstat", "--no-renames", "--format=%x1e%H",
        f"--after={after.isoformat()}", f"--before={before.isoformat()}", "--", *paths,
    ], root)
    deleted: dict[str, int] = defaultdict(int)
    for rec in (out or "").split(_REC):
        lines = rec.strip("\n").splitlines()
        if not lines:
            continue
        sha = lines[0].strip().lower()
        if not sha or sha in exclude:
            continue
        for line in lines[1:]:
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            try:
                deleted[parts[2]] += int(parts[1])
            except ValueError:
                continue
    return dict(deleted)


# --- The pass --------------------------------------------------------------------

@dataclass
class MatchResult:
    sessions_scanned: int = 0
    rows_written: int = 0
    repos_indexed: int = 0
    repos_missing: int = 0
    rework_commits_checked: int = 0
    by_source: dict[str, int] = field(default_factory=dict)


@dataclass
class _Session:
    session_id: str
    repo_root: str | None
    repo_remote: str | None
    branch_start: str | None
    branch_end: str | None
    user_email: str | None
    bridge_session_id: str | None
    started_at: datetime
    ended_at: datetime

    @property
    def window(self) -> tuple[datetime, datetime]:
        return self.started_at - WINDOW_BEFORE, self.ended_at + WINDOW_AFTER


def _candidate_sessions(conn, limit: int, now: datetime) -> list[_Session]:
    """Sessions never scanned, changed since their scan (`ended_at` moved), or
    recent enough that late evidence may still arrive (see `RESCAN_AFTER`)."""
    rows = conn.execute(
        """
        SELECT s.session_id, s.repo_root, s.repo_remote, s.branch_start, s.branch_end,
               s.user_email, s.bridge_session_id, s.started_at,
               COALESCE(s.ended_at, s.started_at) AS ended_at
        FROM sessions s
        LEFT JOIN session_commit_scans sc ON sc.session_id = s.session_id
        WHERE (s.repo_root IS NOT NULL OR s.repo_remote IS NOT NULL)
          AND s.started_at IS NOT NULL
          AND (sc.session_id IS NULL
               OR sc.ended_at IS NULL
               OR COALESCE(s.ended_at, s.started_at) > sc.ended_at
               OR (sc.scanned_at < $2 AND COALESCE(s.ended_at, s.started_at) > $3))
        ORDER BY COALESCE(s.ended_at, s.started_at) DESC
        LIMIT $1
        """,
        [limit, now - RESCAN_AFTER, now - RESCAN_HORIZON],
    ).fetchall()
    out: list[_Session] = []
    for r in rows:
        started, ended = _utc(r[7]), _utc(r[8])
        if started is None or ended is None:
            continue
        out.append(_Session(r[0], r[1], r[2], r[3], r[4], r[5], r[6], started, ended))
    return out


def _resolve_roots(conn, sessions: list[_Session]) -> tuple[dict[str, str], int, dict[str, str]]:
    """`session_id -> usable repo root`. A session whose own root is gone
    (a deleted worktree) borrows any live root with the same remote: worktrees
    share one object store, so `git log --all` there sees its branches."""
    live_by_remote: dict[str, str] = {}
    for root, remote in conn.execute(
        "SELECT DISTINCT repo_root, repo_remote FROM sessions "
        "WHERE repo_root IS NOT NULL AND repo_remote IS NOT NULL"
    ).fetchall():
        if remote not in live_by_remote and os.path.isdir(root):
            live_by_remote[remote] = root
    resolved: dict[str, str] = {}
    missing = 0
    for s in sessions:
        if s.repo_root and os.path.isdir(s.repo_root):
            resolved[s.session_id] = s.repo_root
        elif s.repo_remote and s.repo_remote in live_by_remote:
            resolved[s.session_id] = live_by_remote[s.repo_remote]
        else:
            missing += 1
    return resolved, missing, live_by_remote


def _commit_tool_spans(
    conn, since: datetime, until: datetime,
) -> tuple[dict[str, list[datetime]], dict[str, tuple[str | None, str | None]]]:
    """`session_id -> timestamps of Bash tool calls that ran git commit`, over
    every session with repo context whose call falls in `[since, until]`,
    not only the sessions due for a scan.

    Every session in the span rather than the candidate set because a
    trailer's rank depends on who else ran the commit (see
    `CommitToolIndex.producers`): a parent session due for a rescan must see
    that a subagent session, scanned weeks ago, is the one whose tool call
    produced the commit. Bounded to the candidates' combined window so the
    read grows with the pass, not with the whole retained history."""
    rows = conn.execute(
        "SELECT sp.session_id, sp.start_time, sp.attributes, s.repo_remote, s.repo_root "
        "FROM spans sp JOIN sessions s ON s.session_id = sp.session_id "
        "WHERE sp.tool_name = 'Bash' "
        "AND sp.start_time >= $1 AND sp.start_time <= $2 "
        "AND (s.repo_root IS NOT NULL OR s.repo_remote IS NOT NULL) "
        "AND CAST(sp.attributes AS VARCHAR) LIKE '%commit%'",
        [since, until],
    ).fetchall()
    out: dict[str, list[datetime]] = defaultdict(list)
    repos: dict[str, tuple[str | None, str | None]] = {}
    for sid, ts, raw, remote, root in rows:
        if is_git_commit_command(_tool_input(raw).get("command")):
            when = _utc(ts)
            if when is not None:
                out[sid].append(when)
                repos[sid] = (remote, root)
    return out, repos


class CommitToolIndex:
    """Every `git commit` tool call in the corpus, time-sorted, answering two
    questions per commit: how close this session's own call was
    (`best_delta`), and which sessions ran a commit inside the match window
    at all (`producers`)."""

    def __init__(self, by_session: dict[str, list[datetime]],
                 repos: dict[str, tuple[str | None, str | None]] | None = None) -> None:
        self._by_session = by_session
        #: session -> (repo_remote, repo_root) of the session that ran it.
        self._repos = repos or {}
        pairs = sorted((t, sid) for sid, times in by_session.items() for t in times)
        self._times = [t for t, _ in pairs]
        self._sids = [sid for _, sid in pairs]

    def times(self, session_id: str) -> list[datetime]:
        return self._by_session.get(session_id, [])

    def best_delta(self, session_id: str, committed_at: datetime) -> float | None:
        """Tool-span time minus commit time for this session's closest call
        inside `MATCH_WINDOW_S`, else None."""
        best: float | None = None
        for t in self.times(session_id):
            delta = (t - committed_at).total_seconds()
            if abs(delta) <= MATCH_WINDOW_S and (best is None or abs(delta) < abs(best)):
                best = delta
        return best

    def producers(
        self, committed_at: datetime, remote: str | None = None, root: str | None = None,
    ) -> set[str]:
        """Sessions whose own Bash tool ran `git commit` within the match
        window of `committed_at`, in the repo `(remote, root)` names.

        Same repo means the same remote when both sides know one (a worktree
        shares its parent's remote), else the same root; a session that ran
        a commit in an unrelated repository at the same second is not
        evidence about this commit, and a session whose repo is unknown on
        both counts never is."""
        window = timedelta(seconds=MATCH_WINDOW_S)
        lo = bisect_left(self._times, committed_at - window)
        hi = bisect_right(self._times, committed_at + window)
        found = set(self._sids[lo:hi])
        if remote is None and root is None:
            return found
        out: set[str] = set()
        for sid in found:
            p_remote, p_root = self._repos.get(sid, (None, None))
            if remote and p_remote and p_remote == remote:
                out.add(sid)
            elif root and p_root and p_root == root:
                out.add(sid)
        return out


def trailer_rank(named: str, producers: set[str]) -> tuple[str, str]:
    """`(confidence, source)` for a `TokenJam-Session:` / `Claude-Session:`
    trailer resolving to `named`, given the sessions whose tool call ran the
    commit. Deterministic when nobody else is known to have run it, or when
    the named session did; `inferred` when a DIFFERENT session's tool span
    matched, because the trailer was then inherited from the environment
    (contracts §2, issue #770 fix 6)."""
    if producers and named not in producers:
        return "inferred", "trailer_session"
    return "deterministic", "trailer_session"


def _session_index(conn) -> tuple[set[str], dict[str, list[tuple[str, datetime, datetime]]]]:
    """Every ingested session id, and bridge suffix -> [(id, start, end)]."""
    ids: set[str] = set()
    by_bridge: dict[str, list[tuple[str, datetime, datetime]]] = defaultdict(list)
    for sid, bridge, started, ended in conn.execute(
        "SELECT session_id, bridge_session_id, started_at, COALESCE(ended_at, started_at) "
        "FROM sessions"
    ).fetchall():
        ids.add(sid)
        suffix = bridge_suffix(bridge)
        if suffix and started is not None:
            by_bridge[suffix].append((sid, _utc(started), _utc(ended)))  # type: ignore[arg-type]
    return ids, by_bridge


def _resolve_bridge(
    suffix: str, at: datetime, by_bridge: dict[str, list[tuple[str, datetime, datetime]]],
) -> str | None:
    """The ingested session a bridge id names. Several local sessions can share
    one bridge id, so prefer the one whose window holds the commit, else the
    nearest by end time."""
    candidates = by_bridge.get(suffix) or []
    if not candidates:
        return None
    inside = [
        c for c in candidates
        if c[1] - WINDOW_BEFORE <= at <= c[2] + WINDOW_AFTER
    ]
    pool = inside or candidates
    return min(pool, key=lambda c: abs((c[2] - at).total_seconds()))[0]


def _index_repo(conn, root: str, remote: str | None, oldest: datetime, now: datetime) -> bool:
    """Bring `repo_commits` up to the default branch's current tip.

    Incremental from the last seen tip; a rewritten branch (old tip no longer
    an ancestor) re-indexes from `oldest`. Returns False when the repo has no
    default branch to index.
    """
    branch = _default_branch(root)
    if branch is None:
        return False
    refs = _default_refs(root, branch)
    tips = [t for t in (_rev(root, r) for r in refs) if t]
    if not tips:
        return False
    tip = ",".join(sorted(set(tips)))
    state = conn.execute(
        "SELECT default_tip FROM ledger_repo_state WHERE repo_root = $1", [root],
    ).fetchone()
    old_tip = state[0] if state else None
    if old_tip == tip:
        return True
    old_tips = [t for t in (old_tip or "").split(",") if t]
    # Incremental only while every previously indexed tip is still reachable
    # from the branch; a rewritten history (force push, reset) re-indexes from
    # scratch, and the rows the old history put here go with it, or a commit
    # force-pushed away would stay "shipped" and a stale revert would keep
    # flipping a session's state.
    intact = bool(old_tips) and all(
        any(_is_ancestor(root, t, ref) for ref in refs) for t in old_tips
    )
    if intact:
        commits = _log_range(root, [*refs, "--not", *old_tips])
    else:
        conn.execute("DELETE FROM repo_commits WHERE repo_root = $1", [root])
        commits = _log_range(root, [*refs, f"--since={(oldest - timedelta(days=1)).isoformat()}"])
    for c in commits:
        conn.execute(
            "INSERT INTO repo_commits (repo_root, commit_sha, repo_remote, author_email, "
            "committed_at, subject, reverts_sha, indexed_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) "
            "ON CONFLICT (repo_root, commit_sha) DO UPDATE SET indexed_at = EXCLUDED.indexed_at",
            [root, c.sha, remote, c.author_email, c.committed_at, c.subject[:200],
             reverted_sha(c.subject, c.body), now],
        )
    conn.execute(
        "INSERT INTO ledger_repo_state (repo_root, default_branch, default_tip, indexed_at) "
        "VALUES ($1,$2,$3,$4) ON CONFLICT (repo_root) DO UPDATE SET "
        "default_branch = EXCLUDED.default_branch, default_tip = EXCLUDED.default_tip, "
        "indexed_at = EXCLUDED.indexed_at",
        [root, branch, tip, now],
    )
    return True


def _match_session(
    s: _Session, root: str, commits: list[GitCommit], tools: CommitToolIndex,
    known_ids: set[str], by_bridge: dict, notes_refs: list[str], now: datetime,
) -> list[SessionCommit]:
    rows: dict[tuple[str, str], SessionCommit] = {}
    branches = {b for b in (s.branch_start, s.branch_end) if b}
    branch_cache: dict[tuple[str, str], bool] = {}

    def on_branch(sha: str) -> bool:
        for b in branches:
            key = (sha, b)
            if key not in branch_cache:
                branch_cache[key] = _is_ancestor(root, sha, b)
            if branch_cache[key]:
                return True
        return False

    def offer(sid: str, c: GitCommit, confidence: str, source: str, delta: float | None) -> None:
        key = (sid, c.sha)
        row = SessionCommit(
            session_id=sid, commit_sha=c.sha, confidence=confidence, source=source,
            repo_remote=s.repo_remote, author_email=c.author_email,
            committed_at=c.committed_at, matched_at=now, match_delta_s=delta,
        )
        prev = rows.get(key)
        if prev is None or row.rank > prev.rank:
            rows[key] = row

    for c in commits:
        # 1. A git commit tool call in this session, closest within the window.
        best = tools.best_delta(s.session_id, c.committed_at)
        if best is not None:
            offer(s.session_id, c, "deterministic", "tool_span_git_log", best)

        # 2. Trailers naming an ingested session (this one or another). A
        #    trailer naming a session OTHER than one whose tool call ran the
        #    commit is inherited evidence and ranks below it (`trailer_rank`).
        trailers = parse_trailers(c.body)
        producers = tools.producers(c.committed_at, s.repo_remote, root)
        for sid in trailers["tokenjam_sessions"]:
            if sid in known_ids:
                offer(sid, c, *trailer_rank(sid, producers), None)
        for suffix in trailers["claude_sessions"]:
            sid = _resolve_bridge(suffix, c.committed_at, by_bridge)
            if sid is not None:
                offer(sid, c, *trailer_rank(sid, producers), None)

        # 3. Git notes (Git AI / Exceeds) naming a session we ingested.
        for ref in notes_refs:
            note = _read_note(root, ref, c.sha)
            if not note:
                continue
            named = note.get("session") or note.get("session_id")
            if isinstance(named, str):
                if named in known_ids:
                    offer(named, c, "deterministic", "git_note", None)
                else:
                    sid = _resolve_bridge(bridge_suffix(named) or "", c.committed_at, by_bridge)
                    if sid is not None:
                        offer(sid, c, "deterministic", "git_note", None)

        # 4. AI co-author trailer on this session's branch, same author, no
        #    stronger evidence: inferred.
        if (
            (s.session_id, c.sha) not in rows
            and trailers["ai"]
            and branches
            and (not s.user_email or not c.author_email
                 or c.author_email.lower() == s.user_email.lower())
            and on_branch(c.sha)
        ):
            offer(s.session_id, c, "inferred", "trailer_window", None)
    return list(rows.values())


def _refresh_rework(conn, root: str, now: datetime, budget: int) -> int:
    """(Re)take rework evidence for this repo's joined commits whose window
    is still open or never measured. Returns commits checked."""
    if budget <= 0:
        return 0
    rows = conn.execute(
        """
        SELECT DISTINCT sc.commit_sha, sc.committed_at, sc.session_id
        FROM session_commits sc
        JOIN sessions s ON s.session_id = sc.session_id
        LEFT JOIN (
            SELECT commit_sha, BOOL_AND(horizon_closed) AS closed
            FROM commit_file_stats WHERE repo_root = $1 GROUP BY commit_sha
        ) f ON f.commit_sha = sc.commit_sha
        WHERE (s.repo_root = $1 OR s.repo_remote IN (
                  SELECT repo_remote FROM sessions WHERE repo_root = $1 AND repo_remote IS NOT NULL))
          AND sc.confidence IN ('deterministic', 'inferred')
          AND sc.committed_at IS NOT NULL
          AND COALESCE(f.closed, FALSE) = FALSE
        ORDER BY sc.committed_at DESC
        LIMIT $2
        """,
        [root, budget],
    ).fetchall()
    if not rows:
        return 0
    # Commits joined to the same session are that session's own iteration,
    # not rework of it, so they are excluded from the "later deletions".
    session_shas: dict[str, set[str]] = defaultdict(set)
    for sha, sid in conn.execute(
        "SELECT commit_sha, session_id FROM session_commits WHERE confidence IN "
        "('deterministic', 'inferred')"
    ).fetchall():
        session_shas[sid].add(sha)
    checked = 0
    seen: set[str] = set()
    for sha, committed_at, sid in rows:
        if sha in seen:
            continue
        seen.add(sha)
        committed = _utc(committed_at)
        if committed is None:
            continue
        horizon = committed + timedelta(days=REWORK_WINDOW_DAYS)
        closed = horizon <= now
        before = min(horizon, now)
        added = _numstat(root, sha)
        checked += 1
        if not added:
            # Merge commit, binary-only or unreadable: record a closed empty
            # row so it is not re-taken forever, and contributes nothing.
            conn.execute(
                "INSERT INTO commit_file_stats (repo_root, commit_sha, path, additions, "
                "later_deletions, horizon_closed, checked_at) VALUES ($1,$2,'',0,0,TRUE,$3) "
                "ON CONFLICT (repo_root, commit_sha, path) DO UPDATE SET checked_at = EXCLUDED.checked_at",
                [root, sha, now],
            )
            continue
        deleted = _later_deletions(root, list(added), committed, before, session_shas.get(sid, set()) | {sha})
        for path, adds in added.items():
            conn.execute(
                "INSERT INTO commit_file_stats (repo_root, commit_sha, path, additions, "
                "later_deletions, horizon_closed, checked_at) VALUES ($1,$2,$3,$4,$5,$6,$7) "
                "ON CONFLICT (repo_root, commit_sha, path) DO UPDATE SET "
                "additions = EXCLUDED.additions, later_deletions = EXCLUDED.later_deletions, "
                "horizon_closed = EXCLUDED.horizon_closed, checked_at = EXCLUDED.checked_at",
                [root, sha, path, adds, min(deleted.get(path, 0), adds), closed, now],
            )
    return checked


def match_sessions_to_commits(db: Any, config: Any = None, *, now: datetime | None = None) -> MatchResult:
    """Join every changed session to the commits it produced (contracts §4).

    Never raises: a failure in one repo or session is logged and the pass
    moves on, and the DuckDB fatal case is left to the caller's classifier.
    Runs from the daemon's analyzer pass (`scan_cycle`) and from a direct-DB
    `tj optimize`; never from a request handler.
    """
    result = MatchResult()
    conn = getattr(db, "conn", None)
    if conn is None or shutil.which("git") is None:
        return result
    now = _utc(now) or utcnow()
    lock = getattr(db, "write_lock", None)
    sessions = _candidate_sessions(conn, MAX_SESSIONS_PER_PASS, now)
    roots, result.repos_missing, live_by_remote = _resolve_roots(conn, sessions)
    known_ids, by_bridge = _session_index(conn)
    if sessions:
        span_lo = min(s.window[0] for s in sessions)
        span_hi = max(s.window[1] for s in sessions)
        tools = CommitToolIndex(*_commit_tool_spans(conn, span_lo, span_hi))
    else:
        tools = CommitToolIndex({})

    by_root: dict[str, list[_Session]] = defaultdict(list)
    for s in sessions:
        root = roots.get(s.session_id)
        if root:
            by_root[root].append(s)
    # Every live repo is re-indexed each pass, not only the ones with a
    # session due for a scan: a merge that lands weeks after the session
    # ended flips its state through the index alone, and the tip comparison
    # makes an unchanged repo a no-op.
    for live_root in live_by_remote.values():
        by_root.setdefault(live_root, [])

    rework_budget = MAX_REWORK_COMMITS_PER_PASS
    for root, group in by_root.items():
        oldest = min((s.started_at for s in group), default=None) or _oldest_session_start(conn, root, now)
        remote: str | None = next((s.repo_remote for s in group if s.repo_remote), None) or next(
            (r for r, rt in live_by_remote.items() if rt == root), None)
        try:
            if _index_repo(conn, root, remote, oldest, now):
                result.repos_indexed += 1
            if not group:
                continue
            notes_refs = [r for r in NOTES_REFS_READ if _has_ref(root, f"refs/notes/{r}")]
            for s in group:
                since, until = s.window
                commits = _log_between(root, since, until)
                rows = _match_session(
                    s, root, commits, tools, known_ids, by_bridge, notes_refs, now,
                )
                written = db.upsert_session_commits(rows)
                result.rows_written += written
                for r in rows:
                    result.by_source[r.source] = result.by_source.get(r.source, 0) + 1
                result.sessions_scanned += 1
            checked = _refresh_rework(conn, root, now, rework_budget)
            rework_budget -= checked
            result.rework_commits_checked += checked
        except Exception as exc:  # noqa: BLE001 - one repo must not sink the pass
            from tokenjam.core.db import is_fatal_db_error

            if is_fatal_db_error(exc):
                raise
            logger.warning("session -> commit matching failed for %s: %s", root, exc)
            continue
        # Watermark: only after the repo's sessions were actually processed.
        with (lock if lock is not None else _NullLock()):
            for s in group:
                conn.execute(
                    "INSERT INTO session_commit_scans (session_id, ended_at, scanned_at) "
                    "VALUES ($1,$2,$3) ON CONFLICT (session_id) DO UPDATE SET "
                    "ended_at = EXCLUDED.ended_at, scanned_at = EXCLUDED.scanned_at",
                    [s.session_id, s.ended_at, now],
                )
    return result


# --- The matcher on the daemon, on request ------------------------------------------

_MATCH_LOCK = threading.Lock()
_MATCH_THREAD: threading.Thread | None = None
_MATCH_LAST: dict[str, Any] = {}


def start_match(
    db_factory: Callable[[], Any], config: Any = None,
) -> tuple[threading.Thread, bool]:
    """Run :func:`match_sessions_to_commits` on a daemon thread with its OWN
    backend, the way the scan cycle and the transcript catch-up run.

    Returns ``(thread, started)``: the thread to wait on, and whether this
    call started it (``False`` means one was already running and the caller
    joins that one instead of starting a second pass over the same
    watermarks). The result of the last completed pass is kept in
    :func:`last_match` for the daemon route to report.

    This is what `POST /api/v1/shipped/match` dispatches, so a CLI that
    found the daemon holding the DuckDB lock (`tj optimize shipped`) can
    still get the join refreshed instead of a `ConnectionException` (issue
    #770, fix 4). A DuckDB fatal is classified where it is recognised and
    recovered off the process-wide record, never swallowed as one job's
    warning (Critical Rule 45).
    """
    global _MATCH_THREAD

    with _MATCH_LOCK:
        if _MATCH_THREAD is not None and _MATCH_THREAD.is_alive():
            return _MATCH_THREAD, False

        def _run() -> None:
            backend = None
            try:
                backend = db_factory()
                result = match_sessions_to_commits(backend, config)
                _MATCH_LAST.clear()
                _MATCH_LAST.update({
                    "sessions_scanned": result.sessions_scanned,
                    "rows_written": result.rows_written,
                    "repos_indexed": result.repos_indexed,
                    "repos_missing": result.repos_missing,
                    "by_source": dict(result.by_source),
                    "finished_at": utcnow().isoformat(),
                    "error": None,
                })
            except Exception as exc:  # noqa: BLE001 - classified below
                from tokenjam.core.db import handle_if_fatal

                if not handle_if_fatal(exc, what="session -> commit match"):
                    logger.warning("session -> commit match failed", exc_info=True)
                _MATCH_LAST.clear()
                _MATCH_LAST.update({"error": str(exc)[:300], "finished_at": utcnow().isoformat()})
            finally:
                from tokenjam.core.db import recover_if_fatal_noted

                recover_if_fatal_noted(what="session -> commit match")
                if backend is not None:
                    try:
                        backend.close()
                    except Exception:  # noqa: BLE001
                        pass

        thread = threading.Thread(target=_run, name="tj-commit-match", daemon=True)
        _MATCH_THREAD = thread
        thread.start()
        return thread, True


def last_match() -> dict[str, Any]:
    """The last completed on-request pass (empty before the first)."""
    return dict(_MATCH_LAST)


def _oldest_session_start(conn, root: str, now: datetime) -> datetime:
    row = conn.execute(
        "SELECT MIN(started_at) FROM sessions WHERE repo_root = $1", [root],
    ).fetchone()
    if row is None or row[0] is None:
        return now
    return _utc(row[0]) or now


class _NullLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


# --- Read side -------------------------------------------------------------------

# One row per (session, commit), always: the default-branch index can hold the
# same sha under several roots of one remote (a clone and a worktree both
# indexed), so membership is an EXISTS, never a join that would multiply rows.
_STATE_SQL = """
    WITH joined AS (
        SELECT sc.session_id, sc.commit_sha, sc.confidence, sc.source,
               EXISTS (SELECT 1 FROM repo_commits rc
                       WHERE rc.commit_sha = sc.commit_sha
                         AND (rc.repo_root = s.repo_root OR rc.repo_remote = s.repo_remote))
                   AS on_default,
               EXISTS (SELECT 1 FROM repo_commits rv
                       WHERE rv.reverts_sha IS NOT NULL
                         AND sc.commit_sha LIKE rv.reverts_sha || '%'
                         AND (rv.repo_root = s.repo_root OR rv.repo_remote = s.repo_remote))
                   AS reverted
        FROM session_commits sc
        JOIN sessions s ON s.session_id = sc.session_id
        WHERE sc.confidence IN ('deterministic', 'inferred')
    )
"""


def _state_from_counts(joined: int, on_default: int, reverted: int) -> str:
    if joined == 0:
        return STATE_UNSHIPPED
    # `on_default` is already "on the default branch AND not reverted".
    if on_default > 0:
        return STATE_SHIPPED
    if reverted >= joined:
        return STATE_REVERTED
    return STATE_COMMITTED


def session_shipped_state(conn, session_id: str) -> str:
    """The derived shipped state of one session (brief §3), pure SQL."""
    row = conn.execute(
        "SELECT repo_root, repo_remote FROM sessions WHERE session_id = $1", [session_id],
    ).fetchone()
    if row is None or (row[0] is None and row[1] is None):
        return STATE_NO_REPO
    counts = conn.execute(
        _STATE_SQL + """
        SELECT COUNT(*), COUNT(*) FILTER (WHERE on_default AND NOT reverted),
               COUNT(*) FILTER (WHERE reverted)
        FROM joined WHERE session_id = $1
        """,
        [session_id],
    ).fetchone()
    return _state_from_counts(int(counts[0]), int(counts[1]), int(counts[2]))


def session_commits_payload(db: Any, session_id: str) -> list[dict[str, Any]]:
    """`commits: [{sha, confidence, source, ...}]` for a session surface."""
    out = []
    for c in db.get_session_commits(session_id):
        out.append({
            "sha": c.commit_sha,
            "confidence": c.confidence,
            "source": c.source,
            "committed_at": c.committed_at.isoformat() if c.committed_at else None,
            "author_email": c.author_email,
            "match_delta_s": c.match_delta_s,
        })
    return out


def shipped_states(conn, session_ids: list[str]) -> dict[str, dict[str, Any]]:
    """`session_id -> {shipped_state, commit_count}` for a list surface, in one
    query rather than one per row."""
    if not session_ids:
        return {}
    placeholders = ", ".join(f"${i + 1}" for i in range(len(session_ids)))
    rows = conn.execute(
        _STATE_SQL + f"""
        SELECT s.session_id,
               (s.repo_root IS NOT NULL OR s.repo_remote IS NOT NULL) AS has_repo,
               COUNT(j.commit_sha),
               COUNT(j.commit_sha) FILTER (WHERE j.on_default AND NOT j.reverted),
               COUNT(j.commit_sha) FILTER (WHERE j.reverted)
        FROM sessions s LEFT JOIN joined j ON j.session_id = s.session_id
        WHERE s.session_id IN ({placeholders})
        GROUP BY s.session_id, has_repo
        """,
        session_ids,
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for sid, has_repo, joined, on_default, reverted in rows:
        state = (
            _state_from_counts(int(joined), int(on_default), int(reverted))
            if has_repo else STATE_NO_REPO
        )
        out[sid] = {"shipped_state": state, "commit_count": int(joined)}
    return out


@dataclass
class ShippedSummary:
    """The read-side figures every shipped surface renders (contracts §4)."""
    window_days: float = 0.0
    sessions_total: int = 0
    sessions_shipped: int = 0
    sessions_unshipped: int = 0
    sessions_committed: int = 0
    sessions_no_repo: int = 0
    cost_shipped_usd: float = 0.0
    cost_unshipped_usd: float = 0.0
    cost_committed_usd: float = 0.0
    cost_rework_usd: float | None = None
    cost_loop_usd: float | None = None
    coverage: float | None = None
    commits_joined: int = 0
    commits_on_default: int = 0
    top_unshipped: list[dict[str, Any]] = field(default_factory=list)
    top_reworked: list[dict[str, Any]] = field(default_factory=list)
    rework_basis: str = ""
    loop_basis: str = ""
    caveat: str = SHIPPED_CAVEAT

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _repo_display(remote: str | None) -> str | None:
    from tokenjam.core.repo_context import repo_name_from_url

    return repo_name_from_url(remote)


def shipped_summary(
    conn,
    since: datetime,
    until: datetime,
    *,
    agent_id: str | None = None,
    persona_scope: str | None = None,
) -> ShippedSummary:
    """Shipped vs unshipped over sessions that STARTED in `[since, until)`.

    Pure SQL over the ledger tables; never shells out, so it is safe on a
    request path. A session with no repo context is counted in
    `sessions_no_repo` and in nothing else: it was never analysed, and folding
    it into "unshipped" would publish a burn figure over a population the
    matcher could not see (Critical Rule 30).
    """
    clauses = ["s.started_at >= $1", "s.started_at < $2"]
    params: list[Any] = [since, until]
    if agent_id:
        params.append(agent_id)
        clauses.append(f"s.agent_id = ${len(params)}")
    add_persona_clause(clauses, persona_scope, column="s.agent_id")
    where = " AND ".join(clauses)

    rows = conn.execute(
        _STATE_SQL + f"""
        SELECT s.session_id, s.started_at, s.repo_remote, COALESCE(s.branch_end, s.branch_start),
               COALESCE(s.total_cost_usd, 0.0),
               (s.repo_root IS NOT NULL OR s.repo_remote IS NOT NULL) AS has_repo,
               COUNT(j.commit_sha),
               COUNT(j.commit_sha) FILTER (WHERE j.on_default AND NOT j.reverted),
               COUNT(j.commit_sha) FILTER (WHERE j.reverted)
        FROM sessions s LEFT JOIN joined j ON j.session_id = s.session_id
        WHERE {where}
        GROUP BY ALL
        """,
        params,
    ).fetchall()

    summary = ShippedSummary(window_days=max((until - since).total_seconds() / 86400.0, 0.0))
    unshipped: list[dict[str, Any]] = []
    for sid, started, remote, branch, cost, has_repo, joined, on_default, reverted in rows:
        cost = float(cost or 0.0)
        if not has_repo:
            summary.sessions_no_repo += 1
            continue
        summary.sessions_total += 1
        summary.commits_joined += int(joined)
        summary.commits_on_default += int(on_default)
        state = _state_from_counts(int(joined), int(on_default), int(reverted))
        if state == STATE_SHIPPED:
            summary.sessions_shipped += 1
            summary.cost_shipped_usd += cost
        elif state == STATE_COMMITTED:
            summary.sessions_committed += 1
            summary.cost_committed_usd += cost
        else:
            summary.sessions_unshipped += 1
            summary.cost_unshipped_usd += cost
            unshipped.append({
                "session_id": sid,
                "cost_usd": round(cost, 6),
                "started_at": started.isoformat() if started else None,
                "repo": _repo_display(remote),
                "branch": branch,
                "state": state,
            })
    unshipped.sort(key=lambda r: -r["cost_usd"])
    summary.top_unshipped = unshipped[:TOP_N]
    summary.cost_shipped_usd = round(summary.cost_shipped_usd, 6)
    summary.cost_unshipped_usd = round(summary.cost_unshipped_usd, 6)
    summary.cost_committed_usd = round(summary.cost_committed_usd, 6)

    summary.coverage = _coverage(conn, since, until, where, params)
    _add_rework(conn, summary, where, params)
    _add_loop(conn, summary, since, until, agent_id, persona_scope)
    return summary


def _coverage(conn, since: datetime, until: datetime, where: str, params: list[Any]) -> float | None:
    """Contracts §1 coverage, OSS form: of the default-branch commits in the
    window on repos this window's sessions ran in, authored by those sessions'
    developers, the share that has a deterministic or inferred row. None when
    there is no such commit to cover."""
    row = conn.execute(
        f"""
        WITH scoped AS (SELECT DISTINCT s.repo_remote, s.repo_root, s.user_email
                        FROM sessions s WHERE {where} AND s.repo_root IS NOT NULL),
             pool AS (
                SELECT DISTINCT rc.commit_sha
                FROM repo_commits rc
                JOIN scoped sc ON (rc.repo_root = sc.repo_root OR rc.repo_remote = sc.repo_remote)
                WHERE rc.committed_at >= $1 AND rc.committed_at < $2
                  AND (sc.user_email IS NULL OR lower(rc.author_email) = lower(sc.user_email))
                  AND rc.reverts_sha IS NULL
             )
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE EXISTS (
                   SELECT 1 FROM session_commits j WHERE j.commit_sha = pool.commit_sha
                     AND j.confidence IN ('deterministic', 'inferred')))
        FROM pool
        """,
        params,
    ).fetchone()
    total = int(row[0] or 0)
    if total == 0:
        return None
    return round(int(row[1] or 0) / total, 4)


def _add_rework(conn, summary: ShippedSummary, where: str, params: list[Any]) -> None:
    rows = conn.execute(
        f"""
        WITH per_commit AS (
            SELECT sc.session_id, sc.commit_sha, f.path,
                   MAX(f.additions) AS additions, MAX(f.later_deletions) AS later_deletions
            FROM session_commits sc
            JOIN sessions s ON s.session_id = sc.session_id
            JOIN commit_file_stats f ON f.commit_sha = sc.commit_sha AND f.path <> ''
            WHERE {where} AND sc.confidence IN ('deterministic', 'inferred')
            GROUP BY sc.session_id, sc.commit_sha, f.path
        ),
        per_session AS (
            SELECT session_id, SUM(additions) AS adds, SUM(later_deletions) AS dels
            FROM per_commit GROUP BY session_id
        )
        SELECT ps.session_id, ps.adds, ps.dels, COALESCE(s.total_cost_usd, 0.0)
        FROM per_session ps JOIN sessions s ON s.session_id = ps.session_id
        """,
        params,
    ).fetchall()
    measured = [r for r in rows if r[1]]
    if not measured:
        summary.rework_basis = (
            "No joined commit in this window has rework evidence yet; the matcher "
            "takes it on its next pass."
        )
        return
    reworked = {r[0]: float(r[3] or 0.0) for r in measured if r[2] / r[1] >= REWORK_SHARE}
    summary.cost_rework_usd = round(float(sum(reworked.values())), 6)
    summary.rework_basis = (
        f"Measured over {len(measured)} session(s) whose joined commits have line "
        f"counts: a session is reworked when at least {int(REWORK_SHARE * 100)}% of "
        f"the lines its commits added were deleted again by other work within "
        f"{REWORK_WINDOW_DAYS} days (git numstat on the commit's own files; a "
        f"commit's own session is excluded). {len(reworked)} qualified."
    )
    if not reworked:
        return
    placeholders = ", ".join(f"${i + 1}" for i in range(len(reworked)))
    paths = conn.execute(
        f"""
        WITH hit AS (
            SELECT DISTINCT f.path, sc.session_id
            FROM session_commits sc
            JOIN commit_file_stats f ON f.commit_sha = sc.commit_sha AND f.path <> ''
            WHERE sc.session_id IN ({placeholders}) AND f.additions > 0
              AND f.later_deletions * 1.0 / f.additions >= {REWORK_SHARE}
        )
        SELECT hit.path, COUNT(*) AS sessions, SUM(COALESCE(s.total_cost_usd, 0.0)) AS cost_usd
        FROM hit JOIN sessions s ON s.session_id = hit.session_id
        GROUP BY hit.path ORDER BY cost_usd DESC, sessions DESC LIMIT {TOP_N}
        """,
        list(reworked),
    ).fetchall()
    summary.top_reworked = [
        {"path": p, "sessions": int(n), "cost_usd": round(float(c or 0.0), 6)} for p, n, c in paths
    ]


_EDIT_TOOLS = ("Edit", "Write", "MultiEdit")


def _add_loop(
    conn, summary: ShippedSummary, since: datetime, until: datetime,
    agent_id: str | None, persona_scope: str | None,
) -> None:
    """Loop cost (contracts §1): consecutive Edit/Write calls on one path whose
    content did not change between them, priced at the LLM turn that emitted
    each repeat."""
    clauses = ["t.start_time >= $1", "t.start_time < $2", "t.tool_name IN ('Edit', 'Write', 'MultiEdit', 'Read')"]
    params: list[Any] = [since, until]
    if agent_id:
        params.append(agent_id)
        clauses.append(f"t.agent_id = ${len(params)}")
    add_persona_clause(clauses, persona_scope, column="t.agent_id")
    rows = conn.execute(
        f"""
        SELECT t.session_id, t.start_time, t.tool_name, t.attributes,
               t.parent_span_id, COALESCE(p.cost_usd, 0.0)
        FROM spans t LEFT JOIN spans p ON p.span_id = t.parent_span_id
        WHERE {" AND ".join(clauses)}
        ORDER BY t.session_id, t.start_time
        """,
        params,
    ).fetchall()
    if not rows:
        summary.loop_basis = "No Edit or Write tool calls in this window."
        return
    charged: set[str] = set()
    loop_cost = 0.0
    loops = 0
    weak = 0
    have_content = False
    prev: dict[str, tuple[str, str | None, str | None]] = {}   # session -> (tool, path, hash)
    for sid, _ts, tool, raw, parent, cost in rows:
        inp = _tool_input(raw)
        path = inp.get("file_path") or inp.get("path")
        if tool == "Read":
            prev.pop(sid, None)
            continue
        body = inp.get("new_string") if tool == "Edit" else inp.get("content") if tool == "Write" else inp.get("edits")
        h = content_hash(body)
        if h is not None:
            have_content = True
        last = prev.get(sid)
        if last is not None and path and last[1] == path:
            same = (h is not None and last[2] == h) or (h is None and last[2] is None and tool == "Edit")
            if same:
                loops += 1
                if h is None:
                    weak += 1
                if parent and parent not in charged:
                    charged.add(parent)
                    loop_cost += float(cost or 0.0)
        prev[sid] = (tool, path, h) if path else (tool, None, None)
    summary.cost_loop_usd = round(loop_cost, 6)
    if have_content:
        summary.loop_basis = (
            f"{loops} repeat edit(s): consecutive Edit/Write calls on the same path "
            f"with identical content, priced at the model turn that issued each repeat."
            + (f" {weak} of them had no captured content and were counted on path alone "
               f"(consecutive edits with no Read between)." if weak else "")
        )
    else:
        summary.loop_basis = (
            f"{loops} repeat edit(s) counted on path alone (consecutive Edit calls on one "
            f"path with no Read between): tool-input content is not captured, so identical "
            f"content could not be checked. Enable [capture] tool_inputs for the exact form."
        )


__all__ = [
    "MATCH_WINDOW_S",
    "REWORK_WINDOW_DAYS",
    "SHIPPED_CAVEAT",
    "STATE_COMMITTED",
    "STATE_NO_REPO",
    "STATE_REVERTED",
    "STATE_SHIPPED",
    "STATE_UNSHIPPED",
    "CommitToolIndex",
    "MatchResult",
    "ShippedSummary",
    "bridge_suffix",
    "is_git_commit_command",
    "last_match",
    "match_sessions_to_commits",
    "parse_trailers",
    "reverted_sha",
    "session_commits_payload",
    "session_shipped_state",
    "shipped_states",
    "shipped_summary",
    "start_match",
    "trailer_rank",
]
