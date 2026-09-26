# The ledger: sessions joined to commits

Every other part of TokenJam answers what a session cost. The ledger answers what it left
behind. It records which repo a session ran in and who ran it, then joins each session to the
commits it produced at a labelled confidence, so a cost figure can sit next to an output figure.

Nothing here is a saving. The ledger reports measured spend against measured output, and it says
plainly where the evidence is thin.

- The analyzer that reads it: [`tj optimize shipped`](../optimize/shipped.md)
- The git hooks that make the join stronger: [hooks and notes](hooks-and-notes.md)
- Forwarding the ledger to a team view: [the Cloud bridge](cloud-bridge.md)

## Repo context and developer identity

When a session is ingested (live capture, the daemon's transcript catch-up, or a backfill), tj
resolves the directory the session ran in into repo context and stamps it on the session row:

| Column | What it holds |
|---|---|
| `repo_remote` | the normalised origin remote, `https://github.com/org/repo`, no `.git`, no credentials, no query or fragment |
| `repo_root` | the absolute path of the repo root |
| `branch_start`, `branch_end` | the branch at session start and at session end |
| `head_sha_start`, `head_sha_end` | HEAD at session start and at session end, when the source records it |
| `user_email` | the git author email configured for that directory |
| `developer_id` | `sha256(lower(user_email))[:16]`, a stable pseudonymous id |

Rules that hold everywhere this runs:

- **Read-only git, two-second timeout, never on a request path.** The API and Lens read stored
  columns. Only the backfill, the daemon and `tj init` shell out.
- **A failure yields absence, never a placeholder.** No git on `PATH`, no repo, an unborn HEAD, a
  cwd that no longer exists: each of those leaves the affected fields null, so a later join can
  never match on an invented value.
- **Temp directories and tj's own invocation cwd are skipped**, since a throwaway checkout's
  remote would attribute work to the wrong project.

Sessions that were ingested before repo context existed are refilled by the daemon's transcript
catch-up, and by `tj init --cloud` before its first push, so an older install fills in rather than
starting from the day you upgraded.

`tj otel-resource-attrs` prints the resource attributes for the current project, including the
ledger ones. The full attribute list is in
[architecture.md](../architecture.md#otel-semconv-extensions-repo-context-and-developer-identity).

## The join

The matcher runs on the daemon's analyzer pass, and ahead of a direct-DB `tj optimize` when no
daemon is running. For each session that changed since its last scan it reads the commits inside
the session's window (from 60 seconds before the session started to 15 minutes after it ended) and
joins each one at the best confidence the evidence supports.

### The confidence enum

`confidence` is one of `deterministic`, `inferred` or `estimated`. `source` names the evidence.
TokenJam OSS writes the first two; `estimated` is a Cloud-side allocation with no session evidence
behind it, and never appears in a local database.

| confidence | source | What it means |
|---|---|---|
| `deterministic` | `tool_span_git_log` | a `git commit` Bash tool call inside the session ran within 30 seconds of the commit's own timestamp. The strongest evidence: the session's own tool call produced the commit |
| `deterministic` | `trailer_session` | the commit body carries `TokenJam-Session: <id>` or `Claude-Session: <url>` naming a session tj has ingested, and no stronger evidence points somewhere else |
| `deterministic` | `git_note` | a git note on the commit names a session tj has ingested (see [notes](hooks-and-notes.md#which-refs-are-read)) |
| `inferred` | `trailer_window` | the commit carries an AI co-author trailer, sits on the session's own branch inside the session window, and has no stronger evidence. When both emails are known, the commit author must match the session's |

**Inherited trailers rank below the tool call that ran the commit.** A trailer is written from the
shell environment the commit inherited, and a subagent inherits its parent's session id, so a
commit the subagent's own Bash tool ran still carries the parent's trailer. When a trailer names a
different session than the tool span does, the trailer is inherited evidence and is stored at
`inferred`. Without that rule a parent session absorbs its subagents' commits at top confidence.

A `Claude-Session:` trailer names the bridge session id rather than the local transcript uuid,
which is why the Claude Code backfill persists `bridge_session_id`: it is what makes such a trailer
resolvable.

### Writes are idempotent and never downgrade

Each `(session_id, commit_sha)` pair is written once, at the best confidence found. A later pass
may upgrade a row from `inferred` to `deterministic` when better evidence turns up; nothing ever
moves a row down. Re-running the matcher over the same history writes nothing new. Sessions less
than 30 days old are rescanned so that evidence arriving late (a note written after the fact, a
branch merged the next day) still lands.

### Where the rows live

`session_commits` in your local DuckDB holds the join, one row per session and commit with its
confidence, source, author email, commit time and the match delta. Three side tables keep every
read in SQL rather than in git: an index of the default branch, per-commit file statistics for the
rework figure, and a per-session scan watermark. A request handler never shells out, so the shipped
state of a session is derived at read time from those tables.

## Session states

Derived in SQL from the join plus the default-branch index:

| State | Means |
|---|---|
| **shipped** | at least one joined commit is reachable from the default branch and was not reverted |
| **committed** | joined commits exist, none of them on the default branch yet. Its own state, never counted as shipped |
| **reverted** | every joined commit was reverted by a commit naming its sha |
| **unshipped** | no joined commit at all |
| **no repo** | the session had no repo context, so it was never analysed. Reported separately, never folded into either side |

## Coverage

Coverage is the share of default-branch commits in the window that carry a `deterministic` or
`inferred` row, restricted to the repos of the window's sessions, the developers who ran them, and
excluding reverts. It is shown next to the shipped figures because it is the honest bound on them:
low coverage means tj is seeing part of your work, not that the rest was unproductive.

The fastest way to raise coverage is [`tj init --hooks`](hooks-and-notes.md), which makes commits
from a plain shell carry a session trailer.

## What reads the ledger

| Surface | What it shows |
|---|---|
| `tj optimize shipped` | the full finding: shipped counts, the cost of each state, coverage, biggest unshipped sessions, rework and loop cost |
| `tj status --agent <id>` | a one-line "Shipped N of M sessions" card |
| `GET /api/v1/shipped` | the same summary the analyzer reads |
| `GET /api/v1/sessions/{id}` | `commits` and `shipped_state` for one session |
| Lens | the shipped column and filter on Sessions, commit chips with the confidence glyph on a session page, the Dashboard's "Shipped this week" tile, the Optimize page's Shipped card |
| TokenJam Cloud | the rows forwarded by [the bridge](cloud-bridge.md), if you connect one |

## The honesty line

Unshipped is measured cost of sessions with no joined commit. A session can ship value without a
commit: research, review, operations. The finding carries no recoverable-savings field, sits
outside the recoverable-waste rollup, and prints that caveat on every surface. Read it as a
question to ask, not as a number to cut.
