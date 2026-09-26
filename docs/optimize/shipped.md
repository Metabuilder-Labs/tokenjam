# Shipped

Product name: **Shipped**. Internal/CLI name: `shipped`.

```bash
tj optimize shipped
```

Every other analyzer asks what a session cost. This one asks what it left
behind. tj joins each session to the commits it produced, at a labelled
confidence, and reports how many of the window's sessions shipped a commit to
the default branch, what the sessions that shipped nothing cost, and how much
of that shipped work was rewritten within two weeks. It is the Measure ROI
half of the product: a cost figure next to an output figure.

## How a session is joined to a commit

The join runs on the daemon's background pass (and before a direct-DB
`tj optimize`), never on a request. It is read-only git, bounded, and skipped
for a repo whose checkout no longer exists. Each `(session, commit)` pair is
written once at the best confidence found and never downgraded.

| Confidence | Source | Evidence |
|---|---|---|
| deterministic | `tool_span_git_log` | a `git commit` Bash tool call in the session within 30s of the commit |
| deterministic | `trailer_session` | the commit body carries `TokenJam-Session: <id>` or `Claude-Session: <url>` resolving to an ingested session |
| deterministic | `git_note` | a `refs/notes/ai` (Git AI), `refs/notes/exceeds-ink` or `refs/notes/tokenjam` note names a session tj ingested |
| inferred | `trailer_window` | an AI co-author trailer, on the session's own branch, inside the session window, with no stronger evidence |

A `Claude-Session:` trailer names the bridge session id, not the local session
uuid, so the Claude Code backfill persists `bridge_session_id` from the
transcript to make it resolvable.

A trailer that names a different session than a tool span does is inherited
evidence, not first-hand evidence: a subagent inherits its parent's session id,
so a commit the subagent's own Bash tool ran still carries the parent's
trailer. In that case the row is written at `inferred`, so a parent never
absorbs its subagents' commits at the top confidence.

`refs/notes/tokenjam` is written only when you opt in with `tj init --notes`;
the other two refs are read and never written. Full detail on the join, the
side tables and coverage is in [the ledger overview](../ledger/overview.md),
and the hooks that raise your confidence are in
[hooks and notes](../ledger/hooks-and-notes.md).

## Shipped state

Derived at read time, in SQL, from the join plus an index of the default
branch the same pass keeps current:

- **shipped**: a joined commit is reachable from the default branch and was not reverted
- **committed**: joined commits exist, none on the default branch yet (its own state, never counted as shipped)
- **reverted**: every joined commit was reverted (`Revert "..."` naming the sha)
- **unshipped**: no joined commit
- **no repo**: the session had no repo context and was never analysed

## What the finding carries

`sessions_shipped / sessions_total`, the measured cost of each state, `coverage`
(the share of your default-branch commits in the window that are joined to a
session), the largest unshipped sessions, `cost_rework_usd` (sessions where at
least half of the lines their commits added were deleted again within 14 days,
by other work), `cost_loop_usd` (consecutive Edit/Write calls on one path with
identical content, priced at the model turn that issued each repeat), and the
caveat.

## What it is not

It is not a saving. Every dollar is **measured** spend on sessions that left
no commit, and a session can ship value without a commit: research, review,
operations. The finding carries no `past_overspend_*` field, sits outside the
recoverable-waste rollup, and prints this on every surface:

> Unshipped is measured cost of sessions with no joined commit; a session can
> ship value without a commit (research, review, ops). Review before acting.

## Surfaces

`tj optimize shipped` (human and `--json`), the `tj status --agent <id>` card
("Shipped N of M sessions"), `GET /api/v1/shipped`, `commits` +
`shipped_state` on `GET /api/v1/sessions/{id}`, and in Lens the Sessions view
(shipped column and filter, commit chips with the confidence glyph on the
session page), the Dashboard's "Shipped this week" tile and the Optimize
page's Shipped card.
