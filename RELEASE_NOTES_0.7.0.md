# TokenJam 0.7.0

> Working draft for the GitHub release. Not part of the shipped package.

The largest release since 0.6. TokenJam has always been able to tell you what your agent sessions
cost. This one starts answering the other half of the question: what those sessions actually
produced.

```bash
pipx install --upgrade tokenjam
tj optimize shipped
```

## The shipped-value ledger

tj now records which repo a session ran in and who ran it, then joins each session to the commits
it produced.

- **Repo context and developer identity on every session.** Remote, repo root, branch and HEAD at
  session start and end, the git author email, and a hashed `developer_id` derived from it. Git
  shell-outs are read-only, time-bounded, and never on a request path. A value that cannot be
  resolved is absent rather than guessed, so nothing joins on an invented key.
- **The session-to-commit join.** A new `session_commits` table, populated on the daemon's
  background pass. Four match sources: a `git commit` tool call inside the session (`tool_span_git_log`),
  a `TokenJam-Session:` or `Claude-Session:` trailer (`trailer_session`), a git note naming a
  session (`git_note`), and an AI co-author trailer inside the session window (`trailer_window`).
  The first three are `deterministic`; the last is `inferred`. A row is written once at the best
  confidence found and never downgraded.
- **The `shipped` analyzer.** `tj optimize shipped` reports how many of the window's sessions
  shipped a commit to the default branch, the measured cost of the ones that shipped nothing,
  rework cost (commits whose added lines were mostly deleted again within 14 days), and loop cost
  (repeated Edit/Write calls on one path with no content change). Sessions also carry a state:
  shipped, committed, reverted, unshipped, or no repo.
- **Coverage next to every figure.** The share of your default-branch commits that tj could join.
  Low coverage means tj is seeing part of your work. It does not mean the rest was unproductive.
- **Surfaces.** `tj optimize shipped` in the terminal, a shipped line on the `tj status` agent
  card, `GET /api/v1/shipped`, commits and shipped state on the session detail route, and in Lens
  the Sessions column and filter, the commit chips with a confidence glyph, the "Shipped this week"
  tile and the Optimize page's Shipped card.

Nothing in the shipped finding is a saving. Every dollar in it is measured spend on sessions with
no joined commit, and a session can ship value without a commit: research, review, operations. The
caveat prints on every surface, and the finding sits outside the recoverable-waste rollup.

## First run and onboarding

- **`tj init` is the primary name.** One command object registered under two names, so `tj init`
  and `tj onboard` take the same flags, ask the same questions and write the same config. Nothing
  about `tj onboard` changed.
- **`tj init --hooks`** installs a `prepare-commit-msg` hook in the current repo so a commit you
  make from a plain shell mid-session carries a `TokenJam-Session:` trailer. That joins it to the
  session at deterministic confidence, which is the cheapest way to raise coverage. The hook lives
  in a managed block, keeps a hook you already have, and refuses a `core.hooksPath` inside the
  worktree because that is a tracked directory.
- **`tj init --notes`** adds a `post-commit` hook that writes the session's measured cost to
  `refs/notes/tokenjam`. It implies `--hooks` and is off unless you ask for it. tj reads
  `refs/notes/ai` (Git AI) and `refs/notes/exceeds-ink` too, so a repo already annotated by one of
  those gets a stronger join with no setup.
- **`tj commit-note [sha]`** writes that note by hand. It never fails the commit it runs from.
- **`tj init --enforce`** turns on the enforcement proxy in suggest mode and prints what it does
  and does not touch. Suggest mode forwards every request unmodified and records what a policy
  would have done. Nothing is blocked or rewritten until you approve it. Subscription-plan traffic
  is never intercepted and subscription OAuth credentials are never proxied or forwarded.
- **`tj init --cloud <key> --org <org>`** connects the machine to a TokenJam Cloud organization.
  The daemon then forwards spans, sessions and the commit joins, resumable from a per-stream
  high-water mark, so a machine that was offline catches up on its own. Before the first byte
  leaves, the command prints what does and does not cross and waits for a yes. Token counts, model
  names, cost, timestamps, tool names, file paths, identifiers, the hashed developer id and the git
  author email leave. Prompt text, completions, tool outputs, file contents, diffs and secrets do
  not, unless you turn on both your local capture toggles and `forward_content`.
  `tj init --cloud off` stops forwarding in place.
- **Existing installs fill in.** The daemon refills repo context on sessions that were ingested
  before any of this existed, and `tj init --cloud` refills and matches before its first push. You
  get history rather than a ledger that starts the day you upgraded.
- **`tj doctor` and `tj uninstall` cover the new surfaces.** doctor reports hook state per repo,
  whether a hand commit right now would carry a trailer, and whether the Cloud endpoint is
  reachable and the key accepted. uninstall strips the managed blocks from every hooks directory tj
  wrote to.

## Correctness

- An inherited session trailer now ranks below the tool span that ran the commit. A subagent
  inherits its parent's session id, so a commit the subagent's own Bash tool ran carried the
  parent's trailer and the parent absorbed it at top confidence. Such a row is now written at
  `inferred`.
- Remote normalisation drops the query and fragment, and branch endpoints are picked by timestamp
  rather than by file order (#762).
- The Cloud forwarder resumes in arrival order rather than event order, so a backfill that inserts
  spans older than everything already sent no longer loses them. Resume marks are scoped to the
  organization and to the database file. A batch the receiver refuses is split until the bad record
  is isolated, then skipped and counted, so one row cannot wedge the stream.
- `sessions.updated_at` moves only when a session value actually changes, which stops the bridge
  re-sending untouched rows forever.
- The transcript catch-up runs once per process, over a window-bounded tool-span index, with
  producers keyed by repo.
- The serve-mode shim gates its write routes with the ingest secret rather than the optional API
  key, and the backfill hand-off goes through the data-access seam instead of around it.
- `tj init --cloud` says so when history leaves without an install id.
- Lens: the hero call to action resolves to the surface that renders the analyzer, instead of a
  retired route.

## Docs

New pages for the ledger, the commit hooks and notes, and the Cloud bridge, plus the ledger OTel
attributes in the architecture reference. See `docs/ledger/`.

## Upgrading

`pipx install --upgrade tokenjam`, or `pip install -U tokenjam` in a venv. Schema changes are
additive, so an existing database is migrated in place. Everything new in this release is opt-in:
the hooks, the notes and the Cloud bridge stay off until you run the flag that turns them on.
