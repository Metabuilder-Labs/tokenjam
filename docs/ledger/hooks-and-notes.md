# Commit hooks and git notes

A commit you make by hand, from a plain shell rather than the agent's Bash tool, leaves no tool
span for [the ledger](overview.md) to key on. It joins the session at `inferred` confidence if an
AI co-author trailer happens to be on it, and usually not at all. Two optional git hooks close that
gap.

Both are off until you ask for them.

```bash
tj init --hooks    # prepare-commit-msg: stamp the session id on your commits
tj init --notes    # also post-commit: write the session's cost to refs/notes/tokenjam
```

`--notes` implies `--hooks`. Both compose with the rest of onboarding, so
`tj init --claude-code --hooks` is one run. Each step is idempotent: re-running in the same repo
reports that the block is already current and changes nothing.

## `--hooks`: the commit trailer

The `prepare-commit-msg` hook appends one trailer to the message:

```
TokenJam-Session: cse_01J9X...
```

The ledger resolves that to `trailer_session`, which is deterministic confidence. The trailer is
appended only when it is absent, so a merge, a squash, an amend or a rebase never accumulates
duplicates, and every trailer you already had is preserved.

**Where the session id comes from.** Claude Code exports `CLAUDE_CODE_SESSION_ID` to the shells it
spawns, and the hook prefers it. For a commit from a terminal Claude Code did not spawn, the hook
reads `~/.tj/active_sessions.json`, which the zero-token statusline writes on every render: one
entry per git worktree root, holding the session id, the cwd and a timestamp. Entries older than
six hours are pruned, and the hook trusts an entry only while it is under 30 minutes old. With
neither source available the hook writes no trailer and the commit proceeds untouched.

## `--notes`: the cost note

The `post-commit` hook runs `tj commit-note <sha>` for a commit that carries a `TokenJam-Session:`
trailer. That writes one JSON object to `refs/notes/tokenjam`:

```json
{
  "v": 1,
  "session_id": "cse_01J9X...",
  "tool": "claude_code",
  "model": "claude-sonnet-4-5",
  "cost_usd": 0.84,
  "pricing_mode": "api",
  "confidence": "deterministic",
  "source": "trailer_session"
}
```

`cost_usd` is the session's measured cost and `pricing_mode` rides with it, so a reader never
renders a subscription session's figure as per-token spend.

`tj commit-note` can also be run by hand on any commit that carries the trailer. It never fails the
commit it is called from: every outcome exits 0, and `-v` says what it did or why it did nothing.

### Which refs are read

The matcher reads notes from three refs and writes to exactly one:

| Ref | Written by | Read |
|---|---|---|
| `refs/notes/tokenjam` | tj, only with `--notes` | yes |
| `refs/notes/ai` | Git AI | yes |
| `refs/notes/exceeds-ink` | Exceeds | yes |

A note from any of the three that names a session tj has ingested joins at `git_note` confidence.
Reading other tools' notes costs nothing and means a repo already annotated by one of them gets a
stronger join on day one. Notes are not pushed anywhere on your behalf; `git push origin
refs/notes/tokenjam` is yours to run if you want them shared.

## What gets written to your repo

Only the two hook files, and only inside a managed block:

```sh
# >>> tokenjam commit trailer (managed) >>>
...
# <<< tokenjam commit trailer <<<
```

- The block is written into the hooks directory git will actually consult, which honours
  `core.hooksPath`.
- **A `core.hooksPath` inside the worktree is refused.** That is a tracked directory (the husky
  pattern), and tj never edits tracked files. The command says so and does nothing.
- A hook file whose shebang is not a POSIX shell is left alone, since a Python hook cannot host a
  `sh` block. The command reports it and moves on.
- An existing hook of your own keeps working: install strips any previous tj block and writes one
  fresh block after the shebang, leaving the rest of the file untouched.
- Every hooks directory tj has written to is listed in `~/.tj/commit_hooks.json`, so uninstall can
  find them all.

The commit message is never touched by anything except the `prepare-commit-msg` hook, and the only
git write in the whole ledger is the note on our own ref.

## Checking and removing

```bash
tj doctor       # reports hook state per repo: current, stale, damaged, foreign, or absent
tj uninstall    # strips the managed blocks from every hooks directory tj wrote to
```

`tj doctor` also answers the question the hooks exist for: would a commit made by hand right now
carry a session trailer. If the active-session record is stale or missing it says that, rather than
reporting the hook as healthy because the file is present.
