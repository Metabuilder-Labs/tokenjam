# `tokenjam` — zero-install TokenJam launcher (command: `tj`)

```bash
npx tokenjam
```

That one command reads your `~/.claude/projects/*.jsonl` session logs
and shows you **where your
Claude Code quota actually goes** — quota composition (re-reading context vs.
net-new work) plus a session timeline. No pip env, no daemon, no onboarding.

This npm package is a **thin launcher** for the real CLI, which is the Python
package [`tokenjam`](https://pypi.org/project/tokenjam/) (command: `tj`). `npx tokenjam`
shells out to the first available Python runner:

1. `uvx --from tokenjam tj …`
2. `pipx run --spec tokenjam tj …`
3. an already-installed `tj` on your `PATH`

All arguments pass straight through, so `npx tokenjam quickstart --since 7d`,
`npx tokenjam context`, `npx tokenjam optimize`, etc. all work.

## Go deeper

`npx tokenjam` is the no-setup front door. When you want live capture, the local
dashboard, and the MCP server for Claude Code, install the full CLI and onboard:

```bash
pipx install tokenjam
tj onboard
```

See the [TokenJam README](https://github.com/Metabuilder-Labs/tokenjam) for the
full feature set.

## Requirements

A Python runner — [`uv`](https://docs.astral.sh/uv/) (recommended) or
[`pipx`](https://pipx.pypa.io/). If neither is present, `npx tokenjam` prints install
guidance.
