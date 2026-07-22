# tokenjam

<div align="center">

<img src="https://raw.githubusercontent.com/Metabuilder-Labs/tokenjam/main/docs/brand/tokenjam-repo-header.png" alt="TokenJam cuts what your AI agents cost by ending the mistakes they keep repeating: it finds the failures your agent re-hits and fixes them, reversibly and human-gated. Runs 100% local." width="760">

[![npm](https://img.shields.io/npm/v/tokenjam?color=3d8eff&labelColor=0d1117)](https://www.npmjs.com/package/tokenjam)
[![npm downloads](https://img.shields.io/npm/dm/tokenjam?color=3d8eff&labelColor=0d1117&label=downloads)](https://www.npmjs.com/package/tokenjam)
[![PyPI](https://img.shields.io/pypi/v/tokenjam?color=3d8eff&labelColor=0d1117)](https://pypi.org/project/tokenjam/)
[![License: MIT](https://img.shields.io/badge/license-MIT-3d8eff?labelColor=0d1117)](https://github.com/Metabuilder-Labs/tokenjam/blob/main/LICENSE)

</div>

TokenJam cuts what your AI agents cost by ending their repeated mistakes: it finds the failures your agent keeps re-hitting and fixes them at the harness level, reversibly and human-gated. It ingests agent telemetry from a multitude of sources (Claude Code sessions, Claude Agent SDK apps, any OTel source) into a local DuckDB, so you get the most out of the tokens you pay for. This package is the zero-install launcher: no pip environment, no manual config.

```bash
npx tokenjam        # a read-only report of where your Claude Code quota goes, no install, nothing kept
npx tokenjam onboard   # set up the loop that finds and fixes recurring mistakes, or: pipx install tokenjam && tj onboard
```

## What you get

`tj onboard` is guided setup: it writes a config, generates an ingest secret, and asks how you use AI agents (Claude Code, Codex, or your own SDK/API agents) to wire the right path. For Claude Code and Codex that means backfilling recent history and installing a statusline and hooks for live capture; restart and you're live. Onboarding ends by showing the mistakes your agent keeps making and offers to enable your first fix, and it unlocks the loop, all thirteen analyzers, the Lens dashboard, and the zero-token statusline in one command.

## Commands

All arguments pass straight through to the Python CLI, so any `tj` subcommand and flag works here too.

| Command | What it does |
|---|---|
| `npx tokenjam` | Zero-install report of where your Claude Code quota goes: re-reads vs. net-new work, plus a session timeline. Nothing installed, nothing kept. |
| `npx tokenjam onboard` | Guided setup for the loop: writes a config, generates an ingest secret, and optionally installs the background daemon for live capture. |
| `npx tokenjam context` | Where your tokens go: re-read vs. net-new share, recurring inclusions, `/compact` candidates. |
| `npx tokenjam optimize` | Savings candidates: model downsizing, cache opportunities, prompt trimming, workflow reuse, subagent right-sizing. |

## Go deeper

`tj onboard` sets up live capture, the local Lens dashboard, and the zero-token statusline in one command. From there:

```bash
tj optimize   # cost-saving candidates from your actual usage
tj serve      # open the Lens dashboard at http://127.0.0.1:7391/
```

- Full feature set, the loop, thirteen analyzers, and Lens screenshots: [github.com/Metabuilder-Labs/tokenjam](https://github.com/Metabuilder-Labs/tokenjam)
- Product site and docs: [tokenjam.dev](https://tokenjam.dev)

## How the launcher works

This npm package is a thin launcher, not the real CLI. `npx tokenjam` shells out to the first available Python runner:

1. `uvx --from tokenjam tj …`
2. `pipx run --spec tokenjam tj …`
3. an already-installed `tj` on your `PATH`

The real CLI is the Python package [`tokenjam`](https://pypi.org/project/tokenjam/) (command: `tj`).

## Requirements

A Python runner: [`uv`](https://docs.astral.sh/uv/) (recommended) or [`pipx`](https://pipx.pypa.io/). If neither is present, `npx tokenjam` prints install guidance instead of failing silently.
