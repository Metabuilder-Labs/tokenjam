---
description: Weekly GitHub traffic archive — orphan-branch design and the pattern for any scheduled repo-writing automation.
paths:
  - ".github/workflows/traffic-archive.yml"
  - "growth/**"
---

# Growth instrumentation — weekly traffic archive

`.github/workflows/traffic-archive.yml` runs every Sunday at 12:00 UTC (also `workflow_dispatch`) and
archives the GitHub Traffic API (views / clones / referrers / paths — the only sources with the
14-day retention problem) into a JSON file at `traffic/<year>-W<week>.json`. **The archive lives on
its own orphan branch `traffic-data`, not on `main`** — code and telemetry are deliberately separated,
and `main`'s branch protection stays intact.

Key facts:
- The workflow uses `${{ secrets.TRAFFIC_PAT }}` (fine-grained PAT, resource owner = `Metabuilder-Labs` org, `Administration: Read` on this repo) for the Traffic API read — the default `GITHUB_TOKEN` returns 403 there. The push step uses the default `GITHUB_TOKEN` since `traffic-data` is unprotected.
- The orphan-branch design was the close-out after several other paths (PAT-as-owner push, classic-protection bypass, rulesets bot-bypass) hit GitHub UI / API limitations. See PRs #171 / #172 / #173 for the trail. Don't try to move the archive back onto `main` — that path was explored and dead-ended.
- The archive is the only growth-instrumentation surface in this repo. Health-check + spreadsheet-fill workflows were considered but descoped (Cowork sandbox couldn't reach external APIs, and GH Actions for the same was deemed overkill for a solo project). Manual review is the current model — fetch the JSON when wanted, hand it to an agent for analysis.
- `growth/README.md` documents the schema + read-it-manually flow.
- TRAFFIC_PAT expires Jun 20 2027 — renew before then.

When adding any new automation that needs to *write* to the repo on a schedule, follow the same
pattern: data on its own unprotected ref, code on `main`. Don't try to make `main` accept bot commits.
