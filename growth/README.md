# Growth instrumentation — weekly traffic archive

GitHub's Traffic API only exposes the **last 14 days** of clones, views, and
referrers. This captures a weekly snapshot so the longitudinal record isn't
lost. That archive is the entire deliverable here.

> **Scope note.** The original brief sketched a second track (a Sunday health
> check that opens an issue if the archive didn't run, and a Monday job that
> fills a Google Sheet — "Cowork"). That was **descoped**. There is no health
> check, no spreadsheet automation, and no second PAT. The weekly archive on the
> `traffic-data` branch is the canonical record; pull it into a sheet by hand
> whenever you want a growth review.

## Where the data lives

On the **`traffic-data`** branch — a data-only [orphan branch](https://github.com/Metabuilder-Labs/tokenjam/tree/traffic-data)
with no shared history with `main`. One file per week:

```
traffic/<ISO-year>-W<week>.json     e.g. traffic/2026-W25.json
```

It lives on its own branch (not `main`) because `main` is protected and blocks
automated direct pushes; `traffic-data` is intentionally unprotected so the
workflow's commit lands without a PR or required checks.

### File schema

```json
{
  "archived_at": "2026-06-20T12:00:00Z",
  "iso_week": "2026-W25",
  "repo": "Metabuilder-Labs/tokenjam",
  "views":  { "count": 0, "uniques": 0, "daily": [ { "timestamp": "...", "count": 0, "uniques": 0 } ] },
  "clones": { "count": 0, "uniques": 0, "daily": [ { "timestamp": "...", "count": 0, "uniques": 0 } ] },
  "referrers": [ { "referrer": "github.com", "count": 0, "uniques": 0 } ],
  "paths":     [ { "path": "/...", "title": "...", "count": 0, "uniques": 0 } ]
}
```

All timestamps are UTC; weeks are ISO `YYYY-W<NN>`.

## How it's produced

`.github/workflows/traffic-archive.yml` (on `main`):

- **Schedule:** Sundays 12:00 UTC, plus manual `workflow_dispatch` (for backfill / testing).
- Calls the four Traffic endpoints (`views`, `clones`, `popular/referrers`,
  `popular/paths`) via `.github/scripts/archive_traffic.py` (stdlib only), then
  publishes the week's JSON to `traffic-data` with the default `GITHUB_TOKEN`.
- Idempotent: re-running a week refreshes that week's snapshot rather than
  duplicating it. Any API failure exits non-zero (red in the Actions tab).

### Auth — the `TRAFFIC_PAT` secret (required)

The Traffic API needs **Administration: Read**, which the default `GITHUB_TOKEN`
*cannot* be granted (`administration` isn't a valid workflow `permissions:` key,
and the default token 403s on these endpoints). So the read step uses a PAT
stored as the **`TRAFFIC_PAT`** repo secret. The push step does **not** use it —
it uses the default `GITHUB_TOKEN` against the unprotected `traffic-data` branch.

`TRAFFIC_PAT` must be a fine-grained PAT with:

- **Resource owner = `Metabuilder-Labs`** (the **org**, not a personal account —
  a personal-owned token can't see this repo and 403s with *"Resource not
  accessible by personal access token"*).
- **Repository access** = `tokenjam`.
- **Repository permissions → Administration: Read** (separate from
  Contents/Metadata; easy to miss).

A classic PAT with the `repo` scope also works. If the secret is missing or
mis-scoped the workflow fails red — that's the signal to re-provision it.

## How to read it manually

```bash
# latest week
git fetch origin traffic-data
git show origin/traffic-data:traffic/2026-W25.json | python3 -m json.tool

# list every archived week
git ls-tree --name-only origin/traffic-data traffic/

# or via the API, no clone needed
gh api repos/Metabuilder-Labs/tokenjam/contents/traffic/2026-W25.json?ref=traffic-data \
  --jq '.content' | base64 -d | python3 -m json.tool
```

Or just browse the [`traffic-data` branch](https://github.com/Metabuilder-Labs/tokenjam/tree/traffic-data) on GitHub.

### For a growth review, the two metrics to fixate on

1. **PyPI weekly "Without Mirrors"** — closest proxy for real install velocity
   (not in these JSONs; pull from pypistats when reviewing).
2. **GitHub unique cloners, 14-day rolling** — leading indicator that moves
   before installs → `clones.uniques` in each weekly JSON.

Everything else (stars, views, referrers, paths) is supporting context.
