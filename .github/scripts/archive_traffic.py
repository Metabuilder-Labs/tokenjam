#!/usr/bin/env python3
"""
Archive this repo's GitHub Traffic API data into traffic/<ISO-year>-W<week>.json.

GitHub's Traffic API only retains the last 14 days, so this runs weekly (see
.github/workflows/traffic-archive.yml) and commits a point-in-time snapshot —
the longitudinal record the 14-day window can't otherwise provide.

The output schema is fixed by the growth-instrumentation brief and consumed by
the Cowork spreadsheet-fill job; keep it stable across releases.

Auth: reads GITHUB_TOKEN + GITHUB_REPOSITORY from the environment (both provided
automatically by GitHub Actions). The default workflow token works because the
workflow grants `administration: read`, which the Traffic API requires.

Failure mode: any API error raises and the process exits non-zero, so the run
shows red in the Actions tab (the Cowork health check then alerts on the missing
file). No-data weeks still write the file — its existence is the success signal.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_ROOT = "https://api.github.com"


def _get(path: str, token: str):
    """GET an api.github.com path and return parsed JSON. Raises on any error."""
    req = urllib.request.Request(
        f"{API_ROOT}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "tokenjam-traffic-archive",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def _daily(payload: dict, key: str) -> list[dict]:
    """Normalize the views/clones inner array (keyed 'views'/'clones') to 'daily'."""
    return [
        {"timestamp": d["timestamp"], "count": d["count"], "uniques": d["uniques"]}
        for d in payload.get(key, [])
    ]


def build_record(views: dict, clones: dict, referrers: list, paths: list,
                 repo: str, now: datetime) -> dict:
    """Assemble the combined archive record matching the fixed brief schema."""
    iso_year, iso_week, _ = now.isocalendar()
    iso_week_str = f"{iso_year}-W{iso_week:02d}"
    return {
        "archived_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "iso_week": iso_week_str,
        "repo": repo,
        "views": {
            "count": views.get("count", 0),
            "uniques": views.get("uniques", 0),
            "daily": _daily(views, "views"),
        },
        "clones": {
            "count": clones.get("count", 0),
            "uniques": clones.get("uniques", 0),
            "daily": _daily(clones, "clones"),
        },
        "referrers": [
            {"referrer": r["referrer"], "count": r["count"], "uniques": r["uniques"]}
            for r in (referrers or [])
        ],
        "paths": [
            {"path": p["path"], "title": p.get("title", ""),
             "count": p["count"], "uniques": p["uniques"]}
            for p in (paths or [])
        ],
    }


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")  # "owner/name"
    if not token or not repo:
        print("error: GITHUB_TOKEN and GITHUB_REPOSITORY must be set", file=sys.stderr)
        return 1

    views = _get(f"/repos/{repo}/traffic/views", token)
    clones = _get(f"/repos/{repo}/traffic/clones", token)
    referrers = _get(f"/repos/{repo}/traffic/popular/referrers", token)
    paths = _get(f"/repos/{repo}/traffic/popular/paths", token)

    now = datetime.now(timezone.utc)
    record = build_record(views, clones, referrers, paths, repo, now)
    iso_week_str = record["iso_week"]

    out_dir = Path("traffic")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{iso_week_str}.json"
    # Overwrite is intentional: a week's archive is a point-in-time snapshot, so
    # a manual re-run or late-firing schedule just refreshes it (idempotent).
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(
        f"wrote {out_path} (views={record['views']['count']}, "
        f"clones={record['clones']['count']}, "
        f"referrers={len(record['referrers'])}, paths={len(record['paths'])})"
    )

    # Expose the ISO week to the workflow for the commit message.
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"iso_week={iso_week_str}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
