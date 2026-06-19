"""CI guard against re-adding tracked dev-secret files to the repo.

`.tj/config.toml` contains a live per-install `ingest_secret` and is regenerated
by `tj onboard` / `tj serve`. It was tracked in error from v0.2.0 through
v0.3.5 (PR #145 untracked it; issue #141 finding #6). This test fails loud if
the file is re-staged so the leak can't recur.

See CLAUDE.md Critical Rule 20 and the "Working with concurrent agents"
section for the operational context.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tj_config_not_tracked():
    result = subprocess.run(
        ["git", "ls-files", ".tj/config.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "", (
        "`.tj/config.toml` is tracked again — it must stay untracked because it "
        "carries a live per-install `ingest_secret`. To fix: "
        "`git rm --cached .tj/config.toml && git commit`. "
        "See PR #145, issue #141 finding #6, and CLAUDE.md Critical Rule 20."
    )
