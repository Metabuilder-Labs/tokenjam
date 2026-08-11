"""CI guard: `.claude/rules/` is the ONLY tracked path under `.claude/`.

`.claude/` holds internal, unpublished material — strategy, competitive
teardowns, pricing work, `settings.local.json` — and was gitignored wholesale
for the life of the repo, so "under `.claude/`" reliably meant "private". The
CLAUDE.md path-scoped split broke that equivalence on purpose: the per-subsystem
guides have to be TRACKED to reach contributors at all, and the only directory
Claude Code loads them from is `.claude/rules/`.

That leaves one subdirectory public inside a tree everyone (human and agent) has
learned to treat as private, and the `*.md`-only restriction in `.gitignore`
does not help — the sensitive documents are markdown too. So the invariant is
pinned here rather than left to memory: exactly two things may be tracked under
`.claude/`, `rules/*.md` and nothing else.

This is the inverse-assertion form Critical Rule 23 argues for: it defends the
correct state (only rules are public) instead of merely forbidding one known-bad
filename, so a competitive note dropped into `rules/` by muscle memory fails CI
rather than shipping to a public repo.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tracked_under_dotclaude() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", ".claude/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_only_rules_markdown_is_tracked_under_dotclaude():
    offenders = [
        path
        for path in _tracked_under_dotclaude()
        if not (path.startswith(".claude/rules/") and path.endswith(".md"))
    ]
    assert offenders == [], (
        "Files under `.claude/` are tracked that are not `.claude/rules/*.md`: "
        f"{offenders}. Everything under `.claude/` except `rules/` is internal "
        "and must never be published — it is separately mirrored to a private "
        "archive. To fix: `git rm --cached <path>` and check the `.gitignore` "
        "`.claude/*` / `!.claude/rules/*.md` hunk is intact."
    )


def test_tracked_rules_are_path_scoped_so_they_are_not_always_resident():
    """A rule without a `paths:` glob loads at launch with CLAUDE.md priority.

    That silently undoes the whole point of the split — the file is written, the
    guidance works, and only the cost is wrong, which is exactly the failure
    `core/summarize/load_semantics` documents as invisible. Classify through
    that module rather than re-implementing the frontmatter check here.
    """
    from tokenjam.core.summarize.load_semantics import PATH_SCOPED, classify

    unscoped = []
    for path in _tracked_under_dotclaude():
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        if classify(path, text) != PATH_SCOPED:
            unscoped.append(path)

    assert unscoped == [], (
        f"Tracked rule files carry no `paths:` glob in their frontmatter: {unscoped}. "
        "Without it the whole body is injected into EVERY session at launch, at "
        "CLAUDE.md priority, which is the always-resident cost the split exists to "
        "remove. Add a `paths:` list, or move the file out of `.claude/rules/`."
    )
