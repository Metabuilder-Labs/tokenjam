"""Run-linkage instrumentation for a fan-out harness (the MCP setup_harness tool).

A harness groups its sessions into a tj *run* by stamping the OTel **resource
attribute** ``tokenjam.run_id`` on the launcher AND every worker it spawns. tj
then groups them automatically (see ``api/routes/runs.py``). The robust way is at
*runtime*: the launcher mints one run id per launch and exports it in
``OTEL_RESOURCE_ATTRIBUTES`` so every spawned ``claude`` inherits the same id —
not a value baked into ``.claude/settings.json`` (that would make every run reuse
one id).

This module produces that instrumentation deterministically: the shell helper to
drop in, the equivalent language snippets, and a scan of the repo for likely
spawn points so the caller can show *where* to wire it. It does the read-only
work (generate text, scan files); the actual helper write happens in the MCP
handler. Purely descriptive — it never edits the harness's own code.

Pure module: reads files only; never imports ``tokenjam.api`` / ``tokenjam.cli``.
"""
from __future__ import annotations

import re
from pathlib import Path

from tokenjam.otel.semconv import TjAttributes

#: Where the drop-in helper is written, relative to the repo root.
HELPER_RELPATH = ".tj/run-env.sh"

#: Bounds for the spawn-point scan so a big repo stays cheap.
_SCAN_EXTENSIONS = frozenset({".sh", ".bash", ".zsh", ".py", ".ts", ".js", ".mjs"})
_SKIP_DIRS = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".tj", "generated", ".next", "target",
    "out", "coverage", ".turbo",
})
MAX_SCAN_FILES = 600
MAX_SPAWN_HITS = 40

#: Path substrings that mark a file as a likely launcher/harness — scanned first
#: so a big monorepo's unrelated files don't exhaust the budget before the real
#: spawn scripts (which sort alphabetically late) are ever read.
_HARNESS_HINTS = (
    "govern", "run-loop", "runloop", "spawn", "launch", "orchestr",
    "worker", "harness", "supervis", "fleet", "agent", "loop",
)

#: Lines that look like they spawn a `claude` worker (or already set the OTEL
#: attrs the run id rides on) — the places a launcher carries the run id into.
#: Deliberately narrow to avoid a false-positive flood (a bare trailing `&`
#: matches unrelated code), but covers the real-world forms: a `claude -p` call
#: even via a `$claude_bin` variable, an existing OTEL_RESOURCE_ATTRIBUTES
#: assignment, and python/node subprocess spawns. Still a labeled guess the
#: caller (Claude) confirms against the real code.
_SPAWN_RE = re.compile(
    r"OTEL_RESOURCE_ATTRIBUTES"                               # instrumentation point
    r"|claude[\w-]*\b[^\n]{0,40}?(?:-p\b|--print\b)"          # a claude -p call ($claude_bin too)
    r"|CLAUDE_BIN|claude_bin"                                 # the claude binary variable
    r"|subprocess\.(?:run|Popen|call|check_output)"           # python subprocess
    r"|os\.(?:system|execvp|spawn[lv]p?e?)"                   # python exec
    r"|child_process|execSync|spawnSync|\.spawn\(",           # node spawns
    re.IGNORECASE,
)


def build_run_env_helper() -> str:
    """The drop-in ``.tj/run-env.sh`` — minted-once run id + OTEL export."""
    attr = TjAttributes.RUN_ID
    return (
        "#!/usr/bin/env bash\n"
        "# tj run-linkage. SOURCE this in your harness LAUNCHER, before it spawns\n"
        "# workers. It mints ONE run id per launch and exports it so every spawned\n"
        f"# `claude` (and the launcher) is tagged {attr}=<run> — which tj groups\n"
        "# into a single run on the dashboard. Workers inherit TJ_RUN_ID from the\n"
        "# environment, so they share the launcher's id. Safe to source twice.\n"
        'if [ -z "${TJ_RUN_ID:-}" ]; then\n'
        '  TJ_RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)-$$"\n'
        "fi\n"
        "export TJ_RUN_ID\n"
        'case ",${OTEL_RESOURCE_ATTRIBUTES:-}," in\n'
        f'  *",{attr}="*) ;;  # already tagged — do not double-add\n'
        "  *) export OTEL_RESOURCE_ATTRIBUTES="
        '"${OTEL_RESOURCE_ATTRIBUTES:+${OTEL_RESOURCE_ATTRIBUTES},}'
        f'{attr}=${{TJ_RUN_ID}}" ;;\n'
        "esac\n"
    )


def python_launcher_snippet() -> str:
    """Equivalent instrumentation for a Python launcher (call once at startup)."""
    attr = TjAttributes.RUN_ID
    return (
        "# tj run-linkage — call once in your launcher BEFORE spawning workers.\n"
        "import os, time\n"
        'run_id = os.environ.setdefault("TJ_RUN_ID",\n'
        "    f\"run-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{os.getpid()}\")\n"
        'attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")\n'
        f'if "{attr}=" not in attrs:\n'
        '    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = '
        f'(attrs + "," if attrs else "") + f"{attr}={{run_id}}"\n'
        "# Spawned subprocesses that inherit os.environ are now tagged automatically.\n"
    )


def scan_spawn_points(repo_root: Path) -> list[dict]:
    """Best-effort scan for lines that spawn a worker, as ``{file,line,text}``.

    Walks the repo (skipping vendored/VCS dirs), reads source-like files, and
    flags lines matching :data:`_SPAWN_RE`. Bounded by :data:`MAX_SCAN_FILES` /
    :data:`MAX_SPAWN_HITS`. A labeled guess — the caller confirms against the
    real code. Returns ``[]`` on any failure.
    """
    if not repo_root.is_dir():
        return []

    # Gather candidates, then scan harness-like files first so a big monorepo's
    # unrelated source doesn't exhaust the budget before the launcher scripts.
    candidates: list[Path] = []
    for path in repo_root.rglob("*"):
        if path.is_dir() or path.suffix not in _SCAN_EXTENSIONS:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        candidates.append(path)

    def _priority(p: Path) -> tuple[int, str]:
        # Rank on the path RELATIVE to the repo root so the repo's own location
        # (e.g. a parent dir literally named "harness") can't skew every file.
        rel = str(p.relative_to(repo_root)).lower()
        if any(h in rel for h in _HARNESS_HINTS):
            return (0, rel)
        if p.suffix in (".sh", ".bash", ".zsh"):
            return (1, rel)
        return (2, rel)

    candidates.sort(key=_priority)

    hits: list[dict] = []
    for path in candidates[:MAX_SCAN_FILES]:
        if len(hits) >= MAX_SPAWN_HITS:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _SPAWN_RE.search(line):
                hits.append({
                    "file": str(path.relative_to(repo_root)),
                    "line": lineno,
                    "text": line.strip()[:160],
                })
                if len(hits) >= MAX_SPAWN_HITS:
                    break
    return hits


__all__ = [
    "HELPER_RELPATH",
    "build_run_env_helper",
    "python_launcher_snippet",
    "scan_spawn_points",
    "MAX_SCAN_FILES",
    "MAX_SPAWN_HITS",
]
