"""The self-improve loop's Apply stage (SPEC.md §4 step 5, §6, §7).

Routes an approved relearn-cluster proposal to a write at the right
intervention-ladder rung (SPEC §6):

  rung 1 (note)        -> append a marked section to a CLAUDE.md
  rung 2 (skill)       -> write a new ``.claude/skills/<slug>/SKILL.md``
  rung 3-5 (hook/wrapper/config) -> write the enforcement artifact (a hook
      script + a staged settings.json patch) DISABLED; a human must call
      ``enable_enforcement`` explicitly before it ever runs

Reversibility (SPEC §10 — "reversible"): every write here is preceded by a
pre-image snapshot (reusing ``core.summarize.apply``'s atomic-write
primitive), and git-committed when the target lives inside a git repo. A
matching ``revert_applied_fix`` restores the pre-image (or deletes a
freshly-created file) in one call and commits that too. Every apply /
enable / disable / revert is recorded to a durable, DB-independent ledger
(``applied_fixes.json`` — see ``record_applied``) shaped for Phase 3
(verify) to read: did this signature's recurrence actually drop afterwards?

Safety (SPEC §7): nothing here runs unless a caller passes ``go=True`` — the
default is a dry-run that returns the would-be diff/content without writing
(mirrors ``apply_staged``'s contract). The "never mid-session" idle-boundary
check is a separate, best-effort concern (``active_session_warning``) so a
caller (the API route) can warn or block *before* even calling apply.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tokenjam.core.config import TjConfig
from tokenjam.core.summarize.apply import _atomic_write
from tokenjam.core.summarize.session import SummarizeRefused

# --- Rungs (SPEC §6) -----------------------------------------------------------

RUNG_NOTE = 1
RUNG_SKILL = 2
ENFORCEMENT_RUNGS = {3, 4, 5}   # hook / wrapper / config — always human-gated to enable

RUNG_KIND = {1: "note", 2: "skill", 3: "hook", 4: "wrapper", 5: "config"}


class RelearnApplyRefused(Exception):
    """Refusing to apply/enable/revert (house-voice message) — callers map to a 409."""


def slugify(text: str) -> str:
    """A filesystem/skill-name-safe slug. Never empty (falls back to 'fix')."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:60] or "fix"


# --- Storage roots ---------------------------------------------------------

def _storage_base_dir(config: TjConfig) -> Path:
    """The directory every relearn_apply/relearn_store artifact is parented
    under: ``<storage-parent>`` (mirrors ``core.summarize.session.
    summary_root``'s DEC-026 convention) when ``storage.path`` names a real
    file, else a TEMP directory — NEVER the real ``~/.tj`` (must-fix #4: an
    InMemory-backed / ``:memory:``-configured caller must not leak writes
    into a real local install).

    The temp dir is minted once and cached ON the config object itself
    (``config._relearn_memory_tmp_root``), so repeated calls against the SAME
    config instance agree — apply, then a later revert against that same
    live config (e.g. within one ``tj serve`` process), must resolve to the
    same path. A DIFFERENT config object (a different process, a different
    test) gets its own temp dir; nothing here ever shares state across them,
    and nothing here ever falls back to ``Path.home()``.
    """
    sp = config.storage.path
    if sp not in ("", ":memory:"):
        return Path(sp).expanduser().parent
    cached = getattr(config, "_relearn_memory_tmp_root", None)
    if isinstance(cached, Path):
        return cached
    tmp_root = Path(tempfile.mkdtemp(prefix="tokenjam-relearn-mem-"))
    try:
        config._relearn_memory_tmp_root = tmp_root   # type: ignore[attr-defined]
    except Exception:
        pass   # best-effort caching only — a read-only/duck-typed config just re-mints each call
    return tmp_root


def relearn_apply_root(config: TjConfig) -> Path:
    """``<storage-parent>/relearn_apply/`` — see ``_storage_base_dir``."""
    return _storage_base_dir(config) / "relearn_apply"


def _backups_dir(config: TjConfig) -> Path:
    return relearn_apply_root(config) / "backups"


def applied_fixes_path(config: TjConfig) -> Path:
    """The durable ledger Phase 3 (verify) reads — see ``_storage_base_dir``."""
    return _storage_base_dir(config) / "applied_fixes.json"


# --- Default target-path suggestion (for the card's scope/target override) ----

def default_target_path(rung: int, scope: str, repo_cwd: str, slug: str) -> str:
    """Best-effort suggested write location — always user-editable before Approve
    (SPEC's "scope override" requirement: repo-identity is noisy, never trust it
    blindly). Returns ``""`` when there isn't enough information (no known cwd
    for a project-scoped cluster) — the UI must then require an explicit path.
    """
    if scope == "user-global":
        home = Path.home() / ".claude"
        if rung == RUNG_NOTE:
            return str(home / "CLAUDE.md")
        if rung == RUNG_SKILL:
            return str(home / "skills" / slug / "SKILL.md")
        return str(home / "hooks" / f"{slug}.py")

    if not repo_cwd:
        return ""
    base = Path(repo_cwd)
    if rung == RUNG_NOTE:
        return _nearest_claude_md(base)
    if rung == RUNG_SKILL:
        return str(base / ".claude" / "skills" / slug / "SKILL.md")
    return str(base / ".claude" / "hooks" / f"{slug}.py")


def _nearest_claude_md(base: Path) -> str:
    """Walk up from ``base`` for an existing CLAUDE.md (a workspace-root doc
    above a sub-repo counts too); default to ``base/CLAUDE.md`` (to be created)
    when none is found."""
    if base.exists():
        for ancestor in [base, *base.parents[:3]]:
            candidate = ancestor / "CLAUDE.md"
            if candidate.is_file():
                return str(candidate)
    return str(base / "CLAUDE.md")


# --- Git helpers (best-effort — never raise; a missing git/user config just
# means the caller falls back to the gzip backup as the sole revert path) ------

def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def git_repo_root(path: Path) -> Path | None:
    """The git repo root containing ``path``'s parent dir, or None."""
    start = path if path.is_dir() else path.parent
    if not start.exists():
        start = start if start.parent.exists() else Path.cwd()
    result = _run_git(["rev-parse", "--show-toplevel"], start)
    if result is None or result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return Path(out) if out else None


def _git_commit(repo_root: Path, rel_paths: list[str], message: str) -> str | None:
    """Stage + commit specific paths; returns the new commit sha, or None on any
    failure (git absent, nothing to commit, no user.email configured, etc.) —
    the gzip backup remains the guaranteed revert path either way."""
    add = _run_git(["add", "--", *rel_paths], repo_root)
    if add is None or add.returncode != 0:
        return None
    commit = _run_git(
        ["commit", "--no-verify", "-m", message, "--", *rel_paths], repo_root,
    )
    if commit is None or commit.returncode != 0:
        return None
    sha = _run_git(["rev-parse", "HEAD"], repo_root)
    if sha is None or sha.returncode != 0:
        return None
    return (sha.stdout or "").strip() or None


def _commit_message(action: str, title: str, signature: str, rung: int) -> str:
    # Neutral, tool-attributed, no internal ticket IDs — this lands in ANY
    # target repo (including public ones), not just tokenjam's own.
    return (
        f"tokenjam: {action} fix (rung {rung} · {RUNG_KIND.get(rung, '?')})\n\n"
        f"{title}\n\n"
        f"Applied by TokenJam's loop (signature: {signature}).\n"
        f"Revert from the Review inbox, or via the applied_fixes ledger."
    )


# --- Backup (pre-image) store — reuses `_atomic_write`, own namespace so it
# never collides with core.summarize's own backups keyed by resolved path ------

def _backup_paths(config: TjConfig, fix_id: str) -> tuple[Path, Path]:
    d = _backups_dir(config)
    return d / f"{fix_id}.orig", d / f"{fix_id}.meta.json"

def _read_pre_image(target: Path) -> str | None:
    """The file's content before an apply, or None if it doesn't exist yet
    (a fresh create — revert then means delete, not restore-to-empty)."""
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _save_backup(config: TjConfig, fix_id: str, target: str, pre_image: str | None) -> None:
    orig_f, meta_f = _backup_paths(config, fix_id)
    orig_f.parent.mkdir(parents=True, exist_ok=True)
    if pre_image is None:
        meta_f.write_text(json.dumps({"target_path": target, "created": True}), encoding="utf-8")
        return
    orig_f.write_text(pre_image, encoding="utf-8")
    meta_f.write_text(json.dumps({"target_path": target, "created": False}), encoding="utf-8")


def _restore_backup(config: TjConfig, fix_id: str) -> dict[str, Any]:
    """Undo one apply: restore the pre-image, or delete the file this apply
    created. Raises ``RelearnApplyRefused`` if there's no backup record, or
    (must-fix #3) if the target has since become a symlink — checked BEFORE
    branching on ``created``/``is_file`` so a swapped-in symlink can't
    redirect either a restore-write or a delete through it.
    """
    orig_f, meta_f = _backup_paths(config, fix_id)
    if not meta_f.is_file():
        raise RelearnApplyRefused(f"no backup found for applied fix {fix_id} — cannot revert.")
    try:
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RelearnApplyRefused(f"backup metadata for {fix_id} is unreadable — cannot revert.") from exc
    target = Path(meta["target_path"]).expanduser()
    if target.is_symlink():
        raise RelearnApplyRefused(f"{target} is a symlink — refusing to revert through it.")
    if meta.get("created"):
        if target.is_file():
            target.unlink()
        return {"target_path": str(target), "action": "deleted"}
    if not orig_f.is_file():
        raise RelearnApplyRefused(f"backup blob for {fix_id} is missing — cannot revert.")
    original = orig_f.read_text(encoding="utf-8")
    if target.is_file():
        try:
            _atomic_write(target, original)
        except SummarizeRefused as exc:
            raise RelearnApplyRefused(str(exc)) from exc
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(original, encoding="utf-8")
    return {"target_path": str(target), "action": "restored"}


# --- The durable applied_fixes ledger (Phase 3 reads this) --------------------

@dataclass
class AppliedFix:
    id:              str
    signature:       str
    family_key:      str | None
    title:           str
    rung:            int
    kind:            str                    # note | skill | hook | wrapper | config
    scope:           str                    # project | user-global
    target_path:     str
    repo_root:       str | None
    applied_at:      str
    diff:            str                    # unified diff (note) or full content (new file)
    enforcement:     dict[str, Any] | None   # {"enabled": bool, "hook_path", "settings_path", "patch"}
    git_commit:      str | None
    state:           str = "applied"        # applied | reverted
    reverted_at:     str | None = None
    revert_commit:   str | None = None
    # Scaffold for Phase 3 (verify, see core.optimize.relearn_verify) —
    # baseline counts at apply time so a later rescan can measure the
    # recurrence delta for this exact signature. `baseline_sessions` is the
    # cluster's distinct AFFECTED sessions (from the proposal); `baseline_
    # total_sessions` is the exposure denominator (ALL sessions in scope up
    # to apply time, counted the same way the post-apply side is) — verify
    # prefers the latter and falls back to the former only if it's missing.
    verify: dict[str, Any] = field(default_factory=lambda: {
        "baseline_sessions": None, "baseline_occurrences": None,
        "baseline_total_sessions": None,
        "recurrence_since_apply": None, "post_sessions_since_apply": None,
        "baseline_rate": None, "post_rate": None, "realized_tokens_saved": None,
        "escalate_candidate": False, "reason": None,
        "last_checked_at": None, "verdict": None,
    })

    def to_dict(self) -> dict:
        return asdict(self)


def _load_ledger(config: TjConfig) -> list[dict]:
    p = applied_fixes_path(config)
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return raw if isinstance(raw, list) else []


def _write_ledger(config: TjConfig, records: list[dict]) -> None:
    p = applied_fixes_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def list_applied(config: TjConfig) -> list[dict]:
    return _load_ledger(config)


def get_applied(config: TjConfig, fix_id: str) -> dict | None:
    for rec in _load_ledger(config):
        if rec.get("id") == fix_id:
            return rec
    return None


def set_verify(config: TjConfig, fix_id: str, verify: dict[str, Any]) -> dict:
    """Overwrite an applied fix's ``verify`` sub-dict — the write side of
    Phase 3 (``core.optimize.relearn_verify``'s rescan). The caller passes
    the FULL merged verify dict (old fields + new), not a partial patch, so
    this stays a plain field overwrite like every other ``_update_record``
    caller."""
    return _update_record(config, fix_id, verify=verify)


def _save_record(config: TjConfig, record: AppliedFix) -> dict:
    records = _load_ledger(config)
    records.append(record.to_dict())
    _write_ledger(config, records)
    return record.to_dict()


def _update_record(config: TjConfig, fix_id: str, **fields: Any) -> dict:
    records = _load_ledger(config)
    for rec in records:
        if rec.get("id") == fix_id:
            rec.update(fields)
            _write_ledger(config, records)
            return rec
    raise RelearnApplyRefused(f"no applied_fixes record {fix_id}.")


# --- Active-session guard (SPEC §7 — never apply mid-session) ------------------

def active_session_warning(conn: Any | None, target_path: str) -> str | None:
    """Best-effort: is there a LIVE session in the repo ``target_path`` lives
    in? Returns a human warning string, or None when it's safe (or unknown —
    a missing/failed DB check never blocks; it just can't vouch for safety).
    Mirrors the ``agent_id`` repo-label convention used by the relearn
    detector itself (``analyzers.relearn._repo_map_from_db``).
    """
    if conn is None:
        return None
    try:
        repo_hint = git_repo_root(Path(target_path).expanduser())
        label = (repo_hint or Path(target_path).expanduser().parent).name
        if not label:
            return None
        from tokenjam.core.models import SESSION_STALE_THRESHOLD
        from tokenjam.utils.time_parse import utcnow

        rows = conn.execute(
            "SELECT agent_id, status, started_at, ended_at FROM sessions "
            "WHERE status = 'active'"
        ).fetchall()
        now = utcnow()
        for agent_id, status, started_at, ended_at in rows:
            if not agent_id or label not in str(agent_id):
                continue
            last = ended_at or started_at
            if last is None:
                continue
            gap = now - last   # TIMESTAMPTZ columns come back tz-aware (UTC)
            if gap <= SESSION_STALE_THRESHOLD:
                return (
                    f"an active session was seen in a repo matching '{label}' "
                    f"within the last {int(SESSION_STALE_THRESHOLD.total_seconds() // 60)} "
                    f"minutes — applying now risks the exact 'file modified since read' "
                    f"relearn this loop exists to fix. Prefer an idle boundary."
                )
        return None
    except Exception:
        return None   # best-effort — never let a DB hiccup block or falsely warn


# --- Rung 1: CLAUDE.md note ------------------------------------------------

NOTE_SECTION_HEADER = "## TokenJam fixes (auto-added)"
_NOTE_INTRO = (
    "Entries below are written by TokenJam's loop after a human "
    "approves a proposed fix in the Review inbox. Safe to hand-edit the prose; "
    "keep the `<!-- tokenjam:relearn:... -->` markers if you want the inbox's "
    "Revert to keep working."
)


def _note_block(cluster: dict, signature: str) -> str:
    repos = cluster.get("repos") or []
    return (
        f"<!-- tokenjam:relearn:{signature} -->\n"
        f"### {cluster.get('title', signature)}\n\n"
        f"{cluster.get('proposed_fix', '')}\n\n"
        f"_Evidence: {cluster.get('sessions', 0)} session(s) across "
        f"{len(repos)} repo(s). Rung {cluster.get('rung')}. "
        f"Revert from the Review inbox._\n"
        f"<!-- /tokenjam:relearn:{signature} -->"
    )


def render_note_content(existing: str, cluster: dict, signature: str) -> str:
    """Idempotent: replaces a prior block for this exact signature (re-apply),
    else appends one — creating the shared section header on first use."""
    block = _note_block(cluster, signature)
    marker_re = re.compile(
        rf"<!-- tokenjam:relearn:{re.escape(signature)} -->.*?"
        rf"<!-- /tokenjam:relearn:{re.escape(signature)} -->",
        re.DOTALL,
    )
    if marker_re.search(existing):
        return marker_re.sub(block, existing)

    out = existing
    if NOTE_SECTION_HEADER not in out:
        sep = "\n\n" if out and not out.endswith("\n\n") else ""
        out = f"{out}{sep}{NOTE_SECTION_HEADER}\n\n{_NOTE_INTRO}\n\n{block}\n"
    else:
        sep = "\n\n" if not out.endswith("\n\n") else ""
        out = f"{out}{sep}{block}\n"
    return out


# --- Rung 2: skill ----------------------------------------------------------

def render_skill_content(cluster: dict, signature: str, slug: str) -> str:
    repos = cluster.get("repos") or []
    title = cluster.get("title", slug)
    description = (cluster.get("proposed_fix") or title)[:200].replace("\n", " ")
    examples = cluster.get("examples") or []
    ev_lines = "\n".join(
        f"- `{ex.get('session_id', '')[:12]}` ({ex.get('repo', '')}): {ex.get('snippet', '')[:160]}"
        for ex in examples
    ) or "- No example sessions captured."
    return (
        f"---\n"
        f"name: {slug}\n"
        f"description: {description}\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"{cluster.get('proposed_fix', '')}\n\n"
        f"## Why this exists\n\n"
        f"Detected by TokenJam's loop: this pattern recurred in "
        f"{cluster.get('sessions', 0)} distinct session(s) across {len(repos)} "
        f"repo(s) (rung 2 · skill).\n\n"
        f"## Evidence\n\n{ev_lines}\n\n"
        f"<!-- tokenjam:relearn:{signature} -->\n"
    )


# --- Rungs 3-5: enforcement artifact (disabled by default) ---------------------
#
# SPEC §6/§10 mandate: precision over coverage. A false positive on normal
# usage is worse than a no-op, so a REAL matcher ships only for families where
# the bad pattern is unmistakable (a PreToolUse guard) or where we can react
# to an ALREADY-FAILED tool call without ever touching a successful one (a
# PostToolUseFailure context-injection). Every other family renders the safe
# STUB below (never blocks, never injects) rather than guess at a matcher.
#
#   sleep_chain            -> GUARD     (PreToolUse, blocks an unmistakable
#                                          `sleep N && ...` chain)
#   cwd_confusion           -> REACTIVE  (PostToolUseFailure, fires only after
#                                          a Bash/Read call already failed with
#                                          a path/cwd error; injects the real
#                                          cwd + a short listing)
#   stale_read_race         -> REACTIVE  (PostToolUseFailure, fires only after
#                                          an Edit/Write/MultiEdit already
#                                          failed with a "modified since read"
#                                          error; suggests re-Read)
#   edit_string_not_found   -> REACTIVE  (PostToolUseFailure, fires only after
#                                          an Edit/MultiEdit already failed
#                                          with a string-not-found error;
#                                          suggests re-Read)
#
# `PostToolUseFailure` (docs.claude.com/en/hooks): fires only when a tool call
# already errored, carries the failure's error text + `tool_name` + `cwd` on
# stdin, and supports `hookSpecificOutput.additionalContext` on stdout to hand
# Claude recovery context for its NEXT turn. It can never block anything and
# never fires on a successful call -- exactly the "reactive, not a guard"
# shape the mandate asks for. The docs don't pin the exact field holding the
# error, so the generated hook reads it defensively (`tool_error` -> `error`
# -> `tool_response.error` -> stringified `tool_response`; see `_error_text`).

_GUARD_FAMILIES = {"sleep_chain"}

# family_key -> (tool names the failure can come from, the SAME validated
# regex the analyzer clusters on, the recovery note to inject, whether to
# append the actual cwd + a short directory listing).
_REACTIVE_SPECS: dict[str, dict[str, Any]] = {
    "cwd_confusion": {
        "tools": ("Bash", "Read"),
        "pattern": (
            r"no such file or directory|"
            r"file does not exist\.\s*note:\s*your current working directory"
        ),
        "note": (
            "This tool call failed with a path/cwd error. Use an absolute "
            "path, or cd into place, before retrying a relative path."
        ),
        "include_cwd_listing": True,
    },
    "stale_read_race": {
        "tools": ("Edit", "Write", "MultiEdit"),
        "pattern": r"modified since (it was last read|read)",
        "note": (
            "This file was modified since it was last read (likely a "
            "formatter/linter hook rewrite) -- Read it again to get the "
            "current bytes before retrying this edit."
        ),
        "include_cwd_listing": False,
    },
    "edit_string_not_found": {
        "tools": ("Edit", "MultiEdit"),
        "pattern": r"string to replace not found|old_string not found|not found in file",
        "note": (
            "The exact string to replace was not found -- Read the file "
            "again for its current exact content (whitespace/indentation or "
            "a prior edit may have changed it) before retrying."
        ),
        "include_cwd_listing": False,
    },
}

_SLEEP_CHAIN_MATCHER = (
    '    if tool_name == "Bash" and re.search(r"^\\s*sleep\\b.*(&&|;)", command, re.IGNORECASE):\n'
    '        return True, ("blocked sleep-chain (TokenJam, signature '
    '{signature}) — use the Monitor tool instead of a busy-wait sleep.")\n'
)
#: Families an enforcement rung can actually be written for: a hand-authored
#: GUARD matcher (`_GUARD_FAMILIES`) or a hand-authored REACTIVE spec
#: (`_REACTIVE_SPECS`). Anything else has no matcher, so a hook written for it
#: could only ever be inert; `render_hook_content` refuses instead of writing
#: one (see its docstring).
def matchered_families() -> set[str]:
    """The families a rung 3-5 hook can be rendered for. Resolved lazily so a
    future matcher addition is picked up without a second registry to keep in
    sync."""
    return set(_GUARD_FAMILIES) | set(_REACTIVE_SPECS)


def _render_guard_hook(title: str, rung: int | None, signature: str) -> str:
    """A PreToolUse GUARD script -- can block, only for `_GUARD_FAMILIES`."""
    matcher = _SLEEP_CHAIN_MATCHER.format(signature=signature)
    return f'''#!/usr/bin/env python3
"""TokenJam hook -- {title} (rung {rung}, signature {signature}).

Auto-generated by TokenJam's loop after a human approved this
fix in the Review inbox. DISABLED by default -- wiring this into
settings.json requires an explicit "Enable enforcement" confirmation; see
the staged patch sitting next to this file (``*.settings-patch.json``).

GUARD: this runs on PreToolUse and can block an unmistakable bad pattern
before it executes. FAIL-OPEN: the whole body runs under a blanket
try/except. Any unexpected error here allows the tool call through (exit 0)
rather than blocking it or crashing the session.
"""
import json
import re
import sys


def _decide(payload: dict) -> tuple[bool, str]:
    """Return (should_block, reason). Never raises -- see main()'s try/except."""
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {{}}
    command = str(tool_input.get("command", "")) if tool_name == "Bash" else ""
{matcher}    return False, ""


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw else {{}}
        blocked, reason = _decide(payload)
        if blocked:
            print(reason, file=sys.stderr)
            return 2
        return 0
    except Exception:
        return 0   # fail-open: never block on our own bug


if __name__ == "__main__":
    sys.exit(main())

# tokenjam:relearn:{signature}
'''


def _render_reactive_hook(
    spec: dict[str, Any], title: str, rung: int | None, signature: str,
) -> str:
    """A PostToolUseFailure REACTIVE script -- never blocks, only fires after
    a tool call already failed with this family's exact validated error
    signature; injects `additionalContext` for the model's next turn."""
    tools_repr = repr(tuple(spec["tools"]))
    pattern_repr = repr(spec["pattern"])
    note_repr = repr(spec["note"])
    include_listing = bool(spec["include_cwd_listing"])
    return f'''#!/usr/bin/env python3
"""TokenJam hook -- {title} (rung {rung}, signature {signature}).

Auto-generated by TokenJam's loop after a human approved this
fix in the Review inbox. DISABLED by default -- wiring this into
settings.json requires an explicit "Enable enforcement" confirmation; see
the staged patch sitting next to this file (``*.settings-patch.json``).

REACTIVE, not a guard: this runs on PostToolUseFailure -- i.e. only AFTER a
tool call has already failed. It never blocks and never touches a successful
call; it only appends recovery context for the model's next turn via
`hookSpecificOutput.additionalContext`. FAIL-OPEN: the whole body runs under
a blanket try/except, and emitting nothing (exit 0, no stdout) is always a
safe no-op -- Claude Code proceeds exactly as if this hook were absent.
"""
import json
import os
import re
import sys

_TOOLS = {tools_repr}
_PATTERN = re.compile({pattern_repr}, re.IGNORECASE)
_NOTE = {note_repr}
_INCLUDE_CWD_LISTING = {include_listing!r}


def _error_text(payload: dict) -> str:
    """The failure's error string, read defensively across field-name variants.

    The PostToolUseFailure input schema does not pin the exact field carrying
    the error, so we try, in order: ``tool_error`` -> ``error`` ->
    ``tool_response.error`` -> a stringified ``tool_response``. The first
    non-empty string wins; if none yields one, return "" (fail-open -- inject
    nothing). Never raises."""
    for key in ("tool_error", "error"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val
    resp = payload.get("tool_response")
    if isinstance(resp, dict):
        inner = resp.get("error")
        if isinstance(inner, str) and inner.strip():
            return inner
    if isinstance(resp, str) and resp.strip():
        return resp
    if resp is not None and not isinstance(resp, str):
        text = str(resp)
        if text.strip():
            return text
    return ""


def _context(payload: dict) -> str:
    """The `additionalContext` string to inject, or "" (no match -- pass
    through). Never raises -- see main()'s try/except."""
    tool_name = payload.get("tool_name", "")
    if tool_name not in _TOOLS:
        return ""
    error_text = _error_text(payload)
    if not error_text or not _PATTERN.search(error_text):
        return ""
    extra = ""
    if _INCLUDE_CWD_LISTING:
        cwd = str(payload.get("cwd") or "")
        listing = ""
        try:
            if cwd and os.path.isdir(cwd):
                entries = sorted(os.listdir(cwd))[:40]
                listing = ", ".join(entries) if entries else "(empty directory)"
        except OSError:
            listing = ""
        extra = f" Actual working directory: {{cwd or '(unknown)'}}. Top-level entries: {{listing}}."
    return f"[TokenJam, signature {signature}] {{_NOTE}}{{extra}}"


def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw else {{}}
        context = _context(payload)
        if not context:
            return 0   # no match -- pass through, nothing printed
        out = {{
            "hookSpecificOutput": {{
                "hookEventName": "PostToolUseFailure",
                "additionalContext": context,
            }}
        }}
        sys.stdout.write(json.dumps(out))
        return 0
    except Exception:
        return 0   # fail-open: never disrupt the tool call on our own bug


if __name__ == "__main__":
    sys.exit(main())

# tokenjam:relearn:{signature}
'''


def render_hook_content(cluster: dict, signature: str) -> str:
    """The enforcement artifact for a rung 3-5 apply.

    A family with no hand-authored matcher (neither a GUARD matcher nor a
    REACTIVE spec) has nothing to fire on, so the only hook that could be
    written for it is one that never does anything. Writing that would put a
    record in the ledger, a file on disk and an "applied" badge on the card
    for a fix that cannot possibly work; the honest move is to refuse and
    offer rung 1 instead, which writes a real note carrying the same guidance.
    """
    family_key = cluster.get("family_key") or ""
    title = cluster.get("title", signature)
    rung = cluster.get("rung")
    if family_key in _GUARD_FAMILIES:
        return _render_guard_hook(title, rung, signature)
    spec = _REACTIVE_SPECS.get(family_key)
    if spec is not None:
        return _render_reactive_hook(spec, title, rung, signature)
    raise RelearnApplyRefused(
        f"no matcher exists for family '{family_key or 'unknown'}', so a rung "
        f"{rung} hook for it would be written but never fire. Refusing to write "
        f"a fix that looks applied and does nothing. Apply this at rung 1 "
        f"instead: it writes the same guidance as a reversible note."
    )


def _enforcement_wiring(family_key: str) -> tuple[str, str]:
    """(event, tool-matcher) for the `settings.json` wiring `apply_relearn_fix`
    stages -- the ONLY event/tools this rung-3 hook is ever invoked for. Only
    reached for a family that HAS a matcher; `render_hook_content` refuses an
    un-matchered family before any wiring is staged, so the trailing default
    below is unreachable belt-and-braces, not a supported path."""
    if family_key in _GUARD_FAMILIES:
        return "PreToolUse", "Bash"
    spec = _REACTIVE_SPECS.get(family_key)
    if spec is not None:
        return "PostToolUseFailure", "|".join(spec["tools"])
    return "PreToolUse", "Bash"


def render_settings_patch(hook_path: Path, event: str = "PreToolUse", matcher: str = "Bash") -> dict:
    """The settings.json fragment ``enable_enforcement`` would merge in — staged
    to disk alongside the hook, never merged until an explicit Enable."""
    return {
        "hooks": {
            event: [
                {"matcher": matcher, "hooks": [{"type": "command", "command": str(hook_path)}]},
            ]
        }
    }


def _write_target(target: Path, content: str) -> None:
    """Write a rung's rendered content — atomically (preserving mode) when the
    target already exists, else a plain create (``_atomic_write`` needs an
    existing file to stat for its mode).

    Refuses a symlinked target outright (must-fix #3): checked here (not just
    inside ``_atomic_write``) because a BROKEN symlink's ``.exists()`` is
    False, which would otherwise fall through to the plain-create branch and
    write through the link unchecked.
    """
    if target.is_symlink():
        raise RelearnApplyRefused(f"{target} is a symlink — refusing to write through it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            _atomic_write(target, content)
        except SummarizeRefused as exc:
            raise RelearnApplyRefused(str(exc)) from exc
    else:
        target.write_text(content, encoding="utf-8")


# --- Rung 1: note target allowlist (must-fix #2) -------------------------------

def _is_allowed_note_target(target: Path, pre_image: str | None) -> bool:
    """A rung-1 note may only land on: (a) a ``*.md`` file (covers
    ``CLAUDE.md``, the intended target — any Markdown doc is a reasonable
    note home), or (b) a file that ALREADY carries a ``tokenjam:relearn:``
    marker (a prior legitimate apply — the same rung 2/3 re-apply allowance).
    Anything else (a ``.py``, ``.zshrc``, etc.) is refused outright, closing
    the "rung=1 + arbitrary target_path corrupts any file" gap."""
    if target.suffix.lower() == ".md":
        return True
    return pre_image is not None and "tokenjam:relearn:" in pre_image


# --- Write plan (shared by dry-run preview and the real apply) ----------------

def _build_write_plan(cluster: dict, target_path: str) -> dict[str, Any]:
    """Render what WOULD be written for ``cluster`` at ``target_path`` — pure,
    no disk writes. Raises ``RelearnApplyRefused`` if a non-TokenJam file
    already sits at a create-only target (skill / hook), the target is a
    symlink, or (rung 1) the target isn't an allowlisted note file."""
    import difflib

    signature = cluster["signature"]
    rung = int(cluster["rung"])
    if rung not in RUNG_KIND:
        raise RelearnApplyRefused(f"unknown rung {rung}.")
    if not target_path:
        raise RelearnApplyRefused("no target path given — pick one before approving.")

    slug = slugify(cluster.get("title") or cluster.get("family_key") or signature)
    target = Path(target_path).expanduser()
    if target.is_symlink():
        raise RelearnApplyRefused(
            f"{target} is a symlink — refusing to write through it. Point target_path "
            f"at the real file."
        )
    pre_image = _read_pre_image(target)

    if rung == RUNG_NOTE:
        if not _is_allowed_note_target(target, pre_image):
            raise RelearnApplyRefused(
                f"{target} is not an allowlisted note target (must be a CLAUDE.md/*.md "
                f"file, or already carry a tokenjam:relearn marker) — refusing to write "
                f"a rung-1 note there."
            )
        new_content = render_note_content(pre_image or "", cluster, signature)
    else:
        if pre_image is not None and "tokenjam:relearn:" not in pre_image:
            raise RelearnApplyRefused(
                f"{target} already exists and wasn't written by TokenJam — refusing to "
                f"overwrite it. Choose a different target path."
            )
        new_content = (
            render_skill_content(cluster, signature, slug) if rung == RUNG_SKILL
            else render_hook_content(cluster, signature)
        )

    diff = "".join(difflib.unified_diff(
        (pre_image or "").splitlines(keepends=True), new_content.splitlines(keepends=True),
        fromfile="before", tofile="after", n=2,
    ))
    return {
        "signature": signature, "rung": rung, "slug": slug, "target_path": str(target),
        "pre_image": pre_image, "new_content": new_content, "diff": diff,
        "kind": RUNG_KIND[rung],
    }


def preview_relearn_fix(cluster: dict, *, target_path: str) -> dict:
    """Dry-run: exactly what ``apply_relearn_fix(..., go=True)`` would write,
    without touching disk. Raises ``RelearnApplyRefused`` on the same guards
    the real apply enforces (unknown rung, no target, hostile overwrite)."""
    return {"dry_run": True, **_build_write_plan(cluster, target_path)}


def apply_relearn_fix(
    config: TjConfig,
    cluster: dict,
    *,
    target_path: str,
    scope: str,
    go: bool = False,
    conn: Any | None = None,
    force: bool = False,
) -> dict:
    """Write an approved proposal at its rung. Default dry-run (mirrors
    ``apply_staged``'s contract) — pass ``go=True`` to actually write.

    Raises ``RelearnApplyRefused`` (callers map to 409) when: the rung/target
    is invalid, a non-TokenJam file already sits at a create-only target, or
    (when ``force`` is not set) a live session was just seen in the target's
    repo (SPEC §7 — never apply mid-session).
    """
    plan = _build_write_plan(cluster, target_path)
    if not go:
        return {"dry_run": True, **plan}

    warning = active_session_warning(conn, plan["target_path"])
    if warning and not force:
        raise RelearnApplyRefused(warning)

    target = Path(plan["target_path"]).expanduser()
    rung = plan["rung"]
    fix_id = uuid.uuid4().hex[:16]

    _save_backup(config, fix_id, str(target), plan["pre_image"])
    _write_target(target, plan["new_content"])

    enforcement: dict[str, Any] | None = None
    if rung in ENFORCEMENT_RUNGS:
        target.chmod(0o755)
        settings_path = target.parent.parent / "settings.json"
        patch_path = target.parent / f"{plan['slug']}.settings-patch.json"
        event, matcher = _enforcement_wiring(cluster.get("family_key") or "")
        patch = render_settings_patch(target, event=event, matcher=matcher)
        patch_path.write_text(json.dumps(patch, indent=2) + "\n", encoding="utf-8")
        enforcement = {
            "enabled": False,
            "hook_path": str(target),
            "settings_path": str(settings_path),
            "patch_path": str(patch_path),
            "patch": patch,
        }

    repo_root = git_repo_root(target)
    commit_sha = None
    if repo_root:
        rel_paths = [str(target.relative_to(repo_root))]
        if enforcement:
            rel_paths.append(str(Path(enforcement["patch_path"]).relative_to(repo_root)))
        commit_sha = _git_commit(
            repo_root, rel_paths,
            _commit_message("apply", cluster.get("title", plan["signature"]), plan["signature"], rung),
        )

    from tokenjam.utils.time_parse import utcnow

    applied_at_dt = utcnow()
    record = AppliedFix(
        id=fix_id, signature=plan["signature"], family_key=cluster.get("family_key"),
        title=cluster.get("title", plan["signature"]), rung=rung, kind=plan["kind"],
        scope=scope, target_path=str(target),
        repo_root=str(repo_root) if repo_root else None,
        applied_at=applied_at_dt.isoformat(), diff=plan["diff"], enforcement=enforcement,
        git_commit=commit_sha,
    )
    record.verify["baseline_sessions"] = cluster.get("sessions")
    record.verify["baseline_occurrences"] = cluster.get("occurrences")
    # Best-effort exposure denominator (Phase 3 verify) — total sessions in
    # this fix's scope up to right now, counted the SAME way a later verify
    # pass counts the post-apply side (core.optimize.relearn_verify.
    # count_sessions_in_scope), so the two rates are comparable. Never lets a
    # scan failure sink the apply itself.
    try:
        from tokenjam.core.optimize import relearn_verify

        repo_filter = repo_root.name if (scope == "project" and repo_root) else None
        record.verify["baseline_total_sessions"] = relearn_verify.count_sessions_in_scope(
            None, conn, repo_filter, before=applied_at_dt,
        )
    except Exception:
        record.verify["baseline_total_sessions"] = None
    return {"dry_run": False, "record": _save_record(config, record)}


# --- Enforcement enable/disable (rungs 3-5 only) -------------------------------

def _read_settings(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _merge_hook_patch(settings: dict, patch: dict) -> dict:
    """Immutable merge: returns a NEW settings dict with ``patch``'s hook
    entries appended, skipping any event/command pair already present."""
    out = json.loads(json.dumps(settings))
    hooks_out = out.setdefault("hooks", {})
    for event, entries in (patch.get("hooks") or {}).items():
        existing = hooks_out.setdefault(event, [])
        for entry in entries:
            entry_commands = {h.get("command") for h in entry.get("hooks", [])}
            already = any(
                {h.get("command") for h in e.get("hooks", [])} & entry_commands
                for e in existing if isinstance(e, dict)
            )
            if not already:
                existing.append(entry)
    return out


def _remove_hook_command(settings: dict, hook_command: str) -> dict:
    """Immutable removal: returns a NEW settings dict with any hook entry
    whose command == ``hook_command`` stripped out (empty entries dropped)."""
    out = json.loads(json.dumps(settings))
    hooks = out.get("hooks") or {}
    for event, entries in list(hooks.items()):
        kept = []
        for entry in entries:
            entry_hooks = [h for h in entry.get("hooks", []) if h.get("command") != hook_command]
            if entry_hooks:
                kept.append({**entry, "hooks": entry_hooks})
        hooks[event] = kept
    out["hooks"] = hooks
    return out


def enable_enforcement(config: TjConfig, fix_id: str, *, confirm: bool) -> dict:
    """Wire a generated hook into settings.json — the ONLY step that makes an
    enforcement rung (3-5) live. Requires an explicit ``confirm=True`` (the
    UI's "this intercepts your tools" warning); never auto-fires."""
    rec = get_applied(config, fix_id)
    if rec is None:
        raise RelearnApplyRefused(f"no applied fix {fix_id}.")
    if rec["rung"] not in ENFORCEMENT_RUNGS:
        raise RelearnApplyRefused("only rung 3-5 (hook/wrapper/config) fixes need enabling.")
    if rec["state"] != "applied":
        raise RelearnApplyRefused("this fix was reverted — re-apply it before enabling.")
    enforcement = rec.get("enforcement") or {}
    if enforcement.get("enabled"):
        return rec
    if not confirm:
        raise RelearnApplyRefused(
            "enabling enforcement wires this hook into settings.json, where it "
            "intercepts your tool calls on every matching call — pass an explicit "
            "confirmation to proceed."
        )

    settings_path = Path(enforcement["settings_path"]).expanduser()
    patch = enforcement["patch"]
    merged = _merge_hook_patch(_read_settings(settings_path), patch)
    _write_target(settings_path, json.dumps(merged, indent=2) + "\n")

    repo_root = git_repo_root(settings_path)
    commit_sha = None
    if repo_root:
        rel = str(settings_path.relative_to(repo_root))
        commit_sha = _git_commit(
            repo_root, [rel],
            _commit_message("enable enforcement for", rec["title"], rec["signature"], rec["rung"]),
        )
    new_enforcement = {**enforcement, "enabled": True, "enable_commit": commit_sha}
    return _update_record(config, fix_id, enforcement=new_enforcement)


def disable_enforcement(config: TjConfig, fix_id: str) -> dict:
    """Unwire the hook from settings.json (the file itself is left in place —
    only the settings.json entry is removed) — a no-op if already disabled."""
    rec = get_applied(config, fix_id)
    if rec is None:
        raise RelearnApplyRefused(f"no applied fix {fix_id}.")
    enforcement = rec.get("enforcement") or {}
    if not enforcement.get("enabled"):
        return rec

    settings_path = Path(enforcement["settings_path"]).expanduser()
    if settings_path.is_file():
        updated = _remove_hook_command(_read_settings(settings_path), enforcement["hook_path"])
        _write_target(settings_path, json.dumps(updated, indent=2) + "\n")
        repo_root = git_repo_root(settings_path)
        if repo_root:
            rel = str(settings_path.relative_to(repo_root))
            _git_commit(
                repo_root, [rel],
                _commit_message("disable enforcement for", rec["title"], rec["signature"], rec["rung"]),
            )
    new_enforcement = {**enforcement, "enabled": False}
    return _update_record(config, fix_id, enforcement=new_enforcement)


# --- Revert (one-step, for any rung) -------------------------------------------

def revert_applied_fix(config: TjConfig, fix_id: str) -> dict:
    """Undo an apply: disables enforcement first (if it was live), restores
    the pre-image (or deletes a freshly-created file), removes the enforcement
    sidecar patch file, and git-commits the revert when the target is tracked.
    Idempotent — reverting an already-reverted fix is a no-op."""
    rec = get_applied(config, fix_id)
    if rec is None:
        raise RelearnApplyRefused(f"no applied fix {fix_id}.")
    if rec["state"] == "reverted":
        return rec

    if (rec.get("enforcement") or {}).get("enabled"):
        disable_enforcement(config, fix_id)
        rec = get_applied(config, fix_id) or rec

    restore = _restore_backup(config, fix_id)
    target = Path(restore["target_path"])

    enforcement = rec.get("enforcement")
    if enforcement:
        patch_path = Path(enforcement["patch_path"])
        if patch_path.is_file():
            patch_path.unlink()

    repo_root = git_repo_root(target)
    commit_sha = None
    if repo_root:
        rel = str(target.relative_to(repo_root))
        add = _run_git(["add", "--", rel], repo_root)
        if add is not None and add.returncode == 0:
            commit = _run_git(
                ["commit", "--no-verify", "-m",
                 _commit_message("revert", rec["title"], rec["signature"], rec["rung"]),
                 "--", rel],
                repo_root,
            )
            if commit is not None and commit.returncode == 0:
                sha = _run_git(["rev-parse", "HEAD"], repo_root)
                commit_sha = (sha.stdout or "").strip() if sha and sha.returncode == 0 else None

    from tokenjam.utils.time_parse import utcnow

    return _update_record(
        config, fix_id, state="reverted",
        reverted_at=utcnow().isoformat(), revert_commit=commit_sha,
    )
