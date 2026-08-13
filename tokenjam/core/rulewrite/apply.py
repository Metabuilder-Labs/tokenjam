"""``stage`` / ``check`` / ``apply`` / ``undo`` for a placed rule write.

The safety model is the one ``core/summarize/apply`` already proves out, and it
is deliberately not re-invented: owner check, symlink refusal, content-hash
guard, gzip backup before every write, atomic write preserving mode, undo that
refuses on drift. **Dry-run is the default**; ``go=True`` writes. No flag
bypasses a guard.

What is new here is that one rule has N destinations, so every verb is
per-destination and partial outcomes are first-class: three project files
written and a fourth skipped because it changed under us is a normal,
reportable result, not an error. The alternative — all-or-nothing across files
the user may not even own — would mean one drifted file blocks a fix for every
other project.

**Nothing in this module knows what a CLAUDE.md is.** How a rule reaches the
agent is a DELIVERY MECHANISM, resolved through ``core/rulewrite/delivery``,
because appending markdown is one way to put guidance in front of an agent and
not the only one — a hook can deliver the same guidance at the moment of the
decision instead of at the top of a long context. Staging, diffing, applying,
undoing and pricing all go through that seam, so a second mechanism is a new
registration plus its renderer, never a rewrite of this file. Today's one
mechanism delegates to ``relearn_apply``'s renderers, which own the marker
format that makes a write idempotent and revertible.
"""
from __future__ import annotations

import difflib
import os
from pathlib import Path

from tokenjam.core.atomic_write import AtomicWriteRefused, atomic_write
from tokenjam.core.config import TjConfig
from tokenjam.core.rulewrite import store
from tokenjam.core.rulewrite.delivery import DEFAULT_DELIVERY, resolve_delivery
from tokenjam.core.rulewrite.types import (
    RuleWrite,
    RuleWriteRefused,
    StagedRuleWrite,
)


def _owned_by_current_user(path: Path) -> bool:
    if not hasattr(os, "getuid"):     # non-POSIX — no ownership model to honour
        return True
    return path.stat().st_uid == os.getuid()


def _diff(path: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))


def stage_rule(config: TjConfig, rule: RuleWrite) -> list[StagedRuleWrite]:
    """Render + stage one diff per destination. Writes nothing to the targets.

    Refuses a rule that is not on offer — already applied, already present in
    the user's own files, or dismissed. Those are the only reasons `offered`
    is ever False; every `apply_capable` rule is otherwise offered, with no
    budget/ceiling gate on top. The rule's text stays copyable either way — a
    withdrawn offer is not a deletion.
    """
    # Checked BEFORE the generic not-offered refusal so the message names the
    # real reason.
    if rule.already_applied:
        raise RuleWriteRefused(
            f"{rule.signature} is already applied — you told tokenjam you made "
            "this change. Revert it first if you want to write it again. What "
            "it cost before you applied it is still reported in full.",
        )
    # Same shape as the applied check above, and for the same reason: the message
    # has to name the real state. Writing this rule would append guidance the file
    # already carries, which is the duplication the presence check exists to stop.
    if rule.already_present:
        raise RuleWriteRefused(
            f"{rule.signature} is already in your instruction files"
            + (f" ({rule.presence_path})" if rule.presence_path else "")
            + " — writing it would duplicate guidance you already have. What the "
            "recurrence cost before is still reported in full.",
        )
    if not rule.offered:
        raise RuleWriteRefused(
            f"{rule.signature} is not on offer: "
            f"{rule.blocked_reason or 'it is not currently offered'}. "
            "Its text is still shown so you can apply it by hand.",
        )
    if not rule.destinations:
        raise RuleWriteRefused(
            f"{rule.signature} has no resolved destination — nothing to stage. "
            "This happens when no session it was derived from recorded a "
            "working directory that still exists.",
        )
    # Resolved ONCE for the whole rule and recorded on every staged entry, so
    # apply re-renders through the same mechanism that produced the diff a
    # reviewer approved rather than through whatever the default is by then.
    kind = resolve_delivery(rule.delivery or DEFAULT_DELIVERY)
    staged: list[StagedRuleWrite] = []
    for destination in rule.destinations:
        target = Path(destination.path).expanduser()
        if target.is_symlink():
            raise RuleWriteRefused(
                f"{destination.path} is a symlink — refusing to write through "
                "it (that would replace the link, not the file).",
            )
        existing = target.read_text(encoding="utf-8") if target.is_file() else ""
        rendered = kind.render(rule, existing)
        entry = StagedRuleWrite(
            signature=rule.signature,
            path=str(target),
            scope=destination.scope,
            title=rule.title,
            analyzer=rule.analyzer,
            delivery=kind.name,
            source_sha256=store.sha256(existing),
            rendered=rendered,
            diff=_diff(destination.path, existing, rendered),
            # Priced by the MECHANISM, which is the only thing that knows.
            # A mechanism that puts tokens in front of the model pays a standing
            # cost; see `delivery.py` on why a context-INJECTING hook cannot
            # inherit an executing hook's zero just for also being a hook.
            standing_tokens_per_session=(
                kind.standing_tokens(rule, rendered, existing)
                if kind.carries_prompt_text else 0
            ),
            sessions=destination.sessions,
            creates_file=not target.is_file(),
        )
        store.stage(config, entry)
        staged.append(entry)
    return staged


def check_staged(config: TjConfig) -> list[dict]:
    """Re-check every staged entry against the file as it stands NOW.

    Read-only. Each row says whether the entry is still applyable and why not
    when it is not — the same conditions :func:`apply_staged` enforces, surfaced
    before a user commits to running it.
    """
    rows: list[dict] = []
    for entry in store.list_staged(config):
        applyable, reason = _applyable(entry)
        rows.append({
            "signature": entry.signature, "path": entry.path,
            "analyzer": entry.analyzer, "title": entry.title,
            "scope": entry.scope, "sessions": entry.sessions,
            "delivery": entry.delivery or DEFAULT_DELIVERY,
            "creates_file": entry.creates_file,
            "applyable": applyable, "reason": reason,
        })
    return rows


def _applyable(entry: StagedRuleWrite) -> tuple[bool, str]:
    target = Path(entry.path).expanduser()
    if target.is_symlink():
        return False, "symlink — refusing to write through it"
    if not target.exists():
        # Staged as a create, and still absent: fine. Staged against a file
        # that has since been deleted: not fine, because the diff a human
        # approved described content that no longer exists.
        if entry.creates_file:
            return True, ""
        return False, "file no longer exists — re-stage it"
    if not target.is_file():
        return False, "not a regular file"
    if entry.creates_file:
        return False, "a file now exists at this path — re-stage it"
    if not _owned_by_current_user(target):
        return False, "owned by another user — refusing to rewrite"
    try:
        current = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False, "cannot read the file"
    if store.sha256(current) != entry.source_sha256:
        return False, "changed since staging — re-stage it"
    return True, ""


def _write(target: Path, text: str) -> None:
    """Atomic write for an existing file; plain create otherwise.

    ``atomic_write`` stats the target for its mode, so it needs the file to
    exist; a create goes through ``write_text``. Both paths translate the
    primitive's refusal into the house exception, so no caller of this package
    ever sees ``AtomicWriteRefused``.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            atomic_write(target, text)
        except AtomicWriteRefused as exc:
            raise RuleWriteRefused(str(exc)) from exc
    else:
        target.write_text(text, encoding="utf-8")


def apply_staged(
    config: TjConfig, signature: str | None = None, *, go: bool = False,
) -> dict:
    """Apply staged rule writes — all of them, or just one rule's.

    Default **dry-run**: it reports what would change and touches nothing.
    ``go=True`` writes, taking a gzip backup of every target first.

    Returns ``{"applied": [...], "skipped": [{"path", "signature", "reason"}],
    "dry_run": bool}``. A skipped destination is reported, never silently
    dropped: "3 of 4 projects written, the fourth changed under us" is the
    honest result, and a caller that only counted successes would report the
    rule as fully applied.
    """
    entries = [
        e for e in store.list_staged(config)
        if signature is None or e.signature == signature
    ]
    applied: list[dict] = []
    skipped: list[dict] = []
    for entry in entries:
        ok, reason = _applyable(entry)
        if not ok:
            skipped.append({
                "path": entry.path, "signature": entry.signature, "reason": reason,
            })
            continue
        if go:
            target = Path(entry.path).expanduser()
            original = target.read_text(encoding="utf-8") if target.is_file() else None
            store.save_backup(
                config, entry.signature, entry.path,
                original=original, output=entry.rendered,
            )
            _write(target, entry.rendered)
            store.clear(config, entry.signature, entry.path)
        applied.append({
            "path": entry.path, "signature": entry.signature,
            "analyzer": entry.analyzer, "scope": entry.scope,
            "delivery": entry.delivery or DEFAULT_DELIVERY,
            "sessions": entry.sessions, "diff": entry.diff,
            "standing_tokens_per_session": entry.standing_tokens_per_session,
            "created_file": entry.creates_file,
        })
    return {"applied": applied, "skipped": skipped, "dry_run": not go}


def undo(
    config: TjConfig, signature: str, path: str | None = None, *, go: bool = False,
) -> dict:
    """Restore what one rule write replaced — one destination, or all of them.

    Default dry-run. Refuses (per destination) when the file has changed since
    apply: undoing over newer edits would lose work the user did after
    approving the rule.
    """
    targets = [
        row for row in store.list_backups(config)
        if row["signature"] == signature
        and (path is None or row["source_path"] == path)
    ]
    if not targets:
        raise RuleWriteRefused(
            f"no applied rule write for {signature}"
            + (f" at {path}" if path else "") + " — nothing to undo.",
        )
    restored: list[dict] = []
    skipped: list[dict] = []
    for row in targets:
        source = row["source_path"]
        target = Path(source).expanduser()
        current = target.read_text(encoding="utf-8") if target.is_file() else None
        try:
            original, created = store.load_backup(config, signature, source, current)
        except RuleWriteRefused as exc:
            skipped.append({"path": source, "reason": str(exc)})
            continue
        if go:
            if created:
                # The write created this file; undoing it removes the file
                # rather than leaving an empty one behind, which would read as
                # a rule the agent still loads.
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
            elif original is not None:
                _write(target, original)
            store.clear_backup(config, signature, source)
        restored.append({"path": source, "removed_file": created})
    return {
        "signature": signature, "restored": restored, "skipped": skipped,
        "dry_run": not go,
    }


# Deliberately no delivery-kind constants here. They live on ``kinds``
# (declared) and ``delivery`` (given behaviour); re-exporting them from the
# apply path would suggest this module decides what an artifact is, which is
# exactly the coupling the delivery seam removed.
__all__ = [
    "apply_staged",
    "check_staged",
    "stage_rule",
    "undo",
]
