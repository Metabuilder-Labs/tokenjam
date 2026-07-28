"""Staging + gzip backup store for rule writes (``~/.tj/rulewrite/``).

Deliberately the same shape as ``core/summarize/session`` + ``core/summarize/
backup``, because it is the same lifecycle over a different artifact: stage a
rendered result keyed on the target path, hash-guard it, gzip the original on
apply, restore on undo. What differs is the key — a rule write is
``(signature, path)``, not ``path`` alone, since one rule lands in several
files and several rules can land in one file.

The store persists rendered OUTPUT, never a recipe. A staged entry that had to
be re-rendered at apply time would let the file drift between the diff a human
approved and the bytes that hit disk, and the whole review step would then be
theatre.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from tokenjam.core.config import TjConfig
from tokenjam.core.rulewrite.types import RuleWriteRefused, StagedRuleWrite
from tokenjam.utils.time_parse import utcnow


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rulewrite_root(config: TjConfig) -> Path:
    """The durable anchor next to the storage DB (``~/.tj/rulewrite/``).

    Same derivation as ``summarize.session.summary_root``: an in-memory or
    unset storage path means there is no DB directory to sit beside, so it
    falls back to the default home rather than writing next to ``:memory:``.
    """
    storage_path = config.storage.path
    base = (
        Path.home() / ".tj"
        if storage_path in ("", ":memory:")
        else Path(storage_path).expanduser().parent
    )
    return base / "rulewrite"


def staged_dir(config: TjConfig) -> Path:
    return rulewrite_root(config) / "staged"


def backups_dir(config: TjConfig) -> Path:
    return rulewrite_root(config) / "backups"


def stage_key(signature: str, path: str) -> str:
    """Collision-free key for one (rule, destination) pair.

    Keyed on the RESOLVED absolute path so two spellings of the same file
    (a relative path, a path through a symlinked parent) cannot stage twice and
    then apply twice into one file.
    """
    resolved = str(Path(path).expanduser().resolve())
    return sha256(f"{signature}\x00{resolved}")


def _staged_file(config: TjConfig, signature: str, path: str) -> Path:
    return staged_dir(config) / f"{stage_key(signature, path)}.json"


def stage(config: TjConfig, entry: StagedRuleWrite) -> Path:
    directory = staged_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    target = _staged_file(config, entry.signature, entry.path)
    target.write_text(
        json.dumps(entry.to_dict(), ensure_ascii=False), encoding="utf-8",
    )
    return target


def list_staged(config: TjConfig) -> list[StagedRuleWrite]:
    """Every staged rule write. A corrupt or half-written entry is skipped, not
    raised on: a partially flushed file must never make ``tj rules list``
    unusable."""
    directory = staged_dir(config)
    if not directory.exists():
        return []
    out: list[StagedRuleWrite] = []
    for f in sorted(directory.glob("*.json")):
        try:
            out.append(StagedRuleWrite.from_dict(
                json.loads(f.read_text(encoding="utf-8")),
            ))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
            continue
    return out


def read_staged(
    config: TjConfig, signature: str, path: str,
) -> StagedRuleWrite | None:
    f = _staged_file(config, signature, path)
    if not f.exists():
        return None
    try:
        return StagedRuleWrite.from_dict(json.loads(f.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None


def clear(
    config: TjConfig, signature: str | None = None, path: str | None = None,
) -> int:
    """Remove one staged entry, every entry for one rule, or all of them."""
    directory = staged_dir(config)
    if not directory.exists():
        return 0
    if signature is not None and path is not None:
        files = [_staged_file(config, signature, path)]
    else:
        files = sorted(directory.glob("*.json"))
    removed = 0
    for f in files:
        if signature is not None and path is None:
            entry = None
            try:
                entry = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if str(entry.get("signature", "")) != signature:
                continue
        try:
            f.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


# --- Backups ------------------------------------------------------------------

def _backup_paths(config: TjConfig, signature: str, path: str) -> tuple[Path, Path]:
    key = stage_key(signature, path)
    directory = backups_dir(config)
    return directory / f"{key}.orig.gz", directory / f"{key}.meta.json"


def save_backup(
    config: TjConfig, signature: str, path: str, *,
    original: str | None, output: str,
) -> None:
    """Stash the pre-write file, gzipped, plus a meta sidecar.

    ``original`` is ``None`` when the write CREATES the file. That is recorded
    explicitly rather than as an empty string, because undo has to know the
    difference: restoring an empty file where none existed leaves litter behind
    and reads as a successful revert.
    """
    directory = backups_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    orig_file, meta_file = _backup_paths(config, signature, path)
    orig_file.write_bytes(gzip.compress((original or "").encode("utf-8")))
    meta_file.write_text(json.dumps({
        "signature": signature,
        "source_path": path,
        "created_file": original is None,
        "original_sha256": None if original is None else sha256(original),
        "output_sha256": sha256(output),
        "applied_at": utcnow().isoformat(),
    }, ensure_ascii=False), encoding="utf-8")


def list_backups(config: TjConfig) -> list[dict[str, Any]]:
    """Every applied rule write that still has a backup — the undo surface.

    Each row carries an ``undoable`` flag plus the reason when false, computed
    with the SAME conditions :func:`load_backup` enforces. Surfacing the reason
    up front is the difference between an Undo control that explains itself and
    one that fails when clicked.
    """
    directory = backups_dir(config)
    if not directory.exists():
        return []
    out: list[dict[str, Any]] = []
    for meta_file in sorted(directory.glob("*.meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        source = str(meta.get("source_path", "") or "")
        if not source:
            continue
        # Pair the blob to THIS meta by filename rather than recomputing the
        # key from source_path: recomputing resolves symlinks, so a target that
        # has since become a link would map to a different key and read as
        # "backup missing" (the same trap `summarize.backup.list_backups`
        # documents).
        orig_file = meta_file.with_name(
            meta_file.name[: -len(".meta.json")] + ".orig.gz",
        )
        target = Path(source).expanduser()
        undoable, reason = True, ""
        if not orig_file.exists():
            undoable, reason = False, "backup file missing"
        elif target.is_symlink():
            undoable, reason = False, "symlink — refusing to restore through it"
        elif not target.exists():
            # A created file that has since been deleted needs no undo: the
            # world is already back where it started.
            undoable = bool(meta.get("created_file"))
            reason = "" if undoable else "file no longer exists"
        else:
            try:
                current = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                undoable, reason = False, "cannot read the file"
            else:
                if sha256(current) != meta.get("output_sha256"):
                    undoable, reason = (
                        False, "changed since apply — undo would lose newer edits",
                    )
        out.append({
            "signature": str(meta.get("signature", "") or ""),
            "source_path": source,
            "applied_at": str(meta.get("applied_at", "") or ""),
            "created_file": bool(meta.get("created_file")),
            "undoable": undoable,
            "reason": reason,
        })
    return out


def load_backup(
    config: TjConfig, signature: str, path: str, current: str | None,
) -> tuple[str | None, bool]:
    """The pre-write content for ``(signature, path)``, and whether the write
    created the file.

    ``(None, True)`` means the write created the file, so undoing it deletes
    the file rather than restoring bytes. Raises ``RuleWriteRefused`` when
    there is no backup or the file has changed since apply — refusing to undo
    over newer edits is the same promise ``summarize undo`` makes.
    """
    orig_file, meta_file = _backup_paths(config, signature, path)
    if not (orig_file.exists() and meta_file.exists()):
        raise RuleWriteRefused(
            f"no rule-write backup for {path} — nothing to undo.",
        )
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if current is not None and sha256(current) != meta.get("output_sha256"):
            raise RuleWriteRefused(
                f"{path} has changed since `tj rules apply` wrote it — refusing "
                "to undo (newer edits would be lost).",
            )
        if meta.get("created_file"):
            return None, True
        return gzip.decompress(orig_file.read_bytes()).decode("utf-8"), False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
        raise RuleWriteRefused(
            f"the rule-write backup for {path} is unreadable or corrupt — "
            "cannot undo.",
        ) from exc


def clear_backup(config: TjConfig, signature: str, path: str) -> None:
    for f in _backup_paths(config, signature, path):
        try:
            f.unlink()
        except FileNotFoundError:
            continue
